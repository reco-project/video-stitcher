//! Raw V4L2 MMAP capture for direct sensor access on Linux.
//!
//! Bypasses GStreamer and NVIDIA's nvargus ISP. Captures raw 10-bit
//! RGGB Bayer frames from `/dev/videoN` using kernel MMAP buffers.
//! Both devices must negotiate tightly packed RG10 at the requested dimensions.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};

// V4L2 ioctl helpers. We compute ioctl request codes using the Linux
// _IOC macro: code = (dir << 30) | (size << 16) | (type << 8) | nr
const fn v4l2_ioc(dir: u32, nr: u32, size: u32) -> libc::c_ulong {
    ((dir << 30) | (size << 16) | ((b'V' as u32) << 8) | nr) as libc::c_ulong
}
const IOC_WRITE: u32 = 1;
const IOC_RW: u32 = 3;

const VIDIOC_S_FMT: libc::c_ulong = v4l2_ioc(IOC_RW, 5, V4L2_FORMAT_SIZE as u32);
const VIDIOC_REQBUFS: libc::c_ulong = v4l2_ioc(IOC_RW, 8, 20);
const VIDIOC_QUERYBUF: libc::c_ulong = v4l2_ioc(IOC_RW, 9, V4L2_BUFFER_SIZE as u32);
const VIDIOC_QBUF: libc::c_ulong = v4l2_ioc(IOC_RW, 15, V4L2_BUFFER_SIZE as u32);
const VIDIOC_DQBUF: libc::c_ulong = v4l2_ioc(IOC_RW, 17, V4L2_BUFFER_SIZE as u32);
const VIDIOC_STREAMON: libc::c_ulong = v4l2_ioc(IOC_WRITE, 18, 4);
const VIDIOC_STREAMOFF: libc::c_ulong = v4l2_ioc(IOC_WRITE, 19, 4);

const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
const V4L2_MEMORY_MMAP: u32 = 1;
const V4L2_PIX_FMT_SRGGB10: u32 =
    (b'R' as u32) | ((b'G' as u32) << 8) | ((b'1' as u32) << 16) | ((b'0' as u32) << 24);

const NUM_BUFFERS: u32 = 4;

// Struct sizes differ between 32-bit and 64-bit Linux due to
// timeval and pointer widths. These must match the kernel's layout.
#[cfg(target_pointer_width = "64")]
const V4L2_BUFFER_SIZE: usize = 88;
#[cfg(target_pointer_width = "32")]
const V4L2_BUFFER_SIZE: usize = 68;
const V4L2_FORMAT_SIZE: usize = 208;

/// v4l2_format (type + 4-byte pad + pix_format union)
#[repr(C)]
struct V4l2Format {
    type_: u32,
    _type_pad: u32,
    // v4l2_pix_format fields within the union
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    priv_: u32,
    flags: u32,
    ycbcr_enc: u32,
    quantization: u32,
    xfer_func: u32,
    _pad: [u8; V4L2_FORMAT_SIZE - 8 - 48],
}

/// v4l2_requestbuffers
#[repr(C)]
struct V4l2Requestbuffers {
    count: u32,
    type_: u32,
    memory: u32,
    capabilities: u32,
    flags: u8,
    _reserved: [u8; 3],
}

/// v4l2_buffer (minimal, zero-initialized fields we don't read)
#[repr(C)]
struct V4l2Buffer {
    index: u32,
    type_: u32,
    bytesused: u32,
    flags: u32,
    field: u32,
    _rest: [u8; V4L2_BUFFER_SIZE - 20],
}

struct MappedBuffer {
    ptr: *mut u8,
    length: usize,
}

/// Configuration for V4L2 direct camera capture.
#[derive(Debug, Clone)]
pub struct V4l2CameraConfig {
    pub device: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub exposure: u32,
    pub gain: u32,
}

impl Default for V4l2CameraConfig {
    fn default() -> Self {
        Self {
            device: "/dev/video0".into(),
            width: 4032,
            height: 3040,
            fps: 30,
            exposure: 780,
            gain: 16,
        }
    }
}

/// Raw V4L2 camera capturing 10-bit RGGB Bayer via MMAP.
struct V4l2Camera {
    file: File,
    buffers: Vec<MappedBuffer>,
    width: u32,
    height: u32,
    streaming: bool,
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn v4l2_ioctl(fd: i32, request: libc::c_ulong, arg: *mut libc::c_void) -> io::Result<()> {
    if libc::ioctl(fd, request, arg) < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

impl V4l2Camera {
    /// Open a V4L2 device and configure for raw Bayer capture.
    ///
    /// Sets format, controls, allocates MMAP buffers. Does NOT start
    /// streaming - call [`Self::start`] when ready.
    fn open(config: &V4l2CameraConfig) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&config.device)?;
        let fd = file.as_raw_fd();

        // Set format: RG10 at requested resolution
        let mut fmt = unsafe { std::mem::zeroed::<V4l2Format>() };
        fmt.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.width = config.width;
        fmt.height = config.height;
        fmt.pixelformat = V4L2_PIX_FMT_SRGGB10;
        unsafe { v4l2_ioctl(fd, VIDIOC_S_FMT, &mut fmt as *mut _ as *mut _)? };

        if (fmt.width, fmt.height, fmt.pixelformat)
            != (config.width, config.height, V4L2_PIX_FMT_SRGGB10)
            || fmt.bytesperline != config.width.saturating_mul(2)
        {
            return Err(io::Error::other(
                "camera did not accept the requested tightly packed RG10 format",
            ));
        }
        let sizeimage = fmt.sizeimage as usize;
        log::info!(
            "V4L2 format set: {}x{} RG10, sizeimage={}",
            fmt.width,
            fmt.height,
            sizeimage
        );

        // Set controls via v4l2-ctl subprocess. The Tegra controls
        // use int64 type which requires VIDIOC_S_EXT_CTRLS (complex
        // struct layout). v4l2-ctl handles this correctly.
        let ctrl_str = format!(
            "bypass_mode=0,override_enable=1,sensor_mode=0,frame_rate={},exposure={},gain={}",
            config.fps * 1_000_000,
            config.exposure,
            config.gain,
        );
        let output = std::process::Command::new("v4l2-ctl")
            .args(["-d", &config.device, "--set-ctrl", &ctrl_str])
            .output();
        match output {
            Ok(o) if o.status.success() => {
                log::info!("V4L2 controls set: {}", ctrl_str);
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                log::warn!("v4l2-ctl set-ctrl partial: {}", stderr.trim());
            }
            Err(e) => {
                log::warn!("v4l2-ctl not found ({}), controls may not be set", e);
            }
        }

        // Request MMAP buffers
        let mut reqbufs = unsafe { std::mem::zeroed::<V4l2Requestbuffers>() };
        reqbufs.count = NUM_BUFFERS;
        reqbufs.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        reqbufs.memory = V4L2_MEMORY_MMAP;
        unsafe { v4l2_ioctl(fd, VIDIOC_REQBUFS, &mut reqbufs as *mut _ as *mut _)? };
        log::info!("V4L2 allocated {} MMAP buffers", reqbufs.count);

        // Query and mmap each buffer
        let mut buffers = Vec::with_capacity(reqbufs.count as usize);
        for i in 0..reqbufs.count {
            let mut buf = unsafe { std::mem::zeroed::<V4l2Buffer>() };
            buf.index = i;
            buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.bytesused = 0;
            // We need offset and length from QUERYBUF. They're at known
            // positions in the _rest blob. On 64-bit: length at offset 52
            // from start of struct (byte 52-55), m.offset at byte 56-59.
            // We'll just read them from the raw bytes.
            unsafe {
                v4l2_ioctl(fd, VIDIOC_QUERYBUF, &mut buf as *mut _ as *mut _)?;
            }

            // Extract length and offset from the raw buffer struct.
            // On aarch64: after the 5 u32 fields (20 bytes) we have
            // timeval (16 bytes), timecode (16 bytes), sequence (4),
            // memory (4) = offset 60, then m.offset (4 or 8 bytes),
            // then length (4).
            let raw = unsafe {
                std::slice::from_raw_parts(&buf as *const _ as *const u8, V4L2_BUFFER_SIZE)
            };
            #[cfg(target_pointer_width = "64")]
            let (offset, length) = {
                let length = u32::from_ne_bytes([raw[72], raw[73], raw[74], raw[75]]) as usize;
                let offset = u32::from_ne_bytes([raw[64], raw[65], raw[66], raw[67]]);
                (offset, length)
            };
            #[cfg(target_pointer_width = "32")]
            let (offset, length) = {
                let length = u32::from_ne_bytes([raw[52], raw[53], raw[54], raw[55]]) as usize;
                let offset = u32::from_ne_bytes([raw[48], raw[49], raw[50], raw[51]]);
                (offset, length)
            };

            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    offset as libc::off_t,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }

            buffers.push(MappedBuffer {
                ptr: ptr as *mut u8,
                length,
            });
        }

        Ok(Self {
            file,
            buffers,
            width: config.width,
            height: config.height,
            streaming: false,
        })
    }

    /// Start streaming. Queues all buffers and enables capture.
    fn start(&mut self) -> io::Result<()> {
        let fd = self.file.as_raw_fd();

        for i in 0..self.buffers.len() as u32 {
            let mut buf = unsafe { std::mem::zeroed::<V4l2Buffer>() };
            buf.index = i;
            buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            // memory field is at a fixed offset, set it via the raw bytes
            let raw = unsafe {
                std::slice::from_raw_parts_mut(&mut buf as *mut _ as *mut u8, V4L2_BUFFER_SIZE)
            };
            #[cfg(target_pointer_width = "64")]
            {
                raw[60] = V4L2_MEMORY_MMAP as u8;
            }
            #[cfg(target_pointer_width = "32")]
            {
                raw[44] = V4L2_MEMORY_MMAP as u8;
            }

            unsafe { v4l2_ioctl(fd, VIDIOC_QBUF, &mut buf as *mut _ as *mut _)? };
        }

        let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        unsafe { v4l2_ioctl(fd, VIDIOC_STREAMON, &mut type_ as *mut _ as *mut _)? };
        self.streaming = true;
        log::info!("V4L2 streaming started");
        Ok(())
    }

    /// Dequeue the next frame and copy raw bytes into `dst`.
    ///
    /// `dst` must have capacity for `width * height * 2` bytes.
    /// Data is raw little-endian u16 Bayer (R16Uint format), copied
    /// directly from the MMAP buffer with no per-pixel conversion.
    pub fn next_frame_into(&mut self, dst: &mut Vec<u8>) -> io::Result<()> {
        let fd = self.file.as_raw_fd();

        let mut buf = unsafe { std::mem::zeroed::<V4l2Buffer>() };
        buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        let raw = unsafe {
            std::slice::from_raw_parts_mut(&mut buf as *mut _ as *mut u8, V4L2_BUFFER_SIZE)
        };
        #[cfg(target_pointer_width = "64")]
        {
            raw[60] = V4L2_MEMORY_MMAP as u8;
        }
        #[cfg(target_pointer_width = "32")]
        {
            raw[44] = V4L2_MEMORY_MMAP as u8;
        }

        unsafe { v4l2_ioctl(fd, VIDIOC_DQBUF, &mut buf as *mut _ as *mut _)? };

        let idx = buf.index as usize;
        let mb = self
            .buffers
            .get(idx)
            .ok_or_else(|| io::Error::other("invalid capture buffer index"))?;
        let byte_count = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| io::Error::other("capture frame size overflow"))?;
        if byte_count > mb.length || byte_count > buf.bytesused as usize {
            return Err(io::Error::other("short RG10 capture buffer"));
        }

        // Bulk memcpy from MMAP buffer (no per-pixel conversion)
        let src = unsafe { std::slice::from_raw_parts(mb.ptr, byte_count) };
        dst.clear();
        dst.extend_from_slice(src);

        // Re-queue the buffer
        unsafe { v4l2_ioctl(fd, VIDIOC_QBUF, &mut buf as *mut _ as *mut _)? };

        Ok(())
    }

    /// Stop streaming.
    fn stop(&mut self) -> io::Result<()> {
        if self.streaming {
            let fd = self.file.as_raw_fd();
            let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            unsafe { v4l2_ioctl(fd, VIDIOC_STREAMOFF, &mut type_ as *mut _ as *mut _)? };
            self.streaming = false;
            log::info!("V4L2 streaming stopped");
        }
        Ok(())
    }
}

impl Drop for V4l2Camera {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

impl Drop for MappedBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.cast(), self.length);
        }
    }
}

/// Two independently streaming RG10 devices, paired in dequeue order.
pub struct V4l2StereoCameraSource {
    cameras: [V4l2Camera; 2],
    pixels: [Vec<u8>; 2],
}

impl V4l2StereoCameraSource {
    /// Open both devices before starting capture.
    pub fn open(left: &V4l2CameraConfig, right: &V4l2CameraConfig) -> io::Result<Self> {
        let mut cameras = [V4l2Camera::open(left)?, V4l2Camera::open(right)?];
        for camera in &mut cameras {
            camera.start()?;
        }
        Ok(Self {
            cameras,
            pixels: Default::default(),
        })
    }

    /// Return one raw Bayer image per camera. Cancellation also interrupts warmup.
    pub fn next_pair(&mut self, interrupted: &AtomicBool) -> io::Result<Option<(&[u8], &[u8])>> {
        let mut ready = [false; 2];
        while !interrupted.load(Ordering::Relaxed) {
            let mut fds = std::array::from_fn::<_, 2, _>(|i| libc::pollfd {
                fd: self.cameras[i].file.as_raw_fd(),
                events: if ready[i] { 0 } else { libc::POLLIN },
                revents: 0,
            });
            // A finite wait also checks cancellation if both cameras stop delivering.
            if unsafe { libc::poll(fds.as_mut_ptr(), 2, 50) } < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error);
            }
            for (i, fd) in fds.iter().enumerate() {
                if fd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                    return Err(io::Error::other(format!(
                        "camera {i} stopped delivering capture buffers"
                    )));
                }
                if fd.revents & libc::POLLIN != 0 {
                    match self.cameras[i].next_frame_into(&mut self.pixels[i]) {
                        Ok(()) => ready[i] = true,
                        Err(e)
                            if matches!(
                                e.kind(),
                                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                            ) => {}
                        Err(e) => return Err(e),
                    }
                }
            }
            if ready == [true, true] {
                return Ok(Some((&self.pixels[0], &self.pixels[1])));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::OwnedFd;
    use std::os::unix::net::UnixStream;

    #[test]
    fn disconnected_camera_errors_even_if_peer_is_stalled() {
        let (left, _writer) = UnixStream::pair().unwrap();
        let (right, writer) = UnixStream::pair().unwrap();
        let cameras = [left, right].map(|s| V4l2Camera {
            file: File::from(OwnedFd::from(s)),
            buffers: Vec::new(),
            width: 2,
            height: 2,
            streaming: false,
        });
        let mut source = V4l2StereoCameraSource {
            cameras,
            pixels: Default::default(),
        };
        assert!(source.next_pair(&AtomicBool::new(true)).unwrap().is_none());
        drop(writer);
        assert!(source.next_pair(&AtomicBool::new(false)).is_err());
    }
}
