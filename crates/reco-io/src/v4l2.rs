//! Raw V4L2 MMAP capture for direct sensor access on Linux.
//!
//! Bypasses GStreamer and NVIDIA's nvargus ISP. Captures raw 10-bit
//! RGGB Bayer frames from `/dev/videoN` using kernel MMAP buffers.
//! Designed for the patched IMX477 driver at 4032x3040@30fps.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
pub struct V4l2Camera {
    file: File,
    buffers: Vec<MappedBuffer>,
    width: u32,
    height: u32,
    streaming: bool,
}

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
    /// streaming - call [`start`] when ready.
    pub fn open(config: &V4l2CameraConfig) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&config.device)?;
        let fd = file.as_raw_fd();

        // Set format: RG10 at requested resolution
        let mut fmt = unsafe { std::mem::zeroed::<V4l2Format>() };
        fmt.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.width = config.width;
        fmt.height = config.height;
        fmt.pixelformat = V4L2_PIX_FMT_SRGGB10;
        unsafe { v4l2_ioctl(fd, VIDIOC_S_FMT, &mut fmt as *mut _ as *mut _)? };

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
    pub fn start(&mut self) -> io::Result<()> {
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
            { raw[60] = V4L2_MEMORY_MMAP as u8; }
            #[cfg(target_pointer_width = "32")]
            { raw[44] = V4L2_MEMORY_MMAP as u8; }

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
        { raw[60] = V4L2_MEMORY_MMAP as u8; }
        #[cfg(target_pointer_width = "32")]
        { raw[44] = V4L2_MEMORY_MMAP as u8; }

        unsafe { v4l2_ioctl(fd, VIDIOC_DQBUF, &mut buf as *mut _ as *mut _)? };

        let idx = buf.index as usize;
        let mb = &self.buffers[idx];
        let byte_count = (self.width * self.height * 2) as usize;

        // Bulk memcpy from MMAP buffer (no per-pixel conversion)
        let src = unsafe { std::slice::from_raw_parts(mb.ptr, byte_count.min(mb.length)) };
        dst.clear();
        dst.extend_from_slice(src);

        // Re-queue the buffer
        unsafe { v4l2_ioctl(fd, VIDIOC_QBUF, &mut buf as *mut _ as *mut _)? };

        Ok(())
    }

    /// Stop streaming.
    pub fn stop(&mut self) -> io::Result<()> {
        if self.streaming {
            let fd = self.file.as_raw_fd();
            let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            unsafe { v4l2_ioctl(fd, VIDIOC_STREAMOFF, &mut type_ as *mut _ as *mut _)? };
            self.streaming = false;
            log::info!("V4L2 streaming stopped");
        }
        Ok(())
    }

    /// Frame dimensions.
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

impl Drop for V4l2Camera {
    fn drop(&mut self) {
        let _ = self.stop();
        for mb in &self.buffers {
            unsafe {
                libc::munmap(mb.ptr as *mut libc::c_void, mb.length);
            }
        }
    }
}

// Safety: the MMAP pointers are only accessed via &mut self methods,
// and the File fd is Send.
unsafe impl Send for V4l2Camera {}

/// Stereo pair of V4L2 cameras for raw Bayer capture.
///
/// Each camera runs in its own thread (parallel DQBUF, no phase-offset
/// penalty). A pairing thread zips left+right into stereo pairs using
/// Arc-shared buffers (no per-frame allocation or copy in the hot path).
pub struct V4l2StereoCameraSource {
    rx: std::sync::mpsc::Receiver<(Arc<Vec<u8>>, Arc<Vec<u8>>)>,
    info: reco_core::source::SourceInfo,
    stop: Arc<AtomicBool>,
}

fn spawn_v4l2_capture_thread(
    config: V4l2CameraConfig,
    label: &'static str,
    stop: Arc<AtomicBool>,
) -> std::sync::mpsc::Receiver<Arc<Vec<u8>>> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<Arc<Vec<u8>>>(2);
    let frame_bytes = (config.width * config.height * 2) as usize;

    std::thread::Builder::new()
        .name(format!("v4l2_{label}"))
        .spawn(move || {
            let mut cam = match V4l2Camera::open(&config) {
                Ok(c) => c,
                Err(e) => {
                    log::error!("V4L2 {label} open failed: {e}");
                    return;
                }
            };
            if let Err(e) = cam.start() {
                log::error!("V4L2 {label} start failed: {e}");
                return;
            }

            let mut buf = Vec::with_capacity(frame_bytes);

            while !stop.load(Ordering::Relaxed) {
                if let Err(e) = cam.next_frame_into(&mut buf) {
                    log::error!("V4L2 {label} capture: {e}");
                    break;
                }
                // Arc::new takes ownership; buf gets a fresh allocation
                // on next iteration only if Arc refcount > 1 (i.e. the
                // consumer hasn't dropped the previous frame yet).
                // In steady state with the channel buffering, this
                // stabilizes to zero allocations per frame.
                let frame = Arc::new(std::mem::take(&mut buf));
                if tx.send(frame).is_err() {
                    break;
                }
                // Reclaim capacity for next frame if possible
                if buf.capacity() < frame_bytes {
                    buf = Vec::with_capacity(frame_bytes);
                }
            }

            let _ = cam.stop();
        })
        .expect("spawn v4l2 capture thread");

    rx
}

impl V4l2StereoCameraSource {
    /// Open and start stereo capture from two V4L2 devices.
    ///
    /// Each camera runs in its own capture thread for parallel DQBUF.
    pub fn open(
        left_config: &V4l2CameraConfig,
        right_config: &V4l2CameraConfig,
    ) -> io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));

        let left_rx = spawn_v4l2_capture_thread(
            left_config.clone(),
            "left",
            stop.clone(),
        );
        let right_rx = spawn_v4l2_capture_thread(
            right_config.clone(),
            "right",
            stop.clone(),
        );

        let (tx, rx) = std::sync::mpsc::sync_channel::<(Arc<Vec<u8>>, Arc<Vec<u8>>)>(2);

        std::thread::Builder::new()
            .name("v4l2_pair".into())
            .spawn(move || {
                while let (Ok(left), Ok(right)) = (left_rx.recv(), right_rx.recv()) {
                    if tx.send((left, right)).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn v4l2 pairing thread");

        let info = reco_core::source::SourceInfo {
            width: left_config.width,
            height: left_config.height,
            fps: left_config.fps as f64,
            fps_rational: None,
            total_frames: None,
        };

        log::info!(
            "V4L2 stereo source ready: {}x{} @ {} fps (RGGB, parallel capture)",
            left_config.width,
            left_config.height,
            left_config.fps,
        );

        Ok(Self { rx, info, stop })
    }

    /// Signal capture threads to stop.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Source metadata.
    pub fn info(&self) -> reco_core::source::SourceInfo {
        self.info.clone()
    }

    /// Get the next stereo frame pair as raw Bayer bytes.
    ///
    /// Each Arc<Vec<u8>> is `width * height * 2` bytes of little-endian
    /// u16 (R16Uint). Arc-shared to avoid copying between threads.
    pub fn next_pair(&mut self) -> io::Result<Option<(Arc<Vec<u8>>, Arc<Vec<u8>>)>> {
        match self.rx.recv() {
            Ok(pair) => Ok(Some(pair)),
            Err(_) => Ok(None),
        }
    }
}

impl Drop for V4l2StereoCameraSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        while self.rx.try_recv().is_ok() {}
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}
