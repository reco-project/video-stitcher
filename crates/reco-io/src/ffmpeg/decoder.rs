//! Video decoder: file → YUV420P frames.
//!
//! Wraps FFmpeg's demuxer and decoder to produce YUV420P plane data
//! frame-by-frame from any video file FFmpeg can read. YUV planes are
//! uploaded directly to the GPU, eliminating CPU-side color conversion.
//!
//! ## Hardware Acceleration
//!
//! When available, the decoder uses hardware acceleration (NVDEC via CUDA,
//! or VAAPI) to offload H.264 decoding to the dedicated ASIC. Decoded
//! frames are transferred back to CPU and converted to YUV420P. Falls
//! back to software decoding transparently.

extern crate ffmpeg_next as ffmpeg;

use ffmpeg::format::{Pixel, input};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context as ScalingContext, flag::Flags as ScalingFlags};
use ffmpeg::util::frame::video::Video as VideoFrame;

// Raw FFI for hardware acceleration (not exposed by ffmpeg-next's safe API).
use ffmpeg::ffi;

/// Create a tracing span guard (no-op when `profiling` feature is disabled).
#[cfg(feature = "profiling")]
macro_rules! profile_scope {
    ($name:expr) => {
        let _span = tracing::info_span!($name).entered();
    };
}

#[cfg(not(feature = "profiling"))]
macro_rules! profile_scope {
    ($name:expr) => {};
}
use std::path::Path;
use std::ptr;
use thiserror::Error;

/// Errors from the video decoder.
#[derive(Debug, Error)]
pub enum DecodeError {
    /// FFmpeg error.
    #[error("FFmpeg: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// No video stream found in the file.
    #[error("no video stream found")]
    NoVideoStream,

    /// Pixel format conversion failed.
    #[error("format conversion: {0}")]
    ConversionError(String),
}

/// A decoded NV12 frame still on the GPU (CUDA device memory).
///
/// Contains CUDA device pointers and strides for the Y and UV planes.
/// These pointers are only valid until the next decode call — the caller
/// must copy the data (via `cuMemcpy2D`) before calling `next_frame_gpu()` again.
pub struct GpuFrame {
    /// CUDA device pointer to the Y (luma) plane.
    pub y_ptr: u64,
    /// CUDA device pointer to the UV (chroma) plane (NV12 interleaved).
    pub uv_ptr: u64,
    /// Row stride (pitch) of the Y plane in bytes.
    pub y_pitch: usize,
    /// Row stride (pitch) of the UV plane in bytes.
    pub uv_pitch: usize,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in microseconds.
    pub timestamp_us: i64,
}

/// A decoded YUV420P frame with timestamp.
///
/// Contains tightly-packed plane data (no stride padding):
/// - Y: `width × height` bytes (luma, full resolution)
/// - U: `(width/2) × (height/2)` bytes (chroma blue, half resolution)
/// - V: `(width/2) × (height/2)` bytes (chroma red, half resolution)
pub struct YuvFrame {
    /// Y (luma) plane data, tightly packed.
    pub y: Vec<u8>,
    /// U (Cb) plane data, tightly packed.
    pub u: Vec<u8>,
    /// V (Cr) plane data, tightly packed.
    pub v: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in microseconds.
    pub timestamp_us: i64,
}

/// Which decode backend is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeBackend {
    /// Software decode (libavcodec).
    Software,
    /// NVIDIA NVDEC via CUDA.
    Cuda,
    /// VA-API (Intel/AMD on Linux).
    Vaapi,
}

impl std::fmt::Display for DecodeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Software => write!(f, "software"),
            Self::Cuda => write!(f, "NVDEC (CUDA)"),
            Self::Vaapi => write!(f, "VA-API"),
        }
    }
}

/// Video decoder that produces YUV420P frames from a video file.
///
/// Automatically attempts hardware-accelerated decoding (NVDEC, VA-API)
/// and falls back to software decode. In all cases, output is YUV420P
/// plane data ready for GPU upload.
///
/// # Example
///
/// ```rust,no_run
/// use reco_io::ffmpeg::decoder::VideoDecoder;
/// use std::path::Path;
///
/// let mut decoder = VideoDecoder::open(Path::new("video.mp4")).unwrap();
/// println!("Using: {}", decoder.backend());
/// while let Some(frame) = decoder.next_frame().unwrap() {
///     println!("Frame: {}x{} @ {}us", frame.width, frame.height, frame.timestamp_us);
/// }
/// ```
pub struct VideoDecoder {
    input: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    /// Converts non-YUV420P frames (e.g. NV12 from hwaccel) to YUV420P.
    scaler: Option<ScalingContext>,
    video_stream_index: usize,
    time_base_num: i64,
    time_base_den: i64,
    width: u32,
    height: u32,
    eof_sent: bool,
    backend: DecodeBackend,
    /// Reusable decode buffer.
    decoded_frame: VideoFrame,
    /// CPU-side frame for hardware decode transfer.
    sw_frame: VideoFrame,
    /// Reusable conversion buffer (only used when scaler is active).
    converted_frame: VideoFrame,
    /// Hardware device context reference (must be kept alive).
    /// Stored as raw pointer; freed via av_buffer_unref on drop.
    _hw_device_ref: *mut ffi::AVBufferRef,
}

// SAFETY: The FFmpeg contexts are only accessed from a single thread
// (the decode thread). The raw pointers are owned exclusively.
unsafe impl Send for VideoDecoder {}

impl Drop for VideoDecoder {
    fn drop(&mut self) {
        if !self._hw_device_ref.is_null() {
            unsafe {
                ffi::av_buffer_unref(&mut self._hw_device_ref);
            }
        }
    }
}

impl VideoDecoder {
    /// Open a video file for decoding.
    ///
    /// Tries hardware acceleration in order: CUDA (NVDEC), VAAPI, then
    /// falls back to software decode. Hardware acceleration is transparent —
    /// the output is always YUV420P regardless of backend.
    ///
    /// Set `RECO_NO_HWACCEL=1` to force software decode (useful for benchmarking).
    pub fn open(path: &Path) -> Result<Self, DecodeError> {
        crate::init();

        let ictx = input(path)?;

        let stream = ictx
            .streams()
            .best(Type::Video)
            .ok_or(DecodeError::NoVideoStream)?;
        let video_stream_index = stream.index();
        let time_base = stream.time_base();

        let mut context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
        // Enable multithreaded decode (frame-level threading).
        // count(0) = auto-detect optimal thread count.
        context.set_threading(ffmpeg::threading::Config::count(0));

        // Try hardware acceleration (unless disabled via env var)
        let (backend, hw_device_ref) = if std::env::var("RECO_NO_HWACCEL").is_ok() {
            log::info!("Hardware acceleration disabled via RECO_NO_HWACCEL");
            (DecodeBackend::Software, ptr::null_mut())
        } else {
            try_hwaccel(&mut context)
        };

        let decoder = context.decoder().video()?;

        let width = decoder.width();
        let height = decoder.height();

        log::info!(
            "Decoder: {}x{} {:?}, time_base={}/{}, backend={}",
            width,
            height,
            decoder.format(),
            time_base.0,
            time_base.1,
            backend,
        );

        Ok(Self {
            input: ictx,
            decoder,
            scaler: None, // Created lazily on first frame (format may change with hwaccel)
            video_stream_index,
            time_base_num: time_base.0 as i64,
            time_base_den: time_base.1 as i64,
            width,
            height,
            eof_sent: false,
            backend,
            decoded_frame: VideoFrame::empty(),
            sw_frame: VideoFrame::empty(),
            converted_frame: VideoFrame::empty(),
            _hw_device_ref: hw_device_ref,
        })
    }

    /// Which decode backend is active.
    pub fn backend(&self) -> DecodeBackend {
        self.backend
    }

    /// Frame width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Frame rate as an FFmpeg rational (numerator/denominator).
    ///
    /// Falls back to 30fps if the decoder cannot determine the frame rate.
    pub fn frame_rate(&self) -> ffmpeg::Rational {
        self.decoder.frame_rate().unwrap_or_else(|| {
            log::warn!("Could not determine frame rate, defaulting to 30fps");
            ffmpeg::Rational(30, 1)
        })
    }

    /// Frame rate as frames per second.
    pub fn fps(&self) -> f64 {
        let r = self.frame_rate();
        r.0 as f64 / r.1 as f64
    }

    /// Decode the next YUV420P frame, or `None` if the video is finished.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "decode_frame")
    )]
    pub fn next_frame(&mut self) -> Result<Option<YuvFrame>, DecodeError> {
        if self.eof_sent {
            return Ok(None);
        }

        // Try to receive a frame from packets already sent to the decoder,
        // or read new packets until we get one.
        loop {
            {
                profile_scope!("h264_decode");
                if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                    return Ok(Some(self.extract_yuv()?));
                }
            }

            // Need more data — read next video packet
            let mut found_packet = false;
            for (stream, packet) in self.input.packets() {
                if stream.index() == self.video_stream_index {
                    profile_scope!("send_packet");
                    self.decoder.send_packet(&packet)?;
                    found_packet = true;
                    break;
                }
            }

            if !found_packet {
                self.eof_sent = true;
                self.decoder.send_eof()?;
                if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                    return Ok(Some(self.extract_yuv()?));
                }
                return Ok(None);
            }
        }
    }

    /// Decode the next frame and return raw GPU (CUDA) pointers.
    ///
    /// Only valid when the backend is [`DecodeBackend::Cuda`]. Returns the
    /// NV12 frame's CUDA device pointers (Y + interleaved UV) and their
    /// pitches. The caller must copy via `cuMemcpy2D` before calling this
    /// method again, as the pointers are reused by the decoder.
    ///
    /// Falls back to [`Self::next_frame`] for non-CUDA backends.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "decode_frame_gpu")
    )]
    pub fn next_frame_gpu(&mut self) -> Result<Option<GpuFrame>, DecodeError> {
        if self.backend != DecodeBackend::Cuda {
            // Fallback: decode to CPU and return None to signal "no GPU frame"
            return Ok(None);
        }

        if self.eof_sent {
            return Ok(None);
        }

        loop {
            {
                profile_scope!("h264_decode");
                if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                    if is_hw_frame(&self.decoded_frame) {
                        return Ok(Some(self.extract_gpu_frame()));
                    }
                    // Frame came back as CPU (unusual) — fallback
                    return Ok(None);
                }
            }

            let mut found_packet = false;
            for (stream, packet) in self.input.packets() {
                if stream.index() == self.video_stream_index {
                    profile_scope!("send_packet");
                    self.decoder.send_packet(&packet)?;
                    found_packet = true;
                    break;
                }
            }

            if !found_packet {
                self.eof_sent = true;
                self.decoder.send_eof()?;
                if self.decoder.receive_frame(&mut self.decoded_frame).is_ok()
                    && is_hw_frame(&self.decoded_frame)
                {
                    return Ok(Some(self.extract_gpu_frame()));
                }
                return Ok(None);
            }
        }
    }

    /// Extract CUDA device pointers from an NVDEC-decoded NV12 frame.
    ///
    /// The frame's `data[0]` is the Y plane device pointer, `data[1]` is
    /// the UV plane device pointer. `linesize[0]` and `linesize[1]` are
    /// the respective pitches.
    fn extract_gpu_frame(&self) -> GpuFrame {
        let raw = unsafe { &*self.decoded_frame.as_ptr() };

        let pts = raw.pts;
        let pts = if pts == ffi::AV_NOPTS_VALUE { 0 } else { pts };
        let timestamp_us = if self.time_base_den != 0 {
            pts * self.time_base_num * 1_000_000 / self.time_base_den
        } else {
            0
        };

        GpuFrame {
            y_ptr: raw.data[0] as u64,
            uv_ptr: raw.data[1] as u64,
            y_pitch: raw.linesize[0] as usize,
            uv_pitch: raw.linesize[1] as usize,
            width: self.width,
            height: self.height,
            timestamp_us,
        }
    }

    /// Extract YUV420P planes from the current decoded frame.
    ///
    /// For hardware-decoded frames, transfers from GPU to CPU first.
    /// If the frame format isn't YUV420P (e.g. NV12 from NVDEC),
    /// swscale converts it (very cheap for NV12→YUV420P: just deinterleave UV).
    fn extract_yuv(&mut self) -> Result<YuvFrame, DecodeError> {
        // If hardware-decoded, transfer GPU frame to CPU
        let cpu_frame = if is_hw_frame(&self.decoded_frame) {
            profile_scope!("hw_transfer");
            unsafe {
                let ret = ffi::av_hwframe_transfer_data(
                    self.sw_frame.as_mut_ptr(),
                    self.decoded_frame.as_ptr(),
                    0,
                );
                if ret < 0 {
                    log::error!("av_hwframe_transfer_data failed: {ret}");
                    // Fall through to use decoded_frame directly (will likely fail)
                    &self.decoded_frame
                } else {
                    // Copy PTS from the original frame
                    (*self.sw_frame.as_mut_ptr()).pts = (*self.decoded_frame.as_ptr()).pts;
                    &self.sw_frame
                }
            }
        } else {
            &self.decoded_frame
        };

        let frame_format = cpu_frame.format();

        // Get or create scaler if needed
        let source = if frame_format != Pixel::YUV420P {
            let needs_new_scaler = self.scaler.is_none()
                || self
                    .scaler
                    .as_ref()
                    .is_some_and(|_| self.converted_frame.format() != Pixel::YUV420P);

            if needs_new_scaler {
                log::info!(
                    "Creating scaler: {:?} → YUV420P ({}x{})",
                    frame_format,
                    self.width,
                    self.height
                );
                self.scaler = Some(
                    ScalingContext::get(
                        frame_format,
                        self.width,
                        self.height,
                        Pixel::YUV420P,
                        self.width,
                        self.height,
                        ScalingFlags::POINT,
                    )
                    .map_err(|e| {
                        DecodeError::ConversionError(format!(
                            "cannot create scaler for {frame_format:?}: {e}"
                        ))
                    })?,
                );
            }

            if let Some(scaler) = &mut self.scaler {
                profile_scope!("swscale");
                scaler
                    .run(cpu_frame, &mut self.converted_frame)
                    .map_err(|e| {
                        DecodeError::ConversionError(format!("swscale failed: {e}"))
                    })?;
                &self.converted_frame
            } else {
                cpu_frame
            }
        } else {
            cpu_frame
        };

        let pts = unsafe { (*self.decoded_frame.as_ptr()).pts };
        let pts = if pts == ffi::AV_NOPTS_VALUE { 0 } else { pts };
        let timestamp_us = if self.time_base_den != 0 {
            pts * self.time_base_num * 1_000_000 / self.time_base_den
        } else {
            0
        };

        let w = self.width as usize;
        let h = self.height as usize;
        let uv_w = w / 2;
        let uv_h = h / 2;

        let y = extract_plane(source.data(0), source.stride(0), w, h);
        let u = extract_plane(source.data(1), source.stride(1), uv_w, uv_h);
        let v = extract_plane(source.data(2), source.stride(2), uv_w, uv_h);

        Ok(YuvFrame {
            y,
            u,
            v,
            width: self.width,
            height: self.height,
            timestamp_us,
        })
    }
}

/// Check if a frame is hardware-decoded (lives on GPU memory).
fn is_hw_frame(frame: &VideoFrame) -> bool {
    unsafe { !(*frame.as_ptr()).hw_frames_ctx.is_null() }
}

/// Try to enable hardware-accelerated decoding on the codec context.
///
/// Attempts CUDA (NVDEC) first, then VAAPI. Returns the active backend
/// and the hw device reference (must be kept alive and freed on drop).
fn try_hwaccel(
    context: &mut ffmpeg::codec::context::Context,
) -> (DecodeBackend, *mut ffi::AVBufferRef) {
    // Hardware device types to try, in priority order
    let candidates = [
        (
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
            DecodeBackend::Cuda,
        ),
        (
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
            DecodeBackend::Vaapi,
        ),
    ];

    for (hw_type, backend) in candidates {
        let hw_type_name = unsafe {
            let name = ffi::av_hwdevice_get_type_name(hw_type);
            if name.is_null() {
                "unknown"
            } else {
                std::ffi::CStr::from_ptr(name).to_str().unwrap_or("unknown")
            }
        };

        // Try to create a hardware device context
        let mut device_ref: *mut ffi::AVBufferRef = ptr::null_mut();
        let ret = unsafe {
            ffi::av_hwdevice_ctx_create(
                &mut device_ref,
                hw_type,
                ptr::null(),     // default device
                ptr::null_mut(), // no options
                0,               // no flags
            )
        };

        if ret < 0 {
            log::debug!("Hardware decode {hw_type_name}: not available (error {ret})");
            continue;
        }

        // Assign the hardware device context to the codec context.
        // The codec context takes a reference (av_buffer_ref), so we keep our own.
        unsafe {
            (*context.as_mut_ptr()).hw_device_ctx = ffi::av_buffer_ref(device_ref);
        }

        log::info!("Hardware decode enabled: {backend} ({hw_type_name})");
        return (backend, device_ref);
    }

    log::info!("No hardware decoder available — using software decode");
    (DecodeBackend::Software, ptr::null_mut())
}

/// Copy one plane from an FFmpeg frame, removing stride padding.
///
/// If stride == width (common for 1920-wide frames), this is a single memcpy.
/// Panics if the `data` slice is too small for the given dimensions.
fn extract_plane(data: &[u8], stride: usize, width: usize, height: usize) -> Vec<u8> {
    assert!(
        stride >= width,
        "extract_plane: stride ({stride}) < width ({width})"
    );
    if height > 0 {
        let required = (height - 1) * stride + width;
        assert!(
            data.len() >= required,
            "extract_plane: buffer too small: need {required} bytes for {width}x{height} (stride {stride}), got {}",
            data.len()
        );
    }

    if stride == width {
        data[..width * height].to_vec()
    } else {
        let mut out = Vec::with_capacity(width * height);
        for row in 0..height {
            let start = row * stride;
            out.extend_from_slice(&data[start..start + width]);
        }
        out
    }
}
