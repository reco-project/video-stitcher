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

// `profile_scope!` is defined and exported by `reco_core`. Using it here
// avoids maintaining a local copy.
use reco_core::profile_scope;

use std::path::Path;
use std::ptr;
use thiserror::Error;

/// Errors from the video decoder. `Clone + Send + Sync` so the
/// calibration background thread can send the full result through
/// an mpsc channel to the UI thread with the typed error preserved.
/// `ffmpeg::Error` is not `Clone`, so it is stringified at the
/// `From` boundary.
#[derive(Debug, Clone, Error)]
pub enum DecodeError {
    /// FFmpeg error (text message; original `ffmpeg::Error` is
    /// stringified at the `From` boundary because it is not `Clone`).
    #[error("FFmpeg: {0}")]
    Ffmpeg(String),

    /// No video stream found in the file.
    #[error("no video stream found")]
    NoVideoStream,

    /// Video has odd width or height, incompatible with YUV420P.
    #[error("odd dimensions {width}x{height}: YUV420P requires even width and height")]
    OddDimensions {
        /// Frame width in pixels.
        width: u32,
        /// Frame height in pixels.
        height: u32,
    },

    /// Pixel format conversion failed.
    #[error("format conversion: {0}")]
    ConversionError(String),
}

impl From<ffmpeg::Error> for DecodeError {
    fn from(e: ffmpeg::Error) -> Self {
        Self::Ffmpeg(e.to_string())
    }
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
    /// GPU pixel format (NV12 or P010). Determines shared texture formats
    /// and CUDA copy byte widths.
    pub pixel_format: reco_core::renderer::GpuPixelFormat,
}

/// A decoded frame from VideoToolbox, holding a `CVPixelBufferRef`.
///
/// The pixel buffer is backed by an IOSurface and can be imported into
/// Metal as a texture via `CVMetalTextureCache` without a CPU copy.
///
/// **Lifetime:** The `CVPixelBufferRef` is only valid until the next
/// `next_frame_vt()` call. The caller must import it into Metal textures
/// (or retain it) before decoding the next frame.
#[cfg(target_os = "macos")]
pub struct VtFrame {
    /// Opaque `CVPixelBufferRef` pointer from the decoded AVFrame.
    /// Cast from `frame->data[3]` per FFmpeg's hwaccel convention.
    pub cv_pixel_buffer: *mut std::ffi::c_void,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in microseconds.
    pub timestamp_us: i64,
}

/// Re-export the canonical YUV frame type from reco-core.
pub use reco_core::source::YuvFrame;

/// Which decode backend is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeBackend {
    /// Software decode (libavcodec).
    Software,
    /// NVIDIA NVDEC via CUDA.
    Cuda,
    /// VA-API (Intel/AMD on Linux).
    Vaapi,
    /// Apple VideoToolbox (macOS).
    VideoToolbox,
    /// D3D11VA (Windows - AMD/Intel/NVIDIA).
    D3d11va,
}

impl std::fmt::Display for DecodeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Software => write!(f, "software"),
            Self::Cuda => write!(f, "NVDEC (CUDA)"),
            Self::Vaapi => write!(f, "VA-API"),
            Self::VideoToolbox => write!(f, "VideoToolbox"),
            Self::D3d11va => write!(f, "D3D11VA"),
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
    scaler: Option<ScalingContext>,
    video_stream_index: usize,
    time_base_num: i64,
    time_base_den: i64,
    width: u32,
    height: u32,
    eof_sent: bool,
    backend: DecodeBackend,
    rotation: i32,
    is_10bit: bool,
    decoded_frame: VideoFrame,
    sw_frame: VideoFrame,
    converted_frame: VideoFrame,
    _hw_device_ref: *mut ffi::AVBufferRef,
    /// Concat manifest kept alive so the temp file isn't deleted while
    /// the demuxer reads it. None for single-file inputs.
    _manifest: Option<tempfile::NamedTempFile>,
    y_buf: Vec<u8>,
    u_buf: Vec<u8>,
    v_buf: Vec<u8>,
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
        Self::from_input_context(ictx)
    }

    /// Open from an `InputPath` (single file or chained segments).
    pub fn open_input(input_path: &crate::stitch_job::InputPath) -> Result<Self, DecodeError> {
        match input_path {
            crate::stitch_job::InputPath::Single(p) => Self::open(p),
            crate::stitch_job::InputPath::Chained(paths) => Self::open_chained(paths),
        }
    }

    fn from_input_context(ictx: ffmpeg::format::context::Input) -> Result<Self, DecodeError> {
        let stream = ictx
            .streams()
            .best(Type::Video)
            .ok_or(DecodeError::NoVideoStream)?;
        let video_stream_index = stream.index();
        let time_base = stream.time_base();

        let mut context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
        // Enable multithreaded decode (frame-level threading).
        // count(0) = auto-detect optimal thread count.
        // kind(Frame) = decode multiple frames in parallel.
        let mut threading = ffmpeg::threading::Config::kind(ffmpeg::threading::Type::Frame);
        threading.count = 0;
        context.set_threading(threading);

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

        // YUV420P chroma subsampling requires even dimensions.
        if width % 2 != 0 || height % 2 != 0 {
            return Err(DecodeError::OddDimensions { width, height });
        }

        // Read rotation from stream side data (display matrix).
        // Common with DJI cameras that store upside-down video with rotation=-180.
        // Uses ffmpeg-next's safe side_data() iterator which handles both
        // FFmpeg <7 (stream side data) and FFmpeg 8+ (codecpar coded_side_data).
        let rotation = ictx
            .streams()
            .best(Type::Video)
            .and_then(|stream| {
                stream
                    .side_data()
                    .find(|sd| sd.kind() == ffmpeg::codec::packet::side_data::Type::DisplayMatrix)
                    .map(|sd| {
                        let data = sd.data();
                        if data.len() >= 36 {
                            // Display matrix is 9 x int32 = 36 bytes
                            let matrix_ptr = data.as_ptr() as *const i32;
                            let angle = unsafe { ffi::av_display_rotation_get(matrix_ptr) };
                            ((-angle).round() as i32 % 360 + 360) % 360
                        } else {
                            0
                        }
                    })
            })
            .unwrap_or(0);

        if rotation != 0 {
            log::info!("Stream has rotation={rotation} degrees, will apply to decoded frames");
        }

        // Detect 10-bit source from the codec parameters pixel format.
        // NVDEC reports Pixel::CUDA as its format, but the underlying surface is
        // P010 when the source is 10-bit HEVC (e.g. DJI Action 4). We check the
        // original stream parameters before hwaccel overrides the format.
        let is_10bit = {
            let raw_codecpar = unsafe { &*stream.parameters().as_ptr() };
            let format_i32 = raw_codecpar.format;
            // Validate the pixel format via FFmpeg before converting to the
            // Rust enum. av_pix_fmt_desc_get returns NULL for unknown/invalid
            // format values, letting us avoid transmute on out-of-range i32.
            let desc = unsafe { ffi::av_pix_fmt_desc_get(raw_i32_to_pix_fmt(format_i32)) };
            if desc.is_null() {
                log::warn!("Unknown pixel format {format_i32}, assuming 8-bit");
                false
            } else {
                let pixel = Pixel::from(raw_i32_to_pix_fmt(format_i32));
                let is_10 = matches!(
                    pixel,
                    Pixel::YUV420P10LE | Pixel::YUV420P10BE | Pixel::P010LE | Pixel::P010BE
                );
                if is_10 {
                    log::info!(
                        "Source is 10-bit ({pixel:?}), GPU textures will use R16Unorm/Rg16Unorm"
                    );
                }
                is_10
            }
        };

        log::debug!(
            "Decoder: {}x{} {:?}, time_base={}/{}, backend={}",
            width,
            height,
            decoder.format(),
            time_base.0,
            time_base.1,
            backend,
        );

        let y_size = width as usize * height as usize;
        let uv_size = (width as usize / 2) * (height as usize / 2);

        Ok(Self {
            input: ictx,
            decoder,
            scaler: None,
            video_stream_index,
            time_base_num: time_base.0 as i64,
            time_base_den: time_base.1 as i64,
            width,
            height,
            eof_sent: false,
            backend,
            rotation,
            is_10bit,
            decoded_frame: VideoFrame::empty(),
            sw_frame: VideoFrame::empty(),
            converted_frame: VideoFrame::empty(),
            _hw_device_ref: hw_device_ref,
            _manifest: None,
            y_buf: Vec::with_capacity(y_size),
            u_buf: Vec::with_capacity(uv_size),
            v_buf: Vec::with_capacity(uv_size),
        })
    }

    /// Open multiple video segments as a single continuous stream.
    ///
    /// Uses FFmpeg's concat demuxer to chain segments transparently.
    /// Timestamps are rebased, hardware acceleration works across
    /// segment boundaries, and seeking spans the full duration.
    pub fn open_chained(paths: &[std::path::PathBuf]) -> Result<Self, DecodeError> {
        use std::io::Write;

        if paths.len() <= 1 {
            return Self::open(paths.first().map(|p| p.as_path()).unwrap_or(Path::new("")));
        }

        crate::init();

        let mut manifest = tempfile::Builder::new()
            .prefix("reco_concat_")
            .suffix(".txt")
            .tempfile()
            .map_err(|e| DecodeError::Ffmpeg(format!("concat manifest: {e}")))?;

        writeln!(manifest, "ffconcat version 1.0")
            .map_err(|e| DecodeError::Ffmpeg(format!("write manifest: {e}")))?;
        for p in paths {
            writeln!(manifest, "file '{}'", p.display())
                .map_err(|e| DecodeError::Ffmpeg(format!("write manifest: {e}")))?;
        }
        manifest
            .flush()
            .map_err(|e| DecodeError::Ffmpeg(format!("flush manifest: {e}")))?;

        log::info!(
            "Concat demuxer: chaining {} segments via {}",
            paths.len(),
            manifest.path().display()
        );

        let concat_fmt_ptr = unsafe { ffi::av_find_input_format(c"concat".as_ptr()) };
        if concat_fmt_ptr.is_null() {
            return Err(DecodeError::Ffmpeg(
                "FFmpeg concat demuxer not available".into(),
            ));
        }
        let concat_fmt = unsafe { ffmpeg::format::format::Input::wrap(concat_fmt_ptr as *mut _) };

        let mut opts = ffmpeg::Dictionary::new();
        opts.set("safe", "0");

        let fmt = ffmpeg::format::format::Format::Input(concat_fmt);
        let ictx = match ffmpeg::format::open_with(manifest.path(), &fmt, opts)? {
            ffmpeg::format::Context::Input(ctx) => ctx,
            _ => return Err(DecodeError::Ffmpeg("expected input context".into())),
        };

        let mut dec = Self::from_input_context(ictx)?;
        dec._manifest = Some(manifest);
        Ok(dec)
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

    /// GPU pixel format for zero-copy shared textures.
    ///
    /// Returns `GpuPixelFormat::P010` for 10-bit sources (R16Unorm/Rg16Unorm)
    /// or `GpuPixelFormat::Nv12` for 8-bit (R8Unorm/Rg8Unorm).
    pub fn pixel_format(&self) -> reco_core::renderer::GpuPixelFormat {
        if self.is_10bit {
            reco_core::renderer::GpuPixelFormat::P010
        } else {
            reco_core::renderer::GpuPixelFormat::Nv12
        }
    }

    /// Rotation from stream metadata (0, 90, 180, 270 degrees).
    ///
    /// The CPU decode path applies this by reversing buffers in `extract_yuv`.
    /// The GPU zero-copy path must apply it in the shader (UV flip).
    pub fn rotation(&self) -> i32 {
        self.rotation
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

    /// Video duration in seconds, or `None` if unknown.
    pub fn duration_secs(&self) -> Option<f64> {
        let dur = self.input.duration();
        if dur > 0 {
            // FFmpeg reports duration in AV_TIME_BASE units (microseconds).
            Some(dur as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE))
        } else {
            None
        }
    }

    /// Seek to approximately the given timestamp in seconds.
    ///
    /// Seeks to the nearest keyframe before the target time, then the
    /// caller should call `next_frame()` to advance to the exact frame.
    /// Flushes the decoder so that subsequent frames are decoded fresh.
    pub fn seek_to_secs(&mut self, secs: f64) -> Result<(), DecodeError> {
        // Convert seconds to AV_TIME_BASE (microseconds)
        let ts = (secs * f64::from(ffmpeg::ffi::AV_TIME_BASE)) as i64;
        self.input.seek(ts, ..ts)?;
        self.decoder.flush();
        self.eof_sent = false;
        Ok(())
    }

    /// Decode the next YUV420P frame, or `None` if the video is finished.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "decode_frame")
    )]
    pub fn next_frame(&mut self) -> Result<Option<YuvFrame>, DecodeError> {
        // After EOF we still may have buffered frames in the decoder
        // (multi-slice encoders like libx264 with `slices=7` or
        // multi-reference H.264 with long GOPs hold several frames in
        // the reorder queue). Keep draining until `receive_frame`
        // stops returning frames — only then signal real end-of-stream.
        if self.eof_sent {
            profile_scope!("h264_decode");
            if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                return Ok(Some(self.extract_yuv()?));
            }
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
                // Fall through to the `eof_sent` branch on subsequent
                // calls; drain one frame now to keep the caller's
                // "poll one at a time" loop moving forward.
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
                        return Ok(Some(self.extract_gpu_frame()?));
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
                    return Ok(Some(self.extract_gpu_frame()?));
                }
                return Ok(None);
            }
        }
    }

    /// Decode the next frame and return a VideoToolbox `CVPixelBufferRef`.
    ///
    /// Returns `Ok(None)` if the backend is not VideoToolbox or EOF is reached.
    /// The returned `VtFrame.cv_pixel_buffer` is only valid until the next call
    /// to this method - the caller must import it into Metal textures before
    /// decoding another frame.
    #[cfg(target_os = "macos")]
    pub fn next_frame_vt(&mut self) -> Result<Option<VtFrame>, DecodeError> {
        if self.backend != DecodeBackend::VideoToolbox {
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
                        return Ok(Some(self.extract_vt_frame()));
                    }
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
                    return Ok(Some(self.extract_vt_frame()));
                }
                return Ok(None);
            }
        }
    }

    /// Convert a raw PTS value to microseconds using the stream time base.
    ///
    /// Treats `AV_NOPTS_VALUE` as 0. Returns 0 if the time base denominator
    /// is zero (shouldn't happen with valid streams, but avoids division by zero).
    fn pts_to_us(&self, raw_pts: i64) -> i64 {
        let pts = if raw_pts == ffi::AV_NOPTS_VALUE {
            0
        } else {
            raw_pts
        };
        if self.time_base_den != 0 {
            pts * self.time_base_num * 1_000_000 / self.time_base_den
        } else {
            0
        }
    }

    /// Extract the `CVPixelBufferRef` from a VideoToolbox-decoded frame.
    ///
    /// Per FFmpeg's hwaccel convention, `frame->data[3]` holds the surface
    /// pointer (CVPixelBufferRef for VideoToolbox).
    #[cfg(target_os = "macos")]
    fn extract_vt_frame(&self) -> VtFrame {
        let raw = unsafe { &*self.decoded_frame.as_ptr() };
        let timestamp_us = self.pts_to_us(raw.pts);

        VtFrame {
            cv_pixel_buffer: raw.data[3] as *mut std::ffi::c_void,
            width: self.width,
            height: self.height,
            timestamp_us,
        }
    }

    /// Extract CUDA device pointers from an NVDEC-decoded NV12 frame.
    ///
    /// The frame's `data[0]` is the Y plane device pointer, `data[1]` is
    /// the UV plane device pointer. `linesize[0]` and `linesize[1]` are
    /// the respective pitches.
    fn extract_gpu_frame(&self) -> Result<GpuFrame, DecodeError> {
        let raw = unsafe { &*self.decoded_frame.as_ptr() };
        let timestamp_us = self.pts_to_us(raw.pts);

        // FFmpeg linesize can be negative for some pixel formats (bottom-to-top).
        // Negative values would wrap to huge usize, causing GPU memory corruption.
        let y_ls = raw.linesize[0];
        let uv_ls = raw.linesize[1];
        if y_ls <= 0 || uv_ls <= 0 {
            return Err(DecodeError::ConversionError(format!(
                "negative or zero linesize: y={y_ls}, uv={uv_ls}"
            )));
        }

        let pixel_format = self.pixel_format();
        log::trace!(
            "GPU frame: y_ptr={:#x}, uv_ptr={:#x}, y_pitch={y_ls}, uv_pitch={uv_ls}, {pixel_format:?}, {}x{}",
            raw.data[0] as u64,
            raw.data[1] as u64,
            self.width,
            self.height
        );
        Ok(GpuFrame {
            y_ptr: raw.data[0] as u64,
            uv_ptr: raw.data[1] as u64,
            y_pitch: y_ls as usize,
            uv_pitch: uv_ls as usize,
            width: self.width,
            height: self.height,
            timestamp_us,
            pixel_format,
        })
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
                log::debug!(
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
                    .map_err(|e| DecodeError::ConversionError(format!("swscale failed: {e}")))?;
                &self.converted_frame
            } else {
                cpu_frame
            }
        } else {
            cpu_frame
        };

        let timestamp_us = self.pts_to_us(unsafe { (*self.decoded_frame.as_ptr()).pts });

        let w = self.width as usize;
        let h = self.height as usize;
        let uv_w = w / 2;
        let uv_h = h / 2;

        extract_plane_into(&mut self.y_buf, source.data(0), source.stride(0), w, h);
        extract_plane_into(
            &mut self.u_buf,
            source.data(1),
            source.stride(1),
            uv_w,
            uv_h,
        );
        extract_plane_into(
            &mut self.v_buf,
            source.data(2),
            source.stride(2),
            uv_w,
            uv_h,
        );

        // Apply rotation from stream metadata.
        // 180 degrees: reverse each row and reverse row order (equivalent to
        // reversing the entire buffer). Zero-copy, in-place.
        if self.rotation == 180 {
            self.y_buf.reverse();
            self.u_buf.reverse();
            self.v_buf.reverse();
        }

        Ok(YuvFrame {
            y: self.y_buf.clone(),
            u: self.u_buf.clone(),
            v: self.v_buf.clone(),
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
        #[cfg(target_os = "windows")]
        (
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
            DecodeBackend::D3d11va,
        ),
        (
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
            DecodeBackend::Vaapi,
        ),
        #[cfg(target_os = "macos")]
        (
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
            DecodeBackend::VideoToolbox,
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

        log::debug!("Hardware decode enabled: {backend} ({hw_type_name})");
        return (backend, device_ref);
    }

    log::info!("No hardware decoder available - using software decode");
    (DecodeBackend::Software, ptr::null_mut())
}

/// Convert a raw `i32` codec format to `ffi::AVPixelFormat`.
///
/// `AVPixelFormat` is `#[repr(i32)]`, so this is a layout-safe reinterpret
/// via `union` rather than `std::mem::transmute`. FFmpeg accepts any i32
/// and returns NULL/error for unknown values, so no Rust-side validation
/// of the enum discriminant is needed.
fn raw_i32_to_pix_fmt(v: i32) -> ffi::AVPixelFormat {
    #[repr(C)]
    union Reinterpret {
        i: i32,
        fmt: ffi::AVPixelFormat,
    }
    // SAFETY: AVPixelFormat is #[repr(i32)], so i32 and AVPixelFormat have
    // identical size and alignment. Reading fmt after writing i is defined
    // because both are integer types with the same representation.
    unsafe { Reinterpret { i: v }.fmt }
}

/// Copy one plane from an FFmpeg frame into a reusable buffer, removing stride padding.
///
/// Clears `buf` and fills it with tightly-packed row data. The buffer's
/// allocation is reused across frames, avoiding per-frame heap allocation.
/// If stride == width (common for 1920-wide frames), this is a single memcpy.
///
/// Panics if the `data` slice is too small for the given dimensions.
fn extract_plane_into(buf: &mut Vec<u8>, data: &[u8], stride: usize, width: usize, height: usize) {
    assert!(
        stride >= width,
        "extract_plane_into: stride ({stride}) < width ({width})"
    );
    if height > 0 {
        let required = (height - 1) * stride + width;
        assert!(
            data.len() >= required,
            "extract_plane_into: buffer too small: need {required} bytes for {width}x{height} (stride {stride}), got {}",
            data.len()
        );
    }

    buf.clear();
    if stride == width {
        buf.extend_from_slice(&data[..width * height]);
    } else {
        buf.reserve(width * height);
        for row in 0..height {
            let start = row * stride;
            buf.extend_from_slice(&data[start..start + width]);
        }
    }
}
