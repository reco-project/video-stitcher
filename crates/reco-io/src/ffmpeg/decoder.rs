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

mod hw_frame;
mod probe;

#[cfg(all(target_os = "windows", feature = "gpu"))]
pub use hw_frame::D3d11Frame;
#[cfg(feature = "gpu")]
pub use hw_frame::GpuFrame;
#[cfg(all(target_os = "macos", feature = "gpu"))]
pub use hw_frame::VtFrame;
pub use hw_frame::{SharedHwDevice, create_shared_hw_device};
use hw_frame::{attach_shared_hw_device, is_hw_frame, raw_i32_to_pix_fmt, try_hwaccel};

pub(crate) use probe::probe_recording;
use probe::{is_plausible_rate, pick_frame_rate, probe_concat, stream_color, stream_rotation};

use ffmpeg::Rescale;
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
#[cfg(target_os = "windows")]
use std::sync::{Arc, Mutex};
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

    /// A live decoder produced an image without presentation time.
    #[error("live input frame has no presentation timestamp")]
    MissingTimestamp,

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

    /// A hardware-only accessor was called on another backend.
    #[error("decoder backend mismatch: expected {expected}, got {actual}")]
    BackendMismatch {
        /// Required backend.
        expected: DecodeBackend,
        /// Active backend.
        actual: DecodeBackend,
    },

    /// A decoder returned a frame with incompatible residency.
    #[error("unexpected decoded frame format: expected {expected:?}, got {actual:?}")]
    UnexpectedFrameFormat {
        /// Required hardware pixel format.
        expected: Pixel,
        /// Received pixel format.
        actual: Pixel,
    },
}

impl From<ffmpeg::Error> for DecodeError {
    fn from(e: ffmpeg::Error) -> Self {
        Self::Ffmpeg(e.to_string())
    }
}

/// Re-export the canonical YUV frame type from reco-core.
pub use reco_core::source::Yuv420pFrame;

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
    time_base: ffmpeg::Rational,
    width: u32,
    height: u32,
    eof_sent: bool,
    backend: DecodeBackend,
    rotation: i32,
    #[cfg(feature = "gpu")]
    is_10bit: bool,
    is_full_range: bool,
    color_metadata: super::ColorMetadata,
    color_metadata_frozen: bool,
    last_frame_color: Option<super::ColorMetadata>,
    require_timestamps: bool,
    live_audio: Option<super::live_audio::LiveAudio>,
    decoded_frame: VideoFrame,
    sw_frame: VideoFrame,
    converted_frame: VideoFrame,
    _hw_device_ref: *mut ffi::AVBufferRef,
    #[cfg(target_os = "windows")]
    pending_hw_init: Option<Arc<Mutex<()>>>,
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

    /// Open with software decode, skipping the hardware-acceleration
    /// probe entirely. The typed form of `RECO_NO_HWACCEL` for callers
    /// that must not touch a GPU (e.g. `reco stitch --cpu`).
    pub fn open_software(path: &Path) -> Result<Self, DecodeError> {
        crate::init();
        let ictx = input(path)?;
        Self::from_input_context_inner(ictx, None, true, false)
    }

    /// [`Self::open_input`] with software decode forced (see
    /// [`Self::open_software`]).
    pub fn open_input_software(
        input_path: &crate::stitch_job::InputPath,
    ) -> Result<Self, DecodeError> {
        match input_path {
            crate::stitch_job::InputPath::Single(p) => Self::open_software(p),
            crate::stitch_job::InputPath::Chained(paths) => {
                Self::open_chained_impl(paths, None, true)
            }
        }
    }

    /// Open from an `InputPath` (single file or chained segments).
    pub fn open_input(input_path: &crate::stitch_job::InputPath) -> Result<Self, DecodeError> {
        match input_path {
            crate::stitch_job::InputPath::Single(p) => Self::open(p),
            crate::stitch_job::InputPath::Chained(paths) => Self::open_chained(paths),
        }
    }

    /// Probe the first segment's stream properties and validate colour across the chain.
    pub(crate) fn open_input_probe(
        source: &crate::stitch_job::InputPath,
        force_software: bool,
    ) -> Result<Self, DecodeError> {
        crate::init();
        let path = source.first_path();
        let context = input(path)?;
        let probe = match source {
            crate::stitch_job::InputPath::Chained(paths) if paths.len() > 1 => {
                Some(probe_concat(path, &context, &paths[1..])?)
            }
            _ => None,
        };
        let mut decoder = Self::from_input_context_inner(context, None, force_software, false)?;
        if let Some(probe) = probe {
            decoder.is_full_range = probe.full_range;
            decoder.color_metadata = probe.color;
        }
        Ok(decoder)
    }

    /// Open with a pre-created shared hw device.
    ///
    /// Both decoders in a stereo pair must share the same device so D3D11VA
    /// textures are on a single device (required by `CopySubresourceRegion`).
    pub fn open_input_with_shared_device(
        input_path: &crate::stitch_job::InputPath,
        shared: &SharedHwDevice,
    ) -> Result<Self, DecodeError> {
        crate::init();
        match input_path {
            crate::stitch_job::InputPath::Single(p) => {
                let ictx = input(p)?;
                Self::from_input_context_with_shared(ictx, shared)
            }
            // Chained must also attach the shared device. Otherwise each
            // concat decoder creates its own D3D11 device, and the staging
            // pool's CopySubresourceRegion copies one side's frame into the
            // other device's texture -> cross-device fault (DEVICE_REMOVED on
            // desktop NVIDIA, driver hang on laptops). See open_chained_impl.
            crate::stitch_job::InputPath::Chained(paths) => {
                Self::open_chained_impl(paths, Some(shared), false)
            }
        }
    }

    fn from_input_context(ictx: ffmpeg::format::context::Input) -> Result<Self, DecodeError> {
        Self::from_input_context_inner(ictx, None, false, false)
    }

    fn from_input_context_with_shared(
        ictx: ffmpeg::format::context::Input,
        shared: &SharedHwDevice,
    ) -> Result<Self, DecodeError> {
        Self::from_input_context_inner(ictx, Some(shared), false, false)
    }

    pub(crate) fn from_live_input(
        input: ffmpeg::format::context::Input,
        software: bool,
        audio: Option<super::live_audio::LiveAudio>,
    ) -> Result<Self, DecodeError> {
        let mut decoder = Self::from_input_context_inner(input, None, software, true)?;
        decoder.live_audio = audio;
        Ok(decoder)
    }

    fn from_input_context_inner(
        ictx: ffmpeg::format::context::Input,
        shared_device: Option<&SharedHwDevice>,
        force_software: bool,
        live: bool,
    ) -> Result<Self, DecodeError> {
        let stream = ictx
            .streams()
            .best(Type::Video)
            .ok_or(DecodeError::NoVideoStream)?;
        let video_stream_index = stream.index();
        let time_base = stream.time_base();

        let mut context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
        let (backend, hw_device_ref) = if force_software {
            log::info!("Software decode: forced by the caller");
            (DecodeBackend::Software, ptr::null_mut())
        } else if std::env::var("RECO_NO_HWACCEL").is_ok() {
            log::info!("Hardware acceleration disabled via RECO_NO_HWACCEL");
            (DecodeBackend::Software, ptr::null_mut())
        } else if let Some(shared) = shared_device {
            attach_shared_hw_device(&mut context, shared)
        } else {
            try_hwaccel(&mut context)
        };

        // Frame threading adds one frame of latency per thread. Live input
        // uses slice parallelism without changing file decode throughput.
        let mut threading = ffmpeg::threading::Config::kind(if live {
            ffmpeg::threading::Type::Slice
        } else {
            ffmpeg::threading::Type::Frame
        });
        // FFmpeg reserves an extra decode surface per frame thread. NVDEC's H.264 32-surface limit must leave room for the stream's reference/reorder pool. Other codecs keep automatic threading.
        threading.count =
            if backend == DecodeBackend::Cuda && context.id() == ffmpeg::codec::Id::H264 {
                1
            } else {
                0
            };
        context.set_threading(threading);
        let decoder = context.decoder().video()?;

        let width = decoder.width();
        let height = decoder.height();

        if live
            && (width > reco_core::calibration::MAX_DIM || height > reco_core::calibration::MAX_DIM)
        {
            return Err(DecodeError::ConversionError(format!(
                "invalid live input dimensions {width}x{height}"
            )));
        }

        // YUV420P chroma subsampling requires even dimensions.
        if width % 2 != 0 || height % 2 != 0 {
            return Err(DecodeError::OddDimensions { width, height });
        }

        // Read rotation from stream side data (display matrix).
        // Common with DJI cameras that store upside-down video with rotation=-180.
        // Uses ffmpeg-next's safe side_data() iterator which handles both
        // FFmpeg <7 (stream side data) and FFmpeg 8+ (codecpar coded_side_data).
        let rotation = stream_rotation(&stream);

        if rotation != 0 {
            log::info!("Stream has rotation={rotation} degrees, will apply to decoded frames");
        }

        // Detect 10-bit source from the codec parameters pixel format.
        // NVDEC reports Pixel::CUDA as its format, but the underlying surface is
        // P010 when the source is 10-bit HEVC (e.g. DJI Action 4). We check the
        // original stream parameters before hwaccel overrides the format.
        #[cfg(feature = "gpu")]
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
                    log::debug!(
                        "Source is 10-bit ({pixel:?}), GPU textures will use R16Unorm/Rg16Unorm"
                    );
                }
                is_10
            }
        };

        let (is_full_range, color_metadata) = stream_color(&stream);
        if is_full_range {
            log::debug!("Source is full-range YUV");
        }
        if !color_metadata.is_empty() {
            log::debug!("Source colour metadata: {color_metadata}");
        }

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
            time_base,
            width,
            height,
            eof_sent: false,
            backend,
            rotation,
            #[cfg(feature = "gpu")]
            is_10bit,
            is_full_range,
            color_metadata,
            color_metadata_frozen: !live,
            last_frame_color: None,
            require_timestamps: live,
            live_audio: None,
            decoded_frame: VideoFrame::empty(),
            sw_frame: VideoFrame::empty(),
            converted_frame: VideoFrame::empty(),
            _hw_device_ref: hw_device_ref,
            #[cfg(target_os = "windows")]
            pending_hw_init: if backend == DecodeBackend::D3d11va {
                shared_device.map(|shared| Arc::clone(&shared.decoder_init))
            } else {
                None
            },
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
        Self::open_chained_impl(paths, None, false)
    }

    /// Chained open that optionally attaches a pre-created shared hw device.
    ///
    /// A stereo pair's two concat decoders must land on the SAME D3D11
    /// device: the zero-copy staging pool copies both sides into textures on
    /// a single device via `CopySubresourceRegion`, which faults across
    /// devices. Single-file open already threads the shared device through;
    /// concat must do the same or it regresses to per-decoder devices.
    fn open_chained_impl(
        paths: &[std::path::PathBuf],
        shared: Option<&SharedHwDevice>,
        force_software: bool,
    ) -> Result<Self, DecodeError> {
        use std::io::Write;

        crate::init();

        if paths.len() <= 1 {
            let p = paths.first().map(|p| p.as_path()).unwrap_or(Path::new(""));
            return match shared {
                Some(s) => Self::from_input_context_with_shared(input(p)?, s),
                None if force_software => Self::open_software(p),
                None => Self::open(p),
            };
        }

        let probe = {
            let first = input(&paths[0])?;
            probe_concat(&paths[0], &first, &paths[1..])?
        };
        let mut manifest = tempfile::Builder::new()
            .prefix("reco_concat_")
            .suffix(".txt")
            .tempfile()
            .map_err(|e| DecodeError::Ffmpeg(format!("concat manifest: {e}")))?;

        writeln!(manifest, "ffconcat version 1.0")
            .map_err(|e| DecodeError::Ffmpeg(format!("write manifest: {e}")))?;
        for (idx, p) in paths.iter().enumerate() {
            writeln!(manifest, "file '{}'", p.display())
                .map_err(|e| DecodeError::Ffmpeg(format!("write manifest: {e}")))?;
            if let Some(durations) = probe.durations.as_ref() {
                writeln!(manifest, "duration {:.6}", durations[idx])
                    .map_err(|e| DecodeError::Ffmpeg(format!("write manifest: {e}")))?;
            }
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

        let mut dec = Self::from_input_context_inner(ictx, shared, force_software, false)?;
        dec.is_full_range = probe.full_range;
        dec.color_metadata = probe.color;
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
    #[cfg(feature = "gpu")]
    pub fn pixel_format(&self) -> reco_core::render::renderer::GpuPixelFormat {
        if self.is_10bit {
            reco_core::render::renderer::GpuPixelFormat::P010
        } else {
            reco_core::render::renderer::GpuPixelFormat::Nv12
        }
    }

    /// Whether the source uses full-range YUV (0-255) rather than limited range (16-235).
    pub fn is_full_range(&self) -> bool {
        self.is_full_range
    }

    /// Source colourimetry metadata (primaries, transfer characteristics, matrix) from stream headers.
    pub fn color_metadata(&self) -> super::ColorMetadata {
        self.color_metadata
    }

    /// Rotation from stream metadata (0, 90, 180, 270 degrees).
    ///
    /// The CPU decode path applies this by reversing buffers in `extract_yuv`.
    /// The GPU zero-copy path must apply it in the shader (UV flip).
    pub fn rotation(&self) -> i32 {
        self.rotation
    }

    /// Frame rate as an FFmpeg rational (numerator/denominator), or `None` if unavailable.
    ///
    /// Probes the container stream first (`r_frame_rate`, then `avg_frame_rate`), then codec timing. Missing or unusable timing is not replaced by an assumed rate.
    pub fn frame_rate(&self) -> Option<ffmpeg::Rational> {
        let stream = self.input.stream(self.video_stream_index);
        let codec_rate = self.decoder.frame_rate();
        let rate = pick_frame_rate(
            stream.as_ref().map(|s| s.rate()),
            stream.as_ref().map(|s| s.avg_frame_rate()),
            codec_rate,
        )?;
        if !codec_rate.as_ref().is_some_and(is_plausible_rate) {
            log::debug!(
                "Bitstream carries no timing info (no VUI); using container frame rate {}/{}",
                rate.0,
                rate.1
            );
        }
        Some(rate)
    }

    /// Frame rate as frames per second, or `None` if unavailable.
    pub fn fps(&self) -> Option<f64> {
        self.frame_rate().map(|r| r.0 as f64 / r.1 as f64)
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
        let ts = self.seek_timestamp_us(secs)?;
        self.input.seek(ts, ..ts)?;
        self.decoder.flush();
        self.eof_sent = false;
        self.last_frame_color = None;
        Ok(())
    }

    /// Round a seek target to the stream's timestamp grid, then convert it exactly as decoded PTS.
    pub(crate) fn seek_timestamp_us(&self, secs: f64) -> Result<i64, DecodeError> {
        if self.time_base.0 <= 0 || self.time_base.1 <= 0 {
            return Err(DecodeError::ConversionError(
                "invalid stream time base".into(),
            ));
        }
        let ticks = (secs * f64::from(self.time_base.1) / f64::from(self.time_base.0)).round();
        if !ticks.is_finite() || ticks <= i64::MIN as f64 || ticks >= i64::MAX as f64 {
            return Err(DecodeError::ConversionError(
                "seek timestamp is outside the stream range".into(),
            ));
        }
        self.pts_to_us(ticks as i64)
    }

    fn validate_frame_color(
        &mut self,
        frame_color: super::ColorMetadata,
    ) -> Result<(), DecodeError> {
        let previous = self.last_frame_color.unwrap_or_default();
        let changed = || {
            format!(
                "decoded color metadata changed under a fixed source description (opening: {}; previous frame: {previous}; frame: {frame_color})",
                self.color_metadata,
            )
        };
        // Stream tags can fill in fields absent from decoded frames.
        // A field observed on a frame must stay present until the decoder seeks.
        if frame_color != previous && frame_color.or(previous) != frame_color {
            return Err(DecodeError::ConversionError(changed()));
        }
        let effective_color = frame_color.or(self.color_metadata);
        if effective_color != self.color_metadata {
            match self.color_metadata.matches_description(effective_color) {
                Ok(true) => {}
                Ok(false) => return Err(DecodeError::ConversionError(changed())),
                Err(reason) => {
                    return Err(DecodeError::ConversionError(format!(
                        "{}: {reason}",
                        changed()
                    )));
                }
            }
        }
        self.last_frame_color = Some(frame_color);
        Ok(())
    }

    /// Receive one frame, supplying input only when the decoder requests it.
    /// A demuxer EOF starts draining; only the decoder's EOF ends the stream.
    fn receive_frame(&mut self) -> Result<bool, DecodeError> {
        // D3D11 decoder profile enumeration can race on a shared device.
        // FFmpeg initializes lazily, so serialize through the first frame,
        // not just codec open. Subsequent decoding remains independent.
        #[cfg(target_os = "windows")]
        let initialization = self.pending_hw_init.clone();
        #[cfg(target_os = "windows")]
        let _guard = initialization.as_ref().map(|gate| {
            // The mutex protects execution, not mutable state.
            gate.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
        });
        loop {
            let received = {
                profile_scope!("h264_decode");
                self.decoder.receive_frame(&mut self.decoded_frame)
            };
            match received {
                Ok(()) => {
                    let width = self.decoded_frame.width();
                    let height = self.decoded_frame.height();
                    if self.require_timestamps && (self.width == 0 || self.height == 0) {
                        if width == 0
                            || height == 0
                            || width > reco_core::calibration::MAX_DIM
                            || height > reco_core::calibration::MAX_DIM
                            || !width.is_multiple_of(2)
                            || !height.is_multiple_of(2)
                        {
                            return Err(DecodeError::ConversionError(format!(
                                "invalid initial live image dimensions {width}x{height}"
                            )));
                        }
                        // SDP need not describe image dimensions. Freeze metadata
                        // from the first actual image, before the source is published.
                        self.width = width;
                        self.height = height;
                        self.is_full_range = is_full_range(
                            self.decoded_frame.format(),
                            self.decoded_frame.color_range(),
                        );
                    }
                    let frame_color = super::ColorMetadata::from_frame(&self.decoded_frame);
                    if !self.color_metadata_frozen {
                        self.color_metadata = self.color_metadata.or(frame_color);
                        self.color_metadata_frozen = true;
                    }
                    if width != self.width || height != self.height {
                        return Err(DecodeError::ConversionError(format!(
                            "decoded dimensions changed from {}x{} to {width}x{height}",
                            self.width, self.height
                        )));
                    }
                    if is_full_range(
                        self.decoded_frame.format(),
                        self.decoded_frame.color_range(),
                    ) != self.is_full_range
                    {
                        return Err(DecodeError::ConversionError(
                            "decoded color range changed; the source requires a fixed range".into(),
                        ));
                    }
                    self.validate_frame_color(frame_color)?;
                    #[cfg(target_os = "windows")]
                    {
                        self.pending_hw_init = None;
                    }
                    return Ok(true);
                }
                Err(ffmpeg::Error::Eof) => {
                    #[cfg(target_os = "windows")]
                    {
                        self.pending_hw_init = None;
                    }
                    return Ok(false);
                }
                Err(ffmpeg::Error::Other { errno })
                    if errno == ffmpeg::error::EAGAIN && !self.eof_sent => {}
                Err(error) => return Err(error.into()),
            }

            // PacketIter retries all non-EOF demux errors indefinitely.
            // Read directly so failures reach the caller instead.
            loop {
                let mut packet = ffmpeg::Packet::empty();
                match packet.read(&mut self.input) {
                    Ok(()) if packet.stream() == self.video_stream_index => {
                        profile_scope!("send_packet");
                        self.decoder.send_packet(&packet)?;
                        break;
                    }
                    Ok(()) => {
                        if let Some(audio) = &self.live_audio
                            && packet.stream() == audio.stream_index
                        {
                            audio.push(packet).map_err(DecodeError::Ffmpeg)?;
                        }
                        continue;
                    }
                    Err(ffmpeg::Error::Eof) => {
                        self.decoder.send_eof()?;
                        self.eof_sent = true;
                        break;
                    }
                    Err(error) => return Err(error.into()),
                }
            }
        }
    }

    #[cfg(feature = "gpu")]
    fn require_backend(&self, expected: DecodeBackend) -> Result<(), DecodeError> {
        if self.backend != expected {
            return Err(DecodeError::BackendMismatch {
                expected,
                actual: self.backend,
            });
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn require_format(&self, expected: Pixel) -> Result<(), DecodeError> {
        let actual = self.decoded_frame.format();
        if actual != expected {
            return Err(DecodeError::UnexpectedFrameFormat { expected, actual });
        }
        Ok(())
    }

    /// Decode the next YUV420P frame, or `None` once the decoder is drained.
    /// Pixel-layout conversion preserves the source range reported by [`Self::is_full_range`].
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "decode_frame")
    )]
    pub fn next_frame(&mut self) -> Result<Option<Yuv420pFrame>, DecodeError> {
        if !self.receive_frame()? {
            return Ok(None);
        }
        self.extract_yuv().map(Some)
    }

    /// Decode the next frame and return raw GPU (CUDA) pointers.
    ///
    /// Requires [`DecodeBackend::Cuda`] and CUDA-resident output frames;
    /// backend or residency mismatches return an error, not end-of-stream.
    /// Returns `None` once the decoder is drained. The returned frame retains
    /// its allocation independently of the decoder; no image data is copied.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "decode_frame_gpu")
    )]
    #[cfg(feature = "gpu")]
    pub fn next_frame_gpu(&mut self) -> Result<Option<GpuFrame>, DecodeError> {
        self.require_backend(DecodeBackend::Cuda)?;
        if !self.receive_frame()? {
            return Ok(None);
        }
        self.require_format(Pixel::CUDA)?;
        self.extract_gpu_frame().map(Some)
    }

    /// Decode the next frame and return the D3D11VA texture + array slice.
    ///
    /// Requires [`DecodeBackend::D3d11va`] and D3D11-resident output frames;
    /// backend or residency mismatches return an error, not end-of-stream.
    /// Returns `None` once the decoder is drained. The caller must stage-copy
    /// the frame before decoding again, as the decoder may reuse the slice.
    #[cfg(target_os = "windows")]
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "decode_frame_d3d11")
    )]
    #[cfg(feature = "gpu")]
    pub fn next_frame_d3d11(&mut self) -> Result<Option<D3d11Frame>, DecodeError> {
        self.require_backend(DecodeBackend::D3d11va)?;
        if !self.receive_frame()? {
            return Ok(None);
        }
        self.require_format(Pixel::D3D11)?;
        self.extract_d3d11_frame().map(Some)
    }

    /// Decode the next frame and return a VideoToolbox `CVPixelBufferRef`.
    ///
    /// Requires [`DecodeBackend::VideoToolbox`] and VideoToolbox-resident frames;
    /// backend or residency mismatches return an error, not end-of-stream.
    /// Returns `None` once the decoder is drained. Retain or import the buffer
    /// before decoding again, as the decoder may reuse it.
    #[cfg(target_os = "macos")]
    #[cfg(feature = "gpu")]
    pub fn next_frame_vt(&mut self) -> Result<Option<VtFrame>, DecodeError> {
        self.require_backend(DecodeBackend::VideoToolbox)?;
        if !self.receive_frame()? {
            return Ok(None);
        }
        self.require_format(Pixel::VIDEOTOOLBOX)?;
        self.extract_vt_frame().map(Some)
    }

    /// Convert a raw PTS value to microseconds using the stream time base.
    ///
    /// Missing PTS retains the frame API's zero placeholder. Invalid time bases and unrepresentable timestamps fail; valid timestamps truncate toward zero without intermediate overflow.
    fn pts_to_us(&self, raw_pts: i64) -> Result<i64, DecodeError> {
        if self.time_base.0 <= 0 || self.time_base.1 <= 0 {
            return Err(DecodeError::ConversionError(
                "invalid stream time base".into(),
            ));
        }
        if raw_pts == ffi::AV_NOPTS_VALUE {
            if self.require_timestamps {
                return Err(DecodeError::MissingTimestamp);
            }
            return Ok(0);
        }
        let timestamp = raw_pts.rescale_with(
            self.time_base,
            ffmpeg::util::mathematics::rescale::TIME_BASE,
            ffmpeg::Rounding::Zero,
        );
        // FFmpeg returns INT64_MIN when the rescaled value cannot be represented.
        if timestamp == i64::MIN {
            return Err(DecodeError::ConversionError(
                "decoded timestamp is outside the microsecond range".into(),
            ));
        }
        Ok(timestamp)
    }

    /// Extract the `CVPixelBufferRef` from a VideoToolbox-decoded frame.
    ///
    /// Per FFmpeg's hwaccel convention, `frame->data[3]` holds the surface
    /// pointer (CVPixelBufferRef for VideoToolbox).
    #[cfg(target_os = "macos")]
    #[cfg(feature = "gpu")]
    fn extract_vt_frame(&self) -> Result<VtFrame, DecodeError> {
        let raw = unsafe { &*self.decoded_frame.as_ptr() };
        let timestamp_us = self.pts_to_us(raw.pts)?;

        Ok(VtFrame {
            cv_pixel_buffer: raw.data[3] as *mut std::ffi::c_void,
            width: self.width,
            height: self.height,
            timestamp_us,
        })
    }

    /// Extract the D3D11VA texture pointer and array slice from a decoded frame.
    ///
    /// Clones the AVFrame so the texture slice stays pinned in the decode
    /// pool until the `D3d11Frame` is dropped (after staging).
    #[cfg(target_os = "windows")]
    #[cfg(feature = "gpu")]
    fn extract_d3d11_frame(&self) -> Result<D3d11Frame, DecodeError> {
        let raw = unsafe { &*self.decoded_frame.as_ptr() };
        let timestamp_us = self.pts_to_us(raw.pts)?;
        let frame_ref = unsafe { ffi::av_frame_clone(self.decoded_frame.as_ptr()) };
        if frame_ref.is_null() {
            return Err(DecodeError::Ffmpeg(
                "failed to retain decoded D3D11 frame".into(),
            ));
        }

        Ok(D3d11Frame {
            texture: raw.data[0] as *mut std::ffi::c_void,
            array_slice: raw.data[1] as usize,
            width: self.width,
            height: self.height,
            timestamp_us,
            _frame_ref: frame_ref,
        })
    }

    /// Extract CUDA device pointers from an NVDEC-decoded NV12 frame.
    ///
    /// The frame's `data[0]` is the Y plane device pointer, `data[1]` is
    /// the UV plane device pointer. `linesize[0]` and `linesize[1]` are
    /// the respective pitches.
    #[cfg(feature = "gpu")]
    fn extract_gpu_frame(&self) -> Result<GpuFrame, DecodeError> {
        let raw = unsafe { &*self.decoded_frame.as_ptr() };
        let timestamp_us = self.pts_to_us(raw.pts)?;

        // FFmpeg linesize can be negative for some pixel formats (bottom-to-top).
        // Negative values would wrap to huge usize, causing GPU memory corruption.
        let y_ls = raw.linesize[0];
        let uv_ls = raw.linesize[1];
        if y_ls <= 0 || uv_ls <= 0 {
            return Err(DecodeError::ConversionError(format!(
                "negative or zero linesize: y={y_ls}, uv={uv_ls}"
            )));
        }

        // The codec's opening metadata can become stale across segments.
        // CUDA frames describe their actual plane format in the frame pool.
        let pixel_format = unsafe {
            if raw.hw_frames_ctx.is_null() || (*raw.hw_frames_ctx).data.is_null() {
                return Err(DecodeError::ConversionError(
                    "CUDA frame has no frame pool".into(),
                ));
            }
            let frames = &*((*raw.hw_frames_ctx).data as *const ffi::AVHWFramesContext);
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            {
                // FFmpeg's device-to-device transfer may still be queued on its own context.
                // The frame pool retains this device while the producer stream is drained.
                if frames.device_ctx.is_null() || (*frames.device_ctx).hwctx.is_null() {
                    return Err(DecodeError::ConversionError(
                        "CUDA frame has no device context".into(),
                    ));
                }
                let device = &*((*frames.device_ctx).hwctx as *const ffi::AVCUDADeviceContext);
                if device.cuda_ctx.is_null() {
                    return Err(DecodeError::ConversionError(
                        "CUDA frame has no CUDA context".into(),
                    ));
                }
                profile_scope!("decode_cuda_ready");
                reco_core::interop::cuda::cuda_wait_stream(
                    device.cuda_ctx.cast(),
                    device.stream.cast(),
                )
                .map_err(|error| DecodeError::ConversionError(error.to_string()))?;
            }
            match frames.sw_format {
                ffi::AVPixelFormat::AV_PIX_FMT_NV12 => {
                    reco_core::render::renderer::GpuPixelFormat::Nv12
                }
                ffi::AVPixelFormat::AV_PIX_FMT_P010LE => {
                    reco_core::render::renderer::GpuPixelFormat::P010
                }
                format => {
                    return Err(DecodeError::ConversionError(format!(
                        "unsupported CUDA plane format: {format:?}"
                    )));
                }
            }
        };
        log::trace!(
            "GPU frame: y_ptr={:#x}, uv_ptr={:#x}, y_pitch={y_ls}, uv_pitch={uv_ls}, {pixel_format:?}, {}x{}",
            raw.data[0] as u64,
            raw.data[1] as u64,
            self.width,
            self.height
        );
        // Video::clone copies pixels. av_frame_ref retains the hardware buffers instead.
        let mut owner = VideoFrame::empty();
        unsafe {
            if owner.as_ptr().is_null() {
                return Err(DecodeError::ConversionError(
                    "cannot allocate frame reference".into(),
                ));
            }
            let result = ffi::av_frame_ref(owner.as_mut_ptr(), self.decoded_frame.as_ptr());
            if result < 0 {
                return Err(ffmpeg::Error::from(result).into());
            }
        }
        Ok(GpuFrame {
            y_ptr: raw.data[0] as u64,
            uv_ptr: raw.data[1] as u64,
            y_pitch: y_ls as usize,
            uv_pitch: uv_ls as usize,
            width: self.width,
            height: self.height,
            timestamp_us,
            pixel_format,
            _owner: owner,
        })
    }

    /// Extract YUV420P planes from the current decoded frame.
    ///
    /// For hardware-decoded frames, transfers from GPU to CPU first.
    /// Ten-bit 4:2:0 is quantized to 8-bit in the source range. Other layouts
    /// use swscale (e.g. deinterleave UV for NV12).
    fn extract_yuv(&mut self) -> Result<Yuv420pFrame, DecodeError> {
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
                    return Err(DecodeError::ConversionError(format!(
                        "av_hwframe_transfer_data: {}",
                        ffmpeg::Error::from(ret)
                    )));
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

        let timestamp_us = self.pts_to_us(unsafe { (*self.decoded_frame.as_ptr()).pts })?;
        let w = self.width as usize;
        let h = self.height as usize;
        if matches!(
            frame_format,
            Pixel::P010LE | Pixel::P010BE | Pixel::YUV420P10LE | Pixel::YUV420P10BE
        ) {
            extract_10bit_yuv(
                cpu_frame,
                self.is_full_range,
                [&mut self.y_buf, &mut self.u_buf, &mut self.v_buf],
            );
        } else {
            // Get or create scaler if needed
            let source = if frame_format != Pixel::YUV420P {
                let needs_new_scaler = self
                    .scaler
                    .as_ref()
                    .is_none_or(|scaler| scaler.input().format != frame_format);

                if needs_new_scaler {
                    log::debug!(
                        "Creating scaler: {:?} → YUV420P ({}x{})",
                        frame_format,
                        self.width,
                        self.height
                    );
                    let mut scaler = ScalingContext::get(
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
                    })?;
                    // Preserve the source range when changing pixel layout; YUVJ defaults would compress it.
                    // SAFETY: the scaler is uniquely owned and FFmpeg owns the constant coefficient table.
                    let result = unsafe {
                        let coefficients = ffi::sws_getCoefficients(super::color::swscale_matrix(
                            self.color_metadata.matrix().unwrap_or_default(),
                        ));
                        ffi::sws_setColorspaceDetails(
                            scaler.as_mut_ptr(),
                            coefficients,
                            i32::from(self.is_full_range),
                            coefficients,
                            i32::from(self.is_full_range),
                            0,
                            1 << 16,
                            1 << 16,
                        )
                    };
                    if result < 0 {
                        return Err(DecodeError::ConversionError(format!(
                            "cannot preserve decoded color range: {}",
                            ffmpeg::Error::from(result)
                        )));
                    }
                    self.scaler = Some(scaler);
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
        }

        // Apply rotation from stream metadata.
        // 180 degrees: reverse each row and reverse row order (equivalent to
        // reversing the entire buffer). Zero-copy, in-place.
        if self.rotation == 180 {
            self.y_buf.reverse();
            self.u_buf.reverse();
            self.v_buf.reverse();
        }

        Ok(Yuv420pFrame {
            y: self.y_buf.clone(),
            u: self.u_buf.clone(),
            v: self.v_buf.clone(),
            width: self.width,
            height: self.height,
            timestamp_us,
        })
    }
}

fn extract_10bit_yuv(frame: &VideoFrame, full_range: bool, output: [&mut Vec<u8>; 3]) {
    let packed = matches!(frame.format(), Pixel::P010LE | Pixel::P010BE);
    let big_endian = matches!(frame.format(), Pixel::P010BE | Pixel::YUV420P10BE);
    let read = if big_endian {
        u16::from_be_bytes
    } else {
        u16::from_le_bytes
    };

    for (plane, buf) in output.into_iter().enumerate() {
        let chroma = plane != 0;
        let plane_index = if packed && chroma { 1 } else { plane };
        let stride = frame.stride(plane_index);
        let data = frame.data(plane_index);
        let width = frame.width() as usize >> usize::from(chroma);
        let height = frame.height() as usize >> usize::from(chroma);
        let (start, step) = if packed && chroma {
            ((plane - 1) * 2, 4)
        } else {
            (0, 2)
        };
        let shift = if packed { 6 } else { 0 };
        buf.clear();
        buf.reserve(width * height);
        for row in data.chunks_exact(stride).take(height) {
            buf.extend((0..width).map(|x| {
                let i = start + x * step;
                let code = i32::from((read([row[i], row[i + 1]]) >> shift) & 1023);
                let sample = if !full_range {
                    // The legal offsets and spans are exactly four times their 8-bit values.
                    (code + 2) / 4
                } else if chroma {
                    // Scale about neutral so 512 maps exactly to 128.
                    128 + ((code - 512) * 255 + 511).div_euclid(1023)
                } else {
                    (code * 255 + 511) / 1023
                };
                sample.clamp(0, 255) as u8
            }));
        }
    }
}

fn is_full_range(format: Pixel, range: ffmpeg::color::Range) -> bool {
    match range {
        ffmpeg::color::Range::JPEG => true,
        ffmpeg::color::Range::MPEG => false,
        ffmpeg::color::Range::Unspecified => {
            matches!(format, Pixel::YUVJ420P | Pixel::YUVJ422P | Pixel::YUVJ444P)
        }
    }
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

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires NVIDIA NVENC/NVDEC; also runs without reco-io's gpu feature"]
    fn ffmpeg_hardware_codecs_preserve_host_frames() {
        use super::{DecodeBackend, VideoDecoder};
        use crate::ffmpeg::encoder::{EncoderConfig, VideoEncoder};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hardware.mp4");
        let config = EncoderConfig {
            encoder_name: Some("h264_nvenc".into()),
            ..Default::default()
        };
        let mut encoder = VideoEncoder::new(&path, 256, 192, (30, 1).into(), &config).unwrap();
        assert_eq!(encoder.encoder_name(), "h264_nvenc");
        for index in 0..12 {
            let mut frame = vec![128; 256 * 192 * 3 / 2];
            frame[..256 * 192].fill(80 + index * 8);
            encoder.write_nv12_frame(&frame, None).unwrap();
        }
        encoder.finish().unwrap();
        let mut decoder = VideoDecoder::open(&path).unwrap();
        assert_eq!(decoder.backend(), DecodeBackend::Cuda);
        for index in 0..12 {
            let frame = decoder
                .next_frame()
                .unwrap()
                .expect("missing hardware-decoded frame");
            assert_eq!(
                decoder.decoded_frame.format(),
                ffmpeg::format::Pixel::CUDA,
                "hardware decode must not silently fall back"
            );
            assert_eq!((frame.width, frame.height), (256, 192));
            assert!(
                frame
                    .y
                    .iter()
                    .all(|value| value.abs_diff(80 + index * 8) <= 2)
            );
            assert!(
                frame
                    .u
                    .iter()
                    .chain(&frame.v)
                    .all(|value| value.abs_diff(128) <= 2)
            );
        }
        assert!(decoder.next_frame().unwrap().is_none());
    }

    use ffmpeg_next as ffmpeg;

    const R60: ffmpeg::Rational = ffmpeg::Rational(60, 1);
    const UNSET: ffmpeg::Rational = ffmpeg::Rational(0, 0);

    #[test]
    fn frame_color_validator_uses_stream_tags_until_frame_tags_appear() {
        use crate::ffmpeg::ColorMetadata;
        use ffmpeg::color::{Primaries, Space, TransferCharacteristic};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stream-color.mov");
        write_mjpeg_60fps(&path);
        let mut decoder = super::VideoDecoder::open_software(&path).unwrap();
        let tagged = ColorMetadata {
            primaries: Some(Primaries::BT2020),
            transfer: Some(TransferCharacteristic::ARIB_STD_B67),
            space: Some(Space::BT2020NCL),
        };
        let unspecified = ColorMetadata::default();
        decoder.color_metadata = tagged;

        for _ in 0..3 {
            decoder.validate_frame_color(unspecified).unwrap();
        }
        decoder.validate_frame_color(tagged).unwrap();
        let error = decoder.validate_frame_color(unspecified).unwrap_err();
        assert!(error.to_string().contains("decoded color metadata changed"));

        decoder.seek_to_secs(0.0).unwrap();
        assert_eq!(decoder.color_metadata(), tagged);
        for _ in 0..3 {
            decoder.validate_frame_color(unspecified).unwrap();
        }
    }

    #[test]
    fn seek_targets_use_decoded_timestamp_precision() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seek.mov");
        write_mjpeg_60fps(&path);
        let mut decoder = super::VideoDecoder::open_software(&path).unwrap();
        for (base, seconds, expected) in [
            ((1, 1000), 1.0 / 30.0, 33_000),
            ((1, 1000), 2.0 / 30.0, 67_000),
            ((1, 1000), 1.001 / 30.0, 33_000),
            ((1, 15360), 1.0 / 30.0, 33_333),
            ((1, 90000), 1.001 / 30.0, 33_366),
            ((1, 1000), -1.0 / 30.0, -33_000),
        ] {
            decoder.time_base = base.into();
            assert_eq!(decoder.seek_timestamp_us(seconds).unwrap(), expected);
        }
        for invalid in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            f64::MIN,
        ] {
            assert!(decoder.seek_timestamp_us(invalid).is_err());
        }
        for base in [(0, 1), (1, 0), (-1, 1), (1, -1)] {
            decoder.time_base = base.into();
            assert!(decoder.seek_timestamp_us(0.0).is_err());
        }
    }

    #[test]
    fn decoded_timestamp_rescaling_handles_large_clock_origins() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clock.mov");
        write_mjpeg_60fps(&path);
        let mut decoder = super::VideoDecoder::open_software(&path).unwrap();
        assert!(decoder.receive_frame().unwrap());
        decoder.time_base = (1, 90_000).into();
        let seconds = 1_753_000_000_i64;
        // SAFETY: the decoder uniquely owns this decoded frame. Change only its timestamp.
        unsafe { (*decoder.decoded_frame.as_mut_ptr()).pts = seconds * 90_000 };
        assert_eq!(
            decoder.extract_yuv().unwrap().timestamp_us,
            seconds * 1_000_000
        );
        for (time_base, pts, expected) in [
            ((1, 3), 1, 333_333),
            ((1, 3), -1, -333_333),
            ((1001, 30_000), 90_000, 3_003_000_000),
            ((1, 1_000_000), i64::MAX, i64::MAX),
            ((1, 1_000_000), i64::MIN + 1, i64::MIN + 1),
            ((1, 90_000), ffmpeg::ffi::AV_NOPTS_VALUE, 0),
        ] {
            decoder.time_base = time_base.into();
            assert_eq!(decoder.pts_to_us(pts).unwrap(), expected);
        }
        decoder.require_timestamps = true;
        assert!(matches!(
            decoder.pts_to_us(ffmpeg::ffi::AV_NOPTS_VALUE),
            Err(super::DecodeError::MissingTimestamp)
        ));
        decoder.require_timestamps = false;
        decoder.time_base = (1, 1).into();
        assert!(decoder.pts_to_us(i64::MAX).is_err());
        assert!(decoder.pts_to_us(i64::MIN + 1).is_err());
        // SAFETY: the uniquely owned frame remains decoded and only its PTS changes.
        unsafe { (*decoder.decoded_frame.as_mut_ptr()).pts = i64::MAX };
        assert!(decoder.extract_yuv().is_err());
        let mut seed = 0x53ab_9000_u64;
        for _ in 0..1000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let pts = seed as i64;
            decoder.time_base = ((seed % 1001 + 1) as i32, (seed % 100_000 + 1) as i32).into();
            let exact = i128::from(pts) * i128::from(decoder.time_base.0) * 1_000_000
                / i128::from(decoder.time_base.1);
            let expected = i64::try_from(exact).ok().filter(|&value| value != i64::MIN);
            assert_eq!(decoder.pts_to_us(pts).ok(), expected);
        }
        for time_base in [(0, 1), (1, 0), (-1, 1), (1, -1)] {
            decoder.time_base = time_base.into();
            assert!(decoder.pts_to_us(1).is_err());
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn shared_initialization_gate_is_only_retained_until_first_result() {
        use super::{Arc, Mutex, VideoDecoder};

        ffmpeg::init().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("software.mov");
        write_mjpeg_60fps(&path);
        let mut decoder = VideoDecoder::open_software(&path).unwrap();
        let gate = Arc::new(Mutex::new(()));
        decoder.pending_hw_init = Some(Arc::clone(&gate));

        decoder.width += 2;
        assert!(decoder.next_frame().is_err());
        assert!(decoder.pending_hw_init.is_some());
        decoder.width -= 2;
        assert!(decoder.next_frame().unwrap().is_some());
        assert!(decoder.pending_hw_init.is_none());
        assert_eq!(Arc::strong_count(&gate), 1);
        while decoder.next_frame().unwrap().is_some() {}

        decoder.pending_hw_init = Some(Arc::clone(&gate));
        assert!(decoder.next_frame().unwrap().is_none());
        assert!(decoder.pending_hw_init.is_none());
        assert_eq!(Arc::strong_count(&gate), 1);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn software_frame_on_hardware_accessor_is_an_error_not_eof() {
        ffmpeg::init().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("software.mov");
        write_mjpeg_60fps(&path);
        let mut decoder = super::VideoDecoder::open_software(&path).unwrap();
        // A hardware device can be selected even for a software-only codec.
        // Exercise that boundary without requiring a GPU on the test host.
        decoder.backend = super::DecodeBackend::Cuda;
        assert!(matches!(
            decoder.next_frame_gpu(),
            Err(super::DecodeError::UnexpectedFrameFormat { .. })
        ));
    }

    /// MJPEG carries no timing in its bitstream, so on decode the codec
    /// context's framerate is unset and only the container knows the rate -
    /// the same shape as camera-original HEVC without VUI timing, which
    /// used to hit the silent 30fps default (60fps club footage exported
    /// at 30fps).
    #[test]
    fn probes_container_rate_when_bitstream_has_no_timing() {
        ffmpeg::init().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mjpeg60.mov");
        write_mjpeg_60fps(&path);

        let dec = super::VideoDecoder::open(&path).unwrap();
        assert_eq!(dec.frame_rate(), Some(R60));
    }

    #[test]
    fn missing_timing_does_not_invent_a_rate_or_prevent_decoding() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mjpeg60.mov");
        write_mjpeg_60fps(&path);
        let mut decoder = super::VideoDecoder::open_software(&path).unwrap();
        let mut stream = decoder
            .input
            .stream_mut(decoder.video_stream_index)
            .unwrap();
        stream.set_rate(UNSET);
        stream.set_avg_frame_rate(UNSET);
        decoder.decoder.set_frame_rate(None::<ffmpeg::Rational>);
        assert_eq!(decoder.frame_rate(), None);
        assert_eq!(decoder.fps(), None);
        assert!(decoder.next_frame().unwrap().is_some());
    }

    #[test]
    fn p010_host_extraction_preserves_range_endpoints_and_neutral() {
        use super::{Pixel, VideoFrame, extract_10bit_yuv};
        for (full, y, cb, cr, expected) in [
            (
                false,
                [64_u16, 940],
                [64_u16, 960],
                512_u16,
                [[16, 235], [16, 240], [128, 128]],
            ),
            (
                true,
                [0, 1023],
                [0, 1023],
                512,
                [[0, 255], [0, 255], [128, 128]],
            ),
        ] {
            for (format, big_endian) in [(Pixel::P010LE, false), (Pixel::P010BE, true)] {
                let mut frame = VideoFrame::new(format, 4, 2);
                let bytes = |code| {
                    let value: u16 = code << 6;
                    if big_endian {
                        value.to_be_bytes()
                    } else {
                        value.to_le_bytes()
                    }
                };
                let stride = frame.stride(0);
                for row in 0..2 {
                    for (i, code) in y.into_iter().cycle().take(4).enumerate() {
                        frame.data_mut(0)[row * stride + i * 2..row * stride + i * 2 + 2]
                            .copy_from_slice(&bytes(code));
                    }
                }
                for (i, code) in [cb[0], cr, cb[1], cr].into_iter().enumerate() {
                    frame.data_mut(1)[i * 2..i * 2 + 2].copy_from_slice(&bytes(code));
                }
                let (mut y, mut cb, mut cr) = (Vec::new(), Vec::new(), Vec::new());
                extract_10bit_yuv(&frame, full, [&mut y, &mut cb, &mut cr]);
                assert_eq!(y, expected[0].repeat(4));
                assert_eq!(cb, expected[1]);
                assert_eq!(cr, expected[2]);
            }
        }
    }

    #[test]
    fn cpu_extraction_preserves_full_range_samples() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("full-range.mov");
        write_mjpeg_60fps(&path);
        let mut decoder = super::VideoDecoder::open_software(&path).unwrap();
        assert!(decoder.is_full_range());
        let frame = decoder.next_frame().unwrap().unwrap();
        assert!(
            frame.y.iter().all(|&value| value.abs_diff(32) <= 1),
            "full-range samples changed: {}",
            frame.y[0]
        );
        use reco_core::source::FrameSource;
        let mut source = crate::adapters::FfmpegFileSource::open_inputs(
            &[crate::InputPath::Single(path)],
            0,
            true,
        )
        .unwrap();
        assert!(source.is_full_range());
        let reco_core::source::FrameSet::Yuv420p(planes) =
            source.next_frame().unwrap().unwrap().images
        else {
            panic!("expected CPU planes")
        };
        assert!(planes[0].y.iter().all(|&value| value.abs_diff(32) <= 1));
        assert!(frame.u.iter().all(|&value| value.abs_diff(64) <= 1));
        assert!(frame.v.iter().all(|&value| value.abs_diff(192) <= 1));
        assert!(planes[0].u.iter().all(|&value| value.abs_diff(64) <= 1));
        assert!(planes[0].v.iter().all(|&value| value.abs_diff(192) <= 1));
    }

    #[test]
    fn cpu_source_rejects_mixed_camera_ranges_in_both_orders() {
        let dir = tempfile::tempdir().unwrap();
        let full = dir.path().join("full.mov");
        let limited = dir.path().join("limited.mp4");
        write_mjpeg_60fps(&full);
        let config = crate::ffmpeg::encoder::EncoderConfig {
            encoder_name: Some("libx264".into()),
            ..Default::default()
        };
        let mut encoder =
            crate::ffmpeg::encoder::VideoEncoder::new(&limited, 64, 64, R60, &config).unwrap();
        encoder
            .write_nv12_frame(&[128; 64 * 64 * 3 / 2], None)
            .unwrap();
        encoder.finish().unwrap();
        for paths in [[full.clone(), limited.clone()], [limited, full]] {
            let inputs = paths.map(crate::InputPath::Single);
            let error = crate::adapters::FfmpegFileSource::open_inputs(&inputs, 0, true)
                .err()
                .expect("mixed range must fail");
            assert!(error.to_string().contains("color ranges differ"));
        }
    }

    pub(super) fn write_mjpeg_60fps(path: &std::path::Path) {
        use ffmpeg::codec;

        let mut octx = ffmpeg::format::output(path).unwrap();
        let mjpeg = ffmpeg::encoder::find(codec::Id::MJPEG).unwrap();
        let mut ost = octx.add_stream(mjpeg).unwrap();
        let mut enc = codec::context::Context::new_with_codec(mjpeg)
            .encoder()
            .video()
            .unwrap();
        enc.set_width(64);
        enc.set_height(64);
        enc.set_format(ffmpeg::format::Pixel::YUVJ420P);
        enc.set_frame_rate(Some(R60));
        let time_base = ffmpeg::Rational(R60.1, R60.0);
        enc.set_time_base(time_base);
        ost.set_parameters(&enc);
        let mut enc = enc.open().unwrap();
        ost.set_parameters(&enc);
        octx.write_header().unwrap();
        let stream_time_base = octx.stream(0).unwrap().time_base();

        let mut frame = ffmpeg::util::frame::Video::new(ffmpeg::format::Pixel::YUVJ420P, 64, 64);
        for (plane, value) in [32, 64, 192].into_iter().enumerate() {
            frame.data_mut(plane).fill(value);
        }
        for i in 0..6 {
            frame.set_pts(Some(i));
            enc.send_frame(&frame).unwrap();
            drain(&mut enc, &mut octx, time_base, stream_time_base);
        }
        enc.send_eof().unwrap();
        drain(&mut enc, &mut octx, time_base, stream_time_base);
        octx.write_trailer().unwrap();
    }

    fn drain(
        enc: &mut ffmpeg::encoder::video::Encoder,
        octx: &mut ffmpeg::format::context::Output,
        encoder_time_base: ffmpeg::Rational,
        stream_time_base: ffmpeg::Rational,
    ) {
        let mut packet = ffmpeg::Packet::empty();
        while enc.receive_packet(&mut packet).is_ok() {
            packet.set_stream(0);
            // Without an explicit duration the mov muxer guesses the last
            // sample's, skewing the demuxed average rate (6 frames / 5
            // ticks = 72fps).
            packet.set_duration(1);
            packet.rescale_ts(encoder_time_base, stream_time_base);
            packet.write_interleaved(octx).unwrap();
        }
    }
}
