//! One-shot file-to-file stitching (Layer 3 API).
//!
//! [`StitchJob`] is the simplest way to stitch two video files into a
//! panoramic output. It handles all orchestration internally: GPU
//! initialization, zero-copy detection, encoder creation, decode thread
//! management, and audio passthrough.
//!
//! # Example
//!
//! ```rust,ignore
//! use reco_io::StitchJob;
//! use reco_io::output::{Codec, Quality};
//!
//! StitchJob::new("left.mp4", "right.mp4", "match.json", "output.mp4")
//!     .codec(Codec::HEVC)
//!     .quality(Quality::High)
//!     .on_progress(|p| println!("{:.0}%", p.percent()))
//!     .run(&interrupted)?;
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::output::{AudioMode, Bitrate, Codec, Format, Quality};
use reco_core::session::StitchSession;
use reco_core::session::types::FrameProgress;
use reco_core::source::FrameSource;

/// One-shot stitching job: video files in, encoded video out.
///
/// Use the builder methods to configure output settings, then call
/// [`run`](Self::run) to execute. All GPU, encoder, and decode lifecycle
/// is managed internally.
pub struct StitchJob {
    left: InputPath,
    right: InputPath,
    calibration: CalibrationSource,
    output: PathBuf,

    // Output settings
    codec: Codec,
    bitrate: Bitrate,
    format: Format,
    audio: AudioMode,
    resolution: Option<(u32, u32)>,
    encoder_name: Option<String>,
    crf: Option<u8>,
    preset: Option<String>,

    // Processing window
    start_time: Option<f64>,
    end_time: Option<f64>,
    max_frames: Option<u64>,
    sync_offset: Option<i64>,
    blend_width: f32,

    // Callbacks
    on_progress: Option<ProgressCallback>,
    // Vec so multiple consumers can hook the session (telemetry, autocam,
    // future dual-output). A trait-based hook system would be cleaner for
    // complex composition but Vec<FnOnce> is sufficient for now.
    session_hooks: Vec<SessionCallback>,

    // Replay recording (M6.5 stacked-video). Opt-in, gated by the
    // `stacked-output` feature so consumers not building it pay
    // nothing.
    #[cfg(feature = "stacked-output")]
    replay_recording: Option<ReplayRecordingConfig>,

    /// Force CPU decode instead of GPU zero-copy. Needed when ORT
    /// CPU detection is wanted but TensorRT is not available.
    force_cpu_decode: bool,

    /// Path for pipeline event JSONL output. When set, attaches a
    /// `JsonlSink` to the session that records every detection,
    /// filter decision, and pan decision for offline analysis.
    events_path: Option<std::path::PathBuf>,
}

/// Configuration for optional replay recording (see
/// [`StitchJob::with_replay_recording`]). Stored rather than eagerly
/// opened so `StitchJob::run` is the single allocation / error
/// boundary for the replay file.
#[cfg(feature = "stacked-output")]
struct ReplayRecordingConfig {
    path: PathBuf,
    encoder_config: Box<crate::stacked_video::encoder::StackedEncoderConfig>,
    /// Optional replay tile downscale `(width, height)`. When
    /// `None`, replay tiles match the source tile dims. When
    /// `Some`, the GPU pack shader downscales each tile via the
    /// sampler's linear filter (free on the GPU path) and the CPU
    /// decorator rejects the config with a warn (CPU path has no
    /// free downscale today; use the GPU path for A19). FRICTION
    /// reco-obs A19.
    scale: Option<(u32, u32)>,
}

/// Boxed progress callback type alias to satisfy clippy::type_complexity.
type ProgressCallback = Box<dyn FnMut(&FrameProgress) + Send>;

/// Boxed session callback type alias to satisfy clippy::type_complexity.
type SessionCallback = Box<dyn FnOnce(&mut StitchSession, &dyn FrameSource) + Send>;

/// Where to load calibration from.
enum CalibrationSource {
    /// Load from a JSON file path.
    File(PathBuf),
    /// Use an in-memory calibration (no file I/O).
    Memory(Box<reco_core::calibration::MatchCalibration>),
}

/// Input video file path(s), supporting chained/segmented recordings.
#[derive(Debug, Clone)]
pub enum InputPath {
    /// A single video file.
    Single(PathBuf),
    /// Multiple segments that form one continuous recording (e.g. DJI 4GB splits).
    Chained(Vec<PathBuf>),
}

impl InputPath {
    /// Get the first file path (for audio extraction, probing, etc.).
    pub fn first_path(&self) -> &Path {
        match self {
            Self::Single(p) => p,
            Self::Chained(v) => &v[0],
        }
    }
}

impl From<&str> for InputPath {
    fn from(s: &str) -> Self {
        Self::Single(PathBuf::from(s))
    }
}

impl From<&Path> for InputPath {
    fn from(p: &Path) -> Self {
        Self::Single(p.to_path_buf())
    }
}

impl From<PathBuf> for InputPath {
    fn from(p: PathBuf) -> Self {
        Self::Single(p)
    }
}

impl<P: AsRef<Path>> From<Vec<P>> for InputPath {
    fn from(paths: Vec<P>) -> Self {
        Self::Chained(paths.iter().map(|p| p.as_ref().to_path_buf()).collect())
    }
}

/// Result of a completed stitch job.
#[derive(Debug)]
pub struct StitchResult {
    pub frames_processed: u64,
    pub elapsed: Duration,
    pub gpu_name: String,
    pub encoder_name: String,
    pub decode_mode: String,
    pub telemetry: Option<reco_core::telemetry::TelemetrySnapshot>,
}

impl StitchResult {
    /// Average frames per second.
    pub fn fps(&self) -> f64 {
        self.frames_processed as f64 / self.elapsed.as_secs_f64()
    }
}

/// Errors from [`StitchJob::run`].
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum StitchError {
    /// Calibration file could not be loaded or parsed.
    #[error("calibration: {0}")]
    Calibration(String),
    /// Source video could not be opened.
    #[error("source: {0}")]
    Source(#[from] reco_core::source::SourceError),
    /// GPU initialization failed.
    #[error("GPU: {0}")]
    Gpu(#[from] reco_core::gpu::GpuError),
    /// Session/pipeline error during stitching.
    #[error("session: {0}")]
    Session(#[from] reco_core::session::types::SessionError),
    /// Encoder error.
    #[error("encoder: {0}")]
    Encoder(#[from] reco_core::encoder::EncodeError),
    /// Encoder completed without error but the output file has no
    /// usable video stream. Typical cause: an encoder ffmpeg listed as
    /// available silently rejected every frame (e.g. AV1/NVENC on
    /// pre-Ada hardware). The file usually still has the audio stream.
    #[error(
        "encoder produced no video frames (output={}): the selected codec may not be supported on this hardware - try a different codec",
        path
    )]
    EmptyOutput {
        /// Path to the output file that was created but contains no
        /// video.
        path: String,
    },
    /// I/O or other error.
    #[error("{0}")]
    Other(String),
}

impl StitchJob {
    /// Create a job from file paths (loads calibration from JSON).
    pub fn new(
        left: impl Into<InputPath>,
        right: impl Into<InputPath>,
        calibration: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Self {
        Self {
            left: left.into(),
            right: right.into(),
            calibration: CalibrationSource::File(calibration.as_ref().to_path_buf()),
            output: output.as_ref().to_path_buf(),
            codec: Codec::default(),
            bitrate: Bitrate::default(),
            format: Format::default(),
            audio: AudioMode::default(),
            resolution: None,
            encoder_name: None,
            crf: None,
            preset: None,
            start_time: None,
            end_time: None,
            max_frames: None,
            sync_offset: None,
            blend_width: 0.15,
            on_progress: None,
            session_hooks: Vec::new(),
            #[cfg(feature = "stacked-output")]
            replay_recording: None,
            force_cpu_decode: false,
            events_path: None,
        }
    }

    /// Create a job with in-memory calibration (no JSON file needed).
    pub fn with_calibration(
        left: impl Into<InputPath>,
        right: impl Into<InputPath>,
        calibration: reco_core::calibration::MatchCalibration,
        output: impl AsRef<Path>,
    ) -> Self {
        let mut job = Self::new(left, right, Path::new(""), output);
        job.calibration = CalibrationSource::Memory(Box::new(calibration));
        job
    }

    // ── Output settings ──

    /// Set the output video codec.
    pub fn codec(mut self, codec: Codec) -> Self {
        self.codec = codec;
        self
    }

    /// Set the bitrate control strategy.
    pub fn bitrate(mut self, bitrate: Bitrate) -> Self {
        self.bitrate = bitrate;
        self
    }

    /// Set the output quality (convenience for `bitrate(Bitrate::Quality(...))`).
    pub fn quality(mut self, quality: Quality) -> Self {
        self.bitrate = Bitrate::Quality(quality);
        self
    }

    /// Set the output container format.
    pub fn format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }

    /// Set the output resolution. Default: match input resolution.
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = Some((width, height));
        self
    }

    /// Set the audio mode. Default: copy audio from the first input.
    pub fn audio(mut self, mode: AudioMode) -> Self {
        self.audio = mode;
        self
    }

    /// Force a specific encoder by name (e.g. `"h264_nvenc"`, `"libx264"`).
    pub fn encoder_name(mut self, name: impl Into<String>) -> Self {
        self.encoder_name = Some(name.into());
        self
    }

    /// Override the CRF/quality value (passed through to the encoder).
    pub fn crf(mut self, crf: u8) -> Self {
        self.crf = Some(crf);
        self
    }

    /// Override the encoder preset string (passed through to the encoder).
    pub fn preset(mut self, preset: impl Into<String>) -> Self {
        self.preset = Some(preset.into());
        self
    }

    // ── Processing window ──

    /// Start processing at a time offset (seconds). Default: beginning.
    ///
    /// Converted to a frame index internally using the source's frame
    /// rate (rounded to the nearest frame).
    pub fn start_time(mut self, secs: f64) -> Self {
        self.start_time = Some(secs);
        self
    }

    /// Stop processing at a time offset (seconds). Default: end of source.
    ///
    /// Must be greater than `start_time` when both are set.
    pub fn end_time(mut self, secs: f64) -> Self {
        self.end_time = Some(secs);
        self
    }

    /// Hard cap on the number of output frames. Default: no limit.
    ///
    /// Applied after the time window: if `end_time` implies 900 frames
    /// but `max_frames` is 300, only 300 are produced.
    pub fn max_frames(mut self, n: u64) -> Self {
        self.max_frames = Some(n);
        self
    }

    /// Override the temporal sync offset between cameras (frames).
    /// Positive: right camera started first. Negative: left started first.
    /// Default: use the value from calibration.
    pub fn sync_offset(mut self, frames: i64) -> Self {
        self.sync_offset = Some(frames);
        self
    }

    /// Set the blend width for seam blending (0.0 - 1.0). Default: 0.15.
    pub fn blend_width(mut self, blend: f32) -> Self {
        self.blend_width = blend;
        self
    }

    // ── Replay recording (M6.5 stacked-video) ──

    /// Record pre-stitch source frames to a stacked-video file at
    /// `path` while the job runs. The file is a grid-layout video
    /// (vertical stack for the default N=2 layout) that can be read
    /// back via [`crate::stacked_video::source::StackedSource`] for
    /// professional replay, web panorama generation, or cloud
    /// deployment.
    ///
    /// Defaults to Matroska container, libx264, 30-frame GOP - see
    /// [`crate::stacked_video::encoder::StackedEncoderConfig`] for
    /// the full default and use [`Self::replay_recording_config`]
    /// to override.
    ///
    /// Recording is best-effort and never fails the stitch run: if
    /// the encoder can't keep up, the replay file gracefully stops
    /// recording while the stitch output completes.
    ///
    /// # Pack path
    ///
    /// At [`Self::run`] time, `StitchJob` picks between two pack
    /// paths based on whether the source delivers frames on CPU or
    /// GPU. The choice is logged explicitly once per run so it's
    /// never a silent decision:
    ///
    /// - **GPU path** (`source.is_gpu_resident() == true`): the
    ///   stacked-video compute shader packs the same GPU textures
    ///   the stitch pipeline samples, then triple-buffers a
    ///   readback of the packed atlas. Used by NVDEC zero-copy
    ///   today; Metal / Jetson ISP will follow the same path once
    ///   wired. No extra upload, no CPU memcpy.
    /// - **CPU path** (default for software-decoded YUV): wraps
    ///   the source in
    ///   [`crate::stacked_video::replay::ReplayRecordingSource`]
    ///   which packs YUV planes on the CPU. Uploading them just
    ///   to pack would lose to this path.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// StitchJob::new("left.mp4", "right.mp4", "match.json", "out.mp4")
    ///     .with_replay_recording("replay.mkv")
    ///     .run(&interrupted)?;
    /// ```
    #[cfg(feature = "stacked-output")]
    pub fn with_replay_recording(mut self, path: impl AsRef<Path>) -> Self {
        self.replay_recording = Some(ReplayRecordingConfig {
            path: path.as_ref().to_path_buf(),
            encoder_config: Box::new(
                crate::stacked_video::encoder::StackedEncoderConfig::default(),
            ),
            scale: None,
        });
        self
    }

    /// Set the replay tile downscale (FRICTION reco-obs A19). Must
    /// be called AFTER [`Self::with_replay_recording`] — no-op with
    /// a warn if replay recording wasn't enabled first.
    ///
    /// `(width, height)` are the per-tile dims after scale; the
    /// atlas height becomes `height * N` for an N-vstack layout. A
    /// 1080p source with `.with_replay_scale(1280, 720)` produces
    /// a 1280x1440 atlas (two 720p tiles stacked).
    ///
    /// The GPU pack path folds this into the compute shader's
    /// sampler at no extra cost (linear filter handles the scale).
    /// The CPU path doesn't support downscale today — enabling it
    /// on a CPU-resident source logs a warn and the replay file
    /// keeps the source dims.
    ///
    /// Dimensions must be YUV420P-aligned: `width` divisible by 4
    /// (pack shader quirk), `height` even.
    /// Force CPU decode instead of GPU zero-copy (NVDEC). Required
    /// when AI tracking uses ORT CPU detection and TensorRT is not
    /// available. CPU decode is ~5-10x slower but gives the detector
    /// access to the frames.
    pub fn force_cpu_decode(mut self) -> Self {
        self.force_cpu_decode = true;
        self
    }

    /// Record pipeline events (detections, filter decisions, pan
    /// decisions) to a JSONL file for offline analysis.
    pub fn events(mut self, path: impl AsRef<Path>) -> Self {
        self.events_path = Some(path.as_ref().to_path_buf());
        self
    }

    #[cfg(feature = "stacked-output")]
    pub fn with_replay_scale(mut self, width: u32, height: u32) -> Self {
        if let Some(ref mut cfg) = self.replay_recording {
            cfg.scale = Some((width, height));
        } else {
            log::warn!(
                "with_replay_scale({width}x{height}) called without with_replay_recording \
                 - ignored; call with_replay_recording(path) first"
            );
        }
        self
    }

    /// Override the encoder configuration used for replay recording.
    /// Only meaningful in combination with [`Self::with_replay_recording`].
    #[cfg(feature = "stacked-output")]
    pub fn replay_recording_config(
        mut self,
        config: crate::stacked_video::encoder::StackedEncoderConfig,
    ) -> Self {
        if let Some(ref mut cfg) = self.replay_recording {
            *cfg.encoder_config = config;
        } else {
            log::warn!("replay_recording_config called without with_replay_recording - ignored");
        }
        self
    }

    // ── Callbacks ──

    /// Set a progress callback. Called periodically during processing.
    pub fn on_progress(mut self, cb: impl FnMut(&FrameProgress) + Send + 'static) -> Self {
        self.on_progress = Some(Box::new(cb));
        self
    }

    /// Hook called after the session is created but before the frame loop.
    ///
    /// Use this to attach a detector, director, or other session configuration
    /// that requires access to the session and source metadata (dimensions, fps).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use reco_autocam::{AutocamConfig, TrackingMode};
    ///
    /// StitchJob::new("left.mp4", "right.mp4", "match.json", "out.mp4")
    ///     .on_session(|session, source| {
    ///         let config = AutocamConfig::new("model.onnx")
    ///             .with_tracking_mode(TrackingMode::Field)
    ///             .with_detection_interval(3);
    ///         reco_autocam::setup_autocam(session, &config, source.info().fps as f32, source.is_gpu_resident()).ok();
    ///     })
    ///     .run(&interrupted)?;
    /// ```
    pub fn on_session(
        mut self,
        cb: impl FnOnce(&mut StitchSession, &dyn FrameSource) + Send + 'static,
    ) -> Self {
        self.session_hooks.push(Box::new(cb));
        self
    }

    // ── Execute ──

    /// Run the stitching job.
    ///
    /// This is a blocking call that processes all frames (or until the
    /// interrupt flag is set). Returns a [`StitchResult`] with statistics.
    pub fn run(mut self, interrupted: &AtomicBool) -> Result<StitchResult, StitchError> {
        crate::init();
        let start = std::time::Instant::now();

        // Load calibration
        let cal = match self.calibration {
            CalibrationSource::File(ref path) => {
                reco_core::calibration::MatchCalibration::from_file(path)
                    .map_err(|e| StitchError::Calibration(format!("{e}")))?
            }
            CalibrationSource::Memory(cal) => *cal,
        };
        let effective_sync = self.sync_offset.unwrap_or(cal.sync_offset);
        if self.sync_offset.is_none() && cal.sync_offset != 0 {
            log::info!("Sync offset: {} frames (from calibration)", effective_sync);
        }

        // Initialize GPU
        let gpu = reco_core::gpu::GpuContext::new_blocking()?;
        let gpu_name = gpu.gpu_name().to_string();

        log::info!("StitchJob::run: force_cpu_decode={}", self.force_cpu_decode);
        let mut source = if self.force_cpu_decode {
            log::info!("Force CPU decode: zero-copy disabled by --no-zero-copy");
            crate::SmartFileSource::open_cpu_only(&self.left, &self.right, effective_sync)?
        } else {
            crate::SmartFileSource::open(&self.left, &self.right, &gpu, effective_sync)?
        };
        let info = source.info();
        let (out_w, out_h) = self.resolution.unwrap_or((1920, 1080));
        if self.resolution.is_none() {
            log::info!("Output resolution not specified, defaulting to {out_w}x{out_h}");
        }
        let decode_mode = source.decode_mode().to_string();

        // Determine input format from source capabilities
        let input_format = if source.is_gpu_resident() {
            reco_core::render::renderer::InputFormat::Nv12
        } else {
            reco_core::render::renderer::InputFormat::Yuv420p
        };

        // Build session
        let viewport = reco_core::render::viewport::ViewportConfig {
            width: out_w,
            height: out_h,
            blend_width: self.blend_width,
            rig_tilt: cal.rig_tilt as f32,
            rig_roll: cal.rig_roll as f32,
            ..Default::default()
        };
        let session_config = reco_core::session::types::SessionConfig {
            calibration: cal,
            viewport,
            input_width: info.width,
            input_height: info.height,
            output_format: reco_core::gpu::OutputFormat::Rgba8Unorm,
            input_format,
            left_rotation: source.left_rotation(),
            right_rotation: source.right_rotation(),
        };
        let mut session = reco_core::session::StitchSession::with_gpu(gpu, session_config)?;

        session.telemetry_mut().set_gpu_name(gpu_name.clone());
        session.telemetry_mut().set_decode_mode(decode_mode.clone());

        // Configure GPU bind groups if source is GPU-resident
        #[cfg(target_os = "linux")]
        if let Some(shared) = source.shared_texture_set() {
            session.setup_gpu_source(shared);
        }

        for hook in self.session_hooks.drain(..) {
            hook(&mut session, &source);
        }

        // Start decode threads now that hooks (ORT/DML init) have completed.
        source.start_decoding();

        // Attach JSONL event sink if requested.
        if let Some(ref events_path) = self.events_path {
            match crate::jsonl_sink::JsonlSink::create(events_path) {
                Ok(sink) => {
                    log::info!("Pipeline events -> {}", events_path.display());
                    session.set_event_sink(Box::new(sink));
                }
                Err(e) => {
                    log::warn!("Failed to open events file {}: {e}", events_path.display());
                }
            }
        }

        // Create encoder with optional audio passthrough.
        let fps_rational = info.fps_rational.unwrap_or_else(|| {
            log::warn!("FPS not available from source metadata, defaulting to 30fps");
            (30, 1)
        });
        let quality = match &self.bitrate {
            Bitrate::Quality(q) => crate::ffmpeg::encoder::Quality::from(*q),
            Bitrate::Crf(_) => crate::ffmpeg::encoder::Quality::Balanced,
        };

        // Resolve audio source path from AudioMode.
        let audio_source = match &self.audio {
            AudioMode::CopyFrom(0) => Some(self.left.first_path().to_path_buf()),
            AudioMode::CopyFrom(1) => Some(self.right.first_path().to_path_buf()),
            AudioMode::CopyFrom(n) => {
                log::warn!("AudioMode::CopyFrom({n}) - only 0 (left) and 1 (right) are valid");
                None
            }
            AudioMode::Disabled => None,
        };

        let enc_config = crate::ffmpeg::encoder::EncoderConfig {
            encoder_name: self.encoder_name.clone(),
            codec: self.codec.into(),
            quality,
            crf: self.crf,
            preset: self.preset.clone(),
            audio_source,
            container: self.format.into(),
            gop_size: None,
            stream_url: None,
        };
        let encoder = crate::adapters::FfmpegFileEncoder::new(
            &self.output,
            out_w,
            out_h,
            (fps_rational.0, fps_rational.1),
            &enc_config,
        )?;
        let enc_name = encoder.encoder_name().to_string();
        session.telemetry_mut().set_encoder_name(enc_name.clone());
        session.set_encoder(Box::new(encoder), 2);

        #[cfg(feature = "stacked-output")]
        if let Some(ref mut cfg) = self.replay_recording
            && cfg.encoder_config.inner.encoder_name.is_none()
        {
            cfg.encoder_config.inner.encoder_name =
                crate::ffmpeg::encoder::VideoEncoder::replay_encoder_name(&enc_name)
                    .map(String::from);
        }

        // Resolve processing window (start_time / end_time / max_frames).
        let fps = if info.fps > 0.0 {
            info.fps as f64
        } else {
            30.0
        };
        let start_secs = self.start_time.unwrap_or(0.0);
        let skip_frames = (start_secs * fps).round() as u64;

        if let Some(end) = self.end_time {
            let end_frames = ((end - start_secs) * fps).round() as i64;
            if end_frames <= 0 {
                return Err(StitchError::Other(format!(
                    "end_time ({end:.2}s) must be greater than start_time ({start_secs:.2}s)"
                )));
            }
        }

        use reco_core::source::FrameSource as _;
        if let Some(total) = source.total_frames()
            && skip_frames >= total
        {
            return Err(StitchError::Other(format!(
                "start_time ({start_secs:.2}s = frame {skip_frames}) \
                 is past the end of the source ({total} frames)"
            )));
        }

        // Skip frames to reach start position.
        if skip_frames > 0 {
            log::info!("skipping {skip_frames} frames (start_time={start_secs:.2}s)");
            for skipped in 0..skip_frames {
                if interrupted.load(Ordering::Relaxed) {
                    return Err(StitchError::Other("cancelled during start skip".into()));
                }
                if source.next_frame()?.is_none() {
                    log::warn!("source ended during skip at frame {skipped}/{skip_frames}");
                    break;
                }
            }
        }

        // Compute frame limit from end_time and max_frames.
        let end_limit = self
            .end_time
            .map(|end| ((end - start_secs) * fps).round() as u64);
        let frame_limit = match (end_limit, self.max_frames) {
            (Some(el), Some(mf)) => {
                let limit = el.min(mf);
                if limit == mf {
                    log::info!("frame limit: {limit} (from max_frames)");
                } else {
                    log::info!(
                        "frame limit: {limit} (from end_time {:.2}s)",
                        self.end_time.unwrap()
                    );
                }
                limit
            }
            (Some(el), None) => {
                log::info!(
                    "frame limit: {el} (from end_time {:.2}s)",
                    self.end_time.unwrap()
                );
                el
            }
            (None, Some(mf)) => {
                log::info!("frame limit: {mf} (from max_frames)");
                mf
            }
            (None, None) => u64::MAX,
        };

        // Optional replay recording: wrap the source with a
        // decorator that writes pre-stitch frames to a stacked-video
        // file alongside the main encode. Replay starts from this
        // point so the recording aligns with the exported window
        // (frames skipped via `start_frame` are already past).
        //
        // Two arms so the replay branch can take ownership of
        // `source` via `Box::new`, while the non-replay arm keeps
        // the borrowed `&mut source` already set up.
        #[cfg(feature = "stacked-output")]
        let replay_cfg = self.replay_recording.take();
        #[cfg(not(feature = "stacked-output"))]
        let replay_cfg: Option<()> = None;

        let frame_count;
        // Tracks a CPU-path replay decorator when that arm is
        // chosen; the finalizer below calls `finish()` after the
        // run. GPU path finalizes via
        // `session.clear_stacked_gpu_recorder()` below.
        #[cfg(feature = "stacked-output")]
        let mut replay_src: Option<crate::stacked_video::replay::ReplayRecordingSource> = None;

        #[cfg(feature = "stacked-output")]
        if let Some(cfg) = replay_cfg {
            // Pack-path selection. GPU-resident sources (NVDEC
            // zero-copy, future Metal / Jetson ISP) send frames
            // through `submit_render_output`; the session's
            // zero-copy driver now taps the pack shader directly
            // against those shared textures (post-#270 wiring in
            // reco-core). CPU-resident sources continue through
            // the `ReplayRecordingSource` decorator: uploading
            // just-to-pack would lose to a straight CPU memcpy.
            //
            // Both arms log the decision explicitly so the pack
            // path is never a silent choice.
            if source.is_gpu_resident() {
                // Resolve the output tile size. `scale` is the
                // per-tile dims after downscale; the layout stays
                // sized by the source tile dims (layout.capacity
                // is N, atlas h = output.height * rows).
                let (out_w, out_h) = cfg.scale.unwrap_or((info.width, info.height));
                let layout = reco_core::gpu::yuv_stack_packer::StackGridLayout::vstack(
                    out_w, out_h, 2,
                )
                .ok_or_else(|| {
                    StitchError::Other(format!(
                        "GPU stacked replay: replay tile dims {out_w}x{out_h} are not YUV420P-aligned \
                         (width must be divisible by 4, height must be even). Source: {}x{}",
                        info.width, info.height,
                    ))
                })?;
                let output_size = if cfg.scale.is_some() {
                    reco_core::gpu::yuv_stack_packer::OutputTileSize::scaled(out_w, out_h)
                } else {
                    reco_core::gpu::yuv_stack_packer::OutputTileSize::unscaled(out_w, out_h)
                };
                session
                    .enable_gpu_stacked_replay(layout, output_size)
                    .map_err(|e| StitchError::Other(format!("enable GPU stacked replay: {e}")))?;
                let (atlas_w, atlas_h) = session.stacked_atlas_dims().ok_or_else(|| {
                    StitchError::Other(
                        "stacked_atlas_dims returned None right after enable; internal bug".into(),
                    )
                })?;
                let recorder = crate::stacked_video::replay::session_gpu_recorder(
                    &cfg.path,
                    *cfg.encoder_config,
                    atlas_w,
                    atlas_h,
                )
                .map_err(|e| StitchError::Other(format!("open GPU replay recorder: {e}")))?;
                session.set_stacked_gpu_recorder(recorder);
                let scale_note = if cfg.scale.is_some() {
                    format!(
                        " [A19 downscale: source {}x{} -> tile {}x{}]",
                        info.width, info.height, out_w, out_h
                    )
                } else {
                    String::new()
                };
                log::info!(
                    "reco-io: replay pack path = GPU (source GPU-resident; tile {}x{}, N=2, atlas {}x{}){} -> {}",
                    out_w,
                    out_h,
                    atlas_w,
                    atlas_h,
                    scale_note,
                    cfg.path.display(),
                );
                frame_count = session.run(
                    &mut source,
                    frame_limit,
                    interrupted,
                    self.on_progress.take(),
                )?;
            } else {
                if cfg.scale.is_some() {
                    log::warn!(
                        "reco-io: --replay-scale requested on a CPU-resident source, but the \
                         CPU ReplayRecordingSource decorator doesn't support downscale today. \
                         Recording at source dims {}x{} instead. Use a GPU-resident source \
                         (zero-copy NVDEC) or disable --replay-scale.",
                        info.width,
                        info.height,
                    );
                }
                log::info!(
                    "reco-io: replay pack path = CPU (source CPU-resident; ReplayRecordingSource decorator, tile {}x{}, N=2) -> {}",
                    info.width,
                    info.height,
                    cfg.path.display(),
                );
                let inner: Box<dyn FrameSource> = Box::new(source);
                let mut replay = crate::stacked_video::replay::ReplayRecordingSource::wrap(
                    inner,
                    &cfg.path,
                    *cfg.encoder_config,
                )
                .map_err(|e| StitchError::Other(format!("replay recording open: {e}")))?;
                frame_count = session.run(
                    &mut replay,
                    frame_limit,
                    interrupted,
                    self.on_progress.take(),
                )?;
                replay_src = Some(replay);
            }
        } else {
            frame_count = session.run(
                &mut source,
                frame_limit,
                interrupted,
                self.on_progress.take(),
            )?;
        }
        #[cfg(not(feature = "stacked-output"))]
        {
            // Without the stacked-output feature, replay recording
            // is unavailable and `replay_cfg` is always None.
            let _ = replay_cfg;
            frame_count = session.run(
                &mut source,
                frame_limit,
                interrupted,
                self.on_progress.take(),
            )?;
        }

        // GPU-path finalize: drop the atlas recorder (its `finish`
        // closes the encoder file) before `session.finish()` so
        // the replay file lands before the main encoder's trailer.
        // No-op when the GPU path wasn't selected.
        #[cfg(feature = "stacked-output")]
        session.clear_stacked_gpu_recorder();
        let telemetry_snap = session.telemetry_snapshot();
        session.finish()?;

        #[cfg(feature = "stacked-output")]
        if let Some(mut replay) = replay_src.take()
            && let Err(e) = replay.finish()
        {
            log::warn!("replay recording finalize failed ({e}); stitch output still valid");
        }

        // Post-run sanity check: re-open the output file and verify it
        // actually contains a video stream. Catches silent encoder
        // failures where ffmpeg accepts every frame but produces a file
        // with only audio (e.g. AV1/NVENC on hardware that doesn't
        // support it). Without this check, the user thinks their export
        // succeeded because StitchResult came back Ok; they only find
        // out when they open the file in a player.
        //
        // Skipped when the caller asked for 0 frames (max_frames=0 or
        // duration=0 short-circuit) because an empty output is the
        // requested outcome.
        if frame_count > 0 {
            let output_path = self.output.clone();
            match crate::ffmpeg::decoder::VideoDecoder::open(&output_path) {
                Ok(d) => {
                    let dur = d.duration_secs().unwrap_or(0.0);
                    if d.width() == 0 || d.height() == 0 || dur <= 0.0 {
                        return Err(StitchError::EmptyOutput {
                            path: output_path.display().to_string(),
                        });
                    }
                }
                Err(e) => {
                    log::warn!(
                        "post-run probe of {} failed ({e}); cannot verify video stream, trusting encoder",
                        output_path.display()
                    );
                }
            }
        }

        Ok(StitchResult {
            frames_processed: frame_count,
            elapsed: start.elapsed(),
            gpu_name,
            encoder_name: enc_name,
            decode_mode,
            telemetry: Some(telemetry_snap),
        })
    }
}
