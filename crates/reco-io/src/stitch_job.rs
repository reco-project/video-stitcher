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
use reco_core::session::{FrameProgress, StitchSession};
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

    // Processing settings
    max_frames: Option<u64>,
    duration: Option<f64>,
    start_frame: u64,
    sync_offset: Option<i64>,
    blend_width: f32,

    // Callbacks
    on_progress: Option<ProgressCallback>,
    on_session: Option<SessionCallback>,
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

impl InputPath {
    /// Get the primary file path (first segment for chained inputs).
    fn primary(&self) -> &Path {
        match self {
            Self::Single(p) => p,
            Self::Chained(v) => &v[0],
        }
    }
}

/// Result of a completed stitch job.
#[derive(Debug)]
pub struct StitchResult {
    /// Number of frames processed.
    pub frames_processed: u64,
    /// Total wall-clock time.
    pub elapsed: Duration,
    /// GPU used for rendering.
    pub gpu_name: String,
    /// Encoder used (e.g. "h264_nvenc", "libx264").
    pub encoder_name: String,
    /// Decode mode (e.g. "GPU zero-copy (CUDA/Vulkan)", "CPU upload").
    pub decode_mode: String,
}

impl StitchResult {
    /// Average frames per second.
    pub fn fps(&self) -> f64 {
        self.frames_processed as f64 / self.elapsed.as_secs_f64()
    }
}

/// Errors from [`StitchJob::run`].
#[derive(Debug, thiserror::Error)]
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
    Session(#[from] reco_core::session::SessionError),
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
            max_frames: None,
            duration: None,
            start_frame: 0,
            sync_offset: None,
            blend_width: 0.15,
            on_progress: None,
            on_session: None,
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

    // ── Processing settings ──

    /// Limit the number of frames to process.
    pub fn max_frames(mut self, n: u64) -> Self {
        self.max_frames = Some(n);
        self
    }

    /// Limit processing to a duration in seconds.
    pub fn duration(mut self, secs: f64) -> Self {
        self.duration = Some(secs);
        self
    }

    /// Skip `n` frames from each input before the first output frame.
    ///
    /// Combine with [`max_frames`](Self::max_frames) or
    /// [`duration`](Self::duration) to select a time window, e.g. export
    /// "0:15 - 0:30" as `.start_frame(450).duration(15.0)` at 30fps.
    ///
    /// Skipped frames are decoded and dropped (no seek), so latency is
    /// proportional to decode rate. Default: `0`.
    pub fn start_frame(mut self, n: u64) -> Self {
        self.start_frame = n;
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
    /// use reco_autocam::setup_autocam;
    ///
    /// StitchJob::new("left.mp4", "right.mp4", "match.json", "out.mp4")
    ///     .on_session(|session, source| {
    ///         let info = source.info();
    ///         setup_autocam(session, "model.onnx", info.width, info.height,
    ///             info.fps as f32, source.is_gpu_resident(), 3, 0.5,
    ///             TrackingMode::Ball, None, false).ok();
    ///     })
    ///     .run(&interrupted)?;
    /// ```
    pub fn on_session(
        mut self,
        cb: impl FnOnce(&mut StitchSession, &dyn FrameSource) + Send + 'static,
    ) -> Self {
        self.on_session = Some(Box::new(cb));
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

        // Open source with auto GPU detection
        let mut source = crate::SmartFileSource::open(
            self.left.primary(),
            self.right.primary(),
            &gpu,
            effective_sync,
        )?;
        let info = source.info();
        let (out_w, out_h) = self.resolution.unwrap_or((1920, 1080));
        if self.resolution.is_none() {
            log::info!("Output resolution not specified, defaulting to {out_w}x{out_h}");
        }
        let decode_mode = source.decode_mode().to_string();

        // Determine input format from source capabilities
        let input_format = if source.is_gpu_resident() {
            reco_core::renderer::InputFormat::Nv12
        } else {
            reco_core::renderer::InputFormat::Yuv420p
        };

        // Build session
        let viewport = reco_core::viewport::ViewportConfig {
            width: out_w,
            height: out_h,
            blend_width: self.blend_width,
            rig_tilt: cal.rig_tilt as f32,
            rig_roll: cal.rig_roll as f32,
            ..Default::default()
        };
        let session_config = reco_core::session::SessionConfig {
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

        // Configure GPU bind groups if source is GPU-resident
        #[cfg(target_os = "linux")]
        if let Some(shared) = source.shared_texture_set() {
            session.setup_gpu_source(shared);
        }

        // Call the session callback for consumer configuration (e.g. autocam)
        if let Some(cb) = self.on_session.take() {
            cb(&mut session, &source);
        }

        // Create encoder with optional audio passthrough.
        let fps_rational = info.fps_rational.unwrap_or_else(|| {
            log::warn!("FPS not available from source metadata, defaulting to 30fps");
            (30, 1)
        });
        let (codec_str, quality_str) = map_output_config(&self.codec, &self.bitrate);

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
            codec: crate::ffmpeg::encoder::VideoCodec::from_str_loose(codec_str)
                .unwrap_or_default(),
            quality: match quality_str {
                "fast" => crate::ffmpeg::encoder::Quality::Fast,
                "high" => crate::ffmpeg::encoder::Quality::High,
                _ => crate::ffmpeg::encoder::Quality::Balanced,
            },
            crf: self.crf,
            preset: self.preset.clone(),
            audio_source,
        };
        let encoder = crate::adapters::FfmpegFileEncoder::new(
            &self.output,
            out_w,
            out_h,
            (fps_rational.0, fps_rational.1),
            &enc_config,
        )?;
        let enc_name = encoder.encoder_name().to_string();
        session.set_encoder(Box::new(encoder), 2);

        // Drain-and-discard frames up to start_frame.
        // Done here (before session.run) so the session's frame counter
        // and progress callback reflect the exported window, not the skip.
        if self.start_frame > 0 {
            log::info!("skipping first {} frames before export", self.start_frame);
            use reco_core::source::FrameSource as _;
            for skipped in 0..self.start_frame {
                if interrupted.load(Ordering::Relaxed) {
                    return Err(StitchError::Other(
                        "cancelled during start_frame skip".into(),
                    ));
                }
                match source.next_frame()? {
                    Some(_) => {} // drop the frame
                    None => {
                        log::warn!(
                            "start_frame={} exceeded source length (stopped at {})",
                            self.start_frame,
                            skipped,
                        );
                        break;
                    }
                }
            }
        }

        // Compute frame limit
        let frame_limit =
            reco_core::session::compute_frame_limit(self.duration, self.max_frames, info.fps);

        // Run the frame loop
        let frame_count = session.run(
            &mut source,
            frame_limit,
            interrupted,
            self.on_progress.take(),
        )?;
        session.finish()?;

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
        })
    }
}

/// Map our output config types to the current encoder's string-based API.
/// This bridges the new OutputConfig types with the existing create_encoder.
fn map_output_config(codec: &Codec, bitrate: &Bitrate) -> (&'static str, &'static str) {
    let codec_str = match codec {
        Codec::H264 => "h264",
        Codec::HEVC => "hevc",
        Codec::AV1 => "av1",
    };
    let quality_str = match bitrate {
        Bitrate::Quality(Quality::Fast) => "fast",
        Bitrate::Quality(Quality::Balanced) => "balanced",
        Bitrate::Quality(Quality::High) => "high",
        // For explicit bitrate control, use balanced preset and let the
        // encoder backend handle the rate control. Full bitrate support
        // requires updating the encoder API (future work).
        _ => "balanced",
    };
    (codec_str, quality_str)
}
