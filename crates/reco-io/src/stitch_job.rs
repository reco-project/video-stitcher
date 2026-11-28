//! One-shot file or live-input stitching (Layer 3 API).
//!
//! [`StitchJob`] is the simplest way to stitch per-camera video files
//! into a panoramic output. It handles all orchestration internally: GPU
//! initialization (or the all-software path via [`StitchJob::cpu`]),
//! zero-copy detection, encoder creation, decode thread management,
//! and audio passthrough.
//! Without the `gpu` feature, rendering uses the CPU executor; FFmpeg can still use hardware decode and encode. [`StitchJob::cpu`] explicitly forces software codecs as well.
//!
//! SDR exports default to 8-bit H.264. HDR exports require GPU rendering and
//! default to 10-bit HEVC Main 10 with the source colour metadata. Native P010
//! decoding retains 10-bit input precision. Host-backed file decoding supplies
//! 8-bit YUV420P to the renderer.
//!
//! # Example
//!
//! ```rust,ignore
//! use reco_io::StitchJob;
//! use reco_io::output::{Codec, Quality, QualityPreset};
//!
//! StitchJob::new(["left.mp4", "right.mp4"], "match.json", "output.mp4")
//!     .codec(Codec::HEVC)
//!     .quality(Quality::Preset(QualityPreset::High))
//!     .on_progress(|p| println!("{:.0}%", p.percent()))
//!     .run(&interrupted)?;
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use crate::output::{AudioMode, BitDepth, Codec, Container, Quality};
use crate::stacked_video::replay::ReplayRecordingSource;
use reco_core::render::viewport::ViewportSize;
use reco_core::session::StitchSession;
use reco_core::session::types::FrameProgress;
use reco_core::source::FrameSource;

mod input;
#[cfg(test)]
mod source_tests;
use input::Inputs;

/// One-shot stitching job: camera-ordered images in, encoded video out.
///
/// Use the builder methods to configure output settings, then call
/// [`run`](Self::run) to execute. All GPU, encoder, and decode lifecycle
/// is managed internally.
pub struct StitchJob {
    /// One input per camera, in projection order (index i pairs with
    /// `calibration.lenses[i]`). Arity is validated against the
    /// calibration's projection in [`run`](Self::run) before any decoder
    /// spins up.
    inputs: Inputs,
    calibration: CalibrationSource,
    auxiliary_inputs: Vec<PathBuf>,
    output: PathBuf,

    // Output settings
    codec: Option<Codec>,
    quality: Quality,
    container: Option<Container>,
    audio: AudioMode,
    resolution: Option<(u32, u32)>,
    encoder_name: Option<String>,
    encoder_device: Option<String>,
    preset: Option<String>,
    stream_url: Option<String>,

    // Processing window
    start_time: Option<f64>,
    end_time: Option<f64>,
    max_frames: Option<u64>,
    sync_offset: Option<i64>,
    /// Seam blend override. `None` (default) respects the calibration
    /// document's saved value - the single home for render params.
    blend_width: Option<f32>,

    // Callbacks
    on_progress: Option<ProgressCallback>,
    on_finalizing: Option<Box<dyn FnOnce() + Send>>,
    on_stream_ready: Option<Box<dyn FnOnce(crate::ffmpeg::encoder::StreamMonitor) + Send>>,
    // Vec so multiple consumers can hook the session (telemetry, autocam,
    // future dual-output). A trait-based hook system would be cleaner for
    // complex composition but Vec<FnOnce> is sufficient for now.
    session_hooks: Vec<SessionCallback>,

    // Optional replay recording (stacked-video).
    replay_recording: Option<ReplayRecordingConfig>,

    /// Force CPU decode instead of GPU zero-copy. Needed when ORT
    /// CPU detection is wanted but TensorRT is not available.
    force_cpu_decode: bool,

    /// Use software rendering and decoding on systems without a GPU.
    cpu_stitch: bool,

    /// Lookahead buffer depth in seconds. When > 0, the session
    /// decodes N frames ahead to give the panner future context.
    lookahead_secs: f64,
    lookahead_memory_bytes: Option<usize>,

    /// Path for pipeline event JSONL output. When set, attaches a
    /// `JsonlSink` to the session that records every detection,
    /// filter decision, and pan decision for offline analysis.
    events_path: Option<std::path::PathBuf>,
}

/// Configuration for optional replay recording (see
/// [`StitchJob::with_replay_recording`]). Stored rather than eagerly
/// opened so `StitchJob::run` is the single allocation / error
/// boundary for the replay file.
struct ReplayRecordingConfig {
    path: PathBuf,
    encoder_config: Box<crate::stacked_video::encoder::StackedEncoderConfig>,
    /// Optional replay tile downscale `(width, height)`. When
    /// `None`, replay tiles match the source tile dims. When
    /// `Some`, the GPU pack shader downscales each tile via the
    /// sampler's linear filter (free on the GPU path) and the CPU
    /// decorator rejects the config with a warn (the CPU path has no
    /// free downscale).
    scale: Option<(u32, u32)>,
}

/// Boxed progress callback type alias to satisfy clippy::type_complexity.
type ProgressCallback = Box<dyn FnMut(&FrameProgress) + Send>;

/// Boxed session callback type alias to satisfy clippy::type_complexity.
type SessionCallback =
    Box<dyn FnOnce(&mut StitchSession, &dyn FrameSource) -> Result<(), StitchError> + Send>;

/// Where to load calibration from.
enum CalibrationSource {
    /// Load from a JSON file path.
    File(PathBuf),
    /// Use an in-memory calibration (no file I/O).
    Memory(Box<reco_core::calibration::Calibration>),
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

    /// All segment paths in order (one element for `Single`). Used for audio
    /// passthrough, which must span every chained segment, not just the first.
    pub fn all_paths(&self) -> Vec<PathBuf> {
        match self {
            Self::Single(p) => vec![p.clone()],
            Self::Chained(v) => v.clone(),
        }
    }

    /// Build from an ordered, non-empty segment list: one path stays
    /// [`Single`](Self::Single), several chain via the concat demuxer.
    /// The vec-shaped twin of [`Self::parse_segments`] for consumers
    /// that collect paths themselves (file pickers, env specs).
    pub fn from_segments(mut paths: Vec<PathBuf>) -> Self {
        debug_assert!(!paths.is_empty(), "an input needs at least one segment");
        if paths.len() == 1 {
            Self::Single(paths.remove(0))
        } else {
            Self::Chained(paths)
        }
    }

    /// Parse a `;`-separated segment string (`a.mp4;b.mp4;c.mp4`) into
    /// a chained input; a single path stays `Single`. Segments are
    /// trimmed and empty entries (doubled or trailing `;`) dropped, so
    /// `"a.mp4;"` is the single input `a.mp4`, not a path with a
    /// stray semicolon. A string with no usable segment keeps the raw
    /// text so `open` reports it as an invalid path.
    pub fn parse_segments(s: &str) -> Self {
        let parts: Vec<PathBuf> = s
            .split(';')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .collect();
        if parts.is_empty() {
            return Self::Single(PathBuf::from(s.trim()));
        }
        if parts.len() > 1 {
            log::info!(
                "input: {} segments, chaining via concat demuxer",
                parts.len()
            );
        }
        Self::from_segments(parts)
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

/// Statistics and stop reason of a finalized stitch job.
#[derive(Debug)]
pub struct StitchResult {
    /// Stop reason observed before output finalization.
    pub completion: reco_core::session::types::RunCompletion,
    pub frames_processed: u64,
    pub elapsed: Duration,
    pub gpu_name: String,
    pub encoder_name: String,
    pub decode_mode: String,
    pub telemetry: Option<reco_core::telemetry::TelemetrySnapshot>,
    /// Final optional-stream state. A failed stream does not imply that the recording failed.
    pub stream_status: Option<crate::ffmpeg::encoder::StreamStatus>,
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
    #[cfg(feature = "gpu")]
    Gpu(#[from] reco_core::gpu::GpuError),
    /// Session/pipeline error during stitching.
    #[error("session: {0}")]
    Session(#[from] reco_core::session::types::SessionError),
    /// Encoder sink error.
    #[error("encoder: {0}")]
    Encoder(#[from] reco_core::sink::SinkError),
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
    fn resolved_container(&self) -> Container {
        self.container
            .unwrap_or_else(|| Container::for_output(&self.output.to_string_lossy()))
    }

    /// Create a job from file paths (loads calibration from JSON).
    ///
    /// `inputs` is one video per camera in projection order; the
    /// calibration's projection decides how many are required (the
    /// L-shape consumes two, the cylinder one).
    pub fn new(
        inputs: impl IntoIterator<Item = impl Into<InputPath>>,
        calibration: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Self {
        Self {
            inputs: Inputs::Files(inputs.into_iter().map(Into::into).collect()),
            calibration: CalibrationSource::File(calibration.as_ref().to_path_buf()),
            auxiliary_inputs: Vec::new(),
            output: output.as_ref().to_path_buf(),
            codec: None,
            quality: Quality::default(),
            container: None,
            audio: AudioMode::default(),
            resolution: None,
            encoder_name: None,
            encoder_device: None,
            preset: None,
            stream_url: None,
            start_time: None,
            end_time: None,
            max_frames: None,
            sync_offset: None,
            blend_width: None,
            on_progress: None,
            on_finalizing: None,
            on_stream_ready: None,
            session_hooks: Vec::new(),
            replay_recording: None,
            force_cpu_decode: false,
            cpu_stitch: false,
            lookahead_secs: 0.0,
            lookahead_memory_bytes: None,
            events_path: None,
        }
    }

    /// Create a job with in-memory calibration (no JSON file needed).
    pub fn with_calibration(
        inputs: impl IntoIterator<Item = impl Into<InputPath>>,
        calibration: reco_core::calibration::Calibration,
        output: impl AsRef<Path>,
    ) -> Self {
        let mut job = Self::new(inputs, Path::new(""), output);
        job.calibration = CalibrationSource::Memory(Box::new(calibration));
        job
    }

    /// Create a job from RTSP video inputs using explicit media-clock mapping and bounded synchronization.
    ///
    /// The source is opened by [`Self::run`], after calibration arity validation. Compatible Linux CUDA/Vulkan jobs retain native images through pairing; other configurations deliver host images. File seek and frame-offset options are rejected: RTSP synchronization belongs in each [`crate::rtsp::RtspInput`]. Selected audio is copied from the same RTSP connection and mapped onto the selected video timeline, without mixing or transcoding.
    pub fn from_rtsp(
        inputs: Vec<crate::rtsp::RtspInput>,
        config: crate::rtsp::RtspConfig,
        calibration: reco_core::calibration::Calibration,
        output: impl AsRef<Path>,
    ) -> Self {
        let mut job = Self::with_calibration(Vec::<InputPath>::new(), calibration, output);
        job.inputs = Inputs::Rtsp { inputs, config };
        job
    }

    /// Create a job from a caller-owned host-backed image source.
    ///
    /// The job takes ownership of the source. On success or error, the job attempts to finalize attached outputs before dropping the session and its sinks. The source is dropped last, including during unwinding. The source owns acquisition, timestamp mapping and cancellation; pass the same interruption flag to its opener and [`Self::run`]. Metadata and image layout must remain fixed. Native GPU sources require device-aware [`StitchSession`] setup and are rejected here.
    ///
    /// Audio is disabled because [`FrameSource`] delivers images only. File start-time and frame-offset settings are unavailable; [`Self::end_time`] converts duration to a frame limit using the source rate, and [`Self::max_frames`] limits the number of images processed.
    pub fn from_source(
        source: impl FrameSource + 'static,
        calibration: reco_core::calibration::Calibration,
        output: impl AsRef<Path>,
    ) -> Self {
        let mut job = Self::with_calibration(Vec::<InputPath>::new(), calibration, output);
        job.inputs = Inputs::Supplied(Box::new(source));
        job.audio = AudioMode::Disabled;
        job
    }

    // ── Output settings ──

    /// Set the output video codec. With automatic encoder selection, the
    /// default is 8-bit H.264 for SDR or 10-bit HEVC Main 10 for HDR.
    /// HDR requires HEVC Main 10 and is incompatible with FLV and RTMP.
    pub fn codec(mut self, codec: Codec) -> Self {
        self.codec = Some(codec);
        self
    }

    /// Set the output quality selection.
    pub fn quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }

    /// Set the output container format.
    pub fn container(mut self, container: Container) -> Self {
        self.container = Some(container);
        self
    }

    /// Set the output resolution. Default: 1920x1080.
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

    /// Select a device for the named encoder: VAAPI render node or NVENC GPU index.
    pub fn encoder_device(mut self, device: impl Into<String>) -> Self {
        self.encoder_device = Some(device.into());
        self
    }

    /// Override the encoder preset string (passed through to the encoder).
    pub fn preset(mut self, preset: impl Into<String>) -> Self {
        self.preset = Some(preset.into());
        self
    }

    /// Also publish the recording's encoded packets to RTMP/RTMPS at media speed.
    /// Requires FFmpeg 8+, H.264 and AAC/MP3 source audio; absent audio produces a silent AAC track on the stream.
    /// A bounded, lossy transport FIFO isolates network stalls from recording. Failures do not reconnect.
    /// Observe failures with [`Self::on_stream_ready`] and [`StitchResult::stream_status`].
    pub fn stream_to(mut self, url: impl Into<String>) -> Self {
        self.stream_url = Some(url.into());
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

    /// Override the calibration's saved seam blend width (0.0 - 1.0).
    /// When not called, the calibration document's value is used.
    pub fn blend_width(mut self, blend: f32) -> Self {
        self.blend_width = Some(blend);
        self
    }

    // ── Replay recording (stacked-video) ──

    /// Record pre-stitch source frames to a stacked-video file at
    /// `path` while the job runs. The file is a grid-layout video
    /// (vertical stack for the default N=2 layout) that can be read
    /// back via [`crate::stacked_video::source::StackedSource`] for
    /// professional replay, web panorama generation, or cloud
    /// deployment.
    ///
    /// Defaults to 8-bit H.264 in Matroska with a 30-frame GOP. See
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
    /// StitchJob::new(["left.mp4", "right.mp4"], "match.json", "out.mp4")
    ///     .with_replay_recording("replay.mkv")
    ///     .run(&interrupted)?;
    /// ```
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

    /// Deliver decoded frames in CPU memory instead of using GPU zero-copy, for consumers such as CPU detectors.
    ///
    /// Hardware decoding may still be used with a transfer to CPU memory. [`Self::cpu`] selects software decoding and rendering together. This setting does not reconfigure a caller-supplied source.
    pub fn force_cpu_decode(mut self) -> Self {
        self.force_cpu_decode = true;
        self
    }

    /// Stitch on the CPU executor: software render + CPU decode, no
    /// GPU initialized. Implies [`Self::force_cpu_decode`] (there is
    /// no GPU to decode into).
    /// A caller-supplied source keeps its own acquisition/decoder configuration.
    /// CPU stitching only supports 8-bit output and rejects HDR export.
    pub fn cpu(mut self) -> Self {
        self.cpu_stitch = true;
        self
    }

    /// Set the lookahead buffer depth in seconds.
    pub fn lookahead(mut self, seconds: f64) -> Self {
        self.lookahead_secs = seconds;
        self
    }

    /// Override the retained lookahead image allowance in bytes.
    ///
    /// Without an override, the application selects an allowance at run time.
    /// This is not a total process-memory limit. Zero rejects enabled lookahead.
    pub fn lookahead_memory_limit(mut self, bytes: usize) -> Self {
        self.lookahead_memory_bytes = Some(bytes);
        self
    }

    /// Protect an application input such as a loaded calibration or detector model from export writers.
    pub fn protect_input(mut self, path: impl AsRef<Path>) -> Self {
        self.auxiliary_inputs.push(path.as_ref().to_owned());
        self
    }

    /// Record pipeline events (detections, filter decisions, pan
    /// decisions) to a JSONL file for offline analysis.
    pub fn events(mut self, path: impl AsRef<Path>) -> Self {
        self.events_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Downscale replay recording on the GPU. Width must be divisible by four and height even.
    /// The CPU executor ignores this setting. Call after [`Self::with_replay_recording`].
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

    /// Hook called after frame processing ends and before encoder finalization and source teardown.
    pub fn on_finalizing(mut self, cb: impl FnOnce() + Send + 'static) -> Self {
        self.on_finalizing = Some(Box::new(cb));
        self
    }

    /// Receive a read-only stream monitor before processing starts, including if optional streaming could not open.
    /// The hook runs on the job thread only when streaming was requested. The monitor can be polled from another thread.
    pub fn on_stream_ready(
        mut self,
        cb: impl FnOnce(crate::ffmpeg::encoder::StreamMonitor) + Send + 'static,
    ) -> Self {
        self.on_stream_ready = Some(Box::new(cb));
        self
    }

    /// Hook called after the session is created but before output creation and the frame loop.
    /// An error aborts setup and prevents subsequent hooks from running.
    ///
    /// Use this to attach a detector, director, or other session configuration
    /// that requires access to the session and source metadata (dimensions, fps).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use reco_autocam::{AutocamConfig, TrackingMode};
    ///
    /// StitchJob::new(["left.mp4", "right.mp4"], "match.json", "out.mp4")
    ///     .on_session(|session, source| {
    ///         let config = AutocamConfig::new("model.onnx")
    ///             .with_tracking_mode(TrackingMode::Field)
    ///             .with_detection_interval(3);
    ///         let active = reco_autocam::setup_autocam(session, &config, source.info().fps as f32, source.is_gpu_resident())
    ///             .map_err(|error| reco_io::stitch_job::StitchError::Other(error.to_string()))?;
    ///         if !active {
    ///             return Err(reco_io::stitch_job::StitchError::Other("tracking unavailable".into()));
    ///         }
    ///         Ok(())
    ///     })
    ///     .run(&interrupted)?;
    /// ```
    pub fn on_session(
        mut self,
        cb: impl FnOnce(&mut StitchSession, &dyn FrameSource) -> Result<(), StitchError>
        + Send
        + 'static,
    ) -> Self {
        self.session_hooks.push(Box::new(cb));
        self
    }

    // ── Execute ──

    /// Run the stitching job.
    ///
    /// This is a blocking call that processes all frames (or until the
    /// interrupt flag is set). Returns a [`StitchResult`] with statistics.
    pub fn run(mut self, interrupted: &Arc<AtomicBool>) -> Result<StitchResult, StitchError> {
        if self.start_time.is_some_and(|t| !t.is_finite() || t < 0.0)
            || self
                .end_time
                .is_some_and(|t| !t.is_finite() || t <= self.start_time.unwrap_or(0.0))
        {
            return Err(StitchError::Other("invalid export time window".into()));
        }
        let mut destinations = vec![ffmpeg_file_path(&self.output)];
        destinations.extend(self.events_path.as_deref());
        destinations.extend(
            self.replay_recording
                .as_ref()
                .map(|cfg| ffmpeg_file_path(&cfg.path)),
        );
        let mut sources = std::mem::take(&mut self.auxiliary_inputs);
        if let Inputs::Files(inputs) = &self.inputs {
            for path in inputs.iter().flat_map(InputPath::all_paths) {
                sources.push(ffmpeg_file_path(&path).to_owned());
            }
        }
        if let CalibrationSource::File(path) = &self.calibration {
            sources.push(path.clone());
        }
        for (index, destination) in destinations.iter().enumerate() {
            if destination.is_symlink() && !destination.exists() {
                return Err(StitchError::Other(
                    "output symlink has no accessible target".into(),
                ));
            }
            for other in destinations[..index]
                .iter()
                .copied()
                .chain(sources.iter().map(|p| p.as_path()))
            {
                if paths_collide(destination, other) {
                    return Err(StitchError::Other(
                        "input, event, replay and output paths must be distinct".into(),
                    ));
                }
            }
        }
        let container = self.resolved_container();
        if self.encoder_device.is_some() && self.encoder_name.is_none() {
            return Err(StitchError::Other(
                "encoder device selection requires a named encoder".into(),
            ));
        }
        crate::init();
        let start = std::time::Instant::now();

        // Load calibration
        let mut cal = match self.calibration {
            CalibrationSource::File(ref path) => {
                reco_core::calibration::Calibration::from_file(path)
                    .map_err(|e| StitchError::Calibration(format!("{e}")))?
            }
            CalibrationSource::Memory(cal) => *cal,
        };
        self.inputs.validate_options(
            self.start_time,
            self.end_time,
            self.sync_offset,
            &self.audio,
        )?;
        let effective_sync = self.sync_offset.unwrap_or(cal.sync_offset);
        if matches!(self.inputs, Inputs::Files(_))
            && self.sync_offset.is_none()
            && cal.sync_offset != 0
        {
            log::info!("Sync offset: {} frames (from calibration)", effective_sync);
        }

        // Input arity must match the calibration's projection before any
        // decoder spins up. The projection decides the camera count -
        // never the number of inputs handed in.
        let camera_count = cal.projection.camera_count();
        if self.inputs.len() != camera_count {
            return Err(StitchError::Other(format!(
                "this calibration's projection consumes {camera_count} camera input(s), \
                 but {} were given",
                self.inputs.len()
            )));
        }
        #[cfg(feature = "gpu")]
        let gpu = if self.cpu_stitch {
            log::info!("CPU stitch: software render + software decode");
            None
        } else {
            Some(reco_core::gpu::GpuContext::new_blocking()?)
        };
        let audio_source = self.inputs.audio_paths(&self.audio);
        let opened = self.inputs.open(
            effective_sync,
            self.cpu_stitch,
            self.force_cpu_decode,
            #[cfg(feature = "gpu")]
            gpu.as_ref(),
            interrupted,
            &self.audio,
        )?;
        let source_color = opened.color_metadata();
        if !source_color.is_empty() {
            log::info!("Source colour metadata: {source_color}");
        }
        let bit_depth = if source_color.is_hdr() {
            BitDepth::Ten
        } else {
            BitDepth::Eight
        };
        if bit_depth == BitDepth::Ten {
            #[cfg(feature = "gpu")]
            let gpu_rendering = gpu.is_some();
            #[cfg(not(feature = "gpu"))]
            let gpu_rendering = false;
            if !gpu_rendering {
                return Err(StitchError::Other(
                    "10-bit HDR export requires GPU rendering (CPU stitching only supports 8-bit output)".into(),
                ));
            }
            if self.codec.is_some_and(|codec| codec != Codec::HEVC) {
                return Err(StitchError::Other(
                    "10-bit HDR export requires HEVC Main 10".into(),
                ));
            }
            log::info!("Selected 10-bit HDR delivery (P010LE, HEVC Main 10)");
        }
        let frames = opened.frames();
        let info = frames.info();
        let fps_rational = info.fps_rational.ok_or_else(|| StitchError::Other(
            "video frame rate is unknown for one or more inputs; cannot export with reliable timing".into()
        ))?;
        let fps = info.fps;
        let start_secs = self.start_time.unwrap_or(0.0);
        if let Some(end) = self.end_time {
            let end_frames = ((end - start_secs) * fps).round() as i64;
            if end_frames <= 0 {
                return Err(StitchError::Other(format!(
                    "end_time ({end:.2}s) must be greater than start_time ({start_secs:.2}s)"
                )));
            }
        }

        if self.replay_recording.is_some()
            && info.camera_sizes.windows(2).any(|pair| pair[0] != pair[1])
        {
            return Err(StitchError::Other(
                "stacked replay requires equal camera dimensions".into(),
            ));
        }
        let (out_w, out_h) = self.resolution.unwrap_or((1920, 1080));
        if self.resolution.is_none() {
            log::info!("Output resolution not specified, defaulting to {out_w}x{out_h}");
        }
        let decode_mode = opened.decode_mode().to_string();

        // Build session. Blend lives on the calibration; an explicit job
        // override replaces it, otherwise the saved value renders as-is.
        // Seamless topologies (the mono cylinder) have no blend to set.
        let seam = match &mut cal.projection {
            reco_core::calibration::Projection::LShape(rig) => Some(rig),
            reco_core::calibration::Projection::Panorama(panorama) => Some(&mut panorama.rig),
            reco_core::calibration::Projection::Cylinder(_) => None,
        };
        match (self.blend_width, seam) {
            (Some(blend), Some(projection)) => {
                log::info!(
                    "seam blend: overriding calibration value {} with {blend}",
                    projection.blend_width
                );
                projection.blend_width = blend;
            }
            (Some(blend), None) => {
                log::warn!("seam blend override {blend} ignored: this projection has no seam");
            }
            (None, _) => {
                log::info!(
                    "seam blend: using calibration value {}",
                    cal.projection.blend_width()
                );
            }
        }
        let viewport_size = ViewportSize {
            width: out_w,
            height: out_h,
        };
        let cpu_session = |cal| -> Result<_, StitchError> {
            let executor = reco_core::stitch::CpuExecutor::new(
                cal,
                viewport_size.clone(),
                info.camera_sizes.clone(),
                false,
            )
            .map_err(|e| StitchError::Other(format!("CPU executor: {e}")))?;
            Ok(StitchSession::with_executor(
                reco_core::stitch::Executor::Cpu(Box::new(executor)),
            )?)
        };
        // Declare the owned source first so the session releases its sinks
        // and pending GPU work before source teardown, including during unwind.
        let mut source;
        let gpu_name;
        #[cfg(feature = "gpu")]
        let mut session = match gpu {
            Some(gpu) => {
                gpu_name = gpu.gpu_name().to_string();
                let session_config = reco_core::session::types::SessionConfig {
                    calibration: cal,
                    viewport_size,
                    source_sizes: info.camera_sizes.clone(),
                    output_format: match bit_depth {
                        BitDepth::Eight => reco_core::gpu::OutputFormat::Rgba8Unorm,
                        BitDepth::Ten => reco_core::gpu::OutputFormat::Rgb10a2Unorm,
                    },
                    input_format: frames.input_format(),
                    left_rotation: frames.left_rotation(),
                    right_rotation: frames.right_rotation(),
                };
                reco_core::session::StitchSession::with_gpu(gpu, session_config)?
            }
            None => {
                gpu_name = "software (CPU)".to_string();
                cpu_session(cal)?
            }
        };

        #[cfg(not(feature = "gpu"))]
        let mut session = {
            gpu_name = "software (CPU)".to_string();
            cpu_session(cal)?
        };

        session.telemetry_mut().set_gpu_name(gpu_name.clone());
        session.telemetry_mut().set_decode_mode(decode_mode.clone());

        // Configure GPU bind groups if source is GPU-resident
        #[cfg(all(feature = "gpu", target_os = "linux"))]
        if let Some(shared) = opened.shared_texture_set() {
            session.setup_gpu_source(shared)?;
        }

        let live_audio = opened.audio();
        source = opened.into_frames();
        let output_color = reco_core::render::color::YuvColor::delivery(source.yuv_color().matrix);

        let skip_frames = (start_secs * fps).round() as u64;
        let setup_result = (|| -> Result<_, StitchError> {
            for hook in self.session_hooks.drain(..) {
                hook(&mut session, source.as_ref())?;
            }

            if session.core().constrained_look() {
                session
                    .core_mut()
                    .prepare_coverage()
                    .map_err(|error| StitchError::Other(error.to_string()))?;
            }

            // Start decode threads now that hooks (ORT/DML init) have completed.
            source.start_decoding();

            // Attach JSONL event sink if requested.
            if let Some(ref events_path) = self.events_path {
                let sink = crate::jsonl_sink::JsonlSink::create(events_path).map_err(|e| {
                    StitchError::Other(format!("could not create requested event file: {e}"))
                })?;
                session.set_event_sink(Box::new(sink));
            }

            // Create encoder with optional audio passthrough.
            // File video synchronization skips the leading frames of one camera.
            // Its passthrough audio must start at that same source moment.
            let audio_sync_skip = match self.audio {
                AudioMode::CopyFrom(0) if effective_sync < 0 => effective_sync.unsigned_abs(),
                AudioMode::CopyFrom(1) if effective_sync > 0 => effective_sync as u64,
                _ => 0,
            };
            let audio_start_time = start_secs + audio_sync_skip as f64 / fps;

            let enc_config = crate::ffmpeg::encoder::EncoderConfig {
                encoder_name: self.encoder_name.clone(),
                device: self.encoder_device.clone(),
                software_only: self.cpu_stitch,
                codec: self.codec,
                bit_depth,
                quality: self.quality,
                preset: self.preset.clone(),
                audio_source,
                audio_start_time,
                container,
                gop_size: self
                    .stream_url
                    .as_ref()
                    .map(|_| (fps * 2.0).round().max(1.0) as u32),
                stream_url: self.stream_url.take(),
                interrupted: Some(interrupted.clone()),
                network_timeout: None,
                color_metadata: source_color,
            };
            let encoder = crate::adapters::FfmpegFileEncoder::with_live_audio(
                &self.output,
                out_w,
                out_h,
                (fps_rational.0, fps_rational.1),
                &enc_config,
                live_audio,
                output_color,
            )?;
            let enc_name = encoder.encoder_name().to_string();
            let stream_monitor = encoder.stream_monitor();
            if let Some(monitor) = &stream_monitor
                && let Some(callback) = self.on_stream_ready.take()
            {
                callback(monitor.clone());
            }
            session.telemetry_mut().set_encoder_name(enc_name.clone());
            session.add_sink(
                Box::new(encoder),
                reco_core::session::SinkOptions::threaded(2),
            )?;

            // Sample the allowance after inference and output resources are initialized.
            if self.lookahead_secs > 0.0 {
                let frames = (self.lookahead_secs * info.fps).round() as usize;
                let budget = match source.is_gpu_resident() {
                    #[cfg(feature = "gpu")]
                    true => crate::lookahead::native_image_budget(
                        session.core().gpu(),
                        self.lookahead_memory_bytes,
                    ),
                    _ => crate::lookahead::image_budget(self.lookahead_memory_bytes),
                }
                .map_err(StitchError::Other)?;
                session.set_lookahead(frames, budget);
            }

            if let Some(ref mut cfg) = self.replay_recording {
                if cfg.encoder_config.inner.encoder_name.is_none() {
                    // Match the primary hardware family when useful. The replay
                    // config still selects its own codec and 8-bit input.
                    cfg.encoder_config.inner.encoder_name =
                        crate::ffmpeg::encoder::VideoEncoder::replay_encoder_name(&enc_name)
                            .map(String::from);
                }
                crate::stacked_video::replay::configure_replay_color(
                    &mut cfg.encoder_config,
                    source.as_ref(),
                    source_color,
                )
                .map_err(|e| StitchError::Other(format!("replay recording color: {e}")))?;
                if cfg.encoder_config.fps.is_none() {
                    cfg.encoder_config.fps = Some(fps_rational);
                    log::info!(
                        "replay recording follows the source rate {}/{}",
                        fps_rational.0,
                        fps_rational.1
                    );
                }
            }

            // Skip frames to reach start position.
            if skip_frames > 0 {
                log::info!("skipping {skip_frames} frames (start_time={start_secs:.2}s)");
                let skipped = source.skip_frames(skip_frames)?;
                if skipped < skip_frames {
                    log::warn!("source ended during skip at frame {skipped}/{skip_frames}");
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

            // Replay starts after skipped frames, on the same output window.
            let replay_encoder = if let Some(cfg) = self.replay_recording.take() {
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
                match source.is_gpu_resident() {
                    #[cfg(feature = "gpu")]
                    true => {
                        // Resolve the output tile size. `scale` is the
                        // per-tile dims after downscale; the layout stays
                        // sized by the source tile dims (layout.capacity
                        // is N, atlas h = output.height * rows).
                        let (out_w, out_h) = cfg
                            .scale
                            .unwrap_or((info.camera_sizes[0].0, info.camera_sizes[0].1));
                        use reco_core::gpu::yuv_stack_packer::{OutputTileSize, StackGridLayout};

                        let layout = StackGridLayout::vstack(out_w, out_h, 2).ok_or_else(|| {
                            let (source_w, source_h) = info.camera_sizes[0];
                            StitchError::Other(format!(
                                "GPU stacked replay: replay tile dims {out_w}x{out_h} are not YUV420P-aligned \
                                 (width must be divisible by 4, height must be even). Source: {source_w}x{source_h}",
                            ))
                        })?;
                        let output_size = if cfg.scale.is_some() {
                            OutputTileSize::scaled(out_w, out_h)
                        } else {
                            OutputTileSize::unscaled(out_w, out_h)
                        };
                        session
                            .enable_gpu_stacked_replay(layout, output_size)
                            .map_err(|e| {
                                StitchError::Other(format!("enable GPU stacked replay: {e}"))
                            })?;
                        let (atlas_w, atlas_h) = session.stacked_atlas_dims().ok_or_else(|| {
                            StitchError::Other(
                                "stacked_atlas_dims returned None right after enable; internal bug"
                                    .into(),
                            )
                        })?;
                        let recorder = crate::stacked_video::replay::session_gpu_recorder(
                            &cfg.path,
                            *cfg.encoder_config,
                            atlas_w,
                            atlas_h,
                        )
                        .map_err(|e| {
                            StitchError::Other(format!("open GPU replay recorder: {e}"))
                        })?;
                        session.set_stacked_gpu_recorder(recorder);
                        let scale_note = if cfg.scale.is_some() {
                            format!(
                                " [A19 downscale: source {}x{} -> tile {}x{}]",
                                info.camera_sizes[0].0, info.camera_sizes[0].1, out_w, out_h
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
                        None
                    }
                    _ => {
                        if cfg.scale.is_some() {
                            log::warn!(
                                "reco-io: --replay-scale requested on a CPU-resident source, but the \
                             CPU ReplayRecordingSource decorator doesn't support downscale today. \
                             Recording at source dims {}x{} instead. Use a GPU-resident source \
                             (zero-copy NVDEC) or disable --replay-scale.",
                                info.camera_sizes[0].0,
                                info.camera_sizes[0].1,
                            );
                        }
                        log::info!(
                            "reco-io: replay pack path = CPU (source CPU-resident; ReplayRecordingSource decorator, tile {}x{}, N=2) -> {}",
                            info.camera_sizes[0].0,
                            info.camera_sizes[0].1,
                            cfg.path.display(),
                        );
                        Some(
                            ReplayRecordingSource::open_encoder(
                                source.as_ref(),
                                &cfg.path,
                                *cfg.encoder_config,
                            )
                            .map_err(|e| {
                                StitchError::Other(format!("replay recording open: {e}"))
                            })?,
                        )
                    }
                }
            } else {
                None
            };
            Ok((enc_name, stream_monitor, frame_limit, replay_encoder))
        })();
        let (enc_name, stream_monitor, frame_limit, replay_encoder) = match setup_result {
            Ok(setup) => setup,
            Err(error) => {
                // Keep the source alive through sink shutdown, including a failed replay open.
                if let Err(finish_error) = session.finish() {
                    log::warn!("output finalization also failed: {finish_error}");
                }
                return Err(error);
            }
        };
        if let Some(encoder) = replay_encoder {
            source = Box::new(ReplayRecordingSource::with_encoder(source, encoder));
        }
        let run_result = session.run(
            source.as_mut(),
            frame_limit,
            interrupted,
            self.on_progress.take(),
        );

        if let Some(cb) = self.on_finalizing.take() {
            cb();
        }

        // GPU-path finalize: drop the atlas recorder (its `finish`
        // closes the encoder file) before `session.finish()` so
        // the replay file lands before the main encoder's trailer.
        // No-op when the GPU path wasn't selected.
        #[cfg(feature = "gpu")]
        session.clear_stacked_gpu_recorder();
        let finish_result = session.finish();
        let telemetry_snap = session.telemetry_snapshot();
        drop(session);
        drop(source);

        // A failed run can still have rendered frames awaiting delivery. Finalize
        // them before returning, retaining the run failure as the primary error.
        if run_result.is_err()
            && let Err(error) = &finish_result
        {
            log::warn!("output finalization also failed: {error}");
        }
        let run_result = run_result?;
        finish_result?;
        let completion = run_result.completion;
        let frame_count = run_result.frames_processed;

        if frame_count == 0
            && skip_frames > 0
            && frame_limit > 0
            && completion == reco_core::session::types::RunCompletion::Completed
        {
            return Err(StitchError::Other(format!(
                "no video frames remain at start_time ({start_secs:.2}s)"
            )));
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
        if frame_count > 0 && !crate::output::is_network_destination(&self.output) {
            let output_path = self.output.clone();
            match crate::ffmpeg::decoder::VideoDecoder::open_software(&output_path) {
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
            completion,
            frames_processed: frame_count,
            elapsed: start.elapsed(),
            gpu_name,
            encoder_name: enc_name,
            decode_mode,
            telemetry: Some(telemetry_snap),
            stream_status: stream_monitor.map(|monitor| monitor.status()),
        })
    }
}

// libavformat's file protocol opens the filename after the prefix verbatim.
fn ffmpeg_file_path(path: &Path) -> &Path {
    path.to_str()
        .and_then(|s| s.strip_prefix("file:"))
        .map(Path::new)
        .unwrap_or(path)
}

fn paths_collide(first: &Path, second: &Path) -> bool {
    if first == second || same_file::is_same_file(first, second).unwrap_or(false) {
        return true;
    }
    // A destination may not exist yet. Compare its filename and resolved parent.
    fn resolved_parent(path: &Path) -> Option<PathBuf> {
        std::path::absolute(path)
            .ok()?
            .parent()?
            .canonicalize()
            .ok()
    }
    first.file_name() == second.file_name()
        && resolved_parent(first).is_some_and(|dir| Some(dir) == resolved_parent(second))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_window_leaves_output_intact() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("export.mp4");
        std::fs::write(&output, b"original").unwrap();
        for (start, end) in [(1.0, 0.5), (0.0, f64::INFINITY), (f64::NAN, 2.0)] {
            let job = StitchJob::new(["missing.mp4"], "missing.json", &output)
                .start_time(start)
                .end_time(end);
            let error = job.run(&Arc::new(AtomicBool::new(false))).unwrap_err();
            assert!(error.to_string().contains("window"));
        }
        assert_eq!(std::fs::read(&output).unwrap(), b"original");
    }

    #[test]
    fn colliding_outputs_preserve_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("output.mp4");
        std::fs::write(&output, b"sentinel").unwrap();
        let alias = dir.path().join("events.jsonl");
        std::fs::hard_link(&output, &alias).unwrap();
        let job = StitchJob::new(["missing.mp4"], "missing.json", &output).events(alias);
        let err = job.run(&Arc::new(AtomicBool::new(false))).unwrap_err();
        assert!(err.to_string().contains("distinct"), "{err}");
        assert_eq!(std::fs::read(&output).unwrap(), b"sentinel");
    }

    #[test]
    fn replay_file_url_cannot_replace_a_chapter() {
        let dir = tempfile::tempdir().unwrap();
        let chapter = dir.path().join("second.mkv");
        std::fs::write(&chapter, b"preserve chapter").unwrap();
        let input = InputPath::Chained(vec![dir.path().join("first.mkv"), chapter.clone()]);
        let job = StitchJob::new([input], "missing.json", dir.path().join("out.mp4"))
            .with_replay_recording(format!("file:{}", chapter.display()));
        let error = job.run(&Arc::new(AtomicBool::new(false))).unwrap_err();
        assert!(error.to_string().contains("distinct"), "{error}");
        assert_eq!(std::fs::read(chapter).unwrap(), b"preserve chapter");
    }

    #[cfg(unix)]
    #[test]
    fn dangling_event_symlink_cannot_create_an_alias_for_a_new_output() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("new.mp4");
        let alias = dir.path().join("events");
        std::os::unix::fs::symlink(&target, &alias).unwrap();
        let job = StitchJob::new(["missing.mp4"], "missing.json", &target).events(alias);
        let error = job.run(&Arc::new(AtomicBool::new(false))).unwrap_err();
        assert!(error.to_string().contains("symlink"), "{error}");
        assert!(!target.exists());
    }

    #[test]
    fn omitted_container_follows_destination_explicit_container_wins() {
        for (path, expected) in [
            ("out.mkv", Container::Mkv),
            ("out.mov", Container::Mov),
            ("out.mp4", Container::Mp4),
        ] {
            let job = StitchJob::new(["missing.mp4"], "missing.json", path);
            assert_eq!(job.resolved_container(), expected);
            assert_eq!(
                job.container(Container::Mp4Fragmented).resolved_container(),
                Container::Mp4Fragmented
            );
        }
    }

    #[test]
    fn device_without_encoder_is_rejected_before_loading_calibration() {
        let job = StitchJob::new(["missing.mp4"], "missing.json", "out.mp4")
            .encoder_device("/dev/dri/renderD129");
        let error = job.run(&Arc::new(AtomicBool::new(false))).unwrap_err();
        assert!(error.to_string().contains("encoder"), "{error}");
        assert!(!matches!(error, StitchError::Calibration(_)));
    }

    /// The arity check runs against the projection BEFORE any decoder or
    /// GPU spins up: with nonexistent input paths, a wrong input count
    /// must surface as the typed arity error, never as a file-open
    /// error (which would prove a decoder was touched first).
    #[test]
    fn run_rejects_arity_mismatch_before_opening_inputs() {
        let cam = || {
            reco_core::calibration::Lens::fisheye(
                640,
                360,
                320.0,
                320.0,
                320.0,
                180.0,
                [-0.02, 0.004, 0.0, 0.0],
            )
        };
        let cal = reco_core::calibration::Calibration::new(
            vec![cam(), cam()],
            reco_core::projection::LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            reco_core::calibration::Framing {
                axis_offset: 0.25,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        let job =
            StitchJob::with_calibration(["does-not-exist.mp4"], cal, "arity-mismatch-out.mp4");
        let interrupted = Arc::new(AtomicBool::new(false));
        let err = job.run(&interrupted).expect_err("1 input vs 2 cameras");
        let msg = err.to_string();
        assert!(
            msg.contains("consumes 2 camera input(s), but 1 were given"),
            "expected the typed arity error, got: {msg}"
        );
    }

    #[test]
    fn parse_segments_handles_single_chained_and_stray_separators() {
        assert!(matches!(
            InputPath::parse_segments("a.mp4"),
            InputPath::Single(p) if p == std::path::Path::new("a.mp4")
        ));
        assert!(matches!(
            InputPath::parse_segments("a.mp4;b.mp4"),
            InputPath::Chained(v) if v.len() == 2
        ));
        // A trailing separator is one input, not a path with a stray
        // semicolon (the old CLI closure wrapped the raw string).
        assert!(matches!(
            InputPath::parse_segments("a.mp4;"),
            InputPath::Single(p) if p == std::path::Path::new("a.mp4")
        ));
        assert!(matches!(
            InputPath::parse_segments(" a.mp4 ; b.mp4 "),
            InputPath::Chained(v) if v == vec![PathBuf::from("a.mp4"), PathBuf::from("b.mp4")]
        ));
        // Nothing usable: the raw text flows into open()'s
        // invalid-path reporting instead of an empty chain.
        assert!(matches!(
            InputPath::parse_segments(";;"),
            InputPath::Single(p) if p == std::path::Path::new(";;")
        ));
    }

    #[test]
    fn all_paths_returns_every_chained_segment() {
        let single = InputPath::Single(PathBuf::from("a.mp4"));
        assert_eq!(single.all_paths(), vec![PathBuf::from("a.mp4")]);

        let chained = InputPath::Chained(vec![
            PathBuf::from("a.mp4"),
            PathBuf::from("b.mp4"),
            PathBuf::from("c.mp4"),
        ]);
        assert_eq!(
            chained.all_paths(),
            vec![
                PathBuf::from("a.mp4"),
                PathBuf::from("b.mp4"),
                PathBuf::from("c.mp4"),
            ]
        );
        // first_path stays the first segment; all_paths must not drop the rest.
        assert_eq!(chained.first_path(), std::path::Path::new("a.mp4"));
    }
}
