//! High-level stitching session.
//!
//! [`StitchSession`] bundles the GPU pipeline with the NV12 converter,
//! providing a single entry point for rendering and encoding stitched
//! panoramic frames. This keeps encode orchestration inside `reco-core`
//! so that every consumer (CLI, GUI, OBS plugin, cloud worker) gets the
//! same optimized frame loop without duplicating pipeline plumbing.
//!
//! ## Two-level API
//!
//! - [`StitchSession::process_frame`] - render one frame and submit it
//!   to an encoder. Use this for interactive/GUI applications or when
//!   the caller controls the frame loop (e.g. zero-copy GPU decode).
//!
//! - [`StitchSession::run`] - batch-process an entire [`FrameSource`]
//!   into an encoder, with optional progress reporting and interrupt
//!   support. Use this for CLI batch encoding.

/// Detection pipeline - also usable standalone without StitchSession.
pub mod detection;
#[cfg(test)]
mod tests;
#[cfg(target_os = "linux")]
mod zero_copy_linux;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod zero_copy_macos;

#[cfg(target_os = "linux")]
pub use zero_copy_linux::SharedTextureSet;

// `LiveStitchSession` + `LiveSessionConfig` + `LiveSessionError` were
// deleted 2026-04-19 (plan-execution §3 M3 step 3). Consumers that
// previously held a `LiveStitchSession` migrate to `StitchCore` (via
// `reco_core::core::StitchCore`) and call `submit_frame_*_at_pose`
// for explicit-pose inputs. reco-obs completed the migration in the
// same commit.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::async_encode::AsyncEncodeThread;
use crate::calibration::MatchCalibration;
use crate::core::{StitchCore, StitchCoreConfig, StitchCoreError};
use crate::detector::Detection;
use crate::director::{Director, DirectorContext, MappedDetection, ViewportPosition};
use crate::encoder::{EncodeError, Encoder, GpuEncoder};
use crate::gpu::{GpuContext, GpuError, OutputFormat};
use crate::nv12_converter::{Nv12Converter, Nv12Error};
use crate::pipeline::{PipelineError, StitchPipeline};
use crate::projection;
use crate::renderer::InputFormat;
use crate::source::{FrameSource, SourceError, StereoFrame};
use crate::viewport::ViewportConfig;

use detection::DetectionPipeline;

use thiserror::Error;

/// Compute the frame limit from optional duration and max-frames constraints.
///
/// Both `duration_secs` and `max_frames` are optional. When both are provided,
/// the stricter (lower) limit wins. Returns [`u64::MAX`] when neither is set,
/// meaning "process all available frames".
pub fn compute_frame_limit(duration_secs: Option<f64>, max_frames: Option<u64>, fps: f64) -> u64 {
    let fps = if fps > 0.0 { fps } else { 30.0 };
    match (duration_secs, max_frames) {
        (Some(dur), Some(mf)) if dur > 0.0 => ((dur * fps) as u64).min(mf),
        (Some(dur), None) if dur > 0.0 => (dur * fps) as u64,
        (_, Some(mf)) => mf,
        _ => u64::MAX,
    }
}

/// Configuration for creating a [`StitchSession`].
#[derive(Debug)]
pub struct SessionConfig {
    /// Camera calibration data.
    pub calibration: MatchCalibration,
    /// Output viewport (dimensions, blend width, FOV).
    pub viewport: ViewportConfig,
    /// Input frame width in pixels.
    pub input_width: u32,
    /// Input frame height in pixels.
    pub input_height: u32,
    /// GPU render target format (typically [`OutputFormat::Rgba8Unorm`] for encoding).
    pub output_format: OutputFormat,
    /// Input pixel format (YUV420P or NV12).
    pub input_format: InputFormat,
    /// Left camera rotation from stream metadata (0, 90, 180, 270 degrees).
    ///
    /// The session applies rotation automatically based on the active path:
    /// the CPU decode path handles rotation via buffer reversal in the decoder,
    /// while the GPU zero-copy path uses a shader UV flip.
    pub left_rotation: i32,
    /// Right camera rotation from stream metadata (0, 90, 180, 270 degrees).
    pub right_rotation: i32,
}

/// Progress information passed to the progress callback.
#[derive(Debug, Clone)]
pub struct FrameProgress {
    /// Number of frames processed so far.
    pub frames_completed: u64,
    /// Elapsed wall-clock time since the run started.
    pub elapsed: std::time::Duration,
}

/// Callback for progress reporting during [`StitchSession::run`].
pub type ProgressCallback = Box<dyn FnMut(&FrameProgress) + Send>;

/// Result from [`StitchSession::step`] - one frame with full session features.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Where the virtual camera pointed for this frame.
    pub viewport: ViewportPosition,
    /// Detections mapped to panorama coordinates (empty if no detector or skipped frame).
    pub detections: Vec<MappedDetection>,
    /// Frame index (0-based).
    pub frame_index: u64,
}

/// Session performance metrics for health monitoring.
///
/// Read via [`StitchSession::metrics`]. Updated per-frame.
#[derive(Debug, Clone, Default)]
pub struct SessionMetrics {
    /// Frames processed so far.
    pub frames_processed: u64,
    /// Frames where errors were skipped (via ErrorPolicy::Skip).
    pub frames_dropped: u64,
    /// Total elapsed time since first frame.
    pub elapsed: std::time::Duration,
    /// Average fps over the session lifetime.
    pub fps_average: f32,
    /// Total frames in the source (if known).
    pub total_frames: Option<u64>,
}

/// Error handling policy for [`StitchSession::run`].
///
/// Controls what happens when a frame fails to decode or render.
#[derive(Default)]
pub enum ErrorPolicy {
    /// Stop processing on the first error (default).
    #[default]
    Abort,
    /// Skip bad frames, up to `max_consecutive` in a row.
    /// If `max_consecutive` consecutive frames fail, abort.
    Skip {
        /// Maximum consecutive frame errors before aborting.
        max_consecutive: u64,
    },
}

/// Boxed error type propagated by a [`DetectionSink`] implementation.
///
/// Sinks return this so I/O failures (disk full, broken pipe) can bubble
/// up through [`StitchSession::run`] as a [`SessionError::DetectionSink`]
/// instead of being swallowed by a logger.
pub type DetectionSinkError = Box<dyn std::error::Error + Send + Sync>;

/// Fallible sink for per-frame tracked detection data.
///
/// Sinks receive detections mapped to panorama coordinates every frame
/// (including frames where no detector ran; the vector will be empty or
/// hold the last known positions). The sink returns a `Result`, so CSV
/// writers, socket senders, and similar consumers can surface I/O
/// failures instead of logging and continuing with a corrupt output.
///
/// Any closure matching the signature `FnMut(&[MappedDetection], u64, f64)
/// -> Result<(), DetectionSinkError>` automatically implements this trait
/// via the blanket impl below, so callers can write:
///
/// ```rust,ignore
/// session.set_detection_sink(Box::new(|dets, frame_idx, ts_ms| {
///     writer.write_csv_row(dets, frame_idx, ts_ms)?;
///     Ok(())
/// }));
/// ```
///
/// A sink is called once per frame. Errors returned from the sink abort
/// the current session call (`step`, `process_frame`, or `run`) with
/// [`SessionError::DetectionSink`].
pub trait DetectionSink: Send {
    /// Receive tracked detections for a single frame.
    ///
    /// `detections` is the same data the director sees (panorama
    /// coordinates, camera origin, confidence). `frame_index` is 0-based.
    /// `timestamp_ms` is measured from session start (not PTS).
    fn on_detections(
        &mut self,
        detections: &[MappedDetection],
        frame_index: u64,
        timestamp_ms: f64,
    ) -> Result<(), DetectionSinkError>;
}

impl<F> DetectionSink for F
where
    F: FnMut(&[MappedDetection], u64, f64) -> Result<(), DetectionSinkError> + Send,
{
    fn on_detections(
        &mut self,
        detections: &[MappedDetection],
        frame_index: u64,
        timestamp_ms: f64,
    ) -> Result<(), DetectionSinkError> {
        (self)(detections, frame_index, timestamp_ms)
    }
}

/// Errors from [`StitchSession`]. `Clone + Send + Sync` so consumers
/// posting session results across thread boundaries (reco-gui export
/// thread, reco-obs async init) can keep the typed enum instead of
/// falling back to `Result<_, String>`.
#[derive(Debug, Clone, Error)]
pub enum SessionError {
    /// GPU initialization error.
    #[error("GPU: {0}")]
    Gpu(#[from] GpuError),

    /// GPU pipeline error.
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),

    /// `StitchCore` error (wraps pipeline / readback / config).
    #[error("core: {0}")]
    Core(#[from] StitchCoreError),

    /// NV12 conversion error.
    #[error("NV12 converter: {0}")]
    Nv12(#[from] Nv12Error),

    /// Encoder error.
    #[error("encoder: {0}")]
    Encode(#[from] EncodeError),

    /// Source error.
    #[error("source: {0}")]
    Source(#[from] SourceError),

    /// Metal interop error (macOS zero-copy).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[error("Metal interop: {0}")]
    MetalInterop(#[from] crate::metal_interop::MetalInteropError),

    /// Zero-copy setup or runtime error.
    #[error("zero-copy: {0}")]
    ZeroCopy(String),

    /// Missing or invalid configuration.
    #[error("config: {0}")]
    Config(String),

    /// A [`DetectionSink`] returned an error.
    ///
    /// Surfaces I/O failures from user-supplied sinks (CSV writers,
    /// network senders, ...) so they are not silently swallowed.
    /// The sink's typed error is stringified at this boundary
    /// because the sink trait uses `Box<dyn Error + Send + Sync>`
    /// which is not `Clone`. Consumers that need the typed
    /// underlying error should catch it before returning from the
    /// sink closure; anything that reaches here is already formatted.
    #[error("detection sink: {0}")]
    DetectionSink(String),
}

// Compile-time assertion (plan step 7): every error type reachable
// from the public session API is `Clone + Send + Sync`, so consumers
// that post results to worker-thread channels (reco-gui export
// thread, reco-obs async init) carry the typed error across the
// boundary instead of stringifying. Regresses if a future variant
// introduces a non-Clone wrapped error.
const _: fn() = || {
    fn assert_clone_send_sync<T: Clone + Send + Sync + 'static>() {}
    assert_clone_send_sync::<SessionError>();
    assert_clone_send_sync::<GpuError>();
    assert_clone_send_sync::<PipelineError>();
    assert_clone_send_sync::<StitchCoreError>();
    assert_clone_send_sync::<Nv12Error>();
    assert_clone_send_sync::<EncodeError>();
    assert_clone_send_sync::<SourceError>();
    assert_clone_send_sync::<crate::detector::DetectorError>();
};

/// Builder for constructing a [`StitchSession`] with sensible defaults.
///
/// Required fields: `calibration` and `input_dimensions`. Everything else
/// has defaults or is optional.
///
/// ```rust,ignore
/// let session = StitchSession::builder()
///     .calibration(cal)
///     .input_dimensions(1920, 1080)
///     .viewport(viewport)
///     .gpu(gpu)
///     .build()?;
/// ```
pub struct StitchSessionBuilder {
    calibration: Option<MatchCalibration>,
    viewport: Option<ViewportConfig>,
    input_width: Option<u32>,
    input_height: Option<u32>,
    output_format: OutputFormat,
    input_format: InputFormat,
    gpu: Option<GpuContext>,
    encoder: Option<(Box<dyn Encoder + Send>, usize)>,
    gpu_encoder: Option<Box<dyn GpuEncoder>>,
    detector: Option<Box<dyn crate::detector::UnifiedDetector>>,
    director: Option<Box<dyn Director>>,
    detection_interval: u64,
    lookahead_frames: usize,
}

impl StitchSessionBuilder {
    /// Set the camera calibration (required).
    pub fn calibration(mut self, cal: MatchCalibration) -> Self {
        self.calibration = Some(cal);
        self
    }

    /// Set the output viewport configuration.
    ///
    /// Defaults to 1920x1080 with blend_width 0.15 if not set.
    pub fn viewport(mut self, viewport: ViewportConfig) -> Self {
        self.viewport = Some(viewport);
        self
    }

    /// Set the input frame dimensions (required).
    pub fn input_dimensions(mut self, width: u32, height: u32) -> Self {
        self.input_width = Some(width);
        self.input_height = Some(height);
        self
    }

    /// Set the GPU render target format.
    ///
    /// Defaults to [`OutputFormat::Rgba8Unorm`] (suitable for encoding).
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Set the input pixel format.
    ///
    /// Defaults to [`InputFormat::Yuv420p`].
    pub fn input_format(mut self, format: InputFormat) -> Self {
        self.input_format = format;
        self
    }

    /// Provide a pre-initialized GPU context.
    ///
    /// If not set, the builder will auto-detect the best available GPU.
    pub fn gpu(mut self, gpu: GpuContext) -> Self {
        self.gpu = Some(gpu);
        self
    }

    /// Attach an encoder with the given double-buffer count.
    pub fn encoder(mut self, encoder: Box<dyn Encoder + Send>, buffer_count: usize) -> Self {
        self.encoder = Some((encoder, buffer_count));
        self
    }

    /// Attach a GPU-resident encoder for zero-copy encode.
    ///
    /// A [`GpuEncoder`] receives `wgpu::Texture` references directly,
    /// avoiding the GPU-to-CPU readback that the regular [`Encoder`] path
    /// requires. No implementations exist yet - this reserves the API slot
    /// for future NVENC/VideoToolbox GPU encode backends.
    pub fn gpu_encoder(mut self, encoder: Box<dyn GpuEncoder>) -> Self {
        self.gpu_encoder = Some(encoder);
        self
    }

    /// Attach a [`UnifiedDetector`](crate::detector::UnifiedDetector).
    pub fn detector(mut self, detector: Box<dyn crate::detector::UnifiedDetector>) -> Self {
        self.detector = Some(detector);
        self
    }

    /// Attach a director for camera panning.
    pub fn director(mut self, director: Box<dyn Director>) -> Self {
        self.director = Some(director);
        self
    }

    /// Set the detection interval (run detection every N frames).
    ///
    /// `1` = every frame (default), `3` = every 3rd frame, etc.
    /// Detection is expensive (YOLO at 2-20ms/frame), so skipping
    /// frames lets the render loop run faster while the director
    /// interpolates using the latest detections.
    pub fn detection_interval(mut self, interval: u64) -> Self {
        self.detection_interval = interval.max(1);
        self
    }

    /// Set the number of frames to buffer ahead for lookahead.
    ///
    /// See [`StitchSession::set_lookahead`] for details.
    pub fn lookahead(mut self, frames: usize) -> Self {
        self.lookahead_frames = frames;
        self
    }

    /// Build the session, initializing GPU if not provided.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`] if required fields are missing or GPU
    /// initialization fails.
    pub fn build(self) -> Result<StitchSession, SessionError> {
        let calibration = self.calibration.ok_or_else(|| {
            SessionError::Config("StitchSessionBuilder: calibration is required".into())
        })?;
        let input_width = self.input_width.ok_or_else(|| {
            SessionError::Config("StitchSessionBuilder: input_dimensions is required".into())
        })?;
        let input_height = self.input_height.ok_or_else(|| {
            SessionError::Config("StitchSessionBuilder: input_dimensions is required".into())
        })?;

        let viewport = self.viewport.unwrap_or(ViewportConfig {
            width: 1920,
            height: 1080,
            blend_width: 0.15,
            ..Default::default()
        });

        let gpu = match self.gpu {
            Some(g) => g,
            None => pollster::block_on(GpuContext::new())?,
        };

        let config = SessionConfig {
            calibration,
            viewport,
            input_width,
            input_height,
            output_format: self.output_format,
            input_format: self.input_format,
            left_rotation: 0,
            right_rotation: 0,
        };

        let mut session = StitchSession::with_gpu(gpu, config)?;
        session
            .detection
            .set_detection_interval(self.detection_interval);
        session.lookahead_frames = self.lookahead_frames;

        if let Some((enc, buf_count)) = self.encoder {
            session.set_encoder(enc, buf_count);
        }
        session.gpu_encoder = self.gpu_encoder;
        if let Some(det) = self.detector {
            session.set_detector(det);
        }
        if let Some(dir) = self.director {
            session.set_director(dir);
        }
        Ok(session)
    }
}

/// A high-level stitching session that owns the GPU pipeline, NV12
/// converter, and optionally an async encoder.
///
/// Created once per encoding job or application lifetime. Call
/// [`set_encoder`](Self::set_encoder) to attach an encoder before
/// rendering, then use [`submit_render_output`](Self::submit_render_output)
/// for per-frame control or [`run`](Self::run) for batch processing.
/// Call [`finish`](Self::finish) to flush the last frame and finalize
/// encoding.
pub struct StitchSession {
    /// The canonical push-first core. Owns the `StitchPipeline`,
    /// readback staging, coverage boundary, and director slot. The
    /// session's director + legacy-detector path delegates pose +
    /// coverage decisions to `self.core` during the plan-step-2
    /// transition; later tranches will migrate the legacy
    /// `DetectionPipeline` into the core too.
    pub(crate) core: StitchCore,
    pub(crate) nv12_converter: Nv12Converter,
    pub(crate) encoder: Option<AsyncEncodeThread>,
    /// Additional encoders for multi-output (stream + record).
    extra_encoders: Vec<AsyncEncodeThread>,
    /// GPU-resident encoder (zero-copy encode path, no implementations yet).
    #[allow(dead_code)]
    gpu_encoder: Option<Box<dyn GpuEncoder>>,
    /// Detection backends, interval, callback, and cached detections.
    pub(crate) detection: DetectionPipeline,
    pub(crate) director: Option<Box<dyn Director>>,
    pub(crate) frame_count: u64,
    /// Session start time for metrics computation.
    session_start: Option<std::time::Instant>,
    /// Error policy for the run() batch loop.
    error_policy: ErrorPolicy,
    /// Dropped frame counter (for metrics).
    frames_dropped: u64,
    /// Number of frames to buffer ahead for lookahead.
    /// When > 0, detection runs ahead of rendering so the director
    /// anticipates action before it reaches the encoder.
    lookahead_frames: usize,
    // ── GPU-resident source state (populated by configure_from_source) ──
    /// Bind groups for GPU-resident shared textures.
    /// Created lazily from the source's textures at the start of run().
    #[cfg(target_os = "linux")]
    gpu_bind_groups: Option<crate::pipeline::GpuSourceBindGroups>,
    /// Slot-free senders for decode backpressure (GPU zero-copy).
    #[cfg(target_os = "linux")]
    gpu_slot_free_tx: Option<(
        std::sync::mpsc::SyncSender<u8>,
        std::sync::mpsc::SyncSender<u8>,
    )>,
    /// CUDA buffer info for GPU detection (GPU zero-copy).
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    gpu_buf_info: Option<(crate::zero_copy::GpuBufInfo, crate::zero_copy::GpuBufInfo)>,

    /// Metal texture cache for importing CVPixelBuffers as wgpu textures.
    /// Created lazily on the first MetalResident frame.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    metal_texture_cache: Option<crate::metal_interop::MetalTextureCache>,

    /// Camera rotation from stream metadata, populated by
    /// [`configure_from_source`](Self::configure_from_source).
    /// Used to tell the GPU detector to flip frames during preprocessing.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    left_rotation: i32,
    /// Right camera rotation from stream metadata.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    right_rotation: i32,
}

impl StitchSession {
    /// Create a builder for configuring and constructing a session.
    pub fn builder() -> StitchSessionBuilder {
        StitchSessionBuilder {
            calibration: None,
            viewport: None,
            input_width: None,
            input_height: None,
            output_format: OutputFormat::Rgba8Unorm,
            input_format: InputFormat::Yuv420p,
            gpu: None,
            encoder: None,
            gpu_encoder: None,
            detector: None,
            director: None,
            detection_interval: 1,
            lookahead_frames: 0,
        }
    }

    /// Create a new session, initializing the GPU automatically.
    pub async fn new(config: SessionConfig) -> Result<Self, SessionError> {
        let gpu = GpuContext::new().await?;
        Self::with_gpu(gpu, config)
    }

    /// Create a session with an existing GPU context.
    ///
    /// Use this when the caller needs to control GPU selection (e.g.
    /// for zero-copy decode where the GPU must match the CUDA device).
    pub fn with_gpu(gpu: GpuContext, config: SessionConfig) -> Result<Self, SessionError> {
        let output_width = config.viewport.width;
        let output_height = config.viewport.height;

        // Build a `StitchCore` as the session's rendering foundation.
        // Core owns the pipeline + readback + coverage + projection +
        // camera_input. The session layers on NV12 conversion, async
        // encoding, lookahead, and the legacy per-platform detection
        // pipeline (until the unified-detector migration of the
        // session body completes).
        //
        // Rotation is NOT applied here. It's handled by:
        // - CPU path: decoder reverses buffers in extract_yuv()
        // - GPU path: configure_from_source() sets shader UV flip in run()
        // SessionConfig.left_rotation/right_rotation are kept for Layer 1
        // consumers who call set_flip_180() manually.
        let core = StitchCore::new(
            gpu,
            StitchCoreConfig {
                calibration: config.calibration,
                viewport: config.viewport,
                input_width: config.input_width,
                input_height: config.input_height,
                // `OutputFormat` -> `wgpu::TextureFormat` via the
                // `From` impl in `crate::gpu`; covers all three
                // session-facing variants (Rgba8Unorm, Rgba8UnormSrgb,
                // Bgra8UnormSrgb).
                output_format: config.output_format.into(),
                input_format: config.input_format,
                projection: None,
                camera_input: None,
                replay_buffer_duration: None,
            },
        )?;

        let nv12_converter = Nv12Converter::new(core.gpu(), output_width, output_height)?;

        Ok(Self {
            core,
            nv12_converter,
            encoder: None,
            gpu_encoder: None,
            detection: DetectionPipeline::new(),
            director: None,
            frame_count: 0,
            extra_encoders: Vec::new(),
            session_start: None,
            error_policy: ErrorPolicy::default(),
            frames_dropped: 0,
            lookahead_frames: 0,
            #[cfg(target_os = "linux")]
            gpu_bind_groups: None,
            #[cfg(target_os = "linux")]
            gpu_slot_free_tx: None,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            gpu_buf_info: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal_texture_cache: None,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            left_rotation: 0,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            right_rotation: 0,
        })
    }

    /// The precomputed coverage boundary for "no-black" viewport constraining.
    ///
    /// Delegates to [`StitchCore::coverage`]; use
    /// [`CoverageBoundary::safe_clamp`](crate::projection::CoverageBoundary::safe_clamp) to
    /// constrain viewport positions, or
    /// [`CoverageBoundary::max_fov_degrees`](crate::projection::CoverageBoundary::max_fov_degrees)
    /// for the zoom-out ceiling.
    pub fn coverage(&self) -> Option<&crate::projection::CoverageBoundary> {
        self.core.coverage()
    }

    /// Full angular extent of the stitched panorama.
    ///
    /// Higher-level shortcut for analytics consumers (heatmaps, zone
    /// statistics) that want the coverage bounds without reaching into
    /// [`CoverageBoundary`](crate::projection::CoverageBoundary). Returns
    /// `None` if the session has no coverage boundary (should not happen
    /// for sessions built from a valid calibration).
    pub fn panorama_extent(&self) -> Option<crate::projection::PanoramaExtent> {
        self.core.coverage().map(|c| {
            let (yaw_min, yaw_max) = c.yaw_range();
            let (pitch_min, pitch_max) = c.pitch_range();
            crate::projection::PanoramaExtent {
                yaw_min,
                yaw_max,
                pitch_min,
                pitch_max,
            }
        })
    }

    /// Attach an encoder to this session.
    ///
    /// The encoder is moved to a background thread for async encoding.
    /// `buffer_count` controls how many frames can be in-flight between
    /// the render thread and the encode thread (typically 2).
    ///
    /// Must be called before [`Self::submit_render_output`], [`Self::process_frame`],
    /// or [`Self::run`].
    pub fn set_encoder(&mut self, encoder: Box<dyn Encoder + Send>, buffer_count: usize) {
        let width = self.nv12_converter.width();
        let height = self.nv12_converter.height();
        self.encoder = Some(AsyncEncodeThread::new(encoder, width, height, buffer_count));
    }

    /// Attach a [`UnifiedDetector`](crate::detector::UnifiedDetector)
    /// for object detection on raw camera frames.
    ///
    /// The backend declares which [`DetectorFrame`](crate::detector::DetectorFrame)
    /// residencies it accepts. Session dispatches CPU frames (YUV /
    /// NV12) and CUDA frames (shared textures) through the same
    /// detector; backends return `UnsupportedFrameKind` for residencies
    /// they cannot handle and session logs+drops those at the boundary.
    pub fn set_detector(&mut self, detector: Box<dyn crate::detector::UnifiedDetector>) {
        self.detection.set_detector(detector);
    }

    /// Set the detection interval (run detection every N frames).
    ///
    /// Default is 1 (every frame). Higher values reduce detection CPU load
    /// at the cost of tracking responsiveness. The director still receives
    /// the last known tracked objects on skipped frames.
    pub fn set_detection_interval(&mut self, interval: u64) {
        self.detection.set_detection_interval(interval);
    }

    /// Set the number of frames to buffer ahead for lookahead.
    ///
    /// When > 0, the CPU batch loop decodes and runs detection on frames
    /// ahead of rendering, so the director "sees" the future relative to
    /// what's being encoded. This makes the camera anticipate action
    /// rather than react to it.
    ///
    /// Typical value: `(fps * 0.5) as usize` for 0.5s lead time.
    /// Only affects the CPU path ([`Self::run`]). Zero-copy paths are
    /// not supported (frames are GPU-resident and can't be buffered).
    pub fn set_lookahead(&mut self, frames: usize) {
        self.lookahead_frames = frames;
    }

    /// Attach a director for AI-driven or scripted camera panning.
    ///
    /// When set, batch methods (`run`, `run_zero_copy_linux`,
    /// `run_zero_copy_macos`) use the director's viewport position
    /// instead of the default centered view.
    ///
    /// The director receives a [`DirectorContext`] each frame containing
    /// tracked objects with panorama coordinates and valid panning bounds.
    pub fn set_director(&mut self, director: Box<dyn Director>) {
        self.director = Some(director);
    }

    /// Set the sink that receives tracked detection data each frame.
    ///
    /// The sink is called once per frame with the current tracked
    /// objects, frame index, and timestamp. Errors returned from the
    /// sink abort the current session call ([`run`](Self::run),
    /// [`step`](Self::step), [`process_frame`](Self::process_frame))
    /// with [`SessionError::DetectionSink`].
    ///
    /// Closures matching `FnMut(&[MappedDetection], u64, f64) -> Result<(),
    /// DetectionSinkError>` implement [`DetectionSink`] automatically via
    /// the blanket impl, so typical usage is:
    ///
    /// ```rust,ignore
    /// session.set_detection_sink(Box::new(|dets, frame_idx, ts_ms| {
    ///     writer.write_row(dets, frame_idx, ts_ms)?;
    ///     Ok(())
    /// }));
    /// ```
    ///
    /// Replaces any previously registered sink.
    pub fn set_detection_sink(&mut self, sink: Box<dyn DetectionSink>) {
        self.detection.set_sink(sink);
    }

    /// Get the current viewport position from the director, or default.
    ///
    /// Clamps the director's raw output to the coverage boundary (no-black
    /// region) and applies FOV limits. This keeps all viewport constraining
    /// in the session, so directors can output unconstrained positions.
    pub fn director_position(&mut self) -> ViewportPosition {
        let mut pos = self
            .director
            .as_ref()
            .map_or(ViewportPosition::default(), |d| d.position());

        // The director outputs world-space coordinates (from ball detections).
        // Clamp in world space, then convert to user space for the renderer
        // (which applies rig_tilt internally in its view matrix).
        if let Some(coverage) = self.core.coverage() {
            if let Some(ref mut fov) = pos.fov_degrees {
                *fov = fov.min(coverage.max_fov_degrees());
            }
            let fov = pos
                .fov_degrees
                .unwrap_or_else(|| self.core.pipeline().fov());
            let aspect = self.core.pipeline().viewport().aspect_ratio();
            // Clamp in world space (no rig_tilt transform needed).
            let clamped = coverage.safe_clamp(pos.yaw, pos.pitch, fov, aspect, 0.0);
            pos.yaw = clamped.yaw;
            // Convert world -> user: the renderer applies rig_tilt as
            // a rotation, so the effective world pitch at a given yaw is
            // user_pitch + rig_tilt * cos(yaw). Invert to get user_pitch.
            let rig_tilt = self.core.pipeline().viewport().rig_tilt;
            pos.pitch = clamped.pitch - rig_tilt * clamped.yaw.cos();
        }

        if let Some(fov) = pos.fov_degrees {
            self.core.pipeline_mut().set_fov(fov);
        }
        pos
    }

    /// Run detection on a stereo frame, track, map to panorama, and update the director.
    ///
    /// Detection only runs every `detection_interval` frames. On skipped
    /// frames, the last tracked objects are reused so the director still
    /// has context. The detection sink fires every frame; an error from
    /// the sink aborts this call with [`SessionError::DetectionSink`].
    pub fn detect_and_update_director(
        &mut self,
        frame: &StereoFrame,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        let should_detect = self.detection.should_detect(self.frame_count);

        if should_detect {
            let (width, height) = self.core.pipeline().source_info();
            let detections = self.detection.run_detection(frame, width, height);
            self.detection.last_detections = self.map_detections(detections);
        }

        self.fire_sink_and_update_director(elapsed, should_detect)
    }

    /// Update the director without detection (zero-copy paths).
    ///
    /// No CPU-accessible frame data is available, so detection is skipped.
    /// The director still receives context with empty objects and valid bounds.
    #[cfg_attr(
        any(target_os = "linux", target_os = "windows"),
        allow(dead_code, reason = "used by macOS zero-copy path")
    )]
    pub(crate) fn update_director(
        &mut self,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        self.fire_sink_and_update_director(elapsed, false)
    }

    /// Run GPU-resident detection and update the director.
    ///
    /// Uses the [`GpuDetector`](crate::detector::GpuDetector) to detect
    /// objects directly from CUDA device pointers (NV12 shared textures),
    /// avoiding any GPU-to-CPU frame readback. Only the small detection
    /// output is transferred to CPU for tracking and director updates.
    ///
    /// Falls back to [`update_director`](Self::update_director) if no
    /// GPU detector is attached.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) fn detect_and_update_director_gpu(
        &mut self,
        left_buf: &crate::zero_copy::GpuBufInfo,
        right_buf: &crate::zero_copy::GpuBufInfo,
        left_slot: u8,
        right_slot: u8,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        let should_detect = self.detection.should_detect(self.frame_count);

        if should_detect && self.detection.has_detector() {
            let detections = self.detection.run_gpu_detection(
                left_buf,
                right_buf,
                left_slot,
                right_slot,
                self.left_rotation,
                self.right_rotation,
            );
            self.detection.last_detections = self.map_detections(detections);
        }

        self.fire_sink_and_update_director(elapsed, should_detect)
    }

    /// Run Metal-resident detection and update the director.
    ///
    /// Dispatches to the attached unified detector through
    /// [`DetectorFrame::Metal`](crate::detector::DetectorFrame::Metal).
    /// The backend (e.g. `MetalYoloDetector`) owns the `GpuContext`
    /// clone it needs for CVPixelBuffer import.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) fn detect_and_update_director_metal(
        &mut self,
        left_cvpb: crate::metal_interop::CVPixelBufferRef,
        right_cvpb: crate::metal_interop::CVPixelBufferRef,
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        use crate::detector::{CameraId, DetectorFrame};

        let should_detect = self.detection.should_detect(self.frame_count);

        if should_detect && self.detection.has_detector() {
            if let Some(ref mut detector) = self.detection.detector {
                let mut raw = Vec::new();
                for (camera, cvpb) in [(CameraId::Left, left_cvpb), (CameraId::Right, right_cvpb)] {
                    let frame = DetectorFrame::Metal {
                        cv_pixel_buffer: cvpb,
                        width,
                        height,
                    };
                    match detector.detect(camera, &frame) {
                        Ok(v) => raw.extend(v),
                        Err(e) => log::warn!("detector '{}' {camera:?}: {e}", detector.name()),
                    }
                }
                self.detection.last_detections = self.map_detections(raw);
            }
        }

        self.fire_sink_and_update_director(elapsed, should_detect)
    }

    /// Fire the detection sink and update the director with current state.
    ///
    /// Shared tail for all detection paths (CPU, GPU, Metal, no-detection).
    /// Fires the sink with `last_detections` (which may be empty if no
    /// detector ran this frame) and passes a [`DirectorContext`] to the
    /// director. Viewport constraining is handled separately by
    /// [`director_position`](Self::director_position).
    ///
    /// Sink errors surface as [`SessionError::DetectionSink`] and abort the
    /// current session call.
    fn fire_sink_and_update_director(
        &mut self,
        elapsed: std::time::Duration,
        fresh_detection: bool,
    ) -> Result<(), SessionError> {
        let timestamp_ms = elapsed.as_secs_f64() * 1000.0;

        // Fire sink for external consumers.
        self.detection
            .fire_sink(self.frame_count, timestamp_ms)
            .map_err(|e| SessionError::DetectionSink(e.to_string()))?;

        // Update director with detections and timing only.
        // Coverage clamping is applied later in director_position().
        if let Some(ref mut director) = self.director {
            let ctx = DirectorContext {
                frame_index: self.frame_count,
                timestamp_ms,
                detections: &self.detection.last_detections,
                fresh_detection,
            };
            director.update(&ctx);
        }
        Ok(())
    }

    /// Map raw detections to panorama coordinates.
    ///
    /// Each detection's camera-space center is projected to panorama
    /// yaw/pitch via [`camera_to_panorama`](projection::camera_to_panorama).
    ///
    /// ROI filtering (discarding detections outside the playing field) is
    /// handled at the detector level by `reco-autocam`'s `RoiFilteredDetector`
    /// decorators, so this method is pure coordinate mapping.
    fn map_detections(&self, detections: Vec<Detection>) -> Vec<MappedDetection> {
        let calibration = self.core.pipeline().calibration();
        let scene = &self.core.pipeline().scene;

        detections
            .iter()
            .map(|d| {
                let position = projection::camera_to_panorama(
                    d.camera,
                    d.center_x,
                    d.center_y,
                    calibration,
                    scene,
                );
                MappedDetection {
                    camera: d.camera,
                    class_id: d.class_id,
                    confidence: d.confidence,
                    camera_center: (d.center_x, d.center_y),
                    camera_size: (d.width, d.height),
                    position,
                }
            })
            .collect()
    }

    /// Render a single CPU-resident stereo frame and submit it to the encoder.
    ///
    /// Handles YUV420P and NV12 input formats. For GPU-resident frames
    /// (zero-copy path), use [`submit_render_output`](Self::submit_render_output)
    /// instead.
    /// Process one frame with full session features: detection, director,
    /// coverage clamping, and encoding.
    ///
    /// This is the recommended API for interactive consumers (GUI apps, OBS
    /// plugins) that control their own frame loop. It combines
    /// `detect_and_update_director()`, `director_position()`, and
    /// `process_frame()` into a single call and returns what happened.
    ///
    /// Pass `override_position` to bypass the director (e.g. when the user
    /// grabs the viewport with their mouse). The director still updates
    /// internally so it stays warm.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_step")
    )]
    pub fn step(
        &mut self,
        frame: &StereoFrame,
        elapsed: std::time::Duration,
        override_position: Option<ViewportPosition>,
    ) -> Result<StepResult, SessionError> {
        // Run detection and update director.
        self.detect_and_update_director(frame, elapsed)?;

        // Get viewport position (from director or override).
        let pos = if let Some(ovr) = override_position {
            if let Some(fov) = ovr.fov_degrees {
                self.core.pipeline_mut().set_fov(fov);
            }
            ovr
        } else {
            self.director_position()
        };

        // Capture detections before render (they'll be overwritten on next detect).
        let detections = self.detection.last_detections.clone();
        let frame_index = self.frame_count;

        // Render + encode.
        self.process_frame(frame, pos.yaw, pos.pitch)?;

        Ok(StepResult {
            viewport: pos,
            detections,
            frame_index,
        })
    }

    /// Render a single CPU-resident stereo frame and submit it to the encoder.
    ///
    /// Handles YUV420P and NV12 input formats. For GPU-resident frames
    /// (zero-copy path), use [`submit_render_output`](Self::submit_render_output)
    /// instead.
    ///
    /// For interactive consumers that want detection + director + encoding in
    /// one call, use [`step()`](Self::step) instead.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_process_frame")
    )]
    pub fn process_frame(
        &mut self,
        frame: &StereoFrame,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), SessionError> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let StereoFrame::MetalResident { left, right } = frame {
            return self.process_metal_frame(left, right, yaw, pitch);
        }

        let render_buf = self.core.render_stereo_frame_at_pose(frame, yaw, pitch)?;
        self.submit_render_output(render_buf)
    }

    /// Process a MetalResident frame: import CVPixelBuffers as textures, render.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn process_metal_frame(
        &mut self,
        left: &crate::metal_interop::RetainedCVPixelBuffer,
        right: &crate::metal_interop::RetainedCVPixelBuffer,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), SessionError> {
        // Lazily create the texture cache on first MetalResident frame.
        if self.metal_texture_cache.is_none() {
            self.metal_texture_cache = Some(crate::metal_interop::MetalTextureCache::new(
                self.core.gpu(),
            )?);
            log::info!("Metal zero-copy: texture cache initialized");
        }
        let cache = self.metal_texture_cache.as_ref().unwrap();

        // SAFETY: RetainedCVPixelBuffer guarantees the pointer is valid.
        let (left_y, left_uv) = unsafe { cache.import_nv12(left.as_ptr(), self.core.gpu())? };
        let (right_y, right_uv) = unsafe { cache.import_nv12(right.as_ptr(), self.core.gpu())? };

        let render_buf = self.core.render_imported_textures_at_pose(
            &left_y.texture,
            &left_uv.texture,
            &right_y.texture,
            &right_uv.texture,
            yaw,
            pitch,
        );
        self.submit_render_output(render_buf)
    }

    /// Render from GPU-resident textures and submit to the async encoder.
    ///
    /// Used with the zero-copy path where decode threads write directly
    /// to shared GPU textures. The caller must configure bind groups via
    /// [`pipeline_mut()`](Self::pipeline_mut) and call
    /// [`StitchPipeline::render_gpu_frame`] to get the command buffer,
    /// then pass it here.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_submit_render")
    )]
    pub fn submit_render_output(
        &mut self,
        render_commands: wgpu::CommandBuffer,
    ) -> Result<(), SessionError> {
        let nv12_data = self.nv12_converter.convert_and_readback(
            self.core.gpu(),
            self.core.pipeline().render_target(),
            render_commands,
        )?;

        // First two calls return None (triple-buffer warmup).
        // From the third call onward, we get data from 2 frames ago.
        if let Some(data) = nv12_data {
            if let Some(ref encoder) = self.encoder {
                encoder.submit(data, self.frame_count as i64)?;
            }
            // Fan out to extra encoders (multi-output).
            for enc in &self.extra_encoders {
                enc.submit(data, self.frame_count as i64)?;
            }
        }

        self.frame_count += 1;
        Ok(())
    }

    /// Process one GPU-resident frame with pre-extracted buffer info.
    ///
    /// The `buf_info` is extracted once before the frame loop to avoid
    /// per-frame clones and satisfy the borrow checker.
    #[cfg(target_os = "linux")]
    fn step_gpu_with_bufs(
        &mut self,
        buf_info: &Option<(crate::zero_copy::GpuBufInfo, crate::zero_copy::GpuBufInfo)>,
        left_slot: u8,
        right_slot: u8,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        if let Some((left_buf, right_buf)) = buf_info {
            self.detect_and_update_director_gpu(
                left_buf, right_buf, left_slot, right_slot, elapsed,
            )?;
        }
        let pos = self.director_position();

        let bind_groups = self.gpu_bind_groups.as_ref().ok_or_else(|| {
            SessionError::ZeroCopy(
                "GPU bind groups not configured - call setup_gpu_source() before run()".into(),
            )
        })?;
        let render_buf = self.core.render_gpu_frame_at_pose(
            bind_groups,
            left_slot,
            right_slot,
            pos.yaw,
            pos.pitch,
        );
        self.submit_render_output(render_buf)?;

        // Release slots for decode thread to reuse
        if let Some((ref left_tx, ref right_tx)) = self.gpu_slot_free_tx {
            if left_tx.send(left_slot).is_err() {
                log::error!(
                    "Failed to release left GPU slot {left_slot} - decode thread may have died"
                );
            }
            if right_tx.send(right_slot).is_err() {
                log::error!(
                    "Failed to release right GPU slot {right_slot} - decode thread may have died"
                );
            }
        }

        Ok(())
    }

    /// Auto-configure the session from source metadata.
    ///
    /// Called at the start of [`run`](Self::run). Applies rotation from
    /// the source's metadata.
    fn configure_from_source(&mut self, source: &dyn FrameSource) {
        // Apply rotation via shader UV flip ONLY for GPU-resident sources.
        // CPU sources handle rotation via buffer reversal in the decoder,
        // so applying the shader flip too would rotate 360 degrees (no-op but wrong).
        if source.is_gpu_resident() {
            let (lr, rr) = (source.left_rotation(), source.right_rotation());
            if lr == 180 || rr == 180 {
                self.core.pipeline_mut().set_flip_180(lr == 180, rr == 180);
                log::info!("Rotation: UV flip left={}, right={}", lr == 180, rr == 180);
            }
            // Store rotation for the GPU detector preprocessing path.
            // The detector needs to flip frames independently of the render shader.
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            {
                self.left_rotation = lr;
                self.right_rotation = rr;
            }
        }
    }

    /// Configure the session for a GPU-resident source.
    ///
    /// Creates bind groups from the source's shared textures and stores
    /// slot-free senders for decode backpressure. Call this before
    /// [`run`](Self::run) when using a GPU-resident [`FrameSource`] like
    /// `SmartFileSource`.
    ///
    /// For the Layer 1 API (`run_zero_copy_linux`), this is handled
    /// internally and you don't need to call it.
    #[cfg(target_os = "linux")]
    pub fn setup_gpu_source(&mut self, shared: &SharedTextureSet) {
        let t = &shared.textures;
        let bind_groups = self.core.pipeline_mut().configure_gpu_source(
            [(&t[0], &t[1]), (&t[2], &t[3])],
            [(&t[4], &t[5]), (&t[6], &t[7])],
        );
        self.gpu_bind_groups = Some(bind_groups);
        self.gpu_slot_free_tx = Some((
            shared.left_slot_free_tx.clone(),
            shared.right_slot_free_tx.clone(),
        ));
        self.gpu_buf_info = Some((shared.left_buf.clone(), shared.right_buf.clone()));
        log::info!("Session configured for GPU-resident source");
    }

    /// Batch-process frames from a source into the encoder.
    ///
    /// Runs the full decode-render-encode loop until the source is
    /// exhausted, the frame limit is reached, or the interrupt flag
    /// is set. Returns the number of frames processed.
    ///
    /// Automatically handles CPU-resident and GPU-resident frames:
    /// - CPU frames (Yuv420p, Nv12): uploaded to GPU, rendered, encoded
    /// - GPU frames (GpuResident): rendered directly from shared textures
    ///
    /// When [`lookahead_frames`](Self::set_lookahead) > 0, decodes and
    /// runs detection on frames ahead of rendering. The director "sees"
    /// N frames into the future, so the camera anticipates action before
    /// it reaches the encoder.
    ///
    /// Does NOT call [`Self::finish`] - the caller must do that after this
    /// returns to flush the last frame and finalize encoding.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_run")
    )]
    pub fn run(
        &mut self,
        source: &mut dyn FrameSource,
        frame_limit: u64,
        interrupted: &AtomicBool,
        mut on_progress: Option<ProgressCallback>,
    ) -> Result<u64, SessionError> {
        self.configure_from_source(source);

        let result = if self.lookahead_frames > 0 {
            self.run_with_lookahead(source, frame_limit, interrupted, &mut on_progress)
        } else {
            self.run_immediate(source, frame_limit, interrupted, &mut on_progress)
        };

        // Drop GPU slot senders so decode threads can exit gracefully.
        // Without this, SmartFileSource::drop() deadlocks because the
        // session's cloned senders keep the decode threads' recv() alive.
        #[cfg(target_os = "linux")]
        {
            self.gpu_slot_free_tx = None;
        }

        result
    }

    /// Standard frame loop without lookahead.
    ///
    /// Handles both CPU-resident and GPU-resident frames transparently.
    fn run_immediate(
        &mut self,
        source: &mut dyn FrameSource,
        frame_limit: u64,
        interrupted: &AtomicBool,
        on_progress: &mut Option<ProgressCallback>,
    ) -> Result<u64, SessionError> {
        let start = std::time::Instant::now();

        // Extract GPU buf info once before the loop to avoid per-frame clones.
        // Needed to satisfy the borrow checker (immutable borrow of buf_info
        // vs mutable borrow for detect_and_update_director_gpu).
        #[cfg(target_os = "linux")]
        let gpu_buf_info = self.gpu_buf_info.clone();

        while self.frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
            let frame = {
                crate::profile_scope!("wait_decode");
                match source.next_frame()? {
                    Some(f) => f,
                    None => break,
                }
            };

            match &frame {
                #[cfg(target_os = "linux")]
                StereoFrame::GpuResident {
                    left_slot,
                    right_slot,
                } => {
                    self.step_gpu_with_bufs(
                        &gpu_buf_info,
                        *left_slot,
                        *right_slot,
                        start.elapsed(),
                    )?;
                }
                _ => {
                    // CPU-resident frames (Yuv420p, Nv12)
                    self.detect_and_update_director(&frame, start.elapsed())?;
                    let pos = self.director_position();
                    self.process_frame(&frame, pos.yaw, pos.pitch)?;
                }
            }

            if let Some(cb) = on_progress.as_mut() {
                cb(&FrameProgress {
                    frames_completed: self.frame_count,
                    elapsed: start.elapsed(),
                });
            }
        }

        Ok(self.frame_count)
    }

    /// Frame loop with lookahead buffering.
    ///
    /// Decodes `lookahead_frames` ahead of rendering so the director
    /// has seen future frames by the time each frame is rendered.
    fn run_with_lookahead(
        &mut self,
        source: &mut dyn FrameSource,
        frame_limit: u64,
        interrupted: &AtomicBool,
        on_progress: &mut Option<ProgressCallback>,
    ) -> Result<u64, SessionError> {
        use std::collections::VecDeque;

        let start = std::time::Instant::now();
        let lookahead = self.lookahead_frames;
        let mut buffer: VecDeque<StereoFrame> = VecDeque::with_capacity(lookahead + 1);

        // Track how many frames have been decoded (for frame_limit).
        let mut decoded_count: u64 = 0;

        // Pre-fill: decode lookahead frames and run detection on each,
        // but don't render yet. This advances the director ahead.
        for _ in 0..lookahead {
            if interrupted.load(Ordering::Relaxed) {
                break;
            }
            let frame = {
                crate::profile_scope!("wait_decode");
                match source.next_frame()? {
                    Some(f) => f,
                    None => break,
                }
            };
            decoded_count += 1;
            self.detect_and_update_director(&frame, start.elapsed())?;
            buffer.push_back(frame);
        }

        log::info!(
            "Lookahead: pre-filled {} frames (requested {})",
            buffer.len(),
            lookahead,
        );

        // Main loop: decode one new frame, detect on it (advancing director),
        // then render+encode the oldest buffered frame with the current
        // director position.
        while self.frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
            // Try to decode one more frame to keep the buffer full.
            if decoded_count < frame_limit {
                let frame = {
                    crate::profile_scope!("wait_decode");
                    source.next_frame()?
                };
                if let Some(f) = frame {
                    decoded_count += 1;
                    self.detect_and_update_director(&f, start.elapsed())?;
                    buffer.push_back(f);
                }
            }

            // Render the oldest buffered frame with the current director state.
            let Some(render_frame) = buffer.pop_front() else {
                break;
            };
            let pos = self.director_position();
            self.process_frame(&render_frame, pos.yaw, pos.pitch)?;

            if let Some(cb) = on_progress.as_mut() {
                cb(&FrameProgress {
                    frames_completed: self.frame_count,
                    elapsed: start.elapsed(),
                });
            }
        }

        Ok(self.frame_count)
    }

    /// Flush the NV12 triple-buffer and finalize the encoder.
    ///
    /// Drains all pending frames from the triple-buffer pipeline and
    /// submits them to the encoder, then shuts down the encode thread
    /// and calls [`Encoder::finish`]. Must be called after the frame loop ends.
    pub fn finish(&mut self) -> Result<(), SessionError> {
        // Flush remaining frames from the NV12 triple-buffer.
        while let Some(nv12_data) = self.nv12_converter.flush_pending(self.core.gpu())? {
            if let Some(ref encoder) = self.encoder {
                encoder.submit(nv12_data, self.frame_count as i64)?;
            }
            for enc in &self.extra_encoders {
                enc.submit(nv12_data, self.frame_count as i64)?;
            }
            self.frame_count += 1;
        }

        // Shut down all encode threads.
        if let Some(mut encoder) = self.encoder.take() {
            encoder.finish()?;
        }
        for mut enc in self.extra_encoders.drain(..) {
            enc.finish()?;
        }

        Ok(())
    }

    /// Convert a pre-rendered frame to NV12 without encoding.
    ///
    /// Returns NV12 data from 2 frames ago (or `None` on the first two calls).
    /// Used by the preview path where the caller displays frames directly
    /// instead of encoding them.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_convert_nv12")
    )]
    pub fn convert_to_nv12(
        &mut self,
        render_commands: wgpu::CommandBuffer,
    ) -> Result<Option<&[u8]>, SessionError> {
        let nv12_data = self.nv12_converter.convert_and_readback(
            self.core.gpu(),
            self.core.pipeline().render_target(),
            render_commands,
        )?;
        self.frame_count += 1;
        Ok(nv12_data)
    }

    /// Number of frames processed so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Shared reference to the underlying pipeline (via `StitchCore`).
    pub fn pipeline(&self) -> &StitchPipeline {
        self.core.pipeline()
    }

    /// Mutable reference to the underlying pipeline (via `StitchCore`).
    ///
    /// Needed for zero-copy setup (configure_gpu_source) and viewport
    /// changes (resize, set_fov).
    pub fn pipeline_mut(&mut self) -> &mut StitchPipeline {
        self.core.pipeline_mut()
    }

    /// Borrow the underlying [`StitchCore`]. Useful for consumers that
    /// want to reach through to the push-first API
    /// (`submit_frame_*`, replay buffer, etc.) without giving up the
    /// session's encode-loop features.
    pub fn core(&self) -> &StitchCore {
        &self.core
    }

    /// Mutable borrow of the underlying [`StitchCore`].
    pub fn core_mut(&mut self) -> &mut StitchCore {
        &mut self.core
    }

    /// Shared reference to the GPU context.
    pub fn gpu(&self) -> &GpuContext {
        self.core.gpu()
    }

    /// The name of the GPU this session is running on.
    pub fn gpu_name(&self) -> &str {
        self.core.pipeline().gpu_name()
    }

    /// Get current session performance metrics.
    pub fn metrics(&self) -> SessionMetrics {
        let elapsed = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let secs = elapsed.as_secs_f32().max(0.001);
        SessionMetrics {
            frames_processed: self.frame_count,
            frames_dropped: self.frames_dropped,
            elapsed,
            fps_average: self.frame_count as f32 / secs,
            total_frames: None, // set by consumer from SourceInfo
        }
    }

    /// Add an additional encoder for multi-output (e.g. record + stream).
    ///
    /// The NV12 data from each rendered frame is fanned out to all attached
    /// encoders. Each encoder runs on its own background thread.
    ///
    /// Use [`set_encoder`](Self::set_encoder) for the primary encoder,
    /// then `add_encoder` for additional outputs.
    pub fn add_encoder(&mut self, encoder: Box<dyn Encoder + Send>, buffer_count: usize) {
        let width = self.nv12_converter.width();
        let height = self.nv12_converter.height();
        self.extra_encoders
            .push(AsyncEncodeThread::new(encoder, width, height, buffer_count));
    }

    /// Set the error policy for the [`run()`](Self::run) batch loop.
    pub fn set_error_policy(&mut self, policy: ErrorPolicy) {
        self.error_policy = policy;
    }

    /// Update calibration parameters and recompute coverage boundary.
    ///
    /// Takes effect on the next render call. For interactive calibration
    /// tweaking during preview or live operation. Delegates to
    /// [`StitchCore::update_calibration`] which re-derives the coverage
    /// boundary in one call.
    pub fn update_calibration(&mut self, calibration: crate::calibration::MatchCalibration) {
        self.core.update_calibration(calibration);
    }
}
