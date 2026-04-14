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
#[cfg(target_os = "macos")]
mod zero_copy_macos;

#[cfg(target_os = "linux")]
pub use zero_copy_linux::SharedTextureSet;

use std::sync::atomic::{AtomicBool, Ordering};

use crate::async_encode::AsyncEncodeThread;
use crate::calibration::MatchCalibration;
use crate::detector::{Detection, Detector};
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

/// Callback for receiving tracked detection data.
///
/// Called each frame with the tracked objects (may be empty on non-detection
/// frames or when no detector is configured). Use this to build external
/// consumers like coaching assistants, VAR systems, or stats pipelines.
///
/// Arguments: `(objects, frame_index, timestamp_ms)`
pub type DetectionCallback = Box<dyn FnMut(&[MappedDetection], u64, f64) + Send>;

/// Errors from [`StitchSession`].
#[derive(Debug, Error)]
pub enum SessionError {
    /// GPU initialization error.
    #[error("GPU: {0}")]
    Gpu(#[from] GpuError),

    /// GPU pipeline error.
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),

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
    #[cfg(target_os = "macos")]
    #[error("Metal interop: {0}")]
    MetalInterop(#[from] crate::metal_interop::MetalInteropError),

    /// Zero-copy setup or runtime error.
    #[error("zero-copy: {0}")]
    ZeroCopy(String),

    /// Missing or invalid configuration.
    #[error("config: {0}")]
    Config(String),
}

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
    detector: Option<Box<dyn Detector>>,
    director: Option<Box<dyn Director>>,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    gpu_detector: Option<Box<dyn crate::detector::GpuDetector>>,
    #[cfg(target_os = "macos")]
    metal_detector: Option<Box<dyn crate::detector::MetalDetector>>,
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

    /// Attach a detector for object detection on raw frames.
    pub fn detector(mut self, detector: Box<dyn Detector>) -> Self {
        self.detector = Some(detector);
        self
    }

    /// Attach a director for camera panning.
    pub fn director(mut self, director: Box<dyn Director>) -> Self {
        self.director = Some(director);
        self
    }

    /// Attach a GPU detector for zero-copy detection on CUDA device pointers.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn gpu_detector(mut self, detector: Box<dyn crate::detector::GpuDetector>) -> Self {
        self.gpu_detector = Some(detector);
        self
    }

    /// Attach a Metal detector for zero-copy detection on CVPixelBuffers.
    #[cfg(target_os = "macos")]
    pub fn metal_detector(mut self, detector: Box<dyn crate::detector::MetalDetector>) -> Self {
        self.metal_detector = Some(detector);
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
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(gpu_det) = self.gpu_detector {
            session.set_gpu_detector(gpu_det);
        }
        #[cfg(target_os = "macos")]
        if let Some(metal_det) = self.metal_detector {
            session.set_metal_detector(metal_det);
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
    pub(crate) pipeline: StitchPipeline,
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
    #[cfg(target_os = "macos")]
    metal_texture_cache: Option<crate::metal_interop::MetalTextureCache>,

    /// Precomputed coverage boundary for "no-black" viewport constraining.
    /// Built from calibration at session creation. In world space.
    /// Rig tilt correction is applied per-corner inside `safe_clamp`.
    coverage: Option<crate::projection::CoverageBoundary>,
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
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            gpu_detector: None,
            #[cfg(target_os = "macos")]
            metal_detector: None,
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

        let pipeline = StitchPipeline::with_gpu(
            gpu,
            config.calibration,
            config.viewport,
            config.input_width,
            config.input_height,
            config.output_format,
            config.input_format,
        )?;

        // Rotation is NOT applied here. It's handled by:
        // - CPU path: decoder reverses buffers in extract_yuv()
        // - GPU path: configure_from_source() sets shader UV flip in run()
        // SessionConfig.left_rotation/right_rotation are kept for Layer 1
        // consumers who call set_flip_180() manually.

        let nv12_converter = Nv12Converter::new(pipeline.gpu(), output_width, output_height)?;

        // Compute world-space coverage boundary from calibration (cheap, <1ms).
        // Rig tilt correction is applied per-corner inside safe_clamp.
        let coverage = crate::projection::CoverageBoundary::from_calibration(
            pipeline.calibration(),
            &pipeline.scene,
        );

        Ok(Self {
            pipeline,
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
            #[cfg(target_os = "macos")]
            metal_texture_cache: None,
            coverage: Some(coverage),
        })
    }

    /// The precomputed coverage boundary for "no-black" viewport constraining.
    ///
    /// Use [`CoverageBoundary::safe_clamp`](crate::projection::CoverageBoundary::safe_clamp) to constrain viewport positions,
    /// or [`CoverageBoundary::max_fov_degrees`](crate::projection::CoverageBoundary::max_fov_degrees) for the zoom-out ceiling.
    pub fn coverage(&self) -> Option<&crate::projection::CoverageBoundary> {
        self.coverage.as_ref()
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

    /// Attach a detector for object detection on raw camera frames.
    ///
    /// When set, the CPU batch loop ([`Self::run`]) runs detection on each
    /// frame's raw YUV data and maps results to panorama coordinates
    /// to the director. Zero-copy paths skip detection (no CPU-accessible
    /// frame data).
    pub fn set_detector(&mut self, detector: Box<dyn Detector>) {
        self.detection.set_detector(detector);
    }

    /// Attach a GPU detector for zero-copy detection on CUDA device pointers.
    ///
    /// When set, the zero-copy frame loop runs detection entirely on GPU
    /// using NV12 device pointers from shared textures. Only the small
    /// detection output is read back to CPU.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn set_gpu_detector(&mut self, detector: Box<dyn crate::detector::GpuDetector>) {
        self.detection.set_gpu_detector(detector);
    }

    /// Attach a Metal detector for zero-copy detection on CVPixelBuffers.
    ///
    /// When set, the macOS zero-copy frame loop runs detection using
    /// Metal compute shaders for preprocessing and CoreML for inference.
    #[cfg(target_os = "macos")]
    pub fn set_metal_detector(&mut self, detector: Box<dyn crate::detector::MetalDetector>) {
        self.detection.set_metal_detector(detector);
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

    /// Set a callback for receiving tracked detection data.
    ///
    /// Called each frame with the current tracked objects, frame index,
    /// and timestamp. Use this to build external consumers like coaching
    /// assistants, VAR systems, or stats pipelines.
    ///
    /// The callback receives the same [`MappedDetection`] data as the director,
    /// including panorama-space coordinates.
    pub fn set_detection_callback(&mut self, cb: DetectionCallback) {
        self.detection.set_callback(cb);
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
        if let Some(ref coverage) = self.coverage {
            if let Some(ref mut fov) = pos.fov_degrees {
                *fov = fov.min(coverage.max_fov_degrees());
            }
            let fov = pos.fov_degrees.unwrap_or_else(|| self.pipeline.fov());
            let aspect = self.pipeline.viewport().aspect_ratio();
            // Clamp in world space (no rig_tilt transform needed).
            let clamped = coverage.safe_clamp(pos.yaw, pos.pitch, fov, aspect, 0.0);
            pos.yaw = clamped.yaw;
            // Convert world -> user: the renderer applies rig_tilt as
            // a rotation, so the effective world pitch at a given yaw is
            // user_pitch + rig_tilt * cos(yaw). Invert to get user_pitch.
            let rig_tilt = self.pipeline.viewport().rig_tilt;
            pos.pitch = clamped.pitch - rig_tilt * clamped.yaw.cos();
        }

        if let Some(fov) = pos.fov_degrees {
            self.pipeline.set_fov(fov);
        }
        pos
    }

    /// Run detection on a stereo frame, track, map to panorama, and update the director.
    ///
    /// Detection only runs every `detection_interval` frames. On skipped
    /// frames, the last tracked objects are reused so the director still
    /// has context. The detection callback fires every frame.
    pub fn detect_and_update_director(
        &mut self,
        frame: &StereoFrame,
        elapsed: std::time::Duration,
    ) {
        let should_detect = self.detection.should_detect(self.frame_count);

        if should_detect {
            let (width, height) = self.pipeline.source_info();
            let detections = self.detection.run_detection(frame, width, height);
            self.detection.last_detections = self.map_detections(detections);
        }

        self.fire_callback_and_update_director(elapsed, should_detect);
    }

    /// Update the director without detection (zero-copy paths).
    ///
    /// No CPU-accessible frame data is available, so detection is skipped.
    /// The director still receives context with empty objects and valid bounds.
    #[cfg_attr(
        any(target_os = "linux", target_os = "windows"),
        allow(dead_code, reason = "used by macOS zero-copy path")
    )]
    pub(crate) fn update_director(&mut self, elapsed: std::time::Duration) {
        self.fire_callback_and_update_director(elapsed, false);
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
    ) {
        let should_detect = self.detection.should_detect(self.frame_count);

        if should_detect && self.detection.has_gpu_detector() {
            let detections = self
                .detection
                .run_gpu_detection(left_buf, right_buf, left_slot, right_slot);
            self.detection.last_detections = self.map_detections(detections);
        }

        self.fire_callback_and_update_director(elapsed, should_detect);
    }

    /// Run Metal-resident detection and update the director.
    ///
    /// Uses the [`MetalDetector`](crate::detector::MetalDetector) to detect
    /// objects directly from CVPixelBuffers via Metal compute shaders,
    /// avoiding any GPU-to-CPU frame readback. Only the small detection
    /// output is transferred to CPU for tracking and director updates.
    ///
    /// Falls back to [`update_director`](Self::update_director) if no
    /// Metal detector is attached.
    #[cfg(target_os = "macos")]
    pub(crate) fn detect_and_update_director_metal(
        &mut self,
        left_cvpb: crate::metal_interop::CVPixelBufferRef,
        right_cvpb: crate::metal_interop::CVPixelBufferRef,
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) {
        let should_detect = self.detection.should_detect(self.frame_count);

        if should_detect && self.detection.has_metal_detector() {
            let gpu = self.pipeline.gpu();
            let detections = self
                .detection
                .run_metal_detection(left_cvpb, right_cvpb, width, height, gpu);
            self.detection.last_detections = self.map_detections(detections);
        }

        self.fire_callback_and_update_director(elapsed, should_detect);
    }

    /// Fire the detection callback and update the director with current state.
    ///
    /// Shared tail for all detection paths (CPU, GPU, Metal, no-detection).
    /// Fires the callback with `last_detections` (which may be empty if no
    /// detector ran this frame) and passes a [`DirectorContext`] to the
    /// director. Viewport constraining is handled separately by
    /// [`director_position`](Self::director_position).
    fn fire_callback_and_update_director(
        &mut self,
        elapsed: std::time::Duration,
        fresh_detection: bool,
    ) {
        let timestamp_ms = elapsed.as_secs_f64() * 1000.0;

        // Fire callback for external consumers.
        self.detection.fire_callback(self.frame_count, timestamp_ms);

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
        let calibration = self.pipeline.calibration();
        let scene = &self.pipeline.scene;

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
        self.detect_and_update_director(frame, elapsed);

        // Get viewport position (from director or override).
        let pos = if let Some(ovr) = override_position {
            if let Some(fov) = ovr.fov_degrees {
                self.pipeline.set_fov(fov);
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
        #[cfg(target_os = "macos")]
        if let StereoFrame::MetalResident { left, right } = frame {
            return self.process_metal_frame(left, right, yaw, pitch);
        }

        let render_buf = self.pipeline.render_stereo_frame(frame, yaw, pitch)?;
        self.submit_render_output(render_buf)
    }

    /// Process a MetalResident frame: import CVPixelBuffers as textures, render.
    #[cfg(target_os = "macos")]
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
                self.pipeline.gpu(),
            )?);
            log::info!("Metal zero-copy: texture cache initialized");
        }
        let cache = self.metal_texture_cache.as_ref().unwrap();

        // SAFETY: RetainedCVPixelBuffer guarantees the pointer is valid.
        let (left_y, left_uv) = unsafe { cache.import_nv12(left.as_ptr(), self.pipeline.gpu())? };
        let (right_y, right_uv) =
            unsafe { cache.import_nv12(right.as_ptr(), self.pipeline.gpu())? };

        let render_buf = self.pipeline.render_imported_textures(
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
            self.pipeline.gpu(),
            self.pipeline.render_target(),
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
            );
        }
        let pos = self.director_position();

        let bind_groups = self.gpu_bind_groups.as_ref().ok_or_else(|| {
            SessionError::ZeroCopy(
                "GPU bind groups not configured - call setup_gpu_source() before run()".into(),
            )
        })?;
        let render_buf =
            self.pipeline
                .render_gpu_frame(bind_groups, left_slot, right_slot, pos.yaw, pos.pitch);
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
                self.pipeline.set_flip_180(lr == 180, rr == 180);
                log::info!("Rotation: UV flip left={}, right={}", lr == 180, rr == 180);
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
        let bind_groups = self.pipeline.configure_gpu_source(
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
                    self.detect_and_update_director(&frame, start.elapsed());
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
            self.detect_and_update_director(&frame, start.elapsed());
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
                    self.detect_and_update_director(&f, start.elapsed());
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
        while let Some(nv12_data) = self.nv12_converter.flush_pending(self.pipeline.gpu())? {
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
            self.pipeline.gpu(),
            self.pipeline.render_target(),
            render_commands,
        )?;
        self.frame_count += 1;
        Ok(nv12_data)
    }

    /// Number of frames processed so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Shared reference to the underlying pipeline.
    pub fn pipeline(&self) -> &StitchPipeline {
        &self.pipeline
    }

    /// Mutable reference to the underlying pipeline.
    ///
    /// Needed for zero-copy setup (configure_gpu_source) and viewport
    /// changes (resize, set_fov).
    pub fn pipeline_mut(&mut self) -> &mut StitchPipeline {
        &mut self.pipeline
    }

    /// Shared reference to the GPU context.
    pub fn gpu(&self) -> &GpuContext {
        self.pipeline.gpu()
    }

    /// The name of the GPU this session is running on.
    pub fn gpu_name(&self) -> &str {
        self.pipeline.gpu_name()
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
    /// tweaking during preview or live operation.
    pub fn update_calibration(&mut self, calibration: crate::calibration::MatchCalibration) {
        self.pipeline.update_calibration(calibration);
        self.coverage = Some(crate::projection::CoverageBoundary::from_calibration(
            self.pipeline.calibration(),
            &self.pipeline.scene,
        ));
    }
}
