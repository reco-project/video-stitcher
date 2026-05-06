//! Type definitions and builder for the stitching session.
//!
//! Extracted from `session/mod.rs` to keep the main module focused on
//! the `StitchSession` implementation. All public items are re-exported
//! from `crate::session` via `pub use types::*`.

use crate::calibration::MatchCalibration;
use crate::core::types::StitchCoreError;
use crate::director::{MappedDetection, ViewportPosition};
use crate::encoder::{EncodeError, Encoder};
use crate::gpu::{GpuContext, GpuError, OutputFormat};
use crate::nv12_converter::Nv12Error;
use crate::pipeline::PipelineError;
use crate::renderer::InputFormat;
use crate::source::SourceError;
use crate::viewport::ViewportConfig;

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

/// Configuration for creating a [`StitchSession`](super::StitchSession).
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

/// Callback for progress reporting during [`StitchSession::run`](super::StitchSession::run).
pub type ProgressCallback = Box<dyn FnMut(&FrameProgress) + Send>;

/// Result from [`StitchSession::step`](super::StitchSession::step) - one frame with full session features.
///
/// Detections are not returned here - consumers that need them should
/// attach a [`DetectionSink`] at construction. Keeping detections off
/// the per-frame return path avoids a `Vec<MappedDetection>` clone that
/// showed up on the plan M7.5 alloc audit with no in-tree consumer.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Where the virtual camera pointed for this frame.
    pub viewport: ViewportPosition,
    /// Frame index (0-based).
    pub frame_index: u64,
}

/// Session performance metrics for health monitoring.
///
/// Read via [`StitchSession::metrics`](super::StitchSession::metrics). Updated per-frame.
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

/// Error handling policy for [`StitchSession::run`](super::StitchSession::run).
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
/// up through [`StitchSession::run`](super::StitchSession::run) as a [`SessionError::DetectionSink`]
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

/// Errors from [`StitchSession`](super::StitchSession). `Clone + Send + Sync` so consumers
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

/// Builder for constructing a [`StitchSession`](super::StitchSession) with sensible defaults.
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
    pub(super) calibration: Option<MatchCalibration>,
    pub(super) viewport: Option<ViewportConfig>,
    pub(super) input_width: Option<u32>,
    pub(super) input_height: Option<u32>,
    pub(super) output_format: OutputFormat,
    pub(super) input_format: InputFormat,
    pub(super) gpu: Option<GpuContext>,
    pub(super) encoder: Option<(Box<dyn Encoder + Send>, usize)>,
    pub(super) detector: Option<Box<dyn crate::detector::UnifiedDetector>>,
    pub(super) detection_interval: u64,
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

    /// Attach a [`UnifiedDetector`](crate::detector::UnifiedDetector).
    pub fn detector(mut self, detector: Box<dyn crate::detector::UnifiedDetector>) -> Self {
        self.detector = Some(detector);
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

    /// Build the session, initializing GPU if not provided.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`] if required fields are missing or GPU
    /// initialization fails.
    pub fn build(self) -> Result<super::StitchSession, SessionError> {
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

        let mut session = super::StitchSession::with_gpu(gpu, config)?;
        session
            .detection
            .set_detection_interval(self.detection_interval);

        if let Some((enc, buf_count)) = self.encoder {
            session.set_encoder(enc, buf_count);
        }
        if let Some(det) = self.detector {
            session.set_detector(det);
        }
        Ok(session)
    }
}
