//! Type definitions and builder for the stitching session.
//!
//! Shared run results, errors and progress types. GPU construction options
//! and the auto-initializing builder are available with the `gpu` feature.

#[cfg(feature = "gpu")]
use crate::calibration::Calibration;
use crate::core::types::StitchCoreError;
#[cfg(feature = "gpu")]
use crate::gpu::nv12_converter::Nv12Error;
#[cfg(feature = "gpu")]
use crate::gpu::{GpuContext, GpuError, OutputFormat};
#[cfg(feature = "gpu")]
use crate::render::pipeline::PipelineError;
#[cfg(feature = "gpu")]
use crate::render::renderer::InputFormat;
#[cfg(feature = "gpu")]
use crate::render::viewport::ViewportSize;
#[cfg(feature = "gpu")]
use crate::session::sinks::SinkOptions;
#[cfg(feature = "gpu")]
use crate::sink::OutputSink;
use crate::sink::SinkError;
use crate::source::SourceError;

use thiserror::Error;

/// Why a frame-processing loop stopped successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunCompletion {
    /// The source ended or the requested frame limit was reached.
    Completed,
    /// The loop observed a cancellation request.
    Cancelled,
}

/// Processing result before pending frames and sinks are finalized.
#[derive(Debug, Clone, Copy)]
pub struct RunResult {
    /// Number of frames processed by the session.
    pub frames_processed: u64,
    /// The stop reason observed by the processing loop.
    pub completion: RunCompletion,
}

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

/// GPU configuration for creating a [`StitchSession`](super::StitchSession).
#[derive(Debug)]
#[cfg(feature = "gpu")]
pub struct SessionConfig {
    /// Camera calibration data.
    pub calibration: Calibration,
    /// Output viewport dimensions.
    pub viewport_size: ViewportSize,
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

/// Pre-extracted state for the frame loop, cloned once before iterating.
///
/// The borrow checker prevents reading `self.gpu_buf_info` immutably
/// while calling `&mut self` methods (detect, render). Cloning the
/// buffer info once and passing it through this struct sidesteps the
/// conflict with trivial cost (~150 bytes of scalars).
#[derive(Default)]
pub(crate) struct FrameLoopContext {
    /// The buffered producer already ran detection and tracking.
    pub(crate) tracking_complete: bool,
    /// Output frame duration, independent of decode and processing time.
    pub(crate) frame_seconds: f32,
    /// CUDA buffer info for GPU detection (Linux zero-copy path).
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    pub(crate) gpu_buf_info: Option<Vec<crate::interop::zero_copy::GpuBufInfo>>,
}

/// Errors from [`StitchSession`](super::StitchSession). `Clone + Send + Sync` so consumers
/// posting session results across thread boundaries (reco-gui export
/// thread, reco-obs async init) can keep the typed enum instead of
/// falling back to `Result<_, String>`.
#[derive(Debug, Clone, Error)]
pub enum SessionError {
    /// GPU initialization error.
    #[error("GPU: {0}")]
    #[cfg(feature = "gpu")]
    Gpu(#[from] GpuError),

    /// GPU pipeline error.
    #[error("pipeline: {0}")]
    #[cfg(feature = "gpu")]
    Pipeline(#[from] PipelineError),

    /// `StitchCore` error (wraps pipeline / readback / config).
    #[error("core: {0}")]
    Core(#[from] StitchCoreError),

    /// NV12 conversion error (GPU converter).
    #[error("NV12 converter: {0}")]
    #[cfg(feature = "gpu")]
    Nv12(#[from] Nv12Error),

    /// NV12 conversion error (CPU kernel).
    #[error("CPU NV12 conversion: {0}")]
    CpuNv12(#[from] crate::render::nv12_cpu::Nv12CpuError),

    /// Output sink error.
    #[error("sink: {0}")]
    Sink(#[from] SinkError),

    /// Source error.
    #[error("source: {0}")]
    Source(#[from] SourceError),

    /// Metal interop error (macOS zero-copy).
    #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios")))]
    #[error("Metal interop: {0}")]
    MetalInterop(#[from] crate::interop::metal::MetalInteropError),

    /// Zero-copy setup or runtime error.
    #[error("zero-copy: {0}")]
    ZeroCopy(String),

    /// Missing or invalid configuration.
    #[error("config: {0}")]
    Config(String),
}

// Compile-time assertion: every error type reachable
// from the public session API is `Clone + Send + Sync`, so consumers
// that post results to worker-thread channels (reco-gui export
// thread, reco-obs async init) carry the typed error across the
// boundary instead of stringifying. Regresses if a future variant
// introduces a non-Clone wrapped error.
const _: fn() = || {
    fn assert_clone_send_sync<T: Clone + Send + Sync + 'static>() {}
    assert_clone_send_sync::<SessionError>();
    assert_clone_send_sync::<StitchCoreError>();
    assert_clone_send_sync::<crate::render::nv12_cpu::Nv12CpuError>();
    assert_clone_send_sync::<SinkError>();
    assert_clone_send_sync::<SourceError>();
    assert_clone_send_sync::<crate::detect::detector::DetectorError>();
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
///     .viewport_size(viewport)
///     .gpu(gpu)
///     .build()?;
/// ```
#[cfg(feature = "gpu")]
pub struct StitchSessionBuilder {
    pub(super) calibration: Option<Calibration>,
    pub(super) viewport_size: Option<ViewportSize>,
    pub(super) input_width: Option<u32>,
    pub(super) input_height: Option<u32>,
    pub(super) output_format: OutputFormat,
    pub(super) input_format: InputFormat,
    pub(super) gpu: Option<GpuContext>,
    pub(super) sinks: Vec<(Box<dyn OutputSink>, SinkOptions)>,
    pub(super) detector: Option<Box<dyn crate::detect::detector::Detector>>,
    pub(super) detection_interval: u64,
}

#[cfg(feature = "gpu")]
impl StitchSessionBuilder {
    /// Set the camera calibration (required).
    pub fn calibration(mut self, cal: Calibration) -> Self {
        self.calibration = Some(cal);
        self
    }

    /// Set the output viewport size. Defaults to 1920x1080 if not set.
    pub fn viewport_size(mut self, viewport_size: ViewportSize) -> Self {
        self.viewport_size = Some(viewport_size);
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

    /// Attach an output sink (see
    /// [`StitchSession::add_sink`](super::StitchSession::add_sink)).
    pub fn sink(mut self, sink: Box<dyn OutputSink>, options: SinkOptions) -> Self {
        self.sinks.push((sink, options));
        self
    }

    /// Attach a [`Detector`](crate::detect::detector::Detector).
    pub fn detector(mut self, detector: Box<dyn crate::detect::detector::Detector>) -> Self {
        self.detector = Some(detector);
        self
    }

    /// Set the detection interval (run detection every N frames).
    ///
    /// `1` = every frame (default), `3` = every 3rd frame, etc.
    /// Detection is expensive (YOLO at 2-20ms/frame), so skipping
    /// frames lets the render loop run faster while the panner
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

        let viewport_size = self.viewport_size.unwrap_or(ViewportSize {
            width: 1920,
            height: 1080,
        });

        let gpu = match self.gpu {
            Some(g) => g,
            None => pollster::block_on(GpuContext::new())?,
        };

        let config = SessionConfig {
            calibration,
            viewport_size,
            input_width,
            input_height,
            output_format: self.output_format,
            input_format: self.input_format,
            left_rotation: 0,
            right_rotation: 0,
        };

        let mut session = super::StitchSession::with_gpu(gpu, config)?;
        session.set_detection_interval(self.detection_interval);

        for (sink, options) in self.sinks {
            session.add_sink(sink, options)?;
        }
        if let Some(det) = self.detector {
            session.set_detector(det);
        }
        Ok(session)
    }
}
