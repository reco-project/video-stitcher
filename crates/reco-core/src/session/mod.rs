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
//! - [`StitchSession::run`] - batch-process an entire `FrameSource`
//!   into an encoder, with optional progress reporting and interrupt
//!   support. Use this for CLI batch encoding.

/// Session type definitions, error types, and builder.
pub mod types;

/// Detection dispatch entry points (detect_and_update_director_* variants).
mod detection_dispatch;
/// Lookahead frame buffer for temporal-aware processing.
pub(crate) mod frame_buffer;
/// Per-frame render and encode methods (step, process_frame, submit_render_output).
mod frame_processing;
/// Batch processing entry points (run, run_immediate, setup_gpu_source).
mod run_loop;
/// Configuration wiring (set/clear/attach methods).
mod wiring;

#[cfg(test)]
mod tests;

use crate::async_encode::AsyncEncodeThread;
use crate::core::StitchCore;
use crate::core::types::StitchCoreError;
use crate::gpu::{GpuContext, OutputFormat};
use crate::render::renderer::InputFormat;
use crate::stitch::{Executor, GpuExecutor, GpuExecutorConfig};

/// Callback type for the NV12 tap: receives `(nv12_data, width, height)`.
pub type Nv12TapFn = Box<dyn FnMut(&[u8], u32, u32) + Send>;

use types::{ErrorPolicy, SessionConfig, SessionError, SessionMetrics, StitchSessionBuilder};

/// A high-level stitching session: a pull-loop orchestrator over the
/// engine, adding frame buffering, encode fan-out, and telemetry.
///
/// Created once per encoding job or application lifetime. Call
/// [`set_encoder`](Self::set_encoder) to attach an encoder before
/// rendering, then use [`submit_render_output`](Self::submit_render_output)
/// for per-frame control or [`run`](Self::run) for batch processing.
/// Call [`finish`](Self::finish) to flush the last frame and finalize
/// encoding.
pub struct StitchSession {
    /// The canonical push-first engine. Owns the render substrate,
    /// readback staging, coverage boundary, and the single AI stack
    /// (detector, trackers, panner, event sink). The session is the
    /// pull-loop orchestrator over it: frame buffering, encode
    /// fan-out, telemetry, progress.
    pub(crate) core: StitchCore,
    pub(crate) encoder: Option<AsyncEncodeThread>,
    /// Additional encoders for multi-output (stream + record).
    pub(crate) extra_encoders: Vec<AsyncEncodeThread>,
    /// When true, `process_frame_any` skips detection (the produce phase
    /// already ran it and stored the WorldState in the buffer).
    pub(crate) skip_detection: bool,
    /// Number of lookahead frames (0 = disabled).
    pub(crate) lookahead_frames: usize,
    pub(crate) frame_count: u64,
    /// Session start time for metrics computation.
    session_start: Option<std::time::Instant>,
    /// Error policy for the run() batch loop.
    pub(crate) error_policy: ErrorPolicy,
    /// Dropped frame counter (for metrics).
    frames_dropped: u64,
    pub(crate) telemetry: crate::telemetry::TelemetryCollector,
    /// Sub-timing from the last `submit_render_output` call.
    /// Used by `process_frame_any` to split "stitch" into
    /// render / readback / encode for accurate telemetry.
    pub(crate) last_readback_time: std::time::Duration,
    pub(crate) last_submit_time: std::time::Duration,
    /// VRAM pool slot for the frame currently being rendered
    /// (buffered lookahead path; the pool itself lives on the
    /// GPU executor).
    pub(crate) current_vram_slot: Option<usize>,

    /// Camera rotation from stream metadata, populated by
    /// [`configure_from_source`](Self::configure_from_source).
    /// Used to tell the GPU detector to flip frames during preprocessing.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) left_rotation: i32,
    /// Right camera rotation from stream metadata.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) right_rotation: i32,
    /// GPU pixel format (NV12 or P010) for D3D11VA staging pool creation.
    pub(crate) gpu_pixel_format: crate::render::renderer::GpuPixelFormat,
    /// Full-range YUV (0-255) vs limited range (16-235).
    pub(crate) is_full_range: bool,
    /// Optional callback invoked with NV12 data after each frame.
    /// Used by reco-cli's snapshot writer for periodic JPEG output.
    /// The callback receives `(nv12_data, width, height)`.
    pub(crate) nv12_tap: Option<Nv12TapFn>,
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
            detector: None,
            detection_interval: 1,
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
        // Build a `StitchCore` as the session's rendering foundation.
        // The executor owns the pipeline + projection + NV12 delivery;
        // the core layers readback + coverage on top. The session
        // layers on async encoding, lookahead, and the per-platform
        // frame dispatch.
        //
        // Rotation is NOT applied here. It's handled by:
        // - CPU path: decoder reverses buffers in extract_yuv()
        // - GPU path: configure_from_source() sets shader UV flip in run()
        // SessionConfig.left_rotation/right_rotation are kept for Layer 1
        // consumers who call set_flip_180() manually.
        let executor = GpuExecutor::new(
            gpu,
            GpuExecutorConfig {
                calibration: config.calibration,
                viewport: config.viewport,
                input_width: config.input_width,
                input_height: config.input_height,
                input_format: config.input_format,
                // `OutputFormat` -> `wgpu::TextureFormat` via the
                // `From` impl in `crate::gpu`; covers all three
                // session-facing variants (Rgba8Unorm, Rgba8UnormSrgb,
                // Bgra8UnormSrgb).
                output_format: config.output_format.into(),
                projection: None,
                full_range: false,
            },
        )
        .map_err(StitchCoreError::from)?;
        let core = StitchCore::new(Executor::Gpu(Box::new(executor)))?;

        Ok(Self {
            core,
            encoder: None,
            skip_detection: false,
            lookahead_frames: 0,
            frame_count: 0,
            extra_encoders: Vec::new(),
            session_start: None,
            error_policy: ErrorPolicy::default(),
            frames_dropped: 0,
            telemetry: crate::telemetry::TelemetryCollector::new(),
            last_readback_time: std::time::Duration::ZERO,
            last_submit_time: std::time::Duration::ZERO,
            current_vram_slot: None,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            left_rotation: 0,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            right_rotation: 0,
            gpu_pixel_format: crate::render::renderer::GpuPixelFormat::Nv12,
            is_full_range: false,
            nv12_tap: None,
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

    /// Number of frames processed so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
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

    /// The engine's GPU executor - the resident-frame surface the
    /// streaming session drives. The batch loop is a GPU-streaming
    /// orchestrator, so its engine always runs the GPU arm.
    pub(crate) fn gpu_exec(&mut self) -> &mut GpuExecutor {
        self.core
            .executor
            .gpu_mut()
            .expect("the streaming session runs on the GPU executor")
    }

    /// Shared-reference sibling of [`Self::gpu_exec`].
    pub(crate) fn gpu_exec_ref(&self) -> &GpuExecutor {
        self.core
            .executor
            .gpu()
            .expect("the streaming session runs on the GPU executor")
    }

    /// Shared reference to the GPU context, for consumers that create
    /// auxiliary resources on the session's device (demosaic kernels,
    /// preview textures).
    pub fn gpu(&self) -> &GpuContext {
        self.gpu_exec_ref().pipeline.gpu()
    }

    /// The name of the GPU this session is running on.
    pub fn gpu_name(&self) -> &str {
        self.gpu().gpu_name()
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
            total_frames: None,
        }
    }

    /// Snapshot of the session's telemetry collector.
    ///
    /// Merges the async encode thread's overlapped encode cost and
    /// backpressure into the snapshot (the collector only sees the
    /// per-frame submit cost).
    pub fn telemetry_snapshot(&self) -> crate::telemetry::TelemetrySnapshot {
        let mut snap = self.telemetry.snapshot();
        if let Some(enc) = &self.encoder {
            let (_frames, avg_encode_ms, bp_stalls, bp_ms) = enc.stats();
            snap.avg_encode_worker_ms = avg_encode_ms;
            snap.backpressure_stalls = bp_stalls;
            snap.backpressure_ms = bp_ms;
        }
        snap
    }

    /// Mutable reference to the telemetry collector.
    pub fn telemetry_mut(&mut self) -> &mut crate::telemetry::TelemetryCollector {
        &mut self.telemetry
    }

    /// Flush the NV12 triple-buffer and finalize the encoder.
    ///
    /// Drains all pending frames from the triple-buffer pipeline and
    /// submits them to the encoder, then shuts down the encode thread
    /// and calls `Encoder::finish`. Must be called after the frame loop ends.
    pub fn finish(&mut self) -> Result<(), SessionError> {
        // Flush remaining frames from the NV12 triple-buffer. Field-path
        // borrow: `nv12_data` borrows the executor inside `core` while
        // the loop body feeds the session-owned encoders.
        while let Some(nv12_data) = self
            .core
            .executor
            .gpu_mut()
            .expect("the streaming session runs on the GPU executor")
            .flush_nv12()?
        {
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
}

impl crate::detect::DetectionTarget for StitchSession {
    fn set_detector(&mut self, detector: Box<dyn crate::detect::detector::UnifiedDetector>) {
        self.set_detector(detector);
    }
    fn set_detection_interval(&mut self, interval: u64) {
        self.set_detection_interval(interval);
    }
    fn set_ball_tracker(&mut self, tracker: Box<dyn crate::detect::tracker::Tracker>) {
        self.set_ball_tracker(tracker);
    }
    fn set_player_tracker(&mut self, tracker: Box<dyn crate::detect::tracker::Tracker>) {
        self.set_player_tracker(tracker);
    }
    fn set_panner(&mut self, panner: Box<dyn crate::detect::panner::Panner>) {
        self.set_panner(panner);
    }
    fn source_info(&self) -> (u32, u32) {
        self.core.source_info()
    }
    fn gpu(&self) -> Option<&crate::gpu::GpuContext> {
        Some(self.gpu())
    }
}
