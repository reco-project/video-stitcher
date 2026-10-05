//! High-level stitching session.
//!
//! [`StitchSession`] is the pull-loop orchestrator over the engine:
//! it renders stitched panoramic frames and fans them out to the
//! attached [`OutputSink`](crate::sink::OutputSink)s (encoders,
//! snapshots, streams). This keeps delivery orchestration inside
//! `reco-core` so that every consumer (CLI, GUI, OBS plugin, cloud
//! worker) gets the same optimized frame loop without duplicating
//! pipeline plumbing.
//!
//! ## Two-level API
//!
//! - [`StitchSession::render_frame`] - render one frame and deliver
//!   it to the sinks. Use this for interactive/GUI applications or
//!   when the caller controls the frame loop (e.g. zero-copy GPU
//!   decode).
//!
//! - [`StitchSession::run`] - batch-process an entire `FrameSource`
//!   into the sinks, with optional progress reporting and interrupt
//!   support. Use this for CLI batch encoding.

/// Sink attachment options and delivery fan-out.
pub mod sinks;
/// Session type definitions, error types, and builder.
pub mod types;

/// Detection dispatch entry points (detect_and_update_world_pose_* variants).
mod detection_dispatch;
/// Lookahead frame buffer for temporal-aware processing.
pub(crate) mod frame_buffer;
/// Per-frame render and encode methods (step, render_frame, submit_render_output).
mod frame_processing;
/// Batch processing entry points (run, run_immediate, setup_gpu_source).
mod run_loop;
/// Configuration wiring (set/clear/attach methods).
mod wiring;

#[cfg(test)]
mod tests;

pub use sinks::{SinkDelivery, SinkErrorPolicy, SinkOptions};

use crate::core::StitchCore;
#[cfg(feature = "gpu")]
use crate::core::types::StitchCoreError;
#[cfg(feature = "gpu")]
use crate::gpu::{GpuContext, OutputFormat};
#[cfg(feature = "gpu")]
use crate::render::renderer::InputFormat;
use crate::stitch::Executor;
#[cfg(feature = "gpu")]
use crate::stitch::{GpuExecutor, GpuExecutorConfig};

use types::SessionError;
#[cfg(feature = "gpu")]
use types::{SessionConfig, StitchSessionBuilder};

/// A high-level stitching session: a pull-loop orchestrator over the
/// engine, adding frame buffering, sink fan-out, and telemetry.
///
/// Created once per encoding job or application lifetime. Call
/// [`add_sink`](Self::add_sink) to attach output sinks before
/// rendering, then use [`render_frame`](Self::render_frame)
/// for per-frame control or [`run`](Self::run) for batch processing.
/// Call [`finish`](Self::finish) to flush the last frame and finalize
/// every sink.
pub struct StitchSession {
    /// The canonical push-first engine. Owns the render substrate,
    /// readback staging, coverage boundary, and the single AI stack
    /// (detector, trackers, panner, event sink). The session is the
    /// pull-loop orchestrator over it: frame buffering, sink
    /// fan-out, telemetry, progress.
    pub(crate) core: StitchCore,
    /// Attached output sinks; every rendered frame fans out to all of
    /// them in attach order.
    pub(crate) sinks: Vec<sinks::AttachedSink>,
    /// Reused software RGBA-to-NV12 conversion storage, empty on GPU sessions.
    pub(crate) cpu_nv12: Vec<u8>,
    /// Number of lookahead frames (0 = disabled).
    pub(crate) lookahead_frames: usize,
    pub(crate) lookahead_memory_bytes: usize,
    pub(crate) frame_count: u64,
    /// Timestamps in the same submission order as the GPU's readback ring.
    #[cfg(feature = "gpu")]
    pub(crate) pending_pts: std::collections::VecDeque<Option<i64>>,
    pub(crate) telemetry: crate::telemetry::TelemetryCollector,
    telemetry_sink: Option<(Box<dyn crate::telemetry::TelemetrySink>, u64)>,
    /// Final worker average maximum, backpressure count and milliseconds after sink release.
    finished_sink_stats: (f32, u64, f32),
    /// Sub-timing from the last `submit_render_output` call.
    /// Used by `process_frame_any` to split "stitch" into
    /// render / readback / encode for accurate telemetry.
    pub(crate) last_readback_time: std::time::Duration,
    pub(crate) last_submit_time: std::time::Duration,
    /// VRAM pool slot for the frame currently being rendered
    /// (buffered lookahead path; the pool itself lives on the
    /// GPU executor).
    #[cfg(feature = "gpu")]
    pub(crate) current_vram_slot: Option<usize>,

    /// Camera rotation from stream metadata, populated by
    /// [`configure_from_source`](Self::configure_from_source).
    /// Used to tell the GPU detector to flip frames during preprocessing.
    #[cfg(all(feature = "gpu", any(target_os = "linux", target_os = "windows")))]
    pub(crate) left_rotation: i32,
    /// Right camera rotation from stream metadata.
    #[cfg(all(feature = "gpu", any(target_os = "linux", target_os = "windows")))]
    pub(crate) right_rotation: i32,
    /// GPU pixel format (NV12 or P010) for D3D11VA staging pool creation.
    #[cfg(feature = "gpu")]
    pub(crate) gpu_pixel_format: crate::render::renderer::GpuPixelFormat,
}

impl StitchSession {
    /// Create a builder for configuring and constructing a session.
    #[cfg(feature = "gpu")]
    pub fn builder() -> StitchSessionBuilder {
        StitchSessionBuilder {
            calibration: None,
            viewport_size: None,
            source_sizes: None,
            output_format: OutputFormat::Rgba8Unorm,
            input_format: InputFormat::Yuv420p,
            gpu: None,
            sinks: Vec::new(),
            detector: None,
            detection_interval: 1,
        }
    }

    /// Create a new session, initializing the GPU automatically.
    #[cfg(feature = "gpu")]
    pub async fn new(config: SessionConfig) -> Result<Self, SessionError> {
        let gpu = GpuContext::new().await?;
        Self::with_gpu(gpu, config)
    }

    /// Create a session with an existing GPU context.
    ///
    /// Use this when the caller needs to control GPU selection (e.g.
    /// for zero-copy decode where the GPU must match the CUDA device).
    #[cfg(feature = "gpu")]
    pub fn with_gpu(gpu: GpuContext, config: SessionConfig) -> Result<Self, SessionError> {
        // The executor owns the pipeline + projection + NV12 delivery;
        // the core layers readback + coverage on top. The session
        // layers on sink delivery, lookahead, and the per-platform
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
                viewport_size: config.viewport_size,
                source_sizes: config.source_sizes,
                input_format: config.input_format,
                // `OutputFormat` -> `wgpu::TextureFormat` via the
                // `From` impl in `crate::gpu`; covers all three
                // session-facing variants (Rgba8Unorm, Rgba8UnormSrgb,
                // Bgra8UnormSrgb).
                output_format: config.output_format.into(),
                full_range: false,
            },
        )
        .map_err(StitchCoreError::from)?;
        Self::with_executor(Executor::Gpu(Box::new(executor)))
    }

    /// Create a session over an already-configured executor.
    ///
    /// A [`CpuExecutor`](crate::stitch::CpuExecutor) uses software rendering
    /// in the shared frame loop without the `gpu` feature or a GPU context.
    /// GPU constructors and resident-frame entry points require the `gpu`
    /// feature; GPU-only methods panic on a CPU executor.
    ///
    /// ```
    /// use reco_core::{calibration::Calibration, render::viewport::ViewportSize};
    /// use reco_core::{session::StitchSession, stitch::{CpuExecutor, Executor}};
    ///
    /// fn cpu_session(calibration: Calibration) -> Result<StitchSession, Box<dyn std::error::Error>> {
    ///     let source_sizes = vec![(1920, 1080); calibration.projection.camera_count()];
    ///     let executor = CpuExecutor::new(
    ///         calibration, ViewportSize { width: 640, height: 360 }, source_sizes, false,
    ///     )?;
    ///     Ok(StitchSession::with_executor(Executor::Cpu(Box::new(executor)))?)
    /// }
    /// ```
    pub fn with_executor(executor: Executor) -> Result<Self, SessionError> {
        let core = StitchCore::new(executor)?;

        Ok(Self {
            core,
            sinks: Vec::new(),
            cpu_nv12: Vec::new(),
            lookahead_frames: 0,
            lookahead_memory_bytes: 0,
            frame_count: 0,
            #[cfg(feature = "gpu")]
            pending_pts: std::collections::VecDeque::with_capacity(3),
            telemetry: crate::telemetry::TelemetryCollector::new(),
            telemetry_sink: None,
            finished_sink_stats: (0.0, 0, 0.0),
            last_readback_time: std::time::Duration::ZERO,
            last_submit_time: std::time::Duration::ZERO,
            #[cfg(feature = "gpu")]
            current_vram_slot: None,
            #[cfg(all(feature = "gpu", any(target_os = "linux", target_os = "windows")))]
            left_rotation: 0,
            #[cfg(all(feature = "gpu", any(target_os = "linux", target_os = "windows")))]
            right_rotation: 0,
            #[cfg(feature = "gpu")]
            gpu_pixel_format: crate::render::renderer::GpuPixelFormat::Nv12,
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

    /// GPU-only entry points use the executor's resident-frame surface.
    /// The shared batch loop dispatches CPU frames without calling this.
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_exec(&mut self) -> &mut GpuExecutor {
        self.core
            .executor
            .gpu_mut()
            .expect("the streaming session runs on the GPU executor")
    }

    /// Shared-reference sibling of [`Self::gpu_exec`].
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_exec_ref(&self) -> &GpuExecutor {
        self.core
            .executor
            .gpu()
            .expect("the streaming session runs on the GPU executor")
    }

    /// Shared reference to the GPU context, for consumers that create
    /// auxiliary resources on the session's device (demosaic kernels,
    /// preview textures). Panics on a CPU-executor session - use
    /// [`DetectionTarget::gpu`](crate::detect::DetectionTarget::gpu)
    /// for executor-agnostic access.
    #[cfg(feature = "gpu")]
    pub fn gpu(&self) -> &GpuContext {
        self.gpu_exec_ref().pipeline.gpu()
    }

    /// NV12 delivery dimensions for the current output viewport:
    /// the shared rounding rule over the executor-agnostic viewport,
    /// so CPU and GPU sessions hand sinks identically-sized frames.
    pub(crate) fn nv12_delivery_dims(&self) -> (u32, u32) {
        let vp = self.core.executor.viewport_size();
        crate::render::nv12_cpu::nv12_dims(vp.width, vp.height)
    }

    /// The name of the GPU this session is running on.
    #[cfg(feature = "gpu")]
    pub fn gpu_name(&self) -> &str {
        self.gpu().gpu_name()
    }

    /// Snapshot of the session's telemetry collector.
    ///
    /// Merges the sink threads' overlapped consume cost and
    /// backpressure into the snapshot (the collector only sees the
    /// per-frame submit cost). With several threaded sinks the worker
    /// cost is the largest sink average and the
    /// backpressure stalls are summed - the render loop delivers
    /// serially, so every sink's stall delays it.
    pub fn telemetry_snapshot(&self) -> crate::telemetry::TelemetrySnapshot {
        let mut snap = self.telemetry.snapshot();
        snap.avg_encode_worker_ms = self.finished_sink_stats.0;
        snap.backpressure_stalls = self.finished_sink_stats.1;
        snap.backpressure_ms = self.finished_sink_stats.2;
        for stats in self.sinks.iter().filter_map(|s| s.thread_stats()) {
            let (_frames, avg_consume_ms, bp_stalls, bp_ms) = stats;
            snap.avg_encode_worker_ms = snap.avg_encode_worker_ms.max(avg_consume_ms);
            snap.backpressure_stalls += bp_stalls;
            snap.backpressure_ms += bp_ms;
        }
        snap
    }

    /// Mutable reference to the telemetry collector.
    pub fn telemetry_mut(&mut self) -> &mut crate::telemetry::TelemetryCollector {
        &mut self.telemetry
    }

    /// Deliver the same enriched snapshot as polling every `interval_frames` submitted frames and after sink finalization.
    pub fn set_telemetry_sink(
        &mut self,
        sink: Box<dyn crate::telemetry::TelemetrySink>,
        interval_frames: u64,
    ) {
        self.telemetry_sink = Some((sink, interval_frames.max(1)));
    }

    fn publish_telemetry(&mut self, final_snapshot: bool) {
        if self.telemetry_sink.as_ref().is_some_and(|(_, interval)| {
            final_snapshot || self.frame_count.is_multiple_of(*interval)
        }) && let Some((mut sink, interval)) = self.telemetry_sink.take()
        {
            sink.on_snapshot(&self.telemetry_snapshot());
            self.telemetry_sink = Some((sink, interval));
        }
    }

    /// Flush any pending delivery frames and finalize every sink.
    ///
    /// The GPU arm's NV12 delivery is triple-buffered, so its last two
    /// frames are drained here and fanned out to the attached sinks;
    /// the CPU arm delivers synchronously and has nothing pending.
    /// Then calls
    /// [`OutputSink::finish`](crate::sink::OutputSink::finish) on each
    /// sink in attach order. Must be called after the frame loop ends.
    pub fn finish(&mut self) -> Result<(), SessionError> {
        #[cfg(not(feature = "gpu"))]
        let delivery_result: Result<(), SessionError> = Ok(());
        #[cfg(feature = "gpu")]
        let delivery_result = (|| -> Result<(), SessionError> {
            #[cfg(target_os = "linux")]
            if let Some(exec) = self.core.executor.gpu_mut() {
                exec.finish_nvmm_reads().map_err(SessionError::ZeroCopy)?;
            }
            let (nv12_width, nv12_height) = self.nv12_delivery_dims();
            loop {
                let Some(exec) = self.core.executor.gpu_mut() else {
                    break;
                };
                let Some(nv12_data) = exec.flush_nv12()? else {
                    break;
                };
                sinks::deliver_frame(
                    &mut self.sinks,
                    nv12_data,
                    nv12_width,
                    nv12_height,
                    self.pending_pts.pop_front().ok_or_else(|| {
                        SessionError::Config("readback has no originating timestamp".into())
                    })?,
                )?;
            }
            Ok(())
        })();

        let stats = &mut self.finished_sink_stats;
        let result = sinks::finish_all(&mut self.sinks, |(_, avg_ms, stalls, ms)| {
            stats.0 = stats.0.max(avg_ms);
            stats.1 += stalls;
            stats.2 += ms;
        });
        self.publish_telemetry(true);
        // Finalize every attached sink even when pending delivery failed.
        if delivery_result.is_err()
            && let Err(error) = &result
        {
            log::warn!("sink finalization also failed: {error}");
        }
        delivery_result?;
        result?;

        Ok(())
    }
}

impl crate::detect::DetectionTarget for StitchSession {
    fn set_detector(&mut self, detector: Box<dyn crate::detect::detector::Detector>) {
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
    fn source_sizes(&self) -> &[(u32, u32)] {
        self.core.source_sizes()
    }
    #[cfg(feature = "gpu")]
    fn gpu(&self) -> Option<&crate::gpu::GpuContext> {
        self.core.executor.gpu().map(|g| g.pipeline.gpu())
    }
}
