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

/// Detection pipeline - also usable standalone without StitchSession.
pub mod detection;
/// Detection dispatch entry points (detect_and_update_director_* variants).
mod detection_dispatch;
/// Per-frame render and encode methods (step, process_frame, submit_render_output).
mod frame_processing;
/// Batch processing entry points (run, run_immediate, setup_gpu_source).
mod run_loop;
/// Configuration wiring (set/clear/attach methods).
mod wiring;

#[cfg(test)]
mod tests;
#[cfg(target_os = "linux")]
mod zero_copy_linux;

#[cfg(target_os = "linux")]
pub use zero_copy_linux::SharedTextureSet;

// `LiveStitchSession` + `LiveSessionConfig` + `LiveSessionError` were
// deleted 2026-04-19 (plan-execution §3 M3 step 3). Consumers that
// previously held a `LiveStitchSession` migrate to `StitchCore` (via
// `reco_core::core::StitchCore`) and call `submit_frame_*_at_pose`
// for explicit-pose inputs. reco-obs completed the migration in the
// same commit.

use crate::async_encode::AsyncEncodeThread;
use crate::core::StitchCore;
use crate::core::types::StitchCoreConfig;
use crate::detect::director::ViewportPosition;
use crate::gpu::nv12_converter::Nv12Converter;
use crate::gpu::{GpuContext, OutputFormat};
use crate::render::pipeline::StitchPipeline;
use crate::render::renderer::InputFormat;

use types::{ErrorPolicy, SessionConfig, SessionError, SessionMetrics, StitchSessionBuilder};

use detection::DetectionPipeline;

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
    pub(crate) extra_encoders: Vec<AsyncEncodeThread>,
    /// Detection backends, interval, callback, and cached detections.
    pub(crate) detection: DetectionPipeline,
    /// Tracker/panner pose resolution. When `panner` is set, it owns
    /// pose resolution each frame; when unset the pose stays at the
    /// pipeline default. Trackers are wired here rather than inside
    /// the panner so multiple panners can share the same tracker
    /// output (e.g. replay + live from the same WorldState).
    pub(crate) ball_tracker: Option<Box<dyn crate::detect::tracker::Tracker>>,
    pub(crate) player_tracker: Option<Box<dyn crate::detect::tracker::Tracker>>,
    pub(crate) panner: Option<Box<dyn crate::detect::panner::Panner>>,
    /// Previous frame's resolved pose (post-clamping), handed to the
    /// panner via [`PanContext::previous_position`](crate::detect::panner::PanContext::previous_position).
    pub(crate) previous_panner_pose: ViewportPosition,
    /// Future WorldStates from the lookahead buffer, passed to the
    /// panner via `decide_with_lookahead`. Empty when lookahead is off.
    pub(crate) lookahead_world_states: Vec<crate::detect::tracker::WorldState>,
    pub(crate) frame_count: u64,
    /// Session start time for metrics computation.
    session_start: Option<std::time::Instant>,
    /// Error policy for the run() batch loop.
    pub(crate) error_policy: ErrorPolicy,
    /// Dropped frame counter (for metrics).
    frames_dropped: u64,
    pub(crate) event_sink: Option<Box<dyn crate::detect::pipeline_event::PipelineEventSink>>,
    pub(crate) telemetry: crate::telemetry::TelemetryCollector,
    /// Sub-timing from the last `submit_render_output` call.
    /// Used by `process_frame_any` to split "stitch" into
    /// render / readback / encode for accurate telemetry.
    pub(crate) last_readback_time: std::time::Duration,
    pub(crate) last_encode_time: std::time::Duration,
    /// Ordered pre-tracker detection filters. Empty by default; each
    /// stage transforms `detection.last_detections` in place before
    /// the trackers run. Emission of the before/after event is gated
    /// on `event_sink`.
    pub(crate) detection_filters: Vec<Box<dyn crate::detect::filter::DetectionFilter>>,
    // ── GPU-resident source state (populated by configure_from_source) ──
    /// Bind groups for GPU-resident shared textures.
    /// Created lazily from the source's textures at the start of run().
    #[cfg(target_os = "linux")]
    pub(crate) gpu_bind_groups: Option<crate::render::pipeline::GpuSourceBindGroups>,
    /// Slot-free senders for decode backpressure (GPU zero-copy).
    #[cfg(target_os = "linux")]
    pub(crate) gpu_slot_free_tx: Option<(
        std::sync::mpsc::SyncSender<u8>,
        std::sync::mpsc::SyncSender<u8>,
    )>,
    /// CUDA buffer info for GPU detection (GPU zero-copy).
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) gpu_buf_info: Option<(
        crate::interop::zero_copy::GpuBufInfo,
        crate::interop::zero_copy::GpuBufInfo,
    )>,
    /// Texture views for the 8 shared zero-copy textures, layout
    /// `[left_y_0, left_uv_0, left_y_1, left_uv_1, right_y_0,
    /// right_uv_0, right_y_1, right_uv_1]`. Stashed at
    /// `setup_gpu_source` time so `step_gpu_with_bufs` can hand
    /// slot-indexed views to the GPU stacked-replay pack without
    /// rebuilding views every frame. TextureView holds an Arc on
    /// the underlying texture so the shared-memory lifetime is
    /// still bound to the SharedTextureSet the source owns.
    #[cfg(target_os = "linux")]
    pub(crate) gpu_shared_views: Option<[wgpu::TextureView; 8]>,

    /// Metal texture cache for importing CVPixelBuffers as wgpu textures.
    /// Created lazily on the first MetalResident frame.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) metal_texture_cache: Option<crate::interop::metal::MetalTextureCache>,

    /// D3D11VA staging pool for zero-copy decode on Windows.
    /// Created lazily when the first D3d11Resident frame arrives.
    #[cfg(target_os = "windows")]
    pub(crate) d3d11_staging_pool: Option<crate::interop::d3d11::D3d11StagingPool>,

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
            detection: DetectionPipeline::new(),
            ball_tracker: None,
            player_tracker: None,
            panner: None,
            previous_panner_pose: ViewportPosition::default(),
            lookahead_world_states: Vec::new(),
            frame_count: 0,
            extra_encoders: Vec::new(),
            session_start: None,
            error_policy: ErrorPolicy::default(),
            frames_dropped: 0,
            event_sink: None,
            telemetry: crate::telemetry::TelemetryCollector::new(),
            last_readback_time: std::time::Duration::ZERO,
            last_encode_time: std::time::Duration::ZERO,
            detection_filters: Vec::new(),
            #[cfg(target_os = "linux")]
            gpu_bind_groups: None,
            #[cfg(target_os = "linux")]
            gpu_slot_free_tx: None,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            gpu_buf_info: None,
            #[cfg(target_os = "linux")]
            gpu_shared_views: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal_texture_cache: None,
            #[cfg(target_os = "windows")]
            d3d11_staging_pool: None,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            left_rotation: 0,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            right_rotation: 0,
            gpu_pixel_format: crate::render::renderer::GpuPixelFormat::Nv12,
            is_full_range: false,
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
            total_frames: None,
        }
    }

    /// Snapshot of the session's telemetry collector.
    pub fn telemetry_snapshot(&self) -> crate::telemetry::TelemetrySnapshot {
        self.telemetry.snapshot()
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
    fn pipeline(&self) -> &crate::render::pipeline::StitchPipeline {
        self.pipeline()
    }
    fn gpu(&self) -> &crate::gpu::GpuContext {
        self.gpu()
    }
}
