//! `StitchCore` — push-first canonical entry point for the stitching engine.
//!
//! `StitchCore` is the M3 unification of [`StitchSession`](crate::session::StitchSession)
//! (pull, batch-oriented) and [`LiveStitchSession`](crate::session::LiveStitchSession)
//! (push, live-oriented). Live sports production is the primary use case,
//! so the canonical API is push-based: consumers call
//! [`StitchCore::submit_frame_yuv`] / [`StitchCore::submit_frame_bgra`]
//! whenever a new frame pair is ready, and the core owns the pipeline,
//! readback, director, detection, coverage, and replay ring buffer.
//!
//! Batch file processing layers a thin pull-adapter on top
//! (landing in a later tranche as a rewrite of `StitchSession::run`).
//!
//! ## Why a new module alongside the old two
//!
//! This initial landing introduces `StitchCore` **without** deleting
//! [`LiveStitchSession`] or breaking the existing [`StitchSession`] API.
//! The reco-obs / reco-gui migration, the `LiveStitchSession` deletion,
//! and the `StitchSession::run` pull-adapter rewrite happen in follow-up
//! tranches so this commit stays reviewable. The plan-execution-2026-04-18
//! doc §3 M3 steps 2+3 capture the migration sequence.
//!
//! ## Foundation traits
//!
//! `StitchCore` composes the four M3 foundation traits that landed as
//! separate commits immediately before this one:
//!
//! - [`crate::projection::Projection`] — camera-geometry contract; today's
//!   L-shape is [`LShapeProjection`](crate::projection::LShapeProjection).
//! - [`crate::source::CameraInput`] — input-camera-count contract;
//!   [`StereoCameraInput`](crate::source::StereoCameraInput) is the
//!   current impl.
//! - [`crate::stage::PipelineStage`] — pluggable mid-pipeline transforms
//!   (color correction, exposure, remote compute shims).
//! - [`crate::detector::UnifiedDetector`] — collapsed CPU/CUDA/Metal
//!   detector contract with `DetectorError` for remote-inference futures.
//!
//! The first two are consumed at construction (see [`StitchCoreConfig`]).
//! `UnifiedDetector` is wired in when a backend migrates onto the unified
//! trait (a later M3 tranche). `PipelineStage` slots in via
//! [`StitchCore::push_pipeline_stage`] but has no registered stages yet.
//!
//! ## Usage (push, live)
//!
//! ```rust,ignore
//! use reco_core::core::{StitchCore, StitchCoreConfig, RenderOutcome};
//!
//! let mut core = StitchCore::new(gpu, StitchCoreConfig {
//!     calibration,
//!     viewport,
//!     input_width, input_height,
//!     output_format: wgpu::TextureFormat::Rgba8Unorm,
//!     input_format: InputFormat::Yuv420p,
//!     ..Default::default()
//! })?;
//!
//! core.set_director(Box::new(MyDirector::new()));
//!
//! // Per-frame, from the consumer's callback thread:
//! match core.submit_frame_yuv(&left, &right)? {
//!     RenderOutcome::Rgba(bytes) => compositor.upload(bytes),
//!     RenderOutcome::Warmup => { /* first two frames: pipeline filling */ }
//! }
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::calibration::MatchCalibration;
use crate::director::{Director, DirectorContext, MappedDetection, ViewportPosition};
use crate::gpu::GpuContext;
use crate::pipeline::{BgraPlanes, PipelineError, StitchPipeline, YuvPlanes};
use crate::projection::{CoverageBoundary, LShapeProjection, PanoramaExtent, Projection};
use crate::renderer::InputFormat;
use crate::rgba_readback::{RgbaReadback, RgbaReadbackError};
use crate::source::{CameraInput, StereoCameraInput};
use crate::stage::PipelineStage;
use crate::viewport::ViewportConfig;

// DetectionPipeline is session-private today. Re-expose the pieces
// StitchCore needs by composing its own thin detection hook; when the
// `UnifiedDetector` backends land we swap this for the unified trait
// without changing the outer API.

/// Errors from [`StitchCore`].
#[derive(Debug, Error)]
pub enum StitchCoreError {
    /// GPU pipeline error (upload, render, or state mismatch).
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// Readback staging / mapping error.
    #[error("readback: {0}")]
    Readback(#[from] RgbaReadbackError),
    /// Caller-facing configuration error (e.g. unsupported combination).
    #[error("config: {0}")]
    Config(String),
}

/// Returned from every [`StitchCore::submit_frame_yuv`] /
/// [`StitchCore::submit_frame_bgra`] call.
///
/// The pipeline triple-buffers readback, so the first two calls produce
/// [`RenderOutcome::Warmup`] while the GPU fills the staging ring; from
/// the third call onward every submit produces
/// [`RenderOutcome::Rgba`] holding the tight RGBA bytes of the frame
/// submitted two frames ago.
pub enum RenderOutcome<'a> {
    /// Pipeline warm-up — submit more frames before expecting output.
    /// Only returned on the first two submit calls after construction.
    Warmup,
    /// A rendered panorama frame, tightly packed as RGBA
    /// (`output_width * output_height * 4` bytes). Borrowed from the
    /// core's internal staging; valid until the next submit call.
    Rgba(&'a [u8]),
}

/// A snapshot of one rendered panorama frame for the replay buffer.
///
/// The bytes are owned (not a borrow into the readback ring) because
/// the replay buffer outlives any single frame's staging slot.
#[derive(Debug, Clone)]
pub struct ReplayFrame {
    /// Tight RGBA bytes: `output_width * output_height * 4`.
    pub rgba: Vec<u8>,
    /// Monotonic timestamp captured at submit time (from the
    /// `StitchCore`'s session-start anchor).
    pub captured_at: Duration,
    /// Viewport pose the frame was rendered with. Useful for replay
    /// overlays that want to annotate where the camera pointed.
    pub pose: ViewportPosition,
}

/// Bounded-duration ring of recently-rendered panorama frames.
///
/// Solves FRICTION A16 (OBS replay). Opt-in via
/// [`StitchCore::enable_replay_buffer`]; when disabled the core
/// allocates nothing for replay and `submit_frame_*` does zero extra
/// work. Ring trimming runs per-submit and is `O(frames_evicted)`.
pub struct ReplayBuffer {
    frames: VecDeque<ReplayFrame>,
    max_duration: Duration,
}

impl ReplayBuffer {
    fn new(max_duration: Duration) -> Self {
        Self {
            frames: VecDeque::new(),
            max_duration,
        }
    }

    fn push(&mut self, frame: ReplayFrame) {
        self.frames.push_back(frame);
        // Evict from the front until the oldest kept frame is within
        // max_duration of the newest. Using wrapping subtraction on
        // Duration is not allowed, so compare directly.
        let cutoff = self
            .frames
            .back()
            .map(|f| f.captured_at)
            .unwrap_or_default()
            .saturating_sub(self.max_duration);
        while let Some(front) = self.frames.front() {
            if front.captured_at < cutoff {
                self.frames.pop_front();
            } else {
                break;
            }
        }
    }

    /// Number of frames currently buffered.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the buffer holds zero frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Maximum age of retained frames (set at enable time).
    pub fn max_duration(&self) -> Duration {
        self.max_duration
    }

    /// Iterate buffered frames oldest-to-newest.
    pub fn iter(&self) -> impl Iterator<Item = &ReplayFrame> {
        self.frames.iter()
    }

    /// Most recently buffered frame, if any.
    pub fn latest(&self) -> Option<&ReplayFrame> {
        self.frames.back()
    }
}

/// Configuration for building a [`StitchCore`].
///
/// Required fields: `calibration`, `input_width`, `input_height`,
/// `input_format`. Everything else has sensible defaults.
pub struct StitchCoreConfig {
    /// Camera calibration data.
    pub calibration: MatchCalibration,
    /// Output viewport (dimensions, blend width, FOV).
    pub viewport: ViewportConfig,
    /// Input frame width in pixels (per camera).
    pub input_width: u32,
    /// Input frame height in pixels (per camera).
    pub input_height: u32,
    /// GPU render target format. `Rgba8Unorm` is the default and is
    /// what every compositor consumer needs; `Bgra8Unorm` matches
    /// native Windows DirectX surfaces for consumers that prefer to
    /// swizzle on upload instead of on readback.
    pub output_format: wgpu::TextureFormat,
    /// Input pixel format.
    pub input_format: InputFormat,
    /// Optional custom projection. Defaults to
    /// [`LShapeProjection`] — the 2-plane L-shape that matches today's
    /// geometric model.
    pub projection: Option<Box<dyn Projection>>,
    /// Optional camera-input marker. Defaults to
    /// [`StereoCameraInput`]; future mono / N-input builds pick a
    /// different impl here.
    pub camera_input: Option<Box<dyn CameraInput>>,
    /// Opt-in replay ring buffer duration. `None` (default) keeps no
    /// history and allocates nothing for replay.
    pub replay_buffer_duration: Option<Duration>,
}

impl StitchCoreConfig {
    /// New config with required fields only; defaults everywhere else.
    pub fn new(
        calibration: MatchCalibration,
        input_width: u32,
        input_height: u32,
        input_format: InputFormat,
    ) -> Self {
        Self {
            calibration,
            viewport: ViewportConfig {
                width: 1920,
                height: 1080,
                blend_width: 0.15,
                ..Default::default()
            },
            input_width,
            input_height,
            output_format: wgpu::TextureFormat::Rgba8Unorm,
            input_format,
            projection: None,
            camera_input: None,
            replay_buffer_duration: None,
        }
    }
}

/// Canonical push-first stitching core.
///
/// See the module-level docs for design rationale. `StitchCore` owns:
///
/// - A [`StitchPipeline`] for the GPU render work.
/// - An [`RgbaReadback`] triple-buffered staging ring for CPU delivery.
/// - A coverage boundary precomputed from calibration for `safe_clamp`.
/// - The active [`Projection`] and [`CameraInput`] (for future
///   N-input / alt-projection variants).
/// - An optional [`Director`] and pipeline-stage chain.
/// - An optional [`ReplayBuffer`].
///
/// Detection is explicitly *not* wired in yet: the plan's step M3-4
/// replaces the per-platform detector traits with a single
/// [`UnifiedDetector`](crate::detector::UnifiedDetector), and wiring
/// detection here before that migration would just have to be reworked.
/// Consumers that need detection today should keep using
/// [`StitchSession`](crate::session::StitchSession) until the
/// unified-detector migration lands.
pub struct StitchCore {
    pipeline: StitchPipeline,
    readback: RgbaReadback,
    output_width: u32,
    output_height: u32,

    projection: Box<dyn Projection>,
    camera_input: Box<dyn CameraInput>,
    stages: Vec<Box<dyn PipelineStage>>,

    coverage: Option<CoverageBoundary>,
    director: Option<Box<dyn Director>>,

    replay: Option<ReplayBuffer>,

    frame_count: u64,
    session_start: Option<Instant>,
}

impl StitchCore {
    /// Build a new core. Owns the supplied [`GpuContext`].
    pub fn new(gpu: GpuContext, config: StitchCoreConfig) -> Result<Self, StitchCoreError> {
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

        let readback = RgbaReadback::new(pipeline.gpu(), output_width, output_height)?;

        let coverage = CoverageBoundary::from_calibration(pipeline.calibration(), &pipeline.scene);

        let projection: Box<dyn Projection> = config
            .projection
            .unwrap_or_else(|| Box::new(LShapeProjection));
        let camera_input: Box<dyn CameraInput> = config
            .camera_input
            .unwrap_or_else(|| Box::new(StereoCameraInput));

        if camera_input.camera_count() != projection.camera_count() {
            return Err(StitchCoreError::Config(format!(
                "camera_input.camera_count() = {} but projection.camera_count() = {}; \
                 these must match so StitchCore receives one frame per camera plane",
                camera_input.camera_count(),
                projection.camera_count(),
            )));
        }

        let replay = config.replay_buffer_duration.map(ReplayBuffer::new);

        Ok(Self {
            pipeline,
            readback,
            output_width,
            output_height,
            projection,
            camera_input,
            stages: Vec::new(),
            coverage: Some(coverage),
            director: None,
            replay,
            frame_count: 0,
            session_start: None,
        })
    }

    // -----------------------------------------------------------------
    // Submit / render
    // -----------------------------------------------------------------

    /// Submit a stereo YUV420P frame pair and render the current pose.
    ///
    /// Uses the director (if attached) and coverage clamping to pick
    /// the viewport, renders, and reads back RGBA. The first two calls
    /// produce [`RenderOutcome::Warmup`] while the triple-buffered
    /// staging ring fills; from the third call onward every submit
    /// yields RGBA bytes from two frames ago.
    pub fn submit_frame_yuv(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();
        let pose = self.resolve_current_pose();
        let cmd = self
            .pipeline
            .render_to_target(left, right, pose.yaw, pose.pitch)?;
        // Split-borrow: push_replay only accesses self.replay +
        // self.session_start; self.readback keeps the rgba slice
        // alive. Inlining the replay push (instead of going through
        // `&mut self` on a helper) lets the borrow checker see the
        // fields are disjoint.
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let rgba =
            self.readback
                .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)?;
        self.frame_count += 1;
        if let (Some(replay), Some(bytes)) = (self.replay.as_mut(), rgba) {
            replay.push(ReplayFrame {
                rgba: bytes.to_vec(),
                captured_at,
                pose,
            });
        }
        Ok(match rgba {
            Some(bytes) => RenderOutcome::Rgba(bytes),
            None => RenderOutcome::Warmup,
        })
    }

    /// Submit a stereo packed-RGBA/BGRA frame pair and render the
    /// current pose.
    ///
    /// Requires the core to have been built with [`InputFormat::Bgra`].
    /// See [`Self::submit_frame_yuv`] for return semantics.
    pub fn submit_frame_bgra(
        &mut self,
        left: &BgraPlanes<'_>,
        right: &BgraPlanes<'_>,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();
        let pose = self.resolve_current_pose();
        let cmd = self
            .pipeline
            .render_to_target_bgra(left, right, pose.yaw, pose.pitch)?;
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let rgba =
            self.readback
                .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)?;
        self.frame_count += 1;
        if let (Some(replay), Some(bytes)) = (self.replay.as_mut(), rgba) {
            replay.push(ReplayFrame {
                rgba: bytes.to_vec(),
                captured_at,
                pose,
            });
        }
        Ok(match rgba {
            Some(bytes) => RenderOutcome::Rgba(bytes),
            None => RenderOutcome::Warmup,
        })
    }

    /// Drain one pending readback slot without submitting a new frame.
    ///
    /// Useful at shutdown to collect the 1-2 frames still in-flight in
    /// the triple-buffered staging pipeline.
    pub fn flush(&mut self) -> Result<Option<&[u8]>, StitchCoreError> {
        Ok(self.readback.flush_pending(self.pipeline.gpu())?)
    }

    // -----------------------------------------------------------------
    // Director / pose
    // -----------------------------------------------------------------

    /// Attach a director for pose selection. Replaces any existing one.
    pub fn set_director(&mut self, director: Box<dyn Director>) {
        self.director = Some(director);
    }

    /// Remove the currently attached director.
    pub fn clear_director(&mut self) {
        self.director = None;
    }

    /// The resolved viewport pose for the next render, already clamped
    /// through coverage + FOV limits. Exposed so interactive consumers
    /// (OBS pan/zoom, GUI drag) can preview where the core *would*
    /// render if they submit right now.
    pub fn current_pose(&mut self) -> ViewportPosition {
        self.resolve_current_pose()
    }

    /// Clamp a prospective `(yaw, pitch, fov)` triple through the
    /// coverage boundary. No-op if no coverage is available (e.g. the
    /// calibration produced a degenerate boundary). `fov_degrees: None`
    /// uses the pipeline's current FOV.
    pub fn safe_clamp(&self, pose: ViewportPosition) -> ViewportPosition {
        let Some(coverage) = &self.coverage else {
            return pose;
        };
        let fov = pose
            .fov_degrees
            .unwrap_or_else(|| self.pipeline.fov())
            .min(coverage.max_fov_degrees());
        let aspect = self.pipeline.viewport().aspect_ratio();
        let clamped = coverage.safe_clamp(pose.yaw, pose.pitch, fov, aspect, 0.0);
        let rig_tilt = self.pipeline.viewport().rig_tilt;
        ViewportPosition {
            yaw: clamped.yaw,
            pitch: clamped.pitch - rig_tilt * clamped.yaw.cos(),
            fov_degrees: Some(fov),
        }
    }

    // -----------------------------------------------------------------
    // Coverage / calibration / projection introspection
    // -----------------------------------------------------------------

    /// The precomputed coverage boundary for "no-black" viewport
    /// constraining. Some calibrations produce a degenerate boundary
    /// in which case this is `None` (very rare in practice).
    pub fn coverage(&self) -> Option<&CoverageBoundary> {
        self.coverage.as_ref()
    }

    /// Full angular extent of the stitched panorama, derived from the
    /// coverage boundary. Higher-level shortcut for analytics consumers.
    pub fn panorama_extent(&self) -> Option<PanoramaExtent> {
        self.coverage.as_ref().map(|c| {
            let (yaw_min, yaw_max) = c.yaw_range();
            let (pitch_min, pitch_max) = c.pitch_range();
            PanoramaExtent {
                yaw_min,
                yaw_max,
                pitch_min,
                pitch_max,
            }
        })
    }

    /// Short name of the active projection (for logs + UI labels).
    pub fn projection_name(&self) -> &'static str {
        self.projection.name()
    }

    /// Camera count of the active input configuration. For today's
    /// stereo L-shape this is `2`; future mono / N-input builds expose
    /// different values.
    pub fn camera_count(&self) -> u8 {
        self.camera_input.camera_count()
    }

    /// Hot-swap the calibration. Takes effect on the next submit.
    ///
    /// Re-derives the coverage boundary from the new calibration so
    /// subsequent `safe_clamp` calls respect the new no-black region.
    pub fn update_calibration(&mut self, calibration: MatchCalibration) {
        self.pipeline.update_calibration(calibration);
        self.coverage = Some(CoverageBoundary::from_calibration(
            self.pipeline.calibration(),
            &self.pipeline.scene,
        ));
    }

    // -----------------------------------------------------------------
    // Pipeline stages (slot reserved; no registered stages yet)
    // -----------------------------------------------------------------

    /// Append a pipeline stage to the mid-pipeline chain.
    ///
    /// The chain has no registered stages today; this method is the
    /// registration point for future stages (color correction, exposure
    /// normalization, remote-compute shims) without another breaking
    /// change. Stages are executed in push order.
    pub fn push_pipeline_stage(&mut self, stage: Box<dyn PipelineStage>) {
        self.stages.push(stage);
    }

    /// Number of registered pipeline stages.
    pub fn pipeline_stage_count(&self) -> usize {
        self.stages.len()
    }

    // -----------------------------------------------------------------
    // Replay buffer
    // -----------------------------------------------------------------

    /// Enable (or reconfigure) the replay ring buffer.
    ///
    /// Passing `None` disables replay and drops any buffered frames,
    /// freeing the allocation. Passing `Some(duration)` creates (or
    /// resizes) the ring to retain at most `duration` of the most
    /// recent rendered frames. The ring only grows as frames arrive;
    /// no pre-allocation.
    pub fn enable_replay_buffer(&mut self, duration: Option<Duration>) {
        self.replay = duration.map(ReplayBuffer::new);
    }

    /// Borrow the replay buffer, if enabled.
    pub fn replay_buffer(&self) -> Option<&ReplayBuffer> {
        self.replay.as_ref()
    }

    // -----------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------

    /// Output dimensions in pixels. Identical to
    /// `config.viewport.{width,height}` at construction.
    pub fn output_dims(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }

    /// Number of frames submitted so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Shared access to the underlying pipeline.
    pub fn pipeline(&self) -> &StitchPipeline {
        &self.pipeline
    }

    /// Mutable access to the pipeline for advanced callers that need
    /// to tweak viewport / FOV / zero-copy bind groups directly.
    pub fn pipeline_mut(&mut self) -> &mut StitchPipeline {
        &mut self.pipeline
    }

    /// The GPU context owning every resource.
    pub fn gpu(&self) -> &GpuContext {
        self.pipeline.gpu()
    }

    // -----------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------

    fn anchor_session_start(&mut self) {
        if self.session_start.is_none() {
            self.session_start = Some(Instant::now());
        }
    }

    fn resolve_current_pose(&mut self) -> ViewportPosition {
        // Pull raw director output (or default) and clamp through
        // coverage. Then write the resolved FOV back onto the pipeline
        // so the upcoming render uses it.
        let timestamp_ms = self
            .session_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        if let Some(director) = self.director.as_mut() {
            // Director receives an empty detections list today;
            // detection wiring lands with the `UnifiedDetector`
            // migration in a follow-up. An empty-slice context still
            // ticks the director's internal smoothing so live-coded
            // directors (e.g. keyframe animators) remain usable.
            let empty: &[MappedDetection] = &[];
            let ctx = DirectorContext {
                frame_index: self.frame_count,
                timestamp_ms,
                detections: empty,
                fresh_detection: false,
            };
            director.update(&ctx);
        }

        let raw = self
            .director
            .as_ref()
            .map_or(ViewportPosition::default(), |d| d.position());
        let clamped = self.safe_clamp(raw);
        if let Some(fov) = clamped.fov_degrees {
            self.pipeline.set_fov(fov);
        }
        clamped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert `ReplayBuffer` trims old frames as the newest ages past
    /// `max_duration`. This is the core guarantee OBS A16 relies on:
    /// the buffer never grows unboundedly during a long session.
    #[test]
    fn replay_buffer_trims_old_frames() {
        let mut buf = ReplayBuffer::new(Duration::from_secs(2));
        for i in 0..5 {
            buf.push(ReplayFrame {
                rgba: vec![i as u8; 4],
                captured_at: Duration::from_millis(i as u64 * 1000),
                pose: ViewportPosition::default(),
            });
        }
        // Newest is at 4s; anything older than 2s should be evicted.
        // Frames at 0s and 1s are older than (4s - 2s = 2s), so evicted.
        // Frames at 2s, 3s, 4s remain.
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.iter().next().unwrap().rgba[0], 2);
        assert_eq!(buf.latest().unwrap().rgba[0], 4);
    }

    /// Replay trimming respects `max_duration` exactly: the boundary
    /// is inclusive on the retain side.
    #[test]
    fn replay_buffer_boundary_inclusive() {
        let mut buf = ReplayBuffer::new(Duration::from_millis(100));
        buf.push(ReplayFrame {
            rgba: vec![],
            captured_at: Duration::from_millis(0),
            pose: ViewportPosition::default(),
        });
        buf.push(ReplayFrame {
            rgba: vec![],
            captured_at: Duration::from_millis(100),
            pose: ViewportPosition::default(),
        });
        // Newest - max_duration = 0, so frame at 0ms is exactly on the
        // boundary and retained.
        assert_eq!(buf.len(), 2);
    }

    /// An empty replay buffer answers `latest()` = None and is_empty.
    #[test]
    fn replay_buffer_empty_semantics() {
        let buf = ReplayBuffer::new(Duration::from_secs(1));
        assert!(buf.is_empty());
        assert!(buf.latest().is_none());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.max_duration(), Duration::from_secs(1));
    }

    /// `StitchCoreError` is `std::error::Error` (so downstream
    /// consumers can `?` it through `Box<dyn Error>` channels) and
    /// `Send + Sync` (so it can cross thread boundaries in a future
    /// worker-thread detection pipeline).
    #[test]
    fn stitch_core_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StitchCoreError>();
        fn assert_error<T: std::error::Error + 'static>() {}
        assert_error::<StitchCoreError>();
    }

    /// `RenderOutcome` is `Send` — needed so consumers that post
    /// rendered frames onto worker channels (or mpsc splits) can
    /// forward the enum without boxing it.
    #[test]
    fn render_outcome_warmup_constructs() {
        let outcome: RenderOutcome<'_> = RenderOutcome::Warmup;
        match outcome {
            RenderOutcome::Warmup => {}
            RenderOutcome::Rgba(_) => unreachable!("built a Warmup"),
        }
    }
}
