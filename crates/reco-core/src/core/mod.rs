//! `StitchCore` - push-first canonical entry point for the stitching engine.
//!
//! `StitchCore` is the M3 unification of what used to be two parallel
//! session APIs: `StitchSession` (pull, batch-oriented) and the former
//! `LiveStitchSession` (push, live-oriented, since deleted). Live sports
//! production is the primary use case, so the canonical API is push-based:
//! consumers call `StitchCore::submit_frame_yuv` / `submit_frame_bgra`
//! whenever a new frame pair is ready, and the core owns the pipeline,
//! readback, director, detection, coverage, and replay ring buffer.
//!
//! Batch file processing layers a thin pull-adapter on top
//! ([`StitchSession`](crate::session::StitchSession)::run).
//!
//! ## Foundation traits
//!
//! `StitchCore` composes the foundation traits:
//!
//! - [`crate::projection::Projection`] - camera-geometry contract; today's
//!   L-shape is [`LShapeProjection`](crate::projection::LShapeProjection).
//! - [`crate::source::CameraInput`] - input-camera-count contract;
//!   [`StereoCameraInput`](crate::source::StereoCameraInput) is the
//!   current impl.
//! - [`crate::detect::detector::UnifiedDetector`] - collapsed CPU/CUDA/Metal
//!   detector contract with `DetectorError` for remote-inference futures.
//!
//! The first two are consumed at construction (see `StitchCoreConfig`).
//! `UnifiedDetector` is wired via `StitchCore::set_detector`; detection
//! runs on every `submit_frame_*` whose frame count is a multiple of
//! `StitchCore::detection_interval`, and raw detections are mapped to
//! panorama coordinates before reaching the director.
//!
//! ## Sub-modules
//!
//! - `types` - error types, config, render outcome, replay frame, recorder traits
//! - `replay_buffer` - bounded-duration ring buffer for replay frames
//! - `render` - submit and render-at-pose methods
//! - `replay_management` - stacked replay recorder wiring (CPU + GPU paths)
//! - `pose` - pose resolution, detection scheduling, panorama mapping

mod pose;
mod render;
pub mod replay_buffer;
mod replay_management;
pub mod types;

use std::time::{Duration, Instant};

use crate::calibration::MatchCalibration;
use crate::detect::detector::UnifiedDetector;
use crate::detect::director::{MappedDetection, ViewportPosition};
use crate::detect::panner::Panner;
use crate::detect::tracker::Tracker;
use crate::gpu::GpuContext;
use crate::gpu::rgba_readback::RgbaReadback;
use crate::gpu::yuv_stack_packer::YuvStackPacker;
use crate::projection::{CoverageBoundary, LShapeProjection, PanoramaExtent, Projection};
use crate::render::pipeline::StitchPipeline;
use crate::source::{CameraInput, StereoCameraInput};

use self::replay_buffer::ReplayBuffer;
use self::types::{
    StackedReplayGpuRecorder, StackedReplayRecorder, StitchCoreConfig, StitchCoreError,
};

/// Canonical push-first stitching core.
///
/// See the module-level docs for design rationale. `StitchCore` owns:
///
/// - A [`StitchPipeline`] for the GPU render work.
/// - An [`RgbaReadback`] triple-buffered staging ring for CPU delivery.
/// - A coverage boundary precomputed from calibration for `safe_clamp`.
/// - The active [`Projection`] and [`CameraInput`] (for future
///   N-input / alt-projection variants).
/// - Optional [`Tracker`]s and an optional [`Panner`] that together
///   drive the viewport pose, plus a pipeline-stage chain.
/// - An optional [`ReplayBuffer`].
///
/// Detection is wired through the [`UnifiedDetector`] trait: attach
/// one via [`StitchCore::set_detector`] and the core will run it on every
/// CPU-resident frame submitted (CUDA / Metal residency dispatch
/// lands in a later tranche that adds GPU-frame `submit_*` methods).
/// Raw detections are mapped to panorama coordinates and fed to the
/// attached director each submit; directors see a non-empty
/// `detections` slice on detection frames, empty otherwise.
pub struct StitchCore {
    pub(crate) pipeline: StitchPipeline,
    pub(crate) readback: RgbaReadback,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,

    pub(crate) projection: Box<dyn Projection>,
    pub(crate) camera_input: Box<dyn CameraInput>,
    pub(crate) coverage: Option<CoverageBoundary>,

    /// Per-class trackers that feed a shared [`WorldState`](crate::detect::tracker::WorldState)
    /// consumed by [`StitchCore::panner`]. Slot-based on purpose:
    /// `ball_tracker` fills `world.ball`, `player_tracker` fills
    /// `world.players`. More slots land with future entity classes.
    ///
    /// The panner only runs when at least one tracker is registered
    /// AND a panner is set. Otherwise the pose stays at the pipeline
    /// default.
    pub(crate) ball_tracker: Option<Box<dyn Tracker>>,
    pub(crate) player_tracker: Option<Box<dyn Tracker>>,
    /// Camera-motion policy. Consumes the assembled
    /// [`WorldState`](crate::detect::tracker::WorldState) each frame and emits
    /// a [`ViewportPosition`]. When unset, the pose stays at the
    /// pipeline default.
    pub(crate) panner: Option<Box<dyn Panner>>,
    /// Previous frame's resolved pose, passed to the panner in its
    /// [`PanContext`](crate::detect::panner::PanContext) so panners can
    /// compute first-order motion deltas statelessly.
    pub(crate) previous_panner_pose: ViewportPosition,

    pub(crate) detector: Option<Box<dyn UnifiedDetector>>,
    /// How often detection runs. 1 = every frame (default), higher =
    /// skip frames. On skipped frames the director still ticks with
    /// the previously tracked detections.
    pub(crate) detection_interval: u64,
    /// Panorama-mapped detections from the last detection frame.
    /// Reused on skipped frames so the director retains context.
    pub(crate) last_detections: Vec<MappedDetection>,

    pub(crate) replay: Option<ReplayBuffer>,

    /// Optional stacked-video replay recorder attached via
    /// [`Self::set_stacked_recorder`]. Fires on every successful
    /// YUV submit (not BGRA - see [`StackedReplayRecorder`] docs).
    /// Decouples reco-core from the actual encoder implementation
    /// (lives in reco-io under `stacked-output`) so mobile / wasm
    /// builds that skip reco-io see no replay-recording code.
    pub(crate) stacked_recorder: Option<Box<dyn StackedReplayRecorder>>,

    /// Optional GPU-pack packer attached via
    /// [`Self::enable_gpu_stacked_replay`]. Holds the compute
    /// pipelines and triple-buffered staging ring. `None` when the
    /// session runs on a CPU-pack (or no replay) path.
    pub(crate) stacked_packer: Option<YuvStackPacker>,

    /// Optional GPU-pack atlas recorder attached via
    /// [`Self::set_stacked_gpu_recorder`]. Receives the packed atlas
    /// bytes every time [`YuvStackPacker::poll_ready`] yields a
    /// completed readback slot. `None` means the pack still runs
    /// (if enabled) but the bytes are dropped - useful when a
    /// consumer wants to attach the recorder lazily.
    pub(crate) stacked_gpu_recorder: Option<Box<dyn StackedReplayGpuRecorder>>,

    /// Whether `resolve_current_pose` clamps output through the
    /// coverage boundary (FRICTION A13 - "constrained look"). `true`
    /// by default so the viewport never reveals black panorama
    /// edges; toggle off when the user wants to explore the raw
    /// panorama space (e.g. to find the edge of coverage during
    /// debugging or a cinematographic effect).
    ///
    /// The public [`Self::safe_clamp`] method remains available
    /// regardless of this flag - it's the primitive consumers use
    /// for ad-hoc clamping outside the render loop.
    pub(crate) constrained_look: bool,

    pub(crate) frame_count: u64,
    pub(crate) session_start: Option<Instant>,
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
            coverage: Some(coverage),
            ball_tracker: None,
            player_tracker: None,
            panner: None,
            previous_panner_pose: ViewportPosition::default(),
            detector: None,
            detection_interval: 1,
            last_detections: Vec::new(),
            replay,
            stacked_recorder: None,
            stacked_packer: None,
            stacked_gpu_recorder: None,
            constrained_look: true,
            frame_count: 0,
            session_start: None,
        })
    }

    // -----------------------------------------------------------------
    // Tracker / panner wiring
    // -----------------------------------------------------------------

    /// Attach a singleton ball tracker. Replaces any existing one.
    ///
    /// The tracker only drives the pose when a [`Panner`] is also
    /// attached via [`set_panner`](Self::set_panner); attached without
    /// a panner it still runs so detection sinks see consistent output
    /// but the pose stays at the pipeline default.
    pub fn set_ball_tracker(&mut self, tracker: Box<dyn Tracker>) {
        log::info!(
            "StitchCore: ball tracker attached (class_id={})",
            tracker.class_id()
        );
        self.ball_tracker = Some(tracker);
    }

    /// Remove the currently attached ball tracker.
    pub fn clear_ball_tracker(&mut self) {
        self.ball_tracker = None;
    }

    /// Attach a multi-entity player tracker. Replaces any existing one.
    ///
    /// The tracker's output populates
    /// [`WorldState::players`](crate::detect::tracker::WorldState::players)
    /// each frame. Phase-5 implementation - until that phase lands,
    /// this setter is usable from consumers but typically left unset.
    pub fn set_player_tracker(&mut self, tracker: Box<dyn Tracker>) {
        log::info!(
            "StitchCore: player tracker attached (class_id={})",
            tracker.class_id()
        );
        self.player_tracker = Some(tracker);
    }

    /// Remove the currently attached player tracker.
    pub fn clear_player_tracker(&mut self) {
        self.player_tracker = None;
    }

    /// Attach a panner. Replaces any existing one.
    ///
    /// Each frame, `resolve_current_pose` runs the registered trackers,
    /// builds a [`WorldState`], and delegates to [`Panner::decide`].
    /// Without a panner the pose stays at the pipeline default.
    ///
    /// [`WorldState`]: crate::detect::tracker::WorldState
    pub fn set_panner(&mut self, panner: Box<dyn Panner>) {
        log::info!("StitchCore: panner attached");
        self.panner = Some(panner);
    }

    /// Remove the currently attached panner. Pose reverts to the
    /// pipeline default until a new panner is set.
    pub fn clear_panner(&mut self) {
        log::info!("StitchCore: panner detached");
        self.panner = None;
    }

    /// Attach a unified-trait detector. Replaces any existing one.
    ///
    /// The detector runs on every `submit_frame_*` call whose frame
    /// count matches [`Self::detection_interval`]. Raw detections are
    /// mapped to panorama coordinates, then handed to each registered
    /// [`Tracker`] and to the detection sink (if any). Detection errors
    /// are logged (at `warn!` level) and swallowed so a transient
    /// inference failure does not abort the render loop.
    pub fn set_detector(&mut self, detector: Box<dyn UnifiedDetector>) {
        self.detector = Some(detector);
    }

    /// Remove the currently attached detector. Cached last detections
    /// are cleared so the director does not keep seeing stale data.
    pub fn clear_detector(&mut self) {
        self.detector = None;
        self.last_detections.clear();
    }

    /// Set how often detection runs.
    ///
    /// `1` (default) = every frame, `3` = every third frame, etc.
    /// Values `< 1` are clamped to `1`. Detection is expensive
    /// (2-20 ms depending on the model and backend); skipping frames
    /// lets the render loop run faster while the director interpolates
    /// using the latest detection output.
    pub fn set_detection_interval(&mut self, interval: u64) {
        self.detection_interval = interval.max(1);
    }

    /// Current detection interval.
    pub fn detection_interval(&self) -> u64 {
        self.detection_interval
    }

    /// The resolved viewport pose for the next render, already clamped
    /// through coverage + FOV limits. Exposed so interactive consumers
    /// (OBS pan/zoom, GUI drag) can preview where the core *would*
    /// render if they submit right now.
    pub fn current_pose(&mut self) -> ViewportPosition {
        // A peek does no detection work of its own, so the director
        // sees `fresh_detection = false`. The next real submit will
        // fire the schedule-driven detection path and pass the actual
        // run result.
        self.resolve_current_pose(false)
    }

    /// Clamp a prospective `(yaw, pitch, fov)` triple through the
    /// coverage boundary. No-op if no coverage is available (e.g. the
    /// calibration produced a degenerate boundary). `fov_degrees: None`
    /// uses the pipeline's current FOV.
    ///
    /// Input is treated as world-space (matches the director-output
    /// contract). Output is user-space via a simple
    /// `user_pitch = world_pitch - rig_tilt` transform. The yaw-weighted
    /// horizon correction is a render-site concern (see Model 3 in
    /// `PoseControl::render_pose`).
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
            pitch: clamped.pitch - rig_tilt,
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

    /// Whether the render loop's pose resolution clamps through the
    /// coverage boundary (FRICTION A13). `true` by default. Consumers
    /// expose this as a UI toggle ("Constrained look") so users can
    /// choose between "never show black edges" (on) and "unrestricted
    /// panning" (off).
    pub fn constrained_look(&self) -> bool {
        self.constrained_look
    }

    /// Enable or disable constrained-look clamping.
    ///
    /// When `true`, [`Self::submit_frame_yuv`] / `..._bgra` /
    /// `submit_frame_*_at_pose` pass the director's (or caller's)
    /// pose through [`Self::safe_clamp`] before rendering.
    /// When `false`, the raw pose is used verbatim; the FOV max is
    /// still respected (pipeline-set) but coverage-based yaw/pitch
    /// clamping is skipped.
    ///
    /// The public [`Self::safe_clamp`] method is unaffected - it
    /// always clamps, regardless of this flag.
    pub fn set_constrained_look(&mut self, enabled: bool) {
        self.constrained_look = enabled;
    }

    /// Toggle the constrained-look flag. Returns the new value.
    /// Consumers handling `HotkeyIntent::ToggleConstrained`
    /// wire it to this method.
    pub fn toggle_constrained_look(&mut self) -> bool {
        self.constrained_look = !self.constrained_look;
        self.constrained_look
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

    /// Mutable borrow of the replay buffer. Consumers wire this to
    /// "Clear replay" / "Save replay + reset" UI buttons which call
    /// [`ReplayBuffer::clear`] or [`ReplayBuffer::take`] respectively.
    pub fn replay_buffer_mut(&mut self) -> Option<&mut ReplayBuffer> {
        self.replay.as_mut()
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
}

impl crate::detect::DetectionTarget for StitchCore {
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::core::replay_buffer::ReplayBuffer;
    use crate::core::types::{RenderOutcome, ReplayFrame, StitchCoreError};
    use crate::detect::director::ViewportPosition;

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
        assert!(buf.oldest().is_none());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.buffered_duration(), Duration::ZERO);
        assert_eq!(buf.max_duration(), Duration::from_secs(1));
    }

    /// A16 "Clear replay" / "Save replay" UI wiring: `clear`,
    /// `snapshot`, `take`, `buffered_duration`, `oldest`.
    #[test]
    fn replay_buffer_snapshot_and_take_preserve_ordering() {
        let mut buf = ReplayBuffer::new(Duration::from_secs(10));
        for i in 0..3u8 {
            buf.push(ReplayFrame {
                rgba: vec![i; 4],
                captured_at: Duration::from_millis(i as u64 * 100),
                pose: ViewportPosition::default(),
            });
        }
        // snapshot returns oldest-to-newest, no consumption.
        let snap = buf.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap[0].rgba[0], 0);
        assert_eq!(snap[2].rgba[0], 2);
        assert_eq!(buf.len(), 3, "snapshot does not drain");

        // take drains and returns owned vec in same order.
        let drained = buf.take();
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].rgba[0], 0);
        assert!(buf.is_empty(), "take empties the buffer");
    }

    #[test]
    fn replay_buffer_clear_drops_frames_keeps_max_duration() {
        let mut buf = ReplayBuffer::new(Duration::from_secs(5));
        buf.push(ReplayFrame {
            rgba: vec![0u8; 4],
            captured_at: Duration::ZERO,
            pose: ViewportPosition::default(),
        });
        assert!(!buf.is_empty());
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(
            buf.max_duration(),
            Duration::from_secs(5),
            "clear preserves the configured window"
        );
    }

    #[test]
    fn replay_buffer_duration_tracks_oldest_newest_spread() {
        let mut buf = ReplayBuffer::new(Duration::from_secs(10));
        buf.push(ReplayFrame {
            rgba: vec![],
            captured_at: Duration::from_millis(100),
            pose: ViewportPosition::default(),
        });
        buf.push(ReplayFrame {
            rgba: vec![],
            captured_at: Duration::from_millis(850),
            pose: ViewportPosition::default(),
        });
        assert_eq!(buf.buffered_duration(), Duration::from_millis(750));
        assert_eq!(
            buf.oldest().unwrap().captured_at,
            Duration::from_millis(100)
        );
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

    /// `RenderOutcome` is `Send` - needed so consumers that post
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
