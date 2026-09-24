//! `StitchCore` - push-first canonical entry point for the stitching engine.
//!
//! Live sports production is the primary use case, so the canonical
//! API is push-based: consumers call `StitchCore::submit_frame`
//! whenever a new frame set is ready, and the
//! core owns the render substrate, readback, detection, pose
//! resolution, coverage, and the replay ring buffer.
//!
//! Batch file processing layers a thin pull-adapter on top
//! (`StitchSession::run`, available with the `gpu` feature).
//!
//! ## Foundation traits
//!
//! `StitchCore` composes the foundation traits:
//!
//! - The internal `Projection` trait - camera-geometry contract,
//!   implemented directly by the calibration topology's parameter
//!   structs ([`LShape`](crate::projection::LShape),
//!   [`Cylinder`](crate::projection::Cylinder)). The executor reads it
//!   through the document; the engine reads it back for coverage
//!   construction.
//! - [`crate::detect::detector::UnifiedDetector`] - collapsed CPU/CUDA/Metal
//!   detector contract with `DetectorError` for remote-inference futures.
//!   Wired via `StitchCore::set_detector`; detection runs on every
//!   `submit_frame_*` whose frame count is a multiple of
//!   `StitchCore::detection_interval`, and raw detections are mapped to
//!   panorama coordinates before reaching the director.
//!
//! ## Sub-modules
//!
//! - `types` - error types, config, render outcome, replay frame, recorder traits
//! - `replay_buffer` - bounded-duration ring buffer for replay frames
//! - `render` - submit and render-at-pose methods
//! - `replay_management` - stacked replay recorder wiring (CPU + GPU paths)
//! - `pose` - pose resolution, detection scheduling, panorama mapping

mod calibration_update;
mod coverage_motion;
mod pose;
mod render;
pub mod replay_buffer;
mod replay_management;
mod resize;
pub mod types;

use std::time::{Duration, Instant};

use crate::calibration::Calibration;
use crate::detect::detector::UnifiedDetector;
use crate::detect::director::MappedDetection;
use crate::detect::panner::Panner;
use crate::detect::tracker::Tracker;
use crate::geometry::Pose;
#[cfg(feature = "gpu")]
use crate::gpu::rgba_readback::RgbaReadback;
#[cfg(feature = "gpu")]
use crate::gpu::yuv_stack_packer::YuvStackPacker;
use crate::stitch::{Executor, StitchExecutor};

use self::replay_buffer::ReplayBuffer;
#[cfg(feature = "gpu")]
use self::types::StackedReplayGpuRecorder;
use self::types::{StackedReplayRecorder, StitchCoreError};

/// Canonical push-first stitching core.
///
/// See the module-level docs for design rationale. `StitchCore` owns:
///
/// - An [`Executor`] (CPU or GPU) - the render substrate, which itself
///   owns the active calibration and its projection geometry
///   (and, on the GPU arm, the wgpu pipeline).
/// - An `RgbaReadback` triple-buffered staging ring for CPU delivery
///   (GPU executors only; the CPU path returns owned bytes).
/// - A lazily derived coverage map shared by manual input and automatic motion.
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
    pub(crate) executor: Executor,
    /// Pipelined RGBA delivery ring - GPU executors only (`Some` iff
    /// the executor is [`Executor::Gpu`]). The CPU path returns owned
    /// bytes synchronously and never allocates it.
    #[cfg(feature = "gpu")]
    pub(crate) readback: Option<RgbaReadback>,
    /// The last CPU-stitched frame - the synchronous dual of the GPU
    /// staging ring. [`RenderOutcome::Rgba`](self::types::RenderOutcome)
    /// borrows from it on the CPU arm; empty on GPU engines.
    pub(crate) cpu_frame: Vec<u8>,
    /// One-shot guard for the mono-detection warning (detection is
    /// L-shape-only until the mono mapping lands).
    // TODO: remove once mono detection lands.
    mono_detection_warned: bool,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,

    pub(crate) coverage_motion: Option<crate::projection::coverage::controller::Controller>,
    pending_coverage: Option<resize::PendingCoverage>,

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
    /// a [`Pose`]. When unset, the pose stays at the
    /// pipeline default.
    pub(crate) panner: Option<Box<dyn Panner>>,
    /// Previous frame's resolved pose, passed to the panner in its
    /// [`PanContext`](crate::detect::panner::PanContext) so panners can
    /// compute first-order motion deltas statelessly.
    pub(crate) previous_panner_pose: Pose,

    /// Structured observability sink for the detect -> track -> pan
    /// chain (see [`crate::detect::pipeline_event`]). Owned by the
    /// engine so push consumers (`submit_frame_*`) and the pull
    /// session trace through the same slot.
    pub(crate) event_sink: Option<Box<dyn crate::detect::pipeline_event::PipelineEventSink>>,

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
    #[cfg(feature = "gpu")]
    pub(crate) stacked_packer: Option<YuvStackPacker>,

    /// Optional GPU-pack atlas recorder attached via
    /// [`Self::set_stacked_gpu_recorder`]. Receives the packed atlas
    /// bytes every time [`YuvStackPacker::poll_ready`] yields a
    /// completed readback slot. `None` means the pack still runs
    /// (if enabled) but the bytes are dropped - useful when a
    /// consumer wants to attach the recorder lazily.
    #[cfg(feature = "gpu")]
    pub(crate) stacked_gpu_recorder: Option<Box<dyn StackedReplayGpuRecorder>>,

    /// Apply coverage constraints to automatic presentation and direct interaction. Explicit render-at-pose methods bypass this policy.
    pub(crate) constrained_look: bool,

    pub(crate) frame_count: u64,
    pub(crate) session_start: Option<Instant>,
}

impl StitchCore {
    /// Build a new core around an executor - GPU or CPU, the engine's
    /// orchestration is the same.
    ///
    /// The executor owns the render substrate and the projection (see
    /// `GpuExecutorConfig` (with the `gpu` feature) /
    /// [`CpuExecutor::new`](crate::stitch::CpuExecutor::new)); the
    /// engine layers orchestration on top: detection, pose resolution,
    /// coverage clamping, replay, and delivery. Enable the replay ring
    /// after construction via [`Self::enable_replay_buffer`].
    pub fn new(executor: Executor) -> Result<Self, StitchCoreError> {
        let (output_width, output_height) = {
            let viewport = executor.viewport_size();
            (viewport.width, viewport.height)
        };
        // The pipelined RGBA ring is GPU delivery machinery; the CPU
        // executor returns owned bytes and needs none.
        #[cfg(feature = "gpu")]
        let readback = match executor.gpu() {
            Some(gpu) => Some(RgbaReadback::new(
                gpu.pipeline.gpu(),
                output_width,
                output_height,
            )?),
            None => None,
        };
        log::info!(
            "StitchCore: engine constructed over the '{}' executor",
            StitchExecutor::name(&executor)
        );

        Ok(Self {
            executor,
            #[cfg(feature = "gpu")]
            readback,
            cpu_frame: Vec::new(),
            mono_detection_warned: false,
            output_width,
            output_height,
            coverage_motion: None,
            pending_coverage: None,
            ball_tracker: None,
            player_tracker: None,
            panner: None,
            previous_panner_pose: Pose::default(),
            event_sink: None,
            detector: None,
            detection_interval: 1,
            last_detections: Vec::new(),
            replay: None,
            stacked_recorder: None,
            #[cfg(feature = "gpu")]
            stacked_packer: None,
            #[cfg(feature = "gpu")]
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

    /// Attach a pipeline event sink for structured observability.
    ///
    /// The sink receives the detect -> track -> pan event stream
    /// (see [`crate::detect::pipeline_event`]) from every pose
    /// dispatch, whichever entry point drives it (push `submit_*`
    /// or the pull session). There is deliberately no clear method -
    /// in a <1.0.0 codebase the engine is re-created for that.
    pub fn set_event_sink(
        &mut self,
        sink: Box<dyn crate::detect::pipeline_event::PipelineEventSink>,
    ) {
        log::info!("StitchCore: event sink attached");
        self.event_sink = Some(sink);
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

    /// The most recent panorama-mapped detections (cached across
    /// skipped frames so the director keeps context).
    pub fn last_detections(&self) -> &[MappedDetection] {
        &self.last_detections
    }

    #[cfg(test)]
    pub(crate) fn current_pose(&self) -> Pose {
        self.coverage_motion
            .as_ref()
            .and_then(|c| c.current())
            .unwrap_or_else(|| self.world_to_render_pose(self.previous_panner_pose))
    }

    /// Orient a world-space pose into the render-space `(yaw, pitch)` the
    /// `view_matrix` consumes - the rig tilt/roll basis inversion without
    /// any coverage clamping;
    /// the unconstrained render path uses it so disabling the clamp never
    /// disables horizon leveling.
    pub fn world_to_render_pose(&self, world: Pose) -> Pose {
        let framing = &self.executor.calibration().framing;
        let cam = self.calibration().virtual_camera();
        let (yaw, pitch) = crate::geometry::world_to_render_pose(
            &cam,
            world.yaw,
            world.pitch,
            framing.tilt as f32,
            framing.roll as f32,
        );
        Pose {
            yaw,
            pitch,
            fov_degrees: world.fov_degrees,
        }
    }

    /// Recover the world-space look direction of a render-space pose under the accepted calibration.
    ///
    /// Inverse of [`Self::world_to_render_pose`] away from the yaw singularity at the poles. Preserves FOV and does not run coverage. Use before a framing edit, then orient the result under the accepted new calibration to retain the look direction.
    pub fn render_to_world_pose(&self, render: Pose) -> Pose {
        let framing = &self.calibration().framing;
        let camera = self.calibration().virtual_camera();
        let view = crate::geometry::view_matrix(
            &camera,
            render.yaw,
            render.pitch,
            framing.tilt as f32,
            framing.roll as f32,
        );
        let direction = nalgebra::Vector3::new(-view[(2, 0)], -view[(2, 1)], -view[(2, 2)]);
        let angles = camera.direction_to_yaw_pitch(&direction);
        Pose {
            yaw: angles.yaw,
            pitch: angles.pitch,
            fov_degrees: render.fov_degrees,
        }
    }

    // -----------------------------------------------------------------
    // Live render parameters (first-class - no pipeline reach-through)
    // -----------------------------------------------------------------

    /// The active calibration document.
    pub fn calibration(&self) -> &Calibration {
        self.executor.calibration()
    }

    /// Resize the output viewport. Returns the accepted `(width, height)`,
    /// or `None` when the request was rejected (zero dimension).
    ///
    /// A resize is a stream discontinuity for the delivery machinery:
    /// the GPU streaming readback ring is rebuilt at the new size
    /// (frames in flight in the old ring are dropped) and the executor's
    /// NV12 delivery re-creates itself on next use (it is keyed by dims).
    /// Configured coverage motion is rebuilt before the resize is accepted. A failed
    /// map rebuild returns `None` without changing the viewport or motion state.
    pub fn resize(&mut self, width: u32, height: u32) -> Option<(u32, u32)> {
        if width == 0 || height == 0 {
            return None;
        }
        if (width, height) == (self.output_width, self.output_height)
            && (width, height)
                == (
                    self.executor.viewport_size().width,
                    self.executor.viewport_size().height,
                )
        {
            return Some((width, height));
        }
        let rebuilt = match self
            .coverage_motion
            .as_ref()
            .filter(|controller| {
                !controller
                    .matches_viewport(&crate::render::viewport::ViewportSize { width, height })
            })
            .map(|controller| {
                controller.rebuild(
                    self.calibration(),
                    &crate::render::viewport::ViewportSize { width, height },
                )
            })
            .transpose()
        {
            Ok(controller) => controller,
            Err(error) => {
                log::warn!("Coverage resize rejected: {error}");
                return None;
            }
        };
        self.apply_resize(width, height, rebuilt)
    }

    fn apply_resize(
        &mut self,
        width: u32,
        height: u32,
        rebuilt: Option<crate::projection::coverage::controller::Controller>,
    ) -> Option<(u32, u32)> {
        let accepted = self.executor.resize(width, height);
        if accepted.is_some() {
            if let Some(controller) = rebuilt {
                self.coverage_motion = Some(controller);
            }
            self.output_width = width;
            self.output_height = height;
            #[cfg(feature = "gpu")]
            if let Some(gpu) = self.executor.gpu() {
                // Infallible here: `Executor::resize` already rejected
                // zero dimensions, the ring's only construction error.
                self.readback = Some(
                    RgbaReadback::new(gpu.pipeline.gpu(), width, height)
                        .expect("resize validated non-zero dimensions"),
                );
                log::info!(
                    "StitchCore: delivery machinery rebuilt for the {width}x{height} resize"
                );
            }
        }
        accepted
    }

    /// Whether source YUV uses full-range (0-255) quantization rather
    /// than limited range (16-235).
    pub fn set_full_range(&mut self, full_range: bool) {
        self.executor.set_full_range(full_range);
    }

    /// Flip each camera's source 180 degrees at sample time (for
    /// rotated mounts). GPU sampling only; CPU sources reverse
    /// buffers at decode.
    pub fn set_flip_180(&mut self, left: bool, right: bool) {
        self.executor.set_flip_180(left, right);
    }

    /// Connected coverage FOV cap, or `None` before the map has been prepared.
    pub fn max_fov_degrees(&self) -> Option<f32> {
        self.coverage_motion.as_ref().map(|c| c.cap())
    }

    // -----------------------------------------------------------------
    // Coverage / calibration / projection introspection
    // -----------------------------------------------------------------

    /// Whether the render loop's pose resolution clamps through the
    /// coverage boundary. `true` by default. Consumers
    /// expose this as a UI toggle ("Constrained look") so users can
    /// choose between "never show black edges" (on) and "unrestricted
    /// panning" (off).
    pub fn constrained_look(&self) -> bool {
        self.constrained_look
    }

    /// Enable or disable constrained-look clamping.
    ///
    /// Applies to [`Self::advance_pose`], director-driven push submits and session
    /// presentation. Disabling it bypasses coverage but retains world-to-render
    /// orientation. Changing the flag resets the motion clock, not the map.
    /// Explicit render-at-pose methods consume an already-resolved render pose.
    ///
    pub fn set_constrained_look(&mut self, enabled: bool) {
        if enabled != self.constrained_look
            && let Some(controller) = &mut self.coverage_motion
        {
            controller.reset();
        }
        self.constrained_look = enabled;
    }

    /// Toggle the constrained-look flag. Returns the new value.
    /// Consumers handling a constrained-look toggle
    /// wire it to this method.
    pub fn toggle_constrained_look(&mut self) -> bool {
        self.set_constrained_look(!self.constrained_look);
        self.constrained_look
    }

    /// Short name of the active projection (for logs + UI labels).
    pub fn projection_name(&self) -> &'static str {
        self.executor.projection().name()
    }

    /// Camera count the active projection consumes (one submitted frame
    /// per camera plane). For today's stereo L-shape this is `2`; a
    /// mono projection exposes `1`.
    pub fn camera_count(&self) -> usize {
        self.executor.projection().camera_count()
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
    /// `config.viewport_size.{width,height}` at construction.
    pub fn output_dims(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }

    /// Source frame dimensions `(width, height)` per camera.
    pub fn source_info(&self) -> (u32, u32) {
        self.executor.source_info()
    }

    /// Number of frames submitted so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// The GPU context, for consumers that create auxiliary resources
    /// on the engine's device (preview surfaces, demosaic kernels,
    /// VRAM queries). Panics when the engine runs the CPU executor;
    /// use [`DetectionTarget::gpu`](crate::detect::DetectionTarget::gpu)
    /// for executor-agnostic access.
    #[cfg(feature = "gpu")]
    pub fn gpu(&self) -> &crate::gpu::GpuContext {
        self.executor
            .gpu()
            .expect("gpu() requires the GPU executor")
            .pipeline
            .gpu()
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
    fn source_info(&self) -> (u32, u32) {
        self.source_info()
    }
    #[cfg(feature = "gpu")]
    fn gpu(&self) -> Option<&crate::gpu::GpuContext> {
        self.executor.gpu().map(|g| g.pipeline.gpu())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::core::replay_buffer::ReplayBuffer;
    use crate::core::types::{RenderOutcome, ReplayFrame, StitchCoreError};
    use crate::geometry::Pose;

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
                pose: Pose::default(),
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
            pose: Pose::default(),
        });
        buf.push(ReplayFrame {
            rgba: vec![],
            captured_at: Duration::from_millis(100),
            pose: Pose::default(),
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
                pose: Pose::default(),
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
            pose: Pose::default(),
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
            pose: Pose::default(),
        });
        buf.push(ReplayFrame {
            rgba: vec![],
            captured_at: Duration::from_millis(850),
            pose: Pose::default(),
        });
        assert_eq!(buf.buffered_duration(), Duration::from_millis(750));
        assert_eq!(
            buf.oldest().unwrap().captured_at,
            Duration::from_millis(100)
        );
    }

    /// The SAME engine constructs over the CPU executor - pure-logic
    /// surfaces (coverage, clamp, live setters, introspection) work
    /// identically, and the GPU-only streaming paths return the typed
    /// error instead of panicking.
    #[test]
    fn engine_over_cpu_executor_pure_logic_works() {
        use crate::core::StitchCore;
        use crate::render::viewport::ViewportSize;
        use crate::stitch::{CpuExecutor, Executor, test_support::calib};

        let (w, h) = (64u32, 36u32);
        let executor = CpuExecutor::new(
            calib(w, h),
            ViewportSize {
                width: w,
                height: h,
            },
            w,
            h,
            false,
        )
        .expect("cpu executor");
        let mut core =
            StitchCore::new(Executor::Cpu(Box::new(executor))).expect("cpu engine constructs");

        assert_eq!(core.output_dims(), (w, h));
        assert_eq!(core.camera_count(), 2);
        assert_eq!(core.projection_name(), "l-shape-stereo-2camera");
        assert!(
            core.prepare_coverage().is_ok(),
            "coverage builds without a GPU"
        );
        assert!(core.max_fov_degrees().is_some());

        let mut calibration = core.calibration().clone();
        calibration.topology.l_shape_mut().unwrap().blend_width = 0.1;
        calibration.framing.tilt = 0.2;
        core.update_calibration(calibration).unwrap();
        assert!((core.calibration().topology.blend_width() - 0.1).abs() < 1e-6);
        assert!((core.calibration().framing.tilt - 0.2).abs() < 1e-6);
        assert_eq!(core.resize(48, 26), Some((48, 26)));
        assert_eq!(core.output_dims(), (48, 26));

        // Coverage clamp works CPU-side (pure geometry).
        let requested = Pose {
            yaw: 10.0,
            pitch: 10.0,
            fov_degrees: 50.0,
        };
        let clamped = core.constrain_render_pose(requested, requested).unwrap();
        assert!(clamped.yaw.is_finite() && clamped.pitch.is_finite());
        assert!(clamped.pitch.abs() < std::f32::consts::FRAC_PI_2);
        assert!(clamped.fov_degrees <= core.max_fov_degrees().unwrap());

        // The YUV submit path works on the CPU arm - synchronously
        // (no warmup), at the resized output dimensions.
        let y = vec![0u8; (w * h) as usize];
        let uv = vec![128u8; (w * h / 4) as usize];
        let cam = crate::source::YuvData {
            y,
            u: uv.clone(),
            v: uv,
        };
        let frames = crate::source::FrameSet::Yuv420p(vec![cam.clone(), cam]);
        match core.submit_frame(&frames, 1.0 / 30.0) {
            Ok(crate::core::types::RenderOutcome::Rgba(bytes)) => {
                assert_eq!(bytes.len(), 48 * 26 * 4);
            }
            Ok(crate::core::types::RenderOutcome::Warmup) => {
                panic!("CPU submit is synchronous - no warmup")
            }
            Err(e) => panic!("cpu submit failed: {e}"),
        }
        assert_eq!(core.frame_count(), 1);

        // Genuinely GPU-only paths stay typed errors, not panics
        // (the methods exist only when the gpu feature compiles them).
        #[cfg(feature = "gpu")]
        {
            assert!(matches!(
                core.flush().unwrap_err(),
                StitchCoreError::RequiresGpu
            ));
        }
    }

    /// The FrameSet arity invariant: a set whose length does not match
    /// the projection's camera count is rejected with a typed error at
    /// the submit boundary (`check_camera_count` on the CPU stitch
    /// path), never silently mis-sampled. This is the enforcement
    /// artifact for the "len == camera_count" claim on [`FrameSet`].
    #[test]
    fn submit_frame_rejects_camera_count_mismatch() {
        use crate::core::StitchCore;
        use crate::render::viewport::ViewportSize;
        use crate::stitch::{CpuExecutor, Executor, test_support::calib};

        let (w, h) = (64u32, 36u32);
        let executor = CpuExecutor::new(
            calib(w, h), // L-shape: camera_count == 2
            ViewportSize {
                width: w,
                height: h,
            },
            w,
            h,
            false,
        )
        .expect("cpu executor");
        let mut core = StitchCore::new(Executor::Cpu(Box::new(executor))).expect("cpu engine");

        let cam = crate::source::YuvData {
            y: vec![0u8; (w * h) as usize],
            u: vec![128u8; (w * h / 4) as usize],
            v: vec![128u8; (w * h / 4) as usize],
        };
        let mono_set = crate::source::FrameSet::Yuv420p(vec![cam]);
        // RenderOutcome borrows the core and is not Debug; unwrap by match.
        let err = match core.submit_frame(&mono_set, 1.0 / 30.0) {
            Ok(_) => panic!("a 1-camera set on a 2-camera projection must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("2 camera(s) but 1 frame(s)"),
            "expected the camera-count mismatch error, got: {err}"
        );
    }

    /// Engine-level executor agreement: the same submit API
    /// (`submit_frame`, NV12 set) driven over a CPU engine and a GPU engine
    /// produces the same frame within the established oracle bounds.
    /// The Step-12 guarantee: swapping the executor does not change
    /// the picture.
    #[test]
    #[cfg(feature = "gpu")]
    fn cpu_and_gpu_engines_agree_via_submit_nv12() {
        use crate::core::StitchCore;
        use crate::core::types::RenderOutcome;
        use crate::render::renderer::InputFormat;
        use crate::render::viewport::ViewportSize;
        use crate::stitch::test_support::{Agreement, AgreementBounds, calib, gpu_or_skip, nv12};
        use crate::stitch::{CpuExecutor, Executor, GpuExecutor, GpuExecutorConfig};

        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let (cam_w, cam_h) = (192u32, 108u32);
        let (out_w, out_h) = (160u32, 90u32);
        let config = ViewportSize {
            width: out_w,
            height: out_h,
        };
        let (ly, luv) = nv12(cam_w, cam_h, 0);
        let (ry, ruv) = nv12(cam_w, cam_h, 30);
        let frames = crate::source::FrameSet::Nv12(vec![
            crate::source::Nv12Data { y: ly, uv: luv },
            crate::source::Nv12Data { y: ry, uv: ruv },
        ]);

        let cpu_exec = CpuExecutor::new(calib(cam_w, cam_h), config.clone(), cam_w, cam_h, false)
            .expect("cpu executor");
        let mut cpu_core = StitchCore::new(Executor::Cpu(Box::new(cpu_exec))).expect("cpu engine");

        let gpu_exec = GpuExecutor::new(
            gpu,
            GpuExecutorConfig {
                viewport_size: config,
                ..GpuExecutorConfig::new(calib(cam_w, cam_h), cam_w, cam_h, InputFormat::Nv12)
            },
        )
        .expect("gpu executor");
        let mut gpu_core = StitchCore::new(Executor::Gpu(Box::new(gpu_exec))).expect("gpu engine");

        // CPU: synchronous - the first submit yields the frame.
        let cpu_rgba = match cpu_core
            .submit_frame(&frames, 1.0 / 30.0)
            .expect("cpu submit")
        {
            RenderOutcome::Rgba(bytes) => bytes.to_vec(),
            RenderOutcome::Warmup => panic!("CPU submit is synchronous - no warmup"),
        };

        // GPU: triple-buffered - the ring yields the first frame by
        // the third submit. Every submit renders the identical frame
        // (same input, same resolved pose), so any yielded frame works.
        let mut gpu_rgba = None;
        for _ in 0..3 {
            if let RenderOutcome::Rgba(bytes) = gpu_core
                .submit_frame(&frames, 1.0 / 30.0)
                .expect("gpu submit")
            {
                gpu_rgba = Some(bytes.to_vec());
                break;
            }
        }
        let gpu_rgba = gpu_rgba.expect("gpu produced a frame by the third submit");
        assert_eq!(cpu_rgba.len(), gpu_rgba.len());
        Agreement::compare(&gpu_rgba, &cpu_rgba)
            .assert_within(AgreementBounds::DEFAULT, "engine cpu-vs-gpu submit_nv12");
    }

    /// Resize is a delivery-machinery discontinuity: the streaming
    /// readback ring must follow the new size, so a submit after a
    /// resize yields frames at the resized dimensions. A ring left at
    /// the construction size fails wgpu copy validation on the next
    /// submit.
    #[test]
    #[cfg(feature = "gpu")]
    fn resize_then_submit_yields_resized_frames() {
        use crate::core::StitchCore;
        use crate::core::types::RenderOutcome;
        use crate::render::renderer::InputFormat;
        use crate::render::viewport::ViewportSize;
        use crate::stitch::test_support::{calib, gpu_or_skip, nv12};
        use crate::stitch::{Executor, GpuExecutor, GpuExecutorConfig};

        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let (cam_w, cam_h) = (192u32, 108u32);
        let exec = GpuExecutor::new(
            gpu,
            GpuExecutorConfig {
                viewport_size: ViewportSize {
                    width: 160,
                    height: 90,
                },
                ..GpuExecutorConfig::new(calib(cam_w, cam_h), cam_w, cam_h, InputFormat::Nv12)
            },
        )
        .expect("gpu executor");
        let mut core = StitchCore::new(Executor::Gpu(Box::new(exec))).expect("engine");

        let (ly, luv) = nv12(cam_w, cam_h, 0);
        let (ry, ruv) = nv12(cam_w, cam_h, 30);
        let frames = crate::source::FrameSet::Nv12(vec![
            crate::source::Nv12Data { y: ly, uv: luv },
            crate::source::Nv12Data { y: ry, uv: ruv },
        ]);

        // Warm the ring at the original size, then resize.
        let _ = core.submit_frame(&frames, 1.0 / 30.0).expect("submit");
        assert_eq!(core.resize(128, 72), Some((128, 72)));

        // The rebuilt ring delivers frames at the new size by the
        // third post-resize submit.
        let mut delivered = None;
        for _ in 0..3 {
            if let RenderOutcome::Rgba(bytes) =
                core.submit_frame(&frames, 1.0 / 30.0).expect("submit")
            {
                delivered = Some(bytes.len());
                break;
            }
        }
        assert_eq!(delivered, Some((128 * 72 * 4) as usize));
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

    /// A stub detector returning no detections - just enough for
    /// scheduling assertions.
    struct NullDetector;

    impl crate::detect::detector::UnifiedDetector for NullDetector {
        fn name(&self) -> &'static str {
            "null"
        }
        fn detect(
            &mut self,
            _camera: crate::geometry::CameraIndex,
            _frame: &crate::detect::detector::DetectorFrame<'_>,
        ) -> Result<Vec<crate::detect::detector::Detection>, crate::detect::detector::DetectorError>
        {
            Ok(Vec::new())
        }
    }

    fn cpu_engine(w: u32, h: u32) -> crate::core::StitchCore {
        use crate::render::viewport::ViewportSize;
        use crate::stitch::{CpuExecutor, Executor, test_support::calib};
        let executor = CpuExecutor::new(
            calib(w, h),
            ViewportSize {
                width: w,
                height: h,
            },
            w,
            h,
            false,
        )
        .expect("cpu executor");
        crate::core::StitchCore::new(Executor::Cpu(Box::new(executor))).expect("cpu engine")
    }

    /// Detection scheduling: no detector means never due; with one,
    /// the interval gates and the setter clamps to >= 1.
    #[test]
    fn detection_due_needs_detector_and_respects_interval() {
        let mut core = cpu_engine(64, 36);
        assert!(!core.detection_due(0), "no detector attached");

        core.set_detector(Box::new(NullDetector));
        assert!(core.detection_due(0));
        assert!(core.detection_due(1));

        core.set_detection_interval(3);
        assert!(core.detection_due(0));
        assert!(!core.detection_due(1));
        assert!(!core.detection_due(2));
        assert!(core.detection_due(3));

        core.set_detection_interval(0);
        assert!(core.detection_due(1), "interval clamps to 1");
    }

    /// `run_detection_frames` dispatches once per camera and caches
    /// the panorama-mapped results for the director.
    #[test]
    fn run_detection_frames_dispatches_both_cameras() {
        use crate::detect::detector::{
            ChromaFormat, Detection, DetectorError, DetectorFrame, RawFrame, UnifiedDetector,
        };
        use crate::geometry::CameraIndex;

        struct RecordingDetector;
        impl UnifiedDetector for RecordingDetector {
            fn name(&self) -> &'static str {
                "recording"
            }
            fn detect(
                &mut self,
                camera: CameraIndex,
                frame: &DetectorFrame<'_>,
            ) -> Result<Vec<Detection>, DetectorError> {
                match frame {
                    DetectorFrame::Cpu(_) => Ok(vec![Detection {
                        camera,
                        class_id: 0,
                        confidence: 0.9,
                        center_x: 0.5,
                        center_y: 0.5,
                        width: 0.1,
                        height: 0.1,
                    }]),
                    _ => Err(DetectorError::UnsupportedFrameKind),
                }
            }
        }

        let mut core = cpu_engine(64, 36);
        core.set_detector(Box::new(RecordingDetector));

        let y = vec![0u8; 8];
        let uv = vec![128u8; 2];
        let frame = |cam| {
            (
                cam,
                DetectorFrame::Cpu(RawFrame {
                    y: &y,
                    chroma: ChromaFormat::Yuv420p { u: &uv, v: &uv },
                    width: 4,
                    height: 2,
                }),
            )
        };
        core.run_detection_frames(&[frame(0), frame(1)]);

        let dets = core.last_detections();
        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].camera, 0);
        assert_eq!(dets[1].camera, 1);
    }
}
