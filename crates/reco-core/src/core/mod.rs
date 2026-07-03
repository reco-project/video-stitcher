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
//!   Bound to the executor at construction (see
//!   [`GpuExecutorConfig`](crate::stitch::GpuExecutorConfig)); the
//!   engine reads it back for coverage construction.
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

mod pose;
mod render;
pub mod replay_buffer;
mod replay_management;
pub mod types;

use std::time::{Duration, Instant};

use crate::calibration::Calibration;
use crate::detect::detector::UnifiedDetector;
use crate::detect::director::MappedDetection;
use crate::detect::panner::Panner;
use crate::detect::tracker::Tracker;
use crate::geometry::ViewportPosition;
use crate::gpu::GpuContext;
use crate::gpu::rgba_readback::RgbaReadback;
use crate::gpu::yuv_stack_packer::YuvStackPacker;
use crate::projection::{CoverageBoundary, PanoramaExtent};
use crate::render::pipeline::StitchPipeline;
use crate::stitch::{Executor, StitchExecutor};

use self::replay_buffer::ReplayBuffer;
use self::types::{StackedReplayGpuRecorder, StackedReplayRecorder, StitchCoreError};

/// Canonical push-first stitching core.
///
/// See the module-level docs for design rationale. `StitchCore` owns:
///
/// - An [`Executor`] (CPU or GPU) - the render substrate, which itself
///   owns the active [`Projection`](crate::projection::Projection)
///   (and, on the GPU arm, the wgpu pipeline).
/// - An [`RgbaReadback`] triple-buffered staging ring for CPU delivery
///   (GPU executors only; the CPU path returns owned bytes).
/// - A coverage boundary precomputed from calibration for `safe_clamp`.
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
    pub(crate) readback: Option<RgbaReadback>,
    /// The last CPU-stitched frame - the synchronous dual of the GPU
    /// staging ring. [`RenderOutcome::Rgba`](self::types::RenderOutcome)
    /// borrows from it on the CPU arm; empty on GPU engines.
    pub(crate) cpu_frame: Vec<u8>,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,

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

    /// NV12 converter for the preview-mode recording tap
    /// ([`Self::render_and_readback_nv12`]); lazy so pure display
    /// consumers never pay for the staging ring.
    pub(crate) preview_nv12: Option<crate::gpu::nv12_converter::Nv12Converter>,

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
    /// Build a new core around an executor - GPU or CPU, the engine's
    /// orchestration is the same.
    ///
    /// The executor owns the render substrate and the projection (see
    /// [`GpuExecutorConfig`](crate::stitch::GpuExecutorConfig) /
    /// [`CpuExecutor::new`](crate::stitch::CpuExecutor::new)); the
    /// engine layers orchestration on top: detection, pose resolution,
    /// coverage clamping, replay, and delivery. Enable the replay ring
    /// after construction via [`Self::enable_replay_buffer`].
    pub fn new(executor: Executor) -> Result<Self, StitchCoreError> {
        let (output_width, output_height) = {
            let viewport = executor.viewport();
            (viewport.width, viewport.height)
        };
        // The pipelined RGBA ring is GPU delivery machinery; the CPU
        // executor returns owned bytes and needs none.
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

        // The projection owns coverage construction: a new projection
        // brings its own boundary representation with it.
        let coverage = executor
            .projection()
            .coverage(executor.calibration(), executor.scene());

        Ok(Self {
            executor,
            readback,
            cpu_frame: Vec::new(),
            output_width,
            output_height,
            coverage: Some(coverage),
            ball_tracker: None,
            player_tracker: None,
            panner: None,
            previous_panner_pose: ViewportPosition::default(),
            detector: None,
            detection_interval: 1,
            last_detections: Vec::new(),
            replay: None,
            preview_nv12: None,
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
    /// contract). Output is render-space: resolved through the shared
    /// `resolve_render_pose` authority (world-space clamp + roll-aware
    /// tilt/roll basis inversion), identical to the export/director path.
    pub fn safe_clamp(&self, pose: ViewportPosition) -> ViewportPosition {
        let Some(coverage) = &self.coverage else {
            return pose;
        };
        let fov = pose
            .fov_degrees
            .unwrap_or_else(|| self.executor.fov())
            .min(coverage.max_fov_degrees());
        let aspect = self.executor.viewport().aspect_ratio();
        let rig_tilt = self.executor.calibration().framing.tilt as f32;
        let rig_roll = self.executor.calibration().framing.roll as f32;
        let cam = crate::geometry::VirtualCamera::new(&self.executor.scene().camera_position);
        let (yaw, pitch) = crate::geometry::resolve_render_pose(
            coverage, &cam, rig_tilt, rig_roll, pose.yaw, pose.pitch, fov, aspect,
        );
        ViewportPosition {
            yaw,
            pitch,
            fov_degrees: Some(fov),
        }
    }

    /// Orient a world-space pose into the render-space `(yaw, pitch)` the
    /// `view_matrix` consumes - the rig tilt/roll basis inversion without
    /// any coverage clamping. The orient half of [`Self::safe_clamp`];
    /// the unconstrained render path uses it so disabling the clamp never
    /// disables horizon leveling.
    pub fn orient_pose(&self, world: ViewportPosition) -> ViewportPosition {
        let framing = &self.executor.calibration().framing;
        let cam = crate::geometry::VirtualCamera::new(&self.executor.scene().camera_position);
        let (yaw, pitch) = crate::geometry::world_to_render_pose(
            &cam,
            world.yaw,
            world.pitch,
            framing.tilt as f32,
            framing.roll as f32,
        );
        ViewportPosition {
            yaw,
            pitch,
            fov_degrees: world.fov_degrees,
        }
    }

    // -----------------------------------------------------------------
    // Live render parameters (first-class - no pipeline reach-through)
    // -----------------------------------------------------------------

    /// The active calibration document.
    pub fn calibration(&self) -> &Calibration {
        self.executor.calibration()
    }

    /// Set the vertical field of view in degrees.
    pub fn set_fov(&mut self, fov_degrees: f32) {
        self.executor.set_fov(fov_degrees);
    }

    /// Current vertical field of view in degrees.
    pub fn fov(&self) -> f32 {
        self.executor.fov()
    }

    /// Resize the output viewport. Returns the previous `(width, height)`
    /// when the size actually changed (see [`StitchPipeline::resize`]).
    pub fn resize(&mut self, width: u32, height: u32) -> Option<(u32, u32)> {
        let prev = self.executor.resize(width, height);
        if prev.is_some() {
            self.output_width = width;
            self.output_height = height;
        }
        prev
    }

    /// Set the seam blend width (per-frame uniform; coverage unaffected).
    pub fn set_blend_width(&mut self, width: f32) {
        self.executor.set_blend_width(width);
    }

    /// Set the lens-correction strength on every lens (`0` = pinhole,
    /// `1` = full KB4).
    pub fn set_lens_correction_amount(&mut self, amount: f32) {
        self.executor.set_lens_correction_amount(amount);
    }

    /// Set rig tilt in radians, keeping the coverage clamp in sync.
    ///
    /// The boundary's roll-aware clamp margins read the rig tilt/roll, so
    /// a live change must refresh them. The sampled boundary itself is
    /// tilt-invariant (a view-time basis rotation), so this is a scalar
    /// update, not a dense resample.
    pub fn set_rig_tilt(&mut self, radians: f32) {
        let mut framing = self.executor.calibration().framing.clone();
        framing.tilt = radians as f64;
        self.executor.update_framing(framing);
        self.refresh_coverage_orientation();
    }

    /// Set rig roll in radians, keeping the coverage clamp in sync.
    /// See [`Self::set_rig_tilt`] for why no dense rebuild is needed.
    pub fn set_rig_roll(&mut self, radians: f32) {
        let mut framing = self.executor.calibration().framing.clone();
        framing.roll = radians as f64;
        self.executor.update_framing(framing);
        self.refresh_coverage_orientation();
    }

    /// Replace the topology (plane placement + seam) and recompute coverage.
    pub fn update_topology(&mut self, topology: crate::calibration::Topology) {
        self.executor.update_topology(topology);
        self.rebuild_coverage();
    }

    /// Replace the framing (axis offset, tilt, roll) and recompute coverage.
    pub fn update_framing(&mut self, framing: crate::calibration::Framing) {
        self.executor.update_framing(framing);
        self.rebuild_coverage();
    }

    /// Replace one or both cameras' intrinsics and recompute the coverage
    /// boundary - it samples the frame edges through the lens model, so
    /// intrinsics changes move the no-black region.
    pub fn update_camera_params(
        &mut self,
        left: Option<crate::calibration::Lens>,
        right: Option<crate::calibration::Lens>,
    ) {
        self.executor.update_camera_params(left, right);
        self.rebuild_coverage();
    }

    /// Maximum vertical FOV (degrees) that fits inside the coverage, or
    /// `None` when the calibration produced a degenerate boundary.
    pub fn max_fov_degrees(&self) -> Option<f32> {
        self.coverage.as_ref().map(|c| c.max_fov_degrees())
    }

    fn rebuild_coverage(&mut self) {
        self.coverage = Some(
            self.executor
                .projection()
                .coverage(self.executor.calibration(), self.executor.scene()),
        );
    }

    fn refresh_coverage_orientation(&mut self) {
        let framing = &self.executor.calibration().framing;
        let (tilt, roll) = (framing.tilt as f32, framing.roll as f32);
        if let Some(coverage) = self.coverage.as_mut() {
            coverage.set_rig_orientation(tilt, roll);
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
        self.executor.projection().name()
    }

    /// Camera count the active projection consumes (one submitted frame
    /// per camera plane). For today's stereo L-shape this is `2`; a
    /// mono projection exposes `1`.
    pub fn camera_count(&self) -> u8 {
        self.executor.projection().camera_count()
    }

    /// Hot-swap the calibration. Takes effect on the next submit.
    ///
    /// Re-derives the coverage boundary from the new calibration so
    /// subsequent `safe_clamp` calls respect the new no-black region.
    pub fn update_calibration(&mut self, calibration: Calibration) {
        self.executor.update_calibration(calibration);
        self.rebuild_coverage();
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
    ///
    /// GPU-executor reach-through (dies with the pipeline demotion in
    /// 9B). Panics when the engine runs the CPU executor - CPU engines
    /// have no pipeline; use the engine's own setters and submit paths.
    pub fn pipeline(&self) -> &StitchPipeline {
        &self
            .executor
            .gpu()
            .expect("pipeline() requires the GPU executor")
            .pipeline
    }

    /// Mutable access to the pipeline for advanced callers that need
    /// to tweak viewport / FOV / zero-copy bind groups directly.
    /// See [`Self::pipeline`] for the CPU-executor caveat.
    pub fn pipeline_mut(&mut self) -> &mut StitchPipeline {
        &mut self
            .executor
            .gpu_mut()
            .expect("pipeline_mut() requires the GPU executor")
            .pipeline
    }

    /// The GPU context owning every resource. See [`Self::pipeline`]
    /// for the CPU-executor caveat.
    pub fn gpu(&self) -> &GpuContext {
        self.pipeline().gpu()
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
    use crate::geometry::ViewportPosition;

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

    /// The keystone-B contract: the SAME engine constructs over the
    /// CPU executor - pure-logic surfaces (coverage, clamp, live
    /// setters, introspection) work identically, and the GPU-only
    /// streaming paths return the typed error instead of panicking.
    #[test]
    fn engine_over_cpu_executor_pure_logic_works() {
        use crate::core::StitchCore;
        use crate::render::pipeline::YuvPlanes;
        use crate::render::viewport::ViewportConfig;
        use crate::stitch::{CpuExecutor, Executor, test_support::calib};

        let (w, h) = (64u32, 36u32);
        let executor = CpuExecutor::new(
            Box::new(crate::projection::LShapeProjection),
            calib(w, h),
            ViewportConfig {
                width: w,
                height: h,
                ..Default::default()
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
        assert!(core.coverage().is_some(), "coverage builds without a GPU");
        assert!(core.max_fov_degrees().is_some());

        // Live setters dispatch to the CPU arm (document mutation).
        core.set_fov(50.0);
        assert!((core.fov() - 50.0).abs() < f32::EPSILON);
        core.set_blend_width(0.1);
        assert!((core.calibration().topology.blend_width - 0.1).abs() < 1e-6);
        core.set_rig_tilt(0.2);
        assert!((core.calibration().framing.tilt - 0.2).abs() < 1e-6);
        assert_eq!(core.resize(48, 26), Some((48, 26)));
        assert_eq!(core.output_dims(), (48, 26));

        // Coverage clamp works CPU-side (pure geometry).
        let clamped = core.safe_clamp(ViewportPosition {
            yaw: 10.0,
            pitch: 10.0,
            fov_degrees: Some(50.0),
        });
        assert!(clamped.yaw.is_finite() && clamped.pitch.is_finite());
        assert!(
            clamped.yaw.abs() < 10.0,
            "clamp actually constrained the pose"
        );

        // The YUV submit path works on the CPU arm - synchronously
        // (no warmup), at the resized output dimensions.
        let y = vec![0u8; (w * h) as usize];
        let uv = vec![128u8; (w * h / 4) as usize];
        let planes = YuvPlanes {
            y: &y,
            u: &uv,
            v: &uv,
        };
        match core.submit_frame_yuv(&planes, &planes) {
            Ok(crate::core::types::RenderOutcome::Rgba(bytes)) => {
                assert_eq!(bytes.len(), 48 * 26 * 4);
            }
            Ok(crate::core::types::RenderOutcome::Warmup) => {
                panic!("CPU submit is synchronous - no warmup")
            }
            Err(e) => panic!("cpu submit failed: {e}"),
        }
        assert_eq!(core.frame_count(), 1);

        // Genuinely GPU-only paths stay typed errors, not panics.
        assert!(matches!(
            core.flush().unwrap_err(),
            StitchCoreError::RequiresGpu
        ));
        assert!(matches!(
            core.render_yuv_at_pose(&planes, &planes, 0.0, 0.0)
                .unwrap_err(),
            StitchCoreError::RequiresGpu
        ));
    }

    /// Engine-level executor agreement: the same submit API
    /// (`submit_frame_nv12`) driven over a CPU engine and a GPU engine
    /// produces the same frame within the established oracle bounds.
    /// The Step-12 guarantee: swapping the executor does not change
    /// the picture.
    #[test]
    #[cfg(feature = "gpu")]
    fn cpu_and_gpu_engines_agree_via_submit_nv12() {
        use crate::core::StitchCore;
        use crate::core::types::RenderOutcome;
        use crate::render::planes::Nv12Planes;
        use crate::render::renderer::InputFormat;
        use crate::render::viewport::ViewportConfig;
        use crate::stitch::test_support::{Agreement, AgreementBounds, calib, gpu_or_skip, nv12};
        use crate::stitch::{CpuExecutor, Executor, GpuExecutor, GpuExecutorConfig};

        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let (cam_w, cam_h) = (192u32, 108u32);
        let (out_w, out_h) = (160u32, 90u32);
        let config = ViewportConfig {
            width: out_w,
            height: out_h,
            ..Default::default()
        };
        let (ly, luv) = nv12(cam_w, cam_h, 0);
        let (ry, ruv) = nv12(cam_w, cam_h, 30);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };

        let cpu_exec = CpuExecutor::new(
            Box::new(crate::projection::LShapeProjection),
            calib(cam_w, cam_h),
            config.clone(),
            cam_w,
            cam_h,
            false,
        )
        .expect("cpu executor");
        let mut cpu_core = StitchCore::new(Executor::Cpu(Box::new(cpu_exec))).expect("cpu engine");

        let gpu_exec = GpuExecutor::new(
            gpu,
            GpuExecutorConfig {
                viewport: config,
                ..GpuExecutorConfig::new(calib(cam_w, cam_h), cam_w, cam_h, InputFormat::Nv12)
            },
        )
        .expect("gpu executor");
        let mut gpu_core = StitchCore::new(Executor::Gpu(Box::new(gpu_exec))).expect("gpu engine");

        // CPU: synchronous - the first submit yields the frame.
        let cpu_rgba = match cpu_core
            .submit_frame_nv12(&left, &right)
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
                .submit_frame_nv12(&left, &right)
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
