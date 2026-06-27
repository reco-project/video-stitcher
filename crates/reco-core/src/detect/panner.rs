//! Panner trait — the camera-motion half of the tracker/panner split.
//!
//! A `Panner` consumes a clean `WorldState` (produced by one or
//! more `Tracker` instances from [`super::tracker`]) and returns a
//! `ViewportPosition` (from [`super::director`]) for the virtual
//! camera. It knows nothing
//! about raw detections, plausibility gates, or identity management —
//! those are tracker concerns. Its sole job is "given where things
//! *are*, where should the camera *look*?"
//!
//! Implementations (shipped in `reco-autocam`):
//! - `FieldPanner` - the production player+ball panner: trimmed-robust
//!   cluster centroid, ball-only follow when no cluster, dynamic FOV,
//!   ball-presence hysteresis, velocity-clamped chase. Lookahead is not
//!   a separate panner - the buffered run loop centered-smooths this
//!   panner's pose stream over past + future frames.
//! - `SweepPanner` - debug-only, ignores world state and slowly pans
//!   left-right within coverage bounds.
//! - `FilePanner` - replays a precomputed pose trajectory from CSV.

use super::director::{MappedDetection, ViewportPosition};
use super::pipeline_event::{PipelineEvent, PipelineEventSink};
use super::tracker::{Tracker, WorldState};
use crate::calibration::Calibration;

/// Per-frame context a [`Panner`] receives alongside the world state.
///
/// The context carries timing plus a borrow of the current
/// calibration, so panners can project between camera and panorama
/// coordinates if needed (e.g. for coverage-aware edge handling).
/// It does NOT include raw detections — those never reach a panner.
#[derive(Debug)]
pub struct PanContext<'a> {
    /// Current frame index (0-based), monotonically increasing.
    pub frame_index: u64,
    /// Elapsed milliseconds since the start of processing.
    pub timestamp_ms: f64,
    /// The viewport position the session reported on the *previous*
    /// frame (after clamping and smoothing), or the session default
    /// if this is the first call. Panners use this to compute
    /// first-order motion deltas without needing their own state.
    pub previous_position: ViewportPosition,
    /// Shared calibration for optional camera↔panorama projection.
    /// Borrowed for the duration of the [`decide`](Panner::decide)
    /// call; panners must not retain it.
    pub calibration: &'a Calibration,
}

/// The contract implemented by every camera-motion policy.
///
/// Implementations must be **stateful over time** (to smooth motion,
/// apply dead-zones, anticipate trajectories) but **pure per call**
/// with respect to their inputs — i.e. `decide(&world, &ctx)` must
/// not mutate `world` or `ctx`, and repeated calls with identical
/// inputs may return different outputs only because of internal
/// state evolution.
///
/// # Invariants
///
/// - [`decide`](Self::decide) is called once per frame, in order.
/// - The returned [`ViewportPosition`]
///   is NOT yet clamped to the coverage boundary; the session applies
///   clamping after the panner returns. Panners should produce their
///   geometric preference and let the coverage math enforce reachability.
pub trait Panner: Send {
    /// Decide where the virtual camera should look this frame.
    fn decide(&mut self, world: &WorldState, ctx: &PanContext<'_>) -> ViewportPosition;

    /// Decide with access to future WorldStates from the lookahead buffer.
    ///
    /// `future` contains WorldStates for frames after the current one,
    /// ordered nearest-to-farthest. Empty when lookahead is disabled.
    /// Default delegates to [`decide`](Self::decide), ignoring the future.
    fn decide_with_lookahead(
        &mut self,
        world: &WorldState,
        future: &[WorldState],
        ctx: &PanContext<'_>,
    ) -> ViewportPosition {
        let _ = future;
        self.decide(world, ctx)
    }

    /// Optional debug snapshot from the last `decide()` call.
    fn debug_event(&self, _frame_index: u64) -> Option<PipelineEvent> {
        None
    }
}

/// Scalar inputs to [`dispatch`] bundled so the function stays under
/// `clippy::too_many_arguments`. The mutable pose + the three
/// `Option<&mut Box<dyn …>>` slots are the only moving parts per
/// caller; everything else fits here.
#[derive(Clone, Copy)]
pub(crate) struct DispatchContext<'a> {
    /// Raw mapped detections the trackers should consume this frame.
    pub detections: &'a [MappedDetection],
    /// Shared calibration handed to the panner via [`PanContext`].
    pub calibration: &'a Calibration,
    /// Current frame index (0-based, monotonically increasing).
    pub frame_index: u64,
    /// Elapsed milliseconds since session start.
    pub timestamp_ms: f64,
    /// Short label used only for the >1-ball warning so log output
    /// still says which caller ran the dispatch.
    pub caller: &'static str,
}

/// Run the shared tracker → panner dispatch one frame's worth.
///
/// Both `StitchCore::resolve_current_pose` and
/// `StitchSession::fire_sink_and_update_director` used to inline this
/// ~50-line algorithm. They still differ on what they do around the
/// dispatch (clamp, fire sinks, return vs. store), but the dispatch
/// itself is identical:
///
/// 1. Build an empty [`WorldState`].
/// 2. Run the player tracker, store into `world.players`.
/// 3. Let the ball tracker `observe_world` (so it sees this frame's
///    players for anchor gating), then `update`; take the first
///    entity (warn if more than one) into `world.ball`.
/// 4. Build a [`PanContext`] carrying the caller's previous pose.
/// 5. Ask the panner to `decide`; update `previous_panner_pose` in
///    place; return the decided pose.
///
/// Returns `None` when no panner is attached.
///
/// When a [`PipelineEventSink`] is supplied via `event_sink`, emits a
/// [`PipelineEvent::WorldState`] right before `panner.decide` and a
/// [`PipelineEvent::PanDecision`] right after. Both sites are part
/// of the Step 6 trace vocabulary.
pub(crate) struct DispatchResult {
    pub pose: ViewportPosition,
    pub world_state: WorldState,
    pub active_tracks: u32,
    pub ball_present: bool,
}

/// Run trackers only (no panner). Returns the WorldState for buffering.
pub(crate) fn dispatch_detect_only(
    player_tracker: Option<&mut Box<dyn Tracker>>,
    ball_tracker: Option<&mut Box<dyn Tracker>>,
    ctx: DispatchContext<'_>,
) -> WorldState {
    let mut world = WorldState::default();

    // Order matters: players first, then ball. The ball tracker's
    // `observe_world` sees the just-computed player positions so a
    // player-anchor gate can run against the current frame rather
    // than the previous one.
    if let Some(t) = player_tracker {
        world.players = t.update(ctx.detections, ctx.timestamp_ms);
    }
    if let Some(t) = ball_tracker {
        t.observe_world(&world);
        let ents = t.update(ctx.detections, ctx.timestamp_ms);
        if ents.len() > 1 {
            log::warn!(
                "{}: ball_tracker returned {} entities (expected <=1); taking first",
                ctx.caller,
                ents.len()
            );
        }
        world.ball = ents.into_iter().next();
    }

    world
}

pub(crate) fn dispatch(
    panner: Option<&mut Box<dyn Panner>>,
    player_tracker: Option<&mut Box<dyn Tracker>>,
    ball_tracker: Option<&mut Box<dyn Tracker>>,
    previous_panner_pose: &mut ViewportPosition,
    mut event_sink: Option<&mut (dyn PipelineEventSink + '_)>,
    future_world_states: &[WorldState],
    ctx: DispatchContext<'_>,
) -> Option<DispatchResult> {
    let panner = panner?;
    // Build the WorldState via the shared tracker-run path (same as the
    // lookahead produce phase) so the two can never silently diverge.
    let world = dispatch_detect_only(player_tracker, ball_tracker, ctx);

    // Trace: WorldState (only pays for the clone when a sink exists).
    if let Some(sink) = event_sink.as_mut() {
        sink.emit(PipelineEvent::WorldState {
            frame_index: ctx.frame_index,
            timestamp_ms: ctx.timestamp_ms,
            players: world.players.clone(),
            ball: world.ball,
        });
    }

    let pan_ctx = PanContext {
        frame_index: ctx.frame_index,
        timestamp_ms: ctx.timestamp_ms,
        previous_position: *previous_panner_pose,
        calibration: ctx.calibration,
    };
    let active_tracks = world.players.len() as u32;
    let ball_present = world
        .ball
        .as_ref()
        .is_some_and(|b| !matches!(b.state, super::tracker::TrackState::Lost));

    let pose = panner.decide_with_lookahead(&world, future_world_states, &pan_ctx);
    *previous_panner_pose = pose;

    if let Some(sink) = event_sink.as_mut() {
        sink.emit(PipelineEvent::PanDecision {
            frame_index: ctx.frame_index,
            pose,
        });
        if let Some(debug) = panner.debug_event(ctx.frame_index) {
            sink.emit(debug);
        }
    }

    Some(DispatchResult {
        pose,
        world_state: world,
        active_tracks,
        ball_present,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{Calibration, CameraParams, PlaneLayout};
    use crate::detect::detector::CameraId;
    use crate::detect::tracker::{TrackState, TrackedEntity, WorldState};

    /// A fixture calibration shaped like the v1 test JSON without
    /// needing disk access or real lens data.
    fn test_calibration() -> Calibration {
        Calibration {
            left: CameraParams {
                width: 1920,
                height: 1080,
                fx: 900.0,
                fy: 900.0,
                cx: 960.0,
                cy: 540.0,
                d: [0.0; 4],
            },
            right: CameraParams {
                width: 1920,
                height: 1080,
                fx: 900.0,
                fy: 900.0,
                cx: 960.0,
                cy: 540.0,
                d: [0.0; 4],
            },
            layout: PlaneLayout {
                camera_axis_offset: 0.24,
                intersect: 0.54,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
            },
            rig_tilt: 0.0,
            rig_roll: 0.0,
            sync_offset: 0,
            field_roi: None,
            lens_correction_amount: 1.0,
            blend_width: 0.05,
        }
    }

    /// Minimal panner that echoes the ball's yaw/pitch when present.
    struct EchoPanner;
    impl Panner for EchoPanner {
        fn decide(&mut self, world: &WorldState, _ctx: &PanContext<'_>) -> ViewportPosition {
            match world.ball {
                Some(b) => ViewportPosition {
                    yaw: b.yaw,
                    pitch: b.pitch,
                    fov_degrees: None,
                },
                None => ViewportPosition::default(),
            }
        }
    }

    #[test]
    fn echo_panner_follows_ball() {
        let cal = test_calibration();
        let mut p: Box<dyn Panner> = Box::new(EchoPanner);
        let world = WorldState {
            ball: Some(TrackedEntity {
                id: 0,
                class_id: 0,
                yaw: 0.3,
                pitch: -0.05,
                confidence: 0.9,
                state: TrackState::Tracking,
                age_frames: 5,
                origin: CameraId::Left,
            }),
            players: vec![],
        };
        let ctx = PanContext {
            frame_index: 0,
            timestamp_ms: 0.0,
            previous_position: ViewportPosition::default(),
            calibration: &cal,
        };
        let out = p.decide(&world, &ctx);
        assert_eq!(out.yaw, 0.3);
        assert_eq!(out.pitch, -0.05);
    }

    #[test]
    fn echo_panner_defaults_without_ball() {
        let cal = test_calibration();
        let mut p = EchoPanner;
        let world = WorldState::default();
        let ctx = PanContext {
            frame_index: 0,
            timestamp_ms: 0.0,
            previous_position: ViewportPosition::default(),
            calibration: &cal,
        };
        let out = p.decide(&world, &ctx);
        assert_eq!(out.yaw, 0.0);
        assert_eq!(out.pitch, 0.0);
    }
}
