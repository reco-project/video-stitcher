//! Panner trait — the camera-motion half of the tracker/panner split.
//!
//! A `Panner` consumes a clean `WorldState` (produced by one or
//! more `Tracker` instances from [`super::tracker`]) and returns a
//! [`WorldPose`] for the virtual
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

use super::mapped_detection::MappedDetection;
use super::pipeline_event::{PipelineEvent, PipelineEventSink};
use super::tracker::{Tracker, WorldState};
use crate::calibration::Calibration;
use crate::geometry::WorldPose;

/// Per-frame context a [`Panner`] receives alongside the world state.
///
/// The context carries timing plus a borrow of the current
/// calibration for calibrated source mapping and coverage queries.
/// It does not include raw detections; panners consume tracked world state.
#[derive(Debug)]
pub struct PanContext<'a> {
    /// Current frame index (0-based), monotonically increasing.
    pub frame_index: u64,
    /// Elapsed milliseconds since the start of processing.
    pub timestamp_ms: f64,
    /// Stored world-space request, initially the default pose. In lookahead mode this can be the smoothed request for an earlier rendered frame, not the immediately preceding panner decision. It is never the coverage-constrained render pose.
    pub previous_pose: WorldPose,
    /// Shared calibration for source-to-world mapping and coverage queries.
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
/// - The returned [`WorldPose`]
///   is NOT yet clamped to the coverage boundary; the session applies
///   clamping after the panner returns. Panners should produce their
///   geometric preference and let the coverage math enforce reachability.
pub trait Panner: Send {
    /// Decide where the virtual camera should look this frame.
    fn decide(&mut self, world: &WorldState, ctx: &PanContext<'_>) -> WorldPose;

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
    ) -> WorldPose {
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

/// Tracking statistics from one tracker/panner update.
pub(crate) struct DispatchResult {
    pub active_tracks: u32,
    pub ball_present: bool,
}

/// Run trackers only (no panner). Returns the WorldState for buffering.
pub(crate) fn run_trackers(
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

/// Update trackers and the world pose request when a panner is attached. Without a panner, returns `None` without running trackers. Emits tracked state before the decision and the requested pose afterward.
pub(crate) fn dispatch(
    panner: Option<&mut Box<dyn Panner>>,
    player_tracker: Option<&mut Box<dyn Tracker>>,
    ball_tracker: Option<&mut Box<dyn Tracker>>,
    requested_world_pose: &mut WorldPose,
    mut event_sink: Option<&mut (dyn PipelineEventSink + '_)>,
    ctx: DispatchContext<'_>,
) -> Option<DispatchResult> {
    let panner = panner?;
    // Build the WorldState via the shared tracker-run path (same as the
    // lookahead produce phase) so the two can never silently diverge.
    let world = run_trackers(player_tracker, ball_tracker, ctx);

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
        previous_pose: *requested_world_pose,
        calibration: ctx.calibration,
    };
    let active_tracks = world.players.len() as u32;
    let ball_present = world
        .ball
        .as_ref()
        .is_some_and(|b| !matches!(b.state, super::tracker::TrackState::Lost));

    // The lookahead-aware path does not come through here: the
    // buffered loop calls `StitchCore::decide_pose_with_lookahead`
    // with the real future window. This immediate-mode dispatch has
    // no future frames by construction.
    let pose = panner.decide_with_lookahead(&world, &[], &pan_ctx);
    *requested_world_pose = pose;

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
        active_tracks,
        ball_present,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{Calibration, Framing, Lens};
    use crate::detect::tracker::{TrackState, TrackedEntity, WorldState};

    use crate::projection::LShape;

    /// A fixture calibration shaped like the v1 test JSON without
    /// needing disk access or real lens data.
    fn test_calibration() -> Calibration {
        let cam = || Lens::fisheye(1920, 1080, 900.0, 900.0, 960.0, 540.0, [0.0; 4]);
        Calibration::new(
            vec![cam(), cam()],
            LShape {
                intersect: 0.54,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            Framing {
                axis_offset: 0.24,
                tilt: 0.0,
                roll: 0.0,
            },
        )
    }

    /// Minimal panner that echoes the ball's yaw/pitch when present.
    struct EchoPanner;
    impl Panner for EchoPanner {
        fn decide(&mut self, world: &WorldState, _ctx: &PanContext<'_>) -> WorldPose {
            match world.ball {
                Some(b) => WorldPose {
                    yaw: b.yaw,
                    pitch: b.pitch,
                    ..Default::default()
                },
                None => WorldPose::default(),
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
                origin: 0,
            }),
            players: vec![],
        };
        let ctx = PanContext {
            frame_index: 0,
            timestamp_ms: 0.0,
            previous_pose: WorldPose::default(),
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
            previous_pose: WorldPose::default(),
            calibration: &cal,
        };
        let out = p.decide(&world, &ctx);
        assert_eq!(out.yaw, 0.0);
        assert_eq!(out.pitch, 0.0);
    }
}
