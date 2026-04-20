//! Panner trait — the camera-motion half of the tracker/panner split.
//!
//! A `Panner` consumes a clean `WorldState` (produced by one or
//! more `Tracker` instances from [`crate::tracker`]) and returns a
//! `ViewportPosition` (from [`crate::director`]) for the virtual
//! camera. It knows nothing
//! about raw detections, plausibility gates, or identity management —
//! those are tracker concerns. Its sole job is "given where things
//! *are*, where should the camera *look*?"
//!
//! Typical implementations (shipped in `reco-autocam`):
//! - `BallPanner` — follows `world.ball` while tracking, holds on
//!   coast, recenters on lost.
//! - `FieldPanner` — blends ball with the `world.players` cluster
//!   centroid, widening FOV when the action spreads.
//! - `SweepPanner` — debug-only, ignores world state and slowly
//!   pans left-right within coverage bounds.
//!
//! Future (not part of this module's contract):
//! - `ReplayPanner` — follows a specific player by ID.
//! - `BroadcastPanner` — learned policy over world state.
//!
//! # Composition
//!
//! Panners compose via decorators in `reco-autocam`:
//!
//! ```text
//! BallPanner → Smoother → Anticipator → DeadZone
//! ```
//!
//! Each decorator is itself a `Panner` wrapping an inner `Panner`.
//! Smoothing, anticipation, and dead-zone handling all live at this
//! layer because they transform camera *motion*, not detections.

use crate::calibration::MatchCalibration;
use crate::director::ViewportPosition;
use crate::tracker::WorldState;

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
    pub calibration: &'a MatchCalibration,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{CameraParams, MatchCalibration, PlaneLayout};
    use crate::detector::CameraId;
    use crate::tracker::{TrackState, TrackedEntity, WorldState};

    /// A fixture calibration shaped like the v1 test JSON without
    /// needing disk access or real lens data.
    fn test_calibration() -> MatchCalibration {
        MatchCalibration {
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
                velocity: None,
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
