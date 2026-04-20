//! Ball-following panner — camera-motion policy consuming
//! `world.ball`.
//!
//! [`BallPanner`] is intentionally tiny: all the detection selection,
//! plausibility checking, search-state machine, and cross-camera
//! handoff logic that used to live in `BallDirector` now live in
//! [`crate::trackers::BallTracker`]. By the time `decide()` is
//! called, the world state carries at most one ball with a clean
//! tri-valued [`TrackState`].
//!
//! The panner's job is therefore just:
//!
//! - While the ball is [`TrackState::Tracking`] or
//!   [`TrackState::Coasting`]: snap the virtual camera to the ball's
//!   yaw/pitch and compute dynamic FOV based on ball pitch
//!   (lower pitch = ball near = wider view; higher pitch = ball
//!   far = zoom in).
//! - While [`TrackState::Lost`] or no ball in world: hold the last
//!   published position (camera does not lurch during brief losses).
//!
//! All smoothing, anticipation, and dead-zone logic live as
//! separate decorator panners ([`crate::panners::Smoother`],
//! [`crate::panners::Anticipator`], [`crate::panners::DeadZone`])
//! composed around this one.

use reco_core::director::ViewportPosition;
use reco_core::panner::{PanContext, Panner};
use reco_core::tracker::{TrackState, WorldState};

/// Default wide FOV (degrees) when the ball is close to the camera
/// (low pitch).
pub const DEFAULT_FOV_WIDE_DEG: f32 = 65.0;
/// Default tight FOV (degrees) when the ball is far (high pitch).
pub const DEFAULT_FOV_TIGHT_DEG: f32 = 35.0;
/// Default pitch (radians) at which FOV reaches full wide.
pub const DEFAULT_PITCH_NEAR: f32 = -0.05;
/// Default pitch (radians) at which FOV reaches full tight.
pub const DEFAULT_PITCH_FAR: f32 = 0.20;
/// Default starting FOV when no ball has ever been seen.
pub const DEFAULT_IDLE_FOV_DEG: f32 = 55.0;

/// A minimal, stateful [`Panner`] that follows a tracked ball.
///
/// Holds only enough state to preserve the last-published position
/// across frames where the ball is [`TrackState::Lost`] or absent.
/// Decorators (smoothing, anticipation, dead-zone) wrap this and
/// add their own state separately.
pub struct BallPanner {
    fov_wide_deg: f32,
    fov_tight_deg: f32,
    pitch_near: f32,
    pitch_far: f32,
    last: ViewportPosition,
}

impl BallPanner {
    /// Build a ball panner with defaults (matches the old
    /// `BallDirector` FOV envelope).
    pub fn new() -> Self {
        Self {
            fov_wide_deg: DEFAULT_FOV_WIDE_DEG,
            fov_tight_deg: DEFAULT_FOV_TIGHT_DEG,
            pitch_near: DEFAULT_PITCH_NEAR,
            pitch_far: DEFAULT_PITCH_FAR,
            last: ViewportPosition {
                yaw: 0.0,
                pitch: 0.0,
                fov_degrees: Some(DEFAULT_IDLE_FOV_DEG),
            },
        }
    }

    /// Override the FOV envelope.
    ///
    /// Wide is the outer (larger) FOV at `pitch_near`; tight is the
    /// inner (smaller) FOV at `pitch_far`. Pitch values in radians.
    pub fn with_fov(
        mut self,
        wide_deg: f32,
        tight_deg: f32,
        pitch_near: f32,
        pitch_far: f32,
    ) -> Self {
        self.fov_wide_deg = wide_deg;
        self.fov_tight_deg = tight_deg;
        self.pitch_near = pitch_near;
        self.pitch_far = pitch_far;
        self
    }

    /// Compute target FOV based on ball pitch. Clamped to the
    /// configured envelope.
    fn target_fov(&self, pitch: f32) -> f32 {
        let span = self.pitch_far - self.pitch_near;
        let t = if span.abs() < 1e-6 {
            0.5
        } else {
            ((pitch - self.pitch_near) / span).clamp(0.0, 1.0)
        };
        self.fov_wide_deg + t * (self.fov_tight_deg - self.fov_wide_deg)
    }
}

impl Default for BallPanner {
    fn default() -> Self {
        Self::new()
    }
}

impl Panner for BallPanner {
    fn decide(&mut self, world: &WorldState, _ctx: &PanContext<'_>) -> ViewportPosition {
        match world.ball {
            Some(ball) => match ball.state {
                TrackState::Tracking | TrackState::Coasting => {
                    let pos = ViewportPosition {
                        yaw: ball.yaw,
                        pitch: ball.pitch,
                        fov_degrees: Some(self.target_fov(ball.pitch)),
                    };
                    self.last = pos;
                    pos
                }
                TrackState::Lost => {
                    // Track was just lost this frame — emit one
                    // held frame, then subsequent calls will see
                    // `world.ball = None` and keep holding.
                    self.last
                }
            },
            None => self.last,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::calibration::{CameraParams, MatchCalibration, PlaneLayout};
    use reco_core::detector::CameraId;
    use reco_core::tracker::TrackedEntity;

    fn test_cal() -> MatchCalibration {
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

    fn ctx<'a>(cal: &'a MatchCalibration, prev: ViewportPosition) -> PanContext<'a> {
        PanContext {
            frame_index: 0,
            timestamp_ms: 0.0,
            previous_position: prev,
            calibration: cal,
        }
    }

    fn ball_at(yaw: f32, pitch: f32, state: TrackState) -> TrackedEntity {
        TrackedEntity {
            id: 0,
            class_id: 0,
            yaw,
            pitch,
            velocity: None,
            confidence: 0.8,
            state,
            age_frames: 3,
            origin: CameraId::Left,
        }
    }

    #[test]
    fn default_panner_starts_at_origin_with_idle_fov() {
        let mut p = BallPanner::new();
        let cal = test_cal();
        let out = p.decide(
            &WorldState::default(),
            &ctx(&cal, ViewportPosition::default()),
        );
        assert_eq!(out.yaw, 0.0);
        assert_eq!(out.pitch, 0.0);
        assert_eq!(out.fov_degrees, Some(DEFAULT_IDLE_FOV_DEG));
    }

    #[test]
    fn tracking_snaps_to_ball() {
        let mut p = BallPanner::new();
        let cal = test_cal();
        let world = WorldState {
            ball: Some(ball_at(0.4, 0.05, TrackState::Tracking)),
            players: vec![],
        };
        let out = p.decide(&world, &ctx(&cal, ViewportPosition::default()));
        assert!((out.yaw - 0.4).abs() < 1e-6);
        assert!((out.pitch - 0.05).abs() < 1e-6);
    }

    #[test]
    fn coasting_still_publishes_ball_position() {
        let mut p = BallPanner::new();
        let cal = test_cal();
        let world = WorldState {
            ball: Some(ball_at(0.1, 0.0, TrackState::Coasting)),
            players: vec![],
        };
        let out = p.decide(&world, &ctx(&cal, ViewportPosition::default()));
        assert!((out.yaw - 0.1).abs() < 1e-6);
    }

    #[test]
    fn lost_holds_last_position() {
        let mut p = BallPanner::new();
        let cal = test_cal();
        // Track to (0.3, 0.05).
        p.decide(
            &WorldState {
                ball: Some(ball_at(0.3, 0.05, TrackState::Tracking)),
                players: vec![],
            },
            &ctx(&cal, ViewportPosition::default()),
        );
        // Ball goes lost.
        let out = p.decide(
            &WorldState {
                ball: Some(ball_at(-0.9, -0.9, TrackState::Lost)),
                players: vec![],
            },
            &ctx(&cal, ViewportPosition::default()),
        );
        // Held the previous position — not the Lost entity's fields.
        assert!((out.yaw - 0.3).abs() < 1e-6);
    }

    #[test]
    fn no_ball_in_world_holds() {
        let mut p = BallPanner::new();
        let cal = test_cal();
        p.decide(
            &WorldState {
                ball: Some(ball_at(0.2, 0.0, TrackState::Tracking)),
                players: vec![],
            },
            &ctx(&cal, ViewportPosition::default()),
        );
        let out = p.decide(
            &WorldState::default(),
            &ctx(&cal, ViewportPosition::default()),
        );
        assert!((out.yaw - 0.2).abs() < 1e-6);
    }

    #[test]
    fn fov_zooms_in_at_high_pitch() {
        let mut p = BallPanner::new();
        let cal = test_cal();
        let low = p.decide(
            &WorldState {
                ball: Some(ball_at(0.0, DEFAULT_PITCH_NEAR, TrackState::Tracking)),
                players: vec![],
            },
            &ctx(&cal, ViewportPosition::default()),
        );
        let high = p.decide(
            &WorldState {
                ball: Some(ball_at(0.0, DEFAULT_PITCH_FAR, TrackState::Tracking)),
                players: vec![],
            },
            &ctx(&cal, ViewportPosition::default()),
        );
        assert!(low.fov_degrees.unwrap() > high.fov_degrees.unwrap());
        assert!((low.fov_degrees.unwrap() - DEFAULT_FOV_WIDE_DEG).abs() < 1e-3);
        assert!((high.fov_degrees.unwrap() - DEFAULT_FOV_TIGHT_DEG).abs() < 1e-3);
    }

    #[test]
    fn with_fov_overrides_envelope() {
        let mut p = BallPanner::new().with_fov(80.0, 20.0, 0.0, 0.5);
        let cal = test_cal();
        let out = p.decide(
            &WorldState {
                ball: Some(ball_at(0.0, 0.0, TrackState::Tracking)),
                players: vec![],
            },
            &ctx(&cal, ViewportPosition::default()),
        );
        assert!((out.fov_degrees.unwrap() - 80.0).abs() < 1e-3);
    }
}
