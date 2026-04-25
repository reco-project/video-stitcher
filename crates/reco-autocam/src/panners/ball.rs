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

/// Maximum angular velocity in radians per second.
const MAX_VELOCITY_RAD_PER_SEC: f32 = 0.12;
/// Velocity smoothing alpha.
const VELOCITY_ALPHA: f32 = 0.04;
/// FOV EMA alpha.
const FOV_ALPHA: f32 = 0.02;

/// Ball-following panner with velocity-based smooth motion.
///
/// Instead of snapping to the ball position, moves at a capped
/// angular velocity toward it. This eliminates jitter from noisy
/// ball detections without needing external Smoother/DeadZone
/// decorators.
pub struct BallPanner {
    fov_wide_deg: f32,
    fov_tight_deg: f32,
    pitch_near: f32,
    pitch_far: f32,
    last: ViewportPosition,
    velocity_yaw: f32,
    velocity_pitch: f32,
    current_fov: f32,
    max_velocity: f32,
}

impl BallPanner {
    /// Build a ball panner with defaults (matches the old
    /// `BallDirector` FOV envelope).
    pub fn new(fps: f32) -> Self {
        let fps = fps.clamp(1.0, 1000.0);
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
            velocity_yaw: 0.0,
            velocity_pitch: 0.0,
            current_fov: DEFAULT_IDLE_FOV_DEG,
            max_velocity: MAX_VELOCITY_RAD_PER_SEC / fps,
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
        Self::new(30.0)
    }
}

impl Panner for BallPanner {
    fn decide(&mut self, world: &WorldState, _ctx: &PanContext<'_>) -> ViewportPosition {
        let target = match world.ball {
            Some(ref ball) if !matches!(ball.state, TrackState::Lost) => {
                Some((ball.yaw, ball.pitch, self.target_fov(ball.pitch)))
            }
            _ => None,
        };

        if let Some((ty, tp, tf)) = target {
            let err_yaw = ty - self.last.yaw;
            let err_pitch = tp - self.last.pitch;

            let desired_yaw = err_yaw.clamp(-self.max_velocity, self.max_velocity);
            let desired_pitch = err_pitch.clamp(-self.max_velocity, self.max_velocity);

            self.velocity_yaw += VELOCITY_ALPHA * (desired_yaw - self.velocity_yaw);
            self.velocity_pitch += VELOCITY_ALPHA * (desired_pitch - self.velocity_pitch);

            self.last.yaw += self.velocity_yaw;
            self.last.pitch += self.velocity_pitch;
            self.current_fov += FOV_ALPHA * (tf - self.current_fov);
            self.last.fov_degrees = Some(self.current_fov);
        } else {
            // Lost or absent: decay velocity to zero, hold position.
            self.velocity_yaw *= 0.9;
            self.velocity_pitch *= 0.9;
            self.last.yaw += self.velocity_yaw;
            self.last.pitch += self.velocity_pitch;
        }

        self.last
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
            confidence: 0.8,
            state,
            age_frames: 3,
            origin: CameraId::Left,
        }
    }

    #[test]
    fn default_panner_starts_at_origin_with_idle_fov() {
        let mut p = BallPanner::new(30.0);
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
    fn tracking_converges_to_ball() {
        let mut p = BallPanner::new(30.0);
        let cal = test_cal();
        let world = WorldState {
            ball: Some(ball_at(0.4, 0.05, TrackState::Tracking)),
            players: vec![],
        };
        let mut out = ViewportPosition::default();
        for _ in 0..300 {
            out = p.decide(&world, &ctx(&cal, ViewportPosition::default()));
        }
        assert!(
            (out.yaw - 0.4).abs() < 0.02,
            "expected ~0.4, got {}",
            out.yaw
        );
    }

    #[test]
    fn coasting_still_moves_toward_ball() {
        let mut p = BallPanner::new(30.0);
        let cal = test_cal();
        let world = WorldState {
            ball: Some(ball_at(0.1, 0.0, TrackState::Coasting)),
            players: vec![],
        };
        let out = p.decide(&world, &ctx(&cal, ViewportPosition::default()));
        assert!(out.yaw > 0.0, "should move toward ball");
    }

    #[test]
    fn lost_does_not_jump() {
        let mut p = BallPanner::new(30.0);
        let cal = test_cal();
        let track_world = WorldState {
            ball: Some(ball_at(0.3, 0.05, TrackState::Tracking)),
            players: vec![],
        };
        for _ in 0..200 {
            p.decide(&track_world, &ctx(&cal, ViewportPosition::default()));
        }
        let before = p.last.yaw;
        let out = p.decide(
            &WorldState {
                ball: Some(ball_at(-0.9, -0.9, TrackState::Lost)),
                players: vec![],
            },
            &ctx(&cal, ViewportPosition::default()),
        );
        assert!(
            (out.yaw - before).abs() < 0.01,
            "lost should not jump: before={before}, after={}",
            out.yaw
        );
    }

    #[test]
    fn no_ball_in_world_holds_roughly() {
        let mut p = BallPanner::new(30.0);
        let cal = test_cal();
        for _ in 0..200 {
            p.decide(
                &WorldState {
                    ball: Some(ball_at(0.2, 0.0, TrackState::Tracking)),
                    players: vec![],
                },
                &ctx(&cal, ViewportPosition::default()),
            );
        }
        let before = p.last.yaw;
        for _ in 0..10 {
            p.decide(
                &WorldState::default(),
                &ctx(&cal, ViewportPosition::default()),
            );
        }
        assert!(
            (p.last.yaw - before).abs() < 0.02,
            "should hold roughly: before={before}, after={}",
            p.last.yaw
        );
    }

    #[test]
    fn fov_zooms_in_at_high_pitch() {
        let p = BallPanner::new(30.0);
        let low = p.target_fov(DEFAULT_PITCH_NEAR);
        let high = p.target_fov(DEFAULT_PITCH_FAR);
        assert!(low > high, "low={low} high={high}");
        assert!((low - DEFAULT_FOV_WIDE_DEG).abs() < 1e-3);
        assert!((high - DEFAULT_FOV_TIGHT_DEG).abs() < 1e-3);
    }

    #[test]
    fn with_fov_overrides_envelope() {
        let p = BallPanner::new(30.0).with_fov(80.0, 20.0, 0.0, 0.5);
        let fov = p.target_fov(0.0);
        assert!((fov - 80.0).abs() < 1e-3);
    }
}
