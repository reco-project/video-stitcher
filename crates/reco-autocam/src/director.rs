//! Ball-tracking director with velocity prediction and dynamic FOV.
//!
//! Follows the ball using a 3-state machine:
//! - **Tracking**: ball is visible, camera smoothly follows it
//! - **Searching**: ball lost, camera coasts on last velocity then holds
//! - **Recovering**: ball reappeared after being lost, camera moves to it
//!
//! Uses exponential moving average (EMA) smoothing with velocity prediction
//! to produce smooth, continuous panning even when detection is intermittent.
//! Dynamic FOV zooms in when the ball is far from the camera (high pitch
//! values) and zooms out when action is near.

use reco_core::director::{Director, DirectorContext, ViewportPosition};

/// Director state machine modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Ball is being tracked. Camera follows with smoothing.
    Tracking,
    /// Ball was lost. Camera coasts on last velocity, then holds.
    Searching,
    /// Ball reappeared. Camera moves toward it (faster smoothing).
    Recovering,
}

/// Ball-tracking director with velocity prediction and dynamic FOV.
///
/// Pans the virtual camera to follow a detected ball using EMA smoothing,
/// velocity-based prediction for gap bridging, dead zone for jitter
/// suppression, and viewport bounds clamping to prevent black edges.
pub struct BallDirector {
    state: State,
    /// Current smoothed position (panorama-space radians).
    yaw: f32,
    pitch: f32,
    /// Target position (from latest detection).
    target_yaw: f32,
    target_pitch: f32,
    /// Smoothed velocity (radians per frame) for prediction during gaps.
    vel_yaw: f32,
    vel_pitch: f32,
    /// Previous target for velocity estimation.
    prev_target_yaw: f32,
    prev_target_pitch: f32,
    /// EMA smoothing factor for tracking (0 = no smoothing, 1 = instant).
    alpha_track: f32,
    /// EMA smoothing factor for recovery (faster than tracking).
    alpha_recover: f32,
    /// EMA smoothing factor for velocity estimation.
    alpha_velocity: f32,
    /// Dead zone as fraction of FOV. No panning if target is within this
    /// fraction of the current position.
    dead_zone: f32,
    /// Frames since ball was last seen.
    frames_without_ball: u32,
    /// How many frames to wait before transitioning to Searching.
    search_delay: u32,
    /// How many frames to coast on velocity while Searching before stopping.
    coast_frames: u32,
    /// How many frames of continuous tracking before leaving Recovering.
    recover_confirm: u32,
    /// Counter for recovery confirmation.
    recover_count: u32,
    /// Current dynamic FOV in degrees.
    current_fov: f32,
    /// Wide FOV (zoomed out) in degrees.
    fov_wide: f32,
    /// Tight FOV (zoomed in) in degrees.
    fov_tight: f32,
    /// EMA smoothing factor for FOV transitions.
    fov_alpha: f32,
    /// Label to track (e.g. "ball").
    target_label: String,
}

impl BallDirector {
    /// Create a new ball director with default parameters.
    ///
    /// `fps` is used to scale timing parameters (delay, recovery, coast)
    /// to the video frame rate. Clamped to `[1.0, 1000.0]` to prevent
    /// degenerate timing from invalid input.
    pub fn new(fps: f32) -> Self {
        let fps = fps.clamp(1.0, 1000.0);
        Self {
            state: State::Searching,
            yaw: 0.0,
            pitch: 0.0,
            target_yaw: 0.0,
            target_pitch: 0.0,
            vel_yaw: 0.0,
            vel_pitch: 0.0,
            prev_target_yaw: 0.0,
            prev_target_pitch: 0.0,
            alpha_track: 0.04,
            alpha_recover: 0.08,
            alpha_velocity: 0.2,
            dead_zone: 0.10,
            frames_without_ball: 0,
            search_delay: (fps * 1.5) as u32, // 1.5 seconds before giving up
            coast_frames: (fps * 2.0) as u32, // coast on velocity for 2s
            recover_confirm: (fps * 0.3) as u32, // 0.3 seconds to confirm
            recover_count: 0,
            current_fov: 55.0,
            fov_wide: 65.0,
            fov_tight: 25.0,
            fov_alpha: 0.02,
            target_label: "ball".into(),
        }
    }

    /// Set the EMA smoothing factor for normal tracking (default: 0.04).
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha_track = alpha.clamp(0.01, 1.0);
        self
    }

    /// Set the dead zone as a fraction of FOV (default: 0.10).
    pub fn with_dead_zone(mut self, dead_zone: f32) -> Self {
        self.dead_zone = dead_zone.clamp(0.0, 0.5);
        self
    }

    /// Set the target label to track (default: "ball").
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.target_label = label.into();
        self
    }

    /// Find the best ball detection from mapped detections.
    fn find_ball<'a>(
        &self,
        ctx: &'a DirectorContext<'_>,
    ) -> Option<&'a reco_core::director::MappedDetection> {
        ctx.detections
            .iter()
            .filter(|d| d.label == self.target_label && d.position.is_some())
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Update velocity estimate from a new target position.
    fn update_velocity(&mut self, new_yaw: f32, new_pitch: f32) {
        let dy = new_yaw - self.prev_target_yaw;
        let dp = new_pitch - self.prev_target_pitch;

        // Only update velocity if the jump is reasonable (not a teleport).
        if dy.abs() < 0.3 && dp.abs() < 0.3 {
            self.vel_yaw = self.vel_yaw * (1.0 - self.alpha_velocity) + dy * self.alpha_velocity;
            self.vel_pitch =
                self.vel_pitch * (1.0 - self.alpha_velocity) + dp * self.alpha_velocity;
        }

        self.prev_target_yaw = new_yaw;
        self.prev_target_pitch = new_pitch;
    }

    /// Apply EMA smoothing toward target, respecting dead zone.
    fn smooth_toward(&mut self, alpha: f32, fov_degrees: f32) {
        let dead_zone_rad = (self.dead_zone * fov_degrees).to_radians();

        let dy = self.target_yaw - self.yaw;
        let dp = self.target_pitch - self.pitch;

        if dy.abs() > dead_zone_rad {
            self.yaw += alpha * dy;
        }
        if dp.abs() > dead_zone_rad {
            self.pitch += alpha * dp;
        }
    }

    /// Coast: advance target using velocity prediction.
    fn coast(&mut self) {
        self.target_yaw += self.vel_yaw;
        self.target_pitch += self.vel_pitch;
        // Decay velocity while coasting so we don't overshoot.
        self.vel_yaw *= 0.95;
        self.vel_pitch *= 0.95;
    }

    /// Compute target FOV based on ball's panorama-space pitch.
    ///
    /// Higher pitch (ball further up the field) -> zoom in tighter.
    /// Lower pitch (ball near the camera sideline) -> zoom out wider.
    fn target_fov(&self) -> f32 {
        // Use the current tracked pitch as the distance proxy.
        // Pitch range for a typical football field: ~-0.05 (near) to ~0.25 (far).
        // Map pitch from [pitch_near, pitch_far] to [fov_wide, fov_tight].
        let pitch_near: f32 = -0.05;
        let pitch_far: f32 = 0.20;
        let t = ((self.target_pitch - pitch_near) / (pitch_far - pitch_near)).clamp(0.0, 1.0);
        self.fov_wide + t * (self.fov_tight - self.fov_wide)
    }

    /// Clamp position to viewport bounds.
    fn clamp_to_bounds(&mut self, ctx: &DirectorContext<'_>) {
        self.yaw = self
            .yaw
            .clamp(ctx.viewport_bounds.min_yaw, ctx.viewport_bounds.max_yaw);
        self.pitch = self
            .pitch
            .clamp(ctx.viewport_bounds.min_pitch, ctx.viewport_bounds.max_pitch);
    }
}

impl Director for BallDirector {
    fn update(&mut self, ctx: &DirectorContext<'_>) {
        reco_core::profile_scope!("ball_director_update");

        let ball = self.find_ball(ctx);

        match (self.state, ball) {
            // Tracking + ball visible: follow it.
            (State::Tracking, Some(obj)) => {
                let pos = obj.position.unwrap();
                self.update_velocity(pos.yaw, pos.pitch);
                self.target_yaw = pos.yaw;
                self.target_pitch = pos.pitch;
                self.smooth_toward(self.alpha_track, ctx.current_fov);
                self.frames_without_ball = 0;
            }

            // Tracking + ball lost: coast on velocity, count frames.
            (State::Tracking, None) => {
                self.frames_without_ball += 1;
                // Coast using predicted velocity.
                self.coast();
                self.smooth_toward(self.alpha_track, ctx.current_fov);

                if self.frames_without_ball >= self.search_delay {
                    self.state = State::Searching;
                    log::debug!(
                        "Director: Tracking -> Searching (lost for {} frames)",
                        self.frames_without_ball
                    );
                }
            }

            // Searching + ball found: start recovering.
            (State::Searching, Some(obj)) => {
                let pos = obj.position.unwrap();
                self.update_velocity(pos.yaw, pos.pitch);
                self.target_yaw = pos.yaw;
                self.target_pitch = pos.pitch;
                self.state = State::Recovering;
                self.recover_count = 0;
                self.frames_without_ball = 0;
                log::debug!("Director: Searching -> Recovering");
            }

            // Searching + still no ball: coast then hold.
            (State::Searching, None) => {
                self.frames_without_ball += 1;
                if self.frames_without_ball < self.search_delay + self.coast_frames {
                    // Still coasting on residual velocity.
                    self.coast();
                    self.smooth_toward(self.alpha_track * 0.3, ctx.current_fov);
                }
                // Beyond coast window: camera holds position.
            }

            // Recovering + ball visible: move toward it, confirm.
            (State::Recovering, Some(obj)) => {
                let pos = obj.position.unwrap();
                self.update_velocity(pos.yaw, pos.pitch);
                self.target_yaw = pos.yaw;
                self.target_pitch = pos.pitch;
                self.smooth_toward(self.alpha_recover, ctx.current_fov);
                self.recover_count += 1;
                self.frames_without_ball = 0;

                if self.recover_count >= self.recover_confirm {
                    self.state = State::Tracking;
                    log::debug!("Director: Recovering -> Tracking (confirmed)");
                }
            }

            // Recovering + ball lost again: coast, maybe back to searching.
            (State::Recovering, None) => {
                self.frames_without_ball += 1;
                self.coast();
                self.smooth_toward(self.alpha_recover * 0.5, ctx.current_fov);

                if self.frames_without_ball >= self.search_delay {
                    self.state = State::Searching;
                    log::debug!("Director: Recovering -> Searching (lost again)");
                }
            }
        }

        // Dynamic FOV: smooth toward target based on ball position.
        let target_fov = self.target_fov();
        self.current_fov += self.fov_alpha * (target_fov - self.current_fov);

        self.clamp_to_bounds(ctx);

        // Log state every 30 frames for debug visibility.
        if ctx.frame_index.is_multiple_of(30) {
            log::debug!(
                "Director frame {}: state={:?}, yaw={:.4}, pitch={:.4}, \
                 target=({:.4},{:.4}), vel=({:.5},{:.5}), fov={:.1}, tracks={}",
                ctx.frame_index,
                self.state,
                self.yaw,
                self.pitch,
                self.target_yaw,
                self.target_pitch,
                self.vel_yaw,
                self.vel_pitch,
                self.current_fov,
                ctx.detections.len(),
            );
        }
    }

    fn position(&self) -> ViewportPosition {
        // Negate yaw: the projection maps left-camera-left-edge to negative
        // yaw, but the renderer pans in the opposite direction.
        ViewportPosition {
            yaw: -self.yaw,
            pitch: self.pitch,
            fov_degrees: Some(self.current_fov),
        }
    }
}
