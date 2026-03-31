//! Ball-tracking director with state machine logic.
//!
//! Follows the ball using a 3-state machine:
//! - **Tracking**: ball is visible, camera smoothly follows it
//! - **Searching**: ball lost, camera holds position and waits
//! - **Recovering**: ball reappeared after being lost, camera moves to it
//!
//! Uses exponential moving average (EMA) smoothing to avoid jerky panning,
//! with a dead zone to suppress micro-movements when the ball is near center.

use reco_core::director::{Director, DirectorContext, ViewportPosition};

/// Director state machine modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Ball is being tracked. Camera follows with smoothing.
    Tracking,
    /// Ball was lost. Camera holds last known position.
    Searching,
    /// Ball reappeared. Camera moves toward it (faster smoothing).
    Recovering,
}

/// Ball-tracking director.
///
/// Pans the virtual camera to follow a detected ball using EMA smoothing,
/// a dead zone to suppress jitter, and viewport bounds clamping to prevent
/// black edges.
pub struct BallDirector {
    state: State,
    /// Current smoothed position.
    yaw: f32,
    pitch: f32,
    /// Target position (from latest detection).
    target_yaw: f32,
    target_pitch: f32,
    /// EMA smoothing factor for tracking (0 = no smoothing, 1 = instant).
    alpha_track: f32,
    /// EMA smoothing factor for recovery (faster than tracking).
    alpha_recover: f32,
    /// Dead zone as fraction of FOV. No panning if target is within this
    /// fraction of the current position.
    dead_zone: f32,
    /// Frames since ball was last seen.
    frames_without_ball: u32,
    /// How many frames to wait before transitioning to Searching.
    search_delay: u32,
    /// How many frames of continuous tracking before leaving Recovering.
    recover_confirm: u32,
    /// Counter for recovery confirmation.
    recover_count: u32,
    /// Optional FOV override in degrees.
    fov: Option<f32>,
    /// Label to track (e.g. "ball").
    target_label: String,
}

impl BallDirector {
    /// Create a new ball director with default parameters.
    ///
    /// `fps` is used to scale timing parameters (delay, recovery) to the
    /// video frame rate.
    pub fn new(fps: f32) -> Self {
        Self {
            state: State::Searching,
            yaw: 0.0,
            pitch: 0.0,
            target_yaw: 0.0,
            target_pitch: 0.0,
            alpha_track: 0.08,
            alpha_recover: 0.15,
            dead_zone: 0.05,
            frames_without_ball: 0,
            search_delay: (fps * 0.5) as u32,    // 0.5 seconds
            recover_confirm: (fps * 0.3) as u32, // 0.3 seconds
            recover_count: 0,
            fov: Some(40.0),
            target_label: "ball".into(),
        }
    }

    /// Set the EMA smoothing factor for normal tracking (default: 0.08).
    ///
    /// Higher values make the camera more responsive but jerkier.
    /// Lower values produce smoother motion but more lag.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha_track = alpha.clamp(0.01, 1.0);
        self
    }

    /// Set the dead zone as a fraction of FOV (default: 0.05).
    ///
    /// The camera won't pan if the ball is within this fraction of the
    /// current viewport center. Prevents micro-movements.
    pub fn with_dead_zone(mut self, dead_zone: f32) -> Self {
        self.dead_zone = dead_zone.clamp(0.0, 0.5);
        self
    }

    /// Set a fixed FOV override in degrees.
    pub fn with_fov(mut self, fov_degrees: f32) -> Self {
        self.fov = Some(fov_degrees);
        self
    }

    /// Set the target label to track (default: "ball").
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.target_label = label.into();
        self
    }

    /// Find the best ball detection from tracked objects.
    ///
    /// Prefers the highest-confidence detection with a panorama position,
    /// prioritizing tracks with higher age (more established).
    fn find_ball<'a>(
        &self,
        ctx: &'a DirectorContext<'_>,
    ) -> Option<&'a reco_core::director::TrackedObject> {
        ctx.objects
            .iter()
            .filter(|obj| obj.label == self.target_label && obj.position.is_some())
            .max_by(|a, b| {
                // Prefer older tracks (more established), then higher confidence.
                let age_cmp = a.age.cmp(&b.age);
                if age_cmp != std::cmp::Ordering::Equal {
                    return age_cmp;
                }
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Apply EMA smoothing toward target, respecting dead zone.
    fn smooth_toward(&mut self, alpha: f32, fov_degrees: f32) {
        let dead_zone_rad = (self.dead_zone * fov_degrees).to_radians();

        let dy = self.target_yaw - self.yaw;
        let dp = self.target_pitch - self.pitch;

        // Only pan if outside dead zone.
        if dy.abs() > dead_zone_rad {
            self.yaw += alpha * dy;
        }
        if dp.abs() > dead_zone_rad {
            self.pitch += alpha * dp;
        }
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
                self.target_yaw = pos.yaw;
                self.target_pitch = pos.pitch;
                self.smooth_toward(self.alpha_track, ctx.current_fov);
                self.frames_without_ball = 0;
            }

            // Tracking + ball lost: start counting.
            (State::Tracking, None) => {
                self.frames_without_ball += 1;
                if self.frames_without_ball >= self.search_delay {
                    self.state = State::Searching;
                    log::debug!(
                        "Director: Tracking -> Searching (lost for {} frames)",
                        self.frames_without_ball
                    );
                }
                // Keep panning toward last known target.
                self.smooth_toward(self.alpha_track * 0.5, ctx.current_fov);
            }

            // Searching + ball found: start recovering.
            (State::Searching, Some(obj)) => {
                let pos = obj.position.unwrap();
                self.target_yaw = pos.yaw;
                self.target_pitch = pos.pitch;
                self.state = State::Recovering;
                self.recover_count = 0;
                self.frames_without_ball = 0;
                log::debug!("Director: Searching -> Recovering");
            }

            // Searching + still no ball: hold position.
            (State::Searching, None) => {
                // Camera stays put.
            }

            // Recovering + ball visible: move toward it, confirm.
            (State::Recovering, Some(obj)) => {
                let pos = obj.position.unwrap();
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

            // Recovering + ball lost again: back to searching.
            (State::Recovering, None) => {
                self.frames_without_ball += 1;
                if self.frames_without_ball >= self.search_delay {
                    self.state = State::Searching;
                    log::debug!("Director: Recovering -> Searching (lost again)");
                }
                self.smooth_toward(self.alpha_recover * 0.5, ctx.current_fov);
            }
        }

        self.clamp_to_bounds(ctx);

        // Log state every 30 frames for debug visibility.
        if ctx.frame_index.is_multiple_of(30) {
            log::debug!(
                "Director frame {}: state={:?}, yaw={:.4}, pitch={:.4}, target=({:.4},{:.4}), tracks={}",
                ctx.frame_index,
                self.state,
                self.yaw,
                self.pitch,
                self.target_yaw,
                self.target_pitch,
                ctx.objects.len(),
            );
        }
    }

    fn position(&self) -> ViewportPosition {
        // Negate yaw: the projection maps left-camera-left-edge to negative
        // yaw, but the renderer pans in the opposite direction.
        ViewportPosition {
            yaw: -self.yaw,
            pitch: self.pitch,
            fov_degrees: self.fov,
        }
    }
}
