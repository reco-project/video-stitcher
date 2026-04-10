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

use reco_core::detector::CameraId;
use reco_core::director::{Director, DirectorContext, MappedDetection, ViewportPosition};

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
    /// Which camera produced the last accepted ball detection.
    ///
    /// Used for deduplication in the overlap region: when both cameras
    /// detect the ball, we prefer the same camera as the previous frame
    /// to prevent oscillation.
    last_camera: Option<CameraId>,
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
            last_camera: None,
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
    ///
    /// When multiple cameras detect the ball (common in the overlap region),
    /// scores each detection by confidence, distance from camera center
    /// (less distortion = better), and camera consistency (prefer the same
    /// camera as the previous frame to prevent oscillation).
    fn find_ball<'a>(&self, ctx: &'a DirectorContext<'_>) -> Option<&'a MappedDetection> {
        let balls: Vec<_> = ctx
            .detections
            .iter()
            .filter(|d| d.label == self.target_label && d.position.is_some())
            .collect();

        if balls.is_empty() {
            return None;
        }
        if balls.len() == 1 {
            return Some(balls[0]);
        }

        // Multiple ball detections (likely from both cameras in the overlap).
        // Score each and pick the best.
        balls.into_iter().max_by(|a, b| {
            let score_a = self.detection_score(a);
            let score_b = self.detection_score(b);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Score a detection for ball selection.
    ///
    /// Higher score = better candidate. Factors:
    /// - **Confidence**: base score from the detector.
    /// - **Center proximity**: detections near the camera center have less
    ///   fisheye distortion and are more reliable. Penalty proportional to
    ///   distance from the normalized center (0.5, 0.5).
    /// - **Camera consistency**: small bonus for matching the previous frame's
    ///   camera, reducing oscillation in the overlap region.
    fn detection_score(&self, det: &MappedDetection) -> f32 {
        let mut score = det.confidence;

        // Prefer detections closer to camera center (less distortion).
        let cx = det.camera_center.0;
        let cy = det.camera_center.1;
        let center_dist = ((cx - 0.5) * (cx - 0.5) + (cy - 0.5) * (cy - 0.5)).sqrt();
        score -= center_dist * 0.2;

        // Prefer the same camera as the previous frame (reduces oscillation).
        if let Some(last_camera) = self.last_camera {
            if det.camera == last_camera {
                score += 0.1;
            }
        }

        score
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

        // Track which camera was used for deduplication consistency.
        if let Some(obj) = ball {
            self.last_camera = Some(obj.camera);
        }

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
                // Reset velocity after a long gap to prevent stale momentum
                // from causing overshoot when the ball reappears.
                if self.frames_without_ball > self.coast_frames {
                    self.vel_yaw = 0.0;
                    self.vel_pitch = 0.0;
                    self.prev_target_yaw = pos.yaw;
                    self.prev_target_pitch = pos.pitch;
                } else {
                    self.update_velocity(pos.yaw, pos.pitch);
                }
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
        if ctx.frame_index % 30 == 0 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::projection::ViewportBounds;

    /// Build a `DirectorContext` with the given detections and a generous
    /// viewport so clamping doesn't interfere with state-machine logic.
    fn ctx_with_detections(
        frame_index: u64,
        detections: &[MappedDetection],
    ) -> DirectorContext<'_> {
        DirectorContext {
            frame_index,
            timestamp_ms: frame_index as f64 * (1000.0 / 30.0),
            detections,
            viewport_bounds: ViewportBounds {
                min_yaw: -2.0,
                max_yaw: 2.0,
                min_pitch: -1.0,
                max_pitch: 1.0,
            },
            current_fov: 55.0,
        }
    }

    /// Shorthand: one ball detection at the given panorama yaw/pitch.
    fn ball_detection(yaw: f32, pitch: f32) -> MappedDetection {
        MappedDetection {
            camera: reco_core::detector::CameraId::Left,
            label: "ball".into(),
            confidence: 0.9,
            camera_center: (0.5, 0.5),
            camera_size: (0.02, 0.02),
            position: Some(ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            }),
        }
    }

    // ---- State machine tests ----

    #[test]
    fn initial_state_is_searching() {
        let dir = BallDirector::new(30.0);
        // The state field is private, but we can observe it via behavior:
        // In Searching state with no ball, update should not panic and
        // position should stay at origin.
        assert_eq!(dir.state, State::Searching);
    }

    #[test]
    fn searching_to_recovering_on_detection() {
        let mut dir = BallDirector::new(30.0);
        assert_eq!(dir.state, State::Searching);

        let det = [ball_detection(0.5, 0.1)];
        let ctx = ctx_with_detections(0, &det);
        dir.update(&ctx);

        // First ball detection in Searching triggers Recovering.
        assert_eq!(dir.state, State::Recovering);
    }

    #[test]
    fn recovering_to_tracking_after_confirmation() {
        let mut dir = BallDirector::new(30.0);
        let det = [ball_detection(0.5, 0.1)];

        // First detection: Searching -> Recovering
        dir.update(&ctx_with_detections(0, &det));
        assert_eq!(dir.state, State::Recovering);

        // Feed detections for recover_confirm frames (0.3s at 30fps = 9 frames).
        // The first update already set recover_count to 0 and state to Recovering.
        // Each subsequent update with a detection increments recover_count.
        let confirm_frames = (30.0 * 0.3) as u64; // 9 frames
        for i in 1..=confirm_frames {
            dir.update(&ctx_with_detections(i, &det));
        }

        assert_eq!(dir.state, State::Tracking);
    }

    #[test]
    fn tracking_to_searching_when_ball_lost() {
        let mut dir = BallDirector::new(30.0);

        // Get into Tracking state.
        let det = [ball_detection(0.5, 0.1)];
        dir.update(&ctx_with_detections(0, &det));
        let confirm_frames = (30.0 * 0.3) as u64;
        for i in 1..=confirm_frames {
            dir.update(&ctx_with_detections(i, &det));
        }
        assert_eq!(dir.state, State::Tracking);

        // Now stop sending detections. After search_delay frames (1.5s = 45
        // frames at 30fps) we should transition to Searching.
        let search_delay = (30.0_f32 * 1.5) as u64; // 45
        let base = confirm_frames + 1;
        for i in 0..search_delay {
            let empty: &[MappedDetection] = &[];
            dir.update(&ctx_with_detections(base + i, empty));
        }

        assert_eq!(dir.state, State::Searching);
    }

    #[test]
    fn recovering_falls_back_to_searching_if_ball_lost_again() {
        let mut dir = BallDirector::new(30.0);

        // Searching -> Recovering
        let det = [ball_detection(0.5, 0.1)];
        dir.update(&ctx_with_detections(0, &det));
        assert_eq!(dir.state, State::Recovering);

        // Lose ball immediately. After search_delay frames: Recovering -> Searching.
        let search_delay = (30.0_f32 * 1.5) as u64;
        for i in 1..=search_delay {
            let empty: &[MappedDetection] = &[];
            dir.update(&ctx_with_detections(i, empty));
        }

        assert_eq!(dir.state, State::Searching);
    }

    // ---- Smoothing (EMA) tests ----

    #[test]
    fn ema_smoothing_moves_toward_target() {
        let mut dir = BallDirector::new(30.0);

        // Get into Tracking state.
        let det = [ball_detection(0.5, 0.1)];
        dir.update(&ctx_with_detections(0, &det));
        let confirm_frames = (30.0 * 0.3) as u64;
        for i in 1..=confirm_frames {
            dir.update(&ctx_with_detections(i, &det));
        }
        assert_eq!(dir.state, State::Tracking);

        // Now jump target to a distant position and track convergence.
        let far_det = [ball_detection(1.0, 0.3)];
        let base = confirm_frames + 1;
        let yaw_before = dir.yaw;

        // Run many frames to let EMA converge. With alpha=0.04,
        // convergence is slow (each frame moves 4% of remaining error),
        // so 500 frames gets us close but not exact.
        for i in 0..500 {
            dir.update(&ctx_with_detections(base + i, &far_det));
        }

        // After many frames, yaw should have moved significantly toward 1.0.
        assert!(
            dir.yaw > yaw_before,
            "yaw should move toward the target: {} > {}",
            dir.yaw,
            yaw_before
        );
        assert!(
            (dir.yaw - 1.0).abs() < 0.15,
            "yaw should converge near 1.0, got {}",
            dir.yaw
        );
    }

    // ---- Dead zone tests ----

    #[test]
    fn dead_zone_suppresses_small_movements() {
        // Use a large dead zone and fast alpha so we can converge fully
        // before testing the dead zone behavior.
        let mut dir = BallDirector::new(30.0).with_dead_zone(0.10).with_alpha(0.5); // fast convergence

        // Get into Tracking at a known position.
        let det = [ball_detection(0.5, 0.1)];
        dir.update(&ctx_with_detections(0, &det));
        let confirm = (30.0 * 0.3) as u64;
        for i in 1..=confirm {
            dir.update(&ctx_with_detections(i, &det));
        }

        // Run enough frames so the camera fully converges to target.
        for i in 0..200 {
            dir.update(&ctx_with_detections(confirm + 1 + i, &det));
        }
        let yaw_settled = dir.yaw;

        // Dead zone = 0.10 * 55.0 degrees = 5.5 degrees = ~0.096 radians.
        // The settled yaw is very close to 0.5 (alpha=0.5 converges fast).
        // Move target by only 0.005 radians - well within the dead zone.
        let tiny_det = [ball_detection(yaw_settled + 0.005, 0.1)];
        for i in 0..10 {
            dir.update(&ctx_with_detections(confirm + 201 + i, &tiny_det));
        }

        assert!(
            (dir.yaw - yaw_settled).abs() < 0.001,
            "yaw should not change for within-dead-zone movement: settled={}, \
             now={}",
            yaw_settled,
            dir.yaw
        );
    }

    #[test]
    fn dead_zone_allows_large_movements() {
        let mut dir = BallDirector::new(30.0).with_dead_zone(0.10);

        // Get into Tracking at origin-ish position.
        let det = [ball_detection(0.1, 0.0)];
        dir.update(&ctx_with_detections(0, &det));
        let confirm = (30.0 * 0.3) as u64;
        for i in 1..=confirm {
            dir.update(&ctx_with_detections(i, &det));
        }
        for i in 0..300 {
            dir.update(&ctx_with_detections(confirm + 1 + i, &det));
        }

        let yaw_before = dir.yaw;

        // Jump the detection far outside the dead zone.
        let far_det = [ball_detection(0.8, 0.0)];
        for i in 0..200 {
            dir.update(&ctx_with_detections(confirm + 301 + i, &far_det));
        }

        assert!(
            (dir.yaw - yaw_before).abs() > 0.1,
            "yaw should move for beyond-dead-zone target: before={}, now={}",
            yaw_before,
            dir.yaw
        );
    }

    // ---- FPS clamping tests ----

    #[test]
    fn fps_clamped_low() {
        let dir = BallDirector::new(0.0);
        // fps=0.0 clamped to 1.0 -> search_delay = (1.0 * 1.5) as u32 = 1
        assert_eq!(dir.search_delay, 1);
    }

    #[test]
    fn fps_clamped_high() {
        let dir = BallDirector::new(100_000.0);
        // fps clamped to 1000.0 -> search_delay = (1000 * 1.5) as u32 = 1500
        assert_eq!(dir.search_delay, 1500);
    }

    #[test]
    fn fps_negative_clamped_to_one() {
        let dir = BallDirector::new(-10.0);
        assert_eq!(dir.search_delay, 1); // (1.0 * 1.5) as u32
        assert_eq!(dir.coast_frames, 2); // (1.0 * 2.0) as u32
    }

    // ---- Viewport bounds clamping ----

    #[test]
    fn position_clamped_to_viewport_bounds() {
        let mut dir = BallDirector::new(30.0);

        // Send a detection far outside the viewport bounds.
        let det = [ball_detection(5.0, 3.0)];
        dir.update(&ctx_with_detections(0, &det));
        let confirm = (30.0 * 0.3) as u64;
        for i in 1..=confirm {
            dir.update(&ctx_with_detections(i, &det));
        }
        for i in 0..500 {
            dir.update(&ctx_with_detections(confirm + 1 + i, &det));
        }

        // The internal yaw should be clamped to viewport max (2.0).
        assert!(
            dir.yaw <= 2.0,
            "yaw should be clamped to max_yaw: {}",
            dir.yaw
        );
        assert!(
            dir.pitch <= 1.0,
            "pitch should be clamped to max_pitch: {}",
            dir.pitch
        );
    }

    // ---- Position output ----

    #[test]
    fn position_negates_yaw() {
        let mut dir = BallDirector::new(30.0);

        // Get into Tracking with positive yaw.
        let det = [ball_detection(0.5, 0.1)];
        dir.update(&ctx_with_detections(0, &det));
        let confirm = (30.0 * 0.3) as u64;
        for i in 1..=confirm {
            dir.update(&ctx_with_detections(i, &det));
        }
        for i in 0..300 {
            dir.update(&ctx_with_detections(confirm + 1 + i, &det));
        }

        let pos = dir.position();
        // Internal yaw is positive -> output yaw should be negative.
        assert!(
            dir.yaw > 0.0,
            "internal yaw should be positive: {}",
            dir.yaw
        );
        assert!(pos.yaw < 0.0, "output yaw should be negated: {}", pos.yaw);
        assert!(
            (pos.yaw + dir.yaw).abs() < 1e-6,
            "pos.yaw should be -dir.yaw"
        );
    }

    #[test]
    fn position_includes_fov() {
        let dir = BallDirector::new(30.0);
        let pos = dir.position();
        assert!(pos.fov_degrees.is_some());
    }

    // ---- find_ball filtering ----

    #[test]
    fn find_ball_ignores_non_ball_labels() {
        let dir = BallDirector::new(30.0);

        let detections = [
            MappedDetection {
                camera: reco_core::detector::CameraId::Left,
                label: "player".into(),
                confidence: 0.95,
                camera_center: (0.5, 0.5),
                camera_size: (0.1, 0.2),
                position: Some(ViewportPosition {
                    yaw: 0.3,
                    pitch: 0.1,
                    fov_degrees: None,
                }),
            },
            ball_detection(0.5, 0.1),
        ];

        let ctx = ctx_with_detections(0, &detections);
        let found = dir.find_ball(&ctx);
        assert!(found.is_some());
        assert_eq!(found.unwrap().label, "ball");
    }

    #[test]
    fn find_ball_picks_highest_confidence() {
        let dir = BallDirector::new(30.0);

        let detections = [
            MappedDetection {
                camera: reco_core::detector::CameraId::Left,
                label: "ball".into(),
                confidence: 0.5,
                camera_center: (0.3, 0.3),
                camera_size: (0.02, 0.02),
                position: Some(ViewportPosition {
                    yaw: 0.1,
                    pitch: 0.0,
                    fov_degrees: None,
                }),
            },
            MappedDetection {
                camera: reco_core::detector::CameraId::Left,
                label: "ball".into(),
                confidence: 0.95,
                camera_center: (0.7, 0.7),
                camera_size: (0.02, 0.02),
                position: Some(ViewportPosition {
                    yaw: 0.8,
                    pitch: 0.2,
                    fov_degrees: None,
                }),
            },
        ];

        let ctx = ctx_with_detections(0, &detections);
        let found = dir.find_ball(&ctx);
        assert!(found.is_some());
        assert!((found.unwrap().confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn find_ball_returns_none_for_no_position() {
        let dir = BallDirector::new(30.0);

        let detections = [MappedDetection {
            camera: reco_core::detector::CameraId::Left,
            label: "ball".into(),
            confidence: 0.9,
            camera_center: (0.5, 0.5),
            camera_size: (0.02, 0.02),
            position: None, // unmapped detection
        }];

        let ctx = ctx_with_detections(0, &detections);
        let found = dir.find_ball(&ctx);
        assert!(found.is_none());
    }

    // ---- Dynamic FOV tests ----

    #[test]
    fn fov_zooms_in_for_high_pitch() {
        let mut dir = BallDirector::new(30.0);

        // High pitch = ball far away -> should zoom in (lower FOV).
        dir.target_pitch = 0.20;
        let tight = dir.target_fov();

        dir.target_pitch = -0.05;
        let wide = dir.target_fov();

        assert!(
            tight < wide,
            "high pitch should produce tighter FOV: tight={}, wide={}",
            tight,
            wide
        );
        assert!(
            (tight - dir.fov_tight).abs() < 0.01,
            "max pitch should give fov_tight: {}",
            tight
        );
        assert!(
            (wide - dir.fov_wide).abs() < 0.01,
            "min pitch should give fov_wide: {}",
            wide
        );
    }

    // ---- Velocity prediction ----

    #[test]
    fn velocity_updated_on_reasonable_movement() {
        let mut dir = BallDirector::new(30.0);
        dir.prev_target_yaw = 0.0;
        dir.prev_target_pitch = 0.0;

        dir.update_velocity(0.01, 0.005);
        assert!(dir.vel_yaw.abs() > 0.0, "velocity should be non-zero");
        assert!(dir.vel_pitch.abs() > 0.0, "velocity should be non-zero");
    }

    #[test]
    fn velocity_not_updated_on_teleport() {
        let mut dir = BallDirector::new(30.0);
        dir.prev_target_yaw = 0.0;
        dir.prev_target_pitch = 0.0;
        dir.vel_yaw = 0.0;
        dir.vel_pitch = 0.0;

        // Jump > 0.3 radians is considered a teleport.
        dir.update_velocity(1.0, 0.0);
        assert!(
            dir.vel_yaw.abs() < 1e-9,
            "velocity should not update on teleport: {}",
            dir.vel_yaw
        );
    }

    // ---- Detection scoring (dual-camera deduplication) ----

    /// Ball detection with explicit camera and camera_center.
    fn ball_detection_with_camera(
        camera: CameraId,
        camera_center: (f32, f32),
        confidence: f32,
        yaw: f32,
        pitch: f32,
    ) -> MappedDetection {
        MappedDetection {
            camera,
            label: "ball".into(),
            confidence,
            camera_center,
            camera_size: (0.02, 0.02),
            position: Some(ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            }),
        }
    }

    #[test]
    fn detection_score_prefers_center() {
        let dir = BallDirector::new(30.0);

        let center_det = ball_detection_with_camera(CameraId::Left, (0.5, 0.5), 0.9, 0.3, 0.1);
        let edge_det = ball_detection_with_camera(CameraId::Right, (0.9, 0.9), 0.9, 0.3, 0.1);

        let center_score = dir.detection_score(&center_det);
        let edge_score = dir.detection_score(&edge_det);

        assert!(
            center_score > edge_score,
            "center detection ({:.4}) should score higher than edge ({:.4})",
            center_score,
            edge_score
        );
    }

    #[test]
    fn detection_score_includes_confidence() {
        let dir = BallDirector::new(30.0);

        let high_conf = ball_detection_with_camera(CameraId::Left, (0.5, 0.5), 0.95, 0.3, 0.1);
        let low_conf = ball_detection_with_camera(CameraId::Left, (0.5, 0.5), 0.50, 0.3, 0.1);

        assert!(
            dir.detection_score(&high_conf) > dir.detection_score(&low_conf),
            "higher confidence should produce a higher score"
        );
    }

    #[test]
    fn last_camera_consistency_bonus() {
        let mut dir = BallDirector::new(30.0);
        dir.last_camera = Some(CameraId::Left);

        let left_det = ball_detection_with_camera(CameraId::Left, (0.5, 0.5), 0.9, 0.3, 0.1);
        let right_det = ball_detection_with_camera(CameraId::Right, (0.5, 0.5), 0.9, 0.3, 0.1);

        let left_score = dir.detection_score(&left_det);
        let right_score = dir.detection_score(&right_det);

        assert!(
            left_score > right_score,
            "same-camera detection ({:.4}) should score higher than other ({:.4})",
            left_score,
            right_score
        );

        // The bonus should be exactly 0.1.
        assert!(
            (left_score - right_score - 0.1).abs() < 1e-6,
            "consistency bonus should be 0.1, got {:.4}",
            left_score - right_score
        );
    }

    #[test]
    fn find_ball_updates_last_camera() {
        let mut dir = BallDirector::new(30.0);
        assert!(dir.last_camera.is_none());

        let det = [ball_detection_with_camera(
            CameraId::Right,
            (0.5, 0.5),
            0.9,
            0.3,
            0.1,
        )];
        let ctx = ctx_with_detections(0, &det);
        dir.update(&ctx);

        assert_eq!(dir.last_camera, Some(CameraId::Right));
    }

    #[test]
    fn find_ball_prefers_same_camera_in_overlap() {
        let mut dir = BallDirector::new(30.0);
        dir.last_camera = Some(CameraId::Left);

        // Two detections at the same position, same confidence, but
        // different cameras. The left camera should win due to consistency.
        let detections = [
            ball_detection_with_camera(CameraId::Left, (0.5, 0.5), 0.9, 0.3, 0.1),
            ball_detection_with_camera(CameraId::Right, (0.5, 0.5), 0.9, 0.3, 0.1),
        ];
        let ctx = ctx_with_detections(0, &detections);
        let found = dir.find_ball(&ctx);

        assert!(found.is_some());
        assert_eq!(found.unwrap().camera, CameraId::Left);
    }

    // ---- Gap interpolation (velocity reset on long gap) ----

    #[test]
    fn velocity_reset_on_long_gap_recovery() {
        let mut dir = BallDirector::new(30.0);

        // Get into Tracking with some velocity.
        let det = [ball_detection(0.5, 0.1)];
        dir.update(&ctx_with_detections(0, &det));
        let confirm = (30.0 * 0.3) as u64;
        for i in 1..=confirm {
            dir.update(&ctx_with_detections(i, &det));
        }
        assert_eq!(dir.state, State::Tracking);

        // Build up some velocity by moving the ball.
        let moving = [ball_detection(0.55, 0.1)];
        for i in 0..10 {
            dir.update(&ctx_with_detections(confirm + 1 + i, &moving));
        }
        assert!(
            dir.vel_yaw.abs() > 0.0,
            "should have velocity from movement"
        );

        // Lose the ball for longer than coast_frames (search_delay + coast_frames).
        let search_delay = (30.0_f32 * 1.5) as u64;
        let coast = (30.0_f32 * 2.0) as u64;
        let gap_start = confirm + 11;
        let empty: &[MappedDetection] = &[];
        for i in 0..(search_delay + coast + 1) {
            dir.update(&ctx_with_detections(gap_start + i, empty));
        }
        assert_eq!(dir.state, State::Searching);

        // Ball reappears after a long gap -> Searching -> Recovering.
        // Velocity should be reset.
        let reappear = [ball_detection(0.8, 0.2)];
        let reappear_frame = gap_start + search_delay + coast + 1;
        dir.update(&ctx_with_detections(reappear_frame, &reappear));

        assert_eq!(dir.state, State::Recovering);
        assert!(
            dir.vel_yaw.abs() < 1e-9,
            "velocity should be reset after long gap: vel_yaw={}",
            dir.vel_yaw
        );
        assert!(
            dir.vel_pitch.abs() < 1e-9,
            "velocity should be reset after long gap: vel_pitch={}",
            dir.vel_pitch
        );
    }
}
