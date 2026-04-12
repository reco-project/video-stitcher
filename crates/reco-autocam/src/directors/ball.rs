//! Ball-tracking director with plausibility rejection and dynamic FOV.
//!
//! Follows the ball using a 2-state machine:
//! - **Tracking**: ball is visible, camera snaps to its position
//! - **Searching**: ball lost, camera holds and waits for confirmed redetection
//!
//! Outputs **raw, unsmoothed** positions. Trajectory smoothing is handled
//! externally by [`SmoothedDirector`](crate::SmoothedDirector), and viewport
//! constraining (coverage clamping) is handled by the session in reco-core.
//! This separation keeps the director focused on ball selection and state
//! machine logic.

use reco_core::detector::CameraId;
use reco_core::director::{Director, DirectorContext, MappedDetection, ViewportPosition};

/// Director state machine modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Ball is being tracked. Camera follows with raw snap.
    Tracking,
    /// Ball was lost. Camera holds position, waits for confirmed redetection.
    Searching,
}

/// Ball-tracking director with plausibility rejection and dynamic FOV.
///
/// Pans the virtual camera to follow a detected ball. Uses plausibility
/// rejection to ignore false positives and multi-frame search confirmation
/// to prevent chasing noise. Outputs raw (unsmoothed) positions - pair
/// with [`SmoothedDirector`](crate::SmoothedDirector) for smooth camera motion.
pub struct BallDirector {
    state: State,
    /// Current raw position (panorama-space radians). This IS the output.
    yaw: f32,
    pitch: f32,
    /// Current raw FOV in degrees (no EMA - smoother handles transitions).
    current_fov: f32,
    /// Wide FOV (zoomed out) in degrees.
    fov_wide: f32,
    /// Tight FOV (zoomed in) in degrees.
    fov_tight: f32,
    /// Class ID to track (e.g. 0 for "ball").
    target_class_id: u16,
    /// Which camera produced the last accepted ball detection.
    /// Used for deduplication in the overlap region.
    last_camera: Option<CameraId>,
    /// Frames since ball was last seen.
    frames_without_ball: u32,
    /// How many frames to wait before transitioning to Searching.
    search_delay: u32,
    /// Max distance (rad) a detection can be from the current position
    /// to be considered plausible ball movement while tracking.
    /// Scaled by sqrt(detection_interval) at runtime.
    max_jump: f32,
    /// Detection interval (frames between actual detections). Set by
    /// the session so the director can scale its jump threshold.
    detection_interval: u32,
    /// Candidate position being confirmed during Searching state.
    search_candidate_yaw: f32,
    search_candidate_pitch: f32,
    /// Consecutive fresh detections near the candidate.
    search_confirm_count: u32,
    /// Fresh detections needed to accept a search candidate.
    search_confirm_needed: u32,
    /// Maximum distance (rad) from the camera to consider a detection
    /// during Searching. Prevents chasing false positives across the panorama.
    search_max_radius: f32,
}

impl BallDirector {
    /// Create a new ball director with default parameters.
    ///
    /// `fps` is used to scale timing parameters to the video frame rate.
    /// Clamped to `[1.0, 1000.0]`.
    pub fn new(fps: f32) -> Self {
        let fps = fps.clamp(1.0, 1000.0);
        Self {
            state: State::Searching,
            yaw: 0.0,
            pitch: 0.0,
            current_fov: 55.0,
            fov_wide: 65.0,
            fov_tight: 35.0,
            target_class_id: 0,
            last_camera: None,
            frames_without_ball: 0,
            search_delay: (fps * 1.5) as u32,
            max_jump: 0.09,
            detection_interval: 1,
            search_candidate_yaw: 0.0,
            search_candidate_pitch: 0.0,
            search_confirm_count: 0,
            search_confirm_needed: 3,
            search_max_radius: 0.70,
        }
    }

    /// Set the target class ID to track (default: 0).
    ///
    /// Resolve label names to class IDs via the detector's `class_names()`.
    pub fn with_class_id(mut self, class_id: u16) -> Self {
        self.target_class_id = class_id;
        self
    }

    /// Set the detection interval so the jump threshold can scale.
    ///
    /// Higher intervals mean more frames between detections, so the ball
    /// can legitimately move further between checks.
    pub fn set_detection_interval(&mut self, interval: u32) {
        self.detection_interval = interval.max(1);
    }

    /// Find the best ball detection from mapped detections.
    ///
    /// When multiple cameras detect the ball (overlap region), scores
    /// each detection by confidence, center proximity, and camera
    /// consistency to pick the best candidate.
    fn find_ball<'a>(&self, ctx: &'a DirectorContext<'_>) -> Option<&'a MappedDetection> {
        let balls: Vec<_> = ctx
            .detections
            .iter()
            .filter(|d| d.class_id == self.target_class_id && d.position.is_some())
            .collect();

        if balls.is_empty() {
            return None;
        }
        if balls.len() == 1 {
            return Some(balls[0]);
        }

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
    /// Higher = better. Factors: confidence, center proximity (less
    /// fisheye distortion), camera consistency (reduces oscillation).
    fn detection_score(&self, det: &MappedDetection) -> f32 {
        super::util::detection_score(det, self.last_camera)
    }

    /// Compute target FOV based on ball's panorama-space pitch.
    ///
    /// Higher pitch (ball far away) -> tighter zoom.
    /// Lower pitch (ball near camera) -> wider view.
    fn target_fov(&self) -> f32 {
        let pitch_near: f32 = -0.05;
        let pitch_far: f32 = 0.20;
        let t = ((self.pitch - pitch_near) / (pitch_far - pitch_near)).clamp(0.0, 1.0);
        self.fov_wide + t * (self.fov_tight - self.fov_wide)
    }

    /// Check if a detection is close enough to the current camera to be
    /// plausible ball movement. Threshold scales with detection interval.
    fn is_plausible(&self, det_yaw: f32, det_pitch: f32) -> bool {
        let scale = (self.detection_interval as f32).sqrt();
        let threshold = self.max_jump * scale;
        let dy = det_yaw - self.yaw;
        let dp = det_pitch - self.pitch;
        (dy * dy + dp * dp).sqrt() < threshold
    }

    /// Check if two positions are near each other (within the plausibility
    /// threshold). Used for search candidate confirmation.
    fn is_near(&self, yaw_a: f32, pitch_a: f32, yaw_b: f32, pitch_b: f32) -> bool {
        let scale = (self.detection_interval as f32).sqrt();
        let threshold = self.max_jump * scale;
        let dy = yaw_a - yaw_b;
        let dp = pitch_a - pitch_b;
        (dy * dy + dp * dp).sqrt() < threshold
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
            // Tracking + ball visible: snap if plausible.
            (State::Tracking, Some(obj)) => {
                let pos = obj.position.unwrap();

                if self.is_plausible(pos.yaw, pos.pitch) {
                    self.yaw = pos.yaw;
                    self.pitch = pos.pitch;
                    self.frames_without_ball = 0;
                } else {
                    self.frames_without_ball += 1;
                    log::trace!(
                        "Director: ignoring implausible detection ({:.3},{:.3}), \
                         camera at ({:.3},{:.3})",
                        pos.yaw,
                        pos.pitch,
                        self.yaw,
                        self.pitch,
                    );
                }
            }

            // Tracking + ball lost: hold position, count frames.
            (State::Tracking, None) => {
                self.frames_without_ball += 1;
                if self.frames_without_ball >= self.search_delay {
                    self.state = State::Searching;
                    self.search_confirm_count = 0;
                    log::debug!(
                        "Director: Tracking -> Searching (lost for {} frames)",
                        self.frames_without_ball
                    );
                }
            }

            // Searching + fresh detection: confirm before accepting.
            // Requires multiple consecutive detections near the same spot
            // and within search_max_radius of the camera.
            (State::Searching, Some(obj)) if ctx.fresh_detection => {
                let pos = obj.position.unwrap();

                let dist = ((pos.yaw - self.yaw).powi(2) + (pos.pitch - self.pitch).powi(2)).sqrt();
                if dist > self.search_max_radius {
                    self.search_confirm_count = 0;
                } else {
                    let near_candidate = self.is_near(
                        pos.yaw,
                        pos.pitch,
                        self.search_candidate_yaw,
                        self.search_candidate_pitch,
                    );

                    if near_candidate && self.search_confirm_count > 0 {
                        self.search_confirm_count += 1;
                    } else {
                        self.search_candidate_yaw = pos.yaw;
                        self.search_candidate_pitch = pos.pitch;
                        self.search_confirm_count = 1;
                    }

                    if self.search_confirm_count >= self.search_confirm_needed {
                        self.yaw = pos.yaw;
                        self.pitch = pos.pitch;
                        self.state = State::Tracking;
                        self.frames_without_ball = 0;
                        self.search_confirm_count = 0;
                        log::debug!("Director: Searching -> Tracking (confirmed)");
                    }
                }
            }

            // Searching + cached detection (not fresh): ignore.
            (State::Searching, Some(_)) => {}

            // Searching + no ball: hold position.
            (State::Searching, None) => {}
        }

        // Dynamic FOV: raw target, no EMA (smoother handles transitions).
        self.current_fov = self.target_fov();

        // Log state every 30 frames for debug visibility.
        if ctx.frame_index % 30 == 0 {
            log::debug!(
                "Director frame {}: state={:?}, yaw={:.4}, pitch={:.4}, fov={:.1}, tracks={}",
                ctx.frame_index,
                self.state,
                self.yaw,
                self.pitch,
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

    /// Build a `DirectorContext` with the given detections.
    fn ctx(frame_index: u64, detections: &[MappedDetection]) -> DirectorContext<'_> {
        DirectorContext {
            frame_index,
            timestamp_ms: frame_index as f64 * (1000.0 / 30.0),
            detections,
            fresh_detection: true,
        }
    }

    /// Context with explicit fresh_detection flag.
    fn ctx_fresh(
        frame_index: u64,
        detections: &[MappedDetection],
        fresh: bool,
    ) -> DirectorContext<'_> {
        DirectorContext {
            frame_index,
            timestamp_ms: frame_index as f64 * (1000.0 / 30.0),
            detections,
            fresh_detection: fresh,
        }
    }

    /// Shorthand: one ball detection at the given panorama yaw/pitch.
    fn ball_detection(yaw: f32, pitch: f32) -> MappedDetection {
        MappedDetection {
            camera: CameraId::Left,
            class_id: 0, // "ball"
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
            class_id: 0, // "ball"
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

    /// Drive the director from Searching to Tracking by feeding
    /// enough confirmed fresh detections at the given position.
    fn drive_to_tracking(dir: &mut BallDirector, yaw: f32, pitch: f32) {
        let det = [ball_detection(yaw, pitch)];
        for i in 0..dir.search_confirm_needed as u64 {
            dir.update(&ctx(i, &det));
        }
        assert_eq!(
            dir.state,
            State::Tracking,
            "should be Tracking after confirmation"
        );
    }

    // ---- State machine tests ----

    #[test]
    fn initial_state_is_searching() {
        let dir = BallDirector::new(30.0);
        assert_eq!(dir.state, State::Searching);
    }

    #[test]
    fn searching_to_tracking_after_confirmation() {
        let mut dir = BallDirector::new(30.0);
        assert_eq!(dir.state, State::Searching);

        let det = [ball_detection(0.5, 0.1)];
        // Need search_confirm_needed (3) fresh detections.
        for i in 0..3 {
            dir.update(&ctx(i, &det));
        }
        assert_eq!(dir.state, State::Tracking);
    }

    #[test]
    fn searching_ignores_cached_detections() {
        let mut dir = BallDirector::new(30.0);

        let det = [ball_detection(0.5, 0.1)];
        // Fresh detection starts confirmation.
        dir.update(&ctx(0, &det));
        assert_eq!(dir.search_confirm_count, 1);

        // Cached (not fresh) detections should not count.
        dir.update(&ctx_fresh(1, &det, false));
        dir.update(&ctx_fresh(2, &det, false));
        assert_eq!(dir.state, State::Searching);
        // Confirm count should not have increased from cached.
        assert_eq!(dir.search_confirm_count, 1);
    }

    #[test]
    fn tracking_to_searching_when_ball_lost() {
        let mut dir = BallDirector::new(30.0);
        drive_to_tracking(&mut dir, 0.5, 0.1);

        let search_delay = dir.search_delay as u64;
        let base = dir.search_confirm_needed as u64;
        let empty: &[MappedDetection] = &[];
        for i in 0..search_delay {
            dir.update(&ctx(base + i, empty));
        }
        assert_eq!(dir.state, State::Searching);
    }

    #[test]
    fn tracking_snaps_to_ball_position() {
        let mut dir = BallDirector::new(30.0);
        drive_to_tracking(&mut dir, 0.5, 0.1);

        // Position should be exactly at the ball (raw, no EMA).
        assert!(
            (dir.yaw - 0.5).abs() < 1e-6,
            "yaw should snap to ball: {}",
            dir.yaw
        );
        assert!(
            (dir.pitch - 0.1).abs() < 1e-6,
            "pitch should snap to ball: {}",
            dir.pitch
        );
    }

    #[test]
    fn tracking_holds_position_when_ball_lost() {
        let mut dir = BallDirector::new(30.0);
        drive_to_tracking(&mut dir, 0.5, 0.1);

        let yaw_before = dir.yaw;
        let pitch_before = dir.pitch;

        // Lose ball for a few frames (not enough for Searching).
        let empty: &[MappedDetection] = &[];
        let base = dir.search_confirm_needed as u64;
        for i in 0..5 {
            dir.update(&ctx(base + i, empty));
        }

        assert_eq!(dir.state, State::Tracking);
        assert!(
            (dir.yaw - yaw_before).abs() < 1e-6,
            "yaw should hold: {} vs {}",
            dir.yaw,
            yaw_before
        );
        assert!((dir.pitch - pitch_before).abs() < 1e-6, "pitch should hold",);
    }

    // ---- Plausibility rejection ----

    #[test]
    fn implausible_detection_ignored_while_tracking() {
        let mut dir = BallDirector::new(30.0);
        drive_to_tracking(&mut dir, 0.5, 0.1);

        let yaw_before = dir.yaw;

        // Detection far away from current position (implausible jump).
        let far_det = [ball_detection(2.0, 0.5)];
        let base = dir.search_confirm_needed as u64;
        dir.update(&ctx(base, &far_det));

        assert!(
            (dir.yaw - yaw_before).abs() < 1e-6,
            "implausible detection should be ignored: {} vs {}",
            dir.yaw,
            yaw_before
        );
    }

    #[test]
    fn plausible_detection_accepted_while_tracking() {
        let mut dir = BallDirector::new(30.0);
        drive_to_tracking(&mut dir, 0.5, 0.1);

        // Small movement (within max_jump).
        let near_det = [ball_detection(0.52, 0.1)];
        let base = dir.search_confirm_needed as u64;
        dir.update(&ctx(base, &near_det));

        assert!(
            (dir.yaw - 0.52).abs() < 1e-6,
            "plausible detection should snap: {}",
            dir.yaw
        );
    }

    // ---- Search confirmation ----

    #[test]
    fn search_rejects_far_detections() {
        let mut dir = BallDirector::new(30.0);
        // Stay in Searching. Detection far beyond search_max_radius.
        let far_det = [ball_detection(5.0, 0.0)];
        for i in 0..10 {
            dir.update(&ctx(i, &far_det));
        }
        assert_eq!(dir.state, State::Searching);
    }

    #[test]
    fn search_confirmation_resets_on_different_location() {
        let mut dir = BallDirector::new(30.0);

        // First detection starts candidate.
        let det1 = [ball_detection(0.3, 0.1)];
        dir.update(&ctx(0, &det1));
        assert_eq!(dir.search_confirm_count, 1);

        // Different location (not near candidate) resets.
        let det2 = [ball_detection(0.6, 0.1)];
        dir.update(&ctx(1, &det2));
        assert_eq!(dir.search_confirm_count, 1); // reset to 1 (new candidate)
    }

    // ---- FPS clamping ----

    #[test]
    fn fps_clamped_low() {
        let dir = BallDirector::new(0.0);
        assert_eq!(dir.search_delay, 1);
    }

    #[test]
    fn fps_clamped_high() {
        let dir = BallDirector::new(100_000.0);
        assert_eq!(dir.search_delay, 1500);
    }

    // ---- Position output ----

    #[test]
    fn position_negates_yaw() {
        let mut dir = BallDirector::new(30.0);
        drive_to_tracking(&mut dir, 0.5, 0.1);

        let pos = dir.position();
        assert!(pos.yaw < 0.0, "output yaw should be negated: {}", pos.yaw);
        assert!(
            (pos.yaw + dir.yaw).abs() < 1e-6,
            "pos.yaw should be -dir.yaw"
        );
    }

    #[test]
    fn position_includes_fov() {
        let dir = BallDirector::new(30.0);
        assert!(dir.position().fov_degrees.is_some());
    }

    // ---- Dynamic FOV ----

    #[test]
    fn fov_zooms_in_for_high_pitch() {
        let mut dir = BallDirector::new(30.0);

        dir.pitch = 0.20;
        let tight = dir.target_fov();

        dir.pitch = -0.05;
        let wide = dir.target_fov();

        assert!(
            tight < wide,
            "high pitch should produce tighter FOV: tight={}, wide={}",
            tight,
            wide
        );
    }

    // ---- find_ball ----

    #[test]
    fn find_ball_ignores_non_ball_class_ids() {
        let dir = BallDirector::new(30.0);

        let detections = [
            MappedDetection {
                camera: CameraId::Left,
                class_id: 1, // "player" - not the target
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

        let c = ctx(0, &detections);
        let found = dir.find_ball(&c);
        assert!(found.is_some());
        assert_eq!(found.unwrap().class_id, 0); // "ball"
    }

    #[test]
    fn find_ball_returns_none_for_no_position() {
        let dir = BallDirector::new(30.0);
        let detections = [MappedDetection {
            camera: CameraId::Left,
            class_id: 0, // "ball"
            confidence: 0.9,
            camera_center: (0.5, 0.5),
            camera_size: (0.02, 0.02),
            position: None,
        }];
        let c = ctx(0, &detections);
        assert!(dir.find_ball(&c).is_none());
    }

    // ---- Detection scoring (dual-camera deduplication) ----

    #[test]
    fn detection_score_prefers_center() {
        let dir = BallDirector::new(30.0);
        let center = ball_detection_with_camera(CameraId::Left, (0.5, 0.5), 0.9, 0.3, 0.1);
        let edge = ball_detection_with_camera(CameraId::Right, (0.9, 0.9), 0.9, 0.3, 0.1);
        assert!(dir.detection_score(&center) > dir.detection_score(&edge));
    }

    #[test]
    fn last_camera_consistency_bonus() {
        let mut dir = BallDirector::new(30.0);
        dir.last_camera = Some(CameraId::Left);

        let left = ball_detection_with_camera(CameraId::Left, (0.5, 0.5), 0.9, 0.3, 0.1);
        let right = ball_detection_with_camera(CameraId::Right, (0.5, 0.5), 0.9, 0.3, 0.1);

        let diff = dir.detection_score(&left) - dir.detection_score(&right);
        assert!((diff - 0.1).abs() < 1e-6, "consistency bonus should be 0.1");
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
        dir.update(&ctx(0, &det));
        assert_eq!(dir.last_camera, Some(CameraId::Right));
    }
}
