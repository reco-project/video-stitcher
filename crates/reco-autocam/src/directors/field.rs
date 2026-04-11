//! Field-aware director combining ball + player tracking.
//!
//! Uses the player cluster to define the action zone. Ball detections
//! within the cluster are followed; detections far from players are
//! rejected as false positives. When the ball is lost, the camera
//! follows the player centroid to stay on the action.

use reco_core::detector::CameraId;
use reco_core::director::{Director, DirectorContext, MappedDetection, ViewportPosition};

/// Director state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Ball detected near the player cluster. Camera follows ball.
    FollowingBall,
    /// Ball lost or unreliable. Camera follows player centroid.
    FollowingPlayers,
}

/// Player cluster computed from detections.
struct Cluster {
    yaw: f32,
    pitch: f32,
    spread: f32,
    count: usize,
}

/// Field-aware director combining ball and player tracking.
///
/// The player cluster defines the action zone. Ball detections are
/// validated against this zone - a ball far from any players is likely
/// a false positive (white lines, corner flags). When the ball is
/// lost, the camera smoothly follows the player centroid.
pub struct FieldDirector {
    state: State,
    /// Current raw output position.
    yaw: f32,
    pitch: f32,
    current_fov: f32,
    /// Labels to filter detections.
    ball_label: String,
    player_label: String,
    /// Minimum player detections to form a valid cluster.
    min_players: usize,
    /// Max distance (rad) from cluster centroid to accept a ball detection.
    cluster_radius: f32,
    /// Frames since ball was last seen near the cluster.
    ball_lost_frames: u32,
    /// Frames before switching from ball to player tracking.
    ball_lost_delay: u32,
    /// Consecutive frames a ball must appear near cluster to confirm.
    ball_confirm_count: u32,
    ball_confirm_needed: u32,
    /// FOV range.
    fov_wide: f32,
    fov_tight: f32,
    /// Detection interval for plausibility scaling.
    detection_interval: u32,
    /// Camera dedup (prefer same camera in overlap region).
    last_camera: Option<CameraId>,
    /// EMA-smoothed cluster centroid (stabilizes frame-to-frame jitter).
    ema_yaw: f32,
    ema_pitch: f32,
    ema_initialized: bool,
    /// EMA alpha for cluster centroid (lower = smoother, higher = responsive).
    cluster_alpha: f32,
    /// Merge distance for seam deduplication (rad). Detections from
    /// different cameras within this distance are the same person.
    dedup_radius: f32,
}

impl FieldDirector {
    /// Create a new field director.
    ///
    /// `fps` scales timing parameters (ball lost delay, confirmation).
    pub fn new(fps: f32) -> Self {
        let fps = fps.clamp(1.0, 1000.0);
        Self {
            state: State::FollowingPlayers,
            yaw: 0.0,
            pitch: 0.0,
            current_fov: 55.0,
            ball_label: "sports ball".into(),
            player_label: "person".into(),
            min_players: 3,
            cluster_radius: 0.30,
            ball_lost_frames: 0,
            ball_lost_delay: (fps * 1.0) as u32,
            ball_confirm_count: 0,
            ball_confirm_needed: 3,
            fov_wide: 60.0,
            fov_tight: 35.0,
            detection_interval: 1,
            last_camera: None,
            ema_yaw: 0.0,
            ema_pitch: 0.0,
            ema_initialized: false,
            cluster_alpha: 0.02,
            dedup_radius: 0.05, // ~2.9 degrees - typical seam overlap width
        }
    }

    /// Set the ball label (default: "sports ball").
    pub fn with_ball_label(mut self, label: impl Into<String>) -> Self {
        self.ball_label = label.into();
        self
    }

    /// Set the player label (default: "person").
    pub fn with_player_label(mut self, label: impl Into<String>) -> Self {
        self.player_label = label.into();
        self
    }

    /// Set the detection interval for plausibility scaling.
    pub fn set_detection_interval(&mut self, interval: u32) {
        self.detection_interval = interval.max(1);
    }

    /// Compute the player cluster centroid and spread.
    ///
    /// Deduplicates cross-camera detections at the seam (same player
    /// seen by both cameras), uses confidence-weighted centroid, and
    /// applies EMA smoothing to stabilize frame-to-frame jitter.
    fn compute_cluster(&mut self, ctx: &DirectorContext<'_>) -> Option<Cluster> {
        let min_confidence = 0.3;
        let players: Vec<&MappedDetection> = ctx
            .detections
            .iter()
            .filter(|d| {
                d.label == self.player_label
                    && d.position.is_some()
                    && d.confidence >= min_confidence
            })
            .collect();

        if players.len() < self.min_players {
            return None;
        }

        // Deduplicate: merge detections from different cameras at
        // nearly the same panorama position (seam overlap).
        let mut unique_players: Vec<(f32, f32, f32)> = Vec::new(); // (yaw, pitch, confidence)
        for p in &players {
            let pos = p.position.unwrap();
            let is_dup = unique_players.iter().any(|&(uy, up, _)| {
                let dy = pos.yaw - uy;
                let dp = pos.pitch - up;
                (dy * dy + dp * dp).sqrt() < self.dedup_radius
            });
            if is_dup {
                // Keep the higher-confidence version.
                if let Some(existing) = unique_players.iter_mut().find(|(uy, up, _)| {
                    let dy = pos.yaw - *uy;
                    let dp = pos.pitch - *up;
                    (dy * dy + dp * dp).sqrt() < self.dedup_radius
                }) {
                    if p.confidence > existing.2 {
                        *existing = (pos.yaw, pos.pitch, p.confidence);
                    }
                }
            } else {
                unique_players.push((pos.yaw, pos.pitch, p.confidence));
            }
        }

        if unique_players.len() < self.min_players {
            return None;
        }

        // Confidence-weighted centroid of deduplicated players.
        let mut sum_yaw = 0.0_f32;
        let mut sum_pitch = 0.0_f32;
        let mut total_weight = 0.0_f32;

        for &(yaw, pitch, conf) in &unique_players {
            sum_yaw += yaw * conf;
            sum_pitch += pitch * conf;
            total_weight += conf;
        }

        let raw_yaw = sum_yaw / total_weight;
        let raw_pitch = sum_pitch / total_weight;

        // EMA smooth the centroid to reduce frame-to-frame jitter.
        // Snap on first valid cluster (no smoothing from origin).
        if !self.ema_initialized {
            self.ema_yaw = raw_yaw;
            self.ema_pitch = raw_pitch;
            self.ema_initialized = true;
        } else {
            let alpha = self.cluster_alpha;
            self.ema_yaw += alpha * (raw_yaw - self.ema_yaw);
            self.ema_pitch += alpha * (raw_pitch - self.ema_pitch);
        }
        let centroid_yaw = self.ema_yaw;
        let centroid_pitch = self.ema_pitch;

        // Spread: 80th percentile distance from centroid (ignores outliers
        // like goalkeeper or isolated defenders far from the main group).
        let mut distances: Vec<f32> = unique_players
            .iter()
            .map(|&(y, p, _)| {
                let dy = y - centroid_yaw;
                let dp = p - centroid_pitch;
                (dy * dy + dp * dp).sqrt()
            })
            .collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p80_idx = (distances.len() as f32 * 0.8) as usize;
        let spread = distances[p80_idx.min(distances.len() - 1)];

        Some(Cluster {
            yaw: centroid_yaw,
            pitch: centroid_pitch,
            spread,
            count: unique_players.len(),
        })
    }

    /// Find the best ball detection that's within the cluster radius.
    fn find_validated_ball<'a>(
        &self,
        ctx: &'a DirectorContext<'_>,
        cluster: &Cluster,
    ) -> Option<&'a MappedDetection> {
        let balls: Vec<&MappedDetection> = ctx
            .detections
            .iter()
            .filter(|d| d.label == self.ball_label && d.position.is_some())
            .collect();

        if balls.is_empty() {
            return None;
        }

        // Filter to balls within cluster radius.
        let near_cluster: Vec<&&MappedDetection> = balls
            .iter()
            .filter(|b| {
                let pos = b.position.unwrap();
                let dy = pos.yaw - cluster.yaw;
                let dp = pos.pitch - cluster.pitch;
                (dy * dy + dp * dp).sqrt() < self.cluster_radius
            })
            .collect();

        if near_cluster.is_empty() {
            return None;
        }

        // Pick highest confidence among validated balls.
        near_cluster
            .into_iter()
            .max_by(|a, b| {
                let score_a = self.detection_score(a);
                let score_b = self.detection_score(b);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }

    /// Score a detection (confidence + center proximity + camera consistency).
    fn detection_score(&self, det: &MappedDetection) -> f32 {
        let mut score = det.confidence;
        let cx = det.camera_center.0;
        let cy = det.camera_center.1;
        let center_dist = ((cx - 0.5) * (cx - 0.5) + (cy - 0.5) * (cy - 0.5)).sqrt();
        score -= center_dist * 0.2;
        if let Some(last_camera) = self.last_camera {
            if det.camera == last_camera {
                score += 0.1;
            }
        }
        score
    }

    /// Compute target FOV from player cluster spread.
    ///
    /// Tight cluster (counterattack) = zoom in.
    /// Spread out (set piece, buildup) = zoom out.
    fn target_fov_from_spread(&self, spread: f32) -> f32 {
        // Spread range: ~0.05 (tight group) to ~0.5 (full width).
        let spread_min = 0.05_f32;
        let spread_max = 0.40;
        let t = ((spread - spread_min) / (spread_max - spread_min)).clamp(0.0, 1.0);
        self.fov_tight + t * (self.fov_wide - self.fov_tight)
    }
}

impl Director for FieldDirector {
    fn update(&mut self, ctx: &DirectorContext<'_>) {
        reco_core::profile_scope!("field_director_update");

        let cluster = self.compute_cluster(ctx);
        let ball = cluster
            .as_ref()
            .and_then(|c| self.find_validated_ball(ctx, c));

        // Track camera for dedup.
        if let Some(obj) = ball {
            self.last_camera = Some(obj.camera);
        }

        match (self.state, ball, &cluster) {
            // Following ball + ball still visible near players.
            (State::FollowingBall, Some(b), _) => {
                let pos = b.position.unwrap();
                self.yaw = pos.yaw;
                self.pitch = pos.pitch;
                self.ball_lost_frames = 0;
            }

            // Following ball + ball lost.
            (State::FollowingBall, None, Some(c)) => {
                self.ball_lost_frames += 1;
                if self.ball_lost_frames >= self.ball_lost_delay {
                    self.yaw = c.yaw;
                    self.pitch = c.pitch;
                    self.state = State::FollowingPlayers;
                    self.ball_confirm_count = 0;
                    log::debug!(
                        "FieldDirector: FollowingBall -> FollowingPlayers (ball lost {} frames)",
                        self.ball_lost_frames
                    );
                }
                // While waiting, hold position (don't jump to cluster yet).
            }

            // Following ball but no players either (unusual).
            (State::FollowingBall, None, None) => {
                self.ball_lost_frames += 1;
                // Hold position.
            }

            // Following players + ball appeared near cluster.
            (State::FollowingPlayers, Some(_), _) if ctx.fresh_detection => {
                self.ball_confirm_count += 1;
                if self.ball_confirm_count >= self.ball_confirm_needed {
                    let pos = ball.unwrap().position.unwrap();
                    self.yaw = pos.yaw;
                    self.pitch = pos.pitch;
                    self.state = State::FollowingBall;
                    self.ball_lost_frames = 0;
                    self.ball_confirm_count = 0;
                    log::debug!("FieldDirector: FollowingPlayers -> FollowingBall (confirmed)");
                } else if let Some(c) = &cluster {
                    self.yaw = c.yaw;
                    self.pitch = c.pitch;
                }
            }

            // Following players + cached detection (not fresh).
            (State::FollowingPlayers, Some(_), _) => {
                if let Some(c) = &cluster {
                    self.yaw = c.yaw;
                    self.pitch = c.pitch;
                }
            }

            // Following players, no ball.
            (State::FollowingPlayers, None, Some(c)) => {
                self.yaw = c.yaw;
                self.pitch = c.pitch;
                self.ball_confirm_count = 0;
            }

            // No players, no ball.
            (State::FollowingPlayers, None, None) => {
                self.ball_confirm_count = 0;
                // Hold position.
            }
        }

        // Per-frame trajectory trace for analysis.
        log::trace!(
            "TRAJ,{},{:.6},{:.6},{:.1},{:?},{},{:.4},{:.4}",
            ctx.frame_index,
            self.yaw,
            self.pitch,
            self.current_fov,
            self.state,
            cluster.as_ref().map_or(0, |c| c.count),
            cluster.as_ref().map_or(0.0, |c| c.yaw),
            ball.map_or(0.0, |b| b.position.unwrap().yaw),
        );

        // Dynamic FOV from cluster spread.
        self.current_fov = cluster
            .as_ref()
            .map(|c| self.target_fov_from_spread(c.spread))
            .unwrap_or(self.current_fov);

        // Log every frame at trace level, every 30 at debug.
        if ctx.frame_index % 30 == 0 {
            log::debug!(
                "FieldDirector frame {}: state={:?}, yaw={:.4}, pitch={:.4}, fov={:.1}, \
                 players={}, ball={}",
                ctx.frame_index,
                self.state,
                self.yaw,
                self.pitch,
                self.current_fov,
                cluster.as_ref().map_or(0, |c| c.count),
                ball.is_some(),
            );
        }
    }

    fn position(&self) -> ViewportPosition {
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

    fn ctx(frame_index: u64, detections: &[MappedDetection]) -> DirectorContext<'_> {
        DirectorContext {
            frame_index,
            timestamp_ms: frame_index as f64 * (1000.0 / 30.0),
            detections,
            fresh_detection: true,
        }
    }

    fn player(yaw: f32, pitch: f32) -> MappedDetection {
        MappedDetection {
            camera: CameraId::Left,
            label: "person".into(),
            confidence: 0.9,
            camera_center: (0.5, 0.5),
            camera_size: (0.05, 0.15),
            position: Some(ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            }),
        }
    }

    fn ball(yaw: f32, pitch: f32) -> MappedDetection {
        MappedDetection {
            camera: CameraId::Left,
            label: "sports ball".into(),
            confidence: 0.8,
            camera_center: (0.5, 0.5),
            camera_size: (0.02, 0.02),
            position: Some(ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            }),
        }
    }

    #[test]
    fn starts_following_players() {
        let dir = FieldDirector::new(30.0);
        assert_eq!(dir.state, State::FollowingPlayers);
    }

    #[test]
    fn follows_player_centroid() {
        let mut dir = FieldDirector::new(30.0);
        let dets = [player(0.1, 0.0), player(0.3, 0.0), player(0.5, 0.0)];
        dir.update(&ctx(0, &dets));
        assert!((dir.yaw - 0.3).abs() < 0.01, "yaw={}", dir.yaw);
    }

    #[test]
    fn switches_to_ball_when_confirmed() {
        let mut dir = FieldDirector::new(30.0);
        let dets = [
            player(0.3, 0.0),
            player(0.4, 0.0),
            player(0.5, 0.0),
            ball(0.35, 0.0),
        ];
        for i in 0..3 {
            dir.update(&ctx(i, &dets));
        }
        assert_eq!(dir.state, State::FollowingBall);
    }

    #[test]
    fn rejects_ball_far_from_players() {
        let mut dir = FieldDirector::new(30.0);
        let dets = [
            player(0.3, 0.0),
            player(0.4, 0.0),
            player(0.5, 0.0),
            ball(2.0, 0.0), // far away
        ];
        for i in 0..5 {
            dir.update(&ctx(i, &dets));
        }
        assert_eq!(dir.state, State::FollowingPlayers);
    }

    #[test]
    fn fov_from_spread() {
        let dir = FieldDirector::new(30.0);
        let tight = dir.target_fov_from_spread(0.05);
        let wide = dir.target_fov_from_spread(0.40);
        assert!(tight < wide, "tight={tight}, wide={wide}");
    }
}
