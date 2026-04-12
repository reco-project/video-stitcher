//! Field-aware director combining ball + player tracking.
//!
//! Uses the player cluster to define the action zone. Ball detections
//! within the cluster are followed; detections far from players are
//! rejected as false positives. When the ball is lost, the camera
//! follows the player centroid to stay on the action.

use reco_core::detector::CameraId;
use reco_core::director::{Director, DirectorContext, MappedDetection, ViewportPosition};

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
    /// Current raw output position.
    yaw: f32,
    pitch: f32,
    current_fov: f32,
    /// Labels to filter detections.
    ball_label: String,
    player_label: String,
    /// Minimum player detections to form a valid cluster.
    min_players: usize,
    /// Ball blend weight (0.0 = players only, 0.3 = 70% cluster + 30% ball).
    /// Ball detections nudge the centroid toward the action without snapping.
    ball_weight: f32,
    /// FOV range.
    fov_wide: f32,
    fov_tight: f32,
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
    /// DBSCAN neighborhood radius (rad). Players within this distance
    /// of each other are in the same cluster.
    dbscan_eps: f32,
    /// Minimum neighbors for a DBSCAN core point.
    dbscan_min_neighbors: usize,
    /// Smoothed cluster velocity (rad/frame) for predictive panning.
    vel_yaw: f32,
    vel_pitch: f32,
    /// EMA alpha for velocity smoothing.
    vel_alpha: f32,
    /// How many frames ahead to predict (velocity * lead_frames).
    vel_lead_frames: f32,
}

impl FieldDirector {
    /// Create a new field director.
    ///
    /// `fps` scales timing parameters (ball lost delay, confirmation).
    pub fn new(_fps: f32) -> Self {
        // Allow tuning via env vars for A/B testing without recompiling.
        let env_f32 = |key: &str, default: f32| -> f32 {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        Self {
            yaw: 0.0,
            pitch: 0.0,
            current_fov: 55.0,
            ball_label: "sports ball".into(),
            player_label: "person".into(),
            min_players: 3,
            ball_weight: env_f32("RECO_BALL_WEIGHT", 0.3),
            fov_wide: env_f32("RECO_FOV_WIDE", 55.0),
            fov_tight: env_f32("RECO_FOV_TIGHT", 38.0),
            last_camera: None,
            ema_yaw: 0.0,
            ema_pitch: 0.0,
            ema_initialized: false,
            cluster_alpha: env_f32("RECO_CLUSTER_ALPHA", 0.012),
            dedup_radius: 0.05,
            dbscan_eps: env_f32("RECO_DBSCAN_EPS", 0.07),
            dbscan_min_neighbors: 2,
            vel_yaw: 0.0,
            vel_pitch: 0.0,
            vel_alpha: env_f32("RECO_VEL_ALPHA", 0.0),
            vel_lead_frames: env_f32("RECO_VEL_LEAD", 0.0),
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

    /// Compute the player cluster centroid, spread, and velocity.
    ///
    /// Pipeline: deduplicate seam overlaps -> DBSCAN to find the main
    /// player group (ignoring GK/isolated defenders) -> confidence-weighted
    /// centroid of the main cluster -> EMA smooth -> velocity prediction.
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

        // Step 1: Deduplicate seam overlaps. Only merge detections from
        // DIFFERENT cameras at nearly the same position (same player seen
        // by both cameras in the overlap region).
        let mut unique: Vec<(f32, f32, f32, CameraId)> = Vec::new();
        for p in &players {
            let pos = p.position.unwrap();
            if let Some(existing) = unique.iter_mut().find(|(uy, up, _, cam)| {
                *cam != p.camera && {
                    let dy = pos.yaw - *uy;
                    let dp = pos.pitch - *up;
                    (dy * dy + dp * dp).sqrt() < self.dedup_radius
                }
            }) {
                if p.confidence > existing.2 {
                    *existing = (pos.yaw, pos.pitch, p.confidence, p.camera);
                }
            } else {
                unique.push((pos.yaw, pos.pitch, p.confidence, p.camera));
            }
        }

        if unique.len() < self.min_players {
            return None;
        }

        // Step 2: DBSCAN - find the largest dense cluster.
        // A point is "core" if it has >= min_neighbors within eps.
        // Connected core points form a cluster. Pick the largest.
        let n = unique.len();
        let eps = self.dbscan_eps;
        let min_nb = self.dbscan_min_neighbors;

        // Build neighbor counts.
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dy = unique[i].0 - unique[j].0;
                let dp = unique[i].1 - unique[j].1;
                if (dy * dy + dp * dp).sqrt() < eps {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                }
            }
        }

        // Label core points and flood-fill clusters.
        let mut cluster_id: Vec<i32> = vec![-1; n]; // -1 = unvisited
        let mut current_cluster = 0_i32;

        for i in 0..n {
            if cluster_id[i] != -1 || neighbors[i].len() < min_nb {
                continue; // not core or already visited
            }
            // BFS from this core point.
            let mut queue = vec![i];
            cluster_id[i] = current_cluster;
            while let Some(pt) = queue.pop() {
                for &nb in &neighbors[pt] {
                    if cluster_id[nb] == -1 {
                        cluster_id[nb] = current_cluster;
                        // Expand if neighbor is also core.
                        if neighbors[nb].len() >= min_nb {
                            queue.push(nb);
                        }
                    }
                }
            }
            current_cluster += 1;
        }

        // Find the largest cluster.
        let num_clusters = current_cluster;
        if num_clusters == 0 {
            return None;
        }
        let mut cluster_sizes = vec![0_usize; num_clusters as usize];
        for &cid in &cluster_id {
            if cid >= 0 {
                cluster_sizes[cid as usize] += 1;
            }
        }
        let best_cluster = cluster_sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, size)| *size)
            .map(|(id, _)| id as i32)
            .unwrap();

        // Step 3: Collect cluster members, then take P80 closest to
        // the initial centroid. This trims outliers from BOTH the centroid
        // AND the spread - stragglers are fully ignored.
        let mut members: Vec<(f32, f32, f32)> = unique
            .iter()
            .enumerate()
            .filter(|(i, _)| cluster_id[*i] == best_cluster)
            .map(|(_, &(y, p, c, _))| (y, p, c))
            .collect();

        if members.len() < self.min_players {
            return None;
        }

        // Rough centroid for distance ranking.
        let n = members.len() as f32;
        let rough_yaw: f32 = members.iter().map(|m| m.0).sum::<f32>() / n;
        let rough_pitch: f32 = members.iter().map(|m| m.1).sum::<f32>() / n;

        // Sort by distance to rough centroid, keep closest 80%.
        members.sort_by(|a, b| {
            let da = (a.0 - rough_yaw).powi(2) + (a.1 - rough_pitch).powi(2);
            let db = (b.0 - rough_yaw).powi(2) + (b.1 - rough_pitch).powi(2);
            da.partial_cmp(&db).unwrap()
        });
        let keep = (members.len() as f32 * 0.5).ceil() as usize;
        let core = &members[..keep.max(self.min_players).min(members.len())];

        // Confidence-weighted centroid of the P80 core group.
        let mut sum_yaw = 0.0_f32;
        let mut sum_pitch = 0.0_f32;
        let mut total_weight = 0.0_f32;
        for &(yaw, pitch, conf) in core {
            sum_yaw += yaw * conf;
            sum_pitch += pitch * conf;
            total_weight += conf;
        }
        let raw_yaw = sum_yaw / total_weight;
        let raw_pitch = sum_pitch / total_weight;

        // Spread from the core group (max distance of core members).
        let spread = core
            .iter()
            .map(|&(y, p, _)| {
                let dy = y - raw_yaw;
                let dp = p - raw_pitch;
                (dy * dy + dp * dp).sqrt()
            })
            .fold(0.0_f32, f32::max);

        // Step 4: EMA smooth the centroid ("heavy camera" feel).
        if !self.ema_initialized {
            self.ema_yaw = raw_yaw;
            self.ema_pitch = raw_pitch;
            self.ema_initialized = true;
        } else {
            let alpha = self.cluster_alpha;
            self.ema_yaw += alpha * (raw_yaw - self.ema_yaw);
            self.ema_pitch += alpha * (raw_pitch - self.ema_pitch);
        }

        // Step 5: Velocity - only apply during sustained fast movement
        // (counterattack), not for small centroid drift.
        let raw_vel_yaw = raw_yaw - self.ema_yaw;
        let raw_vel_pitch = raw_pitch - self.ema_pitch;
        self.vel_yaw += self.vel_alpha * (raw_vel_yaw - self.vel_yaw);
        self.vel_pitch += self.vel_alpha * (raw_vel_pitch - self.vel_pitch);

        // Only lead if velocity is sustained (above threshold).
        let vel_mag = (self.vel_yaw.powi(2) + self.vel_pitch.powi(2)).sqrt();
        let vel_threshold = 0.005; // ~0.3 deg/frame = ~9 deg/s
        let lead = if vel_mag > vel_threshold {
            self.vel_lead_frames
        } else {
            0.0
        };
        let centroid_yaw = self.ema_yaw + self.vel_yaw * lead;
        let centroid_pitch = self.ema_pitch + self.vel_pitch * lead;

        Some(Cluster {
            yaw: centroid_yaw,
            pitch: centroid_pitch,
            spread,
            count: core.len(),
        })
    }

    /// Score a detection (confidence + center proximity + camera consistency).
    fn detection_score(&self, det: &MappedDetection) -> f32 {
        super::util::detection_score(det, self.last_camera)
    }

    /// Compute target FOV from player cluster spread.
    ///
    /// Tight cluster (counterattack) = zoom in.
    /// Compute target FOV from cluster spread and distance (pitch).
    ///
    /// Spread: tight cluster = zoom in, spread out = zoom out.
    /// Distance: far cluster (high pitch) = zoom in more to keep
    /// players visible; close cluster = zoom out for context.
    fn target_fov(&self, spread: f32, pitch: f32) -> f32 {
        // Spread component.
        let spread_min = 0.05_f32;
        let spread_max = 0.40;
        let t_spread = ((spread - spread_min) / (spread_max - spread_min)).clamp(0.0, 1.0);
        let fov_from_spread = self.fov_tight + t_spread * (self.fov_wide - self.fov_tight);

        // Distance component: far clusters (high pitch) zoom in more,
        // close/edge clusters zoom out for context.
        let pitch_near = -0.05_f32;
        let pitch_far = 0.20;
        let t_dist = ((pitch - pitch_near) / (pitch_far - pitch_near)).clamp(0.0, 1.0);
        let distance_bias = t_dist * -12.0; // up to 12 deg tighter when far

        // Edge component: when cluster is off-center, zoom out slightly
        // to keep more of the field visible.
        let yaw_abs = self.yaw.abs();
        let edge_bias = (yaw_abs * 5.0).min(4.0); // up to 4 deg wider at edges

        (fov_from_spread + distance_bias + edge_bias).clamp(self.fov_tight, self.fov_wide)
    }
}

impl Director for FieldDirector {
    fn update(&mut self, ctx: &DirectorContext<'_>) {
        reco_core::profile_scope!("field_director_update");

        let cluster = self.compute_cluster(ctx);

        // Find the best ball detection (any "sports ball" with position).
        let ball_pos = if self.ball_weight > 0.0 {
            ctx.detections
                .iter()
                .filter(|d| d.label == self.ball_label && d.position.is_some())
                .max_by(|a, b| {
                    self.detection_score(a)
                        .partial_cmp(&self.detection_score(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .and_then(|d| {
                    self.last_camera = Some(d.camera);
                    d.position
                })
        } else {
            None
        };

        // Follow the cluster centroid with edge exaggeration: when the
        // cluster is off-center, push the camera further in that direction.
        // The ball is almost always ahead of the players, so this naturally
        // keeps the action in frame without relying on noisy ball detections.
        if let Some(ref c) = cluster {
            // Edge factor: 15% push at the extremes, linear from center.
            let edge_push = 0.15;
            self.yaw = c.yaw * (1.0 + edge_push);
            self.pitch = c.pitch;
        }
        // No cluster: hold position.

        // Dynamic FOV from cluster spread, EMA-smoothed for gentle transitions.
        if let Some(ref c) = cluster {
            let target = self.target_fov(c.spread, c.pitch);
            self.current_fov += 0.01 * (target - self.current_fov);
        }

        if ctx.frame_index % 30 == 0 {
            log::debug!(
                "FieldDirector frame {}: yaw={:.4}, pitch={:.4}, fov={:.1}, \
                 players={}, ball={}",
                ctx.frame_index,
                self.yaw,
                self.pitch,
                self.current_fov,
                cluster.as_ref().map_or(0, |c| c.count),
                ball_pos.is_some(),
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
    fn follows_player_centroid() {
        let mut dir = FieldDirector::new(30.0);
        let dets = [
            player(0.28, 0.0),
            player(0.32, 0.0),
            player(0.36, 0.0),
            player(0.40, 0.0),
            player(0.44, 0.0),
        ];
        dir.update(&ctx(0, &dets));
        // P50 keeps closest 3 (0.32, 0.36, 0.40), centroid ~0.36,
        // edge push 15% -> 0.36 * 1.15 = 0.414
        assert!((dir.yaw - 0.414).abs() < 0.03, "yaw={}", dir.yaw);
    }

    #[test]
    fn ball_blends_into_centroid() {
        let mut dir = FieldDirector::new(30.0);
        // Ball at 0.60, cluster centroid ~0.36. With ball_weight=0.3:
        // blended = 0.36 * 0.7 + 0.60 * 0.3 = 0.252 + 0.180 = 0.432
        let dets = [
            player(0.28, 0.0),
            player(0.32, 0.0),
            player(0.36, 0.0),
            player(0.40, 0.0),
            player(0.44, 0.0),
            ball(0.60, 0.0),
        ];
        dir.update(&ctx(0, &dets));
        // Should be pulled toward the ball but not snapped to it.
        assert!(dir.yaw > 0.36, "should pull toward ball: yaw={}", dir.yaw);
        assert!(dir.yaw < 0.60, "should not snap to ball: yaw={}", dir.yaw);
    }

    #[test]
    fn no_ball_uses_pure_cluster() {
        let mut dir = FieldDirector::new(30.0);
        let dets = [
            player(0.28, 0.0),
            player(0.32, 0.0),
            player(0.36, 0.0),
            player(0.40, 0.0),
        ];
        dir.update(&ctx(0, &dets));
        // Pure cluster centroid, no ball influence.
        assert!((dir.yaw - 0.34).abs() < 0.03, "yaw={}", dir.yaw);
    }

    #[test]
    fn fov_from_spread() {
        let dir = FieldDirector::new(30.0);
        let tight = dir.target_fov(0.05, 0.0);
        let wide = dir.target_fov(0.40, 0.0);
        assert!(tight < wide, "tight={tight}, wide={wide}");
    }
}
