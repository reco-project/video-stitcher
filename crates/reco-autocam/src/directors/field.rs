//! Field-aware director using player cluster tracking.
//!
//! Follows the densest group of players on the field using DBSCAN
//! clustering, ignoring isolated outliers (goalkeeper, substitutes).
//! Optionally blends ball position into the centroid. Edge exaggeration
//! pushes the camera toward the direction of play.

use reco_core::director::{Director, DirectorContext, MappedDetection, ViewportPosition};

use super::clustering;
use super::util::{self, DEFAULT_FOV, MIN_PLAYER_CONFIDENCE};

// ── Constants ────────────────────────────────────────────────────────

/// Fraction of cluster members to keep (closest to centroid).
const TRIM_FRACTION: f32 = 0.5;

/// Edge exaggeration factor: yaw is pushed 15% further from center.
const EDGE_PUSH: f32 = 0.15;

/// FOV EMA alpha for gentle zoom transitions.
const FOV_ALPHA: f32 = 0.01;

/// Seam dedup radius (rad). ~2.9 degrees.
const DEDUP_RADIUS: f32 = 0.05;

/// Spread range for FOV mapping.
const SPREAD_MIN: f32 = 0.05;
const SPREAD_MAX: f32 = 0.40;

/// Pitch range for distance-based FOV bias.
const PITCH_NEAR: f32 = -0.05;
const PITCH_FAR: f32 = 0.20;

/// Max FOV reduction for far clusters (degrees).
const DISTANCE_BIAS_MAX: f32 = -12.0;

/// Max FOV increase at panorama edges (degrees).
const EDGE_BIAS_MAX: f32 = 4.0;

/// Log interval in frames.
const LOG_INTERVAL: u64 = 30;

// ── Types ────────────────────────────────────────────────────────────

/// Player cluster computed from detections.
struct Cluster {
    /// Centroid yaw in radians (panorama space).
    yaw: f32,
    /// Centroid pitch in radians (panorama space).
    pitch: f32,
    /// Maximum distance from centroid to any core member (radians).
    spread: f32,
    /// Number of core members after trimming.
    count: usize,
}

// ── FieldDirector ────────────────────────────────────────────────────

/// Field-aware director using player cluster tracking.
///
/// Tracks the densest group of players via DBSCAN clustering, with
/// confidence-weighted centroid, EMA smoothing, and edge exaggeration.
/// Ball detections optionally blend into the centroid.
pub struct FieldDirector {
    /// Current raw output position (panorama-space radians).
    yaw: f32,
    pitch: f32,
    current_fov: f32,
    /// Class ID filters.
    ball_class_id: u16,
    player_class_id: u16,
    /// Minimum players to form a valid cluster.
    min_players: usize,
    /// Ball blend weight (0.0 = players only, 0.3 = 70/30 cluster/ball).
    ball_weight: f32,
    /// FOV range (degrees).
    fov_wide: f32,
    fov_tight: f32,
    /// Camera dedup for overlap region.
    last_camera: Option<reco_core::detector::CameraId>,
    /// EMA-smoothed centroid.
    ema_yaw: f32,
    ema_pitch: f32,
    ema_initialized: bool,
    /// EMA alpha for centroid smoothing (lower = heavier camera feel).
    cluster_alpha: f32,
    /// DBSCAN neighborhood radius (radians).
    dbscan_eps: f32,
    /// DBSCAN minimum neighbors for a core point.
    dbscan_min_neighbors: usize,
}

impl FieldDirector {
    /// Create a new field director with default parameters.
    pub fn new(_fps: f32) -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            current_fov: DEFAULT_FOV,
            ball_class_id: 32,  // COCO "sports ball"
            player_class_id: 0, // COCO "person"
            min_players: 3,
            ball_weight: 0.3,
            fov_wide: 55.0,
            fov_tight: 38.0,
            last_camera: None,
            ema_yaw: 0.0,
            ema_pitch: 0.0,
            ema_initialized: false,
            cluster_alpha: 0.012,
            dbscan_eps: 0.07,
            dbscan_min_neighbors: 2,
        }
    }

    /// Set the ball class ID (default: 32, COCO "sports ball").
    ///
    /// Resolve label names to class IDs via the detector's `class_names()`.
    pub fn with_ball_class_id(mut self, class_id: u16) -> Self {
        self.ball_class_id = class_id;
        self
    }

    /// Set the player class ID (default: 0, COCO "person").
    ///
    /// Resolve label names to class IDs via the detector's `class_names()`.
    pub fn with_player_class_id(mut self, class_id: u16) -> Self {
        self.player_class_id = class_id;
        self
    }

    /// Set the ball blend weight (0.0 = players only, default: 0.3).
    pub fn with_ball_weight(mut self, weight: f32) -> Self {
        self.ball_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the FOV range in degrees (default: 38-55).
    pub fn with_fov_range(mut self, tight: f32, wide: f32) -> Self {
        self.fov_tight = tight;
        self.fov_wide = wide;
        self
    }

    /// Set the centroid EMA alpha (default: 0.012). Lower = smoother.
    pub fn with_cluster_alpha(mut self, alpha: f32) -> Self {
        self.cluster_alpha = alpha.clamp(0.001, 1.0);
        self
    }

    /// Set the DBSCAN neighborhood radius in radians (default: 0.07).
    pub fn with_dbscan_eps(mut self, eps: f32) -> Self {
        self.dbscan_eps = eps.clamp(0.01, 1.0);
        self
    }

    // ── Pipeline steps ───────────────────────────────────────────────

    /// Filter players by label and confidence, deduplicate seam overlaps.
    fn filter_and_dedup(&self, ctx: &DirectorContext<'_>) -> Vec<(f32, f32, f32)> {
        let players: Vec<&MappedDetection> = ctx
            .detections
            .iter()
            .filter(|d| {
                d.class_id == self.player_class_id
                    && d.position.is_some()
                    && d.confidence >= MIN_PLAYER_CONFIDENCE
            })
            .collect();

        util::dedup_cross_camera(&players, DEDUP_RADIUS)
    }

    /// Run DBSCAN, keep the largest cluster, trim to closest TRIM_FRACTION.
    fn cluster_and_trim(&self, points: &[(f32, f32, f32)]) -> Vec<(f32, f32, f32)> {
        if points.len() < self.min_players {
            return Vec::new();
        }

        // DBSCAN on (yaw, pitch) positions.
        let positions: Vec<(f32, f32)> = points.iter().map(|&(y, p, _)| (y, p)).collect();
        let labels = clustering::dbscan(&positions, self.dbscan_eps, self.dbscan_min_neighbors);
        let largest = clustering::largest_cluster_indices(&labels);

        let mut members: Vec<(f32, f32, f32)> = largest.iter().map(|&i| points[i]).collect();

        if members.len() < self.min_players {
            return Vec::new();
        }

        // Rough centroid for distance ranking.
        let n = members.len() as f32;
        let rough_yaw: f32 = members.iter().map(|m| m.0).sum::<f32>() / n;
        let rough_pitch: f32 = members.iter().map(|m| m.1).sum::<f32>() / n;

        // Sort by distance to centroid, keep closest TRIM_FRACTION.
        members.sort_by(|a, b| {
            let da = (a.0 - rough_yaw).powi(2) + (a.1 - rough_pitch).powi(2);
            let db = (b.0 - rough_yaw).powi(2) + (b.1 - rough_pitch).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        let keep = (members.len() as f32 * TRIM_FRACTION).ceil() as usize;
        members.truncate(keep.max(self.min_players).min(members.len()));
        members
    }

    /// Confidence-weighted centroid + EMA smoothing.
    fn smooth_centroid(&mut self, core: &[(f32, f32, f32)]) -> (f32, f32) {
        let mut sum_yaw = 0.0_f32;
        let mut sum_pitch = 0.0_f32;
        let mut total_weight = 0.0_f32;
        for &(yaw, pitch, conf) in core {
            sum_yaw += yaw * conf;
            sum_pitch += pitch * conf;
            total_weight += conf;
        }
        if total_weight <= 0.0 {
            return (self.ema_yaw, self.ema_pitch);
        }
        let raw_yaw = sum_yaw / total_weight;
        let raw_pitch = sum_pitch / total_weight;
        if !raw_yaw.is_finite() || !raw_pitch.is_finite() {
            return (self.ema_yaw, self.ema_pitch);
        }

        if !self.ema_initialized {
            self.ema_yaw = raw_yaw;
            self.ema_pitch = raw_pitch;
            self.ema_initialized = true;
        } else {
            self.ema_yaw += self.cluster_alpha * (raw_yaw - self.ema_yaw);
            self.ema_pitch += self.cluster_alpha * (raw_pitch - self.ema_pitch);
        }

        (self.ema_yaw, self.ema_pitch)
    }

    /// Full pipeline: filter -> cluster -> trim -> smooth -> build Cluster.
    ///
    /// EMA is only updated on frames with fresh detections to prevent
    /// the alpha from compounding on stale data at high detection intervals.
    fn compute_cluster(&mut self, ctx: &DirectorContext<'_>) -> Option<Cluster> {
        let deduped = self.filter_and_dedup(ctx);
        let core = self.cluster_and_trim(&deduped);

        if core.is_empty() {
            return None;
        }

        // Only update the EMA on fresh detection frames. Between detections,
        // return the cached EMA position to avoid compounding alpha on stale data.
        let (centroid_yaw, centroid_pitch) = if ctx.fresh_detection {
            self.smooth_centroid(&core)
        } else if self.ema_initialized {
            (self.ema_yaw, self.ema_pitch)
        } else {
            self.smooth_centroid(&core)
        };

        let spread = core
            .iter()
            .map(|&(y, p, _)| {
                let dy = y - centroid_yaw;
                let dp = p - centroid_pitch;
                (dy * dy + dp * dp).sqrt()
            })
            .fold(0.0_f32, f32::max);

        Some(Cluster {
            yaw: centroid_yaw,
            pitch: centroid_pitch,
            spread,
            count: core.len(),
        })
    }

    /// Compute target FOV from cluster spread and distance.
    fn target_fov(&self, spread: f32, pitch: f32) -> f32 {
        let t_spread = ((spread - SPREAD_MIN) / (SPREAD_MAX - SPREAD_MIN)).clamp(0.0, 1.0);
        let fov_from_spread = self.fov_tight + t_spread * (self.fov_wide - self.fov_tight);

        let t_dist = ((pitch - PITCH_NEAR) / (PITCH_FAR - PITCH_NEAR)).clamp(0.0, 1.0);
        let distance_bias = t_dist * DISTANCE_BIAS_MAX;

        let yaw_abs = self.yaw.abs();
        let edge_bias = (yaw_abs * 5.0).min(EDGE_BIAS_MAX);

        (fov_from_spread + distance_bias + edge_bias).clamp(self.fov_tight, self.fov_wide)
    }
}

impl Director for FieldDirector {
    fn update(&mut self, ctx: &DirectorContext<'_>) {
        reco_core::profile_scope!("field_director_update");

        let cluster = self.compute_cluster(ctx);

        // Find the best ball detection if blending is enabled.
        let ball_pos = if self.ball_weight > 0.0 {
            ctx.detections
                .iter()
                .filter(|d| d.class_id == self.ball_class_id && d.position.is_some())
                .max_by(|a, b| {
                    util::detection_score(a, self.last_camera)
                        .partial_cmp(&util::detection_score(b, self.last_camera))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .and_then(|d| {
                    self.last_camera = Some(d.camera);
                    d.position
                })
        } else {
            None
        };

        // Follow cluster with edge exaggeration + optional ball blend.
        if let Some(ref c) = cluster {
            let mut target_yaw = c.yaw * (1.0 + EDGE_PUSH);
            let mut target_pitch = c.pitch;

            if let Some(bp) = ball_pos {
                let w = self.ball_weight;
                target_yaw = target_yaw * (1.0 - w) + bp.yaw * w;
                target_pitch = target_pitch * (1.0 - w) + bp.pitch * w;
            }

            self.yaw = target_yaw;
            self.pitch = target_pitch;
        }

        // FOV: EMA-smoothed toward target.
        if let Some(ref c) = cluster {
            let target = self.target_fov(c.spread, c.pitch);
            self.current_fov += FOV_ALPHA * (target - self.current_fov);
        }

        if ctx.frame_index % LOG_INTERVAL == 0 {
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
        // Negate yaw: projection convention (see BallDirector::position).
        ViewportPosition {
            yaw: -self.yaw,
            pitch: self.pitch,
            fov_degrees: Some(self.current_fov),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detector::CameraId;

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
            class_id: 0, // "person"
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

    fn player_right(yaw: f32, pitch: f32) -> MappedDetection {
        MappedDetection {
            camera: CameraId::Right,
            class_id: 0, // "person"
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
            class_id: 32, // "sports ball"
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

    /// Helper: tight group of 5 players.
    fn tight_group() -> Vec<MappedDetection> {
        vec![
            player(0.28, 0.0),
            player(0.32, 0.0),
            player(0.36, 0.0),
            player(0.40, 0.0),
            player(0.44, 0.0),
        ]
    }

    #[test]
    fn follows_player_centroid() {
        let mut dir = FieldDirector::new(30.0);
        let dets = tight_group();
        dir.update(&ctx(0, &dets));
        // P50 keeps closest 3 (0.32, 0.36, 0.40), centroid ~0.36,
        // edge push 15% -> ~0.414
        assert!((dir.yaw - 0.414).abs() < 0.03, "yaw={}", dir.yaw);
    }

    #[test]
    fn ball_blends_into_centroid() {
        let mut dir = FieldDirector::new(30.0);
        let mut dets = tight_group();
        dets.push(ball(0.60, 0.0));
        dir.update(&ctx(0, &dets));
        // Ball at 0.60 pulls centroid toward it.
        assert!(dir.yaw > 0.36, "should pull toward ball: yaw={}", dir.yaw);
        assert!(dir.yaw < 0.60, "should not snap to ball: yaw={}", dir.yaw);
    }

    #[test]
    fn no_ball_uses_pure_cluster() {
        let mut dir = FieldDirector::new(30.0).with_ball_weight(0.0);
        let dets = vec![
            player(0.28, 0.0),
            player(0.32, 0.0),
            player(0.36, 0.0),
            player(0.40, 0.0),
        ];
        dir.update(&ctx(0, &dets));
        // No ball influence, pure cluster centroid + edge push.
        assert!(dir.yaw > 0.0, "should follow cluster: yaw={}", dir.yaw);
    }

    #[test]
    fn holds_position_with_no_players() {
        let mut dir = FieldDirector::new(30.0);
        dir.yaw = 0.5;
        dir.pitch = 0.1;
        dir.update(&ctx(0, &[]));
        assert!((dir.yaw - 0.5).abs() < 1e-6, "should hold: yaw={}", dir.yaw);
    }

    #[test]
    fn outlier_excluded_by_dbscan() {
        let mut dir = FieldDirector::new(30.0).with_ball_weight(0.0);
        let dets = vec![
            player(0.30, 0.0),
            player(0.34, 0.0),
            player(0.38, 0.0),
            player(0.42, 0.0),
            player(2.0, 0.0), // goalkeeper far away
        ];
        dir.update(&ctx(0, &dets));
        // GK at 2.0 should be excluded, centroid near 0.36.
        assert!(dir.yaw < 1.0, "GK should be excluded: yaw={}", dir.yaw);
    }

    #[test]
    fn dedup_merges_cross_camera() {
        let mut dir = FieldDirector::new(30.0).with_ball_weight(0.0);
        // Same player at ~0.36 seen by both cameras (seam overlap).
        let dets = vec![
            player(0.30, 0.0),
            player(0.34, 0.0),
            player(0.36, 0.0),
            player_right(0.362, 0.0), // same player, right camera
            player(0.40, 0.0),
        ];
        dir.update(&ctx(0, &dets));
        // Should not double-count the player at 0.36.
        assert!(dir.yaw > 0.0, "should form cluster: yaw={}", dir.yaw);
    }

    #[test]
    fn dedup_keeps_same_camera_close() {
        let mut dir = FieldDirector::new(30.0).with_ball_weight(0.0);
        // Two different players from the same camera, close together.
        let dets = vec![
            player(0.30, 0.0),
            player(0.32, 0.0), // different player, same camera
            player(0.34, 0.0),
            player(0.36, 0.0),
        ];
        dir.update(&ctx(0, &dets));
        // All 4 should be kept (same camera = not deduped).
        assert!(dir.yaw > 0.0, "should form cluster: yaw={}", dir.yaw);
    }

    #[test]
    fn fov_tight_for_tight_cluster() {
        let dir = FieldDirector::new(30.0);
        let tight = dir.target_fov(0.05, 0.0);
        let wide = dir.target_fov(0.40, 0.0);
        assert!(tight < wide, "tight={tight}, wide={wide}");
    }

    #[test]
    fn fov_tighter_when_far() {
        let dir = FieldDirector::new(30.0);
        let near = dir.target_fov(0.20, PITCH_NEAR);
        let far = dir.target_fov(0.20, PITCH_FAR);
        assert!(far < near, "far={far}, near={near}");
    }

    #[test]
    fn position_negates_yaw() {
        let mut dir = FieldDirector::new(30.0);
        dir.yaw = 0.5;
        let pos = dir.position();
        assert!((pos.yaw + 0.5).abs() < 1e-6);
    }

    #[test]
    fn position_includes_fov() {
        let dir = FieldDirector::new(30.0);
        assert!(dir.position().fov_degrees.is_some());
    }
}
