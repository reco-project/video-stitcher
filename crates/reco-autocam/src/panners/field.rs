//! Field-aware panner following the densest player cluster,
//! optionally blending the ball position.
//!
//! # Pipeline (per `decide` call)
//!
//! 1. Take the current-frame tracked players from `world.players`.
//!    The tracker already enforces class filtering and stable IDs;
//!    entities in [`TrackState::Lost`] are dropped before clustering.
//! 2. Huber-weighted robust centroid on (yaw, pitch) in panorama
//!    space. Outliers (goalkeeper, substitutes) get down-weighted
//!    smoothly via IRLS rather than binary-accepted or -rejected,
//!    so frame-to-frame cluster membership flips can no longer
//!    teleport the centroid.
//! 3. Confidence-weighted centroid + EMA smoothing.
//! 4. Edge-push (yaw *= 1.15) exaggerates side-of-pitch motion so
//!    the camera leads into the direction of play.
//! 5. Optional ball blend: weighted linear combination with
//!    `world.ball.yaw/pitch` when both are available.
//! 6. Dynamic FOV from cluster spread, pitch (distance proxy), and
//!    absolute yaw (panorama-edge bias), all EMA-smoothed.
//!
//! # Logging
//!
//! Following reco's explicit-decision principle: every behavior
//! branch (ball blend vs cluster-only, cluster lost, NaN rejection)
//! emits a log line so an operator can reconstruct the camera's
//! decisions from the log alone.

use reco_core::director::ViewportPosition;
use reco_core::panner::{PanContext, Panner};
use reco_core::tracker::{TrackState, TrackedEntity, WorldState};

const LOG_INTERVAL: u64 = 30;

/// All tunable parameters for the field panner.
///
/// Safe defaults are tuned for football (soccer) at 30fps. Override
/// individual fields for other sports or frame rates.
#[derive(Debug, Clone)]
pub struct FieldPannerConfig {
    pub huber_c: f32,
    pub huber_scale_floor: f32,
    pub huber_iters: usize,
    pub huber_converge_eps: f32,
    pub min_cluster: usize,
    pub edge_push: f32,
    pub fov_alpha: f32,
    pub pitch_near: f32,
    pub pitch_far: f32,
    pub distance_bias_max: f32,
    pub edge_bias_max: f32,
    pub fov_tight: f32,
    pub fov_wide: f32,
    pub fov_default: f32,
    pub cluster_alpha: f32,
    pub max_velocity_rad_per_sec: f32,
    pub velocity_alpha: f32,
    pub pitch_bias: f32,
    pub ball_presence_decay: f32,
    pub ball_presence_attack: f32,
    pub velocity_fov_bias_max: f32,
    pub ball_frame_margin_deg: f32,
    pub ball_max_dist_from_cluster: f32,
    pub ball_weight: f32,
}

impl Default for FieldPannerConfig {
    fn default() -> Self {
        Self {
            huber_c: 1.1,
            huber_scale_floor: 0.02,
            huber_iters: 8,
            huber_converge_eps: 1e-6,
            min_cluster: 2,
            edge_push: 0.15,
            fov_alpha: 0.01,
            pitch_near: -0.05,
            pitch_far: 0.20,
            distance_bias_max: -12.0,
            edge_bias_max: 4.0,
            fov_tight: 22.0,
            fov_wide: 58.0,
            fov_default: 40.0,
            cluster_alpha: 0.012,
            max_velocity_rad_per_sec: 0.18,
            velocity_alpha: 0.06,
            pitch_bias: 0.05,
            ball_presence_decay: 0.99,
            ball_presence_attack: 0.15,
            velocity_fov_bias_max: 10.0,
            ball_frame_margin_deg: 3.0,
            ball_max_dist_from_cluster: 0.85,
            ball_weight: 0.0,
        }
    }
}

/// Intermediate cluster descriptor produced by the pipeline.
struct Cluster {
    yaw: f32,
    pitch: f32,
    spread: f32,
    count: usize,
}

/// Field-aware panner that follows the densest group of tracked
/// players, with optional ball blending and dynamic FOV.
pub struct FieldPanner {
    config: FieldPannerConfig,
    yaw: f32,
    pitch: f32,
    current_fov: f32,
    ema_yaw: f32,
    ema_pitch: f32,
    ema_initialized: bool,
    velocity_yaw: f32,
    velocity_pitch: f32,
    max_velocity: f32,
    ball_presence: f32,
    last_ball_yaw: f32,
    last_ball_pitch: f32,
    frame_index: u64,
}

impl FieldPanner {
    pub fn new(fps: f32) -> Self {
        Self::with_config(fps, FieldPannerConfig::default())
    }

    pub fn with_config(fps: f32, config: FieldPannerConfig) -> Self {
        let fps = fps.clamp(1.0, 1000.0);
        let current_fov = config.fov_default;
        let max_velocity = config.max_velocity_rad_per_sec / fps;
        Self {
            config,
            yaw: 0.0,
            pitch: 0.0,
            current_fov,
            ema_yaw: 0.0,
            ema_pitch: 0.0,
            ema_initialized: false,
            velocity_yaw: 0.0,
            velocity_pitch: 0.0,
            max_velocity,
            ball_presence: 0.0,
            last_ball_yaw: 0.0,
            last_ball_pitch: 0.0,
            frame_index: 0,
        }
    }

    pub fn with_ball_weight(mut self, weight: f32) -> Self {
        self.config.ball_weight = weight.clamp(0.0, 1.0);
        self
    }

    pub fn with_fov_range(mut self, tight: f32, wide: f32) -> Self {
        self.config.fov_tight = tight;
        self.config.fov_wide = wide;
        self
    }

    pub fn with_cluster_alpha(mut self, alpha: f32) -> Self {
        self.config.cluster_alpha = alpha.clamp(0.001, 1.0);
        self
    }

    /// Convert live tracked players to `(yaw, pitch, confidence)`
    /// tuples for the clustering pipeline.
    ///
    /// Lost entities are dropped — the tracker may report them for
    /// one final frame so consumers can observe the transition, but
    /// they have no meaningful centroid contribution. Class and
    /// cross-camera deduplication are already the tracker's job;
    /// the panner simply trusts the IDs.
    fn to_points(&self, players: &[TrackedEntity]) -> Vec<(f32, f32, f32)> {
        players
            .iter()
            .filter(|p| !matches!(p.state, TrackState::Lost))
            .map(|p| (p.yaw, p.pitch, p.confidence))
            .collect()
    }

    /// Huber-weighted robust centroid. Returns the inlier set
    /// (weight=1) after IRLS convergence.
    ///
    /// Replaces the older DBSCAN + trim-to-closest-half pipeline.
    /// The old one binary-accepted or -rejected each point based
    /// on neighborhood density, so a marginal player on the
    /// cluster's boundary could flip in / out between frames and
    /// teleport the centroid. Huber weights shrink smoothly with
    /// residual, so a boundary point's influence changes by 1% per
    /// frame instead of 100%.
    fn cluster_and_trim(&self, points: &[(f32, f32, f32)]) -> Vec<(f32, f32, f32)> {
        if points.len() < self.config.min_cluster {
            return Vec::new();
        }

        // Seed centroid at the confidence-weighted mean.
        let total_conf: f32 = points.iter().map(|(_, _, c)| c).sum();
        if total_conf <= 0.0 {
            return Vec::new();
        }
        let mut cy: f32 = points.iter().map(|(y, _, c)| y * c).sum::<f32>() / total_conf;
        let mut cp: f32 = points.iter().map(|(_, p, c)| p * c).sum::<f32>() / total_conf;

        // IRLS: residual -> robust scale (MAD * 1.4826) -> Huber
        // weight * confidence -> new centroid. 8 iterations is
        // plenty for Huber; early-exit on sub-epsilon centroid shift.
        let mut residuals = vec![0.0_f32; points.len()];
        let mut abs_r = Vec::with_capacity(points.len());
        let mut weights = vec![0.0_f32; points.len()];
        for _ in 0..self.config.huber_iters {
            for (i, &(y, p, _)) in points.iter().enumerate() {
                let dy = y - cy;
                let dp = p - cp;
                residuals[i] = (dy * dy + dp * dp).sqrt();
            }
            abs_r.clear();
            abs_r.extend_from_slice(&residuals);
            abs_r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = abs_r[abs_r.len() / 2];
            let scale = (1.4826 * mad).max(self.config.huber_scale_floor);
            let threshold = self.config.huber_c * scale;

            let mut total_w = 0.0_f32;
            for (i, &(_, _, conf)) in points.iter().enumerate() {
                let r = residuals[i];
                let huber = if r <= threshold {
                    1.0
                } else {
                    threshold / r.max(1e-6)
                };
                let w = conf * huber;
                weights[i] = w;
                total_w += w;
            }
            if total_w <= 0.0 {
                break;
            }

            let new_cy: f32 = points
                .iter()
                .zip(weights.iter())
                .map(|((y, _, _), w)| y * w)
                .sum::<f32>()
                / total_w;
            let new_cp: f32 = points
                .iter()
                .zip(weights.iter())
                .map(|((_, p, _), w)| p * w)
                .sum::<f32>()
                / total_w;

            let shift_sq = (new_cy - cy).powi(2) + (new_cp - cp).powi(2);
            cy = new_cy;
            cp = new_cp;
            if shift_sq < self.config.huber_converge_eps {
                break;
            }
        }

        // Final inlier set: points whose residual at the converged
        // centroid is within the robust band.
        for (i, &(y, p, _)) in points.iter().enumerate() {
            let dy = y - cy;
            let dp = p - cp;
            residuals[i] = (dy * dy + dp * dp).sqrt();
        }
        abs_r.clear();
        abs_r.extend_from_slice(&residuals);
        abs_r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = abs_r[abs_r.len() / 2];
        let scale = (1.4826 * mad).max(self.config.huber_scale_floor);
        let threshold = self.config.huber_c * scale;

        let inliers: Vec<(f32, f32, f32)> = points
            .iter()
            .zip(residuals.iter())
            .filter_map(|(&pt, &r)| if r <= threshold { Some(pt) } else { None })
            .collect();

        if inliers.len() < self.config.min_cluster {
            return Vec::new();
        }
        inliers
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

        // First-stage EMA: smooths step changes in the raw centroid
        // into ramps. The output EMA (POSE_ALPHA) then smooths the
        // ramps further, naturally bounding acceleration. Two cascaded
        // EMAs = second-order filter with smooth accel/decel.
        if !self.ema_initialized {
            self.ema_yaw = raw_yaw;
            self.ema_pitch = raw_pitch;
            self.ema_initialized = true;
        } else {
            self.ema_yaw += self.config.cluster_alpha * (raw_yaw - self.ema_yaw);
            self.ema_pitch += self.config.cluster_alpha * (raw_pitch - self.ema_pitch);
        }

        (self.ema_yaw, self.ema_pitch)
    }

    fn compute_cluster(&mut self, players: &[TrackedEntity]) -> Option<Cluster> {
        let points = self.to_points(players);
        let core = self.cluster_and_trim(&points);
        if core.is_empty() {
            return None;
        }
        let (centroid_yaw, centroid_pitch) = self.smooth_centroid(&core);
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
}

impl FieldPanner {
    /// Dynamic FOV: player spread + distance + edge + velocity biases,
    /// then widened if needed to keep the ball in frame.
    fn target_fov(&self, spread: f32, pitch: f32, velocity_mag: f32) -> f32 {
        let spread_deg = spread.to_degrees();
        let fov_from_spread = (2.0 * spread_deg).max(self.config.fov_tight);

        let t_dist = ((pitch - self.config.pitch_near) / (self.config.pitch_far - self.config.pitch_near)).clamp(0.0, 1.0);
        let distance_bias = t_dist * self.config.distance_bias_max;

        let edge_bias = (self.yaw.abs() * 5.0).min(self.config.edge_bias_max);

        let vel_ratio = (velocity_mag / self.max_velocity).clamp(0.0, 1.0);
        let velocity_bias = vel_ratio * self.config.velocity_fov_bias_max;

        let mut fov = fov_from_spread + distance_bias + edge_bias + velocity_bias;

        if self.ball_presence > 0.01 {
            let ball_offset = ((self.last_ball_yaw - self.yaw).powi(2)
                + (self.last_ball_pitch - self.pitch).powi(2))
            .sqrt()
            .to_degrees();
            let needed = (ball_offset + self.config.ball_frame_margin_deg) * 2.0;
            fov = fov.max(needed);
        }

        fov.clamp(self.config.fov_tight, self.config.fov_wide)
    }
}

impl Default for FieldPanner {
    fn default() -> Self {
        Self::new(30.0)
    }
}

impl Panner for FieldPanner {
    fn decide(&mut self, world: &WorldState, _ctx: &PanContext<'_>) -> ViewportPosition {
        reco_core::profile_scope!("field_panner_decide");

        self.frame_index = self.frame_index.wrapping_add(1);
        let cluster = self.compute_cluster(&world.players);

        let ball_detected = self.config.ball_weight > 0.0
            && world
                .ball
                .as_ref()
                .is_some_and(|b| !matches!(b.state, TrackState::Lost));

        let ball_near_cluster = ball_detected
            && cluster.as_ref().is_some_and(|c| {
                let b = world.ball.as_ref().unwrap();
                let dist = ((b.yaw - c.yaw).powi(2) + (b.pitch - c.pitch).powi(2)).sqrt();
                dist < self.config.ball_max_dist_from_cluster
            });

        if ball_near_cluster {
            let b = world.ball.as_ref().unwrap();
            self.last_ball_yaw = b.yaw;
            self.last_ball_pitch = b.pitch;
            self.ball_presence += self.config.ball_presence_attack * (1.0 - self.ball_presence);
        } else {
            self.ball_presence *= self.config.ball_presence_decay;
        }
        self.ball_presence = self.ball_presence.clamp(0.0, 1.0);

        if let Some(ref c) = cluster {
            let mut target_yaw = c.yaw * (1.0 + self.config.edge_push);
            let mut target_pitch = c.pitch + self.config.pitch_bias;

            let effective_w = self.config.ball_weight * self.ball_presence;
            if effective_w > 0.001 {
                target_yaw = target_yaw * (1.0 - effective_w) + self.last_ball_yaw * effective_w;
                target_pitch =
                    target_pitch * (1.0 - effective_w) + self.last_ball_pitch * effective_w;
            }

            if target_yaw.is_finite() && target_pitch.is_finite() {
                let err_yaw = target_yaw - self.yaw;
                let err_pitch = target_pitch - self.pitch;

                let desired_yaw = err_yaw.clamp(-self.max_velocity, self.max_velocity);
                let desired_pitch = err_pitch.clamp(-self.max_velocity, self.max_velocity);

                self.velocity_yaw += self.config.velocity_alpha * (desired_yaw - self.velocity_yaw);
                self.velocity_pitch += self.config.velocity_alpha * (desired_pitch - self.velocity_pitch);

                self.yaw += self.velocity_yaw;
                self.pitch += self.velocity_pitch;
            }

            let vel_mag = (self.velocity_yaw.powi(2) + self.velocity_pitch.powi(2)).sqrt();
            let target_fov = self.target_fov(c.spread, c.pitch, vel_mag);
            if target_fov.is_finite() {
                self.current_fov += self.config.fov_alpha * (target_fov - self.current_fov);
            } else {
                log::warn!(
                    "FieldPanner: non-finite FOV target ({target_fov}) from \
                     spread={} pitch={}; keeping current_fov={}",
                    c.spread,
                    c.pitch,
                    self.current_fov,
                );
            }
        } else {
            log::trace!(
                "FieldPanner: no cluster this frame (players={}, min={})",
                world.players.len(),
                self.config.min_cluster
            );
        }

        if self.frame_index.is_multiple_of(LOG_INTERVAL) {
            log::debug!(
                "FieldPanner frame {}: yaw={:.4} pitch={:.4} fov={:.1} players={} spread={:.3} world_players={} ball_blend={}",
                self.frame_index,
                self.yaw,
                self.pitch,
                self.current_fov,
                cluster.as_ref().map_or(0, |c| c.count),
                cluster.as_ref().map_or(0.0, |c| c.spread),
                world.players.len(),
                self.ball_presence > 0.001
            );
        }

        ViewportPosition {
            yaw: self.yaw,
            pitch: self.pitch,
            fov_degrees: Some(self.current_fov),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detector::CameraId;

    fn player(yaw: f32, pitch: f32, id: u64) -> TrackedEntity {
        TrackedEntity {
            id,
            class_id: 0,
            yaw,
            pitch,
            confidence: 0.9,
            state: TrackState::Tracking,
            age_frames: 5,
            origin: CameraId::Left,
        }
    }

    fn ball(yaw: f32, pitch: f32) -> TrackedEntity {
        TrackedEntity {
            id: 0,
            class_id: 32,
            yaw,
            pitch,
            confidence: 0.8,
            state: TrackState::Tracking,
            age_frames: 1,
            origin: CameraId::Left,
        }
    }

    fn cal() -> reco_core::calibration::MatchCalibration {
        use reco_core::calibration::{CameraParams, MatchCalibration, PlaneLayout};
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

    fn ctx<'a>(
        frame_index: u64,
        cal: &'a reco_core::calibration::MatchCalibration,
    ) -> PanContext<'a> {
        PanContext {
            frame_index,
            timestamp_ms: frame_index as f64 * (1000.0 / 30.0),
            previous_position: ViewportPosition::default(),
            calibration: cal,
        }
    }

    fn tight_world() -> WorldState {
        WorldState {
            ball: None,
            players: vec![
                player(0.28, 0.0, 1),
                player(0.32, 0.0, 2),
                player(0.36, 0.0, 3),
                player(0.40, 0.0, 4),
                player(0.44, 0.0, 5),
            ],
        }
    }

    #[test]
    fn follows_player_centroid() {
        let mut p = FieldPanner::new(30.0);
        let cal = cal();
        let w = tight_world();
        // Per-frame delta clamp (0.015 rad) means the pose needs many
        // frames to converge on the target. Run until it settles.
        let mut out = p.decide(&w, &ctx(0, &cal));
        for i in 1..200 {
            out = p.decide(&w, &ctx(i, &cal));
        }
        // Huber centroid on 5 evenly-spaced players at 0.28..0.44
        // converges on the mean 0.36. Edge push 15% -> 0.414.
        assert!(
            (out.yaw - 0.414).abs() < 0.03,
            "expected ~0.414, got {}",
            out.yaw
        );
    }

    #[test]
    fn no_cluster_holds_position() {
        let mut p = FieldPanner::new(30.0);
        let cal = cal();
        // Seed a pose then send an empty world — must not move.
        p.yaw = 0.3;
        p.pitch = 0.05;
        let out = p.decide(
            &WorldState {
                ball: None,
                players: vec![],
            },
            &ctx(0, &cal),
        );
        assert!((out.yaw - 0.3).abs() < 1e-6);
        assert!((out.pitch - 0.05).abs() < 1e-6);
    }

    #[test]
    fn huber_excludes_goalkeeper_outlier() {
        let mut p = FieldPanner::new(30.0);
        let cal = cal();
        let w = WorldState {
            ball: None,
            players: vec![
                player(0.30, 0.0, 1),
                player(0.34, 0.0, 2),
                player(0.38, 0.0, 3),
                player(0.42, 0.0, 4),
                player(2.0, 0.0, 99), // goalkeeper far away
            ],
        };
        let out = p.decide(&w, &ctx(0, &cal));
        assert!(
            out.yaw < 1.0,
            "goalkeeper must not drag centroid: yaw={}",
            out.yaw
        );
    }

    #[test]
    fn ball_blend_pulls_toward_ball() {
        let mut p = FieldPanner::new(30.0).with_ball_weight(0.3);
        let cal = cal();
        let mut w = tight_world();
        w.ball = Some(ball(0.80, 0.0));
        let mut out = p.decide(&w, &ctx(0, &cal));
        for i in 1..200 {
            out = p.decide(&w, &ctx(i, &cal));
        }
        // Ball at 0.80 pulls toward it, but not all the way.
        assert!(out.yaw > 0.414, "ball should pull yaw up, got {}", out.yaw);
        assert!(out.yaw < 0.80, "ball should not dominate, got {}", out.yaw);
    }

    #[test]
    fn ball_lost_ignored_in_blend() {
        let mut p = FieldPanner::new(30.0).with_ball_weight(0.3);
        let cal = cal();
        let mut w = tight_world();
        let mut lost = ball(0.80, 0.0);
        lost.state = TrackState::Lost;
        w.ball = Some(lost);
        let mut out = p.decide(&w, &ctx(0, &cal));
        for i in 1..200 {
            out = p.decide(&w, &ctx(i, &cal));
        }
        // Lost ball must not pull the centroid — output should match
        // pure-cluster output for the same players.
        assert!(
            (out.yaw - 0.414).abs() < 0.03,
            "lost ball must not pull, got {}",
            out.yaw
        );
    }

    #[test]
    fn lost_players_excluded() {
        let mut p = FieldPanner::new(30.0);
        let cal = cal();
        // Four live players at tight cluster + one lost player far
        // away. The lost one must be ignored so it can't drag the
        // centroid off the cluster.
        let mut lost = player(2.0, 0.0, 99);
        lost.state = TrackState::Lost;
        let w = WorldState {
            ball: None,
            players: vec![
                player(0.28, 0.0, 1),
                player(0.32, 0.0, 2),
                player(0.36, 0.0, 3),
                player(0.40, 0.0, 4),
                lost,
            ],
        };
        let out = p.decide(&w, &ctx(0, &cal));
        assert!(
            out.yaw < 1.0,
            "lost player must not drag centroid: yaw={}",
            out.yaw
        );
    }

    #[test]
    fn fov_narrows_for_tight_cluster() {
        let p = FieldPanner::new(30.0);
        let tight = p.target_fov(0.05, 0.0, 0.0);
        let wide = p.target_fov(0.40, 0.0, 0.0);
        assert!(tight < wide, "tight={tight} wide={wide}");
    }

    #[test]
    fn fov_tighter_when_far() {
        let p = FieldPanner::new(30.0);
        let defaults = FieldPannerConfig::default();
        let near = p.target_fov(0.20, defaults.pitch_near, 0.0);
        let far = p.target_fov(0.20, defaults.pitch_far, 0.0);
        assert!(far < near, "far={far} near={near}");
    }

    #[test]
    fn fov_ema_does_not_latch_on_nan() {
        let mut p = FieldPanner::new(30.0);
        let cal = cal();
        let baseline = p.current_fov;
        let nan_players: Vec<TrackedEntity> =
            (0..5).map(|i| player(f32::NAN, f32::NAN, i)).collect();
        let w = WorldState {
            ball: None,
            players: nan_players,
        };
        p.decide(&w, &ctx(0, &cal));
        assert!(p.current_fov.is_finite(), "FOV latched NaN");
        assert!((p.current_fov - baseline).abs() < 1e-6);
    }

    #[test]
    fn yaw_pitch_do_not_latch_on_nan() {
        let mut p = FieldPanner::new(30.0);
        let cal = cal();
        p.yaw = 0.3;
        p.pitch = 0.05;
        let nan_players: Vec<TrackedEntity> =
            (0..5).map(|i| player(f32::NAN, f32::NAN, i)).collect();
        let w = WorldState {
            ball: None,
            players: nan_players,
        };
        p.decide(&w, &ctx(0, &cal));
        assert!(p.yaw.is_finite());
        assert!(p.pitch.is_finite());
        assert!((p.yaw - 0.3).abs() < 1e-6);
        assert!((p.pitch - 0.05).abs() < 1e-6);
    }

    #[test]
    fn position_includes_fov() {
        let mut p = FieldPanner::new(30.0);
        let cal = cal();
        let out = p.decide(&tight_world(), &ctx(0, &cal));
        assert!(out.fov_degrees.is_some());
    }
}
