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

/// Huber tuning constant. Residuals greater than `c * scale` get
/// a weight of `c * scale / r` instead of 1. `1.345` gives ~95%
/// asymptotic efficiency under Gaussian noise (standard choice).
const HUBER_C: f32 = 1.345;

/// Floor for the robust scale estimate to avoid divide-by-tiny when
/// every player sits on top of the centroid (degenerate cluster).
/// `0.02` rad ≈ 1.15° - below that, the weight of every point
/// collapses to 1 anyway.
const HUBER_SCALE_FLOOR: f32 = 0.02;

/// IRLS iteration budget. Huber converges fast; 8 is comfortable
/// even on 20-player frames.
const HUBER_ITERS: usize = 8;

/// Residual change below this ends IRLS early (squared-distance in
/// rad^2). ~0.2% of a typical cluster spread.
const HUBER_CONVERGE_EPS: f32 = 1e-6;

/// Minimum effective cluster size. Below this, no camera update
/// happens. The effective count is the number of points whose final
/// Huber weight equals 1 (i.e., residual within the inlier band).
const MIN_CLUSTER: usize = 2;

/// Edge exaggeration factor: yaw is pushed 15% further from center.
const EDGE_PUSH: f32 = 0.15;

/// FOV EMA alpha for gentle zoom transitions.
const FOV_ALPHA: f32 = 0.01;

/// Pitch range for distance-based FOV bias.
const PITCH_NEAR: f32 = -0.05;
const PITCH_FAR: f32 = 0.20;

/// Max FOV reduction for far clusters (degrees).
const DISTANCE_BIAS_MAX: f32 = -12.0;

/// Max FOV increase at panorama edges (degrees).
const EDGE_BIAS_MAX: f32 = 4.0;

/// Default FOV envelope (degrees).
const DEFAULT_FOV_TIGHT: f32 = 22.0;
const DEFAULT_FOV_WIDE: f32 = 58.0;
const DEFAULT_FOV: f32 = 40.0;

/// EMA alpha for yaw centroid smoothing (lower = smoother).
const DEFAULT_CLUSTER_ALPHA: f32 = 0.012;

/// Maximum angular velocity in radians per second.
/// ~14 deg/s - allows faster pans on long runs across the field.
const MAX_VELOCITY_RAD_PER_SEC: f32 = 0.25;

/// Velocity smoothing alpha. Controls how quickly the camera changes
/// direction. Lower = smoother reversals, higher = snappier tracking.
const VELOCITY_ALPHA: f32 = 0.06;

/// Pitch bias added to the cluster centroid (radians).
const PITCH_BIAS: f32 = 0.05;

/// Log interval in frames.
const LOG_INTERVAL: u64 = 30;

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
    /// Current raw published pose (panorama-space radians, degrees FOV).
    yaw: f32,
    pitch: f32,
    current_fov: f32,

    /// Minimum cluster size before any camera update (default 3).
    min_players: usize,

    /// Ball blend weight (0.0 = players only, 0.1 ≈ 90/10 cluster/ball).
    ball_weight: f32,

    /// FOV envelope (degrees).
    fov_wide: f32,
    fov_tight: f32,

    /// Centroid EMA state (noise filter).
    ema_yaw: f32,
    ema_pitch: f32,
    ema_initialized: bool,

    /// EMA alpha for yaw centroid.
    cluster_alpha: f32,

    /// Current angular velocity (rad/frame) for yaw and pitch.
    velocity_yaw: f32,
    velocity_pitch: f32,

    /// Max velocity in rad/frame (derived from fps).
    max_velocity: f32,

    /// Frame counter for log throttling.
    frame_index: u64,
}

impl FieldPanner {
    /// Build a field panner with defaults (identical envelope to
    /// `FieldDirector` (predecessor)).
    pub fn new(fps: f32) -> Self {
        let fps = fps.clamp(1.0, 1000.0);
        Self {
            yaw: 0.0,
            pitch: 0.0,
            current_fov: DEFAULT_FOV,
            min_players: MIN_CLUSTER,
            ball_weight: 0.0,
            fov_wide: DEFAULT_FOV_WIDE,
            fov_tight: DEFAULT_FOV_TIGHT,
            ema_yaw: 0.0,
            ema_pitch: 0.0,
            ema_initialized: false,
            cluster_alpha: DEFAULT_CLUSTER_ALPHA,
            velocity_yaw: 0.0,
            velocity_pitch: 0.0,
            max_velocity: MAX_VELOCITY_RAD_PER_SEC / fps,
            frame_index: 0,
        }
    }

    /// Override the ball blend weight. `0.0` = pure cluster centroid;
    /// `1.0` = pure ball. Default is `0.0`.
    pub fn with_ball_weight(mut self, weight: f32) -> Self {
        self.ball_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Override the FOV envelope in degrees.
    pub fn with_fov_range(mut self, tight: f32, wide: f32) -> Self {
        self.fov_tight = tight;
        self.fov_wide = wide;
        self
    }

    /// Override the centroid EMA alpha. Lower = smoother, more lag.
    pub fn with_cluster_alpha(mut self, alpha: f32) -> Self {
        self.cluster_alpha = alpha.clamp(0.001, 1.0);
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
        if points.len() < self.min_players {
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
        for _ in 0..HUBER_ITERS {
            for (i, &(y, p, _)) in points.iter().enumerate() {
                let dy = y - cy;
                let dp = p - cp;
                residuals[i] = (dy * dy + dp * dp).sqrt();
            }
            abs_r.clear();
            abs_r.extend_from_slice(&residuals);
            abs_r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = abs_r[abs_r.len() / 2];
            let scale = (1.4826 * mad).max(HUBER_SCALE_FLOOR);
            let threshold = HUBER_C * scale;

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
            if shift_sq < HUBER_CONVERGE_EPS {
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
        let scale = (1.4826 * mad).max(HUBER_SCALE_FLOOR);
        let threshold = HUBER_C * scale;

        let inliers: Vec<(f32, f32, f32)> = points
            .iter()
            .zip(residuals.iter())
            .filter_map(|(&pt, &r)| if r <= threshold { Some(pt) } else { None })
            .collect();

        if inliers.len() < self.min_players {
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
            self.ema_yaw += self.cluster_alpha * (raw_yaw - self.ema_yaw);
            self.ema_pitch += self.cluster_alpha * (raw_pitch - self.ema_pitch);
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
    /// Dynamic FOV proportional to the cluster's angular extent.
    ///
    /// `spread` is the max inlier distance from the centroid (radians).
    /// The viewport should be roughly `2 * spread` wide plus margin.
    /// A 1.6x margin keeps players comfortably inside the frame.
    fn target_fov(&self, spread: f32, pitch: f32) -> f32 {
        let spread_deg = spread.to_degrees();
        let fov_from_spread = (2.0 * spread_deg * 1.0).max(self.fov_tight);

        let t_dist = ((pitch - PITCH_NEAR) / (PITCH_FAR - PITCH_NEAR)).clamp(0.0, 1.0);
        let distance_bias = t_dist * DISTANCE_BIAS_MAX;

        let yaw_abs = self.yaw.abs();
        let edge_bias = (yaw_abs * 5.0).min(EDGE_BIAS_MAX);

        (fov_from_spread + distance_bias + edge_bias).clamp(self.fov_tight, self.fov_wide)
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

        let ball_pos = if self.ball_weight > 0.0 {
            world
                .ball
                .as_ref()
                .filter(|b| !matches!(b.state, TrackState::Lost))
                .map(|b| (b.yaw, b.pitch))
        } else {
            None
        };

        if let Some(ref c) = cluster {
            let mut target_yaw = c.yaw * (1.0 + EDGE_PUSH);
            let mut target_pitch = c.pitch + PITCH_BIAS;

            if let Some((by, bp)) = ball_pos {
                let w = self.ball_weight;
                target_yaw = target_yaw * (1.0 - w) + by * w;
                target_pitch = target_pitch * (1.0 - w) + bp * w;
                log::trace!(
                    "FieldPanner: blend cluster({:.3},{:.3}) + ball({:.3},{:.3}) w={:.2}",
                    c.yaw,
                    c.pitch,
                    by,
                    bp,
                    w
                );
            }

            if target_yaw.is_finite() && target_pitch.is_finite() {
                let err_yaw = target_yaw - self.yaw;
                let err_pitch = target_pitch - self.pitch;

                let desired_yaw = err_yaw.clamp(-self.max_velocity, self.max_velocity);
                let desired_pitch = err_pitch.clamp(-self.max_velocity, self.max_velocity);

                self.velocity_yaw += VELOCITY_ALPHA * (desired_yaw - self.velocity_yaw);
                self.velocity_pitch += VELOCITY_ALPHA * (desired_pitch - self.velocity_pitch);

                self.yaw += self.velocity_yaw;
                self.pitch += self.velocity_pitch;
            } else {
                log::warn!(
                    "FieldPanner: non-finite target yaw={target_yaw} pitch={target_pitch}; \
                     keeping yaw={} pitch={}",
                    self.yaw,
                    self.pitch,
                );
            }

            let target_fov = self.target_fov(c.spread, c.pitch);
            if target_fov.is_finite() {
                self.current_fov += FOV_ALPHA * (target_fov - self.current_fov);
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
                self.min_players
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
                ball_pos.is_some()
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
        let tight = p.target_fov(0.05, 0.0);
        let wide = p.target_fov(0.40, 0.0);
        assert!(tight < wide, "tight={tight} wide={wide}");
    }

    #[test]
    fn fov_tighter_when_far() {
        let p = FieldPanner::new(30.0);
        let near = p.target_fov(0.20, PITCH_NEAR);
        let far = p.target_fov(0.20, PITCH_FAR);
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
