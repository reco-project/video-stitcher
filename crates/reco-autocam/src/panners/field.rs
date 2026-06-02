//! Field-aware panner. Two framing modes share one motion pipeline:
//! [`FramingMode::Action`] follows the densest player cluster (broadcast
//! style), [`FramingMode::FrameAll`] keeps every player in frame.
//!
//! # Pipeline (per `decide` call)
//!
//! 1. Take the current-frame players from `world.players` (the provider
//!    self-filters by class; `Lost` entities are dropped before
//!    clustering).
//! 2. Resolve the look-at center per [`FramingMode`]:
//!    - **Action** - trimmed centroid: keep the densest `keep_fraction`
//!      of players, drop the rest (goalkeeper, substitutes) as outliers,
//!      then a (optionally confidence-weighted) mean + EMA. Edge-push and
//!      pitch-bias lean the aim into the direction of play; an optional
//!      ball blend pulls toward `world.ball`.
//!    - **FrameAll** - the EMA-smoothed geometric midpoint of every
//!      player's bounding box; no trim, no weighting, no biases.
//! 3. Lookahead lead, velocity-clamped chase, and soft dead-zone move
//!    the aim toward that center (mode-agnostic).
//! 4. Dynamic FOV: Action sizes from cluster spread + distance/edge/
//!    velocity biases (widened to keep the ball framed); FrameAll sizes
//!    to the bounding-box extent plus a margin. Both EMA-smoothed.
//!
//! # Logging
//!
//! Following reco's explicit-decision principle: every behavior
//! branch (ball blend vs cluster-only, cluster lost, NaN rejection)
//! emits a log line so an operator can reconstruct the camera's
//! decisions from the log alone.

use reco_core::detect::director::ViewportPosition;
use reco_core::detect::panner::{PanContext, Panner};
use reco_core::detect::tracker::{TrackState, TrackedEntity, WorldState};
use serde::{Deserialize, Serialize};

const LOG_INTERVAL: u64 = 30;

/// How the panner chooses what to frame each frame.
///
/// The two modes share all of the motion machinery (velocity-clamped
/// chase, dead-zone, lookahead lead, pose smoothing); they differ only
/// in *where* the camera aims and *how wide* it zooms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FramingMode {
    /// Follow the action: the trimmed, (optionally) confidence-weighted
    /// player cluster, with edge-push, pitch-bias, ball blending, and
    /// the dynamic FOV biases. The broadcast-style default.
    #[default]
    Action,
    /// Keep every player in frame: aim at the geometric midpoint of the
    /// players' bounding box and size the FOV to the box extent plus a
    /// margin. No trim, no confidence weighting, no FOV biases, no ball
    /// blend - the "show the whole team" mode (frisbee, training, etc.).
    FrameAll,
}

/// All tunable parameters for the field panner.
///
/// Safe defaults are tuned for football (soccer) at 30fps. Override
/// individual fields for other sports or frame rates. `#[serde(default)]`
/// means a config file may specify only the knobs it wants to change;
/// the rest fall back to [`Default`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FieldPannerConfig {
    /// Fraction of players to keep when computing the cluster
    /// centroid, in `(0, 1]`. The farthest `1 - keep_fraction`
    /// (goalkeeper, substitutes) are trimmed as outliers; `0.8`
    /// follows the densest 80%.
    pub keep_fraction: f32,
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
    /// When lookahead is active, the base chase runs this many times
    /// more reactive (looser velocity clamp + faster velocity EMA). The
    /// loop's centered smoother removes the resulting jitter lag-free,
    /// so the base can afford to actually keep up. `1.0` = unchanged
    /// (the reactive/no-buffer profile).
    pub lookahead_reactivity: f32,
    /// Lookahead lead gain: the fraction of the predicted action
    /// displacement (mean of the future window minus the current
    /// target) added to the aim, so the camera moves *ahead* of the
    /// play. `0.0` = no lead. Only applies when future frames exist.
    pub lead_gain: f32,
    /// EMA factor for the lead, in `(0, 1]`. The lead is a *trend*, so
    /// it must be smoothed - otherwise per-frame centroid noise makes
    /// the aim wobble left-right. Smaller = smoother (and laggier) lead.
    pub lead_alpha: f32,
    /// Soft radial dead-zone (radians): the camera holds when the target
    /// is within this distance of the current aim, and larger errors are
    /// shrunk by it - removes residual micro-wobble on near-static play.
    /// Viable because lookahead absorbs the latency a dead-zone adds.
    /// `0.0` = off.
    pub dead_zone_rad: f32,
    /// Framing mode: follow the action ([`FramingMode::Action`], default)
    /// or keep every player in frame ([`FramingMode::FrameAll`]).
    pub framing: FramingMode,
    /// Whether the [`FramingMode::Action`] centroid weights players by
    /// detection confidence. `true` (default) lets high-confidence
    /// players anchor the aim; `false` is a pure geometric mean of the
    /// kept players. Ignored in [`FramingMode::FrameAll`], whose midpoint
    /// is geometric by construction.
    pub confidence_weighted: bool,
    /// Extra padding (degrees, each side) added to the player bounding
    /// box in [`FramingMode::FrameAll`] so players near the edge are not
    /// clipped at the viewport boundary.
    pub frame_all_margin_deg: f32,
    /// Horizontal-only panning: hold the tilt at [`locked_pitch_rad`] and
    /// pan in yaw alone, keeping a fixed height. Composes with either
    /// framing mode. `false` (default) lets pitch track the action.
    ///
    /// (Zoom-range lock needs no flag: set `fov_tight == fov_wide` to pin
    /// the FOV to a constant.)
    ///
    /// [`locked_pitch_rad`]: Self::locked_pitch_rad
    pub lock_pitch: bool,
    /// Tilt held when [`lock_pitch`](Self::lock_pitch) is set, in radians
    /// (panorama frame; `0.0` = level). Ignored otherwise.
    pub locked_pitch_rad: f32,
}

impl Default for FieldPannerConfig {
    fn default() -> Self {
        Self {
            keep_fraction: 0.8,
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
            ball_weight: 0.5,
            lookahead_reactivity: 2.0,
            lead_gain: 1.0,
            lead_alpha: 0.1,
            dead_zone_rad: 0.087,
            framing: FramingMode::Action,
            confidence_weighted: true,
            frame_all_margin_deg: 8.0,
            lock_pitch: false,
            locked_pitch_rad: 0.0,
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
    /// EMA-smoothed lookahead lead offset (rad), kept across frames so
    /// the lead tracks the action's trend rather than per-frame noise.
    lead_yaw: f32,
    lead_pitch: f32,
    frame_index: u64,
    /// Latched true the first frame a non-empty lookahead buffer is
    /// seen, so the reactive damping profile stays stable through the
    /// drain tail (where the future window shrinks back to empty).
    lookahead_active: bool,
    last_debug: Option<FieldPannerDebug>,
}

struct FieldPannerDebug {
    cluster_yaw: f32,
    cluster_pitch: f32,
    cluster_spread: f32,
    n_players: u32,
    ball_near_cluster: bool,
    ball_presence: f32,
    effective_ball_weight: f32,
    target_yaw: f32,
    target_pitch: f32,
    fov_target: f32,
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
            lead_yaw: 0.0,
            lead_pitch: 0.0,
            frame_index: 0,
            lookahead_active: false,
            last_debug: None,
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
    /// Lost entities are dropped — a tracker may report them for one
    /// final frame so consumers can observe the transition, but they
    /// have no meaningful centroid contribution. Class filtering and
    /// cross-camera handling are the provider's job; the panner just
    /// consumes the points.
    fn to_points(&self, players: &[TrackedEntity]) -> Vec<(f32, f32, f32)> {
        players
            .iter()
            .filter(|p| !matches!(p.state, TrackState::Lost))
            .map(|p| (p.yaw, p.pitch, p.confidence))
            .collect()
    }

    /// Trimmed robust centroid: keep the densest `keep_fraction` of
    /// the players, drop the rest as outliers (goalkeeper, subs).
    ///
    /// Take the confidence-weighted mean, measure each player's
    /// distance to it, and return the closest `keep_fraction` as the
    /// inlier set - one pass, one knob. The downstream centroid EMA in
    /// [`smooth_centroid`](Self::smooth_centroid) absorbs the small
    /// step when a boundary player flips in or out, so the hard trim
    /// does not teleport the camera.
    fn cluster_and_trim(&self, points: &[(f32, f32, f32)]) -> Vec<(f32, f32, f32)> {
        if points.len() < self.config.min_cluster {
            return Vec::new();
        }
        // Trim reference: confidence-weighted mean, or a plain geometric
        // mean when `confidence_weighted` is off (weight every player 1).
        let weight = |c: f32| {
            if self.config.confidence_weighted {
                c
            } else {
                1.0
            }
        };
        let total_w: f32 = points.iter().map(|(_, _, c)| weight(*c)).sum();
        if total_w <= 0.0 {
            return Vec::new();
        }
        let cy: f32 = points.iter().map(|(y, _, c)| y * weight(*c)).sum::<f32>() / total_w;
        let cp: f32 = points.iter().map(|(_, p, c)| p * weight(*c)).sum::<f32>() / total_w;
        if !cy.is_finite() || !cp.is_finite() {
            return Vec::new();
        }

        // Keep the closest `keep_fraction` of players, but never fewer
        // than `min_cluster`.
        let keep_fraction = self.config.keep_fraction.clamp(0.0, 1.0);
        let keep_count = ((points.len() as f32 * keep_fraction).round() as usize)
            .clamp(self.config.min_cluster, points.len());

        let dist_sq = |&(y, p, _): &(f32, f32, f32)| (y - cy).powi(2) + (p - cp).powi(2);
        let mut sorted = points.to_vec();
        sorted.sort_by(|a, b| {
            dist_sq(a)
                .partial_cmp(&dist_sq(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(keep_count);
        sorted
    }

    /// Cluster centroid (confidence-weighted unless the config opts out)
    /// followed by EMA smoothing.
    fn smooth_centroid(&mut self, core: &[(f32, f32, f32)]) -> (f32, f32) {
        let weight = |c: f32| {
            if self.config.confidence_weighted {
                c
            } else {
                1.0
            }
        };
        let mut sum_yaw = 0.0_f32;
        let mut sum_pitch = 0.0_f32;
        let mut total_weight = 0.0_f32;
        for &(yaw, pitch, conf) in core {
            let w = weight(conf);
            sum_yaw += yaw * w;
            sum_pitch += pitch * w;
            total_weight += w;
        }
        if total_weight <= 0.0 {
            return (self.ema_yaw, self.ema_pitch);
        }
        let raw_yaw = sum_yaw / total_weight;
        let raw_pitch = sum_pitch / total_weight;
        if !raw_yaw.is_finite() || !raw_pitch.is_finite() {
            return (self.ema_yaw, self.ema_pitch);
        }
        self.ema_step(raw_yaw, raw_pitch)
    }

    /// Advance the centroid EMA toward a raw `(yaw, pitch)` and return
    /// the smoothed value. Shared by the action centroid and the
    /// frame-all bounding-box midpoint.
    ///
    /// First-stage EMA: smooths step changes in the raw center into
    /// ramps. The output pose EMA in the run loop then smooths the ramps
    /// further, naturally bounding acceleration - two cascaded EMAs give
    /// a second-order filter with smooth accel/decel.
    fn ema_step(&mut self, raw_yaw: f32, raw_pitch: f32) -> (f32, f32) {
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
        match self.config.framing {
            FramingMode::Action => self.cluster_action(&points),
            FramingMode::FrameAll => self.cluster_frame_all(&points),
        }
    }

    /// Action framing: trimmed, (optionally) confidence-weighted cluster
    /// centroid; `spread` is the max radial distance of a kept player, so
    /// the FOV path frames the cluster tightly.
    fn cluster_action(&mut self, points: &[(f32, f32, f32)]) -> Option<Cluster> {
        let core = self.cluster_and_trim(points);
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

    /// Frame-all framing: aim at the geometric midpoint of every player's
    /// bounding box (EMA-smoothed so a new extreme player does not snap
    /// the camera), and report `spread` as the larger half-extent so the
    /// FOV path widens to contain the whole box. No trim, no confidence
    /// weighting - every player counts equally.
    fn cluster_frame_all(&mut self, points: &[(f32, f32, f32)]) -> Option<Cluster> {
        if points.len() < self.config.min_cluster {
            return None;
        }
        let (cy, cp, half_yaw, half_pitch) = bbox(points)?;
        let (yaw, pitch) = self.ema_step(cy, cp);
        // The viewport FOV is the horizontal angle, so the yaw extent
        // usually dominates; keep the larger half-extent so a tall, narrow
        // formation is not clipped vertically either.
        let spread = half_yaw.max(half_pitch);
        Some(Cluster {
            yaw,
            pitch,
            spread,
            count: points.len(),
        })
    }

    /// Instantaneous look-at target from a world state, with no
    /// smoothing or internal state, else the live ball, else None. Used
    /// to estimate where the action is heading for the lookahead lead, so
    /// it must mirror the framing the real target uses: the action
    /// centroid (edge-pushed, pitch-biased) or the frame-all bbox
    /// midpoint.
    fn raw_target(&self, world: &WorldState) -> Option<(f32, f32)> {
        let points = self.to_points(&world.players);
        match self.config.framing {
            FramingMode::Action => {
                let core = self.cluster_and_trim(&points);
                if !core.is_empty() {
                    let weight = |c: f32| {
                        if self.config.confidence_weighted {
                            c
                        } else {
                            1.0
                        }
                    };
                    let total: f32 = core.iter().map(|(_, _, c)| weight(*c)).sum();
                    if total > 0.0 {
                        let cy = core.iter().map(|(y, _, c)| y * weight(*c)).sum::<f32>() / total;
                        let cp = core.iter().map(|(_, p, c)| p * weight(*c)).sum::<f32>() / total;
                        return Some((
                            cy * (1.0 + self.config.edge_push),
                            cp + self.config.pitch_bias,
                        ));
                    }
                }
            }
            FramingMode::FrameAll => {
                if points.len() >= self.config.min_cluster
                    && let Some((cy, cp, _, _)) = bbox(&points)
                {
                    return Some((cy, cp));
                }
            }
        }
        world
            .ball
            .as_ref()
            .filter(|b| !matches!(b.state, TrackState::Lost))
            .map(|b| (b.yaw, b.pitch))
    }
}

/// Axis-aligned bounding box of the finite points: returns
/// `(center_yaw, center_pitch, half_yaw_extent, half_pitch_extent)`,
/// or `None` if no point has finite coordinates.
fn bbox(points: &[(f32, f32, f32)]) -> Option<(f32, f32, f32, f32)> {
    let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_p, mut max_p) = (f32::INFINITY, f32::NEG_INFINITY);
    for &(y, p, _) in points {
        if !y.is_finite() || !p.is_finite() {
            continue;
        }
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        min_p = min_p.min(p);
        max_p = max_p.max(p);
    }
    if min_y.is_finite() && max_y.is_finite() {
        Some((
            0.5 * (min_y + max_y),
            0.5 * (min_p + max_p),
            0.5 * (max_y - min_y),
            0.5 * (max_p - min_p),
        ))
    } else {
        None
    }
}

impl FieldPanner {
    /// Dynamic FOV: player spread + distance + edge + velocity biases,
    /// then widened if needed to keep the ball in frame.
    ///
    /// In [`FramingMode::FrameAll`] the biases are bypassed entirely: the
    /// FOV is just the player bounding-box extent (`2 * spread`) plus the
    /// configured margin, so every player stays framed regardless of
    /// distance or velocity.
    fn target_fov(&self, spread: f32, pitch: f32, velocity_mag: f32) -> f32 {
        if self.config.framing == FramingMode::FrameAll {
            let extent_deg = (2.0 * spread).to_degrees();
            return (extent_deg + 2.0 * self.config.frame_all_margin_deg)
                .clamp(self.config.fov_tight, self.config.fov_wide);
        }

        let spread_deg = spread.to_degrees();
        let fov_from_spread = (2.0 * spread_deg).max(self.config.fov_tight);

        let t_dist = ((pitch - self.config.pitch_near)
            / (self.config.pitch_far - self.config.pitch_near))
            .clamp(0.0, 1.0);
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
    fn decide(&mut self, world: &WorldState, ctx: &PanContext<'_>) -> ViewportPosition {
        // No buffer: empty future, so the reactive (non-lookahead)
        // damping profile and no lead.
        self.decide_with_lookahead(world, &[], ctx)
    }

    fn decide_with_lookahead(
        &mut self,
        world: &WorldState,
        future: &[WorldState],
        _ctx: &PanContext<'_>,
    ) -> ViewportPosition {
        reco_core::profile_scope!("field_panner_decide");

        // Once a non-empty lookahead buffer is seen, stay in the
        // reactive damping profile (latched, so the drain tail does not
        // snap back to over-damped). The loop's centered smoother is
        // what removes the resulting jitter, lag-free.
        self.lookahead_active |= !future.is_empty();
        let reactivity = if self.lookahead_active {
            self.config.lookahead_reactivity.max(1.0)
        } else {
            1.0
        };

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

        // Resolve where to look: the player cluster (optionally
        // ball-blended) when we have one, else the ball alone, else
        // hold. Ball-only follow lets FieldPanner serve ball-centric
        // sports and ball-only detectors without a separate panner -
        // the player cluster simply isn't available to size the FOV.
        let mut cluster_target: Option<(f32, f32, f32)> = None; // (target_yaw, target_pitch, effective_ball_weight)
        let mut target = if let Some(ref c) = cluster {
            match self.config.framing {
                FramingMode::Action => {
                    let mut target_yaw = c.yaw * (1.0 + self.config.edge_push);
                    let mut target_pitch = c.pitch + self.config.pitch_bias;

                    let effective_w = self.config.ball_weight * self.ball_presence;
                    if effective_w > 0.001 {
                        target_yaw =
                            target_yaw * (1.0 - effective_w) + self.last_ball_yaw * effective_w;
                        target_pitch =
                            target_pitch * (1.0 - effective_w) + self.last_ball_pitch * effective_w;
                    }
                    cluster_target = Some((target_yaw, target_pitch, effective_w));
                    Some((target_yaw, target_pitch))
                }
                // Frame-all aims at the raw bbox midpoint: no edge-push,
                // no pitch-bias, no ball blend - just frame the team.
                FramingMode::FrameAll => {
                    cluster_target = Some((c.yaw, c.pitch, 0.0));
                    Some((c.yaw, c.pitch))
                }
            }
        } else if ball_detected {
            // No player cluster: follow the ball directly.
            let b = world.ball.as_ref().unwrap();
            Some((b.yaw, b.pitch))
        } else {
            None
        };

        // Lookahead lead: aim toward where the action is heading. The
        // lead is the displacement from the current target to the MEAN
        // of the future window (averaging kills per-frame centroid
        // noise), then EMA-smoothed across frames so it tracks the
        // trend rather than jitter - a noisy lead makes the camera
        // wobble left-right. This is the only consumer of the future
        // world-states, and the reason the camera moves ahead of play.
        if !future.is_empty()
            && let Some((ty, tp)) = target
            && let Some((cur_y, cur_p)) = self.raw_target(world)
        {
            let mut sum_y = 0.0_f32;
            let mut sum_p = 0.0_f32;
            let mut n = 0u32;
            for w in future {
                if let Some((y, p)) = self.raw_target(w) {
                    sum_y += y;
                    sum_p += p;
                    n += 1;
                }
            }
            if n > 0 {
                let g = self.config.lead_gain;
                let raw_lead_y = (sum_y / n as f32 - cur_y) * g;
                let raw_lead_p = (sum_p / n as f32 - cur_p) * g;
                let a = self.config.lead_alpha;
                self.lead_yaw += a * (raw_lead_y - self.lead_yaw);
                self.lead_pitch += a * (raw_lead_p - self.lead_pitch);
                target = Some((ty + self.lead_yaw, tp + self.lead_pitch));
            }
        }

        // Horizontal-only: pin the tilt so the camera pans in yaw alone.
        // Applied after the lead/blend so nothing can reintroduce pitch
        // motion downstream.
        if self.config.lock_pitch
            && let Some((ty, _)) = target
        {
            target = Some((ty, self.config.locked_pitch_rad));
        }

        // Velocity-clamped chase toward the resolved target (shared by
        // the cluster and ball-only paths). When lookahead is active the
        // clamp and EMA are loosened by `reactivity` so the base keeps
        // up; the centered smoother downstream removes the jitter.
        if let Some((target_yaw, target_pitch)) = target
            && target_yaw.is_finite()
            && target_pitch.is_finite()
        {
            let max_v = self.max_velocity * reactivity;
            let v_alpha = (self.config.velocity_alpha * reactivity).min(1.0);
            let mut err_yaw = target_yaw - self.yaw;
            let mut err_pitch = target_pitch - self.pitch;

            // Soft radial dead-zone: hold for sub-threshold moves, and
            // shrink larger errors by the threshold so the camera eases
            // in/out instead of a hard hold-then-jump.
            let dz = self.config.dead_zone_rad;
            if dz > 0.0 {
                let mag = (err_yaw * err_yaw + err_pitch * err_pitch).sqrt();
                if mag > 1e-6 {
                    let scale = (mag - dz).max(0.0) / mag;
                    err_yaw *= scale;
                    err_pitch *= scale;
                }
            }

            let desired_yaw = err_yaw.clamp(-max_v, max_v);
            let desired_pitch = err_pitch.clamp(-max_v, max_v);

            self.velocity_yaw += v_alpha * (desired_yaw - self.velocity_yaw);
            self.velocity_pitch += v_alpha * (desired_pitch - self.velocity_pitch);

            self.yaw += self.velocity_yaw;
            self.pitch += self.velocity_pitch;
        }

        // Dynamic FOV needs the cluster spread; a ball-only target has
        // none, so the FOV holds at its current value.
        if let Some(ref c) = cluster {
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

            let (target_yaw, target_pitch, effective_w) =
                cluster_target.unwrap_or((c.yaw, c.pitch, 0.0));
            self.last_debug = Some(FieldPannerDebug {
                cluster_yaw: c.yaw,
                cluster_pitch: c.pitch,
                cluster_spread: c.spread,
                n_players: c.count as u32,
                ball_near_cluster,
                ball_presence: self.ball_presence,
                effective_ball_weight: effective_w,
                target_yaw,
                target_pitch,
                fov_target: target_fov,
            });
        } else {
            self.last_debug = None;
            if target.is_none() {
                log::trace!(
                    "FieldPanner: no cluster and no ball this frame (players={}, min={})",
                    world.players.len(),
                    self.config.min_cluster
                );
            }
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

    fn debug_event(
        &self,
        frame_index: u64,
    ) -> Option<reco_core::detect::pipeline_event::PipelineEvent> {
        let d = self.last_debug.as_ref()?;
        Some(
            reco_core::detect::pipeline_event::PipelineEvent::PannerDebug {
                frame_index,
                cluster_yaw: d.cluster_yaw,
                cluster_pitch: d.cluster_pitch,
                cluster_spread: d.cluster_spread,
                n_players: d.n_players,
                ball_near_cluster: d.ball_near_cluster,
                ball_presence: d.ball_presence,
                effective_ball_weight: d.effective_ball_weight,
                target_yaw: d.target_yaw,
                target_pitch: d.target_pitch,
                fov_target: d.fov_target,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detect::detector::CameraId;

    fn player(yaw: f32, pitch: f32, id: u64) -> TrackedEntity {
        player_conf(yaw, pitch, id, 0.9)
    }

    fn player_conf(yaw: f32, pitch: f32, id: u64, confidence: f32) -> TrackedEntity {
        TrackedEntity {
            id,
            class_id: 0,
            yaw,
            pitch,
            confidence,
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
        // Trimmed centroid on 5 evenly-spaced players at 0.28..0.44
        // (keep 80% = 4) converges near the mean 0.36. Edge push 15%.
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
    fn trim_excludes_goalkeeper_outlier() {
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
    fn keep_fraction_trims_farthest_players() {
        let p = FieldPanner::with_config(
            30.0,
            FieldPannerConfig {
                keep_fraction: 0.6,
                ..Default::default()
            },
        );
        // 3 tight players + 2 far outliers. keep 60% of 5 = 3.
        let points = vec![
            (0.30, 0.0, 1.0),
            (0.32, 0.0, 1.0),
            (0.34, 0.0, 1.0),
            (1.50, 0.0, 1.0),
            (1.60, 0.0, 1.0),
        ];
        let kept = p.cluster_and_trim(&points);
        assert_eq!(kept.len(), 3, "keep 0.6 of 5 = 3");
        assert!(
            kept.iter().all(|&(y, _, _)| y < 0.5),
            "the tight trio survives, outliers dropped: {kept:?}"
        );
    }

    #[test]
    fn keep_fraction_respects_min_cluster_floor() {
        // A tiny keep_fraction still keeps at least min_cluster players.
        let p = FieldPanner::with_config(
            30.0,
            FieldPannerConfig {
                keep_fraction: 0.01,
                min_cluster: 2,
                ..Default::default()
            },
        );
        let points = vec![
            (0.00, 0.0, 1.0),
            (0.10, 0.0, 1.0),
            (0.20, 0.0, 1.0),
            (5.00, 0.0, 1.0),
        ];
        let kept = p.cluster_and_trim(&points);
        assert_eq!(kept.len(), 2, "min_cluster floors the keep count");
    }

    #[test]
    fn ball_only_follows_ball_without_cluster() {
        // No players (no cluster) but a ball is present: FieldPanner
        // follows the ball directly instead of holding. This is what
        // lets one panner serve ball-centric modes too.
        let mut p = FieldPanner::new(30.0); // default ball_weight 0.5 > 0
        let cal = cal();
        let w = WorldState {
            ball: Some(ball(0.5, 0.1)),
            players: vec![],
        };
        let mut out = p.decide(&w, &ctx(0, &cal));
        for i in 1..300 {
            out = p.decide(&w, &ctx(i, &cal));
        }
        assert!(
            (out.yaw - 0.5).abs() < 0.05,
            "ball-only should converge near ball yaw 0.5, got {}",
            out.yaw
        );
        assert!(
            (out.pitch - 0.1).abs() < 0.05,
            "ball-only should converge near ball pitch 0.1, got {}",
            out.pitch
        );
    }

    #[test]
    fn lookahead_lead_aims_ahead_of_current() {
        let cal = cal();
        // Current cluster centered at yaw 0; the action is heading to 0.5.
        let cur = WorldState {
            ball: None,
            players: vec![
                player(-0.02, 0.0, 1),
                player(0.02, 0.0, 2),
                player(0.0, 0.0, 3),
            ],
        };
        let fut = WorldState {
            ball: None,
            players: vec![
                player(0.48, 0.0, 1),
                player(0.52, 0.0, 2),
                player(0.50, 0.0, 3),
            ],
        };
        // Isolate the lead: reactivity 1.0 (no damping boost) and
        // dead_zone 0.0 (so the lead isn't masked by the hold). Run a
        // few frames so the EMA-smoothed lead builds and moves the aim.
        let cfg = FieldPannerConfig {
            lookahead_reactivity: 1.0,
            lead_gain: 1.0,
            dead_zone_rad: 0.0,
            ..Default::default()
        };
        let mut no_lead = FieldPanner::with_config(30.0, cfg.clone());
        let mut with_lead = FieldPanner::with_config(30.0, cfg);
        let mut out_no = no_lead.decide(&cur, &ctx(0, &cal));
        let mut out_la =
            with_lead.decide_with_lookahead(&cur, std::slice::from_ref(&fut), &ctx(0, &cal));
        for i in 1..20 {
            out_no = no_lead.decide(&cur, &ctx(i, &cal));
            out_la =
                with_lead.decide_with_lookahead(&cur, std::slice::from_ref(&fut), &ctx(i, &cal));
        }
        assert!(
            out_la.yaw > out_no.yaw,
            "lead should push the aim toward the future cluster: no-lead {} vs lead {}",
            out_no.yaw,
            out_la.yaw
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

    fn frame_all_config() -> FieldPannerConfig {
        FieldPannerConfig {
            framing: FramingMode::FrameAll,
            dead_zone_rad: 0.0, // let the aim converge exactly for the assert
            ..Default::default()
        }
    }

    #[test]
    fn frame_all_aims_at_bbox_midpoint_not_mean() {
        // Players bunched low + one high: midpoint of [0.0, 0.5] = 0.25,
        // whereas the mean would sit at ~0.2. Frame-all must use the
        // midpoint (and apply no edge-push), so the lone player isn't
        // clipped.
        let mut p = FieldPanner::with_config(30.0, frame_all_config());
        let cal = cal();
        let w = WorldState {
            ball: None,
            players: vec![
                player(0.0, 0.0, 1),
                player(0.1, 0.0, 2),
                player(0.5, 0.0, 3),
            ],
        };
        let mut out = p.decide(&w, &ctx(0, &cal));
        for i in 1..400 {
            out = p.decide(&w, &ctx(i, &cal));
        }
        assert!(
            (out.yaw - 0.25).abs() < 0.02,
            "frame-all should aim at the bbox midpoint 0.25, got {}",
            out.yaw
        );
    }

    #[test]
    fn frame_all_fov_grows_with_player_extent() {
        let p = FieldPanner::with_config(30.0, frame_all_config());
        // spread is the half-extent; a wider squad needs a wider FOV.
        let tight = p.target_fov(0.05, 0.0, 0.0);
        let wide = p.target_fov(0.30, 0.0, 0.0);
        assert!(tight < wide, "tight={tight} wide={wide}");
        // The FOV must cover the full extent (2*spread) plus the margin.
        let half: f32 = 0.20;
        let got = p.target_fov(half, 0.0, 0.0);
        let needed = (2.0 * half).to_degrees() + 2.0 * p.config.frame_all_margin_deg;
        assert!(
            got >= needed - 1e-3 || (got - p.config.fov_wide).abs() < 1e-3,
            "frame-all FOV {got} must cover extent+margin {needed} (or be clamped to fov_wide)"
        );
    }

    #[test]
    fn confidence_weighting_can_be_disabled() {
        // One low-confidence player near 0.0 and one high-confidence
        // player at 0.4. Weighted, the aim leans toward the confident
        // player; unweighted, it sits at the geometric mean.
        let cal = cal();
        let w = WorldState {
            ball: None,
            players: vec![
                player_conf(0.0, 0.0, 1, 0.2),
                player_conf(0.4, 0.0, 2, 0.95),
            ],
        };
        let base = FieldPannerConfig {
            keep_fraction: 1.0, // keep both so weighting is what differs
            dead_zone_rad: 0.0,
            ball_weight: 0.0,
            ..Default::default()
        };
        let mut weighted = FieldPanner::with_config(
            30.0,
            FieldPannerConfig {
                confidence_weighted: true,
                ..base.clone()
            },
        );
        let mut unweighted = FieldPanner::with_config(
            30.0,
            FieldPannerConfig {
                confidence_weighted: false,
                ..base
            },
        );
        let (mut ow, mut ou) = (
            weighted.decide(&w, &ctx(0, &cal)),
            unweighted.decide(&w, &ctx(0, &cal)),
        );
        for i in 1..400 {
            ow = weighted.decide(&w, &ctx(i, &cal));
            ou = unweighted.decide(&w, &ctx(i, &cal));
        }
        assert!(
            ow.yaw > ou.yaw,
            "confidence weighting should pull toward the confident player: weighted {} vs unweighted {}",
            ow.yaw,
            ou.yaw
        );
    }

    #[test]
    fn lock_pitch_holds_tilt_but_pans_yaw() {
        // Players sit at pitch 0.2; with lock_pitch the camera must
        // converge to the locked tilt (0.0) while still tracking yaw.
        let cal = cal();
        let cfg = FieldPannerConfig {
            lock_pitch: true,
            locked_pitch_rad: 0.0,
            dead_zone_rad: 0.0,
            ball_weight: 0.0,
            ..Default::default()
        };
        let mut p = FieldPanner::with_config(30.0, cfg);
        let w = WorldState {
            ball: None,
            players: vec![
                player(0.30, 0.2, 1),
                player(0.34, 0.2, 2),
                player(0.38, 0.2, 3),
            ],
        };
        let mut out = p.decide(&w, &ctx(0, &cal));
        for i in 1..400 {
            out = p.decide(&w, &ctx(i, &cal));
        }
        assert!(
            out.pitch.abs() < 1e-3,
            "lock_pitch should hold tilt at 0.0, got {}",
            out.pitch
        );
        assert!(
            out.yaw > 0.2,
            "yaw should still track the cluster, got {}",
            out.yaw
        );
    }
}
