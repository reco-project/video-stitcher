//! Lookahead-aware panner using a frame buffer for anticipation.
//! Pipeline: pre-smooth -> future blend -> dead zone -> EMA chase.
//! Post-smooth (centered moving average) is applied in the run loop
//! where both past and future poses are available.

use std::collections::VecDeque;

use reco_core::detect::director::ViewportPosition;
use reco_core::detect::panner::{PanContext, Panner};
use reco_core::detect::tracker::{TrackState, TrackedEntity, WorldState};

/// Configuration for the lookahead panner.
#[derive(Debug, Clone)]
pub struct LookaheadPannerConfig {
    /// Ball influence when detected (0.0 = players only, 1.0 = ball only).
    pub ball_weight: f32,
    /// Decay rate for ball memory when ball is lost (per frame).
    pub ball_memory_decay: f32,
    /// Edge push factor (exaggerates side-of-pitch motion).
    pub edge_push: f32,
    /// Pitch bias added to the cluster centroid.
    pub pitch_bias: f32,
    /// Exponential decay factor for future target weighting.
    pub future_decay: f32,
    /// Velocity lead multiplier.
    pub lead_multiplier: f32,
    /// EMA alpha for yaw chase.
    pub yaw_alpha: f32,
    /// EMA alpha for pitch chase.
    pub pitch_alpha: f32,
    /// Pre-smooth window size (moving average on raw targets).
    pub pre_smooth_window: usize,
    /// Dead zone radius in radians. Camera holds when target is within this distance.
    pub dead_zone_rad: f32,
}

impl Default for LookaheadPannerConfig {
    fn default() -> Self {
        Self {
            ball_weight: 0.20,
            ball_memory_decay: 0.97,
            edge_push: 0.15,
            pitch_bias: 0.05,
            future_decay: 0.6,
            lead_multiplier: 0.3,
            yaw_alpha: 0.04,
            pitch_alpha: 0.015,
            pre_smooth_window: 5,
            dead_zone_rad: 0.087, // ~5 degrees
        }
    }
}

/// Ring buffer for computing a causal moving average.
struct SmoothBuffer {
    buf: VecDeque<f32>,
    capacity: usize,
}

impl SmoothBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    fn push_and_average(&mut self, val: f32) -> f32 {
        self.buf.push_back(val);
        if self.buf.len() > self.capacity {
            self.buf.pop_front();
        }
        self.buf.iter().sum::<f32>() / self.buf.len() as f32
    }
}

/// Panner that blends future WorldStates for camera anticipation.
pub struct LookaheadPanner {
    config: LookaheadPannerConfig,
    yaw: f32,
    pitch: f32,
    fov: f32,
    ball_memory_yaw: Option<f32>,
    ball_memory_pitch: Option<f32>,
    ball_decay: f32,
    pre_yaw: SmoothBuffer,
    pre_pitch: SmoothBuffer,
}

impl Default for LookaheadPanner {
    fn default() -> Self {
        Self::new()
    }
}

impl LookaheadPanner {
    pub fn new() -> Self {
        Self::with_config(LookaheadPannerConfig::default())
    }

    pub fn with_config(config: LookaheadPannerConfig) -> Self {
        let pre_yaw = SmoothBuffer::new(config.pre_smooth_window);
        let pre_pitch = SmoothBuffer::new(config.pre_smooth_window);
        Self {
            config,
            yaw: 0.0,
            pitch: 0.0,
            fov: 45.0,
            ball_memory_yaw: None,
            ball_memory_pitch: None,
            ball_decay: 0.0,
            pre_yaw,
            pre_pitch,
        }
    }

    /// Compute the raw target (yaw, pitch) from a single WorldState.
    fn target_from_world(&self, world: &WorldState) -> Option<(f32, f32)> {
        let ball = world
            .ball
            .as_ref()
            .filter(|b| !matches!(b.state, TrackState::Lost));

        let active: Vec<&TrackedEntity> = world
            .players
            .iter()
            .filter(|p| !matches!(p.state, TrackState::Lost))
            .collect();

        let has_cluster = active.len() >= 2;
        let total_conf: f32 = active.iter().map(|p| p.confidence).sum();

        // Ball-only: return ball position directly.
        if !has_cluster || total_conf <= 0.0 {
            return ball.map(|b| (b.yaw, b.pitch));
        }

        // Cluster centroid.
        let cluster_yaw = active.iter().map(|p| p.yaw * p.confidence).sum::<f32>() / total_conf;
        let cluster_pitch = active.iter().map(|p| p.pitch * p.confidence).sum::<f32>() / total_conf;

        let mut ty = cluster_yaw * (1.0 + self.config.edge_push);
        let mut tp = cluster_pitch + self.config.pitch_bias;

        // Blend ball when available.
        if let Some(b) = ball {
            let w = self.config.ball_weight;
            ty = ty * (1.0 - w) + b.yaw * w;
            tp = tp * (1.0 - w) + b.pitch * w;
        }

        Some((ty, tp))
    }

    /// Blend the current target with future targets using exponential
    /// decay weighting + velocity lead.
    fn blend_with_future(&self, current: (f32, f32), future: &[WorldState]) -> (f32, f32) {
        if future.is_empty() {
            return current;
        }

        let n = future.len();
        let mut sum_yaw = current.0;
        let mut sum_pitch = current.1;
        let mut sum_weight = 1.0_f32;

        for (i, ws) in future.iter().enumerate() {
            if let Some((fy, fp)) = self.target_from_world(ws) {
                let w = (-(i as f32 + 1.0) / (n as f32 * self.config.future_decay)).exp();
                sum_yaw += fy * w;
                sum_pitch += fp * w;
                sum_weight += w;
            }
        }

        let blended_yaw = sum_yaw / sum_weight;
        let blended_pitch = sum_pitch / sum_weight;

        // Velocity lead via linear regression on the first few targets.
        let mut lead_yaw = 0.0_f32;
        let mut lead_pitch = 0.0_f32;
        let fit_len = n.min(10);
        if fit_len >= 3 {
            let mut targets = vec![current];
            for ws in future.iter().take(fit_len) {
                if let Some(t) = self.target_from_world(ws) {
                    targets.push(t);
                }
            }
            if targets.len() >= 3 {
                let m = targets.len() as f32;
                let sum_t: f32 = (0..targets.len()).map(|i| i as f32).sum();
                let sum_y: f32 = targets.iter().map(|(y, _)| *y).sum();
                let sum_p: f32 = targets.iter().map(|(_, p)| *p).sum();
                let sum_tt: f32 = (0..targets.len()).map(|i| (i * i) as f32).sum();
                let sum_ty: f32 = targets
                    .iter()
                    .enumerate()
                    .map(|(i, (y, _))| i as f32 * y)
                    .sum();
                let sum_tp: f32 = targets
                    .iter()
                    .enumerate()
                    .map(|(i, (_, p))| i as f32 * p)
                    .sum();

                let denom = m * sum_tt - sum_t * sum_t;
                if denom.abs() > 1e-6 {
                    let slope_yaw = (m * sum_ty - sum_t * sum_y) / denom;
                    let slope_pitch = (m * sum_tp - sum_t * sum_p) / denom;
                    lead_yaw = slope_yaw * n as f32 * self.config.lead_multiplier;
                    lead_pitch = slope_pitch * n as f32 * self.config.lead_multiplier;
                }
            }
        }

        (blended_yaw + lead_yaw, blended_pitch + lead_pitch)
    }
}

impl Panner for LookaheadPanner {
    fn decide(&mut self, world: &WorldState, ctx: &PanContext<'_>) -> ViewportPosition {
        self.decide_with_lookahead(world, &[], ctx)
    }

    fn decide_with_lookahead(
        &mut self,
        world: &WorldState,
        future: &[WorldState],
        _ctx: &PanContext<'_>,
    ) -> ViewportPosition {
        // Ball memory decay.
        let ball = world
            .ball
            .as_ref()
            .filter(|b| !matches!(b.state, TrackState::Lost));

        if let Some(b) = ball {
            self.ball_memory_yaw = Some(b.yaw);
            self.ball_memory_pitch = Some(b.pitch);
            self.ball_decay = 1.0;
        } else {
            self.ball_decay *= self.config.ball_memory_decay;
        }

        // Build effective target with ball memory.
        let current = if let Some((ty, tp)) = self.target_from_world(world) {
            if let (Some(by), Some(bp)) = (self.ball_memory_yaw, self.ball_memory_pitch) {
                let d = self.ball_decay;
                (d * by + (1.0 - d) * ty, d * bp + (1.0 - d) * tp)
            } else {
                (ty, tp)
            }
        } else if let (Some(by), Some(bp)) = (self.ball_memory_yaw, self.ball_memory_pitch) {
            (by, bp)
        } else {
            return ViewportPosition {
                yaw: self.yaw,
                pitch: self.pitch,
                fov_degrees: Some(self.fov),
            };
        };

        // Step 1: Pre-smooth the raw target.
        let smooth_target = (
            self.pre_yaw.push_and_average(current.0),
            self.pre_pitch.push_and_average(current.1),
        );

        // Step 2: Blend with future lookahead window.
        let (mut target_yaw, mut target_pitch) = self.blend_with_future(smooth_target, future);

        // Step 2.5: Dead zone - hold target when ball barely moved.
        if self.config.dead_zone_rad > 0.0 {
            let dist =
                ((target_yaw - self.yaw).powi(2) + (target_pitch - self.pitch).powi(2)).sqrt();
            if dist < self.config.dead_zone_rad {
                target_yaw = self.yaw;
                target_pitch = self.pitch;
            }
        }

        // Step 3: EMA chase.
        self.yaw += self.config.yaw_alpha * (target_yaw - self.yaw);
        self.pitch += self.config.pitch_alpha * (target_pitch - self.pitch);

        ViewportPosition {
            yaw: self.yaw,
            pitch: self.pitch,
            fov_degrees: Some(self.fov),
        }
    }
}
