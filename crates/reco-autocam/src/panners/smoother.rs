//! Forward/backward One Euro trajectory smoother wrapped as a
//! [`Panner`] decorator.
//!
//! Two smoothing modes are selected by the `lookahead_frames`
//! parameter:
//! - `lookahead_frames <= 1`: persistent causal (forward-only) One
//!   Euro filter. Some phase lag, but no look-ahead cost.
//! - `lookahead_frames > 1`: forward-backward pass over a rolling
//!   buffer of `lookahead_frames` most-recent panner outputs. Zero
//!   phase lag, stronger noise suppression, adds `lookahead_frames`
//!   frames of end-to-end latency.
//!
//! The causal filter always runs so its state stays warm; when the
//! buffer is full, the bidirectional result replaces it for the
//! frame we publish.
//!
//! Reference: Casiez, Roussel, Vogel. "1 Euro Filter: A Simple
//! Speed-based Low-pass Filter for Noisy Input in Interactive
//! Systems." CHI 2012.

use std::collections::VecDeque;

use reco_core::director::ViewportPosition;
use reco_core::panner::{PanContext, Panner};
use reco_core::tracker::WorldState;

/// Fallback FOV (degrees) when an inner panner publishes
/// `fov_degrees = None`. The pipeline defaults to 75°; tracking FOV
/// is intentionally narrower for a tighter view of the action.
const FALLBACK_FOV: f32 = 55.0;

/// Single-axis One Euro filter state.
struct OneEuroAxis {
    x_prev: f32,
    dx_prev: f32,
    initialized: bool,
}

fn smoothing_factor(dt: f32, cutoff: f32) -> f32 {
    let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

impl OneEuroAxis {
    fn new() -> Self {
        Self {
            x_prev: 0.0,
            dx_prev: 0.0,
            initialized: false,
        }
    }

    fn filter(&mut self, x: f32, dt: f32, min_cutoff: f32, beta: f32, d_cutoff: f32) -> f32 {
        if !self.initialized {
            self.x_prev = x;
            self.dx_prev = 0.0;
            self.initialized = true;
            return x;
        }
        let dx = (x - self.x_prev) / dt;
        let alpha_d = smoothing_factor(dt, d_cutoff);
        let dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev;
        let cutoff = min_cutoff + beta * dx_hat.abs();
        let alpha = smoothing_factor(dt, cutoff);
        let x_hat = alpha * x + (1.0 - alpha) * self.x_prev;
        self.x_prev = x_hat;
        self.dx_prev = dx_hat;
        x_hat
    }
}

/// Bidirectional One Euro filter for trajectory smoothing.
///
/// Given a window of raw `ViewportPosition`s, applies a forward One Euro
/// pass, then a backward One Euro pass, and returns the smoothed position
/// for the oldest frame. The bidirectional pass approximately cancels
/// phase lag: forward introduces lag, backward introduces equal lag in
/// the opposite direction. The result is a zero-phase-lag smoothed
/// signal with squared amplitude response.
struct TrajectorySmoother {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    fps: f32,
}

impl TrajectorySmoother {
    fn new(fps: f32) -> Self {
        Self {
            min_cutoff: 0.5,
            beta: 0.007,
            d_cutoff: 1.0,
            fps: fps.clamp(1.0, 1000.0),
        }
    }

    /// Smooth a window of positions and return the position for the
    /// oldest frame (index 0). Input must be ordered oldest-to-newest.
    fn smooth(&self, positions: &[ViewportPosition]) -> ViewportPosition {
        if positions.is_empty() {
            return ViewportPosition::default();
        }
        if positions.len() == 1 {
            return positions[0];
        }

        let dt = 1.0 / self.fps;
        let n = positions.len();

        let mut fwd_yaw = OneEuroAxis::new();
        let mut fwd_pitch = OneEuroAxis::new();
        let mut fwd_fov = OneEuroAxis::new();
        let mut forward: Vec<(f32, f32, f32)> = Vec::with_capacity(n);
        for pos in positions {
            let y = fwd_yaw.filter(pos.yaw, dt, self.min_cutoff, self.beta, self.d_cutoff);
            let p = fwd_pitch.filter(pos.pitch, dt, self.min_cutoff, self.beta, self.d_cutoff);
            let f = fwd_fov.filter(
                pos.fov_degrees.unwrap_or(FALLBACK_FOV),
                dt,
                self.min_cutoff,
                self.beta,
                self.d_cutoff,
            );
            forward.push((y, p, f));
        }

        let mut bwd_yaw = OneEuroAxis::new();
        let mut bwd_pitch = OneEuroAxis::new();
        let mut bwd_fov = OneEuroAxis::new();
        let mut backward: Vec<(f32, f32, f32)> = Vec::with_capacity(n);
        for pos in positions.iter().rev() {
            let y = bwd_yaw.filter(pos.yaw, dt, self.min_cutoff, self.beta, self.d_cutoff);
            let p = bwd_pitch.filter(pos.pitch, dt, self.min_cutoff, self.beta, self.d_cutoff);
            let f = bwd_fov.filter(
                pos.fov_degrees.unwrap_or(FALLBACK_FOV),
                dt,
                self.min_cutoff,
                self.beta,
                self.d_cutoff,
            );
            backward.push((y, p, f));
        }
        backward.reverse();

        let (fy, fp, ff) = forward[0];
        let (by, bp, bf) = backward[0];
        ViewportPosition {
            yaw: (fy + by) * 0.5,
            pitch: (fp + bp) * 0.5,
            fov_degrees: Some((ff + bf) * 0.5),
        }
    }
}

/// One Euro smoother decorator for any [`Panner`] implementation.
pub struct Smoother {
    inner: Box<dyn Panner>,
    smoother: TrajectorySmoother,
    buffer: VecDeque<ViewportPosition>,
    capacity: usize,
    causal_yaw: OneEuroAxis,
    causal_pitch: OneEuroAxis,
    causal_fov: OneEuroAxis,
    dt: f32,
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
}

impl Smoother {
    /// Wrap a panner with trajectory smoothing.
    ///
    /// `fps` is used to convert the One Euro cutoff frequencies
    /// into per-frame alphas; clamped to `[1.0, 1000.0]`.
    /// `lookahead_frames` selects causal-only (<=1) vs
    /// bidirectional (>1) smoothing; typical value is
    /// `(fps * 0.5) as usize`.
    pub fn new(inner: Box<dyn Panner>, fps: f32, lookahead_frames: usize) -> Self {
        let fps = fps.clamp(1.0, 1000.0);
        let capacity = lookahead_frames.max(1);
        let smoother = TrajectorySmoother::new(fps);
        Self {
            inner,
            dt: 1.0 / fps,
            // Parameters match TrajectorySmoother defaults; keeping
            // them local so the causal filter can evolve without
            // touching the bidirectional helper.
            min_cutoff: 0.5,
            beta: 0.007,
            d_cutoff: 1.0,
            smoother,
            buffer: VecDeque::with_capacity(capacity + 1),
            capacity,
            causal_yaw: OneEuroAxis::new(),
            causal_pitch: OneEuroAxis::new(),
            causal_fov: OneEuroAxis::new(),
        }
    }
}

impl Panner for Smoother {
    fn decide(&mut self, world: &WorldState, ctx: &PanContext<'_>) -> ViewportPosition {
        // Drop the oldest entry once we reach capacity.
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        let raw = self.inner.decide(world, ctx);
        self.buffer.push_back(raw);

        // Causal filter always runs so its state stays warm.
        let cy =
            self.causal_yaw
                .filter(raw.yaw, self.dt, self.min_cutoff, self.beta, self.d_cutoff);
        let cp = self.causal_pitch.filter(
            raw.pitch,
            self.dt,
            self.min_cutoff,
            self.beta,
            self.d_cutoff,
        );
        let cf = self.causal_fov.filter(
            raw.fov_degrees.unwrap_or(FALLBACK_FOV),
            self.dt,
            self.min_cutoff,
            self.beta,
            self.d_cutoff,
        );
        let causal = ViewportPosition {
            yaw: cy,
            pitch: cp,
            fov_degrees: Some(cf),
        };

        // With lookahead, prefer the zero-phase-lag bidirectional result.
        if self.buffer.len() >= 2 {
            let slice: Vec<ViewportPosition> = self.buffer.iter().copied().collect();
            self.smoother.smooth(&slice)
        } else {
            causal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::calibration::{CameraParams, MatchCalibration, PlaneLayout};
    use reco_core::detector::CameraId;
    use reco_core::tracker::{TrackState, TrackedEntity, WorldState};

    fn test_cal() -> MatchCalibration {
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

    /// A panner that snaps directly to world.ball when Tracking.
    struct EchoPanner;
    impl Panner for EchoPanner {
        fn decide(&mut self, world: &WorldState, _ctx: &PanContext<'_>) -> ViewportPosition {
            match world.ball {
                Some(b) => ViewportPosition {
                    yaw: b.yaw,
                    pitch: b.pitch,
                    fov_degrees: Some(55.0),
                },
                None => ViewportPosition::default(),
            }
        }
    }

    fn ball(yaw: f32) -> TrackedEntity {
        TrackedEntity {
            id: 0,
            class_id: 0,
            yaw,
            pitch: 0.0,
            velocity: None,
            confidence: 0.8,
            state: TrackState::Tracking,
            age_frames: 1,
            origin: CameraId::Left,
        }
    }

    #[test]
    fn smoother_attenuates_jumps() {
        let cal = test_cal();
        let mut s = Smoother::new(Box::new(EchoPanner), 30.0, 8);
        let mut positions = Vec::new();
        for (i, yaw) in [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0].into_iter().enumerate() {
            let world = WorldState {
                ball: Some(ball(yaw)),
                players: vec![],
            };
            let ctx = PanContext {
                frame_index: i as u64,
                timestamp_ms: i as f64 * 33.3,
                previous_position: ViewportPosition::default(),
                calibration: &cal,
            };
            positions.push(s.decide(&world, &ctx).yaw);
        }
        // Step function 0 -> 1 at index 3. A smoother attenuates the
        // immediate jump so the FIRST post-step sample is less than 1.
        assert!(
            positions[3] < 0.9,
            "expected attenuated step, got {}",
            positions[3]
        );
    }

    #[test]
    fn smoother_passes_steady_state_through() {
        let cal = test_cal();
        let mut s = Smoother::new(Box::new(EchoPanner), 30.0, 4);
        for i in 0..20 {
            let world = WorldState {
                ball: Some(ball(0.5)),
                players: vec![],
            };
            let ctx = PanContext {
                frame_index: i,
                timestamp_ms: i as f64 * 33.3,
                previous_position: ViewportPosition::default(),
                calibration: &cal,
            };
            let out = s.decide(&world, &ctx);
            if i > 10 {
                assert!(
                    (out.yaw - 0.5).abs() < 1e-2,
                    "steady-state drifted: {}",
                    out.yaw
                );
            }
        }
    }
}
