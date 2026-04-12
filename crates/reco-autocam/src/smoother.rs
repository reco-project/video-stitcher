//! Bidirectional One Euro trajectory smoother and SmoothedDirector decorator.
//!
//! Applies a forward-backward One Euro filter over a window of raw
//! director positions, producing a smoothed position with approximately
//! zero phase lag. The One Euro filter is adaptive: slow movements get
//! heavy smoothing (no jitter), fast movements get light smoothing
//! (no lag).
//!
//! The [`SmoothedDirector`] wraps any [`Director`] implementation and
//! applies trajectory smoothing transparently. The session in reco-core
//! sees a plain `Director` - all smoothing is encapsulated here.
//!
//! Reference: Casiez, Roussel, Vogel. "1 Euro Filter: A Simple Speed-based
//! Low-pass Filter for Noisy Input in Interactive Systems." CHI 2012.

use std::collections::VecDeque;

use reco_core::director::{Director, DirectorContext, ViewportPosition};

use crate::directors::util::DEFAULT_FOV;

/// Smoothing factor from cutoff frequency and time step.
fn smoothing_factor(dt: f32, cutoff: f32) -> f32 {
    let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

/// Single-axis One Euro filter state.
///
/// Tracks position and derivative for one dimension (e.g. yaw or pitch).
/// Call [`filter`](Self::filter) once per sample in temporal order.
struct OneEuroAxis {
    x_prev: f32,
    dx_prev: f32,
    initialized: bool,
}

impl OneEuroAxis {
    fn new() -> Self {
        Self {
            x_prev: 0.0,
            dx_prev: 0.0,
            initialized: false,
        }
    }

    /// Filter one sample. Returns the smoothed value.
    ///
    /// - `x`: raw input value
    /// - `dt`: time step in seconds (1/fps)
    /// - `min_cutoff`: minimum cutoff frequency (Hz) - controls jitter at rest
    /// - `beta`: speed coefficient - controls lag during fast motion
    /// - `d_cutoff`: derivative cutoff frequency (Hz) - smooths the speed estimate
    fn filter(&mut self, x: f32, dt: f32, min_cutoff: f32, beta: f32, d_cutoff: f32) -> f32 {
        if !self.initialized {
            self.x_prev = x;
            self.dx_prev = 0.0;
            self.initialized = true;
            return x;
        }

        // Estimate derivative.
        let dx = (x - self.x_prev) / dt;
        let alpha_d = smoothing_factor(dt, d_cutoff);
        let dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev;

        // Adaptive cutoff: faster motion -> higher cutoff -> less smoothing.
        let cutoff = min_cutoff + beta * dx_hat.abs();
        let alpha = smoothing_factor(dt, cutoff);

        // Low-pass filter.
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
/// for the oldest frame (the one about to be rendered).
///
/// The bidirectional pass approximately cancels phase lag: the forward
/// pass introduces lag, the backward pass introduces equal lag in the
/// opposite direction. The result is a zero-phase-lag smoothed signal
/// with squared amplitude response (stronger noise reduction than a
/// single pass).
pub struct TrajectorySmoother {
    /// Minimum cutoff frequency (Hz). Lower = smoother at rest.
    min_cutoff: f32,
    /// Speed coefficient. Higher = less lag during fast motion.
    beta: f32,
    /// Derivative cutoff frequency (Hz). Smooths the speed estimate.
    d_cutoff: f32,
    /// Frame rate for time-step computation.
    fps: f32,
}

impl TrajectorySmoother {
    /// Create a smoother with default parameters tuned for camera paths.
    ///
    /// Defaults: `min_cutoff=0.5`, `beta=0.007`, `d_cutoff=1.0`.
    pub fn new(fps: f32) -> Self {
        let fps = fps.clamp(1.0, 1000.0);
        Self {
            min_cutoff: 0.5,
            beta: 0.007,
            d_cutoff: 1.0,
            fps,
        }
    }

    /// Override the minimum cutoff frequency.
    pub fn with_min_cutoff(mut self, min_cutoff: f32) -> Self {
        self.min_cutoff = min_cutoff;
        self
    }

    /// Override the speed coefficient.
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Smooth a window of positions and return the position for the
    /// oldest frame (index 0).
    ///
    /// `positions` must be ordered oldest-to-newest (matching the
    /// lookahead buffer's iteration order).
    pub fn smooth(&self, positions: &[ViewportPosition]) -> ViewportPosition {
        if positions.is_empty() {
            return ViewportPosition::default();
        }
        if positions.len() == 1 {
            return positions[0];
        }

        let dt = 1.0 / self.fps;
        let n = positions.len();

        // Forward pass: oldest -> newest.
        let mut fwd_yaw = OneEuroAxis::new();
        let mut fwd_pitch = OneEuroAxis::new();
        let mut fwd_fov = OneEuroAxis::new();
        let mut forward: Vec<(f32, f32, f32)> = Vec::with_capacity(n);

        for pos in positions {
            let y = fwd_yaw.filter(pos.yaw, dt, self.min_cutoff, self.beta, self.d_cutoff);
            let p = fwd_pitch.filter(pos.pitch, dt, self.min_cutoff, self.beta, self.d_cutoff);
            let f = fwd_fov.filter(
                pos.fov_degrees.unwrap_or(DEFAULT_FOV),
                dt,
                self.min_cutoff,
                self.beta,
                self.d_cutoff,
            );
            forward.push((y, p, f));
        }

        // Backward pass: newest -> oldest.
        let mut bwd_yaw = OneEuroAxis::new();
        let mut bwd_pitch = OneEuroAxis::new();
        let mut bwd_fov = OneEuroAxis::new();
        let mut backward: Vec<(f32, f32, f32)> = Vec::with_capacity(n);

        for pos in positions.iter().rev() {
            let y = bwd_yaw.filter(pos.yaw, dt, self.min_cutoff, self.beta, self.d_cutoff);
            let p = bwd_pitch.filter(pos.pitch, dt, self.min_cutoff, self.beta, self.d_cutoff);
            let f = bwd_fov.filter(
                pos.fov_degrees.unwrap_or(DEFAULT_FOV),
                dt,
                self.min_cutoff,
                self.beta,
                self.d_cutoff,
            );
            backward.push((y, p, f));
        }
        backward.reverse();

        // Average forward and backward at index 0 (the render frame).
        let (fy, fp, ff) = forward[0];
        let (by, bp, bf) = backward[0];

        ViewportPosition {
            yaw: (fy + by) * 0.5,
            pitch: (fp + bp) * 0.5,
            fov_degrees: Some((ff + bf) * 0.5),
        }
    }
}

/// Smoothing decorator for any [`Director`] implementation.
///
/// Wraps an inner director and applies One Euro trajectory smoothing.
/// Two modes depending on lookahead availability:
///
/// - **Causal** (no lookahead / buffer=1): persistent forward-only One Euro
///   filter. Smooth output with some phase lag - gives the camera a physical,
///   broadcast-like feel.
/// - **Bidirectional** (with lookahead / buffer>1): forward-backward One Euro
///   over the buffer window. Zero phase lag, stronger noise reduction.
///
/// The causal filter always runs to keep its state warm. When the buffer
/// has multiple entries, the bidirectional result is used instead.
pub struct SmoothedDirector {
    inner: Box<dyn Director>,
    smoother: TrajectorySmoother,
    buffer: VecDeque<ViewportPosition>,
    capacity: usize,
    /// Persistent causal filter state (updated every frame).
    causal_yaw: OneEuroAxis,
    causal_pitch: OneEuroAxis,
    causal_fov: OneEuroAxis,
    /// Filter parameters.
    dt: f32,
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    /// Pre-computed smoothed position (set in update, read in position).
    smoothed_position: ViewportPosition,
    /// Maximum pan speed (radians per frame).
    ///
    /// Prevents teleporting when the ball jumps across the field.
    max_slew: f32,
    /// Whether the first frame has been processed (skip slew on first frame).
    initialized: bool,
}

impl SmoothedDirector {
    /// Wrap a director with trajectory smoothing.
    ///
    /// `lookahead_frames` controls the bidirectional smoothing window.
    /// Even with 0 lookahead, the persistent causal filter provides
    /// smooth, broadcast-like camera motion.
    /// Typical lookahead: `(fps * 0.5) as usize` for 0.5s lead time.
    pub fn new(inner: Box<dyn Director>, fps: f32, lookahead_frames: usize) -> Self {
        let fps = fps.clamp(1.0, 1000.0);
        let capacity = lookahead_frames.max(1);
        let smoother = TrajectorySmoother::new(fps);
        Self {
            inner,
            dt: 1.0 / fps,
            min_cutoff: smoother.min_cutoff,
            beta: smoother.beta,
            d_cutoff: smoother.d_cutoff,
            smoother,
            buffer: VecDeque::with_capacity(capacity + 1),
            capacity,
            causal_yaw: OneEuroAxis::new(),
            causal_pitch: OneEuroAxis::new(),
            causal_fov: OneEuroAxis::new(),
            smoothed_position: ViewportPosition::default(),
            // 30 deg/s at the given fps.
            max_slew: 30.0_f32.to_radians() / fps,
            initialized: false,
        }
    }
}

impl Director for SmoothedDirector {
    fn update(&mut self, ctx: &DirectorContext<'_>) {
        // Pop the already-rendered entry from the previous iteration
        // once the buffer is full.
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }

        self.inner.update(ctx);
        let raw = self.inner.position();
        self.buffer.push_back(raw);

        // First frame: snap to the raw position (no slew from origin).
        if !self.initialized {
            self.smoothed_position = raw;
            self.initialized = true;
        }

        // Always feed the causal filter to keep state warm.
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
            raw.fov_degrees.unwrap_or(DEFAULT_FOV),
            self.dt,
            self.min_cutoff,
            self.beta,
            self.d_cutoff,
        );

        let filtered = if self.buffer.len() > 1 {
            // Bidirectional: zero-lag smoothing over the lookahead window.
            let positions: Vec<ViewportPosition> = self.buffer.iter().copied().collect();
            self.smoother.smooth(&positions)
        } else {
            // Causal: persistent filter gives broadcast-like camera motion.
            ViewportPosition {
                yaw: cy,
                pitch: cp,
                fov_degrees: Some(cf),
            }
        };

        // Slew rate limit: cap pan speed and FOV change rate to prevent
        // teleporting. FOV is slew-limited too so position and zoom change
        // at similar rates (prevents black edges during fast pans).
        let prev = self.smoothed_position;
        let dy = (filtered.yaw - prev.yaw).clamp(-self.max_slew, self.max_slew);
        let dp = (filtered.pitch - prev.pitch).clamp(-self.max_slew, self.max_slew);
        // FOV slew: ~10 deg/s at 30fps = 0.33 deg/frame.
        let max_fov_slew = 10.0 * self.dt;
        let prev_fov = prev.fov_degrees.unwrap_or(DEFAULT_FOV);
        let target_fov = filtered.fov_degrees.unwrap_or(DEFAULT_FOV);
        let df = (target_fov - prev_fov).clamp(-max_fov_slew, max_fov_slew);
        self.smoothed_position = ViewportPosition {
            yaw: prev.yaw + dy,
            pitch: prev.pitch + dp,
            fov_degrees: Some(prev_fov + df),
        };
    }

    fn position(&self) -> ViewportPosition {
        self.smoothed_position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_euro_axis_constant_signal_converges() {
        let mut axis = OneEuroAxis::new();
        let dt = 1.0 / 30.0;
        let mut v = 0.0;
        for _ in 0..200 {
            v = axis.filter(1.0, dt, 0.5, 0.007, 1.0);
        }
        assert!(
            (v - 1.0).abs() < 0.001,
            "constant signal should converge, got {v}"
        );
    }

    #[test]
    fn one_euro_axis_first_sample_passthrough() {
        let mut axis = OneEuroAxis::new();
        let v = axis.filter(42.0, 1.0 / 30.0, 0.5, 0.007, 1.0);
        assert!((v - 42.0).abs() < f32::EPSILON);
    }

    #[test]
    fn bidirectional_smoothing_reduces_jitter() {
        let smoother = TrajectorySmoother::new(30.0);
        // Alternating jitter around 0.5.
        let positions: Vec<ViewportPosition> = (0..15)
            .map(|i| ViewportPosition {
                yaw: 0.5 + if i % 2 == 0 { 0.02 } else { -0.02 },
                pitch: 0.0,
                fov_degrees: Some(55.0),
            })
            .collect();
        let smoothed = smoother.smooth(&positions);
        // Smoothed should be close to the mean (0.5).
        assert!(
            (smoothed.yaw - 0.5).abs() < 0.015,
            "jitter should be reduced, got yaw={}",
            smoothed.yaw
        );
    }

    #[test]
    fn smoothing_preserves_linear_trend() {
        let smoother = TrajectorySmoother::new(30.0);
        // Linear ramp from 0.0 to 0.14.
        let positions: Vec<ViewportPosition> = (0..15)
            .map(|i| ViewportPosition {
                yaw: i as f32 * 0.01,
                pitch: 0.0,
                fov_degrees: Some(55.0),
            })
            .collect();
        let smoothed = smoother.smooth(&positions);
        // First frame smoothed should be near the start of the ramp.
        assert!(
            smoothed.yaw < 0.05,
            "should preserve trend start, got yaw={}",
            smoothed.yaw
        );
    }

    #[test]
    fn single_position_returns_as_is() {
        let smoother = TrajectorySmoother::new(30.0);
        let pos = ViewportPosition {
            yaw: 0.5,
            pitch: 0.1,
            fov_degrees: Some(55.0),
        };
        let result = smoother.smooth(&[pos]);
        assert!((result.yaw - 0.5).abs() < f32::EPSILON);
        assert!((result.pitch - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn empty_positions_returns_default() {
        let smoother = TrajectorySmoother::new(30.0);
        let result = smoother.smooth(&[]);
        assert!((result.yaw - 0.0).abs() < f32::EPSILON);
        assert!((result.pitch - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn fov_is_smoothed_independently() {
        let smoother = TrajectorySmoother::new(30.0);
        // Constant position but varying FOV.
        let positions: Vec<ViewportPosition> = (0..15)
            .map(|i| ViewportPosition {
                yaw: 0.0,
                pitch: 0.0,
                fov_degrees: Some(if i % 2 == 0 { 60.0 } else { 40.0 }),
            })
            .collect();
        let smoothed = smoother.smooth(&positions);
        // FOV should be smoothed toward mean (~50).
        let fov = smoothed.fov_degrees.unwrap();
        assert!(
            (fov - 50.0).abs() < 8.0,
            "FOV should be smoothed, got {fov}"
        );
    }

    // ---- SmoothedDirector tests ----

    /// Minimal director that snaps to a fixed position.
    struct FixedDirector {
        pos: ViewportPosition,
    }

    impl Director for FixedDirector {
        fn update(&mut self, _ctx: &DirectorContext<'_>) {}
        fn position(&self) -> ViewportPosition {
            self.pos
        }
    }

    #[test]
    fn smoothed_director_passthrough_with_single_buffer() {
        let inner = FixedDirector {
            pos: ViewportPosition {
                yaw: 0.5,
                pitch: 0.1,
                fov_degrees: Some(55.0),
            },
        };
        let mut dir = SmoothedDirector::new(Box::new(inner), 30.0, 0);

        let ctx = DirectorContext {
            frame_index: 0,
            timestamp_ms: 0.0,
            detections: &[],
            fresh_detection: false,
        };
        dir.update(&ctx);
        let pos = dir.position();
        assert!((pos.yaw - 0.5).abs() < f32::EPSILON);
        assert!((pos.pitch - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn smoothed_director_buffer_management() {
        let inner = FixedDirector {
            pos: ViewportPosition {
                yaw: 0.0,
                pitch: 0.0,
                fov_degrees: Some(55.0),
            },
        };
        let mut dir = SmoothedDirector::new(Box::new(inner), 30.0, 5);

        let ctx = DirectorContext {
            frame_index: 0,
            timestamp_ms: 0.0,
            detections: &[],
            fresh_detection: false,
        };

        // Pre-fill: 5 updates
        for _ in 0..5 {
            dir.update(&ctx);
        }
        assert_eq!(dir.buffer.len(), 5);

        // Steady state: buffer stays at capacity
        for _ in 0..10 {
            dir.update(&ctx);
        }
        assert_eq!(dir.buffer.len(), 5);
    }
}
