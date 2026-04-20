//! Trajectory anticipation decorator for [`Director`] implementations.
//!
//! Wraps an inner director and projects its output forward along the
//! observed yaw/pitch velocity, so the rendered camera arrives at the
//! ball's position instead of trailing it.
//!
//! Lead is damped when the trajectory curvature is high, to avoid
//! overshooting on sharp direction changes. Lead is also hard-capped
//! to prevent runaway extrapolation when velocity estimates are noisy.
//!
//! Typical chain:
//!
//! ```text
//! BallDirector -> SmoothedDirector -> AnticipatingDirector
//! ```
//!
//! Running anticipation *after* smoothing keeps the velocity estimate
//! clean (computed from smoothed positions, not raw jumpy detections).

use std::collections::VecDeque;

use reco_core::director::{Director, DirectorContext, ViewportPosition};

/// Minimum history length before anticipation activates.
const MIN_HISTORY: usize = 3;

/// Default history window (samples).
const DEFAULT_HISTORY: usize = 8;

/// Default lookahead in seconds.
const DEFAULT_LEAD_SECONDS: f32 = 0.15;

/// Default per-frame cap on added lead, in radians. ~3 degrees.
const DEFAULT_MAX_LEAD_RAD: f32 = 0.052;

/// Wrap a [`Director`] with velocity-based trajectory anticipation.
///
/// On each frame the decorator:
/// 1. Updates the inner director.
/// 2. Records the inner director's reported position + timestamp.
/// 3. Estimates instantaneous yaw/pitch velocity via least-squares fit
///    over the recent history window.
/// 4. Dampens the lead by an estimated curvature factor (sharper turns
///    get less lead, keeping the camera from overshooting a cut-back).
/// 5. Exposes `inner_position + lead * velocity` (hard-capped) as the
///    published position.
pub struct AnticipatingDirector {
    inner: Box<dyn Director>,
    history: VecDeque<Sample>,
    max_history: usize,
    lead_seconds: f32,
    max_lead_rad: f32,
    out: ViewportPosition,
}

#[derive(Clone, Copy)]
struct Sample {
    yaw: f32,
    pitch: f32,
    t_ms: f64,
}

impl AnticipatingDirector {
    /// Wrap a director with default anticipation parameters.
    ///
    /// Defaults: 8-sample history, 0.15s lead, ~3° per-frame lead cap.
    pub fn new(inner: Box<dyn Director>) -> Self {
        Self::with_params(inner, DEFAULT_LEAD_SECONDS, DEFAULT_MAX_LEAD_RAD)
    }

    /// Wrap a director with custom anticipation parameters.
    ///
    /// - `lead_seconds`: how far ahead to project along the velocity
    ///   vector. Practical values are 0.05-0.30s; higher values over-
    ///   extrapolate and feel unnatural.
    /// - `max_lead_rad`: per-frame cap on added lead. Protects against
    ///   runaway extrapolation when velocity estimates are noisy.
    pub fn with_params(inner: Box<dyn Director>, lead_seconds: f32, max_lead_rad: f32) -> Self {
        Self {
            inner,
            history: VecDeque::with_capacity(DEFAULT_HISTORY + 1),
            max_history: DEFAULT_HISTORY,
            lead_seconds: lead_seconds.max(0.0),
            max_lead_rad: max_lead_rad.max(0.0),
            out: ViewportPosition::default(),
        }
    }

    /// Estimate (v_yaw, v_pitch) in radians per second via least-squares
    /// fit of position vs. time over the history window. Returns `None`
    /// when there are too few samples or the time span is degenerate.
    fn estimate_velocity(&self) -> Option<(f32, f32)> {
        if self.history.len() < MIN_HISTORY {
            return None;
        }

        let n = self.history.len() as f64;
        let t_mean: f64 = self.history.iter().map(|s| s.t_ms).sum::<f64>() / n;
        let y_mean: f64 = self.history.iter().map(|s| s.yaw as f64).sum::<f64>() / n;
        let p_mean: f64 = self.history.iter().map(|s| s.pitch as f64).sum::<f64>() / n;

        let mut tt = 0.0_f64;
        let mut ty = 0.0_f64;
        let mut tp = 0.0_f64;
        for s in &self.history {
            let dt = s.t_ms - t_mean;
            tt += dt * dt;
            ty += dt * (s.yaw as f64 - y_mean);
            tp += dt * (s.pitch as f64 - p_mean);
        }

        if tt < 1e-6 {
            return None;
        }

        // slope is per-millisecond; convert to per-second.
        let vy = (ty / tt) * 1000.0;
        let vp = (tp / tt) * 1000.0;
        Some((vy as f32, vp as f32))
    }

    /// Estimate trajectory curvature in `[0, 1]`. Higher = sharper turn.
    ///
    /// Computed from the angle between the average velocity vectors of
    /// the first and second halves of the history window. A curvature
    /// near 1 means the direction flipped, and lead should be damped
    /// aggressively.
    fn estimate_curvature(&self) -> f32 {
        let n = self.history.len();
        if n < 4 {
            return 0.0;
        }
        let mid = n / 2;
        let early_start = &self.history[0];
        let early_end = &self.history[mid - 1];
        let late_start = &self.history[mid];
        let late_end = &self.history[n - 1];

        let v1 = (
            early_end.yaw - early_start.yaw,
            early_end.pitch - early_start.pitch,
        );
        let v2 = (
            late_end.yaw - late_start.yaw,
            late_end.pitch - late_start.pitch,
        );

        let mag1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
        let mag2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
        if mag1 < 1e-6 || mag2 < 1e-6 {
            return 0.0;
        }

        // cos(theta) in [-1, 1]; map to curvature in [0, 1].
        let cos = ((v1.0 * v2.0 + v1.1 * v2.1) / (mag1 * mag2)).clamp(-1.0, 1.0);
        ((1.0 - cos) * 0.5).clamp(0.0, 1.0)
    }
}

impl Director for AnticipatingDirector {
    fn update(&mut self, ctx: &DirectorContext<'_>) {
        self.inner.update(ctx);
        let raw = self.inner.position();

        self.history.push_back(Sample {
            yaw: raw.yaw,
            pitch: raw.pitch,
            t_ms: ctx.timestamp_ms,
        });
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }

        let mut lead_y = 0.0;
        let mut lead_p = 0.0;
        if let Some((vy, vp)) = self.estimate_velocity() {
            // Damp lead during direction changes: curv=0 keeps full lead,
            // curv=1 collapses it. Keeps the camera stable on cut-backs.
            let curv = self.estimate_curvature();
            let damp = 1.0 - curv;
            let lead_sec = self.lead_seconds * damp;
            lead_y = (vy * lead_sec).clamp(-self.max_lead_rad, self.max_lead_rad);
            lead_p = (vp * lead_sec).clamp(-self.max_lead_rad, self.max_lead_rad);
        }

        self.out = ViewportPosition {
            yaw: raw.yaw + lead_y,
            pitch: raw.pitch + lead_p,
            fov_degrees: raw.fov_degrees,
        };
    }

    fn position(&self) -> ViewportPosition {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedDirector(ViewportPosition);
    impl Director for FixedDirector {
        fn update(&mut self, _ctx: &DirectorContext<'_>) {}
        fn position(&self) -> ViewportPosition {
            self.0
        }
    }

    fn ctx(frame: u64, t_ms: f64) -> DirectorContext<'static> {
        DirectorContext {
            frame_index: frame,
            timestamp_ms: t_ms,
            detections: &[],
            fresh_detection: true,
        }
    }

    #[test]
    fn no_history_no_lead() {
        let inner = Box::new(FixedDirector(ViewportPosition {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: None,
        }));
        let mut d = AnticipatingDirector::new(inner);
        d.update(&ctx(0, 0.0));
        assert_eq!(d.position().yaw, 0.0);
    }

    #[test]
    fn constant_velocity_produces_positive_lead() {
        // Inner emits yaw increasing by 0.01 rad per frame.
        struct Ramp(f32);
        impl Director for Ramp {
            fn update(&mut self, _ctx: &DirectorContext<'_>) {
                self.0 += 0.01;
            }
            fn position(&self) -> ViewportPosition {
                ViewportPosition {
                    yaw: self.0,
                    pitch: 0.0,
                    fov_degrees: None,
                }
            }
        }
        let mut d = AnticipatingDirector::with_params(Box::new(Ramp(0.0)), 0.15, 0.1);
        for i in 0..10 {
            d.update(&ctx(i, i as f64 * 1000.0 / 30.0));
        }
        let raw_yaw = 0.01 * 10.0;
        let out_yaw = d.position().yaw;
        assert!(
            out_yaw > raw_yaw,
            "expected anticipated yaw > raw; raw={raw_yaw}, out={out_yaw}"
        );
    }

    #[test]
    fn direction_reversal_dampens_lead() {
        // Inner bounces yaw sharply — curvature ~ 1.
        struct Zigzag {
            frame: u32,
        }
        impl Director for Zigzag {
            fn update(&mut self, _ctx: &DirectorContext<'_>) {
                self.frame += 1;
            }
            fn position(&self) -> ViewportPosition {
                // First half positive yaw, second half negative.
                let y = if self.frame <= 4 {
                    self.frame as f32 * 0.02
                } else {
                    0.08 - (self.frame as f32 - 4.0) * 0.02
                };
                ViewportPosition {
                    yaw: y,
                    pitch: 0.0,
                    fov_degrees: None,
                }
            }
        }
        let mut d = AnticipatingDirector::with_params(Box::new(Zigzag { frame: 0 }), 0.15, 0.1);
        for i in 0..8 {
            d.update(&ctx(i, i as f64 * 1000.0 / 30.0));
        }
        // After the reversal, curvature should be high; lead_y should be
        // near zero. Bound loosely (the fit is noisy).
        let raw = d.inner.position().yaw;
        let diff = (d.position().yaw - raw).abs();
        assert!(
            diff < 0.03,
            "lead should be damped by curvature; raw={raw}, out={}, diff={diff}",
            d.position().yaw
        );
    }
}
