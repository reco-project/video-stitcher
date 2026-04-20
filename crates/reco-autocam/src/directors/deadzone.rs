//! Adaptive dead-zone decorator for [`Director`] implementations.
//!
//! Wraps an inner director and suppresses micro-movements of the
//! published camera position when the observed velocity is low. Large,
//! intentional pans always pass through unchanged; the dead-zone only
//! hides the sub-pixel jitter that remains after smoothing when the
//! ball is effectively stationary.
//!
//! The dead-zone radius is adaptive: at zero velocity it is
//! `max_radius_rad`, and it linearly collapses to zero as the observed
//! angular velocity approaches `velocity_threshold_rad_per_s`. This
//! keeps the camera locked during idle play and responsive during
//! active motion.
//!
//! Typical chain:
//!
//! ```text
//! BallDirector -> SmoothedDirector -> AnticipatingDirector -> DeadZoneDirector
//! ```
//!
//! Running the dead-zone last means the final rendered position is what
//! gets held - any upstream smoothing/anticipation feeds in, but micro
//! oscillations never reach the renderer.

use reco_core::director::{Director, DirectorContext, ViewportPosition};

/// Default dead-zone radius at zero velocity (radians).
/// Roughly 0.34 degrees — smaller than a ball at typical tracking FOV.
const DEFAULT_MAX_RADIUS_RAD: f32 = 0.006;

/// Default velocity at which the dead-zone fully collapses (rad/s).
const DEFAULT_VELOCITY_THRESHOLD: f32 = 0.30;

/// Exponential-smoothing factor for the internal velocity estimate.
const VELOCITY_SMOOTH_ALPHA: f32 = 0.3;

/// Wrap a [`Director`] with an adaptive dead-zone.
///
/// At each update the decorator:
/// 1. Runs the inner director.
/// 2. Estimates the inner director's angular velocity (EMA of frame-over-
///    frame position delta).
/// 3. Computes a dead-zone radius that scales inversely with velocity.
/// 4. Publishes the inner position if it moved further than the radius
///    from the held position; otherwise holds the previous output.
pub struct DeadZoneDirector {
    inner: Box<dyn Director>,
    max_radius_rad: f32,
    velocity_threshold: f32,
    held: Option<ViewportPosition>,
    last_sample: Option<(f32, f32, f64)>,
    velocity_est: f32,
    out: ViewportPosition,
}

impl DeadZoneDirector {
    /// Wrap a director with default dead-zone parameters.
    pub fn new(inner: Box<dyn Director>) -> Self {
        Self::with_params(inner, DEFAULT_MAX_RADIUS_RAD, DEFAULT_VELOCITY_THRESHOLD)
    }

    /// Wrap a director with custom dead-zone parameters.
    ///
    /// - `max_radius_rad`: dead-zone radius at zero velocity. Higher =
    ///   more aggressive idle-holding. Values around 0.003-0.015 rad
    ///   (0.17-0.86°) work well for football panoramas.
    /// - `velocity_threshold_rad_per_s`: velocity at which the radius
    ///   collapses to zero. Above this, no dead-zone is applied. The
    ///   default 0.30 rad/s covers typical in-play ball motion.
    pub fn with_params(
        inner: Box<dyn Director>,
        max_radius_rad: f32,
        velocity_threshold_rad_per_s: f32,
    ) -> Self {
        Self {
            inner,
            max_radius_rad: max_radius_rad.max(0.0),
            velocity_threshold: velocity_threshold_rad_per_s.max(1e-3),
            held: None,
            last_sample: None,
            velocity_est: 0.0,
            out: ViewportPosition::default(),
        }
    }

    /// Current adaptive radius, for tests and debug.
    fn current_radius(&self, velocity_rad_per_s: f32) -> f32 {
        let shrink = (velocity_rad_per_s / self.velocity_threshold).clamp(0.0, 1.0);
        self.max_radius_rad * (1.0 - shrink)
    }
}

impl Director for DeadZoneDirector {
    fn update(&mut self, ctx: &DirectorContext<'_>) {
        self.inner.update(ctx);
        let target = self.inner.position();

        // Update velocity estimate from inner position deltas.
        if let Some((prev_y, prev_p, prev_t)) = self.last_sample {
            let dt_ms = ctx.timestamp_ms - prev_t;
            if dt_ms > 0.1 {
                let dy = target.yaw - prev_y;
                let dp = target.pitch - prev_p;
                let mag = (dy * dy + dp * dp).sqrt();
                let per_sec = mag / (dt_ms as f32 / 1000.0);
                self.velocity_est = (1.0 - VELOCITY_SMOOTH_ALPHA) * self.velocity_est
                    + VELOCITY_SMOOTH_ALPHA * per_sec;
            }
        }
        self.last_sample = Some((target.yaw, target.pitch, ctx.timestamp_ms));

        let radius = self.current_radius(self.velocity_est);

        match self.held {
            Some(held) => {
                let dy = target.yaw - held.yaw;
                let dp = target.pitch - held.pitch;
                let dist = (dy * dy + dp * dp).sqrt();
                if dist < radius {
                    // Inside the dead-zone: keep holding, but let FOV
                    // updates through so zoom keeps tracking smoothly.
                    self.out = ViewportPosition {
                        yaw: held.yaw,
                        pitch: held.pitch,
                        fov_degrees: target.fov_degrees,
                    };
                } else {
                    self.held = Some(target);
                    self.out = target;
                }
            }
            None => {
                self.held = Some(target);
                self.out = target;
            }
        }
    }

    fn position(&self) -> ViewportPosition {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scripted {
        positions: Vec<(f32, f32)>,
        idx: usize,
    }
    impl Director for Scripted {
        fn update(&mut self, _ctx: &DirectorContext<'_>) {
            self.idx = (self.idx + 1).min(self.positions.len() - 1);
        }
        fn position(&self) -> ViewportPosition {
            let (y, p) = self.positions[self.idx];
            ViewportPosition {
                yaw: y,
                pitch: p,
                fov_degrees: Some(55.0),
            }
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
    fn micro_movement_is_suppressed_at_rest() {
        // Inner position wiggles by 0.0005 rad while effectively stationary.
        let wiggles: Vec<(f32, f32)> = (0..12)
            .map(|i| ((i as f32 * 0.0005) % 0.001, 0.0))
            .collect();
        let inner = Box::new(Scripted {
            positions: wiggles,
            idx: 0,
        });
        let mut d = DeadZoneDirector::with_params(inner, 0.006, 0.30);
        // First call establishes hold.
        d.update(&ctx(0, 0.0));
        let held = d.position().yaw;
        for i in 1..12 {
            d.update(&ctx(i, i as f64 * 1000.0 / 30.0));
        }
        // Output should still equal the held value (wiggles < 0.006 rad).
        assert!(
            (d.position().yaw - held).abs() < 1e-6,
            "expected held yaw, got {}",
            d.position().yaw
        );
    }

    #[test]
    fn large_movement_passes_through() {
        // Inner makes a big move.
        let positions = vec![(0.0, 0.0), (0.0, 0.0), (0.3, 0.05)];
        let inner = Box::new(Scripted { positions, idx: 0 });
        let mut d = DeadZoneDirector::new(inner);
        d.update(&ctx(0, 0.0));
        d.update(&ctx(1, 33.3));
        d.update(&ctx(2, 66.6));
        assert!(
            d.position().yaw > 0.1,
            "expected large move to pass through; got {}",
            d.position().yaw
        );
    }

    #[test]
    fn radius_collapses_at_high_velocity() {
        let inner = Box::new(Scripted {
            positions: vec![(0.0, 0.0)],
            idx: 0,
        });
        let d = DeadZoneDirector::with_params(inner, 0.006, 0.30);
        assert!((d.current_radius(0.0) - 0.006).abs() < 1e-6);
        assert!((d.current_radius(0.15) - 0.003).abs() < 1e-6);
        assert!(d.current_radius(0.30) < 1e-6);
        assert!(d.current_radius(5.0) < 1e-6);
    }

    #[test]
    fn fov_updates_pass_through_while_held() {
        // Inner stays nearly stationary but emits different FOVs.
        struct FovScripted {
            idx: u64,
        }
        impl Director for FovScripted {
            fn update(&mut self, _ctx: &DirectorContext<'_>) {
                self.idx += 1;
            }
            fn position(&self) -> ViewportPosition {
                ViewportPosition {
                    yaw: 0.0005,
                    pitch: 0.0,
                    fov_degrees: Some(40.0 + self.idx as f32),
                }
            }
        }
        let mut d = DeadZoneDirector::with_params(Box::new(FovScripted { idx: 0 }), 0.006, 0.30);
        d.update(&ctx(0, 0.0));
        let f0 = d.position().fov_degrees.unwrap();
        d.update(&ctx(1, 33.3));
        let f1 = d.position().fov_degrees.unwrap();
        assert!(f1 > f0, "expected FOV to track inner; f0={f0}, f1={f1}");
    }
}
