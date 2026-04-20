//! Adaptive dead-zone decorator for [`Panner`].
//!
//! Port of [`crate::directors::DeadZoneDirector`] to the [`Panner`]
//! contract. The math (EMA-tracked angular velocity, radius scales
//! inversely with velocity) is identical; only the wrapping
//! interface changed.
//!
//! The dead-zone suppresses sub-pixel oscillation on idle frames
//! without affecting intentional pans: the radius shrinks to zero
//! as soon as the inner panner is moving at
//! `velocity_threshold_rad_per_s` or more.

use reco_core::director::ViewportPosition;
use reco_core::panner::{PanContext, Panner};
use reco_core::tracker::WorldState;

const DEFAULT_MAX_RADIUS_RAD: f32 = 0.006;
const DEFAULT_VELOCITY_THRESHOLD: f32 = 0.30;
const VELOCITY_SMOOTH_ALPHA: f32 = 0.3;

/// Wrap a [`Panner`] with an adaptive dead-zone.
pub struct DeadZone {
    inner: Box<dyn Panner>,
    max_radius_rad: f32,
    velocity_threshold: f32,
    held: Option<ViewportPosition>,
    last_sample: Option<(f32, f32, f64)>,
    velocity_est: f32,
}

impl DeadZone {
    /// Defaults — small idle radius (0.006 rad ≈ 0.34°), velocity
    /// threshold tuned to in-play ball motion (0.30 rad/s).
    pub fn new(inner: Box<dyn Panner>) -> Self {
        Self::with_params(inner, DEFAULT_MAX_RADIUS_RAD, DEFAULT_VELOCITY_THRESHOLD)
    }

    /// Custom parameters.
    pub fn with_params(
        inner: Box<dyn Panner>,
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
        }
    }

    fn current_radius(&self, velocity_rad_per_s: f32) -> f32 {
        let shrink = (velocity_rad_per_s / self.velocity_threshold).clamp(0.0, 1.0);
        self.max_radius_rad * (1.0 - shrink)
    }
}

impl Panner for DeadZone {
    fn decide(&mut self, world: &WorldState, ctx: &PanContext<'_>) -> ViewportPosition {
        let target = self.inner.decide(world, ctx);

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

        let out = match self.held {
            Some(held) => {
                let dy = target.yaw - held.yaw;
                let dp = target.pitch - held.pitch;
                let dist = (dy * dy + dp * dp).sqrt();
                if dist < radius {
                    // Inside dead-zone: hold position but allow FOV updates
                    // so zoom stays synced with upstream.
                    ViewportPosition {
                        yaw: held.yaw,
                        pitch: held.pitch,
                        fov_degrees: target.fov_degrees,
                    }
                } else {
                    target
                }
            }
            None => target,
        };
        self.held = Some(out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::calibration::{CameraParams, MatchCalibration, PlaneLayout};

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

    struct Scripted {
        positions: Vec<(f32, f32)>,
        idx: usize,
    }
    impl Panner for Scripted {
        fn decide(&mut self, _w: &WorldState, _c: &PanContext<'_>) -> ViewportPosition {
            let (y, p) = self.positions[self.idx.min(self.positions.len() - 1)];
            self.idx += 1;
            ViewportPosition {
                yaw: y,
                pitch: p,
                fov_degrees: Some(55.0),
            }
        }
    }

    fn ctx<'a>(cal: &'a MatchCalibration, i: u64) -> PanContext<'a> {
        PanContext {
            frame_index: i,
            timestamp_ms: i as f64 * 33.3,
            previous_position: ViewportPosition::default(),
            calibration: cal,
        }
    }

    #[test]
    fn radius_collapses_at_high_velocity() {
        let dz = DeadZone::with_params(
            Box::new(Scripted {
                positions: vec![(0.0, 0.0)],
                idx: 0,
            }),
            0.006,
            0.30,
        );
        assert!((dz.current_radius(0.0) - 0.006).abs() < 1e-6);
        assert!((dz.current_radius(0.15) - 0.003).abs() < 1e-6);
        assert!(dz.current_radius(0.30) < 1e-6);
    }

    #[test]
    fn micro_movement_at_rest_is_held() {
        let cal = test_cal();
        let wiggle: Vec<(f32, f32)> = (0..12)
            .map(|i| ((i as f32 * 0.0005) % 0.001, 0.0))
            .collect();
        let mut dz = DeadZone::with_params(
            Box::new(Scripted {
                positions: wiggle,
                idx: 0,
            }),
            0.006,
            0.30,
        );
        dz.decide(&WorldState::default(), &ctx(&cal, 0));
        let held = dz.held.unwrap().yaw;
        for i in 1..12 {
            dz.decide(&WorldState::default(), &ctx(&cal, i));
        }
        assert!((dz.held.unwrap().yaw - held).abs() < 1e-6);
    }

    #[test]
    fn large_move_passes_through() {
        let cal = test_cal();
        let mut dz = DeadZone::new(Box::new(Scripted {
            positions: vec![(0.0, 0.0), (0.0, 0.0), (0.3, 0.05)],
            idx: 0,
        }));
        dz.decide(&WorldState::default(), &ctx(&cal, 0));
        dz.decide(&WorldState::default(), &ctx(&cal, 1));
        let out = dz.decide(&WorldState::default(), &ctx(&cal, 2));
        assert!(out.yaw > 0.1);
    }
}
