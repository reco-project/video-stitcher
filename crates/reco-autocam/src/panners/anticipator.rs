//! Velocity-anticipation decorator for [`Panner`].
//!
//! On each frame:
//! 1. Forward to inner panner for the "raw" target.
//! 2. Record `(yaw, pitch, t_ms)` into a short history.
//! 3. Estimate instantaneous velocity via least-squares fit.
//! 4. Damp the lead contribution by estimated trajectory curvature
//!    so cut-backs don't cause the camera to over-extrapolate.
//! 5. Publish `raw + lead * velocity`, hard-capped per axis.

use std::collections::VecDeque;

use reco_core::director::ViewportPosition;
use reco_core::panner::{PanContext, Panner};
use reco_core::tracker::WorldState;

const MIN_HISTORY: usize = 3;
const DEFAULT_HISTORY: usize = 8;
const DEFAULT_LEAD_SECONDS: f32 = 0.15;
const DEFAULT_MAX_LEAD_RAD: f32 = 0.052;

#[derive(Clone, Copy)]
struct Sample {
    yaw: f32,
    pitch: f32,
    t_ms: f64,
}

/// Wrap a [`Panner`] with velocity-based trajectory anticipation.
pub struct Anticipator {
    inner: Box<dyn Panner>,
    history: VecDeque<Sample>,
    max_history: usize,
    lead_seconds: f32,
    max_lead_rad: f32,
}

impl Anticipator {
    /// Defaults: 8-sample history, 0.15 s lead, ~3° per-frame cap.
    pub fn new(inner: Box<dyn Panner>) -> Self {
        Self::with_params(inner, DEFAULT_LEAD_SECONDS, DEFAULT_MAX_LEAD_RAD)
    }

    /// Custom parameters.
    ///
    /// - `lead_seconds`: extrapolation horizon (practical 0.05-0.30s).
    /// - `max_lead_rad`: hard cap on added lead per axis.
    pub fn with_params(inner: Box<dyn Panner>, lead_seconds: f32, max_lead_rad: f32) -> Self {
        Self {
            inner,
            history: VecDeque::with_capacity(DEFAULT_HISTORY + 1),
            max_history: DEFAULT_HISTORY,
            lead_seconds: lead_seconds.max(0.0),
            max_lead_rad: max_lead_rad.max(0.0),
        }
    }

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
        // ms → seconds conversion: slope is per ms, × 1000 → per s.
        Some(((ty / tt) as f32 * 1000.0, (tp / tt) as f32 * 1000.0))
    }

    /// Curvature in `[0, 1]`; 0 = straight, 1 = reversed direction.
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
        let m1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
        let m2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
        if m1 < 1e-6 || m2 < 1e-6 {
            return 0.0;
        }
        let cos = ((v1.0 * v2.0 + v1.1 * v2.1) / (m1 * m2)).clamp(-1.0, 1.0);
        ((1.0 - cos) * 0.5).clamp(0.0, 1.0)
    }
}

impl Panner for Anticipator {
    fn decide(&mut self, world: &WorldState, ctx: &PanContext<'_>) -> ViewportPosition {
        let raw = self.inner.decide(world, ctx);
        self.history.push_back(Sample {
            yaw: raw.yaw,
            pitch: raw.pitch,
            t_ms: ctx.timestamp_ms,
        });
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
        let (mut lead_y, mut lead_p) = (0.0_f32, 0.0_f32);
        if let Some((vy, vp)) = self.estimate_velocity() {
            let curv = self.estimate_curvature();
            let damp = 1.0 - curv;
            let lead_sec = self.lead_seconds * damp;
            lead_y = (vy * lead_sec).clamp(-self.max_lead_rad, self.max_lead_rad);
            lead_p = (vp * lead_sec).clamp(-self.max_lead_rad, self.max_lead_rad);
        }
        ViewportPosition {
            yaw: raw.yaw + lead_y,
            pitch: raw.pitch + lead_p,
            fov_degrees: raw.fov_degrees,
        }
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

    struct RampPanner {
        yaw: f32,
    }
    impl Panner for RampPanner {
        fn decide(&mut self, _w: &WorldState, _c: &PanContext<'_>) -> ViewportPosition {
            self.yaw += 0.01;
            ViewportPosition {
                yaw: self.yaw,
                pitch: 0.0,
                fov_degrees: None,
            }
        }
    }

    struct FixedPanner(ViewportPosition);
    impl Panner for FixedPanner {
        fn decide(&mut self, _w: &WorldState, _c: &PanContext<'_>) -> ViewportPosition {
            self.0
        }
    }

    fn ctx<'a>(cal: &'a MatchCalibration, i: u64) -> PanContext<'a> {
        PanContext {
            frame_index: i,
            timestamp_ms: i as f64 * 1000.0 / 30.0,
            previous_position: ViewportPosition::default(),
            calibration: cal,
        }
    }

    #[test]
    fn no_history_means_no_lead() {
        let cal = test_cal();
        let mut a = Anticipator::new(Box::new(FixedPanner(ViewportPosition {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: None,
        })));
        let out = a.decide(&WorldState::default(), &ctx(&cal, 0));
        assert_eq!(out.yaw, 0.0);
    }

    #[test]
    fn constant_velocity_produces_positive_lead() {
        let cal = test_cal();
        let mut a = Anticipator::with_params(Box::new(RampPanner { yaw: 0.0 }), 0.15, 0.1);
        for i in 0..10 {
            a.decide(&WorldState::default(), &ctx(&cal, i));
        }
        let out = a.decide(&WorldState::default(), &ctx(&cal, 10));
        // Inner yaw after 11 calls = 0.11; output should be larger by the lead.
        assert!(out.yaw > 0.11, "expected lead, got {}", out.yaw);
    }
}
