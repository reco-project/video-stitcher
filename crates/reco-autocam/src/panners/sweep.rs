//! Debugging panner that slowly pans left-right across the full
//! coverage.
//!
//! No AI, no tracking — just a deterministic sinusoidal sweep driven
//! by the per-frame [`PanContext::frame_index`]. Ignores
//! [`WorldState`] entirely. Useful for verifying stitch quality across
//! the full FOV and for smoke-testing panner dispatch paths.

use reco_core::detect::director::ViewportPosition;
use reco_core::detect::panner::{PanContext, Panner};
use reco_core::detect::tracker::WorldState;

/// A debugging panner that sweeps the virtual camera left-right.
///
/// Pans sinusoidally from `-yaw_range` to `+yaw_range` over
/// `cycle_secs` seconds. FOV defaults to 50° (narrow enough to stay
/// inside typical coverage boundaries without the safe-clamp pinning
/// the camera to one edge).
pub struct SweepPanner {
    yaw_range: f32,
    cycle_secs: f32,
    fov_degrees: f32,
    fov_min: f32,
    fov_max: f32,
    zoom_cycle_secs: f32,
    /// Source frame rate used to turn the session's monotonically
    /// increasing `frame_index` into a seconds-valued phase. Matches
    /// the director's hard-coded 30 fps for behavior parity.
    fps: f32,
}

impl SweepPanner {
    /// Create a new sweep panner.
    ///
    /// - `yaw_range`: maximum yaw in radians.
    /// - `cycle_secs`: seconds per full left-right-left cycle.
    pub fn new(yaw_range: f32, cycle_secs: f32) -> Self {
        Self {
            yaw_range,
            cycle_secs: cycle_secs.max(0.1),
            fov_degrees: 50.0,
            fov_min: 0.0,
            fov_max: 0.0,
            zoom_cycle_secs: 0.0,
            fps: 30.0,
        }
    }

    /// Override the fixed FOV in degrees (disables zoom).
    pub fn with_fov(mut self, fov_degrees: f32) -> Self {
        self.fov_degrees = fov_degrees;
        self
    }

    /// Enable sinusoidal zoom between `fov_min` and `fov_max` degrees
    /// over `cycle_secs`. Uses a different period than the yaw sweep
    /// so the zoom and pan don't synchronize.
    pub fn with_zoom(mut self, fov_min: f32, fov_max: f32, cycle_secs: f32) -> Self {
        self.fov_min = fov_min;
        self.fov_max = fov_max;
        self.zoom_cycle_secs = cycle_secs.max(0.1);
        self
    }

    /// Override the frame rate used to compute phase.
    pub fn with_fps(mut self, fps: f32) -> Self {
        self.fps = fps.max(1.0);
        self
    }
}

impl Panner for SweepPanner {
    fn decide(&mut self, _world: &WorldState, ctx: &PanContext<'_>) -> ViewportPosition {
        let t = ctx.frame_index as f32 / self.fps;
        let yaw_phase = (t * std::f32::consts::TAU / self.cycle_secs).sin();

        let fov = if self.zoom_cycle_secs > 0.0 {
            let zoom_phase = (t * std::f32::consts::TAU / self.zoom_cycle_secs).sin();
            let mid = (self.fov_min + self.fov_max) * 0.5;
            let amp = (self.fov_max - self.fov_min) * 0.5;
            mid + zoom_phase * amp
        } else {
            self.fov_degrees
        };

        ViewportPosition {
            yaw: yaw_phase * self.yaw_range,
            pitch: 0.0,
            fov_degrees: Some(fov),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::calibration::{Calibration, Framing, Lens, Topology};

    fn test_cal() -> Calibration {
        let cam = || Lens::fisheye(1920, 1080, 900.0, 900.0, 960.0, 540.0, [0.0; 4]);
        Calibration::new(
            vec![cam(), cam()],
            Topology {
                intersect: 0.54,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            Framing {
                axis_offset: 0.24,
                tilt: 0.0,
                roll: 0.0,
            },
        )
    }

    fn ctx<'a>(frame_index: u64, cal: &'a Calibration) -> PanContext<'a> {
        PanContext {
            frame_index,
            timestamp_ms: frame_index as f64 * (1000.0 / 30.0),
            previous_position: ViewportPosition::default(),
            calibration: cal,
        }
    }

    #[test]
    fn zero_phase_at_origin() {
        let mut p = SweepPanner::new(0.8, 10.0);
        let cal = test_cal();
        let out = p.decide(&WorldState::default(), &ctx(0, &cal));
        assert!(
            out.yaw.abs() < 1e-6,
            "yaw should be 0 at start: {}",
            out.yaw
        );
    }

    #[test]
    fn quarter_cycle_reaches_range_peak() {
        let mut p = SweepPanner::new(0.8, 10.0);
        let cal = test_cal();
        // Quarter cycle at 30 fps * 2.5s = 75 frames.
        let out = p.decide(&WorldState::default(), &ctx(75, &cal));
        assert!(
            (out.yaw - 0.8).abs() < 0.05,
            "yaw should approach +range at quarter cycle: {}",
            out.yaw
        );
    }

    #[test]
    fn stays_within_range() {
        let mut p = SweepPanner::new(0.8, 10.0);
        let cal = test_cal();
        for i in 0..300 {
            let out = p.decide(&WorldState::default(), &ctx(i, &cal));
            assert!(out.yaw.abs() <= 0.8 + 1e-6);
        }
    }

    #[test]
    fn ignores_world_state() {
        let mut p = SweepPanner::new(0.8, 10.0);
        let cal = test_cal();
        // Seed with a fake ball far from origin — sweep must not react.
        let w = WorldState {
            ball: Some(reco_core::detect::tracker::TrackedEntity {
                id: 0,
                class_id: 0,
                yaw: 1.5,
                pitch: 0.3,
                confidence: 1.0,
                state: reco_core::detect::tracker::TrackState::Tracking,
                age_frames: 1,
                origin: reco_core::detect::detector::CameraId::Left,
            }),
            players: Vec::new(),
        };
        let a = p.decide(&w, &ctx(0, &cal));
        let b = p.decide(&WorldState::default(), &ctx(0, &cal));
        assert!((a.yaw - b.yaw).abs() < 1e-6);
    }

    #[test]
    fn fov_override_applied() {
        let mut p = SweepPanner::new(0.8, 10.0).with_fov(42.0);
        let cal = test_cal();
        let out = p.decide(&WorldState::default(), &ctx(0, &cal));
        assert_eq!(out.fov_degrees, Some(42.0));
    }
}
