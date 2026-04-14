//! Debugging director that slowly pans left-right across the full coverage.
//!
//! No AI, no tracking - just a deterministic sinusoidal sweep.
//! Useful for verifying stitch quality across the entire FOV.

use reco_core::director::{Director, DirectorContext, ViewportPosition};

/// A debugging director that sweeps the virtual camera left-right.
///
/// Pans sinusoidally from `-yaw_range` to `+yaw_range` at a configurable
/// speed. No detection input is used. The pitch stays at 0.
///
/// # Example
///
/// ```rust,ignore
/// use reco_autocam::directors::SweepDirector;
///
/// // Sweep +/- 0.8 radians over 10 seconds
/// let director = SweepDirector::new(0.8, 10.0);
/// ```
pub struct SweepDirector {
    /// Maximum yaw in radians (sweeps from -yaw_range to +yaw_range).
    yaw_range: f32,
    /// Seconds for one full left-right-left cycle.
    cycle_secs: f32,
    /// Current yaw position.
    yaw: f32,
}

impl SweepDirector {
    /// Create a new sweep director.
    ///
    /// - `yaw_range`: maximum yaw in radians (e.g. 0.8 for ~46 degrees each side)
    /// - `cycle_secs`: seconds for one full sweep cycle (left-right-left)
    pub fn new(yaw_range: f32, cycle_secs: f32) -> Self {
        Self {
            yaw_range,
            cycle_secs: cycle_secs.max(0.1),
            yaw: 0.0,
        }
    }
}

impl Director for SweepDirector {
    fn update(&mut self, ctx: &DirectorContext<'_>) {
        let t = ctx.timestamp_ms / 1000.0;
        let phase = (t as f32 * std::f32::consts::TAU / self.cycle_secs).sin();
        self.yaw = phase * self.yaw_range;
    }

    fn position(&self) -> ViewportPosition {
        ViewportPosition {
            yaw: self.yaw,
            pitch: 0.0,
            fov_degrees: None,
        }
    }
}
