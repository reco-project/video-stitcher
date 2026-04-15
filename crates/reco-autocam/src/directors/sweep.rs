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
/// use reco_autocam::SweepDirector;
///
/// // Sweep +/- 0.8 radians over 10 seconds
/// let director = SweepDirector::new(0.8, 10.0);
/// ```
pub struct SweepDirector {
    /// Maximum yaw in radians (sweeps from -yaw_range to +yaw_range).
    yaw_range: f32,
    /// Seconds for one full left-right-left cycle.
    cycle_secs: f32,
    /// FOV in degrees. Should be less than the coverage boundary's
    /// max FOV, otherwise safe_clamp will pin the camera.
    fov_degrees: f32,
    /// Current yaw position.
    yaw: f32,
}

impl SweepDirector {
    /// Create a new sweep director.
    ///
    /// - `yaw_range`: maximum yaw in radians (e.g. 0.8 for ~46 degrees each side)
    /// - `cycle_secs`: seconds for one full sweep cycle (left-right-left)
    ///
    /// Uses a default FOV of 50 degrees. Use [`with_fov`](Self::with_fov)
    /// to set a FOV that fits your coverage boundary (must be less than
    /// `CoverageBoundary::max_fov_degrees()`).
    pub fn new(yaw_range: f32, cycle_secs: f32) -> Self {
        Self {
            yaw_range,
            cycle_secs: cycle_secs.max(0.1),
            fov_degrees: 50.0,
            yaw: 0.0,
        }
    }

    /// Set the FOV in degrees.
    ///
    /// Should be smaller than the coverage boundary's max FOV to
    /// allow the sweep to move freely without safe_clamp pinning it.
    pub fn with_fov(mut self, fov_degrees: f32) -> Self {
        self.fov_degrees = fov_degrees;
        self
    }
}

impl Director for SweepDirector {
    fn update(&mut self, ctx: &DirectorContext<'_>) {
        let video_time_secs = ctx.frame_index as f32 / 30.0;
        let phase = (video_time_secs * std::f32::consts::TAU / self.cycle_secs).sin();
        self.yaw = phase * self.yaw_range;
    }

    fn position(&self) -> ViewportPosition {
        ViewportPosition {
            yaw: self.yaw,
            pitch: 0.0,
            // Use a narrow FOV to ensure the viewport fits within coverage.
            // The default 75 degrees often exceeds the coverage boundary,
            // causing safe_clamp to pin the camera to one position.
            fov_degrees: Some(self.fov_degrees),
        }
    }
}
