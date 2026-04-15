//! Viewport cropping from the panoramic render.
//!
//! The viewport defines the 16:9 (or user-chosen) rectangle that is
//! extracted from the full panoramic view. The [`crate::director::Director`]
//! controls the viewport position via yaw/pitch.

use crate::director::ViewportPosition;

/// Configuration for the output viewport.
#[derive(Debug, Clone)]
pub struct ViewportConfig {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Vertical field of view in degrees.
    ///
    /// Controls how "zoomed in" the output is. Larger values show more
    /// of the panorama. Default: 75.0 (matches v1 Three.js camera FOV).
    /// Note: this is vertical FOV per nalgebra's `Perspective3` convention.
    pub fov_degrees: f32,
    /// Seam blend width in UV space (0.0–1.0).
    ///
    /// Controls how much of the right plane's left edge fades in over the
    /// left plane using a smoothstep alpha gradient. `0.0` = hard seam,
    /// `0.15` = blend over 15% of the plane width. Default: 0.15.
    pub blend_width: f32,
    /// Rig tilt in radians (forward lean from vertical).
    ///
    /// Rotates the entire scene (both planes) to compensate for a
    /// physically tilted camera rig. When panning, this creates a
    /// natural roll correction that straightens vertical lines at the
    /// edges. `0.0` = no correction. Default: 0.0.
    pub rig_tilt: f32,
    /// Rig roll in radians (lateral lean).
    ///
    /// Rotates the scene around the forward axis to compensate for a
    /// laterally tilted camera rig. `0.0` = no correction. Default: 0.0.
    pub rig_roll: f32,
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fov_degrees: 75.0,
            blend_width: 0.05,
            rig_tilt: 0.0,
            rig_roll: 0.0,
        }
    }
}

impl ViewportConfig {
    /// Aspect ratio of the output (width / height).
    ///
    /// Returns 1.0 if height is zero (degenerate viewport).
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            return 1.0;
        }
        self.width as f32 / self.height as f32
    }

    /// Validate the viewport configuration.
    ///
    /// Returns an error description if any field is invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err(format!(
                "viewport dimensions must be non-zero, got {}x{}",
                self.width, self.height
            ));
        }
        if !(1.0..179.0).contains(&self.fov_degrees) {
            return Err(format!(
                "fov_degrees must be in (1, 179), got {}",
                self.fov_degrees
            ));
        }
        if !(0.0..=1.0).contains(&self.blend_width) {
            return Err(format!(
                "blend_width must be in [0, 1], got {}",
                self.blend_width
            ));
        }
        Ok(())
    }
}

/// Resolved viewport state for a single frame.
///
/// Combines the viewport configuration with the director's pan position
/// to produce the final camera parameters for rendering.
#[derive(Debug, Clone)]
pub struct ResolvedViewport {
    /// The viewport configuration.
    pub config: ViewportConfig,
    /// The pan position for this frame.
    pub position: ViewportPosition,
}
