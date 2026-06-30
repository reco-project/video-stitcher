//! Viewport cropping from the panoramic render.
//!
//! The viewport defines the 16:9 (or user-chosen) rectangle that is
//! extracted from the full panoramic view. The
//! [`crate::detect::panner::Panner`] emits the per-frame yaw/pitch that
//! positions this rectangle.

use crate::detect::director::ViewportPosition;

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
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fov_degrees: 75.0,
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
