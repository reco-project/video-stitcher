//! Viewport cropping from the panoramic render.
//!
//! The viewport defines the 16:9 (or user-chosen) rectangle that is
//! extracted from the full panoramic view. The
//! [`crate::detect::panner::Panner`] emits the per-frame yaw/pitch that
//! positions this rectangle.

use crate::geometry::Pose;

/// Configuration for the output viewport: pure output geometry.
///
/// Zoom is NOT here - fov travels with yaw/pitch in every
/// [`Pose`], so there is no retained zoom state to fall out of sync
/// with the pose stream.
#[derive(Debug, Clone)]
pub struct ViewportConfig {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
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
        Ok(())
    }
}

/// Resolved viewport state for a single frame.
///
/// Combines the output geometry with the pose (pan + zoom) to produce
/// the final camera parameters for rendering.
#[derive(Debug, Clone)]
pub struct ResolvedViewport {
    /// The viewport configuration.
    pub config: ViewportConfig,
    /// The pose for this frame; its fov drives the projection matrix.
    pub position: Pose,
}
