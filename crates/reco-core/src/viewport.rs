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
    /// Horizontal field of view in degrees.
    ///
    /// Controls how "zoomed in" the output is. Larger values show more
    /// of the panorama. Default: 75.0 (matches v1 Three.js camera FOV).
    pub fov_degrees: f32,
    /// Seam blend width in UV space (0.0–1.0).
    ///
    /// Controls how much of the right plane's left edge fades in over the
    /// left plane using a smoothstep alpha gradient. `0.0` = hard seam,
    /// `0.15` = blend over 15% of the plane width. Default: 0.15.
    pub blend_width: f32,
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fov_degrees: 75.0,
            blend_width: 0.15,
        }
    }
}

impl ViewportConfig {
    /// Aspect ratio of the output (width / height).
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
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
