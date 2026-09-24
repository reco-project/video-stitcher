//! Output viewport dimensions. Pose and FOV select the view rendered into this rectangle; no intermediate panorama image is required.

/// Output viewport geometry: the rectangle the panorama is rendered
/// into. Zoom travels with yaw/pitch in every
/// [`RenderPose`](crate::geometry::RenderPose), never here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewportSize {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

impl Default for ViewportSize {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
        }
    }
}

impl ViewportSize {
    /// Convert an output pixel centre to normalized device coordinates, with positive Y pointing up.
    pub(crate) fn pixel_ndc(&self, x: u32, y: u32) -> [f64; 2] {
        [
            (x as f64 + 0.5) / self.width as f64 * 2.0 - 1.0,
            1.0 - (y as f64 + 0.5) / self.height as f64 * 2.0,
        ]
    }

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
