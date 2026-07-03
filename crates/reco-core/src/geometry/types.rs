//! The pose and source-identity currency every layer shares.
//!
//! `ViewportPosition` travels from human input and AI directors through
//! the clamp and orient stages into the render; `CameraId` names which
//! source a detection or sample came from. Both live in the geometry
//! leaf so no layer has to import another layer's domain to talk about
//! a pose.

/// Specifies the yaw, pitch, and field of view of the virtual camera.
/// The FOV allows directors to express zoom: narrow FOV = zoomed in on
/// action, wide FOV = zoomed out for context.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ViewportPosition {
    /// Horizontal pan angle in radians.
    ///
    /// `0.0` = centered on the seam between cameras.
    pub yaw: f32,

    /// Vertical tilt angle in radians.
    ///
    /// `0.0` = level. Positive = looking up.
    pub pitch: f32,

    /// Field of view in degrees, or `None` to use the pipeline's
    /// default FOV.
    ///
    /// Typical range: 30.0 (zoomed in) to 120.0 (wide). The pipeline
    /// default is 75.0.
    pub fov_degrees: Option<f32>,
}

impl Default for ViewportPosition {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: None,
        }
    }
}

/// Which camera produced this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CameraId {
    /// Left camera (plane in X-Z space).
    Left,
    /// Right camera (plane in X-Y space).
    Right,
}

impl std::fmt::Display for CameraId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Left => f.write_str("L"),
            Self::Right => f.write_str("R"),
        }
    }
}
