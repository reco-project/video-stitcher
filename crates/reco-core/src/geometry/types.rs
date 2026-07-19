//! The pose and source-identity currency every layer shares.
//!
//! `Pose` travels from human input and AI directors through
//! the clamp and orient stages into the render; `CameraIndex` names which
//! source a detection or sample came from. Both live in the geometry
//! leaf so no layer has to import another layer's domain to talk about
//! a pose.

/// Specifies the yaw, pitch, and field of view of the virtual camera.
/// The FOV allows directors to express zoom: narrow FOV = zoomed in on
/// action, wide FOV = zoomed out for context.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Pose {
    /// Horizontal pan angle in radians.
    ///
    /// `0.0` = centered on the seam between cameras.
    pub yaw: f32,

    /// Vertical tilt angle in radians.
    ///
    /// `0.0` = level. Positive = looking up.
    pub pitch: f32,

    /// Vertical field of view in degrees (nalgebra `Perspective3`
    /// convention).
    ///
    /// Typical range: 30.0 (zoomed in) to 120.0 (wide). The single
    /// home for fov: it travels with yaw/pitch in every pose, and the
    /// render boundary clamps it to `(1, 179)` before building a
    /// projection matrix. There is no retained pipeline fov state.
    pub fov_degrees: f32,
}

impl Pose {
    /// The fov this pose renders with: clamped to `(1, 179)` degrees.
    ///
    /// The single home for the render-boundary guard - 0 or 180 would
    /// produce a NaN/Inf projection matrix (GPU) or a degenerate
    /// frustum tangent (CPU maps). Every place a pose becomes a
    /// matrix or ray basis goes through this.
    pub fn render_fov(&self) -> f32 {
        self.fov_degrees.clamp(1.0, 179.0)
    }
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: 75.0,
        }
    }
}

/// Which camera produced a frame or detection: the index into
/// `calibration.lenses`, in projection order (the L-shape's camera 0
/// is the X-Z plane, camera 1 the X-Y plane).
///
/// A plain index, not an enum: the calibration document is the
/// authority on how many cameras exist, and serialized forms (the
/// events JSONL) carry the same integer.
pub type CameraIndex = usize;
