//! World-reference and render-space view poses, angular directions and source identity.

/// A requested view in the virtual camera's world-reference frame, before rig tilt/roll correction.
///
/// Panners produce this type. Convert through the calibrated world-to-render operation before rendering; copying the angles into a [`RenderPose`] does not perform that conversion.
///
/// A rendered result cannot be submitted as another world-space request:
/// ```compile_fail
/// use reco_core::{core::StitchCore, geometry::RenderPose};
/// fn advance(core: &mut StitchCore) {
///     core.advance_pose(RenderPose::default(), 0, 1.0 / 30.0);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorldPose {
    /// Horizontal angle in radians relative to the virtual camera's reference forward direction.
    pub yaw: f32,
    /// Elevation in radians above the world-reference horizontal plane.
    pub pitch: f32,
    /// Vertical field of view in degrees.
    pub fov_degrees: f32,
}

impl Default for WorldPose {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: 75.0,
        }
    }
}

/// A view expressed in the renderer's rig-adjusted yaw/pitch coordinates.
///
/// Direct viewport controls and render-at-pose methods use this type. It identifies a coordinate frame, not coverage validity: callers must constrain the view separately when source support is required.
///
/// World-reference requests cannot be passed directly to render-space APIs:
/// ```compile_fail
/// use reco_core::{core::StitchCore, geometry::WorldPose};
/// fn constrain(core: &mut StitchCore) {
///     core.constrain_render_pose(WorldPose::default(), WorldPose::default());
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RenderPose {
    /// Yaw in radians around the rig-adjusted yaw axis.
    pub yaw: f32,

    /// Pitch in radians around the renderer's pitch axis after yaw.
    pub pitch: f32,

    /// Vertical field of view in degrees (nalgebra `Perspective3`
    /// convention).
    ///
    /// Typical range: 30.0 (zoomed in) to 120.0 (wide). The single
    /// home for fov: it travels with yaw/pitch in every pose, and the
    /// render boundary clamps it to `[1, 179]` before building a
    /// projection matrix. There is no retained pipeline fov state.
    pub fov_degrees: f32,
}

/// An angular direction in the virtual camera's reference frame, without zoom.
/// Rig tilt and roll are not applied to these world-reference angles.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct YawPitch {
    /// Horizontal angle in radians.
    pub yaw: f32,
    /// Vertical angle in radians; positive looks up.
    pub pitch: f32,
}

impl RenderPose {
    /// The fov this pose renders with: clamped to `[1, 179]` degrees.
    ///
    /// The single home for the render-boundary guard - 0 or 180 would
    /// produce a NaN/Inf projection matrix (GPU) or a degenerate
    /// frustum tangent (CPU maps). Every place a pose becomes a
    /// matrix or ray basis goes through this.
    pub fn render_fov(&self) -> f32 {
        self.fov_degrees.clamp(1.0, 179.0)
    }
}

impl Default for RenderPose {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_coordinate_types_preserve_serialized_fields() {
        let value = serde_json::json!({"yaw": 0.25, "pitch": -0.5, "fov_degrees": 60.0});
        let world: WorldPose = serde_json::from_value(value.clone()).unwrap();
        let render: RenderPose = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(world).unwrap(), value);
        assert_eq!(serde_json::to_value(render).unwrap(), value);
    }
}
