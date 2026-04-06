//! 3D scene model: two camera planes and a virtual camera.
//!
//! Replicates the v1 Three.js geometric model in Rust. Two textured
//! planes are arranged in an L-shape, and a virtual camera at the
//! corner renders the stitched panoramic view.
//!
//! ## Coordinate System
//!
//! ```text
//!          Z (up in plane space)
//!          │
//!          │  ┌──────────┐
//!          │  │  Left     │  (X-Z plane, faces +X direction)
//!          │  │  Camera   │
//!          └──┼──────────┐│──── X
//!             │  Right   ││
//!             │  Camera  ││
//!             └──────────┘│
//!                         │
//!   Camera at [d, 0, d] where d = camera_axis_offset
//! ```

use crate::calibration::PlaneLayout;
use nalgebra::{Matrix4, Translation3, UnitQuaternion};

/// Computed 3D positions and rotations for the two camera planes.
///
/// Derived from a [`PlaneLayout`] by applying the intersection offset
/// and rotation corrections.
#[derive(Debug, Clone)]
pub struct SceneGeometry {
    /// Left plane position `[x, y, z]`.
    pub left_position: [f32; 3],
    /// Left plane rotation `[rx, ry, rz]` in radians.
    pub left_rotation: [f32; 3],
    /// Right plane position `[x, y, z]`.
    pub right_position: [f32; 3],
    /// Right plane rotation `[rx, ry, rz]` in radians.
    pub right_rotation: [f32; 3],
    /// Virtual camera position `[x, y, z]`.
    pub camera_position: [f32; 3],
    /// Plane width (normalized to 1.0).
    pub plane_width: f32,
    /// Plane aspect ratio (width / height), default 16:9.
    pub plane_aspect: f32,
}

impl SceneGeometry {
    /// Compute the 3D scene geometry from a plane layout.
    ///
    /// This mirrors the v1 `VideoPlane` component positioning:
    /// - Left plane: `position = [0, 0, (w/2) × (1 - intersect)]`,
    ///   `rotation = [zRx, π/2, 0]`
    /// - Right plane: `position = [(w/2) × (1 - intersect), xTy, 0]`,
    ///   `rotation = [0, 0, xRz]`
    pub fn from_layout(layout: &PlaneLayout) -> Self {
        let plane_width: f32 = 1.0;
        let aspect: f32 = crate::renderer::PLANE_ASPECT;
        let half_offset = (plane_width / 2.0) * (1.0 - layout.intersect as f32);

        Self {
            left_position: [0.0, 0.0, half_offset],
            left_rotation: [
                layout.z_rx as f32,
                std::f32::consts::FRAC_PI_2,
                layout.z_rz as f32,
            ],
            right_position: [half_offset, layout.x_ty as f32, 0.0],
            right_rotation: [layout.x_rx as f32, 0.0, layout.x_rz as f32],
            camera_position: [
                layout.camera_axis_offset as f32,
                0.0,
                layout.camera_axis_offset as f32,
            ],
            plane_width,
            plane_aspect: aspect,
        }
    }

    /// Model matrix for the left camera plane.
    ///
    /// The z-plane base rotation is π/2 around Y (faces sideways).
    /// `z_rx` is applied as a post-rotation around X so it acts as
    /// a roll around the plane's final normal. `z_rz` is applied
    /// as a pre-rotation (tilt correction).
    pub fn model_matrix_left(&self) -> Matrix4<f32> {
        let t = Translation3::new(
            self.left_position[0],
            self.left_position[1],
            self.left_position[2],
        );
        // Base: π/2 Y rotation + z_rz tilt
        let base = UnitQuaternion::from_euler_angles(
            0.0,
            self.left_rotation[1], // π/2
            self.left_rotation[2], // z_rz
        );
        // Post-rotate: z_rx as roll around X (the plane's final normal)
        let roll = UnitQuaternion::from_euler_angles(
            self.left_rotation[0], // z_rx
            0.0,
            0.0,
        );
        let r = roll * base;
        t.to_homogeneous() * r.to_homogeneous()
    }

    /// Model matrix for the right camera plane.
    pub fn model_matrix_right(&self) -> Matrix4<f32> {
        let t = Translation3::new(
            self.right_position[0],
            self.right_position[1],
            self.right_position[2],
        );
        let r = UnitQuaternion::from_euler_angles(
            self.right_rotation[0],
            self.right_rotation[1],
            self.right_rotation[2],
        );
        t.to_homogeneous() * r.to_homogeneous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::PlaneLayout;

    #[test]
    fn geometry_from_default_layout() {
        let layout = PlaneLayout {
            camera_axis_offset: 0.25,
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
        };

        let geom = SceneGeometry::from_layout(&layout);

        // Half offset = 0.5 * (1 - 0.5) = 0.25
        assert!((geom.left_position[2] - 0.25).abs() < 1e-5);
        assert!((geom.right_position[0] - 0.25).abs() < 1e-5);
        assert!((geom.camera_position[0] - 0.25).abs() < 1e-5);
        assert!((geom.camera_position[2] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn geometry_with_corrections() {
        let layout = PlaneLayout {
            camera_axis_offset: 0.24,
            intersect: 0.55,
            x_ty: 0.005,
            x_rz: 0.008,
            z_rx: -0.004,
            x_rx: 0.0,
            z_rz: 0.0,
        };

        let geom = SceneGeometry::from_layout(&layout);

        // Right plane should have the xTy correction
        assert!((geom.right_position[1] - 0.005).abs() < 1e-5);
        // Rotations should be applied
        assert!((geom.right_rotation[2] - 0.008).abs() < 1e-5);
        assert!((geom.left_rotation[0] - (-0.004)).abs() < 1e-5);
    }
}
