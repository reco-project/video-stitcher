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
//!   Camera at [d, 0, d] where d = framing.axis_offset
//! ```

use crate::calibration::{Framing, Topology};
use nalgebra::{Matrix4, Translation3, UnitQuaternion};

/// Computed 3D positions and rotations for the two camera planes.
///
/// Derived from a [`Topology`] by applying the intersection offset
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
    /// Derive the 3D scene geometry from the calibration's topology + framing.
    ///
    /// `aspect` is the source frame `width / height`. Mirrors the v1 plane
    /// positioning:
    /// - Left plane: `position = [0, 0, (w/2)(1 - intersect)]`, `rotation = [z_rx, π/2, z_rz]`
    /// - Right plane: `position = [(w/2)(1 - intersect), x_ty, 0]`, `rotation = [x_rx, 0, x_rz]`
    /// - Virtual camera at `[axis_offset, 0, axis_offset]`.
    pub fn new(topology: &Topology, framing: &Framing, aspect: f32) -> Self {
        let plane_width: f32 = 1.0;
        let half_offset = (plane_width / 2.0) * (1.0 - topology.intersect as f32);
        let axis = framing.axis_offset as f32;

        Self {
            left_position: [0.0, 0.0, half_offset],
            left_rotation: [
                topology.z_rx as f32,
                std::f32::consts::FRAC_PI_2,
                topology.z_rz as f32,
            ],
            right_position: [half_offset, topology.x_ty as f32, 0.0],
            right_rotation: [topology.x_rx as f32, 0.0, topology.x_rz as f32],
            camera_position: [axis, 0.0, axis],
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
    use crate::calibration::{Framing, Topology};

    fn topo(intersect: f64) -> Topology {
        Topology {
            intersect,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        }
    }

    fn framing(axis_offset: f64) -> Framing {
        Framing {
            axis_offset,
            tilt: 0.0,
            roll: 0.0,
        }
    }

    #[test]
    fn geometry_from_default_layout() {
        let geom = SceneGeometry::new(&topo(0.5), &framing(0.25), 16.0 / 9.0);

        // Half offset = 0.5 * (1 - 0.5) = 0.25
        assert!((geom.left_position[2] - 0.25).abs() < 1e-5);
        assert!((geom.right_position[0] - 0.25).abs() < 1e-5);
        assert!((geom.camera_position[0] - 0.25).abs() < 1e-5);
        assert!((geom.camera_position[2] - 0.25).abs() < 1e-5);
        assert!((geom.plane_aspect - 16.0 / 9.0).abs() < 1e-5);
    }

    #[test]
    fn geometry_with_corrections() {
        let topology = Topology {
            intersect: 0.55,
            x_ty: 0.005,
            x_rz: 0.008,
            z_rx: -0.004,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        };

        let geom = SceneGeometry::new(&topology, &framing(0.24), 16.0 / 9.0);

        // Right plane should have the x_ty correction
        assert!((geom.right_position[1] - 0.005).abs() < 1e-5);
        // Rotations should be applied
        assert!((geom.right_rotation[2] - 0.008).abs() < 1e-5);
        assert!((geom.left_rotation[0] - (-0.004)).abs() < 1e-5);
    }

    #[test]
    fn geometry_with_custom_aspect() {
        let aspect_4_3 = 4.0 / 3.0;
        let geom = SceneGeometry::new(&topo(0.5), &framing(0.25), aspect_4_3);
        assert!((geom.plane_aspect - aspect_4_3).abs() < 1e-5);
    }
}
