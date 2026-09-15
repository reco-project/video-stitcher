//! Virtual camera basis for yaw/pitch decomposition.

use crate::geometry::Pose;

use nalgebra::{Point3, Vector3};

/// A virtual camera's position and horizontal reference orientation.
///
/// `base_right = base_forward x world_up` so it semantically points
/// to the viewer's right (intuitive). That makes the triple
/// left-handed, which on its own inverts yaw sign between a right-
/// hand rotation around `world_up` (what `view_matrix` does) and the
/// naive `atan2(h . base_right, h . base_forward)` decomposition.
/// The yaw API compensates by negating `h . base_right` in the
/// atan2, so `direction_to_yaw_pitch` and `view_matrix` agree on
/// yaw sign without any downstream reconciliation. `yaw_pitch_to_direction`
/// mirrors the same negation for symmetry.
///
/// Rig tilt and rig roll are NOT part of this type: `rig_frame` layers
/// them on top, and `view_matrix` + `world_to_render_pose` both compose
/// through that one construction.
#[derive(Debug, Clone, Copy)]
pub struct VirtualCamera {
    /// World-space viewpoint position, independent of reference orientation.
    pub(crate) eye: Point3<f32>,
    /// Unit reference forward vector in the horizontal XZ plane.
    pub(crate) base_forward: Vector3<f32>,
    /// Unit vector to the viewer's right (`base_forward x world_up`,
    /// left-handed triple). The yaw API compensates for the
    /// handedness so there is no sign divergence downstream.
    pub(crate) base_right: Vector3<f32>,
}

impl VirtualCamera {
    /// World up axis. Constant `+Y`; exposed as an associated method
    /// rather than a field because every VirtualCamera agrees on it.
    pub fn world_up() -> Vector3<f32> {
        Vector3::new(0.0, 1.0, 0.0)
    }

    /// Build the reference basis derived by a topology from a validated calibration.
    ///
    /// Position must be finite; forward must be horizontal with a finite, nonzero length in f32. Rig tilt and roll are applied separately.
    pub(crate) fn new(camera_position: &[f32; 3], forward: &[f32; 3]) -> Self {
        let eye = Point3::new(camera_position[0], camera_position[1], camera_position[2]);
        let base_forward = Vector3::new(forward[0], forward[1], forward[2]).normalize();
        let base_right = base_forward.cross(&Self::world_up()).normalize();
        Self {
            eye,
            base_forward,
            base_right,
        }
    }

    /// Decompose a world-space direction into yaw/pitch relative to
    /// the base forward axis.
    pub fn direction_to_yaw_pitch(&self, dir: &Vector3<f32>) -> Pose {
        // Pitch: elevation angle from the horizontal plane.
        let pitch = dir.y.clamp(-1.0, 1.0).asin();

        // Yaw: horizontal angle relative to base_forward. The minus
        // sign on (h . base_right) compensates for the left-handed
        // basis so yaw matches view_matrix's right-hand rotation
        // around world_up (see type doc comment).
        let horizontal = Vector3::new(dir.x, 0.0, dir.z);
        let h_len = horizontal.norm();
        let yaw = if h_len > 1e-6 {
            let h = horizontal / h_len;
            let cos_yaw = h.dot(&self.base_forward).clamp(-1.0, 1.0);
            let sin_yaw = -h.dot(&self.base_right);
            sin_yaw.atan2(cos_yaw)
        } else {
            0.0
        };

        Pose {
            yaw,
            pitch,
            // A direction has no zoom; the default fov stands in so
            // detection-mapping consumers (which read yaw/pitch only)
            // get a well-formed pose.
            ..Default::default()
        }
    }

    /// Exact inverse of [`direction_to_yaw_pitch`](Self::direction_to_yaw_pitch).
    /// `pitch` is expected in `(-pi/2, pi/2)`; at the poles yaw is
    /// undefined and the round-trip through `direction_to_yaw_pitch`
    /// collapses.
    pub fn yaw_pitch_to_direction(&self, yaw: f32, pitch: f32) -> Vector3<f32> {
        let cos_pitch = pitch.cos();
        // Matching sign compensation: `-sin(yaw) * base_right` pairs
        // with the `-h . base_right` in direction_to_yaw_pitch so the
        // round-trip is exact.
        let horizontal =
            self.base_forward * (cos_pitch * yaw.cos()) - self.base_right * (cos_pitch * yaw.sin());
        Vector3::new(horizontal.x, pitch.sin(), horizontal.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_does_not_determine_reference_orientation() {
        let origin = VirtualCamera::new(&[0.0, 0.0, 0.0], &[0.0, 0.0, -2.0]);
        let translated = VirtualCamera::new(&[2.0, 3.0, 4.0], &[0.0, 0.0, -2.0]);
        assert_eq!(origin.eye, Point3::origin());
        assert_eq!(translated.eye, Point3::new(2.0, 3.0, 4.0));
        assert_eq!(origin.base_forward, Vector3::new(0.0, 0.0, -1.0));
        assert_eq!(origin.base_right, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(origin.base_forward, translated.base_forward);
        assert_eq!(origin.base_right, translated.base_right);
    }

    #[test]
    fn explicit_l_shape_forward_preserves_origin_facing_basis() {
        for offset in [0.01, 0.2398, 0.5, 2.0] {
            let camera = VirtualCamera::new(&[offset, 0.0, offset], &[-offset, 0.0, -offset]);
            let forward = (Point3::origin() - camera.eye).normalize();
            assert_eq!(camera.base_forward, forward);
            assert_eq!(
                camera.base_right,
                forward.cross(&VirtualCamera::world_up()).normalize()
            );
        }
    }
}
