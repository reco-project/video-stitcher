//! Virtual camera basis for yaw/pitch decomposition.

use crate::director::ViewportPosition;

use nalgebra::Vector3;

/// The virtual camera's orthonormal basis: single source of truth for
/// `(base_forward, base_right, world_up)` shared by `view_matrix` and
/// yaw/pitch decomposition.
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
/// Pre-Step-2 the two APIs used the literal `atan2(h . base_right,
/// h . base_forward)` and `cos(yaw)*bf + sin(yaw)*br` forms, so a
/// `yaw=+theta` on one side meant `yaw=-theta` on the other. The Step 1e
/// regression test locked that bug in; this type's yaw convention
/// un-ignores it.
///
/// Rig tilt and rig roll are NOT part of this type. `view_matrix`
/// layers them on top; Step 4 unifies them under `RigCorrection`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VirtualCamera {
    /// World-space camera eye position (copy of `camera_position`).
    pub eye: Vector3<f32>,
    /// Unit vector from eye toward the scene origin (the L-shape corner).
    pub base_forward: Vector3<f32>,
    /// Unit vector to the viewer's right (`base_forward x world_up`,
    /// left-handed triple). The yaw API compensates for the
    /// handedness so there is no sign divergence downstream.
    pub base_right: Vector3<f32>,
}

impl VirtualCamera {
    /// World up axis. Constant `+Y`; exposed as an associated method
    /// rather than a field because every VirtualCamera agrees on it.
    pub fn world_up() -> Vector3<f32> {
        Vector3::new(0.0, 1.0, 0.0)
    }

    /// Build the basis from a world-space camera position.
    pub fn new(camera_position: &[f32; 3]) -> Self {
        let eye = Vector3::new(camera_position[0], camera_position[1], camera_position[2]);
        let base_forward = (-eye).normalize();
        let base_right = base_forward.cross(&Self::world_up()).normalize();
        Self {
            eye,
            base_forward,
            base_right,
        }
    }

    /// Decompose a world-space direction into yaw/pitch relative to
    /// the base forward axis.
    pub fn direction_to_yaw_pitch(&self, dir: &Vector3<f32>) -> ViewportPosition {
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

        ViewportPosition {
            yaw,
            pitch,
            fov_degrees: None,
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
