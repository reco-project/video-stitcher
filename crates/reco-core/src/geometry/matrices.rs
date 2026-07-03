//! Virtual-camera matrices and clip constants (wgpu-free).
//!
//! The rasterization-side half of the geometry leaf: the view matrix the
//! render pose feeds, the projection-space correction, and the clip
//! planes - shared verbatim by the GPU pipeline and the CPU inverse maps
//! so the two executors agree by construction.

use nalgebra::{Matrix4, Point3, UnitQuaternion};

/// Near clipping plane for the perspective projection.
pub const NEAR_PLANE: f32 = 0.01;
/// Far clipping plane for the perspective projection.
pub const FAR_PLANE: f32 = 5.0;

/// Build the view matrix for the virtual camera.
///
/// Camera sits at `position` and looks at the origin (corner where the two
/// planes meet) by default. This matches v1 Three.js where the OrbitControls
/// target is `[0, 0, 0]`. `yaw` rotates around Y (left/right from center),
/// `pitch` rotates around X (up/down).
pub fn view_matrix(
    position: &[f32; 3],
    yaw: f32,
    pitch: f32,
    rig_tilt: f32,
    rig_roll: f32,
) -> Matrix4<f32> {
    // One basis for the whole crate: the tilted+rolled reference frame
    // comes from the same rig_frame the pose inverse and the viewport-roll
    // margin computation use, so the three can never drift. The
    // world_to_render_pose round-trip test locks the pair together.
    let cam = super::virtual_camera::VirtualCamera::new(position);
    let eye = Point3::from(cam.eye);
    let (up_frame, rest_forward) = super::rig_correction::rig_frame(&cam, rig_tilt, rig_roll);

    // Yaw rotates around the (tilted+rolled) up axis; pitch around the
    // yaw-rotated right axis.
    let yaw_q = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(up_frame), yaw);
    let right = yaw_q * cam.base_right;
    let pitch_q = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(right), pitch);
    let rotation = pitch_q * yaw_q;
    let forward = rotation * rest_forward;
    let up = rotation * up_frame;
    let target = Point3::from(eye.coords + forward);
    nalgebra::Isometry3::look_at_rh(&eye, &target, &up).to_homogeneous()
}

/// Convert a nalgebra `Matrix4` to column-major `[[f32; 4]; 4]` for wgpu.
#[cfg(feature = "gpu")]
pub(crate) fn matrix4_to_columns(m: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let s = m.as_slice();
    [
        [s[0], s[1], s[2], s[3]],
        [s[4], s[5], s[6], s[7]],
        [s[8], s[9], s[10], s[11]],
        [s[12], s[13], s[14], s[15]],
    ]
}

/// OpenGL to wgpu clip space correction: Z from \[-1,1\] to \[0,1\].
///
/// nalgebra's `Perspective3` uses OpenGL conventions. wgpu expects
/// clip space Z in [0, 1], so we apply this correction.
#[rustfmt::skip]
pub(crate) fn opengl_to_wgpu_matrix() -> Matrix4<f32> {
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opengl_to_wgpu_maps_z() {
        let m = opengl_to_wgpu_matrix();
        // Point at Z = -1 (OpenGL near) should map to Z = 0 (wgpu near)
        let p = m * nalgebra::Vector4::new(0.0, 0.0, -1.0, 1.0);
        assert!((p.z - (-0.5 + 0.5)).abs() < 1e-5); // -0.5 + 0.5 = 0
        // Point at Z = 1 (OpenGL far) should map to Z = 1 (wgpu far)
        let p = m * nalgebra::Vector4::new(0.0, 0.0, 1.0, 1.0);
        assert!((p.z - 1.0).abs() < 1e-5);
    }

    #[test]
    fn view_matrix_self_consistent_with_direction_to_yaw_pitch() {
        // Step 1e (un-ignored by Step 2's VirtualCamera basis fix):
        // directions synthesized at a known (yaw, pitch), run through
        // direction_to_yaw_pitch, then fed to view_matrix, must
        // transform a point on the dir ray to the camera's -Z axis
        // (the right-hand convention nalgebra::Isometry3::look_at_rh
        // uses).
        //
        // rig_tilt and rig_roll are both zero here: direction_to_yaw_pitch
        // does not take them (Model 4), so any non-zero tilt/roll
        // would break the round-trip by definition. Step 4 lands
        // RigCorrection and unblocks the full (yaw, pitch, tilt, roll)
        // version of this test.
        let camera_position = [0.24_f32, 0.0, 0.24];
        let yaw_steps = [-1.0_f32, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0];
        let pitch_steps = [-0.6_f32, -0.2, 0.0, 0.2, 0.6];

        for &yaw in &yaw_steps {
            for &pitch in &pitch_steps {
                let dir = crate::projection::yaw_pitch_to_direction(yaw, pitch, &camera_position);
                let pos = crate::projection::direction_to_yaw_pitch(&dir, &camera_position);

                let view = view_matrix(&camera_position, pos.yaw, pos.pitch, 0.0, 0.0);

                // A point at eye + dir (unit step along the direction)
                // must land on camera-space -Z at distance 1.
                let target = nalgebra::Vector4::new(
                    camera_position[0] + dir.x,
                    camera_position[1] + dir.y,
                    camera_position[2] + dir.z,
                    1.0,
                );
                let cam = view * target;

                assert!(
                    cam.x.abs() < 1e-4,
                    "x should be zero (on camera forward axis), got {} at yaw={yaw} pitch={pitch}",
                    cam.x
                );
                assert!(
                    cam.y.abs() < 1e-4,
                    "y should be zero (on camera forward axis), got {} at yaw={yaw} pitch={pitch}",
                    cam.y
                );
                assert!(
                    (cam.z + 1.0).abs() < 1e-4,
                    "z should be -1 (camera looks down -Z), got {} at yaw={yaw} pitch={pitch}",
                    cam.z
                );
            }
        }
    }
}
