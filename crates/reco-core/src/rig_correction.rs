//! Rig tilt + roll correction — one type, one formula, three use sites.
//!
//! The camera rig is a physical object mounted at the edge of a pitch.
//! When it is tilted up/down (`tilt`) or leans to one side (`roll`),
//! the stitched panorama inherits the same skew. `RigCorrection`
//! encapsulates the math that lets every layer of the pipeline agree
//! on how to account for that skew:
//!
//! - [`RigCorrection::user_to_world_pose`] / [`world_to_user_pose`] —
//!   bijective mapping between the user's (yaw, pitch) controls (where
//!   the horizon stays level across pans) and the world-space
//!   (yaw, pitch) the coverage boundary and panorama geometry are in.
//! - [`RigCorrection::apply_to_view_matrix_basis`] — rotates the
//!   virtual camera basis by tilt+roll the way `view_matrix` needs
//!   right before yaw/pitch rotations happen.
//!
//! The pose bijection uses the yaw-dependent closed form
//! `world_pitch = user_pitch + tilt * (1 - cos(yaw))`, empirically
//! validated on rig_tilt=15° XTU footage. Pre-Step-4 this same formula
//! lived in three places under three different names (`safe_clamp`'s
//! constant `+ rig_tilt` offset, `pose_control::render_pose`'s
//! closed form, and nothing at all in `direction_to_yaw_pitch`) —
//! none of them agreed. The one source of truth now lives here.

use crate::projection::VirtualCamera;
use nalgebra::{UnitQuaternion, Vector3};

/// Rig mount correction: tilt around the camera's right axis, roll
/// around the forward axis. Both in radians.
///
/// `Default` is zero tilt and zero roll (no correction).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RigCorrection {
    /// Tilt angle around the rig's right axis (radians). Positive
    /// tilts the rig upward.
    pub tilt: f32,
    /// Roll angle around the rig's forward axis (radians). Positive
    /// rolls the rig clockwise as seen from behind.
    pub roll: f32,
}

impl RigCorrection {
    /// Build a correction from explicit tilt + roll.
    pub fn new(tilt: f32, roll: f32) -> Self {
        Self { tilt, roll }
    }

    /// `true` when both components are near zero — callers can skip
    /// the whole correction chain in the fast path.
    pub fn is_identity(&self) -> bool {
        self.tilt.abs() <= 1e-6 && self.roll.abs() <= 1e-6
    }

    /// Map a user-space `(yaw, pitch)` pair to world-space.
    ///
    /// The user's `pitch=0` keeps the horizon level across pans. Due
    /// to the way a tilted rig folds pitch into roll during pan
    /// (around the tilted up axis in `view_matrix`'s quaternion
    /// model), the world-space pitch that renders a level horizon is
    /// `user_pitch + tilt * (1 - cos(yaw))`. Yaw is unchanged.
    ///
    /// Roll does not enter the pose bijection — it only rotates the
    /// output image around the view axis.
    pub fn user_to_world_pose(&self, yaw: f32, pitch: f32) -> (f32, f32) {
        (yaw, pitch + self.tilt * (1.0 - yaw.cos()))
    }

    /// Inverse of [`user_to_world_pose`]. Yaw is unchanged between
    /// spaces, so the inverse is `-tilt * (1 - cos(yaw))` on pitch.
    pub fn world_to_user_pose(&self, yaw: f32, pitch: f32) -> (f32, f32) {
        (yaw, pitch - self.tilt * (1.0 - yaw.cos()))
    }

    /// Apply tilt + roll to the virtual camera basis, producing the
    /// basis `view_matrix` uses for its yaw/pitch rotations.
    ///
    /// Returns `(base_forward, base_right, world_up)` after tilt
    /// (around `base_right`) and roll (around `base_forward`) have
    /// been applied. `base_right` is invariant under both operations
    /// at the basis-setup phase (tilt rotates around it, roll does
    /// not touch it), so it is returned unchanged.
    ///
    /// This is Model 1 from the audit — geometrically exact.
    ///
    /// `pub(crate)` because `VirtualCamera` is crate-internal; callers
    /// outside reco-core use the pose bijection and let `view_matrix`
    /// handle the basis rotation.
    pub(crate) fn apply_to_view_matrix_basis(
        &self,
        cam: &VirtualCamera,
    ) -> (Vector3<f32>, Vector3<f32>, Vector3<f32>) {
        let mut base_forward = cam.base_forward;
        let mut world_up = VirtualCamera::world_up();

        if self.tilt.abs() > 1e-6 {
            let tilt_q = UnitQuaternion::from_axis_angle(
                &nalgebra::Unit::new_normalize(cam.base_right),
                self.tilt,
            );
            base_forward = tilt_q * base_forward;
            world_up = tilt_q * world_up;
        }

        if self.roll.abs() > 1e-6 {
            // Negated: roll describes the camera's lean direction, so
            // the output must rotate the opposite way to straighten
            // the horizon. Tilt is not negated because it shifts the
            // view center to match where the camera points.
            let roll_q = UnitQuaternion::from_axis_angle(
                &nalgebra::Unit::new_normalize(base_forward),
                -self.roll,
            );
            world_up = roll_q * world_up;
        }

        (base_forward, cam.base_right, world_up)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_correction_is_identity_on_pose() {
        let rig = RigCorrection::default();
        assert!(rig.is_identity());
        let (y, p) = rig.user_to_world_pose(0.5, 0.3);
        assert_eq!(y, 0.5);
        assert_eq!(p, 0.3);
    }

    #[test]
    fn user_to_world_and_back_is_exact() {
        // Bijection check on a yaw x pitch grid at a realistic tilt.
        let rig = RigCorrection::new(15.0_f32.to_radians(), 0.0);
        for yaw_deg in (-180..=180).step_by(20) {
            for pitch_deg in (-30..=30).step_by(10) {
                let yaw = (yaw_deg as f32).to_radians();
                let pitch = (pitch_deg as f32).to_radians();
                let (wy, wp) = rig.user_to_world_pose(yaw, pitch);
                let (uy, up) = rig.world_to_user_pose(wy, wp);
                assert!(
                    (uy - yaw).abs() < 1e-6,
                    "yaw mismatch at ({yaw_deg}, {pitch_deg}): sent {yaw}, got {uy}"
                );
                assert!(
                    (up - pitch).abs() < 1e-6,
                    "pitch mismatch at ({yaw_deg}, {pitch_deg}): sent {pitch}, got {up}"
                );
            }
        }
    }

    #[test]
    fn world_pitch_shifts_by_twice_tilt_at_yaw_pi() {
        // The Model 3 formula's defining feature: at yaw=0 the
        // correction is zero (1 - cos(0) = 0), at yaw=pi the
        // correction is 2*tilt (1 - cos(pi) = 2). This guards the
        // formula against accidental edits.
        let rig = RigCorrection::new(0.2, 0.0);
        let (_, p0) = rig.user_to_world_pose(0.0, 0.1);
        assert!((p0 - 0.1).abs() < 1e-6);
        let (_, pp) = rig.user_to_world_pose(std::f32::consts::PI, 0.1);
        assert!((pp - (0.1 + 0.4)).abs() < 1e-6);
    }

    #[test]
    fn identity_basis_matches_virtual_camera_basis() {
        let rig = RigCorrection::default();
        let cam = VirtualCamera::new(&[0.24, 0.0, 0.24]);
        let (bf, br, wu) = rig.apply_to_view_matrix_basis(&cam);
        assert!((bf - cam.base_forward).norm() < 1e-6);
        assert!((br - cam.base_right).norm() < 1e-6);
        assert!((wu - VirtualCamera::world_up()).norm() < 1e-6);
    }

    #[test]
    fn tilt_rotates_base_forward_toward_up() {
        // Positive tilt rotates base_forward around base_right by the
        // tilt angle. For the default diagonal camera, that moves
        // base_forward upward (acquires a +Y component).
        let rig = RigCorrection::new(0.25, 0.0);
        let cam = VirtualCamera::new(&[0.24, 0.0, 0.24]);
        let (bf, _, _) = rig.apply_to_view_matrix_basis(&cam);
        assert!(
            bf.y > 0.0,
            "positive tilt must rotate base_forward upward, got y={}",
            bf.y
        );
    }
}
