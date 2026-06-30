//! Rig tilt + roll correction.
//!
//! Maps a world-space look direction (the panorama's native coordinate
//! system) into the render-space pose the `view_matrix` consumes, so the
//! rendered horizon stays level under pan on a tilted/rolled rig.
//!
//! - `world_to_render_pose` (crate-internal): the orient leaf - exact
//!   quaternion inversion of the tilt+roll basis.
//! - `resolve_render_pose` (crate-internal): coverage-clamp + orient,
//!   the combined authority for the auto/AI path.
//!
//! Derivation in vault at
//! `architecture/rig-correction-v2-derivation-2026-04-23.md`.

use crate::projection::{CoverageBoundary, VirtualCamera};

/// Exact world-to-render mapping for the AI panner path.
///
/// Given a world-space (yaw, pitch) the panner wants to look at,
/// returns the (render_yaw, render_pitch) that makes `view_matrix`
/// point at that world direction. Uses quaternion inversion of the
/// tilt+roll basis rotation, so it handles the tilt+roll coupling at
/// non-zero yaw exactly (no closed-form approximation).
pub(crate) fn world_to_render_pose(
    cam: &VirtualCamera,
    world_yaw: f32,
    world_pitch: f32,
    rig_tilt: f32,
    rig_roll: f32,
) -> (f32, f32) {
    if rig_tilt.abs() < 1e-6 && rig_roll.abs() < 1e-6 {
        return (world_yaw, world_pitch);
    }

    // 1. Get the 3D direction the panner wants (in world space).
    let world_dir = cam.yaw_pitch_to_direction(world_yaw, world_pitch);

    // 2. Build the same tilt+roll quaternion that view_matrix applies.
    let base_right = cam.base_right;
    let mut base_forward = cam.base_forward;
    let mut world_up = VirtualCamera::world_up();

    let mut combined_q = nalgebra::UnitQuaternion::identity();
    if rig_tilt.abs() > 1e-6 {
        let tilt_q = nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(base_right),
            rig_tilt,
        );
        base_forward = tilt_q * base_forward;
        world_up = tilt_q * world_up;
        combined_q = tilt_q;
    }
    if rig_roll.abs() > 1e-6 {
        let roll_q = nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(base_forward),
            -rig_roll,
        );
        let _ = roll_q * world_up;
        combined_q = roll_q * combined_q;
    }

    // 3. Invert: the un-tilted direction that view_matrix's
    //    tilt rotation will map to world_dir.
    let render_dir = combined_q.inverse() * world_dir;

    // 4. Decompose using the un-tilted camera basis.
    let pos = cam.direction_to_yaw_pitch(&render_dir);
    (pos.yaw, pos.pitch)
}

/// Resolve a world-space target look-direction into the render-space
/// `(yaw, pitch)` the `view_matrix` consumes.
///
/// The auto/director and AI-panner paths (StitchCore + StitchSession)
/// route through this, so their coverage clamp and rig tilt+roll
/// correction can never drift into per-call copies. Interactive
/// consumers clamp ([`CoverageBoundary::safe_clamp`] via
/// `PoseControl`) and orient (`StitchRenderer::orient_pose`) as separate
/// steps, sharing the same [`world_to_render_pose`] leaf.
///
/// It bridges the two halves of the geometry seam:
/// 1. Clamp the world target to the coverage boundary. This stage is
///    *projection-coupled*: `CoverageBoundary` and its clamp encode a
///    bounded, non-wrapping panorama (today's L-shape) - a cylinder or
///    sphere would need its own. The target is already world-space and
///    coverage is panorama-native, so the clamp is pure world-space.
/// 2. Invert `view_matrix`'s tilt+roll basis via [`world_to_render_pose`]
///    so the horizon stays level under pan. This stage *is* projection
///    agnostic (pure virtual-camera orientation) and roll-aware (exact
///    quaternion inversion, no closed-form approximation).
///
/// `fov` and `aspect` size the clamp margins; capping `fov` against
/// `coverage.max_fov_degrees()` is the caller's policy, kept out of here.
///
/// (Steps 6-8 will dispatch stage 1 through the `Projection` trait so a
/// new projection brings its own clamp; stage 2 stays shared.)
pub(crate) fn resolve_render_pose(
    coverage: &CoverageBoundary,
    cam: &VirtualCamera,
    rig_tilt: f32,
    rig_roll: f32,
    world_yaw: f32,
    world_pitch: f32,
    fov: f32,
    aspect: f32,
) -> (f32, f32) {
    let clamped = coverage.safe_clamp(world_yaw, world_pitch, fov, aspect);
    world_to_render_pose(cam, clamped.yaw, clamped.pitch, rig_tilt, rig_roll)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> VirtualCamera {
        VirtualCamera::new(&[1.0, 0.0, 1.0])
    }

    #[test]
    fn world_to_render_identity_when_no_tilt_roll() {
        let cam = cam();
        for yaw_i in -8..=8 {
            let yaw = yaw_i as f32 * 0.1;
            for pitch_i in -4..=4 {
                let pitch = pitch_i as f32 * 0.05;
                let (ry, rp) = world_to_render_pose(&cam, yaw, pitch, 0.0, 0.0);
                assert!(
                    (ry - yaw).abs() < 1e-6 && (rp - pitch).abs() < 1e-6,
                    "identity failed at ({yaw}, {pitch}): got ({ry}, {rp})"
                );
            }
        }
    }

    #[test]
    fn world_to_render_compensates_tilt_and_roll_under_pan() {
        // On a tilted+rolled rig, looking at the world horizon (pitch=0)
        // must produce a non-zero render (yaw, pitch) so view_matrix
        // re-levels the frame; and the correction must vary with yaw
        // (this is exactly what the deleted closed-form render_pitch got
        // wrong for roll and off-axis yaw).
        let cam = cam();
        let (tilt, roll) = (0.2618_f32, 0.1222_f32); // ~15deg tilt, ~7deg roll
        let (ry0, rp0) = world_to_render_pose(&cam, 0.0, 0.0, tilt, roll);
        let (ry1, rp1) = world_to_render_pose(&cam, 0.6, 0.0, tilt, roll);
        assert!(
            rp0.abs() > 1e-3,
            "rest render pitch should be nonzero, got {rp0}"
        );
        assert!(
            (rp0 - rp1).abs() > 1e-4 || (ry1 - 0.6 - (ry0 - 0.0)).abs() > 1e-4,
            "correction must vary with yaw: ({ry0},{rp0}) vs ({ry1},{rp1})"
        );
    }

    #[test]
    fn world_to_render_exact_at_yaw_zero() {
        // At yaw=0 the rig axis points straight ahead, so a tilt T must
        // resolve to exactly render pitch -T (and unchanged yaw): the
        // view_matrix then tilts the frame back up by T to level it.
        let cam = cam();
        let tilt = 0.2_f32;
        let (ry, rp) = world_to_render_pose(&cam, 0.0, 0.0, tilt, 0.0);
        assert!(ry.abs() < 1e-5, "yaw should stay 0 at yaw=0, got {ry}");
        assert!(
            (rp + tilt).abs() < 1e-4,
            "render pitch should be -tilt={}, got {rp}",
            -tilt
        );
    }
}
