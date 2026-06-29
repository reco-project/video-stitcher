//! Rig tilt + roll correction.
//!
//! Maps between the "human frame" (where pitch=0 means level horizon
//! regardless of camera tilt) and "world space" (the panorama's
//! native coordinate system).
//!
//! Operations:
//! - `world_to_render_pose` / `resolve_render_pose` (crate-internal):
//!   produce the render-space pose `view_matrix` needs (roll-aware) so
//!   the rendered horizon stays level as yaw changes. `world_to_render_pose`
//!   is the orient leaf (exact quaternion inversion of the tilt+roll
//!   basis); `resolve_render_pose` is the coverage-clamp + orient bridge.
//! - [`human_to_world_pitch`] / [`world_to_human_pitch`]: bijective
//!   pitch mapping for safe_clamp so coverage is checked in world space.
//!
//! Derivation in vault at
//! `architecture/rig-correction-v2-derivation-2026-04-23.md`.

use crate::projection::{CoverageBoundary, VirtualCamera};

/// Map a human-frame (yaw, pitch) to world-space pitch.
///
/// In the human frame, pitch=0 means level horizon. In world space,
/// pitch=0 is the un-tilted level. This function bridges the two.
///
/// `D = sqrt(cos²(yaw)*sin²(tilt) + cos²(tilt))` is the coupling
/// factor: at yaw=0, D=1 (full tilt effect); at yaw=π/2, D=cos(tilt)
/// (tilt decoupled from pitch).
pub fn human_to_world_pitch(yaw: f32, human_pitch: f32, rig_tilt: f32) -> f32 {
    if rig_tilt.abs() < 1e-6 {
        return human_pitch;
    }
    let d = coupling_factor(yaw, rig_tilt);
    (human_pitch.sin() * d).asin()
}

/// Inverse of [`human_to_world_pitch`].
pub fn world_to_human_pitch(yaw: f32, world_pitch: f32, rig_tilt: f32) -> f32 {
    if rig_tilt.abs() < 1e-6 {
        return world_pitch;
    }
    let d = coupling_factor(yaw, rig_tilt);
    if d.abs() < 1e-6 {
        return world_pitch;
    }
    (world_pitch.sin() / d).clamp(-1.0, 1.0).asin()
}

fn coupling_factor(yaw: f32, rig_tilt: f32) -> f32 {
    let cos_yaw = yaw.cos();
    let sin_t = rig_tilt.sin();
    let cos_t = rig_tilt.cos();
    (cos_yaw * cos_yaw * sin_t * sin_t + cos_t * cos_t).sqrt()
}

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
/// This is the single pose-resolution authority. Every render site
/// (auto/director, AI panner, and - once converged - interactive
/// consumers) routes through it, so the coverage clamp and the rig
/// tilt+roll correction can never drift apart into per-call copies.
///
/// It bridges the two halves of the geometry seam:
/// 1. Clamp the world target to the coverage boundary. This stage is
///    *projection-coupled*: `CoverageBoundary` and its clamp encode a
///    bounded, non-wrapping panorama (today's L-shape) - a cylinder or
///    sphere would need its own. `rig_tilt = 0` on purpose: the target
///    is already world-space and coverage is panorama-native, so no
///    human<->world mapping is wanted here.
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
    let clamped = coverage.safe_clamp(world_yaw, world_pitch, fov, aspect, 0.0);
    world_to_render_pose(cam, clamped.yaw, clamped.pitch, rig_tilt, rig_roll)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_to_world_identity_when_no_tilt() {
        for yaw_i in -10..=10 {
            let yaw = yaw_i as f32 * 0.1;
            for pitch_i in -5..=5 {
                let pitch = pitch_i as f32 * 0.05;
                let wp = human_to_world_pitch(yaw, pitch, 0.0);
                assert!(
                    (wp - pitch).abs() < 1e-6,
                    "identity failed at ({yaw}, {pitch}): got {wp}"
                );
            }
        }
    }

    #[test]
    fn human_zero_pitch_maps_to_world_zero() {
        let tilt = 0.2_f32;
        for yaw_i in -10..=10 {
            let yaw = yaw_i as f32 * 0.3;
            let wp = human_to_world_pitch(yaw, 0.0, tilt);
            assert!(
                wp.abs() < 1e-6,
                "human_pitch=0 should map to world_pitch=0 at yaw={yaw}, got {wp}"
            );
        }
    }

    #[test]
    fn world_to_human_roundtrip() {
        let tilt = 0.15_f32;
        for yaw_i in -5..=5 {
            let yaw = yaw_i as f32 * 0.3;
            for pitch_i in -3..=3 {
                let pitch = pitch_i as f32 * 0.1;
                let wp = human_to_world_pitch(yaw, pitch, tilt);
                let back = world_to_human_pitch(yaw, wp, tilt);
                assert!(
                    (back - pitch).abs() < 1e-4,
                    "roundtrip failed at yaw={yaw}, pitch={pitch}: world={wp}, back={back}"
                );
            }
        }
    }
}
