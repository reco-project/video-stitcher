//! Rig tilt + roll correction.
//!
//! Maps a world-space look direction (the panorama's native coordinate
//! system) into the render-space pose the `view_matrix` consumes, so the
//! rendered horizon stays level under pan on a tilted/rolled rig.
//!
//! - `world_to_render_pose` (crate-internal): the orient leaf - exact
//!   inverse of `view_matrix`'s yaw/pitch composition in the tilted+rolled
//!   reference frame.
//! - `resolve_render_pose` (crate-internal): coverage-clamp + orient,
//!   the combined authority for the auto/AI path.
//!
//! Derivation in vault at
//! `architecture/rig-correction-v2-derivation-2026-04-23.md`.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use nalgebra::{Unit, UnitQuaternion, Vector3};

use crate::projection::CoverageBoundary;

use super::virtual_camera::VirtualCamera;

/// The tilted+rolled reference frame `view_matrix` composes before applying
/// yaw/pitch: the yaw axis `u` (world up after tilt, then roll) and the rest
/// forward `f0` (base forward after tilt; roll rotates *around* it, so roll
/// leaves it unchanged). Must mirror `view_matrix`'s construction exactly -
/// the round-trip test in this module is the guard.
pub(crate) fn rig_frame(
    cam: &VirtualCamera,
    rig_tilt: f32,
    rig_roll: f32,
) -> (Vector3<f32>, Vector3<f32>) {
    let mut f0 = cam.base_forward;
    let mut u = VirtualCamera::world_up();
    if rig_tilt.abs() > 1e-6 {
        let tilt_q =
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(cam.base_right), rig_tilt);
        f0 = tilt_q * f0;
        u = tilt_q * u;
    }
    if rig_roll.abs() > 1e-6 {
        let roll_q = UnitQuaternion::from_axis_angle(&Unit::new_normalize(f0), -rig_roll);
        u = roll_q * u;
    }
    (u, f0)
}

/// Exact world-to-render mapping for every render site.
///
/// Given a world-space (yaw, pitch) to look at, returns the
/// (render_yaw, render_pitch) that makes `view_matrix` point at that
/// world direction. `view_matrix` composes yaw around the tilted+rolled
/// up axis and pitch around the yaw-rotated (unrolled) right axis, so the
/// two axes do not form the frame a single quaternion conjugation can
/// invert; instead this solves the composition directly:
///
/// 1. The look direction always stays perpendicular to the pitch axis
///    `R(u, yaw) * base_right`, which pins yaw to
///    `A cos(yaw) + B sin(yaw) + C = 0` (Rodrigues expansion) - solved in
///    closed form, keeping the root that faces the target.
/// 2. Pitch is then the signed angle from the yawed rest-forward to the
///    target around that pitch axis.
///
/// Exact for any tilt+roll (the round-trip test against `view_matrix`
/// is the guard); the previous quaternion-conjugation inverse was exact
/// for tilt but drifted up to ~4 deg under combined tilt+roll.
pub fn world_to_render_pose(
    cam: &VirtualCamera,
    world_yaw: f32,
    world_pitch: f32,
    rig_tilt: f32,
    rig_roll: f32,
) -> (f32, f32) {
    if rig_tilt.abs() < 1e-6 && rig_roll.abs() < 1e-6 {
        return (world_yaw, world_pitch);
    }

    let d = cam.yaw_pitch_to_direction(world_yaw, world_pitch);
    let (u, f0) = rig_frame(cam, rig_tilt, rig_roll);
    let br = cam.base_right;

    // d . (R(u, yaw) * br) = 0, Rodrigues-expanded:
    let a = d.dot(&br) - u.dot(&br) * d.dot(&u);
    let b = d.dot(&u.cross(&br));
    let c = u.dot(&br) * d.dot(&u);
    let r = (a * a + b * b).sqrt();
    if r < 1e-9 {
        // Target (anti)parallel to the yaw axis: yaw is ill-defined (a
        // clamped pose never reaches the pole). Keep the requested yaw
        // and aim the pitch at the pole.
        let pole = if d.dot(&u) >= 0.0 {
            FRAC_PI_2
        } else {
            -FRAC_PI_2
        };
        return (world_yaw, pole);
    }

    // A cos(yaw) + B sin(yaw) = -C  =>  yaw = atan2(B, A) +- acos(-C / r).
    let phi = b.atan2(a);
    let delta = (-c / r).clamp(-1.0, 1.0).acos();

    // Two roots: camera facing toward the target, or away from it with
    // pitch flipped past the pole. Keep the toward-facing one.
    let mut best = (f32::NEG_INFINITY, 0.0_f32, 0.0_f32);
    for yaw in [phi + delta, phi - delta] {
        let yaw_q = UnitQuaternion::from_axis_angle(&Unit::new_normalize(u), yaw);
        let g = yaw_q * f0;
        let along = g.dot(&d);
        if along > best.0 {
            let p_axis = yaw_q * br;
            let pitch = g.cross(&d).dot(&p_axis).atan2(along);
            best = (along, yaw, pitch);
        }
    }
    let (_, yaw, pitch) = best;
    // Principal yaw for downstream clamps.
    let yaw = (yaw + PI).rem_euclid(TAU) - PI;
    (yaw, pitch)
}

/// Signed roll of the rendered viewport relative to the panorama's
/// upright frame at a world-space pose: the angle between `view_matrix`'s
/// rendered up vector and the zero-roll up for that look direction.
///
/// Zero on a level rig (panning never rolls the frame); grows with pan
/// on a tilted/rolled rig - the intentional "natural roll" `view_matrix`
/// documents. [`CoverageBoundary::safe_clamp`] uses it to size its
/// margins against the *rotated* viewport rectangle instead of an
/// axis-aligned one, which under tilt leaked black corners past the
/// clamp.
pub(crate) fn render_viewport_roll(
    cam: &VirtualCamera,
    world_yaw: f32,
    world_pitch: f32,
    rig_tilt: f32,
    rig_roll: f32,
) -> f32 {
    if rig_tilt.abs() < 1e-6 && rig_roll.abs() < 1e-6 {
        return 0.0;
    }
    let (ry, rp) = world_to_render_pose(cam, world_yaw, world_pitch, rig_tilt, rig_roll);
    let (u, f0) = rig_frame(cam, rig_tilt, rig_roll);
    // Rebuild view_matrix's rotation at the render pose.
    let yaw_q = UnitQuaternion::from_axis_angle(&Unit::new_normalize(u), ry);
    let pitch_q = UnitQuaternion::from_axis_angle(&Unit::new_normalize(yaw_q * cam.base_right), rp);
    let rotation = pitch_q * yaw_q;
    let fwd = (rotation * f0).normalize();
    // Both ups orthogonalized against the look direction (look_at_rh does
    // the same internally), then the signed angle between them around it.
    let perp = |v: Vector3<f32>| v - fwd * v.dot(&fwd);
    let up_rendered = perp(rotation * u);
    let up_ref = perp(VirtualCamera::world_up());
    if up_rendered.norm() < 1e-6 || up_ref.norm() < 1e-6 {
        return 0.0; // looking along an up axis: roll reference undefined
    }
    up_ref
        .cross(&up_rendered)
        .dot(&fwd)
        .atan2(up_ref.dot(&up_rendered))
}

/// Resolve a world-space target look-direction into the render-space
/// `(yaw, pitch)` the `view_matrix` consumes.
///
/// The auto/director and AI-panner paths (StitchCore + StitchSession)
/// route through this, so their coverage clamp and rig tilt+roll
/// correction can never drift into per-call copies. Interactive
/// consumers clamp ([`CoverageBoundary::safe_clamp`] via
/// `PoseControl`) and orient (`StitchCore::orient_pose`) as separate
/// steps, sharing the same [`world_to_render_pose`] leaf.
///
/// It bridges the two halves of the geometry seam:
/// 1. Clamp the world target to the coverage boundary. This stage is
///    *projection-coupled*: `CoverageBoundary` and its clamp encode a
///    bounded, non-wrapping panorama (today's L-shape) - a cylinder or
///    sphere would need its own. The target is already world-space and
///    coverage is panorama-native, so the clamp is pure world-space.
/// 2. Invert `view_matrix`'s tilt+roll composition via
///    [`world_to_render_pose`] so the horizon stays level under pan. This
///    stage *is* projection agnostic (pure virtual-camera orientation)
///    and roll-aware (exact solve, guarded by the round-trip test).
///
/// `fov` and `aspect` size the clamp margins; capping `fov` against
/// `coverage.max_fov_degrees()` is the caller's policy, kept out of here.
///
/// (Steps 6-8 will dispatch stage 1 through the `Projection` trait so a
/// new projection brings its own clamp; stage 2 stays shared.)
pub fn resolve_render_pose(
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

    /// The round-trip guard for the orient leaf: feeding the resolved
    /// (render_yaw, render_pitch) to `view_matrix` must aim the camera
    /// exactly at the requested world direction, for any tilt+roll.
    ///
    /// This is the only oracle pose resolution can have: the CPU/GPU
    /// agreement suite feeds both executors the SAME resolved pose, so
    /// it can never see a resolution error (both agree while both point
    /// the wrong way).
    #[test]
    fn world_to_render_inverts_view_matrix_under_tilt_and_roll() {
        use crate::geometry::matrices::view_matrix;
        let position = [1.0_f32, 0.0, 1.0];
        let cam = VirtualCamera::new(&position);
        for &(tilt, roll) in &[
            (0.0_f32, 0.0_f32),
            (0.15, 0.0),
            (0.33, 0.0),
            (0.0, 0.12),
            (0.33, 0.12),
            (0.26, -0.1),
        ] {
            for yaw_i in -4..=4i32 {
                let wy = yaw_i as f32 * 0.22;
                for pitch_i in -3..=3i32 {
                    let wp = pitch_i as f32 * 0.11;
                    let (ry, rp) = world_to_render_pose(&cam, wy, wp, tilt, roll);
                    let view = view_matrix(&position, ry, rp, tilt, roll);
                    let dir = cam.yaw_pitch_to_direction(wy, wp);
                    let target = nalgebra::Vector4::new(
                        position[0] + dir.x,
                        position[1] + dir.y,
                        position[2] + dir.z,
                        1.0,
                    );
                    let c = view * target;
                    // Angular error between camera -Z and the target.
                    let err = (c.x * c.x + c.y * c.y).sqrt().atan2(-c.z);
                    assert!(
                        err.abs() < 5e-4,
                        "pointing error {err} rad at tilt={tilt} roll={roll} yaw={wy} pitch={wp}"
                    );
                }
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
