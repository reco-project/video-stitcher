//! Rig tilt + roll correction.
//!
//! Maps between the "human frame" (where pitch=0 means level horizon
//! regardless of camera tilt) and "world space" (the panorama's
//! native coordinate system).
//!
//! Two operations:
//! - [`render_pitch`]: produces the pitch that `view_matrix` needs so
//!   the rendered horizon stays level as yaw changes.
//! - [`human_to_world_pitch`] / [`world_to_human_pitch`]: bijective
//!   mapping for safe_clamp so coverage is checked in world space.
//!
//! Derivation in vault at
//! `architecture/rig-correction-v2-derivation-2026-04-23.md`.

/// Compute the pitch to pass to `view_matrix` so the horizon stays
/// level at the user's requested (yaw, user_pitch).
///
/// The view_matrix applies `rig_tilt` as a basis rotation. Without
/// compensation, the horizon tilts as you pan. This function
/// subtracts the yaw-dependent tilt offset to keep it level.
pub fn render_pitch(user_yaw: f32, user_pitch: f32, rig_tilt: f32) -> f32 {
    if rig_tilt.abs() < 1e-6 {
        return user_pitch;
    }
    user_pitch - (user_yaw.cos() * rig_tilt.tan()).atan()
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_pitch_identity_when_no_tilt() {
        for yaw_i in -10..=10 {
            let yaw = yaw_i as f32 * 0.1;
            for pitch_i in -5..=5 {
                let pitch = pitch_i as f32 * 0.05;
                let rp = render_pitch(yaw, pitch, 0.0);
                assert!(
                    (rp - pitch).abs() < 1e-6,
                    "identity failed at ({yaw}, {pitch}): got {rp}"
                );
            }
        }
    }

    #[test]
    fn render_pitch_compensates_tilt_at_yaw_zero() {
        let tilt = 0.15_f32;
        let rp = render_pitch(0.0, 0.0, tilt);
        assert!(
            (rp + tilt).abs() < 1e-4,
            "at yaw=0, render_pitch should be -tilt={}, got {rp}",
            -tilt
        );
    }

    #[test]
    fn render_pitch_no_compensation_at_yaw_90() {
        let tilt = 0.15_f32;
        let rp = render_pitch(std::f32::consts::FRAC_PI_2, 0.0, tilt);
        assert!(
            rp.abs() < 1e-4,
            "at yaw=π/2, render_pitch should be ~0, got {rp}"
        );
    }

    #[test]
    fn render_pitch_positive_at_yaw_pi() {
        let tilt = 0.15_f32;
        let rp = render_pitch(std::f32::consts::PI, 0.0, tilt);
        assert!(
            (rp - tilt).abs() < 1e-4,
            "at yaw=π, render_pitch should be +tilt={tilt}, got {rp}"
        );
    }

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
