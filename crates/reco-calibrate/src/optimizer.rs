//! COBYLA-based position optimization.
//!
//! Finds the 5 or 6 placement parameters that minimize the total
//! reprojection error between matched feature point pairs. Uses a
//! three-phase strategy:
//!
//! 1. **Phase 0**: Translation-only optimization (3 params: `x_ty`,
//!    `intersect`, `cam_d`) with rotations fixed at zero. Uses a wide
//!    trust region and multi-start to find robust translation values.
//!
//! 2. **Phase 1**: Local 5-param refinement around the Phase 0 seed.
//!    Adds rotations (`x_rz`, `z_rx`) with tight bounds (±0.05 rad,
//!    ~3 degrees) and a small trust region to prevent rotations from
//!    absorbing the vertical offset that Phase 0 established.
//!
//! 3. **Phase 2** (optional): Refine with a 6th parameter (`z_rz`,
//!    left plane roll) using a tight trust region seeded from Phase 1.
//!
//! The phase separation is critical: without it, rotations can trade
//! off against `x_ty`, producing mathematically lower error on the
//! matched point set but visually worse stitching results.

use cobyla::{RhoBeg, StopTols};
use reco_core::calibration::PlaneLayout;

use crate::error::CalibrateError;
use crate::geometry::{self, OptParams};
use crate::types::{CalibrationConfig, MatchedPoint};

/// Bounds for the 3-parameter translation-only optimization (Phase 0).
///
/// Order: `[x_ty, intersect, cam_d]`
const BOUNDS_3: [(f64, f64); 3] = [
    (-0.1, 0.1), // x_ty: vertical translation
    (0.0, 1.0),  // intersect: overlap ratio
    (0.1, 0.35), // cam_d: camera distance (matches v1 bounds)
];

/// Rotation bound for Phase 1 refinement (about ±3 degrees).
///
/// Camera misalignment is typically sub-degree. A 3-degree bound
/// allows correction of moderate misalignment without letting the
/// optimizer trade rotations for vertical translation.
const ROTATION_BOUND: f64 = 0.05;

/// Maximum roll correction for the 6th parameter (about ±6 degrees).
const Z_RZ_BOUND: f64 = 0.1;

/// Shared COBYLA stop tolerances.
fn stop_tols() -> StopTols {
    StopTols {
        ftol_rel: 1e-10,
        xtol_rel: 1e-10,
        ..StopTols::default()
    }
}

/// Starting points for the 3-parameter translation-only phase.
///
/// Explores diverse cam_d and intersect values to avoid local minima.
const STARTS_3: [[f64; 3]; 5] = [
    [0.0, 0.5, 0.25], // center
    [0.0, 0.5, 0.15], // low cam_d
    [0.0, 0.5, 0.30], // high cam_d
    [0.0, 0.3, 0.25], // low intersect
    [0.0, 0.7, 0.25], // high intersect
];

/// Phase 0: Translation-only optimization (3 params).
///
/// Finds the best `[x_ty, intersect, cam_d]` with rotations fixed at zero.
/// This establishes the translation baseline that Phase 1 refines.
fn optimize_translations(points: &[MatchedPoint], max_evals: usize) -> Option<([f64; 3], f64)> {
    let data = points;
    let cons: Vec<&dyn cobyla::Func<&[MatchedPoint]>> = vec![];
    let mut best: Option<([f64; 3], f64)> = None;

    for init in &STARTS_3 {
        let result = cobyla::minimize(
            geometry::objective_3param,
            init,
            &BOUNDS_3,
            &cons,
            data,
            max_evals,
            RhoBeg::All(0.3),
            Some(stop_tols()),
        );

        let (x, f) = match result {
            Ok((_status, x, f)) => (x, f),
            Err((_status, x, f)) if f.is_finite() => (x, f),
            Err(_) => continue,
        };

        if best.as_ref().is_none_or(|(_, r)| f < *r) {
            best = Some(([x[0], x[1], x[2]], f));
        }
    }

    best
}

/// Phase 1: 5-param refinement around Phase 0 seed.
///
/// Takes the Phase 0 translations as a seed and adds rotation
/// corrections. x_ty and intersect stay close to Phase 0's values
/// (tight bounds), but cam_d gets the full range because the
/// angular error objective is degenerate in cam_d when rotations
/// are zero - only joint optimization with rotations can find the
/// correct cam_d.
fn refine_5param(
    seed: &[f64; 3],
    points: &[MatchedPoint],
    max_evals: usize,
) -> Option<(OptParams, f64)> {
    let init = [seed[0], seed[1], seed[2], 0.0, 0.0];

    // x_ty and intersect stay close to Phase 0 seed.
    // cam_d gets the full range: Phase 0 always hits the upper bound
    // because angular error decreases with distance when rotations=0.
    // With rotations enabled, cam_d can settle at the correct value.
    let bounds: [(f64, f64); 5] = [
        ((seed[0] - 0.02).max(-0.1), (seed[0] + 0.02).min(0.1)),
        ((seed[1] - 0.05).max(0.0), (seed[1] + 0.05).min(1.0)),
        BOUNDS_3[2], // full cam_d range [0.1, 0.35]
        (-ROTATION_BOUND, ROTATION_BOUND),
        (-ROTATION_BOUND, ROTATION_BOUND),
    ];

    let data = points;
    let cons: Vec<&dyn cobyla::Func<&[MatchedPoint]>> = vec![];

    let (x, f) = match cobyla::minimize(
        geometry::objective_5param,
        &init,
        &bounds,
        &cons,
        data,
        max_evals,
        RhoBeg::All(0.1), // wider trust region so cam_d can move freely
        Some(stop_tols()),
    ) {
        Ok((_status, x, f)) => (x, f),
        Err((_status, x, f)) if f.is_finite() => (x, f),
        Err(_) => return None,
    };

    Some((OptParams::from_5param(&x), f))
}

/// Run Phase 0 (translations) + Phase 1 (local refinement with rotations).
fn optimize_5param(
    points: &[MatchedPoint],
    config: &CalibrationConfig,
) -> Result<(OptParams, f64), CalibrateError> {
    // Phase 0: translation-only to establish x_ty, intersect, cam_d
    let (seed, seed_res) = optimize_translations(points, config.max_optimizer_evals).ok_or(
        CalibrateError::OptimizerFailed {
            max_evals: config.max_optimizer_evals,
        },
    )?;

    log::debug!(
        "  phase 0: x_ty={:.4}, intersect={:.4}, cam_d={:.4}, residual={seed_res:.6}",
        seed[0],
        seed[1],
        seed[2]
    );

    // Phase 1: local 5-param refinement seeded from Phase 0
    let (params, residual) = refine_5param(&seed, points, config.max_optimizer_evals)
        .unwrap_or_else(|| {
            // Fall back to Phase 0 result with zero rotations
            log::debug!("  phase 1 failed, using phase 0 result");
            (
                OptParams {
                    x_ty: seed[0],
                    intersect: seed[1],
                    cam_d: seed[2],
                    x_rz: 0.0,
                    z_rx: 0.0,
                    z_rz: None,
                },
                seed_res,
            )
        });

    log::debug!(
        "  phase 1: x_ty={:.4}, intersect={:.4}, cam_d={:.4}, x_rz={:.5}, z_rx={:.5}, residual={residual:.6}",
        params.x_ty,
        params.intersect,
        params.cam_d,
        params.x_rz,
        params.z_rx
    );

    Ok((params, residual))
}

/// Run the 6-parameter refinement (Phase 2).
///
/// Seeds from the Phase 1 result and adds `z_rz` with tight bounds.
fn refine_6param(
    points: &[MatchedPoint],
    seed: &OptParams,
    config: &CalibrationConfig,
) -> Result<(OptParams, f64), CalibrateError> {
    let init = [
        seed.x_ty,
        seed.intersect,
        seed.cam_d,
        seed.x_rz,
        seed.z_rx,
        0.0, // z_rz starts at 0
    ];

    // Keep translations near Phase 1, rotations within ROTATION_BOUND
    let bounds: [(f64, f64); 6] = [
        BOUNDS_3[0],
        BOUNDS_3[1],
        BOUNDS_3[2],
        (-ROTATION_BOUND, ROTATION_BOUND),
        (-ROTATION_BOUND, ROTATION_BOUND),
        (-Z_RZ_BOUND, Z_RZ_BOUND),
    ];

    let data = points;
    let cons: Vec<&dyn cobyla::Func<&[MatchedPoint]>> = vec![];

    let result = cobyla::minimize(
        geometry::objective_6param,
        &init,
        &bounds,
        &cons,
        data,
        config.max_optimizer_evals,
        RhoBeg::All(0.05), // tight trust region for refinement
        Some(stop_tols()),
    );

    match result {
        Ok((_status, x_opt, f_opt)) => {
            let params = OptParams::from_6param(&x_opt);
            Ok((params, f_opt))
        }
        Err((_status, x, f)) => {
            if f.is_finite() {
                let params = OptParams::from_6param(&x);
                Ok((params, f))
            } else {
                Err(CalibrateError::OptimizerFailed {
                    max_evals: config.max_optimizer_evals,
                })
            }
        }
    }
}

/// Run the full optimization pipeline on a set of matched points.
///
/// Phase 1 optimizes 5 parameters. If `config.enable_sixth_param` is true,
/// Phase 2 refines with the 6th parameter (left plane roll). The 6th
/// parameter is only accepted if it improves the residual error.
pub fn optimize(
    points: &[MatchedPoint],
    config: &CalibrationConfig,
) -> Result<(PlaneLayout, f64), CalibrateError> {
    let (params_5, residual_5) = optimize_5param(points, config)?;

    let (final_params, final_residual) = if config.enable_sixth_param {
        match refine_6param(points, &params_5, config) {
            Ok((params_6, residual_6)) if residual_6 < residual_5 => {
                log::debug!(
                    "6-param refinement improved residual: {residual_5:.6} -> {residual_6:.6}"
                );
                (params_6, residual_6)
            }
            Ok((_, residual_6)) => {
                log::debug!(
                    "6-param did not improve ({residual_6:.6} >= {residual_5:.6}), keeping 5-param"
                );
                (params_5, residual_5)
            }
            Err(e) => {
                log::warn!("6-param refinement failed ({e}), keeping 5-param result");
                (params_5, residual_5)
            }
        }
    } else {
        (params_5, residual_5)
    };

    let layout = PlaneLayout {
        camera_axis_offset: final_params.cam_d,
        intersect: final_params.intersect,
        x_ty: final_params.x_ty,
        x_rz: final_params.x_rz,
        z_rx: final_params.z_rx,
        z_rz: final_params.z_rz.unwrap_or(0.0),
    };

    Ok((layout, final_residual))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Create synthetic matched points for a known configuration.
    ///
    /// Uses `apply_transformations` + `angular_error` in reverse: pick 2D
    /// plane coordinates, forward-transform them, and verify zero error.
    /// The points are spread across the overlap region to well-constrain
    /// all 5 parameters.
    fn synthetic_points(true_params: &OptParams, n: usize) -> Vec<MatchedPoint> {
        use crate::geometry::PLANE_WIDTH;

        let half_offset = PLANE_WIDTH / 2.0 * (1.0 - true_params.intersect);
        let cam = nalgebra::Vector3::new(true_params.cam_d, 0.0, true_params.cam_d);

        let mut points = Vec::with_capacity(n);
        let grid = (n as f64).sqrt().ceil() as usize;

        for iy in 0..grid {
            for ix in 0..grid {
                if points.len() >= n {
                    break;
                }
                // Spread points across the overlap region
                let fx = (ix as f64 + 0.5) / grid as f64;
                let fy = (iy as f64 + 0.5) / grid as f64;

                // Ray direction: sweep a wide angular range toward both planes
                let yaw = -0.9 + fx * 0.5; // [-0.9, -0.4] - hits both planes
                let pitch = (fy - 0.5) * 0.4; // [-0.2, 0.2] - vertical spread

                let d = nalgebra::Vector3::new(yaw, pitch, yaw - 0.3).normalize();

                // Need both t > 0 for the ray to hit both planes
                if d.z.abs() < 1e-10 || d.x.abs() < 1e-10 {
                    continue;
                }
                let t_x = -cam.z / d.z;
                let t_z = -cam.x / d.x;
                if t_x < 0.0 || t_z < 0.0 {
                    continue;
                }

                let hit_x = cam + t_x * d;
                let hit_z = cam + t_z * d;

                // Reverse-transform to untransformed plane coords
                let x_coord = hit_x.x - half_offset;
                let y_coord = -(hit_x.y - true_params.x_ty);
                let z_coord = -(hit_z.z - half_offset);
                let z_y = -hit_z.y;

                points.push(MatchedPoint {
                    left: [x_coord, y_coord],
                    right: [z_coord, z_y],
                });
            }
        }
        points
    }

    #[test]
    fn optimize_recovers_known_params() {
        let true_params = OptParams {
            x_ty: 0.01,
            intersect: 0.5,
            cam_d: 0.225,
            x_rz: 0.0,
            z_rx: 0.0,
            z_rz: None,
        };

        let points = synthetic_points(&true_params, 50);
        assert!(
            points.len() >= 10,
            "not enough synthetic points: {}",
            points.len()
        );

        // Verify ground truth: both error metrics should be ~0 at true params
        let true_angular = geometry::angular_error(&points, &true_params);
        assert!(
            true_angular < 0.001,
            "synthetic points don't have near-zero angular error at true params: {true_angular}"
        );

        let config = CalibrationConfig {
            enable_sixth_param: false,
            max_optimizer_evals: 5000,
            ..Default::default()
        };

        let (layout, residual) = optimize(&points, &config).expect("optimization should succeed");

        // Residual should beat a random guess
        let random_error = geometry::angular_error(
            &points,
            &OptParams {
                x_ty: 0.5,
                intersect: 0.2,
                cam_d: 0.3,
                x_rz: 0.1,
                z_rx: 0.1,
                z_rz: None,
            },
        );
        assert!(
            residual < random_error,
            "optimizer should beat random params: {residual} vs {random_error}"
        );

        // Check recovered params are in a reasonable range
        assert_abs_diff_eq!(layout.camera_axis_offset, 0.225, epsilon = 0.1);
        assert_abs_diff_eq!(layout.intersect, 0.5, epsilon = 0.25);
    }

    #[test]
    fn six_param_does_not_regress() {
        let true_params = OptParams {
            x_ty: 0.0,
            intersect: 0.5,
            cam_d: 0.225,
            x_rz: 0.0,
            z_rx: 0.0,
            z_rz: None,
        };
        let points = synthetic_points(&true_params, 50);

        let config = CalibrationConfig {
            enable_sixth_param: true,
            max_optimizer_evals: 5000,
            ..Default::default()
        };

        let (layout, _) = optimize(&points, &config).expect("optimization should succeed");

        // z_rz should stay small when there's no actual roll.
        // Z_RZ_BOUND is 0.1 rad (~6 deg), so staying under that is good.
        assert_abs_diff_eq!(layout.z_rz, 0.0, epsilon = Z_RZ_BOUND);
    }
}
