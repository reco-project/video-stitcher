//! COBYLA-based position optimization.
//!
//! Finds the 5 or 6 placement parameters that minimize the symmetric
//! plane-to-plane reprojection error. Uses multi-start COBYLA to avoid
//! local minima.
//!
//! The reprojection error objective has a proper global minimum in all
//! parameters (including cam_d), unlike the angular error used in v1
//! which is degenerate in camera distance.

use cobyla::{RhoBeg, StopTols};
use reco_core::calibration::PlaneLayout;

use crate::error::CalibrateError;
use crate::geometry::{self, OptParams};
use crate::types::{CalibrationConfig, MatchedPoint};

/// Bounds for the 5-parameter optimization.
///
/// Order: `[x_ty, intersect, cam_d, x_rz, z_rx]`
const BOUNDS_5: [(f64, f64); 5] = [
    (-0.1, 0.1),   // x_ty: vertical translation
    (0.0, 1.0),    // intersect: overlap ratio
    (0.1, 0.35),   // cam_d: camera distance
    (-0.05, 0.05), // x_rz: ~±3 degrees
    (-0.05, 0.05), // z_rx: ~±3 degrees
];

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

/// Starting points for multi-start optimization.
///
/// Explores diverse cam_d and intersect values. Rotations always
/// start at zero since the reprojection error landscape is well-behaved
/// and COBYLA can find the rotation corrections from any cam_d start.
const STARTS: [[f64; 5]; 6] = [
    [0.0, 0.5, 0.225, 0.0, 0.0], // center (matches v1 initial guess)
    [0.0, 0.5, 0.15, 0.0, 0.0],  // low cam_d
    [0.0, 0.5, 0.30, 0.0, 0.0],  // high cam_d
    [0.0, 0.3, 0.225, 0.0, 0.0], // low intersect
    [0.0, 0.7, 0.225, 0.0, 0.0], // high intersect
    [0.0, 0.5, 0.20, 0.0, 0.0],  // alternative cam_d
];

/// Optimize 5 parameters using multi-start Powell coordinate descent.
///
/// Powell's method with conjugate direction updates naturally finds
/// local minima in the reprojection error landscape. Multiple starts
/// ensure we explore different basins.
fn optimize_5param(
    points: &[MatchedPoint],
    _config: &CalibrationConfig,
) -> Result<(OptParams, f64), CalibrateError> {
    let mut best: Option<(OptParams, f64)> = None;

    let eval = |p: &[f64; 5]| -> f64 {
        let params = OptParams::from_5param(p);
        geometry::reprojection_error(points, &params)
    };

    for init in &STARTS {
        let result = powell_minimize(&eval, init, &BOUNDS_5, 20, 1e-12);

        if let Some((x, f)) = result {
            log::debug!(
                "  start: x_ty={:.4}, intersect={:.4}, cam_d={:.4}, x_rz={:.5}, z_rx={:.5}, residual={f:.6}",
                x[0],
                x[1],
                x[2],
                x[3],
                x[4]
            );

            if best.as_ref().is_none_or(|(_, r)| f < *r) {
                best = Some((OptParams::from_5param(&x), f));
            }
        }
    }

    best.ok_or(CalibrateError::OptimizerFailed { max_evals: 0 })
}

/// Powell's method with conjugate direction updates and bounded line search.
///
/// Iteratively minimizes along coordinate directions, then replaces
/// the direction of largest decrease with the conjugate (overall
/// movement) direction. Uses golden-section line search within bounds.
fn powell_minimize(
    eval: &dyn Fn(&[f64; 5]) -> f64,
    init: &[f64; 5],
    bounds: &[(f64, f64); 5],
    max_cycles: usize,
    tol: f64,
) -> Option<([f64; 5], f64)> {
    let n = 5;
    let mut x = *init;
    let mut best_f = eval(&x);

    let mut dirs: Vec<[f64; 5]> = (0..n)
        .map(|i| {
            let mut d = [0.0; 5];
            d[i] = 1.0;
            d
        })
        .collect();

    let line_min =
        |base: &[f64; 5], dir: &[f64; 5], eval_fn: &dyn Fn(&[f64; 5]) -> f64| -> (f64, f64) {
            let mut t_lo = f64::NEG_INFINITY;
            let mut t_hi = f64::INFINITY;
            for i in 0..n {
                if dir[i].abs() > 1e-15 {
                    let lo = (bounds[i].0 - base[i]) / dir[i];
                    let hi = (bounds[i].1 - base[i]) / dir[i];
                    let (lo, hi) = if lo < hi { (lo, hi) } else { (hi, lo) };
                    t_lo = t_lo.max(lo);
                    t_hi = t_hi.min(hi);
                }
            }
            t_lo = t_lo.max(-2.0);
            t_hi = t_hi.min(2.0);

            if t_lo >= t_hi {
                return (0.0, eval_fn(base));
            }

            let golden = 0.381966011250105;
            let mut a = t_lo;
            let mut b = t_hi;

            let point_at = |t: f64| -> [f64; 5] {
                let mut p = *base;
                for i in 0..n {
                    p[i] = (base[i] + t * dir[i]).clamp(bounds[i].0, bounds[i].1);
                }
                p
            };

            for _ in 0..50 {
                if (b - a) < 1e-14 {
                    break;
                }
                let t1 = a + golden * (b - a);
                let t2 = b - golden * (b - a);
                if eval_fn(&point_at(t1)) < eval_fn(&point_at(t2)) {
                    b = t2;
                } else {
                    a = t1;
                }
            }
            let t_best = (a + b) / 2.0;
            let p_best = point_at(t_best);
            (t_best, eval_fn(&p_best))
        };

    for _cycle in 0..max_cycles {
        let prev_f = best_f;
        let x0 = x;

        let mut max_decrease = 0.0f64;
        let mut max_decrease_idx = 0;

        for (di, dir) in dirs.iter().enumerate() {
            let f_before = eval(&x);
            let (t, f_after) = line_min(&x, dir, eval);
            for i in 0..n {
                x[i] = (x[i] + t * dir[i]).clamp(bounds[i].0, bounds[i].1);
            }
            best_f = f_after;

            let decrease = f_before - f_after;
            if decrease > max_decrease {
                max_decrease = decrease;
                max_decrease_idx = di;
            }
        }

        // Conjugate direction update
        let mut new_dir = [0.0; 5];
        let mut norm_sq = 0.0;
        for i in 0..n {
            new_dir[i] = x[i] - x0[i];
            norm_sq += new_dir[i] * new_dir[i];
        }

        if norm_sq > 1e-20 {
            let norm = norm_sq.sqrt();
            for d in &mut new_dir {
                *d /= norm;
            }

            let (t, f_conj) = line_min(&x, &new_dir, eval);
            if f_conj < best_f {
                for i in 0..n {
                    x[i] = (x[i] + t * new_dir[i]).clamp(bounds[i].0, bounds[i].1);
                }
                best_f = f_conj;
                dirs[max_decrease_idx] = new_dir;
            }
        }

        if (prev_f - best_f).abs() < tol {
            break;
        }
    }

    Some((x, best_f))
}

/// Refine with a 6th parameter (z_rz, left plane roll).
///
/// Seeds from the 5-param result and adds z_rz with tight bounds.
/// Uses COBYLA since we're refining near a known good solution.
fn refine_6param(
    points: &[MatchedPoint],
    seed: &OptParams,
    config: &CalibrationConfig,
) -> Result<(OptParams, f64), CalibrateError> {
    let init = seed.to_6param();

    let bounds: [(f64, f64); 6] = [
        BOUNDS_5[0],
        BOUNDS_5[1],
        BOUNDS_5[2],
        BOUNDS_5[3],
        BOUNDS_5[4],
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
        RhoBeg::All(0.05),
        Some(stop_tols()),
    );

    match result {
        Ok((_status, x, f)) => Ok((OptParams::from_6param(&x), f)),
        Err((_status, x, f)) if f.is_finite() => Ok((OptParams::from_6param(&x), f)),
        Err(_) => Err(CalibrateError::OptimizerFailed {
            max_evals: config.max_optimizer_evals,
        }),
    }
}

/// Run the full optimization pipeline on a set of matched points.
///
/// Optimizes 5 parameters with multi-start COBYLA. If `config.enable_sixth_param`
/// is true, refines with a 6th parameter (left plane roll), accepted only
/// if it improves the residual.
pub fn optimize(
    points: &[MatchedPoint],
    config: &CalibrationConfig,
) -> Result<(PlaneLayout, f64), CalibrateError> {
    let (params_5, residual_5) = optimize_5param(points, config)?;

    let (final_params, final_residual) = if config.enable_sixth_param {
        match refine_6param(points, &params_5, config) {
            Ok((params_6, residual_6)) if residual_6 < residual_5 => {
                log::debug!("6-param improved: {residual_5:.6} -> {residual_6:.6}");
                (params_6, residual_6)
            }
            Ok((_, residual_6)) => {
                log::debug!(
                    "6-param did not improve ({residual_6:.6} >= {residual_5:.6}), keeping 5-param"
                );
                (params_5, residual_5)
            }
            Err(e) => {
                log::warn!("6-param failed ({e}), keeping 5-param");
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
        x_rx: 0.0,
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
    /// Traces rays from the camera through both planes, producing
    /// point pairs that are perfectly consistent with the given params.
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
                let fx = (ix as f64 + 0.5) / grid as f64;
                let fy = (iy as f64 + 0.5) / grid as f64;

                let yaw = -0.9 + fx * 0.5;
                let pitch = (fy - 0.5) * 0.4;
                let d = nalgebra::Vector3::new(yaw, pitch, yaw - 0.3).normalize();

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

        // Verify ground truth: reprojection error should be ~0
        let true_err = geometry::reprojection_error(&points, &true_params);
        assert!(
            true_err < 0.001,
            "synthetic points don't have near-zero error at true params: {true_err}"
        );

        let config = CalibrationConfig {
            enable_sixth_param: false,
            max_optimizer_evals: 5000,
            ..Default::default()
        };

        let (layout, _) = optimize(&points, &config).expect("optimization should succeed");

        assert_abs_diff_eq!(layout.camera_axis_offset, 0.225, epsilon = 0.05);
        assert_abs_diff_eq!(layout.intersect, 0.5, epsilon = 0.1);
        assert_abs_diff_eq!(layout.x_ty, 0.01, epsilon = 0.02);
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
        assert_abs_diff_eq!(layout.z_rz, 0.0, epsilon = Z_RZ_BOUND);
    }
}
