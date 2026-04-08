//! Position optimization for stereo camera calibration.
//!
//! Finds the placement parameters that minimize the seam-weighted
//! symmetric reprojection error between matched feature point pairs.
//!
//! ## Architecture
//!
//! The optimizer is behind a trait ([`Optimizer`]) so the backend is
//! swappable. The default implementation ([`NelderMeadOptimizer`]) uses
//! multi-start Nelder-Mead via the `argmin` crate, which was proven to
//! converge on all test footages (GoPro, DJI, XTU) where Powell and
//! COBYLA diverged on wide-overlap rigs.
//!
//! ## Seam weighting
//!
//! The objective function weights each point by proximity to the stitch
//! seam via a Gaussian (sigma configurable, default 0.08). This ensures
//! the optimizer prioritizes alignment where it matters most visually.

use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use reco_core::calibration::PlaneLayout;

use crate::error::CalibrateError;
use crate::geometry::{self, OptParams};
use crate::types::{CalibrationConfig, MatchedPoint};

// ---------------------------------------------------------------------------
// Optimizer trait (swappable backend)
// ---------------------------------------------------------------------------

/// Trait for calibration parameter optimizers.
///
/// Implementations take a set of matched points and configuration, and
/// return the optimal [`PlaneLayout`] with its residual error. This
/// abstraction allows swapping optimization backends (Nelder-Mead, Powell,
/// L-BFGS-B, etc.) without changing the calibration pipeline.
pub trait Optimizer {
    /// Find the placement parameters that minimize the calibration error.
    fn optimize(
        &self,
        points: &[MatchedPoint],
        config: &CalibrationConfig,
    ) -> Result<(PlaneLayout, f64), CalibrateError>;
}

// ---------------------------------------------------------------------------
// Nelder-Mead implementation
// ---------------------------------------------------------------------------

/// Bounds for the base 5 parameters.
///
/// Order: `[cam_d, intersect, x_ty, x_rz, z_rx]`
///
/// Rotation bounds set to ±0.3 rad (~17 deg) to accommodate cameras
/// with larger mounting misalignment (e.g. DJI Action 4 with rotation-
/// corrected left video requires z_rx ≈ -0.14 rad).
const BOUNDS_5: [(f64, f64); 5] = [
    (0.1, 0.30), // cam_d: camera distance
    (0.0, 1.0),  // intersect: overlap ratio
    (-0.1, 0.1), // x_ty: vertical translation
    (-0.3, 0.3), // x_rz: right plane roll (~17 deg)
    (-0.3, 0.3), // z_rx: left plane tilt (~17 deg)
];

/// Bound for the optional 6th parameter (x_rx, right plane pitch).
const X_RX_BOUND: (f64, f64) = (-0.3, 0.3); // ~17 deg

/// Get bounds for the active parameter count.
fn active_bounds(enable_x_rx: bool, lock_cam_d: bool) -> Vec<(f64, f64)> {
    let mut b = if lock_cam_d {
        // Locked: [intersect, x_ty, x_rz, z_rx]
        BOUNDS_5[1..].to_vec()
    } else {
        BOUNDS_5.to_vec()
    };
    if enable_x_rx {
        b.push(X_RX_BOUND);
    }
    b
}

/// Starting points for multi-start optimization (base 5 params).
///
/// Explores diverse cam_d and intersect values. Rotations always
/// start at zero.
const STARTS_5: [[f64; 5]; 8] = [
    [0.225, 0.5, 0.0, 0.0, 0.0], // center
    [0.15, 0.5, 0.0, 0.0, 0.0],  // low cam_d
    [0.30, 0.5, 0.0, 0.0, 0.0],  // high cam_d
    [0.225, 0.3, 0.0, 0.0, 0.0], // low intersect
    [0.225, 0.7, 0.0, 0.0, 0.0], // high intersect
    [0.20, 0.5, 0.0, 0.0, 0.0],  // alternative cam_d
    [0.19, 0.65, 0.0, 0.0, 0.0], // DJI-like: low cam_d, high intersect
    [0.15, 0.7, 0.0, 0.0, 0.0],  // wide-FOV: very low cam_d, high intersect
];

/// Starting points when cam_d is locked (4 params).
///
/// Order: `[intersect, x_ty, x_rz, z_rx]`.
/// cam_d is derived as `0.5 * (1 - intersect)`.
const STARTS_4: [[f64; 4]; 5] = [
    [0.5, 0.0, 0.0, 0.0], // center
    [0.3, 0.0, 0.0, 0.0], // low intersect (= high cam_d)
    [0.7, 0.0, 0.0, 0.0], // high intersect (= low cam_d)
    [0.4, 0.0, 0.0, 0.0], // below center
    [0.6, 0.0, 0.0, 0.0], // above center
];

/// Perturbation scale for building the initial simplex around each start.
///
/// For an n-dimensional problem, Nelder-Mead needs n+1 vertices. We
/// generate them by perturbing each dimension of the start point by
/// this fraction of the parameter range.
const SIMPLEX_PERTURBATION: f64 = 0.10;

/// Maximum iterations per Nelder-Mead run.
const MAX_ITERS: u64 = 5000;

/// Cost function for argmin's Nelder-Mead solver.
///
/// Wraps the seam-weighted reprojection error with a penalty term
/// for out-of-bounds parameters (Nelder-Mead is unconstrained).
struct CalibrationCost<'a> {
    points: &'a [MatchedPoint],
    sigma: f64,
    bounds: Vec<(f64, f64)>,
    /// When true, cam_d is derived from intersect (cam_d = half_offset).
    /// The parameter vector has 4 elements: `[intersect, x_ty, x_rz, z_rx]`.
    lock_cam_d: bool,
    /// Fraction of worst points to drop (0.0 = no trimming, 0.2 = drop 20%).
    trim_fraction: f64,
}

impl CostFunction for CalibrationCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let params = if self.lock_cam_d {
            params_from_vec_locked(p)
        } else {
            params_from_vec(p)
        };
        let err = if self.trim_fraction > 0.0 {
            geometry::trimmed_seam_weighted_reprojection_error(
                self.points,
                &params,
                self.sigma,
                self.trim_fraction,
            )
        } else {
            geometry::seam_weighted_reprojection_error(self.points, &params, self.sigma)
        };

        // Quadratic penalty for out-of-bounds parameters.
        let penalty = bounds_penalty(p, &self.bounds);

        Ok(err + penalty)
    }
}

/// Convert a parameter vector to [`OptParams`].
///
/// Base order: `[cam_d, intersect, x_ty, x_rz, z_rx]`.
/// If 6 elements, the 6th is `x_rx` (right plane pitch).
fn params_from_vec(p: &[f64]) -> OptParams {
    OptParams {
        cam_d: p[0],
        intersect: p[1],
        x_ty: p[2],
        x_rz: p[3],
        z_rx: p[4],
        z_rz: if p.len() > 5 { Some(p[5]) } else { None },
    }
}

/// Convert a locked parameter vector to [`OptParams`].
///
/// cam_d is derived from intersect: `cam_d = 0.5 * (1 - intersect)`.
/// Order: `[intersect, x_ty, x_rz, z_rx]`.
/// If 5 elements, the 5th is `x_rx` (right plane pitch).
fn params_from_vec_locked(p: &[f64]) -> OptParams {
    let intersect = p[0];
    OptParams {
        cam_d: 0.5 * (1.0 - intersect),
        intersect,
        x_ty: p[1],
        x_rz: p[2],
        z_rx: p[3],
        z_rz: if p.len() > 4 { Some(p[4]) } else { None },
    }
}

/// Quadratic penalty for parameters outside bounds.
fn bounds_penalty(p: &[f64], bounds: &[(f64, f64)]) -> f64 {
    let scale = 1e4;
    let mut penalty = 0.0;
    for (i, &val) in p.iter().enumerate() {
        if i >= bounds.len() {
            break;
        }
        let (lo, hi) = bounds[i];
        if val < lo {
            let d = lo - val;
            penalty += scale * d * d;
        } else if val > hi {
            let d = val - hi;
            penalty += scale * d * d;
        }
    }
    penalty
}

/// Build an initial simplex (n+1 vertices) around a start point.
fn build_simplex(start: &[f64], bounds: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let n = start.len();
    let mut vertices: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    vertices.push(start.to_vec());

    for i in 0..n {
        let mut vertex = start.to_vec();
        let range = bounds[i].1 - bounds[i].0;
        let delta = SIMPLEX_PERTURBATION * range;
        if vertex[i] + delta <= bounds[i].1 {
            vertex[i] += delta;
        } else {
            vertex[i] -= delta;
        }
        vertices.push(vertex);
    }
    vertices
}

/// Run a single Nelder-Mead optimization from a start point.
fn run_nelder_mead(cost: &CalibrationCost<'_>, start: &[f64]) -> Option<(Vec<f64>, f64)> {
    let simplex = build_simplex(start, &cost.bounds);
    let solver: NelderMead<Vec<f64>, f64> =
        NelderMead::new(simplex).with_sd_tolerance(1e-12).ok()?;

    let res = Executor::new(cost.clone(), solver)
        .configure(|state| state.max_iters(MAX_ITERS))
        .run()
        .ok()?;

    let p = res.state().get_best_param()?.clone();
    let f = res.state().get_best_cost();
    Some((p, f))
}

/// Multi-start Nelder-Mead optimizer.
///
/// Runs Nelder-Mead from multiple starting points and returns the
/// best result. Uses seam-weighted reprojection error as the objective
/// with quadratic penalty for bound enforcement.
///
/// This is the default [`Optimizer`] implementation, proven to converge
/// on all test footages where Powell and COBYLA failed.
pub struct NelderMeadOptimizer;

impl Optimizer for NelderMeadOptimizer {
    fn optimize(
        &self,
        points: &[MatchedPoint],
        config: &CalibrationConfig,
    ) -> Result<(PlaneLayout, f64), CalibrateError> {
        let lock = config.lock_cam_d;
        let bounds = active_bounds(config.enable_x_rx, lock);
        let cost = CalibrationCost {
            points,
            sigma: config.seam_sigma,
            bounds: bounds.clone(),
            lock_cam_d: lock,
            trim_fraction: config.trim_fraction,
        };

        let mut best: Option<(Vec<f64>, f64)> = None;

        // IMU seeds for rotation parameters
        let xrx_default = config.imu_xrx_seed.unwrap_or(0.0);
        let zrx_default = config.imu_zrx_seed.unwrap_or(0.0);

        if lock {
            // Locked mode: 4 params [intersect, x_ty, x_rz, z_rx]
            for base_start in &STARTS_4 {
                let mut start = base_start.to_vec();
                start[3] = zrx_default;
                if config.enable_x_rx {
                    start.push(xrx_default);
                }
                if let Some((p, f)) = run_nelder_mead(&cost, &start) {
                    log::debug!("NM locked: intersect={:.3}: cost={f:.8}", start[0]);
                    if best.as_ref().is_none_or(|(_, r)| f < *r) {
                        best = Some((p, f));
                    }
                }
            }

            if let Some(xrz_seed) = config.imu_xrz_seed {
                let mut imu_start = STARTS_4[0].to_vec();
                imu_start[2] = xrz_seed;
                if config.enable_x_rx {
                    imu_start.push(xrx_default);
                }
                if let Some((p, f)) = run_nelder_mead(&cost, &imu_start) {
                    log::debug!("NM locked IMU-seeded: cost={f:.8}");
                    if best.as_ref().is_none_or(|(_, r)| f < *r) {
                        best = Some((p, f));
                    }
                }
            }
        } else {
            // Standard mode: 5 params [cam_d, intersect, x_ty, x_rz, z_rx]
            for base_start in &STARTS_5 {
                let mut start = base_start.to_vec();
                start[4] = zrx_default;
                if config.enable_x_rx {
                    start.push(xrx_default);
                }
                if let Some((p, f)) = run_nelder_mead(&cost, &start) {
                    log::debug!(
                        "NM start: cam_d={:.3}, intersect={:.3}: cost={f:.8}",
                        start[0],
                        start[1]
                    );
                    if best.as_ref().is_none_or(|(_, r)| f < *r) {
                        best = Some((p, f));
                    }
                }
            }

            if let Some(xrz_seed) = config.imu_xrz_seed {
                let mut imu_start = STARTS_5[0].to_vec();
                imu_start[3] = xrz_seed;
                if config.enable_x_rx {
                    imu_start.push(xrx_default);
                }
                if let Some((p, f)) = run_nelder_mead(&cost, &imu_start) {
                    log::debug!("NM IMU-seeded start: cost={f:.8}");
                    if best.as_ref().is_none_or(|(_, r)| f < *r) {
                        best = Some((p, f));
                    }
                }
            }
        }

        let (best_p, best_cost) = best.ok_or(CalibrateError::OptimizerFailed {
            max_evals: config.max_optimizer_evals,
        })?;

        let params = if lock {
            params_from_vec_locked(&best_p)
        } else {
            params_from_vec(&best_p)
        };
        let layout = PlaneLayout {
            camera_axis_offset: params.cam_d,
            intersect: params.intersect,
            x_ty: params.x_ty,
            x_rz: params.x_rz,
            z_rx: params.z_rx,
            x_rx: if config.enable_x_rx {
                params.z_rz.unwrap_or(0.0)
            } else {
                0.0
            },
            z_rz: 0.0,
        };

        Ok((layout, best_cost))
    }
}

// We need Clone for CalibrationCost to run multiple Executor instances
impl Clone for CalibrationCost<'_> {
    fn clone(&self) -> Self {
        Self {
            points: self.points,
            sigma: self.sigma,
            bounds: self.bounds.clone(),
            lock_cam_d: self.lock_cam_d,
            trim_fraction: self.trim_fraction,
        }
    }
}

/// Run the default optimizer (Nelder-Mead) on a set of matched points.
///
/// This is the convenience entry point used by the calibration pipeline.
/// For custom optimizer backends, use the [`Optimizer`] trait directly.
pub fn optimize(
    points: &[MatchedPoint],
    config: &CalibrationConfig,
) -> Result<(PlaneLayout, f64), CalibrateError> {
    NelderMeadOptimizer.optimize(points, config)
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

                points.push(MatchedPoint::from_planes(
                    [x_coord, y_coord],
                    [z_coord, z_y],
                ));
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

        let config = CalibrationConfig::default();
        let (layout, _) = optimize(&points, &config).expect("optimization should succeed");

        assert_abs_diff_eq!(layout.camera_axis_offset, 0.225, epsilon = 0.05);
        assert_abs_diff_eq!(layout.intersect, 0.5, epsilon = 0.1);
        assert_abs_diff_eq!(layout.x_ty, 0.01, epsilon = 0.02);
    }

    #[test]
    fn optimizer_handles_small_rotations() {
        let true_params = OptParams {
            x_ty: 0.005,
            intersect: 0.55,
            cam_d: 0.24,
            x_rz: 0.01,
            z_rx: -0.005,
            z_rz: None,
        };
        let points = synthetic_points(&true_params, 50);

        let config = CalibrationConfig::default();
        let (layout, _) = optimize(&points, &config).expect("optimization should succeed");
        assert_abs_diff_eq!(layout.camera_axis_offset, 0.24, epsilon = 0.05);
        assert_abs_diff_eq!(layout.intersect, 0.55, epsilon = 0.1);
    }

    #[test]
    fn bounds_penalty_zero_inside() {
        let bounds = active_bounds(false, false);
        let inside = vec![0.225, 0.5, 0.0, 0.0, 0.0];
        assert_abs_diff_eq!(bounds_penalty(&inside, &bounds), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn bounds_penalty_nonzero_outside() {
        let bounds = active_bounds(false, false);
        let outside = vec![0.05, 0.5, 0.0, 0.0, 0.0]; // cam_d below 0.1
        assert!(bounds_penalty(&outside, &bounds) > 0.0);
    }

    #[test]
    fn simplex_has_correct_size() {
        let bounds = active_bounds(false, false);
        let start = STARTS_5[0].to_vec();
        let simplex = build_simplex(&start, &bounds);
        assert_eq!(simplex.len(), 6); // n+1 = 5+1
        for v in &simplex {
            for (i, &val) in v.iter().enumerate() {
                assert!(
                    val >= bounds[i].0 - 1e-10 && val <= bounds[i].1 + 1e-10,
                    "vertex dim {i} = {val} out of bounds [{}, {}]",
                    bounds[i].0,
                    bounds[i].1
                );
            }
        }
    }

    #[test]
    fn simplex_6d_with_x_rx() {
        let bounds = active_bounds(true, false);
        let mut start = STARTS_5[0].to_vec();
        start.push(0.0); // x_rx
        let simplex = build_simplex(&start, &bounds);
        assert_eq!(simplex.len(), 7); // n+1 = 6+1
    }
}
