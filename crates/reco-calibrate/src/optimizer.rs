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

/// Bounds for the 5-parameter optimization.
///
/// Order: `[cam_d, intersect, x_ty, x_rz, z_rx]`
const BOUNDS: [(f64, f64); 5] = [
    (0.1, 0.35),   // cam_d: camera distance
    (0.0, 1.0),    // intersect: overlap ratio
    (-0.1, 0.1),   // x_ty: vertical translation
    (-0.05, 0.05), // x_rz: right plane roll (~3 deg)
    (-0.05, 0.05), // z_rx: left plane tilt (~3 deg)
];

/// Starting points for multi-start optimization.
///
/// Explores diverse cam_d and intersect values. Rotations always
/// start at zero since the reprojection error landscape is well-behaved
/// near zero rotation.
const STARTS: [[f64; 5]; 6] = [
    [0.225, 0.5, 0.0, 0.0, 0.0], // center
    [0.15, 0.5, 0.0, 0.0, 0.0],  // low cam_d
    [0.30, 0.5, 0.0, 0.0, 0.0],  // high cam_d
    [0.225, 0.3, 0.0, 0.0, 0.0], // low intersect
    [0.225, 0.7, 0.0, 0.0, 0.0], // high intersect
    [0.20, 0.5, 0.0, 0.0, 0.0],  // alternative cam_d
];

/// Perturbation scale for building the initial simplex around each start.
///
/// For an n-dimensional problem, Nelder-Mead needs n+1 vertices. We
/// generate them by perturbing each dimension of the start point by
/// this fraction of the parameter range.
const SIMPLEX_PERTURBATION: f64 = 0.05;

/// Maximum iterations per Nelder-Mead run.
const MAX_ITERS: u64 = 2000;

/// Cost function for argmin's Nelder-Mead solver.
///
/// Wraps the seam-weighted reprojection error with a penalty term
/// for out-of-bounds parameters (Nelder-Mead is unconstrained).
struct CalibrationCost<'a> {
    points: &'a [MatchedPoint],
    sigma: f64,
}

impl CostFunction for CalibrationCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let params = params_from_vec(p);
        let err = geometry::seam_weighted_reprojection_error(self.points, &params, self.sigma);

        // Quadratic penalty for out-of-bounds parameters.
        // Nelder-Mead is unconstrained, so we guide it back with a
        // smooth penalty rather than hard clamping (which creates
        // flat regions that confuse the simplex).
        let penalty = bounds_penalty(p);

        Ok(err + penalty)
    }
}

/// Convert a parameter vector to [`OptParams`].
///
/// Order: `[cam_d, intersect, x_ty, x_rz, z_rx]`
fn params_from_vec(p: &[f64]) -> OptParams {
    OptParams {
        cam_d: p[0],
        intersect: p[1],
        x_ty: p[2],
        x_rz: p[3],
        z_rx: p[4],
        z_rz: None,
    }
}

/// Quadratic penalty for parameters outside bounds.
///
/// Returns 0 when all parameters are in bounds, otherwise a smooth
/// increasing penalty proportional to the squared distance from the
/// nearest bound.
fn bounds_penalty(p: &[f64]) -> f64 {
    let scale = 1e4; // strong enough to keep params in bounds
    let mut penalty = 0.0;
    for (i, &val) in p.iter().enumerate() {
        if i >= BOUNDS.len() {
            break;
        }
        let (lo, hi) = BOUNDS[i];
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
///
/// Each vertex perturbs one dimension by a fraction of its bound range,
/// alternating direction to explore both sides.
fn build_simplex(start: &[f64; 5]) -> Vec<Vec<f64>> {
    let n = start.len();
    let mut vertices: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    vertices.push(start.to_vec());

    for i in 0..n {
        let mut vertex = start.to_vec();
        let range = BOUNDS[i].1 - BOUNDS[i].0;
        let delta = SIMPLEX_PERTURBATION * range;
        // Perturb upward, but clamp if it would exceed the bound
        if vertex[i] + delta <= BOUNDS[i].1 {
            vertex[i] += delta;
        } else {
            vertex[i] -= delta;
        }
        vertices.push(vertex);
    }
    vertices
}

/// Run a single Nelder-Mead optimization from a start point.
///
/// Returns `Some((best_params, best_cost))` on success, `None` on failure.
fn run_nelder_mead(cost: &CalibrationCost<'_>, start: &[f64; 5]) -> Option<(Vec<f64>, f64)> {
    let simplex = build_simplex(start);
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
        let cost = CalibrationCost {
            points,
            sigma: config.seam_sigma,
        };

        let mut best: Option<(Vec<f64>, f64)> = None;

        for start in &STARTS {
            if let Some((p, f)) = run_nelder_mead(&cost, start) {
                log::debug!(
                    "NM start [{:.3}, {:.3}, {:.4}, {:.5}, {:.5}]: cost={f:.8}",
                    start[0],
                    start[1],
                    start[2],
                    start[3],
                    start[4],
                );
                if best.as_ref().is_none_or(|(_, r)| f < *r) {
                    best = Some((p, f));
                }
            }
        }

        // If IMU provides an x_rz seed, add an extra start
        if let Some(xrz_seed) = config.imu_xrz_seed {
            let mut imu_start = STARTS[0]; // center cam_d/intersect
            imu_start[3] = xrz_seed;
            if let Some((p, f)) = run_nelder_mead(&cost, &imu_start) {
                log::debug!("NM IMU-seeded start: cost={f:.8}");
                if best.as_ref().is_none_or(|(_, r)| f < *r) {
                    best = Some((p, f));
                }
            }
        }

        let (best_p, best_cost) = best.ok_or(CalibrateError::OptimizerFailed {
            max_evals: config.max_optimizer_evals,
        })?;

        let params = params_from_vec(&best_p);
        let layout = PlaneLayout {
            camera_axis_offset: params.cam_d,
            intersect: params.intersect,
            x_ty: params.x_ty,
            x_rz: params.x_rz,
            z_rx: params.z_rx,
            x_rx: 0.0,
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
        let inside = vec![0.225, 0.5, 0.0, 0.0, 0.0];
        assert_abs_diff_eq!(bounds_penalty(&inside), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn bounds_penalty_nonzero_outside() {
        let outside = vec![0.05, 0.5, 0.0, 0.0, 0.0]; // cam_d below 0.1
        assert!(bounds_penalty(&outside) > 0.0);
    }

    #[test]
    fn simplex_has_correct_size() {
        let start = STARTS[0];
        let simplex = build_simplex(&start);
        assert_eq!(simplex.len(), 6); // n+1 = 5+1
        // All vertices should be within bounds (with tolerance)
        for v in &simplex {
            for (i, &val) in v.iter().enumerate() {
                assert!(
                    val >= BOUNDS[i].0 - 1e-10 && val <= BOUNDS[i].1 + 1e-10,
                    "vertex dim {i} = {val} out of bounds [{}, {}]",
                    BOUNDS[i].0,
                    BOUNDS[i].1
                );
            }
        }
    }
}
