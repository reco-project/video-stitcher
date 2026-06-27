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
//! converge reliably on all test footages (GoPro, DJI, XTU), including
//! wide-overlap rigs.
//!
//! ## Seam weighting
//!
//! The objective function weights each point by proximity to the stitch
//! seam via a Gaussian (sigma configurable, default 0.08). This ensures
//! the optimizer prioritizes alignment where it matters most visually.

use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use reco_core::calibration::{Framing, Topology};

use crate::error::CalibrateError;
use crate::geometry::{self, OptParams};
use crate::types::{CalibrationConfig, MatchedPoint};

// ---------------------------------------------------------------------------
// Optimizer trait (swappable backend)
// ---------------------------------------------------------------------------

/// Trait for calibration parameter optimizers.
///
/// Implementations take a set of matched points and configuration, and
/// return the optimal [`Topology`] with its residual error. This
/// abstraction allows swapping optimization backends without changing
/// the calibration pipeline.
pub trait Optimizer {
    /// Find the placement parameters that minimize the calibration error.
    fn optimize(
        &self,
        points: &[MatchedPoint],
        config: &CalibrationConfig,
    ) -> Result<(Topology, Framing, f64), CalibrateError>;
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
fn active_bounds(enable_x_rx: bool, lock_cam_d: bool, lock_z_rx: bool) -> Vec<(f64, f64)> {
    let mut b = if lock_cam_d {
        // Locked cam_d: [intersect, x_ty, x_rz, z_rx]
        BOUNDS_5[1..].to_vec()
    } else {
        BOUNDS_5.to_vec()
    };
    // Remove z_rx (last element before x_rx) if locked
    if lock_z_rx {
        b.pop(); // remove z_rx
    }
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

/// Cost function for argmin's Nelder-Mead solver.
///
/// Wraps the seam-weighted reprojection error with a penalty term
/// for out-of-bounds parameters (Nelder-Mead is unconstrained).
struct CalibrationCost<'a> {
    points: &'a [MatchedPoint],
    sigma: f64,
    bounds: Vec<(f64, f64)>,
    /// When true, cam_d is derived from intersect (cam_d = half_offset).
    lock_cam_d: bool,
    /// When true, z_rx is fixed at 0 (z-plane only translates, no rotation).
    lock_z_rx: bool,
    /// Fraction of worst points to drop (0.0 = no trimming, 0.2 = drop 20%).
    trim_fraction: f64,
}

impl CostFunction for CalibrationCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let params = match (self.lock_cam_d, self.lock_z_rx) {
            (true, true) => params_from_vec_locked_no_zrx(p),
            (true, false) => params_from_vec_locked(p),
            (false, true) => params_from_vec_no_zrx(p),
            (false, false) => params_from_vec(p),
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
/// If 6 elements, the 6th is stored in `x_rx` (right plane pitch)
/// when `enable_x_rx` is set.
fn params_from_vec(p: &[f64]) -> OptParams {
    debug_assert!(p.len() >= 5, "need at least 5 params, got {}", p.len());
    OptParams {
        cam_d: p[0],
        intersect: p[1],
        x_ty: p[2],
        x_rz: p[3],
        z_rx: p[4],
        z_rz: None,
        x_rx: if p.len() > 5 { Some(p[5]) } else { None },
    }
}

/// Convert a parameter vector to [`OptParams`] with z_rx fixed at 0.
///
/// Order: `[cam_d, intersect, x_ty, x_rz]`.
fn params_from_vec_no_zrx(p: &[f64]) -> OptParams {
    debug_assert!(p.len() >= 4, "need at least 4 params, got {}", p.len());
    OptParams {
        cam_d: p[0],
        intersect: p[1],
        x_ty: p[2],
        x_rz: p[3],
        z_rx: 0.0,
        z_rz: None,
        x_rx: if p.len() > 4 { Some(p[4]) } else { None },
    }
}

/// Convert a locked parameter vector to [`OptParams`].
///
/// cam_d is derived from intersect: `cam_d = 0.5 * (1 - intersect)`.
/// Order: `[intersect, x_ty, x_rz, z_rx]`.
/// If 5 elements, the 5th is `x_rx` (right plane pitch).
fn params_from_vec_locked(p: &[f64]) -> OptParams {
    debug_assert!(p.len() >= 4, "need at least 4 params, got {}", p.len());
    let intersect = p[0];
    OptParams {
        cam_d: 0.5 * (1.0 - intersect),
        intersect,
        x_ty: p[1],
        x_rz: p[2],
        z_rx: p[3],
        z_rz: None,
        x_rx: if p.len() > 4 { Some(p[4]) } else { None },
    }
}

/// Convert a locked parameter vector with z_rx fixed at 0.
///
/// cam_d derived from intersect, z_rx = 0.
/// Order: `[intersect, x_ty, x_rz]`.
fn params_from_vec_locked_no_zrx(p: &[f64]) -> OptParams {
    debug_assert!(p.len() >= 3, "need at least 3 params, got {}", p.len());
    let intersect = p[0];
    OptParams {
        cam_d: 0.5 * (1.0 - intersect),
        intersect,
        x_ty: p[1],
        x_rz: p[2],
        z_rx: 0.0,
        z_rz: None,
        x_rx: if p.len() > 3 { Some(p[3]) } else { None },
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
fn run_nelder_mead(
    cost: &CalibrationCost<'_>,
    start: &[f64],
    max_iters: u64,
) -> Option<(Vec<f64>, f64)> {
    let simplex = build_simplex(start, &cost.bounds);
    let solver: NelderMead<Vec<f64>, f64> =
        NelderMead::new(simplex).with_sd_tolerance(1e-12).ok()?;

    let res = Executor::new(cost.clone(), solver)
        .configure(|state| state.max_iters(max_iters))
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
/// reliably on all test footages.
pub struct NelderMeadOptimizer;

impl Optimizer for NelderMeadOptimizer {
    fn optimize(
        &self,
        points: &[MatchedPoint],
        config: &CalibrationConfig,
    ) -> Result<(Topology, Framing, f64), CalibrateError> {
        let lock = config.optimizer.lock_cam_d;
        let lock_zrx = config.optimizer.lock_z_rx;
        let enable_xrx = config.optimizer.enable_x_rx;
        let max_iters = config.optimizer.max_iters as u64;
        let bounds = active_bounds(enable_xrx, lock, lock_zrx);
        let cost = CalibrationCost {
            points,
            sigma: config.optimizer.seam_sigma,
            bounds: bounds.clone(),
            lock_cam_d: lock,
            lock_z_rx: lock_zrx,
            trim_fraction: config.optimizer.trim_fraction,
        };

        let mut best: Option<(Vec<f64>, f64)> = None;

        // IMU seeds for rotation parameters
        let xrx_default = config.imu_xrx_seed.unwrap_or(0.0);
        let zrx_default = config.imu_zrx_seed.unwrap_or(0.0);

        // Build starting points based on which parameters are active
        let raw_starts: Vec<Vec<f64>> = if lock {
            STARTS_4.iter().map(|s| s.to_vec()).collect()
        } else {
            STARTS_5.iter().map(|s| s.to_vec()).collect()
        };

        for base_start in &raw_starts {
            let mut start = base_start.clone();

            // Set z_rx seed (last element before x_rx) if not locked
            if !lock_zrx {
                let zrx_idx = if lock { 3 } else { 4 };
                if zrx_idx < start.len() {
                    start[zrx_idx] = zrx_default;
                }
            } else {
                // Remove z_rx from start vector
                let zrx_idx = if lock { 3 } else { 4 };
                if zrx_idx < start.len() {
                    start.remove(zrx_idx);
                }
            }

            if enable_xrx {
                start.push(xrx_default);
            }

            if let Some((p, f)) = run_nelder_mead(&cost, &start, max_iters) {
                log::debug!("NM start: cost={f:.8}");
                if best.as_ref().is_none_or(|(_, r)| f < *r) {
                    best = Some((p, f));
                }
            }
        }

        {
            // IMU-seeded extra start
            if let Some(xrz_seed) = config.imu_xrz_seed {
                let mut imu_start = if lock {
                    STARTS_4[0].to_vec()
                } else {
                    STARTS_5[0].to_vec()
                };
                let xrz_idx = if lock { 2 } else { 3 };
                imu_start[xrz_idx] = xrz_seed;
                if lock_zrx {
                    let zrx_idx = if lock { 3 } else { 4 };
                    if zrx_idx < imu_start.len() {
                        imu_start.remove(zrx_idx);
                    }
                }
                if enable_xrx {
                    imu_start.push(xrx_default);
                }
                if let Some((p, f)) = run_nelder_mead(&cost, &imu_start, max_iters) {
                    log::debug!("NM IMU-seeded start: cost={f:.8}");
                    if best.as_ref().is_none_or(|(_, r)| f < *r) {
                        best = Some((p, f));
                    }
                }
            }
        }

        let (best_p, best_cost) = best.ok_or(CalibrateError::OptimizerFailed {
            max_evals: config.optimizer.max_iters,
        })?;

        let params = match (lock, lock_zrx) {
            (true, true) => params_from_vec_locked_no_zrx(&best_p),
            (true, false) => params_from_vec_locked(&best_p),
            (false, true) => params_from_vec_no_zrx(&best_p),
            (false, false) => params_from_vec(&best_p),
        };
        let topology = Topology {
            intersect: params.intersect,
            x_ty: params.x_ty,
            x_rz: params.x_rz,
            z_rx: params.z_rx,
            x_rx: if enable_xrx {
                params.x_rx.unwrap_or(0.0)
            } else {
                0.0
            },
            z_rz: 0.0,
            blend_width: 0.05,
        };
        let framing = Framing {
            axis_offset: params.cam_d,
            tilt: 0.0,
            roll: 0.0,
        };

        Ok((topology, framing, best_cost))
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
            lock_z_rx: self.lock_z_rx,
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
) -> Result<(Topology, Framing, f64), CalibrateError> {
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
            x_rx: None,
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
        let (topology, framing, _) =
            optimize(&points, &config).expect("optimization should succeed");

        assert_abs_diff_eq!(framing.axis_offset, 0.225, epsilon = 0.05);
        assert_abs_diff_eq!(topology.intersect, 0.5, epsilon = 0.1);
        assert_abs_diff_eq!(topology.x_ty, 0.01, epsilon = 0.02);
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
            x_rx: None,
        };
        let points = synthetic_points(&true_params, 50);

        let config = CalibrationConfig::default();
        let (topology, framing, _) =
            optimize(&points, &config).expect("optimization should succeed");
        assert_abs_diff_eq!(framing.axis_offset, 0.24, epsilon = 0.05);
        assert_abs_diff_eq!(topology.intersect, 0.55, epsilon = 0.1);
    }

    #[test]
    fn bounds_penalty_zero_inside() {
        let bounds = active_bounds(false, false, false);
        let inside = vec![0.225, 0.5, 0.0, 0.0, 0.0];
        assert_abs_diff_eq!(bounds_penalty(&inside, &bounds), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn bounds_penalty_nonzero_outside() {
        let bounds = active_bounds(false, false, false);
        let outside = vec![0.05, 0.5, 0.0, 0.0, 0.0]; // cam_d below 0.1
        assert!(bounds_penalty(&outside, &bounds) > 0.0);
    }

    #[test]
    fn simplex_has_correct_size() {
        let bounds = active_bounds(false, false, false);
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
        let bounds = active_bounds(true, false, false);
        let mut start = STARTS_5[0].to_vec();
        start.push(0.0); // x_rx
        let simplex = build_simplex(&start, &bounds);
        assert_eq!(simplex.len(), 7); // n+1 = 6+1
    }
}
