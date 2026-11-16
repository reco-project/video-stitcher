//! Bounded direct-image fitting with sensor-domain constraints.

use super::kernel::Stage;
use basin::{
    BoxConstraints, ConstraintJacobian, CostFunction, DenseMatrix, Executor, Gradient, Jacobian,
    NonlinearConstraints, Residual, RobustLeastSquares, Slsqp, SoftL1Loss, TerminationReason, Trf,
};
use rayon::prelude::*;
use reco_core::calibration::Calibration;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[cfg(test)]
#[path = "solver_diagnostics.rs"]
pub(crate) mod diagnostics;

// Seven placement parameters, then focal scale and four KB4 terms per camera.
pub(super) const N: usize = 7 + 2 * 5;
const SCALE: [f64; 7] = [0.06, 0.03, 0.08, 0.08, 0.08, 0.08, 0.04];
const OPTICS_SCALE: [f64; 5] = [0.2, 0.1, 0.1, 0.05, 0.02];

pub(crate) struct Control<'a> {
    pub(crate) interrupted: &'a AtomicBool,
    pub(crate) deadline: Instant,
}

impl Control<'_> {
    pub(crate) fn remaining(&self) -> Result<Duration, String> {
        self.check()?;
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| "calibration time limit reached".into())
    }

    pub(crate) fn check(&self) -> Result<(), String> {
        if self.interrupted.load(Ordering::Relaxed) {
            Err("calibration cancelled".into())
        } else if Instant::now() >= self.deadline {
            Err("calibration time limit reached".into())
        } else {
            Ok(())
        }
    }
}

pub(crate) fn decode_parameters(seed: &Calibration, x: &[f64]) -> Result<Calibration, String> {
    if ![7, N].contains(&x.len()) || x.iter().any(|x| !x.is_finite()) {
        return Err("invalid parameter vector".into());
    }
    let mut calibration = seed.clone();
    let shape = calibration
        .projection
        .l_shape_mut()
        .ok_or("expected L-shape calibration")?;
    for (i, field) in [
        &mut shape.intersect,
        &mut shape.x_ty,
        &mut shape.x_rz,
        &mut shape.x_rx,
        &mut shape.z_rx,
        &mut shape.z_rz,
    ]
    .into_iter()
    .enumerate()
    {
        *field += x[i] * SCALE[i];
    }
    calibration.framing.axis_offset += x[6] * SCALE[6];
    if x.len() == N {
        for (camera, lens) in calibration.lenses.iter_mut().enumerate() {
            let offset = 7 + camera * 5;
            let factor = (x[offset] * OPTICS_SCALE[0]).exp();
            lens.fx *= factor;
            lens.fy *= factor;
            for i in 0..4 {
                lens.distortion[i] += x[offset + 1 + i] * OPTICS_SCALE[1 + i];
            }
        }
    }
    Ok(calibration)
}

pub(crate) fn parameter_bounds(
    seed: &Calibration,
    count: usize,
    overlap: [f64; 2],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if ![7, N].contains(&count) {
        return Err("invalid parameter count".into());
    }
    let mut lower = vec![-3.0_f64; count];
    let mut upper = vec![3.0_f64; count];
    let intersect = seed
        .projection
        .l_shape()
        .ok_or("expected L-shape calibration")?
        .intersect;
    lower[0] = lower[0].max((overlap[0] - intersect) / SCALE[0]);
    upper[0] = upper[0].min((overlap[1] - intersect) / SCALE[0]);
    lower[6] = lower[6].max((0.05 - seed.framing.axis_offset) / SCALE[6]);
    if lower
        .iter()
        .zip(&upper)
        .any(|(&a, &b)| !a.is_finite() || !b.is_finite() || a >= b || a > 0.0 || b < 0.0)
    {
        return Err("empty bounds or infeasible seed".into());
    }
    Ok((lower, upper))
}

fn sensor_margin(calibration: &Calibration) -> Result<f64, String> {
    calibration
        .lenses
        .iter()
        .try_fold(f64::INFINITY, |minimum, lens| {
            let margin = reco_core::lens::sensor_domain_margin(lens);
            if margin.is_finite() {
                Ok(minimum.min(margin))
            } else {
                Err("nonfinite optical domain".into())
            }
        })
}

struct Model<'a> {
    seed: Calibration,
    stage: &'a Stage,
    pool: &'a rayon::ThreadPool,
    control: &'a Control<'a>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    sensor_penalty: bool,
    active_penalty_derivative: bool,
    difference_step: f64,
    optics: [bool; 2],
    residuals: AtomicUsize,
    jacobians: AtomicUsize,
    derivative_nanos: AtomicU64,
}

impl<'a> Model<'a> {
    fn new(
        seed: Calibration,
        stage: &'a Stage,
        pool: &'a rayon::ThreadPool,
        sensor_penalty: bool,
        parameters: usize,
        control: &'a Control<'a>,
    ) -> Result<Self, String> {
        control.check()?;
        let cal = &seed;
        cal.validate().map_err(|e| e.to_string())?;
        if cal.lenses.len() != 2 || cal.projection.l_shape().is_none() {
            return Err("calibration requires a two-source L-shape".into());
        }
        if stage.output_len() > 100_000 {
            return Err("excessive native model".into());
        }
        let margin = sensor_margin(cal)?;
        let (lower, upper) = parameter_bounds(&seed, parameters, [0.05, 0.95])?;
        if parameters == 7 && margin < 0.0 {
            return Err("fixed optics do not cover both sensors".into());
        }
        Ok(Self {
            seed,
            stage,
            pool,
            control,
            lower,
            upper,
            sensor_penalty: sensor_penalty && parameters == N,
            active_penalty_derivative: false,
            difference_step: 0.002,
            optics: [true; 2],
            residuals: AtomicUsize::new(0),
            jacobians: AtomicUsize::new(0),
            derivative_nanos: AtomicU64::new(0),
        })
    }

    fn decode(&self, x: &[f64]) -> Result<Calibration, String> {
        if x.len() != self.lower.len() {
            return Err("invalid parameter vector".into());
        }
        let mut cal = decode_parameters(&self.seed, x)?;
        for (i, free) in self.optics.into_iter().enumerate() {
            if !free {
                cal.lenses[i] = self.seed.lenses[i].clone();
            }
        }
        Ok(cal)
    }

    fn calibration(&self, x: &[f64]) -> Result<Calibration, String> {
        self.control.check()?;
        self.decode(x)
    }

    fn margin(&self, x: &[f64]) -> Result<f64, String> {
        let cal = self.calibration(x)?;
        sensor_margin(&cal)
    }

    fn sensor_boundary_active(&self, x: &[f64]) -> Result<bool, String> {
        if !self.sensor_penalty {
            return Ok(false);
        }
        if self.margin(x)? < 0.0 {
            return Ok(true);
        }
        // A stencil crossing the hinge cannot establish smooth interior convergence.
        // Placement does not affect sensor feasibility, so only probe optical columns.
        for i in 7..x.len() {
            let (hi, lo) = self.perturb(x, i);
            if self.margin(&hi)? < 0.0 || self.margin(&lo)? < 0.0 {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn values(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        self.control.check()?;
        self.residuals.fetch_add(1, Ordering::Relaxed);
        let cal = self.calibration(x)?;
        let mut values = Vec::new();
        self.stage
            .residual_checked(&cal, &mut values, || self.control.check())?;
        let frames = self.stage.frame_count();
        let scale = (values.len() as f64 / frames as f64).sqrt();
        let penalty = if self.sensor_penalty {
            Some((-sensor_margin(&cal)?).max(0.0) * 50.0 * scale)
        } else {
            None
        };
        // Repeat fixed-weight priors instead of changing their robust-loss branch as pairs are added.
        for _ in 0..frames {
            values.extend(
                x.iter()
                    .enumerate()
                    .map(|(i, x)| x * if i < 7 { 0.002 } else { 0.005 } * scale),
            );
            if let Some(penalty) = penalty {
                values.push(penalty);
            }
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite residual".into());
        }
        Ok(values)
    }

    fn perturb(&self, x: &[f64], i: usize) -> (Vec<f64>, Vec<f64>) {
        let (mut hi, mut lo) = (x.to_vec(), x.to_vec());
        hi[i] = (x[i] + self.difference_step).min(self.upper[i]);
        lo[i] = (x[i] - self.difference_step).max(self.lower[i]);
        (hi, lo)
    }

    fn columns(&self, x: &[f64]) -> Result<Vec<Vec<f64>>, String> {
        let started = Instant::now();
        self.jacobians.fetch_add(1, Ordering::Relaxed);
        let margin = self.margin(x)?;
        let result = self.pool.install(|| {
            (0..x.len())
                .into_par_iter()
                .map(|i| {
                    let (hi, lo) = self.perturb(x, i);
                    let delta = hi[i] - lo[i];
                    if delta <= 0.0 {
                        return Err("invalid difference interval".into());
                    }
                    let a = self.values(&hi)?;
                    let b = self.values(&lo)?;
                    let mut column: Vec<_> =
                        a.iter().zip(&b).map(|(a, b)| (a - b) / delta).collect();
                    if self.sensor_penalty && self.active_penalty_derivative {
                        let derivative = (self.margin(&hi)? - self.margin(&lo)?) / delta;
                        let frames = self.stage.frame_count();
                        let value = penalty_derivative(margin, derivative)
                            * 50.0
                            * (self.stage.output_len() as f64 / frames as f64).sqrt();
                        for frame in 0..frames {
                            column[self.stage.output_len() + frame * (x.len() + 1) + x.len()] =
                                value;
                        }
                    }
                    Ok(column)
                })
                .collect()
        });
        self.derivative_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        result
    }

    fn constrained_cost(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, String> {
        let values = self.values(x)?;
        let count = self.stage.output_len();
        let cost = values
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                if i < count {
                    0.09 * ((1.0 + (r / 0.3).powi(2)).sqrt() - 1.0)
                } else {
                    0.5 * r * r
                }
            })
            .sum::<f64>()
            / count as f64;
        if let Some(gradient) = gradient {
            for (g, col) in gradient.iter_mut().zip(self.columns(x)?) {
                *g = col
                    .iter()
                    .zip(&values)
                    .enumerate()
                    .map(|(i, (&j, &r))| {
                        j * r
                            / if i < count {
                                (1.0 + (r / 0.3).powi(2)).sqrt()
                            } else {
                                1.0
                            }
                    })
                    .sum::<f64>()
                    / count as f64;
            }
        }
        Ok(cost)
    }
}

fn penalty_derivative(margin: f64, derivative: f64) -> f64 {
    if margin < 0.0 { -derivative } else { 0.0 }
}

impl CostFunction for &Model<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = String;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, String> {
        Ok(self.values(x)?.iter().map(|v| v * v * 0.5).sum())
    }
}
impl Residual for &Model<'_> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = String;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, String> {
        self.values(x)
    }
}
impl Jacobian for &Model<'_> {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, String> {
        let columns = self.columns(x)?;
        Ok(DenseMatrix::from_fn(
            columns[0].len(),
            columns.len(),
            |i, j| columns[j][i],
        ))
    }
}
impl BoxConstraints for &Model<'_> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

struct Constrained<'a>(&'a Model<'a>);

impl CostFunction for Constrained<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = String;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, String> {
        self.0.constrained_cost(x, None)
    }
}

impl Gradient for Constrained<'_> {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, String> {
        let mut gradient = vec![0.0; x.len()];
        self.0.constrained_cost(x, Some(&mut gradient))?;
        Ok(gradient)
    }
}

impl NonlinearConstraints for Constrained<'_> {
    type Matrix = DenseMatrix;
    fn num_nonlinear_constraints(&self) -> usize {
        1
    }
    fn nonlinear_constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, String> {
        Ok(vec![-self.0.margin(x)?])
    }
    fn lower(&self) -> Option<&Vec<f64>> {
        Some(&self.0.lower)
    }
    fn upper(&self) -> Option<&Vec<f64>> {
        Some(&self.0.upper)
    }
}

impl ConstraintJacobian for Constrained<'_> {
    fn constraint_jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, String> {
        let mut gradient = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let (hi, lo) = self.0.perturb(x, i);
            gradient.push(-(self.0.margin(&hi)? - self.0.margin(&lo)?) / (hi[i] - lo[i]));
        }
        Ok(DenseMatrix::from_fn(1, x.len(), |_, i| gradient[i]))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Method {
    LeastSquares,
    Constrained,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SolveOptions {
    pub(crate) method: Method,
    pub(crate) budget: u64,
    pub(crate) tolerance: f64,
    pub(crate) difference_step: f64,
    pub(crate) active_penalty: bool,
    pub(crate) optics: [bool; 2],
}

#[derive(Debug)]
pub(crate) struct Solution {
    pub(crate) calibration: Calibration,
    pub(crate) x: Vec<f64>,
    pub(crate) cost: f64,
    pub(crate) status: String,
    pub(crate) success: bool,
    pub(crate) iterations: u64,
    pub(crate) elapsed: Duration,
    pub(crate) margin: f64,
    pub(crate) sensor_boundary_active: bool,
    pub(crate) residuals: usize,
    pub(crate) jacobians: usize,
    pub(crate) derivative_elapsed: Duration,
}

/// Fit a prepared stage without serialization or process transport.
/// The caller owns the deadline for the entire workflow, not just this stage.
pub(crate) fn solve_native(
    seed: Calibration,
    x: Vec<f64>,
    options: SolveOptions,
    stage: &Stage,
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<Solution, String> {
    control.check()?;
    if !(1..=1000).contains(&options.budget) {
        return Err("invalid solver budget".into());
    }
    if !options.tolerance.is_finite() || !(1e-9..=1e-3).contains(&options.tolerance) {
        return Err("invalid tolerance".into());
    }
    if !options.difference_step.is_finite() || !(0.000002..=0.01).contains(&options.difference_step)
    {
        return Err("invalid numerical difference step".into());
    }
    let mut model = Model::new(
        seed,
        stage,
        pool,
        options.method != Method::Constrained,
        x.len(),
        control,
    )?;
    model.optics = options.optics;
    model.active_penalty_derivative = options.active_penalty;
    model.difference_step = options.difference_step;
    model.decode(&x)?;
    if x.iter()
        .zip(&model.lower)
        .zip(&model.upper)
        .any(|((&v, &lo), &hi)| v < lo || v > hi)
    {
        return Err("initial parameters outside bounds".into());
    }
    let started = Instant::now();
    let (x, cost, status, success, iterations) = if options.method != Method::Constrained {
        let problem = RobustLeastSquares::new(&model, SoftL1Loss).with_scale(0.3);
        let result = {
            let solver = Trf::new()
                .with_absolute_scaled_gradient_tolerance(options.tolerance)
                .with_relative_cost_change_tolerance(options.tolerance)
                .with_relative_step_tolerance(options.tolerance);
            Executor::from_start(problem, solver, x)
                .max_iter(options.budget)
                .max_time(control.remaining()?)
                .run()?
        };
        let success = matches!(
            result.reason,
            TerminationReason::SolverConverged
                | TerminationReason::RelativeCostTolerance
                | TerminationReason::RelativeParamTolerance
                | TerminationReason::GradientTolerance
                | TerminationReason::ProjectedGradientTolerance
        );
        (
            result.param().clone(),
            result.cost(),
            format!("{:?}", result.reason),
            success,
            result.iter(),
        )
    } else {
        let result = Executor::from_start(
            Constrained(&model),
            Slsqp::new().with_absolute_accuracy_tolerance(1e-9),
            x,
        )
        .max_iter(options.budget)
        .max_cost_evals(options.budget)
        .max_time(control.remaining()?)
        .run()?;
        let success = result.reason == TerminationReason::SolverConverged;
        let status = if let Some(failure) = result.state.failure() {
            format!("{:?}: {failure:?}", result.reason)
        } else {
            format!("{:?}", result.reason)
        };
        (
            result.param().clone(),
            result.cost(),
            status,
            success,
            result.iter(),
        )
    };
    control.check()?;
    let solution = Solution {
        calibration: model.decode(&x)?,
        margin: model.margin(&x)?,
        sensor_boundary_active: model.sensor_boundary_active(&x)?,
        x,
        cost,
        status,
        success,
        iterations,
        elapsed: started.elapsed(),
        residuals: model.residuals.load(Ordering::Relaxed),
        jacobians: model.jacobians.load(Ordering::Relaxed),
        derivative_elapsed: Duration::from_nanos(model.derivative_nanos.load(Ordering::Relaxed)),
    };
    log::debug!(
        "Direct fit {:?}: {} iterations, {} residuals, {} Jacobians, {:.3}s (derivatives {:.3}s), cost={}",
        options.method,
        solution.iterations,
        solution.residuals,
        solution.jacobians,
        solution.elapsed.as_secs_f64(),
        solution.derivative_elapsed.as_secs_f64(),
        solution.cost
    );
    Ok(solution)
}

#[cfg(test)]
#[path = "solver_tests.rs"]
mod tests;
