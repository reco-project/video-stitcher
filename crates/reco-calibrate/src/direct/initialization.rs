//! Bounded placement fitting from measured correspondences.

use super::solver::{Control, decode_parameters, parameter_bounds};
use basin::{
    BoxConstraints, CostFunction, DenseMatrix, Executor, Jacobian, Residual, RobustLeastSquares,
    SoftL1Loss, TerminationReason, Trf,
};
use reco_core::calibration::Calibration;
#[cfg(test)]
use std::time::Duration;
use std::time::Instant;

#[cfg(test)]
#[path = "initialization_diagnostics.rs"]
pub(crate) mod diagnostics;

const N: usize = 7;

struct Placement<'a> {
    seed: Calibration,
    source: Vec<[f64; 2]>,
    target: &'a [[f64; 2]],
    target_dimensions: [f64; 2],
    lower: Vec<f64>,
    upper: Vec<f64>,
    control: &'a Control<'a>,
}

impl Placement<'_> {
    fn values(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        self.control.check()?;
        let cal = decode_parameters(&self.seed, x)?;
        cal.validate().map_err(|e| e.to_string())?;
        let mapping = cal.source_correspondence(0, 1).map_err(|e| e.to_string())?;
        let mut values = Vec::with_capacity(self.source.len() * 2 + N);
        for (source, target) in self.source.iter().zip(self.target) {
            self.control.check()?;
            let error = mapping.map(*source).map(|uv| {
                [
                    uv[0] * self.target_dimensions[0] - 0.5 - target[0],
                    uv[1] * self.target_dimensions[1] - 0.5 - target[1],
                ]
            });
            values.extend(
                error
                    .filter(|e| e.iter().all(|v| v.is_finite()))
                    .unwrap_or([1000.0; 2]),
            );
        }
        values.extend(
            x.iter()
                .map(|x| 0.03 * (self.source.len() as f64).sqrt() * x),
        );
        Ok(values)
    }
}

impl CostFunction for &Placement<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = String;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, String> {
        Ok(self.values(x)?.iter().map(|v| v * v * 0.5).sum())
    }
}

impl Residual for &Placement<'_> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = String;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, String> {
        self.values(x)
    }
}

impl Jacobian for &Placement<'_> {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, String> {
        let mut columns = Vec::with_capacity(N);
        for i in 0..N {
            let (mut hi, mut lo) = (x.clone(), x.clone());
            hi[i] = (hi[i] + 0.001).min(self.upper[i]);
            lo[i] = (lo[i] - 0.001).max(self.lower[i]);
            let delta = hi[i] - lo[i];
            if delta <= 0.0 {
                return Err("invalid initializer difference interval".into());
            }
            columns.push(
                self.values(&hi)?
                    .iter()
                    .zip(self.values(&lo)?)
                    .map(|(a, b)| (a - b) / delta)
                    .collect::<Vec<_>>(),
            );
        }
        Ok(DenseMatrix::from_fn(columns[0].len(), N, |r, c| {
            columns[c][r]
        }))
    }
}

impl BoxConstraints for &Placement<'_> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[derive(Debug)]
#[cfg(test)]
pub(crate) struct InitializationStage {
    pub(crate) scale: f64,
    pub(crate) status: String,
    pub(crate) iterations: u64,
    pub(crate) cost: f64,
    pub(crate) calibration: Calibration,
}

#[derive(Debug)]
pub(crate) struct Initialization {
    pub(crate) calibration: Calibration,
    #[cfg(test)]
    pub(crate) x: Vec<f64>,
    #[cfg(test)]
    pub(crate) stages: Vec<InitializationStage>,
    #[cfg(test)]
    pub(crate) elapsed: Duration,
}

/// Refine placement from measured image coordinates, keeping supplied optics unchanged.
pub(crate) fn initialize_native(
    seed: Calibration,
    source: &[[f64; 2]],
    target: &[[f64; 2]],
    dimensions: [[usize; 2]; 2],
    control: &Control<'_>,
) -> Result<Initialization, String> {
    let started = Instant::now();
    control.check()?;
    if !(15..=2000).contains(&source.len())
        || source.len() != target.len()
        || dimensions
            .iter()
            .flatten()
            .any(|v| !(2..=16_384).contains(v))
    {
        return Err("invalid initialization layout".into());
    }
    for (points, dims) in [(source, dimensions[0]), (target, dimensions[1])] {
        if points.iter().any(|p| {
            p.iter()
                .zip(dims)
                .any(|(&v, d)| !v.is_finite() || v < 0.0 || v >= d as f64)
        }) {
            return Err("invalid initialization coordinates".into());
        }
    }
    seed.validate().map_err(|e| e.to_string())?;
    if seed.lenses.len() != 2 || seed.projection.l_shape().is_none() {
        return Err("initializer requires two-source L-shape".into());
    }
    let (lower, upper) = parameter_bounds(&seed, N, [0.02, 0.98])?;
    let model = Placement {
        seed,
        source: source
            .iter()
            .map(|p| {
                [
                    (p[0] + 0.5) / dimensions[0][0] as f64,
                    (p[1] + 0.5) / dimensions[0][1] as f64,
                ]
            })
            .collect(),
        target,
        target_dimensions: dimensions[1].map(|v| v as f64),
        lower,
        upper,
        control,
    };
    let mut x = vec![0.0; N];
    #[cfg(test)]
    let mut stages = Vec::new();
    for scale in [4.0, 1.0] {
        let result = Executor::from_start(
            RobustLeastSquares::new(&model, SoftL1Loss).with_scale(scale),
            Trf::new()
                .with_absolute_scaled_gradient_tolerance(1e-6)
                .with_relative_cost_change_tolerance(1e-7)
                .with_relative_step_tolerance(1e-7),
            x,
        )
        .max_iter(100)
        .max_time(control.remaining()?)
        .run()?;
        model.control.check()?;
        if !matches!(
            result.reason,
            TerminationReason::SolverConverged
                | TerminationReason::RelativeCostTolerance
                | TerminationReason::RelativeParamTolerance
                | TerminationReason::GradientTolerance
                | TerminationReason::ProjectedGradientTolerance
        ) {
            return Err(format!(
                "placement initialization at scale {scale} did not converge after {} iterations: {:?}",
                result.iter(),
                result.reason
            ));
        }
        x = result.param().clone();
        #[cfg(test)]
        stages.push(InitializationStage {
            scale,
            status: format!("{:?}", result.reason),
            iterations: result.iter(),
            cost: result.cost(),
            calibration: decode_parameters(&model.seed, &x)?,
        });
    }
    control.check()?;
    log::debug!(
        "Placement initialization: {:.3}s",
        started.elapsed().as_secs_f64()
    );
    Ok(Initialization {
        calibration: decode_parameters(&model.seed, &x)?,
        #[cfg(test)]
        x,
        #[cfg(test)]
        stages,
        #[cfg(test)]
        elapsed: started.elapsed(),
    })
}

#[cfg(test)]
mod tests {
    use super::diagnostics::initialize;
    use super::*;
    use reco_core::calibration::{Framing, Lens};
    use reco_core::projection::LShape;
    use serde_json::json;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn recovers_known_mapping_without_touching_optics_or_other_fields() {
        let lens = Lens::fisheye(640, 480, 384.0, 384.0, 320.0, 240.0, [0.0; 4]);
        let calibration = Calibration::new(
            vec![lens.clone(), lens],
            LShape {
                intersect: 0.59,
                x_ty: 0.0,
                x_rz: 0.0,
                x_rx: 0.0,
                z_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            Framing {
                axis_offset: 0.27,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        let mut seed = serde_json::to_value(calibration).unwrap();
        seed["sync_offset"] = json!(-17);
        let truth = decode_parameters(
            &serde_json::from_value(seed.clone()).unwrap(),
            &[0.12, -0.03, 0.04, 0.02, -0.02, 0.01, 0.02],
        )
        .unwrap();
        let mapping = truth.source_correspondence(0, 1).unwrap();
        let mut source = Vec::new();
        let mut target = Vec::new();
        for y in 0..10 {
            for x in 0..10 {
                let point = [0.6 + x as f64 * 0.03, 0.15 + y as f64 * 0.07];
                if let Some(uv) = mapping.map(point)
                    && uv.iter().all(|v| (0.01..0.99).contains(v))
                {
                    source.push([point[0] * 640.0 - 0.5, point[1] * 480.0 - 0.5]);
                    target.push([uv[0] * 1280.0 - 0.5, uv[1] * 960.0 - 0.5]);
                }
            }
        }
        assert!(source.len() >= 15);
        let request = json!({"seed":seed,"source":source,"target":target,"dimensions":[[640,480],[1280,960]]});
        let result = initialize(&request, &AtomicBool::new(false)).unwrap();
        assert_eq!(result["calibration"]["lenses"], seed["lenses"]);
        assert_eq!(result["calibration"]["sync_offset"], seed["sync_offset"]);
        assert_eq!(
            result["calibration"]["framing"]["roll"],
            seed["framing"]["roll"]
        );
        let fitted: Calibration = serde_json::from_value(result["calibration"].clone()).unwrap();
        let fitted_map = fitted.source_correspondence(0, 1).unwrap();
        let mut maximum = 0.0_f64;
        for (source, target) in source.iter().zip(target) {
            let uv = fitted_map
                .map([(source[0] + 0.5) / 640.0, (source[1] + 0.5) / 480.0])
                .unwrap();
            maximum = maximum
                .max((uv[0] * 1280.0 - 0.5 - target[0]).hypot(uv[1] * 960.0 - 0.5 - target[1]));
        }
        assert!(maximum < 0.02, "known mapping residual: {maximum}");
        let mut malformed = request.clone();
        malformed["source"][0][0] = json!(-1.0);
        assert_eq!(
            initialize(&malformed, &AtomicBool::new(false)).unwrap_err(),
            "invalid initialization coordinates"
        );
    }

    #[test]
    fn rejects_cancelled_or_invalid_input_before_mapping() {
        assert_eq!(
            initialize(&json!({}), &AtomicBool::new(true)).unwrap_err(),
            "calibration cancelled"
        );
        assert_eq!(
            initialize(&json!({"time_limit_ms":0}), &AtomicBool::new(false)).unwrap_err(),
            "invalid initialization time limit"
        );
        let request = json!({"source":vec![[1.0,1.0];14],"target":vec![[1.0,1.0];14],"dimensions":[[640,480],[1280,960]]});
        assert_eq!(
            initialize(&request, &AtomicBool::new(false)).unwrap_err(),
            "invalid initialization layout"
        );
    }

    #[test]
    fn transport_preserves_supplied_lens_coefficient_bits() {
        let expected = -0.0009822150067739363_f64;
        let decoded: f64 = serde_json::from_str("-0.0009822150067739363").unwrap();
        assert_eq!(decoded.to_bits(), expected.to_bits());
    }

    #[test]
    fn typed_initializer_obeys_the_callers_expired_deadline() {
        let lens = Lens::fisheye(640, 480, 384.0, 384.0, 320.0, 240.0, [0.0; 4]);
        let seed = Calibration::new(
            vec![lens.clone(), lens],
            LShape {
                intersect: 0.59,
                x_ty: 0.0,
                x_rz: 0.0,
                x_rx: 0.0,
                z_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            Framing {
                axis_offset: 0.27,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        let interrupted = AtomicBool::new(false);
        let control = Control {
            interrupted: &interrupted,
            deadline: Instant::now(),
        };
        assert_eq!(
            initialize_native(seed, &[], &[], [[640, 480]; 2], &control).unwrap_err(),
            "calibration time limit reached"
        );
    }
}
