//! Bounded, bidirectional pixel correspondence using prepared core geometry.

use super::kernel::{normalize, sample, sample_with_gradient};
use basin::{
    BoxConstraints, CostFunction, DenseMatrix, Executor, Jacobian, Residual, TerminationReason, Trf,
};
use rayon::prelude::*;
use reco_core::{calibration::Calibration, projection::SourceCorrespondence};

pub(crate) struct Image<T = f64> {
    width: usize,
    height: usize,
    values: Vec<T>,
}

impl<T: Copy + Into<f64>> Image<T> {
    pub(crate) fn dimensions(&self) -> [usize; 2] {
        [self.width, self.height]
    }

    pub(crate) fn values(&self) -> &[T] {
        &self.values
    }

    pub(crate) fn into_values(self) -> Vec<T> {
        self.values
    }

    pub(crate) fn valid_layout(width: usize, height: usize) -> bool {
        width >= 4
            && height >= 4
            && width.checked_mul(height).is_some_and(|n| {
                n <= 16_000_000
                    && n.checked_mul(std::mem::size_of::<T>())
                        .is_some_and(|b| b <= 64_000_000)
            })
    }

    pub(crate) fn new(width: usize, height: usize, values: Vec<T>) -> Result<Self, String> {
        if !Self::valid_layout(width, height)
            || width * height != values.len()
            || values.iter().any(|x| {
                let x: f64 = (*x).into();
                !x.is_finite() || !(0.0..=1.0).contains(&x)
            })
        {
            return Err("invalid matching image".into());
        }
        Ok(Self {
            width,
            height,
            values,
        })
    }

    fn patch(&self, points: &[[f64; 2]], shift: [f64; 2]) -> Vec<f64> {
        points
            .iter()
            .map(|p| {
                sample(
                    &self.values,
                    self.width,
                    self.height,
                    [p[0] + shift[0], p[1] + shift[1]],
                )
            })
            .collect()
    }

    fn interior(&self, point: [f64; 2]) -> bool {
        point[0] > 1.0
            && point[1] > 1.0
            && point[0] < self.width as f64 - 2.0
            && point[1] < self.height as f64 - 2.0
    }
}

fn unit(values: &mut [f64]) -> f64 {
    let std = normalize(values);
    let norm = (values.len() as f64).sqrt();
    for value in values {
        *value /= norm;
    }
    std
}

struct Translation<'a, C, T> {
    image: &'a Image<T>,
    points: &'a [[f64; 2]],
    template: &'a [f64],
    lower: Vec<f64>,
    upper: Vec<f64>,
    check: &'a C,
}

impl<C: Fn() -> Result<(), String>, T: Copy + Into<f64>> CostFunction for &Translation<'_, C, T> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = String;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, String> {
        Ok(self.residual(x)?.iter().map(|r| 0.5 * r * r).sum())
    }
}

impl<C: Fn() -> Result<(), String>, T: Copy + Into<f64>> Residual for &Translation<'_, C, T> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = String;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, String> {
        (self.check)()?;
        let mut values = self.image.patch(self.points, [x[0], x[1]]);
        unit(&mut values);
        for (value, template) in values.iter_mut().zip(self.template) {
            *value -= template;
        }
        Ok(values)
    }
}

impl<C: Fn() -> Result<(), String>, T: Copy + Into<f64>> Jacobian for &Translation<'_, C, T> {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, String> {
        (self.check)()?;
        let (mut values, gradients): (Vec<_>, Vec<_>) = self
            .points
            .iter()
            .map(|p| {
                sample_with_gradient(
                    &self.image.values,
                    self.image.width,
                    self.image.height,
                    [p[0] + x[0], p[1] + x[1]],
                )
            })
            .unzip();
        let norm = unit(&mut values) * (values.len() as f64).sqrt();
        if norm == 0.0 {
            return Ok(DenseMatrix::from_fn(values.len(), 2, |_, _| 0.0));
        }
        let means: [f64; 2] = std::array::from_fn(|axis| {
            gradients.iter().map(|g| g[axis]).sum::<f64>() / values.len() as f64
        });
        let along: [f64; 2] = std::array::from_fn(|axis| {
            values
                .iter()
                .zip(&gradients)
                .map(|(v, g)| v * g[axis])
                .sum()
        });
        Ok(DenseMatrix::from_fn(values.len(), 2, |i, axis| {
            (gradients[i][axis] - means[axis] - values[i] * along[axis]) / norm
        }))
    }
}

impl<C: Fn() -> Result<(), String>, T: Copy + Into<f64>> BoxConstraints for &Translation<'_, C, T> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct Measurement {
    pub target: Option<[f64; 2]>,
    pub forward: bool,
    pub accepted: bool,
    pub peak: f64,
    pub margin: f64,
    pub curvature: f64,
    pub status: Option<String>,
}

fn correlate<C: Fn() -> Result<(), String>, T: Copy + Into<f64>>(
    left: &Image<T>,
    right: &Image<T>,
    points: &[[f64; 2]],
    mapped: &[[f64; 2]],
    check: &C,
) -> Result<Measurement, String> {
    let mut template = left.patch(points, [0.0; 2]);
    if unit(&mut template) <= 0.025 {
        return Ok(Measurement::default());
    }
    let mut shift = [0.0; 2];
    let mut result = Measurement::default();
    let mut boundary = false;
    for (radius, step) in [(32i32, 4usize), (4, 1)] {
        let mut scores: Vec<([f64; 2], f64)> = Vec::new();
        let mut best = 0;
        for dy in (-radius..=radius).step_by(step) {
            for dx in (-radius..=radius).step_by(step) {
                check()?;
                let offset = [f64::from(dx), f64::from(dy)];
                let mut values = right.patch(mapped, [shift[0] + offset[0], shift[1] + offset[1]]);
                unit(&mut values);
                let score: f64 = values.iter().zip(&template).map(|(a, b)| a * b).sum();
                if scores.is_empty() || score > scores[best].1 {
                    best = scores.len();
                }
                scores.push((offset, score));
            }
        }
        let (offset, score) = scores[best];
        if step == 4 {
            boundary = offset.iter().any(|x| x.abs() == f64::from(radius));
            let second = scores
                .iter()
                .filter(|(p, _)| (p[0] - offset[0]).powi(2) + (p[1] - offset[1]).powi(2) >= 64.0)
                .map(|(_, s)| *s)
                .fold(-1.0_f64, f64::max);
            result.margin = score - second;
        }
        result.peak = score;
        shift = [shift[0] + offset[0], shift[1] + offset[1]];
    }
    let middle = mapped.len() / 2;
    result.target = Some([mapped[middle][0] + shift[0], mapped[middle][1] + shift[1]]);
    if boundary || result.peak <= 0.85 || result.margin <= 0.025 {
        return Ok(result);
    }
    let problem = Translation {
        image: right,
        points: mapped,
        template: &template,
        lower: shift.iter().map(|x| x - 1.0).collect(),
        upper: shift.iter().map(|x| x + 1.0).collect(),
        check,
    };
    let solver = Trf::new()
        .with_absolute_scaled_gradient_tolerance(1e-7)
        .with_relative_cost_change_tolerance(1e-7)
        .with_relative_step_tolerance(1e-7);
    let fitted = Executor::from_start(&problem, solver, shift.to_vec())
        .max_iter(25)
        .run()?;
    result.status = Some(format!("{:?}", fitted.reason));
    let converged = matches!(
        fitted.reason,
        TerminationReason::SolverConverged
            | TerminationReason::RelativeCostTolerance
            | TerminationReason::RelativeParamTolerance
            | TerminationReason::GradientTolerance
            | TerminationReason::ProjectedGradientTolerance
    );
    let x = fitted.param();
    result.target = Some([mapped[middle][0] + x[0], mapped[middle][1] + x[1]]);
    let jac = (&problem).jacobian(x)?;
    let mut gram = nalgebra::Matrix2::<f64>::zeros();
    for i in 0..mapped.len() {
        let row = nalgebra::Vector2::new(jac.get(i, 0), jac.get(i, 1));
        gram += row * row.transpose();
    }
    let eigen = gram.symmetric_eigen().eigenvalues;
    result.curvature = (eigen.min().max(0.0) / eigen.max().max(1e-18)).sqrt();
    result.forward = converged
        && result.curvature > 0.12
        && x.iter()
            .zip(shift)
            .all(|(x, start)| (x - start).abs() < 1.0 - 1e-7)
        && mapped
            .iter()
            .all(|p| right.interior([p[0] + x[0], p[1] + x[1]]));
    Ok(result)
}

fn map_patch<T>(
    map: &SourceCorrespondence,
    points: &[[f64; 2]],
    source: &Image<T>,
    target: &Image<T>,
) -> Option<Vec<[f64; 2]>> {
    points
        .iter()
        .map(|p| {
            map.map([
                (p[0] + 0.5) / source.width as f64,
                (p[1] + 0.5) / source.height as f64,
            ])
            .map(|q| {
                [
                    q[0] * target.width as f64 - 0.5,
                    q[1] * target.height as f64 - 0.5,
                ]
            })
        })
        .collect()
}

pub(crate) fn measure<T: Copy + Into<f64> + Sync>(
    calibration: &Calibration,
    images: &[Image<T>; 2],
    patches: &[Vec<[f64; 2]>],
    pool: &rayon::ThreadPool,
    check: &(impl Fn() -> Result<(), String> + Sync),
) -> Result<Vec<Measurement>, String> {
    check()?;
    calibration.validate().map_err(|e| e.to_string())?;
    if patches.len() > 2000
        || patches.iter().map(Vec::len).sum::<usize>() > 200_000
        || patches.iter().any(|p| {
            !(3..=4095).contains(&p.len())
                || p.len() % 2 != 1
                || p.iter().any(|p| {
                    p.iter().any(|x| !x.is_finite())
                        || p[0] < 0.0
                        || p[1] < 0.0
                        || p[0] >= (images[0].width - 1) as f64
                        || p[1] >= (images[0].height - 1) as f64
                })
        })
    {
        return Err("invalid matching patches".into());
    }
    let forward = calibration
        .source_correspondence(0, 1)
        .map_err(|e| e.to_string())?;
    let reverse = calibration
        .source_correspondence(1, 0)
        .map_err(|e| e.to_string())?;
    let measured = pool.install(|| {
        patches
            .par_iter()
            .map(|points| {
                check()?;
                let Some(mapped) = map_patch(&forward, points, &images[0], &images[1]) else {
                    return Ok(Measurement::default());
                };
                let mut result = correlate(&images[0], &images[1], points, &mapped, check)?;
                if result.forward {
                    let target = result.target.ok_or("missing refined target")?;
                    let middle = points.len() / 2;
                    let backwards: Vec<_> = points
                        .iter()
                        .map(|p| {
                            [
                                target[0] + p[0] - points[middle][0],
                                target[1] + p[1] - points[middle][1],
                            ]
                        })
                        .collect();
                    if backwards.iter().all(|p| images[1].interior(*p))
                        && let Some(mapped) =
                            map_patch(&reverse, &backwards, &images[1], &images[0])
                    {
                        let back = correlate(&images[1], &images[0], &backwards, &mapped, check)?;
                        if let Some(returned) = back.target {
                            result.accepted = back.forward
                                && (returned[0] - points[middle][0])
                                    .hypot(returned[1] - points[middle][1])
                                    < 1.0;
                        }
                    }
                }
                Ok(result)
            })
            .collect()
    });
    check()?;
    measured
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(shift: [f64; 2], stripes: bool) -> Image {
        let mut values = Vec::new();
        for y in 0..160 {
            for x in 0..160 {
                let x = x as f64 - shift[0];
                let y = y as f64 - shift[1];
                let value = if stripes {
                    0.2 * (x * std::f64::consts::PI / 4.0).cos()
                } else {
                    let mut total = 0.0;
                    let mut weights = 0.0;
                    for i in 1..=24 {
                        let i = i as f64;
                        let angle = i * 2.399963;
                        let frequency = 0.11 + i * 0.023;
                        let weight = 1.0 / i.sqrt();
                        total += weight
                            * (frequency * (x * angle.cos() + y * angle.sin()) + i * 0.73).sin();
                        weights += weight;
                    }
                    0.4 * total / weights
                };
                values.push(0.5 + value);
            }
        }
        Image::new(160, 160, values).unwrap()
    }

    #[test]
    fn known_subpixel_translations_and_ambiguous_inputs() {
        let left = image([0.0; 2], false);
        let points: Vec<_> = (-16..=16)
            .step_by(2)
            .flat_map(|y| {
                (-16..=16)
                    .step_by(2)
                    .map(move |x| [80.0 + x as f64, 80.0 + y as f64])
            })
            .collect();
        for shift in [[5.0, -3.0], [2.3, -1.6], [-3.2, 2.4]] {
            let right = image(shift, false);
            let result = correlate(&left, &right, &points, &points, &|| Ok(())).unwrap();
            assert!(result.forward, "{result:?}");
            let target = result.target.unwrap();
            assert!(
                (target[0] - 80.0 - shift[0]).hypot(target[1] - 80.0 - shift[1]) < 0.1,
                "{result:?}"
            );
        }
        let flat = Image::new(160, 160, vec![0.5; 160 * 160]).unwrap();
        assert!(
            !correlate(&flat, &flat, &points, &points, &|| Ok(()))
                .unwrap()
                .forward
        );
        let stripes = image([0.0; 2], true);
        assert!(
            !correlate(&stripes, &stripes, &points, &points, &|| Ok(()))
                .unwrap()
                .forward
        );
    }

    #[test]
    fn cancellation_interrupts_the_search() {
        let left = image([0.0; 2], false);
        let points: Vec<_> = (-16..=16)
            .step_by(2)
            .flat_map(|y| {
                (-16..=16)
                    .step_by(2)
                    .map(move |x| [80.0 + x as f64, 80.0 + y as f64])
            })
            .collect();
        let count = std::cell::Cell::new(0);
        let result = correlate(&left, &left, &points, &points, &|| {
            count.set(count.get() + 1);
            if count.get() == 10 {
                Err("cancelled".into())
            } else {
                Ok(())
            }
        });
        assert_eq!(result.unwrap_err(), "cancelled");
        assert_eq!(count.get(), 10);
    }

    #[test]
    fn analytic_normalized_patch_derivative_matches_finite_differences() {
        let image = image([0.0; 2], false);
        let points: Vec<_> = (0..49)
            .map(|i| [60.17 + (i % 7) as f64 * 2.0, 70.29 + (i / 7) as f64 * 2.0])
            .collect();
        let mut template = image.patch(&points, [0.0; 2]);
        unit(&mut template);
        let problem = Translation {
            image: &image,
            points: &points,
            template: &template,
            lower: vec![-1.0; 2],
            upper: vec![1.0; 2],
            check: &|| Ok(()),
        };
        let x = vec![0.03, -0.06];
        let jac = (&problem).jacobian(&x).unwrap();
        for axis in 0..2 {
            let (mut hi, mut lo) = (x.clone(), x.clone());
            hi[axis] += 1e-5;
            lo[axis] -= 1e-5;
            let a = (&problem).residual(&hi).unwrap();
            let b = (&problem).residual(&lo).unwrap();
            for i in 0..points.len() {
                assert!((jac.get(i, axis) - (a[i] - b[i]) / 2e-5).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn gain_offset_and_border_checks_survive_native_refinement() {
        let left = image([0.0; 2], false);
        let right = image([2.3, -1.6], false);
        let changed = Image::new(
            right.width,
            right.height,
            right.values.iter().map(|v| v * 0.65 + 0.12).collect(),
        )
        .unwrap();
        let points: Vec<_> = (-16..=16)
            .step_by(2)
            .flat_map(|y| {
                (-16..=16)
                    .step_by(2)
                    .map(move |x| [80.0 + x as f64, 80.0 + y as f64])
            })
            .collect();
        let a = correlate(&left, &right, &points, &points, &|| Ok(())).unwrap();
        let b = correlate(&left, &changed, &points, &points, &|| Ok(())).unwrap();
        assert!(a.forward && b.forward);
        for (a, b) in a.target.unwrap().iter().zip(b.target.unwrap()) {
            assert!((a - b).abs() < 1e-9);
        }
        let border: Vec<_> = points.iter().map(|p| [p[0] - 60.0, p[1]]).collect();
        let outside = image([-10.0, 0.0], false);
        assert!(
            !correlate(&left, &outside, &border, &border, &|| Ok(()))
                .unwrap()
                .forward
        );
    }

    #[test]
    fn invalid_images_are_rejected_before_sampling() {
        assert!(Image::<f64>::new(usize::MAX, 4, vec![]).is_err());
        assert!(Image::new(3, 4, vec![0.0; 12]).is_err());
        assert!(Image::new(4, 4, vec![0.0; 15]).is_err());
        for value in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
            assert!(Image::new(4, 4, vec![value; 16]).is_err());
        }
    }

    #[test]
    fn compact_images_preserve_interpolation_and_gradients_exactly() {
        let pixels: Vec<f32> = (0..32 * 24)
            .map(|i| ((i * 139 % 255) as f32) / 255.0)
            .collect();
        let wide: Vec<f64> = pixels.iter().copied().map(f64::from).collect();
        for j in -4..105 {
            for i in -4..135 {
                let point = [i as f64 * 0.247 + 0.037, j as f64 * 0.239 + 0.043];
                assert_eq!(
                    sample_with_gradient(&pixels, 32, 24, point),
                    sample_with_gradient(&wide, 32, 24, point)
                );
            }
        }
    }

    #[test]
    fn compact_images_preserve_refined_measurements_exactly() {
        let compact = |shift| {
            let image = image(shift, false);
            Image::new(
                image.width,
                image.height,
                image.values.iter().map(|v| *v as f32).collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let widen = |im: &Image<f32>| {
            Image::new(
                im.width,
                im.height,
                im.values.iter().copied().map(f64::from).collect(),
            )
            .unwrap()
        };
        let a = compact([0.0; 2]);
        let b = compact([2.3, -1.6]);
        let points: Vec<_> = (-16..=16)
            .step_by(2)
            .flat_map(|y| {
                (-16..=16)
                    .step_by(2)
                    .map(move |x| [80.0 + x as f64, 80.0 + y as f64])
            })
            .collect();
        let result = correlate(&a, &b, &points, &points, &|| Ok(())).unwrap();
        assert!(result.forward);
        assert_eq!(
            result,
            correlate(&widen(&a), &widen(&b), &points, &points, &|| Ok(())).unwrap()
        );
    }

    #[test]
    fn image_storage_is_bounded_before_allocation() {
        assert!(Image::<f32>::valid_layout(5312, 2988));
        assert!(!Image::<f64>::valid_layout(5312, 2988));
        assert!(Image::<f32>::valid_layout(4000, 4000));
        assert!(!Image::<f32>::valid_layout(4001, 4000));
        assert!(!Image::<f32>::valid_layout(usize::MAX, 4));
        assert!(!Image::<f32>::valid_layout(3, 16));
        for value in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
            assert!(Image::new(4, 4, vec![value; 16]).is_err());
        }
    }
}
