//! Bounded image preparation for normalized direct-image fitting.

use super::kernel::{Pair, Stage, normalize, sample, validate_layout};
use super::matching::Image;
use super::solver::{Control, Method, Solution, SolveOptions, solve_native};
use reco_core::calibration::Calibration;

pub(crate) struct CoarseFit {
    pub(crate) stages: Vec<Solution>,
    pub(crate) training_rmse: f64,
}

impl CoarseFit {
    pub(super) fn into_calibration(mut self) -> Result<Calibration, String> {
        self.stages
            .pop()
            .map(|s| s.calibration)
            .ok_or("missing coarse fit".into())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
    Coarse,
    Refine,
    Constrained,
}

/// Explore a starting hypothesis with fixed support, keeping nonconvergence explicit.
pub(crate) fn coarse(
    seed: Calibration,
    images: &[Image; 2],
    patches: &[Vec<[f64; 2]>],
    estimate_optics: [bool; 2],
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<CoarseFit, String> {
    fit_pass(
        seed,
        images,
        patches,
        estimate_optics,
        pool,
        control,
        Pass::Coarse,
    )
}

/// Refine fixed source patches at successive blur scales with one shared deadline.
/// This is fitting only: convergence and a valid sensor domain are not confirmation.
pub(crate) fn refine(
    seed: Calibration,
    images: &[Image; 2],
    patches: &[Vec<[f64; 2]>],
    estimate_optics: [bool; 2],
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<Vec<Solution>, String> {
    let mut fit = fit_pass(
        seed,
        images,
        patches,
        estimate_optics,
        pool,
        control,
        Pass::Refine,
    )?;
    let final_broad = fit.stages.last().ok_or("missing fit")?;
    if final_broad.sensor_boundary_active {
        let repair = fit_pass(
            final_broad.calibration.clone(),
            images,
            patches,
            estimate_optics,
            pool,
            control,
            Pass::Constrained,
        )?;
        fit.stages.extend(repair.stages);
    }
    control.check()?;
    if fit.stages.last().is_none_or(|result| result.margin < 0.0) {
        return Err("refined optics do not cover both sensors".into());
    }
    Ok(fit.stages)
}

fn fit_pass(
    seed: Calibration,
    images: &[Image; 2],
    patches: &[Vec<[f64; 2]>],
    estimate_optics: [bool; 2],
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
    pass: Pass,
) -> Result<CoarseFit, String> {
    control.check()?;
    seed.validate().map_err(|e| e.to_string())?;
    if patches.len() < 15 {
        return Err("insufficient fitting patches".into());
    }
    let parameters = if estimate_optics.iter().any(|&v| v) {
        super::solver::N
    } else {
        7
    };
    let mut solutions = Vec::new();
    let method = if pass == Pass::Constrained {
        Method::Constrained
    } else {
        Method::LeastSquares
    };
    // Placement has already been initialized from bidirectional pixel matches at refinement.
    let sigmas: &[f64] = if pass == Pass::Coarse {
        &[8.0, 4.0, 2.0, 0.7]
    } else {
        &[2.0, 0.7]
    };
    let mut x = vec![0.0; parameters];
    let mut training_rmse = f64::INFINITY;
    for &sigma in sigmas {
        control.check()?;
        let stage = prepare(images, patches, sigma, || control.check())?;
        if solutions.is_empty() {
            let [width, height] = images[0].dimensions();
            let mut informative = 0;
            for patch in patches {
                control.check()?;
                let mut values: Vec<_> = patch
                    .iter()
                    .map(|&p| sample(images[0].values(), width, height, p))
                    .collect();
                if normalize(&mut values) > 0.012 {
                    informative += 1;
                }
            }
            if informative < 15 {
                return Err("insufficient textured fitting patches".into());
            }
        }
        let result = solve_native(
            seed.clone(),
            x,
            SolveOptions {
                method,
                budget: if method == Method::Constrained {
                    1000
                } else {
                    75
                },
                tolerance: 1e-5,
                difference_step: 0.00002,
                active_penalty: false,
                optics: estimate_optics,
            },
            &stage,
            pool,
            control,
        )?;
        if !result.success && pass != Pass::Coarse {
            return Err(format!(
                "direct refinement at blur {sigma} did not converge after {} iterations: {}",
                result.iterations, result.status
            ));
        }
        if sigma == 0.7 {
            let mut values = Vec::new();
            let lost_patches =
                stage.residual_checked(&result.calibration, &mut values, || control.check())?;
            training_rmse =
                (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt();
            if pass != Pass::Coarse && result.margin >= 0.0 {
                require_support(lost_patches)?;
            }
        }
        x = result.x.clone();
        solutions.push(result);
    }
    control.check()?;
    Ok(CoarseFit {
        stages: solutions,
        training_rmse,
    })
}

fn require_support(lost: usize) -> Result<(), String> {
    if lost != 0 {
        return Err("refined mapping lost fitting support".into());
    }
    Ok(())
}

// Reflection excludes the border pixel, so a linear ramp has symmetric support.
fn reflected(index: isize, length: usize) -> usize {
    let period = 2 * (length as isize - 1);
    let index = index.rem_euclid(period);
    index.min(period - index) as usize
}

/// Separable Gaussian with four-sigma support and reflected image boundaries.
/// At most two full-size scratch images are retained; cancellation is checked per row.
pub(crate) fn blur(
    image: &Image,
    sigma: f64,
    check: impl Fn() -> Result<(), String>,
) -> Result<Image, String> {
    check()?;
    if !sigma.is_finite() || !(0.1..=8.0).contains(&sigma) {
        return Err("invalid preparation blur scale".into());
    }
    let [width, height] = image.dimensions();
    if width * height > 4_000_000 {
        return Err("excessive preparation image".into());
    }
    let length = ((8.0 * sigma + 1.0).round_ties_even() as usize) | 1;
    let radius = (length / 2) as isize;
    let mut kernel: Vec<f64> = (-radius..=radius)
        .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
        .collect();
    let sum: f64 = kernel.iter().sum();
    for weight in &mut kernel {
        *weight /= sum;
    }
    let mut horizontal = vec![0.0; width * height];
    for (y, row) in horizontal.chunks_exact_mut(width).enumerate() {
        check()?;
        let input = &image.values()[y * width..(y + 1) * width];
        for (x, value) in row.iter_mut().enumerate() {
            *value = kernel
                .iter()
                .enumerate()
                .map(|(k, weight)| {
                    weight * input[reflected(x as isize + k as isize - radius, width)]
                })
                .sum();
        }
    }
    let mut values = vec![0.0; width * height];
    for (y, row) in values.chunks_exact_mut(width).enumerate() {
        check()?;
        for (k, weight) in kernel.iter().enumerate() {
            let source_y = reflected(y as isize + k as isize - radius, height);
            for (output, input) in row
                .iter_mut()
                .zip(&horizontal[source_y * width..(source_y + 1) * width])
            {
                *output += weight * input;
            }
        }
        // Positive normalized weights preserve [0, 1], apart from rounding at one.
        for value in row {
            *value = value.clamp(0.0, 1.0);
        }
    }
    check()?;
    Image::new(width, height, values)
}

/// Prepare one fitting stage from raw grayscale images and source pixel patches.
/// Each source pixel centre `(x, y)` maps to `((x + .5) / width, (y + .5) / height)`.
pub(crate) fn prepare(
    images: &[Image; 2],
    patches: &[Vec<[f64; 2]>],
    sigma: f64,
    check: impl Fn() -> Result<(), String>,
) -> Result<Stage, String> {
    check()?;
    let samples = patches.first().ok_or("missing fitting patches")?.len();
    let count = patches
        .len()
        .checked_mul(samples)
        .ok_or("excessive fitting patches")?;
    let [width, height] = images[1].dimensions();
    validate_layout(width, height, count, samples, 1)?;
    let [source_width, source_height] = images[0].dimensions();
    if patches.iter().any(|p| {
        p.len() != samples
            || p.iter().any(|point| {
                !point.iter().all(|v| v.is_finite())
                    || point[0] < 0.0
                    || point[1] < 0.0
                    || point[0] >= (source_width - 1) as f64
                    || point[1] >= (source_height - 1) as f64
            })
    }) {
        return Err("invalid fitting patch coordinates".into());
    }
    let left = blur(&images[0], sigma, &check)?;
    let mut target = Vec::with_capacity(count);
    let mut points = Vec::with_capacity(count);
    for patch in patches {
        check()?;
        let start = target.len();
        for &point in patch {
            target.push(sample(left.values(), source_width, source_height, point));
            points.push([
                (point[0] + 0.5) / source_width as f64,
                (point[1] + 0.5) / source_height as f64,
            ]);
        }
        normalize(&mut target[start..]);
    }
    drop(left);
    let right = blur(&images[1], sigma, &check)?.into_values();
    check()?;
    Stage::new(width, height, samples, points, vec![Pair { target, right }])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn reflection_handles_kernels_larger_than_the_image() {
        assert_eq!(
            (-5..=8).map(|i| reflected(i, 4)).collect::<Vec<_>>(),
            [1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2]
        );
        let image = Image::new(4, 4, vec![0.7; 16]).unwrap();
        for sigma in [0.7, 1.2, 2.0, 4.0, 8.0] {
            let result = blur(&image, sigma, || Ok(())).unwrap();
            assert!(result.values().iter().all(|v| (v - 0.7).abs() < 1e-14));
        }
    }

    #[test]
    fn preparation_rejects_bad_coordinates_and_invalid_scales() {
        let image = || Image::new(16, 16, vec![0.5; 256]).unwrap();
        let images = [image(), image()];
        for sigma in [f64::NAN, f64::INFINITY, -1.0, 0.0, 8.1] {
            assert!(blur(&images[0], sigma, || Ok(())).is_err());
        }
        for bad in [[f64::NAN, 5.0], [-0.1, 5.0], [15.0, 5.0]] {
            assert!(prepare(&images, &[vec![bad; 49]], 0.7, || Ok(())).is_err());
        }
        assert!(prepare(&images, &[], 0.7, || Ok(())).is_err());
        assert!(
            prepare(
                &images,
                &[vec![[5.0, 5.0]; 49], vec![[5.0, 5.0]; 48]],
                0.7,
                || Ok(())
            )
            .is_err()
        );
        assert_eq!(
            prepare(&images, &[vec![[5.0, 5.0]; 49]], 0.7, || Ok(()))
                .unwrap()
                .output_len(),
            49
        );
    }

    #[test]
    fn blur_cancels_mid_preparation_without_returning_an_image() {
        let image = Image::new(128, 128, vec![0.5; 128 * 128]).unwrap();
        let calls = Cell::new(0);
        let result = blur(&image, 8.0, || {
            calls.set(calls.get() + 1);
            if calls.get() == 4 {
                Err("cancelled".into())
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(message) if message == "cancelled"));
        assert_eq!(calls.get(), 4);
        assert!(blur(&image, 0.7, || Ok(())).is_ok());
    }

    #[test]
    fn refinement_rejects_cancelled_expired_or_untextured_input() {
        use reco_core::calibration::{Framing, Lens};
        use reco_core::projection::LShape;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, Instant};
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
            deadline: Instant::now() + Duration::from_secs(10),
        };
        let unsupported = Stage::new(
            32,
            32,
            49,
            vec![[0.0, 0.0]; 49],
            vec![Pair {
                target: vec![0.0; 49],
                right: vec![0.5; 1024],
            }],
        )
        .unwrap();
        assert!(
            matches!(require_support(unsupported.residual_checked(&seed, &mut Vec::new(), || control.check()).unwrap()),
            Err(e) if e == "refined mapping lost fitting support")
        );
        let images = [
            Image::new(32, 32, vec![0.5; 1024]).unwrap(),
            Image::new(32, 32, vec![0.5; 1024]).unwrap(),
        ];
        let patches = vec![vec![[12.0, 12.0]; 49]; 15];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let interrupted = AtomicBool::new(true);
        let mut control = Control {
            interrupted: &interrupted,
            deadline: Instant::now() + Duration::from_secs(1),
        };
        let run = |control: &Control<'_>| {
            refine(seed.clone(), &images, &patches, [false; 2], &pool, control)
        };
        assert_eq!(run(&control).unwrap_err(), "calibration cancelled");
        interrupted.store(false, Ordering::Relaxed);
        assert_eq!(
            run(&control).unwrap_err(),
            "insufficient textured fitting patches"
        );
        control.deadline = Instant::now();
        assert_eq!(run(&control).unwrap_err(), "calibration time limit reached");
    }
}
