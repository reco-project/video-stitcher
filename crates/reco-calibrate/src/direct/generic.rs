//! Training-only generic optical starts; never a reusable lens-profile estimate.

use super::features::Correspondences;
use super::initialization::initialize_native;
use super::matching::Image;
use super::preparation::{CoarseFit, coarse};
use super::solver::Control;
use super::support::{Patch, select_generic};
use reco_core::calibration::Calibration;

const FOCALS: [f64; 3] = [0.4, 0.65, 1.0];

pub(crate) struct Hypothesis {
    pub(crate) focal: f64,
    pub(crate) fit: Result<CoarseFit, String>,
}

pub(crate) struct GenericStart {
    pub(crate) selected: usize,
    pub(crate) hypotheses: Vec<Hypothesis>,
}

fn guess(
    template: &Calibration,
    focal: f64,
    distortion: [f64; 4],
    optics: [bool; 2],
) -> Result<Calibration, String> {
    let mut seed = template.clone();
    if seed.lenses.len() != 2 {
        return Err("generic start requires two source sensors".into());
    }
    for (lens, free) in seed.lenses.iter_mut().zip(optics) {
        if !free {
            continue;
        }
        lens.fx = lens.width as f64 * focal;
        lens.fy = lens.fx;
        lens.cx = lens.width as f64 * 0.5;
        lens.cy = lens.height as f64 * 0.5;
        lens.distortion = distortion;
        lens.correction = 1.0;
    }
    let shape = seed
        .projection
        .l_shape_mut()
        .ok_or("generic start requires L-shape geometry")?;
    shape.intersect = 0.6;
    shape.x_ty = 0.0;
    shape.x_rx = 0.0;
    shape.x_rz = 0.0;
    shape.z_rx = 0.0;
    shape.z_rz = 0.0;
    seed.framing.axis_offset = 0.23;
    seed.framing.tilt = 0.0;
    seed.framing.roll = 0.0;
    seed.validate().map_err(|e| e.to_string())?;
    Ok(seed)
}

/// Build the demonstrated common image support without using supplied optics/placement.
pub(crate) fn support(
    template: &Calibration,
    source: &Image,
    optics: [bool; 2],
    control: &Control<'_>,
) -> Result<Vec<Patch>, String> {
    control.check()?;
    let mut seeds = Vec::new();
    for distortion in [
        [0.0; 4],
        [1.0 / 3.0, 2.0 / 15.0, 17.0 / 315.0, 62.0 / 2835.0],
    ] {
        for focal in FOCALS {
            seeds.push(guess(template, focal, distortion, optics)?);
        }
    }
    select_generic(&seeds, source, || control.check())
}

/// Choose a starting hypothesis from training residuals, not confirmation evidence.
/// Only optical models selected by the per-camera mask may change.
pub(crate) fn discover(
    template: &Calibration,
    images: &[Image; 2],
    features: Option<&Correspondences>,
    optics: [bool; 2],
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<GenericStart, String> {
    let patches: Vec<_> = support(template, &images[0], optics, control)?
        .into_iter()
        .filter(|p| p.fitting)
        .map(|p| p.pixels)
        .collect();
    if patches.len() < 15 {
        return Err("insufficient textured generic support".into());
    }
    let mut hypotheses = Vec::new();
    for focal in FOCALS {
        control.check()?;
        let fit = (|| {
            let mut seed = guess(template, focal, [0.0; 4], optics)?;
            if let Some(features) = features {
                seed = initialize_native(
                    seed,
                    &features.source,
                    &features.target,
                    [images[0].dimensions(), images[1].dimensions()],
                    control,
                )?
                .calibration;
            }
            coarse(seed, images, &patches, optics, pool, control)
        })();
        // Cancellation/expiry belongs to the whole call, not just one hypothesis.
        control.check()?;
        hypotheses.push(Hypothesis { focal, fit });
    }
    let selected = select_best(&hypotheses)?;
    Ok(GenericStart {
        selected,
        hypotheses,
    })
}

fn select_best(hypotheses: &[Hypothesis]) -> Result<usize, String> {
    hypotheses
        .iter()
        .enumerate()
        .filter_map(|(index, hypothesis)| {
            let fit = hypothesis.fit.as_ref().ok()?;
            fit.stages.last().filter(|stage| stage.success)?;
            fit.training_rmse
                .is_finite()
                .then_some((index, fit.training_rmse))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(index, _)| index)
        .ok_or_else(|| "no final-stage-converged generic hypothesis".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::solver::Solution;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    #[test]
    fn generic_guess_preserves_each_supplied_lens() {
        let original = template();
        for fixed in 0..2 {
            let mut free = [true; 2];
            free[fixed] = false;
            let guessed = guess(&original, 0.4, [0.1; 4], free).unwrap();
            assert_eq!(guessed.lenses[fixed], original.lenses[fixed]);
            assert_ne!(guessed.lenses[1 - fixed], original.lenses[1 - fixed]);
        }
    }

    fn template() -> Calibration {
        serde_json::from_value(serde_json::json!({
            "schema_version":2,
            "lenses":[{"width":640,"height":480,"fx":400.0,"fy":401.0,"cx":318.0,"cy":241.0,"distortion":[0.01,0.0,0.0,0.0]},
                      {"width":800,"height":600,"fx":600.0,"fy":601.0,"cx":410.0,"cy":302.0,"distortion":[0.02,0.0,0.0,0.0]}],
            "projection":{"type":"l-shape","intersect":0.7,"x_ty":0.02,"x_rx":0.03,"blend_width":0.1},
            "framing":{"axis_offset":0.27,"tilt":0.1,"roll":0.2},"sync_offset":7
        })).unwrap()
    }

    #[test]
    fn generic_guesses_use_sensor_sizes_not_saved_optics_or_placement() {
        let template = template();
        let snapshot = template.clone();
        let seed = guess(&template, 0.65, [0.0; 4], [true; 2]).unwrap();
        assert_eq!(template, snapshot);
        assert_eq!(seed.sync_offset, template.sync_offset);
        assert_eq!(seed.projection.l_shape().unwrap().blend_width, 0.1);
        assert_eq!(seed.projection.l_shape().unwrap().intersect, 0.6);
        assert_eq!(seed.framing.axis_offset, 0.23);
        assert_eq!(seed.framing.tilt, 0.0);
        for lens in seed.lenses {
            assert_eq!(lens.fx, lens.width as f64 * 0.65);
            assert_eq!(lens.fy, lens.fx);
            assert_eq!(lens.cx, lens.width as f64 * 0.5);
            assert_eq!(lens.cy, lens.height as f64 * 0.5);
            assert_eq!(lens.distortion, [0.0; 4]);
        }
    }

    fn hypothesis(score: f64, success: bool) -> Hypothesis {
        Hypothesis {
            focal: 0.65,
            fit: Ok(CoarseFit {
                stages: vec![Solution {
                    calibration: template(),
                    x: vec![0.0; 12],
                    cost: 0.0,
                    status: String::new(),
                    success,
                    iterations: 1,
                    elapsed: Duration::ZERO,
                    margin: -0.1,
                    sensor_boundary_active: true,
                    residuals: 0,
                    jacobians: 0,
                    derivative_elapsed: Duration::ZERO,
                }],
                training_rmse: score,
            }),
        }
    }

    #[test]
    fn selection_uses_only_finite_training_scores_of_converged_final_stages() {
        let options = [
            hypothesis(0.01, false),
            hypothesis(0.4, true),
            hypothesis(0.3, true),
            hypothesis(f64::NAN, true),
        ];
        assert_eq!(select_best(&options).unwrap(), 2);
        assert!(select_best(&options[..1]).is_err());
        assert!(select_best(&[]).is_err());
        assert_eq!(
            select_best(&[hypothesis(0.3, true), hypothesis(0.3, true)]).unwrap(),
            0
        );
    }

    #[test]
    fn discovery_rejects_cancellation_expiry_and_untextured_images() {
        let cancelled = AtomicBool::new(true);
        let control = Control {
            interrupted: &cancelled,
            deadline: Instant::now() + Duration::from_secs(10),
        };
        let images = [
            Image::new(64, 48, vec![0.5; 64 * 48]).unwrap(),
            Image::new(64, 48, vec![0.5; 64 * 48]).unwrap(),
        ];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        assert!(
            matches!(discover(&template(), &images, None, [true;2], &pool, &control), Err(e) if e == "calibration cancelled")
        );
        let active = AtomicBool::new(false);
        let expired = Control {
            interrupted: &active,
            deadline: Instant::now() - Duration::from_millis(1),
        };
        assert!(
            matches!(discover(&template(), &images, None, [true;2], &pool, &expired), Err(e) if e == "calibration time limit reached")
        );
        let active_control = Control {
            interrupted: &active,
            deadline: Instant::now() + Duration::from_secs(10),
        };
        assert!(
            matches!(discover(&template(), &images, None, [true;2], &pool, &active_control), Err(e) if e == "insufficient textured generic support")
        );
    }
}
