//! Direct alignment of decoded luma, with independent-frame overlap verification.

mod confirmation;
mod features;
mod generic;
mod initialization;
mod intake;
mod kernel;
mod matching;
mod preparation;
mod solver;
mod support;
mod workflow;

use crate::{CalibrateError, CalibrationResult, Yuv420pFrame};
use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::projection::LShape;
use solver::Control;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

pub(crate) fn calibrate(
    frames: &[(Yuv420pFrame, Yuv420pFrame)],
    profiles: [Option<&Lens>; 2],
    interrupted: &AtomicBool,
) -> Result<CalibrationResult, CalibrateError> {
    run(frames, profiles, interrupted).map_err(CalibrateError::Alignment)
}

fn run(
    frames: &[(Yuv420pFrame, Yuv420pFrame)],
    profiles: [Option<&Lens>; 2],
    interrupted: &AtomicBool,
) -> Result<CalibrationResult, String> {
    let started = Instant::now();
    let control = Control {
        interrupted,
        deadline: started + Duration::from_secs(60),
    };
    control.check()?;
    if frames.is_empty() || frames.len() > 16 {
        return Err("calibration requires 1 to 16 synchronized frame pairs".into());
    }
    let dimensions = [
        [frames[0].0.width, frames[0].0.height],
        [frames[0].1.width, frames[0].1.height],
    ];
    for pair in frames {
        for (side, frame) in [&pair.0, &pair.1].into_iter().enumerate() {
            control.check()?;
            frame.validate().map_err(|e| e.to_string())?;
            if [frame.width, frame.height] != dimensions[side] {
                return Err(format!(
                    "camera {side} dimensions changed during calibration"
                ));
            }
            if !intake::valid_dimensions(frame.width, frame.height) {
                return Err("unsupported calibration image dimensions".into());
            }
        }
    }
    let lenses = dimensions
        .iter()
        .zip(profiles)
        .map(|(&[w, h], supplied)| {
            supplied.cloned().unwrap_or_else(|| {
                Lens::fisheye(
                    w,
                    h,
                    0.65 * f64::from(w),
                    0.65 * f64::from(w),
                    f64::from(w) / 2.0,
                    f64::from(h) / 2.0,
                    [0.0; 4],
                )
            })
        })
        .collect();
    let mut seed = Calibration::new(
        lenses,
        LShape {
            intersect: 0.6,
            x_ty: 0.0,
            x_rx: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        },
        Framing {
            axis_offset: 0.23,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    seed.validate().map_err(|e| e.to_string())?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().map_or(1, |n| n.get().min(4)))
        .build()
        .map_err(|e| e.to_string())?;
    let prepare = |pair: &(Yuv420pFrame, Yuv420pFrame)| -> Result<_, String> {
        Ok([
            intake::prepare(&pair.0.y, pair.0.width, pair.0.height, &control)?,
            intake::prepare(&pair.1.y, pair.1.width, pair.1.height, &control)?,
        ])
    };
    let images = prepare(&frames[0])?;
    let prepared_at = started.elapsed();
    let optics = profiles.map(|p| p.is_none());
    let discover = |seed: &Calibration| -> Result<_, String> {
        let discovered = workflow::discover_and_propose(seed, optics, &images, &pool, &control)?;
        log::debug!(
            "Optical discovery: focal={}, features={}",
            discovered.focal,
            discovered.used_features
        );
        Ok(discovered.proposal)
    };
    let proposal = if optics.iter().all(|&v| v) {
        discover(&seed)?
    } else {
        if optics[0] != optics[1] {
            let fixed = usize::from(optics[0]);
            let free = 1 - fixed;
            let [w, h] = dimensions[free];
            // A supplied lens is a starting estimate for its peer, never a fixed substitute.
            seed.lenses[free] = seed.lenses[fixed]
                .rescaled(w, h)
                .map_err(|e| e.to_string())?;
        }
        workflow::from_optics(seed, optics, &images, &pool, &control).map_err(|error| {
            format!("{error}. Supplied optics were kept fixed; check the image order and sync, or explicitly estimate optics.")
        })?
    };
    drop(images);
    let solution = proposal
        .refinement
        .last()
        .ok_or("missing final calibration")?;
    let calibration = solution.calibration.clone();
    calibration.validate().map_err(|e| e.to_string())?;
    let fitting_at = started.elapsed();
    let mut checks = Vec::new();
    // The first pair alone fits the model. No unused pair feeds back into fitting or selection.
    for pair in frames.iter().skip(1) {
        control.check()?;
        if pair.0.timestamp_us == frames[0].0.timestamp_us
            || pair.1.timestamp_us == frames[0].1.timestamp_us
        {
            continue;
        }
        checks.push(confirmation::check(
            &calibration,
            &prepare(pair)?,
            &pool,
            &control,
        )?);
    }
    let assessment = confirmation::assess(&checks)?;
    control.check()?;
    log::info!(
        "Calibration: preparation={:.3}s, fitting={:.3}s, confirmation={:.3}s; {:?}",
        prepared_at.as_secs_f64(),
        (fitting_at - prepared_at).as_secs_f64(),
        (started.elapsed() - fitting_at).as_secs_f64(),
        assessment
    );
    log::debug!(
        "Alignment diagnostics: support={}/{}/{}, matching={:?}, proposal={:?}, initialization={:?}, stages={:?}",
        proposal.initial_patches,
        proposal.matched_training_patches,
        proposal.fitting_patches,
        proposal.matching_elapsed,
        proposal.elapsed,
        proposal.initialization,
        proposal.refinement
    );
    Ok(CalibrationResult {
        calibration,
        total_matches: proposal.matched_training_patches,
        frames_used: 1,
        assessment,
        overlap_checks: checks,
        left_lens_profile: optics[0].then(estimated_profile),
        right_lens_profile: optics[1].then(estimated_profile),
    })
}

fn estimated_profile() -> crate::LensProfileInfo {
    crate::LensProfileInfo {
        camera: "".into(),
        lens: "Estimated from overlap".into(),
        source: crate::ProfileSource::Estimated,
        path: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires RECO_CALIBRATION_FIXTURES containing align_luma reports and Y planes"]
    fn real_mixed_optics_preserve_supplied_lenses_or_reject_the_fit() {
        let root = std::path::PathBuf::from(
            std::env::var_os("RECO_CALIBRATION_FIXTURES").expect("fixture directory"),
        );
        for rig in ["gopro", "dji", "xtu"] {
            let report: serde_json::Value =
                serde_json::from_slice(&std::fs::read(root.join(format!("{rig}-0.json"))).unwrap())
                    .unwrap();
            let known: Calibration =
                serde_json::from_value(report["result"]["calibration"].clone()).unwrap();
            let read = |index, side: usize| {
                let lens = &known.lenses[side];
                let name = ["left", "right"][side];
                Yuv420pFrame {
                    width: lens.width,
                    height: lens.height,
                    y: std::fs::read(root.join(format!("{rig}-{name}-{index}.y"))).unwrap(),
                    u: vec![128; (lens.width.div_ceil(2) * lens.height.div_ceil(2)) as usize],
                    v: vec![128; (lens.width.div_ceil(2) * lens.height.div_ceil(2)) as usize],
                    timestamp_us: i64::from(index) * 1_000_000,
                }
            };
            let frames: Vec<_> = (0..4).map(|i| (read(i, 0), read(i, 1))).collect();
            for fixed in 0..2 {
                let mut profiles = [None; 2];
                profiles[fixed] = Some(&known.lenses[fixed]);
                let result = match run(&frames, profiles, &AtomicBool::new(false)) {
                    Ok(result) => result,
                    Err(error) => {
                        // A stitching estimate is not necessarily a reusable lens profile.
                        assert_ne!(rig, "gopro", "camera {fixed}: {error}");
                        assert!(
                            error.contains("insufficient image-measured initialization")
                                || error.contains("did not converge"),
                            "{rig} camera {fixed}: {error}"
                        );
                        eprintln!("{rig} fixed camera {fixed}: rejected ({error})");
                        continue;
                    }
                };
                assert_eq!(result.calibration.lenses[fixed], known.lenses[fixed]);
                let provenance = [result.left_lens_profile, result.right_lens_profile];
                assert!(provenance[fixed].is_none());
                assert!(matches!(
                    provenance[1 - fixed].as_ref().unwrap().source,
                    crate::ProfileSource::Estimated
                ));
                assert_eq!(result.overlap_checks.len(), 3);
                eprintln!("{rig} fixed camera {fixed}: {:?}", result.assessment);
            }
        }
    }

    #[test]
    fn cancellation_and_malformed_input_never_produce_a_calibration() {
        assert_eq!(
            run(&[], [None, None], &AtomicBool::new(true)).unwrap_err(),
            "calibration cancelled"
        );
        assert!(run(&[], [None, None], &AtomicBool::new(false)).is_err());
        let frame = Yuv420pFrame {
            width: 64,
            height: 64,
            y: vec![0; 4096],
            u: vec![0; 1024],
            v: vec![0; 1024],
            timestamp_us: 0,
        };
        for plane in 0..3 {
            let mut malformed = frame.clone();
            [&mut malformed.y, &mut malformed.u, &mut malformed.v][plane].pop();
            assert!(
                run(
                    &[(malformed, frame.clone())],
                    [None, None],
                    &AtomicBool::new(false)
                )
                .is_err()
            );
        }
        assert!(
            run(
                &vec![(frame.clone(), frame.clone()); 17],
                [None, None],
                &AtomicBool::new(false)
            )
            .is_err()
        );
        let larger = Yuv420pFrame {
            width: 128,
            y: vec![0; 8192],
            u: vec![0; 2048],
            v: vec![0; 2048],
            ..frame.clone()
        };
        assert!(
            run(
                &[(frame.clone(), frame.clone()), (larger, frame.clone())],
                [None, None],
                &AtomicBool::new(false)
            )
            .unwrap_err()
            .contains("dimensions changed")
        );
        assert!(
            run(
                &[(frame.clone(), frame)],
                [None, None],
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
}
