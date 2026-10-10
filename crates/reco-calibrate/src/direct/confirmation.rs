//! Verify unused images against the fitted mapping, without feeding observations back to fitting.

use super::{
    matching::{Image, measure},
    preparation::blur,
    solver::Control,
    support,
};
use crate::{CalibrationAssessment, OverlapCheck};
use reco_core::calibration::Calibration;

pub(super) fn check(
    cal: &Calibration,
    images: &[Image; 2],
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<OverlapCheck, String> {
    let prepared = [
        blur(&images[0], 1.2, || control.check())?,
        blur(&images[1], 1.2, || control.check())?,
    ];
    let patches: Vec<_> = support::select(cal, &prepared[0], || control.check())?
        .into_iter()
        .map(|p| p.pixels)
        .collect();
    let observations = measure(cal, &prepared, &patches, pool, &|| control.check())?;
    let dims = images[0].dimensions().map(|v| v as f64);
    let target_dims = images[1].dimensions().map(|v| v as f64);
    let mapping = cal.source_correspondence(0, 1).map_err(|e| e.to_string())?;
    let mut bands = [0; 3];
    let mut errors = Vec::new();
    for (patch, observation) in patches.iter().zip(observations) {
        control.check()?;
        if !observation.accepted {
            continue;
        }
        let point = patch[patch.len() / 2];
        let uv = [(point[0] + 0.5) / dims[0], (point[1] + 0.5) / dims[1]];
        if let (Some(target), Some(mapped)) = (observation.target, mapping.map(uv)) {
            let error = (mapped[0] * target_dims[0] - 0.5 - target[0])
                .hypot(mapped[1] * target_dims[1] - 0.5 - target[1]);
            errors.push(error * 960.0 / target_dims[0].max(target_dims[1]));
            bands[if uv[1] < 0.4 {
                0
            } else if uv[1] < 0.65 {
                1
            } else {
                2
            }] += 1;
        }
    }
    errors.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        (!errors.is_empty()).then(|| errors[((errors.len() - 1) as f64 * p).ceil() as usize])
    };
    Ok(OverlapCheck {
        matches: errors.len(),
        bands,
        median_pixels: percentile(0.5),
        p95_pixels: percentile(0.95),
    })
}

pub(super) fn assess(checks: &[OverlapCheck]) -> Result<CalibrationAssessment, String> {
    if checks.iter().any(|c| {
        c.matches >= 15
            && (c.median_pixels.is_some_and(|x| x > 3.0) || c.p95_pixels.is_some_and(|x| x > 6.0))
    }) {
        return Err("independent images contradict the fitted alignment".into());
    }
    if !checks.is_empty()
        && checks.iter().all(|c| {
            c.bands.iter().all(|&n| n >= 8)
                && c.median_pixels.is_some_and(|x| x <= 1.0)
                && c.p95_pixels.is_some_and(|x| x <= 2.0)
        })
    {
        Ok(CalibrationAssessment::Confirmed)
    } else {
        Ok(CalibrationAssessment::NeedsReview {
            reason: "Independent overlap checks lack sufficient coverage or precision; inspect the seam before applying.".into()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires RECO_CALIBRATION_FIXTURES containing align_luma reports and Y planes"]
    fn real_unused_frames_confirm_alignment_and_reject_displaced_geometry() {
        let root = std::path::PathBuf::from(
            std::env::var_os("RECO_CALIBRATION_FIXTURES").expect("fixture directory"),
        );
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("gopro-0.json")).unwrap()).unwrap();
        let cal: Calibration =
            serde_json::from_value(report["result"]["calibration"].clone()).unwrap();
        let cancelled = std::sync::atomic::AtomicBool::new(false);
        let control = Control {
            interrupted: &cancelled,
            deadline: std::time::Instant::now() + std::time::Duration::from_secs(30),
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let read = |side: usize| {
            let name = if side == 0 { "left" } else { "right" };
            let data = std::fs::read(root.join(format!("gopro-{name}-1.y"))).unwrap();
            super::super::intake::prepare(
                &data,
                cal.lenses[side].width,
                cal.lenses[side].height,
                &control,
            )
            .unwrap()
        };
        let images = [read(0), read(1)];
        let baseline = check(&cal, &images, &pool, &control).unwrap();
        assert_eq!(
            assess(&[baseline]).unwrap(),
            CalibrationAssessment::Confirmed
        );
        let mut rejected = 0;
        for delta in [-0.04, -0.02, 0.02, 0.04] {
            let mut displaced = cal.clone();
            displaced.projection.l_shape_mut().unwrap().x_ty += delta;
            let evidence = check(&displaced, &images, &pool, &control).unwrap();
            eprintln!("displacement={delta}: {evidence:?}");
            let verdict = assess(&[evidence]);
            assert!(!matches!(verdict, Ok(CalibrationAssessment::Confirmed)));
            rejected += usize::from(verdict.is_err());
        }
        assert!(
            rejected >= 2,
            "displaced geometry should contradict measured correspondences"
        );
    }
    #[test]
    fn confirmation_requires_spread_precision_and_unused_evidence() {
        let good = OverlapCheck {
            matches: 24,
            bands: [8; 3],
            median_pixels: Some(0.4),
            p95_pixels: Some(0.9),
        };
        assert_eq!(
            assess(std::slice::from_ref(&good)).unwrap(),
            CalibrationAssessment::Confirmed
        );
        assert!(matches!(
            assess(&[]).unwrap(),
            CalibrationAssessment::NeedsReview { .. }
        ));
        let mut poor = good.clone();
        poor.bands = [0, 24, 0];
        assert!(matches!(
            assess(&[poor]).unwrap(),
            CalibrationAssessment::NeedsReview { .. }
        ));
        let mut poor = good.clone();
        poor.median_pixels = Some(4.0);
        assert!(assess(&[poor]).is_err());
        let mut uncertain = good.clone();
        uncertain.p95_pixels = Some(2.1);
        assert!(matches!(
            assess(&[good, uncertain]).unwrap(),
            CalibrationAssessment::NeedsReview { .. }
        ));
    }
}
