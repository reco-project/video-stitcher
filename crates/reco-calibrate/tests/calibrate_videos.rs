//! Integration test for the `calibrate_videos` one-call API.
//!
//! Exercises native fitting and independent confirmation through the real decoder.
//! Requires the named local footage, not a GPU. Run with `cargo test -p reco-calibrate --features io --test calibrate_videos -- --ignored`.

#![cfg(feature = "io")]

use std::path::Path;
use std::sync::atomic::AtomicBool;

use reco_calibrate::video::{CalibrateVideosOptions, calibrate_videos};

#[test]
#[ignore = "requires RECO_TEST_FOOTAGE with synchronized reference fixtures"]
fn calibrate_videos_confirms_real_decoded_overlap() {
    let footage = std::env::var_os("RECO_TEST_FOOTAGE").expect("set RECO_TEST_FOOTAGE");
    #[derive(serde::Deserialize)]
    struct Expected {
        sync_offset: i64,
        skip_start_secs: f64,
        skip_end_secs: f64,
    }
    let expected: Expected = serde_json::from_slice(
        &std::fs::read(Path::new(&footage).join("expected.json")).unwrap(),
    )
    .unwrap();
    let left = Path::new(&footage).join("left.mp4");
    let right = Path::new(&footage).join("right.mp4");
    assert!(
        left.exists() && right.exists(),
        "test footage is unavailable"
    );
    let interrupted = AtomicBool::new(false);
    let reference = reco_core::calibration::Calibration::from_file(
        &Path::new(&footage).join("reference.json"),
    )
    .unwrap();
    for estimate_optics in [false, true] {
        let result = calibrate_videos(
            &left,
            &right,
            CalibrateVideosOptions {
                left_params: (!estimate_optics).then(|| reference.lenses[0].clone()),
                right_params: (!estimate_optics).then(|| reference.lenses[1].clone()),
                sync_offset: Some(expected.sync_offset),
                config: Some(reco_calibrate::CalibrationConfig {
                    num_frames: 4,
                    skip_start_secs: expected.skip_start_secs,
                    skip_end_secs: expected.skip_end_secs,
                }),
            },
            &mut |p| eprintln!("[test] {}: {}", p.step, p.detail),
            &interrupted,
        )
        .expect("calibrate_videos failed");

        result.calibration.validate().unwrap();
        assert_eq!(result.calibration.sync_offset, expected.sync_offset);
        if estimate_optics {
            assert!(matches!(
                result.assessment,
                reco_calibrate::CalibrationAssessment::NeedsReview { .. }
            ));
            assert!(
                result
                    .overlap_checks
                    .iter()
                    .any(|check| check.bands.iter().any(|&n| n < 8))
            );
        } else {
            assert_eq!(result.calibration.lenses, reference.lenses);
            assert_eq!(
                result.assessment,
                reco_calibrate::CalibrationAssessment::Confirmed
            );
        }
        assert_eq!(result.frames_used, 1);
        assert_eq!(result.overlap_checks.len(), 3);
        assert!(
            result.total_matches > 10,
            "too few matches: {}",
            result.total_matches
        );

        for profile in [result.left_lens_profile, result.right_lens_profile] {
            assert_eq!(
                matches!(
                    profile.unwrap().source,
                    reco_calibrate::ProfileSource::Estimated
                ),
                estimate_optics
            );
        }
    }
}
