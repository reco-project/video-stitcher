//! Integration test for the `calibrate_videos` one-call API.
//!
//! Verifies that `calibrate_videos` produces results consistent with
//! a known-good calibration from the same footage. Requires real video
//! files and a GPU. Run with `cargo test -p reco-calibrate --features io -- --ignored`.

#![cfg(feature = "io")]

use std::path::Path;
use std::sync::atomic::AtomicBool;

use reco_calibrate::video::{CalibrateVideosOptions, calibrate_videos};

const LEFT_4K: &str = "/media/guelzim/HDD/reco-test-footage/gopro-hero10/left_4k.mp4";
const RIGHT_4K: &str = "/media/guelzim/HDD/reco-test-footage/gopro-hero10/right_4k.mp4";
const KNOWN_GOOD: &str = "/media/guelzim/HDD/reco-test-footage/gopro-hero10/match_4k.json";

fn have_test_footage() -> bool {
    Path::new(LEFT_4K).exists() && Path::new(KNOWN_GOOD).exists()
}

#[test]
#[ignore]
fn calibrate_videos_matches_known_good() {
    if !have_test_footage() {
        eprintln!("Skipping: test footage not available");
        return;
    }

    let interrupted = AtomicBool::new(false);
    let result = calibrate_videos(
        Path::new(LEFT_4K),
        Path::new(RIGHT_4K),
        CalibrateVideosOptions::default(),
        &mut |p| eprintln!("[test] {}: {}", p.step, p.detail),
        &interrupted,
    )
    .expect("calibrate_videos failed");

    // Load known-good calibration for comparison
    let known: reco_core::calibration::MatchCalibration =
        reco_core::calibration::MatchCalibration::from_file(Path::new(KNOWN_GOOD))
            .expect("failed to load known-good calibration");

    // Sync offset should match exactly (both use IMU)
    assert_eq!(
        result.calibration.sync_offset, known.sync_offset,
        "sync offset mismatch"
    );

    // Placement parameters should be close (AKAZE/RANSAC have random elements)
    let tol = 0.01;
    let got = &result.calibration.layout;
    let exp = &known.layout;

    assert!(
        (got.camera_axis_offset - exp.camera_axis_offset).abs() < tol,
        "camera_axis_offset: got {}, expected {} (tol {tol})",
        got.camera_axis_offset,
        exp.camera_axis_offset,
    );
    assert!(
        (got.intersect - exp.intersect).abs() < tol,
        "intersect: got {}, expected {} (tol {tol})",
        got.intersect,
        exp.intersect,
    );
    assert!(
        (got.x_ty - exp.x_ty).abs() < tol,
        "x_ty: got {}, expected {} (tol {tol})",
        got.x_ty,
        exp.x_ty,
    );
    assert!(
        (got.x_rz - exp.x_rz).abs() < tol,
        "x_rz: got {}, expected {} (tol {tol})",
        got.x_rz,
        exp.x_rz,
    );
    assert!(
        (got.z_rx - exp.z_rx).abs() < tol,
        "z_rx: got {}, expected {} (tol {tol})",
        got.z_rx,
        exp.z_rx,
    );

    // Quality metrics
    assert!(
        result.confidence > 0.5,
        "low confidence: {}",
        result.confidence
    );
    assert!(result.frames_used > 0, "no frames used");
    assert!(
        result.total_matches > 10,
        "too few matches: {}",
        result.total_matches
    );

    println!("calibrate_videos passed:");
    println!("  sync_offset: {}", result.calibration.sync_offset);
    println!("  confidence: {:.1}%", result.confidence * 100.0);
    println!("  frames_used: {}", result.frames_used);
    println!("  matches: {}", result.total_matches);
    println!("  cam_d: {:.4}", got.camera_axis_offset);
    println!("  intersect: {:.4}", got.intersect);
}
