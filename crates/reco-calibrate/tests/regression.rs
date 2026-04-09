//! Regression tests for calibration optimizer.
//!
//! Loads pre-extracted matched points from real footage (GoPro 10, DJI
//! Action 4, XTU Max3) and verifies the optimizer produces parameters
//! within tolerance of validated reference values.
//!
//! These tests run without GPU or video files - they only need the
//! JSON point data and the optimizer.

use reco_calibrate::optimizer;
use reco_calibrate::types::{CalibrationConfig, MatchedPoint};

use serde::Deserialize;
use std::path::Path;

/// Test dataset loaded from JSON.
#[derive(Deserialize)]
#[allow(dead_code)]
struct TestDataset {
    camera: String,
    n_points: usize,
    expected: ExpectedParams,
    points: Vec<TestPoint>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ExpectedParams {
    cam_d: f64,
    intersect: f64,
    x_ty: f64,
    x_rz_deg: f64,
    z_rx_deg: f64,
    ratio: f64,
}

#[derive(Deserialize)]
struct TestPoint {
    left: [f64; 2],
    right: [f64; 2],
    left_pixel_nx: f64,
    right_pixel_nx: f64,
}

fn load_dataset(name: &str) -> TestDataset {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(format!("{name}.json"));
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()))
}

fn to_matched_points(dataset: &TestDataset) -> Vec<MatchedPoint> {
    dataset
        .points
        .iter()
        .map(|p| MatchedPoint {
            left: p.left,
            right: p.right,
            left_pixel_nx: p.left_pixel_nx,
            right_pixel_nx: p.right_pixel_nx,
        })
        .collect()
}

/// Run the optimizer and return (cam_d, intersect, x_ty, x_rz_deg, z_rx_deg, ratio).
fn run_calibration(points: &[MatchedPoint], trim: f64) -> (f64, f64, f64, f64, f64, f64) {
    let config = CalibrationConfig {
        trim_fraction: trim,
        seam_sigma: 0.08,
        ..Default::default()
    };

    let (layout, _residual) =
        optimizer::optimize(points, &config).expect("optimizer should converge");

    let half_offset = 0.5 * (1.0 - layout.intersect);
    let ratio = if half_offset > 1e-10 {
        layout.camera_axis_offset / half_offset
    } else {
        f64::INFINITY
    };

    (
        layout.camera_axis_offset,
        layout.intersect,
        layout.x_ty,
        layout.x_rz.to_degrees(),
        layout.z_rx.to_degrees(),
        ratio,
    )
}

/// Assert calibration parameters are within tolerance of expected values.
fn assert_calibration(
    label: &str,
    result: (f64, f64, f64, f64, f64, f64),
    expected: &ExpectedParams,
    rotation_tol_deg: f64,
    ratio_tol: f64,
    intersect_tol: f64,
) {
    let (cam_d, intersect, _x_ty, x_rz_deg, z_rx_deg, ratio) = result;

    eprintln!("{label}:");
    eprintln!(
        "  cam_d={cam_d:.4} (exp {:.4})  ratio={ratio:.3} (exp {:.3})",
        expected.cam_d, expected.ratio
    );
    eprintln!("  intersect={intersect:.4} (exp {:.4})", expected.intersect);
    eprintln!(
        "  x_rz={x_rz_deg:.2} deg (exp {:.2})  z_rx={z_rx_deg:.2} deg (exp {:.2})",
        expected.x_rz_deg, expected.z_rx_deg
    );

    assert!(
        (intersect - expected.intersect).abs() < intersect_tol,
        "{label}: intersect {intersect:.4} not within {intersect_tol} of {:.4}",
        expected.intersect
    );

    assert!(
        (ratio - expected.ratio).abs() < ratio_tol,
        "{label}: ratio {ratio:.3} not within {ratio_tol} of {:.3}",
        expected.ratio
    );

    assert!(
        (x_rz_deg - expected.x_rz_deg).abs() < rotation_tol_deg,
        "{label}: x_rz {x_rz_deg:.2} deg not within {rotation_tol_deg} deg of {:.2}",
        expected.x_rz_deg
    );

    assert!(
        (z_rx_deg - expected.z_rx_deg).abs() < rotation_tol_deg,
        "{label}: z_rx {z_rx_deg:.2} deg not within {rotation_tol_deg} deg of {:.2}",
        expected.z_rx_deg
    );
}

#[test]
fn regression_gopro10_4k() {
    let ds = load_dataset("gopro10_4k");
    assert_eq!(ds.n_points, ds.points.len());

    let points = to_matched_points(&ds);
    let result = run_calibration(&points, 0.3);

    // GoPro has lots of points (148) so we expect tight convergence
    assert_calibration(
        "GoPro 10 4K",
        result,
        &ds.expected,
        2.0,  // rotation tolerance: 2 degrees
        0.2,  // ratio tolerance
        0.05, // intersect tolerance
    );
}

#[test]
fn regression_dji_action4() {
    let ds = load_dataset("dji_action4");
    assert_eq!(ds.n_points, ds.points.len());

    let points = to_matched_points(&ds);
    let result = run_calibration(&points, 0.3);

    assert_calibration(
        "DJI Action 4",
        result,
        &ds.expected,
        2.0,  // rotation tolerance
        0.2,  // ratio tolerance
        0.05, // intersect tolerance
    );
}

#[test]
fn regression_xtu_max3() {
    let ds = load_dataset("xtu_max3_4k");
    assert_eq!(ds.n_points, ds.points.len());

    let points = to_matched_points(&ds);
    let result = run_calibration(&points, 0.3);

    assert_calibration(
        "XTU Max3 4K",
        result,
        &ds.expected,
        2.0,  // rotation tolerance
        0.2,  // ratio tolerance
        0.05, // intersect tolerance
    );
}

#[test]
fn dji_basin_costs() {
    // Verify Rust cost function ranks Basin A (amazing) lower than Basin B (off)
    // when using seam-weighted trimmed cost. If this fails, the seam weighting
    // implementation doesn't match the Python reference.
    let ds = load_dataset("dji_action4");
    let points = to_matched_points(&ds);

    let sigma = 0.08;
    let trim = 0.2;

    // Basin A: the visually amazing result (seam-weighted optimum)
    let params_a = reco_calibrate::geometry::OptParams {
        cam_d: 0.1847,
        intersect: 0.6576,
        x_ty: -0.0034,
        x_rz: 0.17_f64.to_radians(),
        z_rx: (-0.74_f64).to_radians(),
        z_rz: None,
    };
    // Basin B: the raw optimum (coherent but visually off)
    let params_b = reco_calibrate::geometry::OptParams {
        cam_d: 0.1657,
        intersect: 0.6579,
        x_ty: -0.0163,
        x_rz: (-1.92_f64).to_radians(),
        z_rx: (-2.82_f64).to_radians(),
        z_rz: None,
    };

    let cost_a = reco_calibrate::geometry::trimmed_seam_weighted_reprojection_error(
        &points, &params_a, sigma, trim,
    );
    let cost_b = reco_calibrate::geometry::trimmed_seam_weighted_reprojection_error(
        &points, &params_b, sigma, trim,
    );

    eprintln!("Basin A (amazing): seam+trim cost = {cost_a:.10}");
    eprintln!("Basin B (off):     seam+trim cost = {cost_b:.10}");
    eprintln!("Ratio B/A: {:.3}", cost_b / cost_a);

    assert!(
        cost_a < cost_b,
        "Seam-weighted cost should rank Basin A lower than Basin B: A={cost_a:.10} B={cost_b:.10}"
    );
}

#[test]
fn regression_ratio_consistency() {
    // All cameras should produce cam_d/half_offset ratio between 0.8 and 1.5
    for name in ["gopro10_4k", "dji_action4", "xtu_max3_4k"] {
        let ds = load_dataset(name);
        let points = to_matched_points(&ds);
        let (_, intersect, _, _, _, ratio) = run_calibration(&points, 0.3);

        eprintln!("{name}: ratio={ratio:.3}, intersect={intersect:.4}");
        assert!(
            (0.8..1.5).contains(&ratio),
            "{name}: ratio {ratio:.3} outside expected range [0.8, 1.5]"
        );
    }
}
