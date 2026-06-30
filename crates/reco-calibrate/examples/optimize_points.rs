//! Load matched points from JSON, remove outliers, run Rust optimizer.
//!
//! Usage: cargo run --release -p reco-calibrate --example optimize_points -- \
//!   <matched_points.json> [outlier_ids...]

use reco_calibrate::geometry;
use reco_calibrate::optimizer;
use reco_calibrate::types::{CalibrationConfig, MatchedPoint};
use reco_core::calibration::{Calibration, Lens};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <matched_points.json> [outlier_ids...]", args[0]);
        std::process::exit(1);
    }

    let json_str = std::fs::read_to_string(&args[1]).unwrap();
    let raw: Vec<serde_json::Value> = serde_json::from_str(&json_str).unwrap();

    let outlier_ids: std::collections::HashSet<u64> =
        args[2..].iter().filter_map(|s| s.parse().ok()).collect();

    let points: Vec<MatchedPoint> = raw
        .iter()
        .filter(|p| !outlier_ids.contains(&p["idx"].as_u64().unwrap()))
        .map(|p| {
            let lp = &p["left_plane_coords"];
            let rp = &p["right_plane_coords"];
            MatchedPoint::from_planes(
                [lp[0].as_f64().unwrap(), lp[1].as_f64().unwrap()],
                [rp[0].as_f64().unwrap(), rp[1].as_f64().unwrap()],
            )
        })
        .collect();

    eprintln!(
        "Loaded {} points ({} removed as outliers)",
        points.len(),
        outlier_ids.len()
    );

    let config = CalibrationConfig::default();

    match optimizer::optimize(&points, &config) {
        Ok((topology, framing, residual)) => {
            eprintln!("\nRust optimizer result:");
            eprintln!("  cameraAxisOffset: {:.6}", framing.axis_offset);
            eprintln!("  intersect:        {:.6}", topology.intersect);
            eprintln!("  xTy:              {:.6}", topology.x_ty);
            eprintln!("  xRz:              {:.6}", topology.x_rz);
            eprintln!("  zRx:              {:.6}", topology.z_rx);
            eprintln!("  zRz:              {:.6}", topology.z_rz);
            eprintln!("  residual:         {:.6}", residual);

            // Compare with v1 reference
            let v1 = geometry::OptParams {
                x_ty: 0.0047,
                intersect: 0.5447,
                cam_d: 0.2407,
                x_rz: 0.0071,
                z_rx: -0.0035,
                z_rz: None,
                x_rx: None,
            };
            let v1_err = geometry::angular_error(&points, &v1);
            eprintln!("\n  v1 reference residual: {:.6}", v1_err);

            // Emit a current-schema calibration (loadable by `reco stitch -c`).
            let lens = || {
                Lens::fisheye(
                    3840,
                    2160,
                    1796.3208206894308,
                    1797.22277342282,
                    1919.372365976781,
                    1063.171593155705,
                    [
                        0.034213889574164644,
                        0.06767320765357862,
                        -0.07408969996955275,
                        0.029944425249175583,
                    ],
                )
            };
            let cal = Calibration::new(vec![lens(), lens()], topology, framing);
            println!("{}", cal.to_json_pretty());
        }
        Err(e) => eprintln!("Optimization failed: {e}"),
    }
}
