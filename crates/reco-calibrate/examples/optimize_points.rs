//! Load matched points from JSON, remove outliers, run Rust optimizer.
//!
//! Usage: cargo run --release -p reco-calibrate --example optimize_points -- \
//!   <matched_points.json> [outlier_ids...]

use reco_calibrate::geometry;
use reco_calibrate::optimizer;
use reco_calibrate::types::{CalibrationConfig, MatchedPoint};

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
            MatchedPoint {
                left: [lp[0].as_f64().unwrap(), lp[1].as_f64().unwrap()],
                right: [rp[0].as_f64().unwrap(), rp[1].as_f64().unwrap()],
            }
        })
        .collect();

    eprintln!(
        "Loaded {} points ({} removed as outliers)",
        points.len(),
        outlier_ids.len()
    );

    let config = CalibrationConfig {
        enable_sixth_param: true,
        max_optimizer_evals: 5000,
        ..Default::default()
    };

    match optimizer::optimize(&points, &config) {
        Ok((layout, residual)) => {
            eprintln!("\nRust optimizer result:");
            eprintln!("  cameraAxisOffset: {:.6}", layout.camera_axis_offset);
            eprintln!("  intersect:        {:.6}", layout.intersect);
            eprintln!("  xTy:              {:.6}", layout.x_ty);
            eprintln!("  xRz:              {:.6}", layout.x_rz);
            eprintln!("  zRx:              {:.6}", layout.z_rx);
            eprintln!("  zRz:              {:.6}", layout.z_rz);
            eprintln!("  residual:         {:.6}", residual);

            // Compare with v1 reference
            let v1 = geometry::OptParams {
                x_ty: 0.0047,
                intersect: 0.5447,
                cam_d: 0.2407,
                x_rz: 0.0071,
                z_rx: -0.0035,
                z_rz: None,
            };
            let v1_err = geometry::angular_error(&points, &v1);
            eprintln!("\n  v1 reference residual: {:.6}", v1_err);

            // Output as JSON
            let json = serde_json::json!({
                "left_uniforms": {
                    "width": 3840, "height": 2160,
                    "fx": 1796.3208206894308, "fy": 1797.22277342282,
                    "cx": 1919.372365976781, "cy": 1063.171593155705,
                    "d": [0.034213889574164644, 0.06767320765357862, -0.07408969996955275, 0.029944425249175583]
                },
                "right_uniforms": {
                    "width": 3840, "height": 2160,
                    "fx": 1796.3208206894308, "fy": 1797.22277342282,
                    "cx": 1919.372365976781, "cy": 1063.171593155705,
                    "d": [0.034213889574164644, 0.06767320765357862, -0.07408969996955275, 0.029944425249175583]
                },
                "params": {
                    "cameraAxisOffset": layout.camera_axis_offset,
                    "intersect": layout.intersect,
                    "xTy": layout.x_ty,
                    "xRz": layout.x_rz,
                    "zRx": layout.z_rx,
                    "zRz": layout.z_rz,
                },
                "sync_offset": 67
            });
            println!("{}", serde_json::to_string_pretty(&json).unwrap());
        }
        Err(e) => eprintln!("Optimization failed: {e}"),
    }
}
