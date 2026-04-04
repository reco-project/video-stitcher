//! Sweep cam_d values, optimizing the other 4 params at each,
//! to visualize how cam_d affects residual error.
//!
//! First run a calibration with --debug-dir to get the matched points,
//! or this example runs the matching pipeline itself.
//!
//! Usage: cargo run --release -p reco-calibrate --example sweep_cam_d -- \
//!   <left.mp4> <right.mp4> <match.json> <sync_offset>

use std::f64::consts::PI;

use cobyla::{RhoBeg, StopTols};
use reco_calibrate::features::{self, DetectRegion};
use reco_calibrate::geometry;
use reco_calibrate::types::MatchedPoint;
use reco_core::calibration::MatchCalibration;
use reco_core::gpu::GpuContext;
use reco_core::undistort::GpuUndistort;

fn main() {
    reco_io::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "Usage: {} <left.mp4> <right.mp4> <match.json> <sync_offset>",
            args[0]
        );
        std::process::exit(1);
    }

    let sync_offset: u64 = args[4].parse().expect("invalid sync_offset");

    let json_str = std::fs::read_to_string(&args[3]).unwrap();
    let cal: MatchCalibration = serde_json::from_str(&json_str).expect("invalid match.json");

    // Extract and match features from multiple frames
    let gpu = pollster::block_on(GpuContext::new()).expect("no GPU");
    let all_points = collect_matches(&args[1], &args[2], &cal, sync_offset, &gpu);
    eprintln!("Total matched points: {}", all_points.len());

    if all_points.is_empty() {
        eprintln!("No matches found!");
        std::process::exit(1);
    }

    // Sweep cam_d from 0.05 to 0.60 in steps of 0.01
    eprintln!("\ncam_d\tresidual\tintersect\tx_ty\t\tx_rz\t\tz_rx");
    let mut results = Vec::new();
    let mut d = 0.05;
    while d <= 0.60 {
        if let Some((residual, params)) = optimize_fixed_cam_d(&all_points, d) {
            eprintln!(
                "{:.3}\t{:.6}\t{:.4}\t\t{:.6}\t{:.6}\t{:.6}",
                d, residual, params[1], params[0], params[3], params[4]
            );
            results.push((d, residual));
        } else {
            eprintln!("{:.3}\tFAILED", d);
        }
        d += 0.01;
    }

    // Also evaluate v1 reference
    let v1 = geometry::OptParams {
        x_ty: 0.0047,
        intersect: 0.5447,
        cam_d: 0.2407,
        x_rz: 0.0071,
        z_rx: -0.0035,
        z_rz: None,
    };
    let v1_err = geometry::angular_error(&all_points, &v1);
    eprintln!("\nv1 reference (cam_d=0.2407): residual={v1_err:.6}");

    // Print best
    if let Some((best_d, best_r)) = results.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
        eprintln!("Best: cam_d={best_d:.3}, residual={best_r:.6}");
    }

    // Output CSV for plotting
    println!("cam_d,residual");
    for (d, r) in &results {
        println!("{d:.3},{r:.6}");
    }
}

/// Optimize 4 params (x_ty, intersect, x_rz, z_rx) with cam_d fixed.
fn optimize_fixed_cam_d(points: &[MatchedPoint], fixed_cam_d: f64) -> Option<(f64, [f64; 5])> {
    // Bounds for the 4 free params: [x_ty, intersect, x_rz, z_rx]
    let bounds_4: [(f64, f64); 4] = [(-1.0, 1.0), (0.0, 1.0), (-PI, PI), (-PI, PI)];

    let starts: [[f64; 4]; 4] = [
        [0.0, 0.5, 0.0, 0.0],
        [0.005, 0.55, 0.007, -0.004],
        [0.0, 0.3, 0.0, 0.0],
        [0.0, 0.7, 0.0, 0.0],
    ];

    let mut best: Option<(f64, [f64; 5])> = None;

    for init in &starts {
        let cam_d = fixed_cam_d;
        let data = (points, cam_d);
        let cons: Vec<&dyn cobyla::Func<(&[MatchedPoint], f64)>> = vec![];

        let stop_tols = StopTols {
            ftol_rel: 1e-10,
            xtol_rel: 1e-10,
            ..StopTols::default()
        };

        let result = cobyla::minimize(
            |x: &[f64], ctx: &mut (&[MatchedPoint], f64)| {
                let params = geometry::OptParams {
                    x_ty: x[0],
                    intersect: x[1],
                    cam_d: ctx.1,
                    x_rz: x[2],
                    z_rx: x[3],
                    z_rz: None,
                };
                geometry::angular_error(ctx.0, &params)
            },
            init,
            &bounds_4,
            &cons,
            data,
            2000,
            RhoBeg::All(0.3),
            Some(stop_tols),
        );

        let (x, f) = match result {
            Ok((_status, x, f)) => (x, f),
            Err((_status, x, f)) if f.is_finite() => (x, f),
            Err(_) => continue,
        };

        let full = [x[0], x[1], cam_d, x[2], x[3]];
        if best.as_ref().is_none_or(|(r, _)| f < *r) {
            best = Some((f, full));
        }
    }

    best
}

/// Collect matched points from multiple frames.
fn collect_matches(
    left_path: &str,
    right_path: &str,
    cal: &MatchCalibration,
    sync_offset: u64,
    gpu: &GpuContext,
) -> Vec<MatchedPoint> {
    let mut left_dec =
        reco_io::ffmpeg::decoder::VideoDecoder::open(std::path::Path::new(left_path)).unwrap();
    let mut right_dec =
        reco_io::ffmpeg::decoder::VideoDecoder::open(std::path::Path::new(right_path)).unwrap();

    // Skip sync offset on right
    for _ in 0..sync_offset {
        right_dec.next_frame().unwrap();
    }

    let left_region = DetectRegion {
        x_min: 0.4,
        x_max: 1.0,
        y_min: 0.3,
        y_max: 1.0,
    };
    let right_region = DetectRegion {
        x_min: 0.0,
        x_max: 0.6,
        y_min: 0.3,
        y_max: 1.0,
    };

    let mut all_points = Vec::new();
    let mut left_undistort: Option<GpuUndistort> = None;
    let mut right_undistort: Option<GpuUndistort> = None;

    // Sample frames every 50 frames, up to 20 frame pairs
    let mut frame_idx = 0u64;
    let mut pairs_done = 0;
    while pairs_done < 20 {
        let left_yuv = match left_dec.next_frame().unwrap() {
            Some(f) => f,
            None => break,
        };
        let right_yuv = match right_dec.next_frame().unwrap() {
            Some(f) => f,
            None => break,
        };
        frame_idx += 1;

        if !frame_idx.is_multiple_of(50) {
            continue;
        }

        let (lw, lh) = (left_yuv.width, left_yuv.height);
        let (rw, rh) = (right_yuv.width, right_yuv.height);

        let lu = left_undistort.get_or_insert_with(|| GpuUndistort::new(gpu, lw, lh));
        let ru = right_undistort.get_or_insert_with(|| GpuUndistort::new(gpu, rw, rh));

        let left_rgba = lu.undistort(gpu, &left_yuv.y, &left_yuv.u, &left_yuv.v, &cal.left);
        let right_rgba = ru.undistort(gpu, &right_yuv.y, &right_yuv.u, &right_yuv.v, &cal.right);

        let (kp_l, desc_l) = features::detect(&left_rgba, lw, lh, Some(left_region), 2000);
        let (kp_r, desc_r) = features::detect(&right_rgba, rw, rh, Some(right_region), 2000);

        let matches = features::match_descriptors(&desc_l, &desc_r, 0.7);

        for m in &matches {
            let lp = &kp_l[m.left_idx];
            let rp = &kp_r[m.right_idx];
            // Swap convention: right camera -> left plane, left camera -> right plane
            all_points.push(MatchedPoint {
                left: geometry::normalize_to_plane(rp.x as f64, rp.y as f64, rw, rh),
                right: geometry::normalize_to_plane(lp.x as f64, lp.y as f64, lw, lh),
            });
        }

        eprintln!(
            "frame {frame_idx}: {} left kp, {} right kp, {} matches (total: {})",
            kp_l.len(),
            kp_r.len(),
            matches.len(),
            all_points.len()
        );
        pairs_done += 1;
    }

    all_points
}
