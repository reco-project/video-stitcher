//! Dump matched feature points with plane coordinates for debugging.
//!
//! Extracts a single frame pair, detects AKAZE features, matches them,
//! and writes the matched points as JSON along with undistorted frame
//! images. Uses the same detection parameters as the calibration pipeline.
//!
//! Usage:
//!   cargo run --release -p reco-calibrate --example dump_points -- \
//!     <left.mp4> <right.mp4> <match.json> <sync_offset> <output_dir> [frame]

use reco_calibrate::features::{self, DetectRegion};
use reco_calibrate::geometry;
use reco_calibrate::types::CalibrationConfig;
use reco_core::calibration::Calibration;
use reco_core::gpu::GpuContext;
use reco_core::lens::undistort::GpuUndistort;

fn main() {
    reco_io::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "Usage: {} <left.mp4> <right.mp4> <match.json> <sync_offset> <output_dir> [frame]",
            args[0]
        );
        std::process::exit(1);
    }

    let sync_offset: u64 = args[4].parse().expect("invalid sync_offset");
    let out_dir = &args[5];
    std::fs::create_dir_all(out_dir).unwrap();
    let target_frame: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(300);

    let json_str = std::fs::read_to_string(&args[3]).unwrap();
    let cal: Calibration = serde_json::from_str(&json_str).expect("invalid match.json");

    // Use calibration pipeline defaults for detection parameters
    let config = CalibrationConfig::default();

    // Decode target frame
    let mut left_dec =
        reco_io::ffmpeg::decoder::VideoDecoder::open(std::path::Path::new(&args[1])).unwrap();
    let mut right_dec =
        reco_io::ffmpeg::decoder::VideoDecoder::open(std::path::Path::new(&args[2])).unwrap();

    for _ in 0..target_frame {
        left_dec.next_frame().unwrap();
    }
    for _ in 0..(target_frame + sync_offset) {
        right_dec.next_frame().unwrap();
    }

    let left_yuv = left_dec.next_frame().unwrap().expect("no left frames");
    let right_yuv = right_dec.next_frame().unwrap().expect("no right frames");
    let (lw, lh) = (left_yuv.width, left_yuv.height);
    let (rw, rh) = (right_yuv.width, right_yuv.height);

    // GPU undistort
    let gpu = pollster::block_on(GpuContext::new()).expect("no GPU");
    let left_undistort = GpuUndistort::new(&gpu, lw, lh, lw as f32 / lh as f32);
    let right_undistort = GpuUndistort::new(&gpu, rw, rh, rw as f32 / rh as f32);
    let left_rgba =
        left_undistort.undistort(&gpu, &left_yuv.y, &left_yuv.u, &left_yuv.v, &cal.lenses[0]);
    let right_rgba = right_undistort.undistort(
        &gpu,
        &right_yuv.y,
        &right_yuv.u,
        &right_yuv.v,
        &cal.lenses[1],
    );

    // Detect features using pipeline defaults
    let left_region = DetectRegion {
        x_min: config.matching.spatial_x_threshold as f32,
        x_max: 1.0,
        y_min: config.akaze.detect_y_min as f32,
        y_max: config.akaze.detect_y_max as f32,
    };
    let right_region = DetectRegion {
        x_min: 0.0,
        x_max: 1.0 - config.matching.spatial_x_threshold as f32,
        y_min: config.akaze.detect_y_min as f32,
        y_max: config.akaze.detect_y_max as f32,
    };

    eprintln!(
        "Detect: x_thresh={}, y=[{}, {}], akaze={}, lowe={}",
        config.matching.spatial_x_threshold,
        config.akaze.detect_y_min,
        config.akaze.detect_y_max,
        config.akaze.threshold,
        config.matching.lowe_ratio,
    );

    let (kp_l, desc_l) = features::detect(
        &left_rgba,
        lw,
        lh,
        Some(left_region),
        config.akaze.max_keypoints,
        config.akaze.threshold,
    );
    let (kp_r, desc_r) = features::detect(
        &right_rgba,
        rw,
        rh,
        Some(right_region),
        config.akaze.max_keypoints,
        config.akaze.threshold,
    );
    let matches = features::match_descriptors(&desc_l, &desc_r, config.matching.lowe_ratio);

    eprintln!(
        "Frame {target_frame}: {} raw matches (max_y_disp={})",
        matches.len(),
        config.matching.max_y_disparity,
    );

    // Build matched points with y-disparity filter and swap convention
    // Right camera -> left plane (x-plane), Left camera -> right plane (z-plane)
    let mut points_json = Vec::new();
    let mut rejected = 0;

    for m in matches.iter() {
        let lp = &kp_l[m.left_idx];
        let rp = &kp_r[m.right_idx];

        // Y-disparity filter
        let ly_norm = lp.y as f64 / lh as f64;
        let ry_norm = rp.y as f64 / rh as f64;
        if (ly_norm - ry_norm).abs() > config.matching.max_y_disparity {
            rejected += 1;
            continue;
        }

        let left_plane = geometry::normalize_to_plane(rp.x as f64, rp.y as f64, rw, rh);
        let right_plane = geometry::normalize_to_plane(lp.x as f64, lp.y as f64, lw, lh);

        let idx = points_json.len();
        points_json.push(serde_json::json!({
            "idx": idx,
            "left_pixel": [lp.x, lp.y],
            "right_pixel": [rp.x, rp.y],
            "left_plane_coords": left_plane,
            "right_plane_coords": right_plane,
            "hamming_distance": m.distance,
        }));
    }
    eprintln!(
        "After y-disparity filter: {} kept, {} rejected",
        points_json.len(),
        rejected
    );

    // Dump points as JSON
    let json = serde_json::to_string_pretty(&points_json).unwrap();
    let json_path = format!("{out_dir}/matched_points.json");
    std::fs::write(&json_path, &json).unwrap();
    eprintln!("Points saved to {json_path}");

    // Save undistorted frames for visualization
    let left_img = image::RgbaImage::from_raw(lw, lh, left_rgba).unwrap();
    let right_img = image::RgbaImage::from_raw(rw, rh, right_rgba).unwrap();
    left_img
        .save(format!("{out_dir}/left_undistorted.png"))
        .unwrap();
    right_img
        .save(format!("{out_dir}/right_undistorted.png"))
        .unwrap();
    eprintln!("Frames saved to {out_dir}/");
}
