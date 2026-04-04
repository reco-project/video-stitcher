//! Visualize matched points with their normalized plane coordinates
//! (the exact values fed to the optimizer).
//!
//! Draws keypoints on the undistorted frames with both pixel coords
//! and plane coords labeled, and dumps the point list as JSON.
//!
//! Usage: cargo run --release -p reco-calibrate --example dump_points -- \
//!   <left.mp4> <right.mp4> <match.json> <sync_offset> <output_dir>

use reco_calibrate::features::{self, DetectRegion};
use reco_calibrate::geometry;
use reco_core::calibration::MatchCalibration;
use reco_core::gpu::GpuContext;
use reco_core::undistort::GpuUndistort;

fn main() {
    reco_io::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "Usage: {} <left.mp4> <right.mp4> <match.json> <sync_offset> <output_dir>",
            args[0]
        );
        std::process::exit(1);
    }

    let sync_offset: u64 = args[4].parse().expect("invalid sync_offset");
    let out_dir = &args[5];
    std::fs::create_dir_all(out_dir).unwrap();
    let target_frame: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(300);

    let json_str = std::fs::read_to_string(&args[3]).unwrap();
    let cal: MatchCalibration = serde_json::from_str(&json_str).expect("invalid match.json");

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
    let left_undistort = GpuUndistort::new(&gpu, lw, lh);
    let right_undistort = GpuUndistort::new(&gpu, rw, rh);
    let left_rgba =
        left_undistort.undistort(&gpu, &left_yuv.y, &left_yuv.u, &left_yuv.v, &cal.left);
    let right_rgba =
        right_undistort.undistort(&gpu, &right_yuv.y, &right_yuv.u, &right_yuv.v, &cal.right);

    // Detect with overlap regions
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

    let (kp_l, desc_l) = features::detect(&left_rgba, lw, lh, Some(left_region), 2000);
    let (kp_r, desc_r) = features::detect(&right_rgba, rw, rh, Some(right_region), 2000);
    let matches = features::match_descriptors(&desc_l, &desc_r, 0.7);

    eprintln!("Frame {target_frame}: {} matches", matches.len());

    // Build the matched points with the swap convention (same as lib.rs)
    // Right camera -> left plane (x-plane), Left camera -> right plane (z-plane)
    let mut points_json = Vec::new();

    for (i, m) in matches.iter().enumerate() {
        let lp = &kp_l[m.left_idx];
        let rp = &kp_r[m.right_idx];

        // Plane coordinates (what the optimizer sees)
        let left_plane = geometry::normalize_to_plane(rp.x as f64, rp.y as f64, rw, rh);
        let right_plane = geometry::normalize_to_plane(lp.x as f64, lp.y as f64, lw, lh);

        points_json.push(serde_json::json!({
            "idx": i,
            "left_pixel": [lp.x, lp.y],
            "right_pixel": [rp.x, rp.y],
            "left_plane_coords": left_plane,
            "right_plane_coords": right_plane,
            "hamming_distance": m.distance,
        }));
    }

    // Dump points as JSON
    let json = serde_json::to_string_pretty(&points_json).unwrap();
    let json_path = format!("{out_dir}/matched_points.json");
    std::fs::write(&json_path, &json).unwrap();
    eprintln!("Points saved to {json_path}");

    // Also save the undistorted frames for the Python visualization
    let left_img = image::RgbaImage::from_raw(lw, lh, left_rgba).unwrap();
    let right_img = image::RgbaImage::from_raw(rw, rh, right_rgba).unwrap();
    left_img
        .save(format!("{out_dir}/left_undistorted.png"))
        .unwrap();
    right_img
        .save(format!("{out_dir}/right_undistorted.png"))
        .unwrap();
    eprintln!("Frames saved to {out_dir}/");

    // Print the points
    eprintln!("\nMatched points (optimizer input):");
    eprintln!(
        "idx | left_px (in right cam) | right_px (in left cam) | x-plane [x,y]     | z-plane [x,y]"
    );
    eprintln!(
        "----+------------------------+------------------------+-------------------+-------------------"
    );
    for (i, m) in matches.iter().enumerate() {
        let lp = &kp_l[m.left_idx];
        let rp = &kp_r[m.right_idx];
        let left_plane = geometry::normalize_to_plane(rp.x as f64, rp.y as f64, rw, rh);
        let right_plane = geometry::normalize_to_plane(lp.x as f64, lp.y as f64, lw, lh);
        eprintln!(
            "{:3} | ({:7.1}, {:7.1})      | ({:7.1}, {:7.1})      | [{:+.4}, {:+.4}] | [{:+.4}, {:+.4}]",
            i, lp.x, lp.y, rp.x, rp.y, left_plane[0], left_plane[1], right_plane[0], right_plane[1],
        );
    }
}
