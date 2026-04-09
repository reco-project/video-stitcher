//! `reco calibrate` subcommand.
//!
//! Extracts frames from two video files, runs the calibration pipeline,
//! and writes a match.json file compatible with `reco stitch`.

use std::path::Path;

use reco_calibrate::types::YuvFrame;
use reco_calibrate::{CalibrateError, CalibrationConfig};
use reco_core::calibration::CameraParams;
use reco_core::gpu::GpuContext;

/// Load a lens profile JSON and extract CameraParams.
///
/// Accepts v1 match.json uniforms format (`fx/fy/cx/cy/d`) or
/// Gyroflow/reco profile format (`camera_matrix/distortion_coeffs/resolution`).
fn load_camera_params(path: &str) -> anyhow::Result<CameraParams> {
    let json_str = std::fs::read_to_string(path)?;
    let v: serde_json::Value = serde_json::from_str(&json_str)?;

    // Try v1 uniforms format first (flat with "d" array)
    if v.get("fx").is_some() && v.get("d").is_some() {
        let params: CameraParams = serde_json::from_str(&json_str)?;
        return Ok(params);
    }

    // Try Gyroflow/reco lens profile format
    if let Some(cm) = v.get("camera_matrix") {
        let res = v
            .get("resolution")
            .or_else(|| v.get("calib_dimension"))
            .ok_or_else(|| anyhow::anyhow!("missing 'resolution' in lens profile"))?;

        let width = res
            .get("width")
            .or_else(|| res.get("w"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("missing 'width' in resolution"))?
            as u32;
        let height = res
            .get("height")
            .or_else(|| res.get("h"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("missing 'height' in resolution"))?
            as u32;

        let fx = cm["fx"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("missing 'fx' in camera_matrix"))?;
        let fy = cm["fy"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("missing 'fy' in camera_matrix"))?;
        let cx = cm["cx"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("missing 'cx' in camera_matrix"))?;
        let cy = cm["cy"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("missing 'cy' in camera_matrix"))?;

        let dc = v
            .get("distortion_coeffs")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("missing 'distortion_coeffs' in lens profile"))?;

        anyhow::ensure!(dc.len() >= 4, "need at least 4 distortion coefficients");
        let d = [
            dc[0].as_f64().unwrap_or(0.0),
            dc[1].as_f64().unwrap_or(0.0),
            dc[2].as_f64().unwrap_or(0.0),
            dc[3].as_f64().unwrap_or(0.0),
        ];

        return Ok(CameraParams {
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            d,
        });
    }

    anyhow::bail!(
        "unrecognized lens profile format in '{path}'. \
         Expected v1 uniforms (fx/fy/cx/cy/d) or reco profile (camera_matrix/distortion_coeffs/resolution)."
    );
}

/// Extract YUV420P frames from a video at the given frame indices.
///
/// Seeks to each target timestamp and grabs the first decoded frame.
/// The exact frame may differ slightly from the target index (lands on
/// the nearest keyframe), but both cameras use the same indices so the
/// left/right pairing and sync offset are preserved.
fn extract_frames(video_path: &str, frame_indices: &[u64]) -> anyhow::Result<Vec<YuvFrame>> {
    use reco_io::ffmpeg::decoder::VideoDecoder;

    let mut decoder = VideoDecoder::open(Path::new(video_path))?;
    let fps = decoder.fps();
    let mut frames = Vec::with_capacity(frame_indices.len());

    for &target_idx in frame_indices {
        let target_secs = target_idx as f64 / fps;
        decoder.seek_to_secs(target_secs)?;

        // Decode forward until we reach the exact target frame.
        // seek_to_secs lands on the nearest keyframe which may be
        // several frames before the target.
        let mut last_frame = None;
        while let Some(yuv) = decoder.next_frame()? {
            let frame_time = yuv.timestamp_us as f64 / 1_000_000.0;
            last_frame = Some(YuvFrame {
                y: yuv.y,
                u: yuv.u,
                v: yuv.v,
                width: yuv.width,
                height: yuv.height,
            });
            // Stop once we've reached or passed the target time
            if frame_time >= target_secs - 0.5 / fps {
                break;
            }
        }
        if let Some(f) = last_frame {
            frames.push(f);
        }
    }

    Ok(frames)
}

/// Run the calibrate subcommand.
#[allow(clippy::too_many_arguments)]
pub fn run_calibrate(
    left: &str,
    right: &str,
    left_profile: Option<&str>,
    right_profile: Option<&str>,
    num_frames: usize,
    iterations: usize,
    auto_imu: bool,
    auto_sync: bool,
    sync_offset: i64,
    skip_start: f64,
    skip_end: f64,
    akaze_threshold: f64,
    lowe_ratio: f64,
    detect_x: f64,
    detect_y_min: f64,
    detect_y_max: f64,
    lock_cam_d: bool,
    lock_z_rx: bool,
    trim: f64,
    seam_sigma: f64,
    debug_dir: Option<&str>,
    output: &str,
) -> anyhow::Result<()> {
    reco_io::init();

    // IMU telemetry extraction (if requested)
    let mut imu_sync_offset: Option<f64> = None;
    let mut imu_xrz_seed: Option<f64> = None;
    let mut imu_xrx_seed: Option<f64> = None;
    let mut imu_zrx_seed: Option<f64> = None;
    let mut enable_x_rx = false;

    if auto_imu {
        eprintln!("Extracting IMU telemetry...");
        let left_path = std::path::Path::new(left);
        let right_path = std::path::Path::new(right);

        match (
            reco_calibrate::telemetry::extract(left_path),
            reco_calibrate::telemetry::extract(right_path),
        ) {
            (Ok(left_imu), Ok(right_imu)) => {
                // Gyro cross-correlation for sync offset
                if let Some(offset) =
                    reco_calibrate::telemetry::estimate_sync_offset(&left_imu, &right_imu)
                {
                    eprintln!("  IMU sync offset: {offset:.3}s");
                    imu_sync_offset = Some(offset);
                }

                // Differential orientation for xRz, xRx, zRx seeds
                if let Some((roll, pitch, tilt)) =
                    reco_calibrate::telemetry::differential_orientation(&left_imu, &right_imu)
                {
                    eprintln!(
                        "  Differential roll: {:.2} deg, pitch: {:.2} deg, tilt: {:.2} deg",
                        roll.to_degrees(),
                        pitch.to_degrees(),
                        tilt.to_degrees(),
                    );
                    imu_xrz_seed = Some(roll);
                    imu_zrx_seed = Some(tilt);
                    if pitch.abs() > 2.0_f64.to_radians() {
                        eprintln!("  Pitch > 2 deg, enabling x_rx seeded at {pitch:.4} rad");
                        enable_x_rx = true;
                        imu_xrx_seed = Some(pitch);
                    }
                }

                // Rig tilt from gravity
                if let Some(tilt) = reco_calibrate::telemetry::rig_tilt(&left_imu) {
                    eprintln!("  Rig tilt: {:.1} deg", tilt.to_degrees());
                }
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("  IMU extraction failed: {e} (continuing without IMU)");
            }
        }
    }

    // Load lens profiles (manual or auto-detect)
    log::info!("loading lens profiles...");
    let (left_params, right_params) = if let Some(lp) = left_profile {
        let left_p = load_camera_params(lp)?;
        let right_p = if let Some(rp) = right_profile {
            load_camera_params(rp)?
        } else {
            left_p.clone() // same profile for both cameras
        };
        (left_p, right_p)
    } else {
        // Auto-detect from video metadata using telemetry-parser
        eprintln!("Auto-detecting lens profiles...");
        let lens_db = reco_calibrate::lens_database::LensDatabase::load_embedded();

        let left_path = Path::new(left);
        let right_path = Path::new(right);

        // Open decoders to get resolution
        let ld = reco_io::ffmpeg::decoder::VideoDecoder::open(left_path)?;
        let rd = reco_io::ffmpeg::decoder::VideoDecoder::open(right_path)?;
        let (lw, lh) = (ld.width(), ld.height());
        let (rw, rh) = (rd.width(), rd.height());
        drop(ld);
        drop(rd);

        // Use telemetry-parser camera identification
        let left_tel = reco_calibrate::telemetry::extract(left_path).ok();
        let right_tel = reco_calibrate::telemetry::extract(right_path).ok();

        let left_p = if let Some(ref tel) = left_tel {
            lens_db
                .find_from_telemetry(&tel.camera_type, tel.camera_model.as_deref(), lw, lh)
                .ok_or_else(|| anyhow::anyhow!(
                    "no lens profile found for left camera: {} {} {}x{}. Use --left-profile to specify manually.",
                    tel.camera_type, tel.camera_model.as_deref().unwrap_or("?"), lw, lh
                ))?
        } else {
            anyhow::bail!(
                "cannot detect left camera type. Use --left-profile to specify the lens profile."
            )
        };

        let right_p = if let Some(ref tel) = right_tel {
            lens_db
                .find_from_telemetry(&tel.camera_type, tel.camera_model.as_deref(), rw, rh)
                .unwrap_or_else(|| {
                    eprintln!("  right camera: no profile found, using left camera profile");
                    left_p.clone()
                })
        } else {
            left_p.clone()
        };

        eprintln!(
            "  left:  {} {}x{}",
            left_tel
                .as_ref()
                .map(|t| t.camera_type.as_str())
                .unwrap_or("?"),
            left_p.width,
            left_p.height
        );
        eprintln!(
            "  right: {} {}x{}",
            right_tel
                .as_ref()
                .map(|t| t.camera_type.as_str())
                .unwrap_or("?"),
            right_p.width,
            right_p.height
        );

        (left_p, right_p)
    };
    log::info!(
        "left: {}x{}, right: {}x{}",
        left_params.width,
        left_params.height,
        right_params.width,
        right_params.height
    );

    // Determine frame count from both videos
    let left_decoder = reco_io::ffmpeg::decoder::VideoDecoder::open(Path::new(left))?;
    let fps = left_decoder.fps();
    let total_frames = {
        let right_decoder = reco_io::ffmpeg::decoder::VideoDecoder::open(Path::new(right))?;
        let left_est = left_decoder
            .duration_secs()
            .map(|d| (d * left_decoder.fps()) as u64)
            .unwrap_or((left_decoder.fps() * 60.0) as u64);
        let right_est = right_decoder
            .duration_secs()
            .map(|d| (d * right_decoder.fps()) as u64)
            .unwrap_or((right_decoder.fps() * 60.0) as u64);
        left_est.min(right_est).max(100)
    };
    drop(left_decoder);

    let frame_indices = reco_calibrate::sampling::select_frame_indices(
        total_frames,
        fps,
        num_frames,
        skip_start,
        skip_end,
    );

    // Determine sync offset: IMU > audio > manual
    let effective_sync = if let Some(imu_offset) = imu_sync_offset {
        let frames = (-imu_offset * fps).round() as i64;
        eprintln!("  IMU sync: {imu_offset:.3}s = {frames} frames @ {fps:.1}fps");
        frames
    } else if auto_sync {
        eprintln!("Auto-detecting sync from audio...");
        match reco_calibrate::audio_sync::estimate_sync_offset(
            Path::new(left),
            Path::new(right),
            fps,
            &reco_calibrate::audio_sync::AudioSyncConfig::default(),
        ) {
            Ok(result) => {
                let frames = result.offset_frames.round() as i64;
                eprintln!(
                    "  Audio sync: {:.3}s = {} frames (confidence={:.2})",
                    result.offset_secs, frames, result.confidence
                );
                frames
            }
            Err(e) => {
                eprintln!("  Audio sync failed: {e}, falling back to --sync-offset {sync_offset}");
                sync_offset
            }
        }
    } else {
        sync_offset
    };

    // Apply sync offset
    let (left_indices, right_indices) = if effective_sync >= 0 {
        let offset = effective_sync as u64;
        (
            frame_indices.clone(),
            frame_indices
                .iter()
                .map(|&i| i + offset)
                .collect::<Vec<_>>(),
        )
    } else {
        let offset = (-effective_sync) as u64;
        (
            frame_indices
                .iter()
                .map(|&i| i + offset)
                .collect::<Vec<_>>(),
            frame_indices.clone(),
        )
    };

    log::info!(
        "extracting {} frames (fps: {fps:.1}, sync_offset: {effective_sync})",
        left_indices.len(),
    );

    eprintln!("Extracting frames from left video...");
    let left_frames = extract_frames(left, &left_indices)?;
    eprintln!("Extracting frames from right video...");
    let right_frames = extract_frames(right, &right_indices)?;

    let pair_count = left_frames.len().min(right_frames.len());
    anyhow::ensure!(pair_count > 0, "no frames could be extracted from videos");

    // Init GPU
    let gpu = pollster::block_on(GpuContext::new())
        .map_err(|e| anyhow::anyhow!("GPU init failed: {e}"))?;
    log::info!("GPU: {}", gpu.gpu_name());

    // Debug: save GPU-undistorted frames
    if let Some(dir) = debug_dir {
        std::fs::create_dir_all(dir)?;
        let (w, h) = (left_frames[0].width, left_frames[0].height);
        let undistort = reco_core::undistort::GpuUndistort::new(&gpu, w, h);
        for (i, (lf, rf)) in left_frames.iter().zip(right_frames.iter()).enumerate() {
            let l_rgba = undistort.undistort(&gpu, &lf.y, &lf.u, &lf.v, &left_params);
            let r_rgba = undistort.undistort(&gpu, &rf.y, &rf.u, &rf.v, &right_params);
            save_rgba_png(&l_rgba, w, h, &format!("{dir}/frame_{i:02}_left.png"))?;
            save_rgba_png(&r_rgba, w, h, &format!("{dir}/frame_{i:02}_right.png"))?;
        }
        eprintln!("Debug frames saved to {dir}/ (GPU-undistorted)");
    }

    let frame_pairs: Vec<(YuvFrame, YuvFrame)> =
        left_frames.into_iter().zip(right_frames).collect();

    let config = CalibrationConfig {
        num_frames,
        iterations,
        skip_start_secs: skip_start,
        skip_end_secs: skip_end,
        akaze_threshold,
        lowe_ratio,
        spatial_x_threshold: detect_x,
        detect_y_min,
        detect_y_max,
        lock_cam_d,
        lock_z_rx,
        trim_fraction: trim,
        seam_sigma,
        imu_xrz_seed,
        enable_x_rx,
        imu_xrx_seed,
        imu_zrx_seed,
        ..Default::default()
    };

    eprintln!(
        "Calibrating with {} frame pairs, {} iterations...",
        frame_pairs.len(),
        config.iterations
    );

    let result =
        reco_calibrate::calibrate(&gpu, &frame_pairs, &left_params, &right_params, &config)
            .map_err(|e: CalibrateError| anyhow::anyhow!("{e}"))?;

    // Write output
    let json = serde_json::to_string_pretty(&result.calibration)?;
    std::fs::write(output, &json)?;

    // Print diagnostics
    eprintln!("\nCalibration results:");
    eprintln!("  Output:          {output}");
    eprintln!(
        "  Frames used:     {}/{}",
        result.frames_used,
        frame_pairs.len()
    );
    eprintln!("  Total matches:   {}", result.total_matches);
    eprintln!("  Confidence:      {:.1}%", result.confidence * 100.0);
    eprintln!("  Residual error:  {:.6}", result.residual_error);
    eprintln!("\nPlacement parameters:");
    let l = &result.calibration.layout;
    eprintln!("  cameraAxisOffset: {:.4}", l.camera_axis_offset);
    eprintln!("  intersect:        {:.4}", l.intersect);
    eprintln!("  xTy:              {:.6}", l.x_ty);
    eprintln!("  xRz:              {:.6}", l.x_rz);
    eprintln!("  zRx:              {:.6}", l.z_rx);
    eprintln!("  zRz:              {:.6}", l.z_rz);

    if !result.per_frame.is_empty() {
        eprintln!("\nPer-frame statistics:");
        for (i, fm) in result.per_frame.iter().enumerate() {
            eprintln!(
                "  frame {i}: {} keypoints (L:{}/R:{}), {} -> {} -> {} -> {} matches",
                fm.points.len(),
                fm.keypoints_left,
                fm.keypoints_right,
                fm.raw_matches,
                fm.post_ratio_test,
                fm.post_spatial_filter,
                fm.post_ransac,
            );
        }
    }

    // Debug: save per-frame match data as JSON + keypoint visualizations
    if let Some(dir) = debug_dir {
        let matches_json = serde_json::to_string_pretty(&result.per_frame)?;
        std::fs::write(format!("{dir}/matches.json"), &matches_json)?;

        // Draw matched keypoints on the debug frames
        let (w, h) = (frame_pairs[0].0.width, frame_pairs[0].0.height);
        for (i, fm) in result.per_frame.iter().enumerate() {
            let left_path = format!("{dir}/frame_{i:02}_left.png");
            let right_path = format!("{dir}/frame_{i:02}_right.png");
            if std::path::Path::new(&left_path).exists()
                && std::path::Path::new(&right_path).exists()
            {
                // Load the saved debug frames and draw matches
                let mut left_img = image::open(&left_path)?.to_rgba8();
                let mut right_img = image::open(&right_path)?.to_rgba8();

                // Convert plane coords back to pixel coords for visualization.
                // Plane coords: x in [-0.5, 0.5], y in [-h/(2w), h/(2w)]
                // Remember the swap: .left is right camera (x-plane), .right is left camera (z-plane)
                for pt in &fm.points {
                    // Right camera point on left image (before swap, .left = right cam)
                    let rx = ((pt.left[0] + 0.5) * w as f64) as i32;
                    let ry = ((pt.left[1] / (h as f64 / w as f64)) + 0.5) * h as f64;
                    let ry = ry as i32;
                    draw_cross(&mut right_img, rx, ry, [0, 255, 0, 255]); // green

                    // Left camera point on right image (before swap, .right = left cam)
                    let lx = ((pt.right[0] + 0.5) * w as f64) as i32;
                    let ly = ((pt.right[1] / (h as f64 / w as f64)) + 0.5) * h as f64;
                    let ly = ly as i32;
                    draw_cross(&mut left_img, lx, ly, [255, 0, 0, 255]); // red
                }

                left_img.save(format!("{dir}/matches_{i:02}_left.png"))?;
                right_img.save(format!("{dir}/matches_{i:02}_right.png"))?;
            }
        }

        eprintln!("Match visualizations saved to {dir}/matches_*_{{left,right}}.png");
    }

    Ok(())
}

/// Draw a cross marker at (cx, cy) on an RGBA image.
fn draw_cross(img: &mut image::RgbaImage, cx: i32, cy: i32, color: [u8; 4]) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let size = 8;
    let thickness = 2;
    for d in -size..=size {
        for t in -thickness..=thickness {
            // Horizontal arm
            let x = cx + d;
            let y = cy + t;
            if x >= 0 && x < w && y >= 0 && y < h {
                img.put_pixel(x as u32, y as u32, image::Rgba(color));
            }
            // Vertical arm
            let x = cx + t;
            let y = cy + d;
            if x >= 0 && x < w && y >= 0 && y < h {
                img.put_pixel(x as u32, y as u32, image::Rgba(color));
            }
        }
    }
}

/// Save RGBA pixel data as a PNG.
fn save_rgba_png(rgba: &[u8], width: u32, height: u32, path: &str) -> anyhow::Result<()> {
    let img = image::RgbaImage::from_raw(width, height, rgba.to_vec())
        .ok_or_else(|| anyhow::anyhow!("invalid frame dimensions"))?;
    img.save(path)?;
    Ok(())
}
