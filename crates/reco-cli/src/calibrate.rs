//! `reco calibrate` subcommand.
//!
//! Thin wrapper around [`reco_calibrate::pipeline::CalibrationPipeline`].
//! Handles file I/O (video decode, audio extraction, debug output) while
//! all calibration logic lives in the library.

use std::path::Path;

use reco_calibrate::CalibrationConfig;
use reco_calibrate::pipeline::{CalibrationPipeline, VideoInfo};
use reco_calibrate::types::YuvFrame;
use reco_core::gpu::GpuContext;
use reco_io::ffmpeg::calibration_io;

/// Run the calibrate subcommand.
#[allow(clippy::too_many_arguments)]
pub fn run_calibrate(
    left: &str,
    right: &str,
    left_profile: Option<&str>,
    right_profile: Option<&str>,
    num_frames: usize,
    no_auto_imu: bool,
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

    // Probe video metadata
    let left_probe = calibration_io::probe_video(Path::new(left))?;
    let right_probe = calibration_io::probe_video(Path::new(right))?;

    let left_info = VideoInfo {
        path: left.into(),
        width: left_probe.width,
        height: left_probe.height,
        fps: left_probe.fps,
        total_frames: left_probe.total_frames,
    };
    let right_info = VideoInfo {
        path: right.into(),
        width: right_probe.width,
        height: right_probe.height,
        fps: right_probe.fps,
        total_frames: right_probe.total_frames,
    };
    let fps = left_info.fps;

    let config = CalibrationConfig {
        num_frames,
        skip_start_secs: skip_start,
        skip_end_secs: skip_end,
        akaze: reco_calibrate::AkazeConfig {
            threshold: akaze_threshold,
            detect_y_min,
            detect_y_max,
            ..Default::default()
        },
        matching: reco_calibrate::MatchConfig {
            lowe_ratio,
            spatial_x_threshold: detect_x,
            ..Default::default()
        },
        optimizer: reco_calibrate::OptimizerConfig {
            lock_cam_d,
            lock_z_rx,
            trim_fraction: trim,
            seam_sigma,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut pipeline = CalibrationPipeline::new(left_info, right_info, config);

    // Step 1: Lens profiles
    if let Some(lp) = left_profile {
        let rp = right_profile.map(Path::new);
        let (lp, rp) = pipeline.load_profiles(Path::new(lp), rp)?;
        eprintln!("  left:  {}x{}", lp.width, lp.height);
        eprintln!("  right: {}x{}", rp.width, rp.height);
    } else {
        eprintln!("Auto-detecting lens profiles...");
        let (lp, rp) = pipeline.detect_profiles()?;
        eprintln!("  left:  {}x{}", lp.width, lp.height);
        eprintln!("  right: {}x{}", rp.width, rp.height);
    }

    // Step 2: Sync - priority: IMU > audio > manual
    if !no_auto_imu {
        eprintln!("Extracting IMU telemetry...");
        match pipeline.imu_sync() {
            Ok(Some(frames)) => {
                eprintln!("  IMU sync: {frames} frames @ {fps:.1}fps");
            }
            Ok(None) => {
                eprintln!("  IMU telemetry available but sync failed, trying audio...");
                try_audio_sync(&mut pipeline, left, right, fps, auto_sync, sync_offset)?;
            }
            Err(e) => {
                eprintln!("  IMU extraction failed: {e}");
                try_audio_sync(&mut pipeline, left, right, fps, auto_sync, sync_offset)?;
            }
        }
    } else {
        try_audio_sync(&mut pipeline, left, right, fps, auto_sync, sync_offset)?;
    }

    // Step 3: Frame indices (sync already applied by pipeline)
    let (left_indices, right_indices) = pipeline.frame_indices();
    log::info!(
        "extracting {} frames (fps: {fps:.1}, sync_offset: {})",
        left_indices.len(),
        pipeline.sync_offset(),
    );

    // Step 4: Extract frames
    eprintln!("Extracting frames from left video...");
    let left_frames = calibration_io::extract_frames(Path::new(left), &left_indices)?;
    eprintln!("Extracting frames from right video...");
    let right_frames = calibration_io::extract_frames(Path::new(right), &right_indices)?;

    let pair_count = left_frames.len().min(right_frames.len());
    anyhow::ensure!(pair_count > 0, "no frames could be extracted from videos");

    // Init GPU
    let gpu = pollster::block_on(GpuContext::new())
        .map_err(|e| anyhow::anyhow!("GPU init failed: {e}"))?;
    log::info!("GPU: {}", gpu.gpu_name());

    // Debug: save GPU-undistorted frames
    if let Some(dir) = debug_dir {
        save_debug_frames(
            &gpu,
            &left_frames,
            &right_frames,
            pipeline
                .left_params()
                .ok_or_else(|| anyhow::anyhow!("lens params not available for left camera"))?,
            pipeline
                .right_params()
                .ok_or_else(|| anyhow::anyhow!("lens params not available for right camera"))?,
            dir,
        )?;
    }

    let frame_pairs: Vec<(YuvFrame, YuvFrame)> =
        left_frames.into_iter().zip(right_frames).collect();

    // Step 5: Calibrate
    eprintln!("Calibrating with {} frame pairs...", frame_pairs.len());
    let result = pipeline
        .calibrate(&gpu, &frame_pairs)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Write output
    let json = serde_json::to_string_pretty(&result.calibration)?;
    std::fs::write(output, &json)?;

    // Print diagnostics
    print_results(&result, output, frame_pairs.len());

    // Debug: save match visualizations
    if let Some(dir) = debug_dir {
        save_match_visualizations(&result, &frame_pairs, dir)?;
    }

    Ok(())
}

/// Try audio sync, falling back to manual offset.
fn try_audio_sync(
    pipeline: &mut CalibrationPipeline,
    left: &str,
    right: &str,
    fps: f64,
    auto_sync: bool,
    manual_offset: i64,
) -> anyhow::Result<()> {
    if auto_sync {
        eprintln!("Auto-detecting sync from audio...");
        match extract_and_sync_audio(pipeline, left, right) {
            Ok(frames) => {
                eprintln!("  Audio sync: {frames} frames @ {fps:.1}fps");
                return Ok(());
            }
            Err(e) => {
                eprintln!("  Audio sync failed: {e}, using manual offset {manual_offset}");
            }
        }
    }
    pipeline.set_sync_offset(manual_offset);
    Ok(())
}

/// Extract audio and run sync through the pipeline.
fn extract_and_sync_audio(
    pipeline: &mut CalibrationPipeline,
    left: &str,
    right: &str,
) -> anyhow::Result<i64> {
    let sample_rate = 44100u32;
    let left_samples = calibration_io::extract_audio_pcm(Path::new(left), sample_rate)?;
    let right_samples = calibration_io::extract_audio_pcm(Path::new(right), sample_rate)?;
    let frames = pipeline.audio_sync(&left_samples, &right_samples, sample_rate)?;
    Ok(frames)
}

// ---------------------------------------------------------------------------
// Debug visualization (CLI only)
// ---------------------------------------------------------------------------

fn save_debug_frames(
    gpu: &GpuContext,
    left_frames: &[YuvFrame],
    right_frames: &[YuvFrame],
    left_params: &reco_core::calibration::CameraParams,
    right_params: &reco_core::calibration::CameraParams,
    dir: &str,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let (w, h) = (left_frames[0].width, left_frames[0].height);
    let aspect = w as f32 / h as f32;
    let undistort = reco_core::undistort::GpuUndistort::new(gpu, w, h, aspect);
    for (i, (lf, rf)) in left_frames.iter().zip(right_frames.iter()).enumerate() {
        let l_rgba = undistort.undistort(gpu, &lf.y, &lf.u, &lf.v, left_params);
        let r_rgba = undistort.undistort(gpu, &rf.y, &rf.u, &rf.v, right_params);
        save_rgba_png(&l_rgba, w, h, &format!("{dir}/frame_{i:02}_left.png"))?;
        save_rgba_png(&r_rgba, w, h, &format!("{dir}/frame_{i:02}_right.png"))?;
    }
    eprintln!("Debug frames saved to {dir}/ (GPU-undistorted)");
    Ok(())
}

fn save_match_visualizations(
    result: &reco_calibrate::CalibrationResult,
    frame_pairs: &[(YuvFrame, YuvFrame)],
    dir: &str,
) -> anyhow::Result<()> {
    let (w, h) = (frame_pairs[0].0.width, frame_pairs[0].0.height);
    for (i, fm) in result.per_frame.iter().enumerate() {
        let left_path = format!("{dir}/frame_{i:02}_left.png");
        let right_path = format!("{dir}/frame_{i:02}_right.png");
        if !Path::new(&left_path).exists() || !Path::new(&right_path).exists() {
            continue;
        }
        let mut left_img = image::open(&left_path)?.to_rgba8();
        let mut right_img = image::open(&right_path)?.to_rgba8();

        for pt in &fm.points {
            let rx = ((pt.left[0] + 0.5) * w as f64) as i32;
            let ry = ((pt.left[1] / (h as f64 / w as f64)) + 0.5) * h as f64;
            draw_cross(&mut right_img, rx, ry as i32, [0, 255, 0, 255]);

            let lx = ((pt.right[0] + 0.5) * w as f64) as i32;
            let ly = ((pt.right[1] / (h as f64 / w as f64)) + 0.5) * h as f64;
            draw_cross(&mut left_img, lx, ly as i32, [255, 0, 0, 255]);
        }

        left_img.save(format!("{dir}/matches_{i:02}_left.png"))?;
        right_img.save(format!("{dir}/matches_{i:02}_right.png"))?;
    }
    eprintln!("Match visualizations saved to {dir}/matches_*_{{left,right}}.png");
    Ok(())
}

fn print_results(result: &reco_calibrate::CalibrationResult, output: &str, total_pairs: usize) {
    eprintln!("\nCalibration results:");
    eprintln!("  Output:          {output}");
    eprintln!("  Frames used:     {}/{total_pairs}", result.frames_used);
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
                fm.min_descriptors,
                fm.post_ratio_test,
                fm.post_spatial_filter,
                fm.post_ransac,
            );
        }
    }
}

fn draw_cross(img: &mut image::RgbaImage, cx: i32, cy: i32, color: [u8; 4]) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let size = 8;
    let thickness = 2;
    for d in -size..=size {
        for t in -thickness..=thickness {
            let x = cx + d;
            let y = cy + t;
            if x >= 0 && x < w && y >= 0 && y < h {
                img.put_pixel(x as u32, y as u32, image::Rgba(color));
            }
            let x = cx + t;
            let y = cy + d;
            if x >= 0 && x < w && y >= 0 && y < h {
                img.put_pixel(x as u32, y as u32, image::Rgba(color));
            }
        }
    }
}

fn save_rgba_png(rgba: &[u8], width: u32, height: u32, path: &str) -> anyhow::Result<()> {
    let img = image::RgbaImage::from_raw(width, height, rgba.to_vec())
        .ok_or_else(|| anyhow::anyhow!("invalid frame dimensions"))?;
    img.save(path)?;
    Ok(())
}
