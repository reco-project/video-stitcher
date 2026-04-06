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
fn extract_frames(video_path: &str, frame_indices: &[u64]) -> anyhow::Result<Vec<YuvFrame>> {
    use reco_io::ffmpeg::decoder::VideoDecoder;

    let mut decoder = VideoDecoder::open(Path::new(video_path))?;
    let mut frames = Vec::with_capacity(frame_indices.len());
    let mut frame_idx: u64 = 0;
    let mut target_ptr = 0;

    while target_ptr < frame_indices.len() {
        match decoder.next_frame()? {
            Some(yuv) => {
                if frame_idx == frame_indices[target_ptr] {
                    frames.push(YuvFrame {
                        y: yuv.y,
                        u: yuv.u,
                        v: yuv.v,
                        width: yuv.width,
                        height: yuv.height,
                    });
                    target_ptr += 1;

                    while target_ptr < frame_indices.len() && frame_indices[target_ptr] == frame_idx
                    {
                        target_ptr += 1;
                    }
                }
                frame_idx += 1;
            }
            None => break,
        }
    }

    Ok(frames)
}

/// Run the calibrate subcommand.
#[allow(clippy::too_many_arguments)]
pub fn run_calibrate(
    left: &str,
    right: &str,
    left_profile: &str,
    right_profile: &str,
    num_frames: usize,
    iterations: usize,
    no_left_roll: bool,
    sync_offset: i64,
    skip_start: f64,
    skip_end: f64,
    debug_dir: Option<&str>,
    output: &str,
) -> anyhow::Result<()> {
    reco_io::init();

    // Load lens profiles
    log::info!("loading lens profiles...");
    let left_params = load_camera_params(left_profile)?;
    let right_params = load_camera_params(right_profile)?;
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

    // Apply sync offset
    let (left_indices, right_indices) = if sync_offset >= 0 {
        let offset = sync_offset as u64;
        (
            frame_indices.clone(),
            frame_indices
                .iter()
                .map(|&i| i + offset)
                .collect::<Vec<_>>(),
        )
    } else {
        let offset = (-sync_offset) as u64;
        (
            frame_indices
                .iter()
                .map(|&i| i + offset)
                .collect::<Vec<_>>(),
            frame_indices.clone(),
        )
    };

    log::info!(
        "extracting {} frames (fps: {fps:.1}, sync_offset: {sync_offset})",
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
        enable_sixth_param: !no_left_roll,
        skip_start_secs: skip_start,
        skip_end_secs: skip_end,
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

    // Debug: save per-frame match data as JSON
    if let Some(dir) = debug_dir {
        let matches_json = serde_json::to_string_pretty(&result.per_frame)?;
        std::fs::write(format!("{dir}/matches.json"), &matches_json)?;
        eprintln!("Match data saved to {dir}/matches.json");
    }

    Ok(())
}

/// Save RGBA pixel data as a PNG.
fn save_rgba_png(rgba: &[u8], width: u32, height: u32, path: &str) -> anyhow::Result<()> {
    let img = image::RgbaImage::from_raw(width, height, rgba.to_vec())
        .ok_or_else(|| anyhow::anyhow!("invalid frame dimensions"))?;
    img.save(path)?;
    Ok(())
}
