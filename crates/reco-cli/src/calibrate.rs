//! `reco calibrate` subcommand.
//!
//! Thin wrapper around [`reco_calibrate::pipeline::CalibrationPipeline`].
//! Handles file I/O (video decode, audio extraction, result publication) while
//! all calibration logic lives in the library.

use std::path::Path;

use reco_calibrate::CalibrationConfig;
use reco_calibrate::pipeline::{CalibrationPipeline, VideoInfo};
use reco_calibrate::types::Yuv420pFrame;
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
    output: &str,
    accept_unconfirmed: bool,
) -> anyhow::Result<()> {
    let config = CalibrationConfig {
        num_frames,
        skip_start_secs: skip_start,
        skip_end_secs: skip_end,
    };
    config.validate()?;
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

    let mut pipeline = CalibrationPipeline::new(left_info, right_info, config);

    let load = |path: Option<&str>| {
        path.map(|path| reco_calibrate::lens_database::load_from_file(Path::new(path)))
            .transpose()
    };
    pipeline.set_profiles(load(left_profile)?, load(right_profile)?);

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
    anyhow::ensure!(
        !left_indices.is_empty(),
        "no overlapping calibration frames could be selected from the videos"
    );
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

    let frame_pairs: Vec<(Yuv420pFrame, Yuv420pFrame)> =
        left_frames.into_iter().zip(right_frames).collect();

    // Step 5: Calibrate
    eprintln!("Calibrating with {} frame pairs...", frame_pairs.len());
    let result = pipeline
        .calibrate(&std::sync::atomic::AtomicBool::new(false), &frame_pairs)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    publish_result(&result, Path::new(output), accept_unconfirmed)?;
    print_results(&result, output, frame_pairs.len());
    Ok(())
}

fn publish_result(
    result: &reco_calibrate::CalibrationResult,
    output: &Path,
    accept_unconfirmed: bool,
) -> anyhow::Result<()> {
    if let reco_calibrate::CalibrationAssessment::NeedsReview { reason } = &result.assessment
        && !accept_unconfirmed
    {
        // A report retains the candidate without publishing it as an active calibration.
        let mut report = output.as_os_str().to_os_string();
        report.push(".review.json");
        let report = std::path::PathBuf::from(report);
        std::fs::write(&report, serde_json::to_string_pretty(&result)?)?;
        anyhow::bail!(
            "{reason} Candidate and measurements: {}. Requested output was not modified.",
            report.display()
        );
    }

    // Write output
    let json = serde_json::to_string_pretty(&result.calibration)?;
    std::fs::write(output, &json)?;

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

fn print_results(result: &reco_calibrate::CalibrationResult, output: &str, total_pairs: usize) {
    eprintln!("\nCalibration results:");
    eprintln!("  Output:          {output}");
    eprintln!("  Frames used:     {}/{total_pairs}", result.frames_used);
    eprintln!("  Total matches:   {}", result.total_matches);
    eprintln!("  Assessment:      {:?}", result.assessment);
    eprintln!("\nPlacement parameters:");
    let l = result
        .calibration
        .projection
        .l_shape()
        .expect("stereo calibration always produces the L-shape projection");
    eprintln!(
        "  cameraAxisOffset: {:.4}",
        result.calibration.framing.axis_offset
    );
    eprintln!("  intersect:        {:.4}", l.intersect);
    eprintln!("  xTy:              {:.6}", l.x_ty);
    eprintln!("  xRz:              {:.6}", l.x_rz);
    eprintln!("  zRx:              {:.6}", l.z_rx);
    eprintln!("  zRz:              {:.6}", l.z_rz);

    for (i, check) in result.overlap_checks.iter().enumerate() {
        eprintln!(
            "  Independent pair {}: {} matches, median {:?}px, p95 {:?}px",
            i + 1,
            check.matches,
            check.median_pixels,
            check.p95_pixels
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_calibrate::{CalibrationAssessment, CalibrationResult};
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::projection::LShape;

    #[test]
    fn review_report_preserves_active_file_until_explicit_acceptance() {
        let directory = std::env::temp_dir().join(format!(
            "reco-calibration-publication-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let output = directory.join("calibration.json");
        std::fs::write(&output, "original document").unwrap();
        let mut result = CalibrationResult {
            calibration: Calibration::new(
                vec![Lens::fisheye(64, 64, 40.0, 40.0, 32.0, 32.0, [0.0; 4]); 2],
                LShape {
                    intersect: 0.6,
                    x_ty: 0.0,
                    x_rx: 0.0,
                    x_rz: 0.0,
                    z_rx: 0.0,
                    z_rz: 0.0,
                    blend_width: 0.05,
                },
                Framing {
                    axis_offset: 0.23,
                    tilt: 0.0,
                    roll: 0.0,
                },
            ),
            total_matches: 24,
            frames_used: 1,
            assessment: CalibrationAssessment::NeedsReview {
                reason: "insufficient evidence".into(),
            },
            overlap_checks: vec![],
            left_lens_profile: None,
            right_lens_profile: None,
        };
        let error = publish_result(&result, &output, false).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Requested output was not modified")
        );
        assert_eq!(
            std::fs::read_to_string(&output).unwrap(),
            "original document"
        );
        let report = directory.join("calibration.json.review.json");
        let candidate: CalibrationResult =
            serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
        assert_eq!(candidate.calibration, result.calibration);
        assert_eq!(candidate.assessment, result.assessment);
        publish_result(&result, &output, true).unwrap();
        assert_eq!(Calibration::from_file(&output).unwrap(), result.calibration);
        result.assessment = CalibrationAssessment::Confirmed;
        result.calibration.framing.roll = 0.1;
        publish_result(&result, &output, false).unwrap();
        assert_eq!(Calibration::from_file(&output).unwrap(), result.calibration);
        std::fs::remove_file(output).unwrap();
        std::fs::remove_file(report).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }
}
