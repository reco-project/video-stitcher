//! Standalone sync-offset detection background job.
//!
//! Mirrors the IMU-then-audio priority `reco_calibrate::video::calibrate_videos`
//! uses internally (same as `reco-cli`'s `calibrate` subcommand: try IMU
//! telemetry cross-correlation first, fall back to audio cross-correlation),
//! but stopping there instead of continuing into AKAZE feature matching and
//! the optimizer. Much cheaper when only the sync offset needs
//! (re-)detecting - e.g. after re-clipping footage without touching an
//! already-good rig calibration.

use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use reco_calibrate::CalibrationConfig;
use reco_calibrate::pipeline::{CalibrationPipeline, VideoInfo};
use reco_io::ffmpeg::calibration_io;

/// Result of a standalone sync-offset computation.
pub struct SyncOffsetResult {
    pub frames: i64,
    /// Which method produced the result, for status-line display.
    pub method: &'static str,
}

/// Spawn a background computation of just the temporal sync offset
/// between two videos. The final result comes back over an `mpsc`
/// channel that the render tick polls (non-blocking `try_recv`).
pub fn spawn_compute_sync_offset(
    left: PathBuf,
    right: PathBuf,
) -> Receiver<Result<SyncOffsetResult, String>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = compute_sync_offset(&left, &right);
        match &result {
            Ok(r) => log::info!(
                "Sync-offset detection complete: {} frames ({})",
                r.frames,
                r.method
            ),
            Err(e) => log::error!("Sync-offset detection failed: {e}"),
        }
        tx.send(result).ok();
    });
    rx
}

fn compute_sync_offset(left: &Path, right: &Path) -> Result<SyncOffsetResult, String> {
    let left_probe = calibration_io::probe_video(left).map_err(|e| e.to_string())?;
    let right_probe = calibration_io::probe_video(right).map_err(|e| e.to_string())?;

    let left_info = VideoInfo {
        path: left.to_path_buf(),
        width: left_probe.width,
        height: left_probe.height,
        fps: left_probe.fps,
        total_frames: left_probe.total_frames,
    };
    let right_info = VideoInfo {
        path: right.to_path_buf(),
        width: right_probe.width,
        height: right_probe.height,
        fps: right_probe.fps,
        total_frames: right_probe.total_frames,
    };

    let mut pipeline =
        CalibrationPipeline::new(left_info, right_info, CalibrationConfig::default());
    // Needed for `has_native_gyro`, which gates `imu_sync` - see its own
    // doc comment on why IMU sync is unreliable on quaternion-only cameras.
    pipeline.detect_profiles().map_err(|e| e.to_string())?;

    if let Some(frames) = pipeline.imu_sync().map_err(|e| e.to_string())? {
        return Ok(SyncOffsetResult {
            frames,
            method: "IMU",
        });
    }

    let sample_rate = 44_100u32;
    let left_samples =
        calibration_io::extract_audio_pcm(left, sample_rate).map_err(|e| e.to_string())?;
    let right_samples =
        calibration_io::extract_audio_pcm(right, sample_rate).map_err(|e| e.to_string())?;
    let frames = pipeline
        .audio_sync(&left_samples, &right_samples, sample_rate)
        .map_err(|e| e.to_string())?;
    Ok(SyncOffsetResult {
        frames,
        method: "audio",
    })
}
