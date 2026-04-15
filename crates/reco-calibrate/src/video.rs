//! One-call calibration from video files.
//!
//! Available when the `io` feature is enabled. Wraps the full
//! calibration pipeline with reco-io's FFmpeg-based I/O into a
//! single function call.
//!
//! For live/in-memory calibration (Jetson cameras, mobile streams),
//! use [`calibrate()`](crate::calibrate) directly with pre-extracted
//! frame pairs instead.
//!
//! ```ignore
//! use std::sync::atomic::AtomicBool;
//! use reco_calibrate::video::{calibrate_videos, CalibrateVideosOptions};
//!
//! let interrupted = AtomicBool::new(false);
//! let result = calibrate_videos(
//!     "left.mp4".as_ref(),
//!     "right.mp4".as_ref(),
//!     CalibrateVideosOptions::default(),
//!     &mut |p| eprintln!("{}: {}", p.step, p.detail),
//!     &interrupted,
//! )?;
//! println!("Confidence: {:.1}%", result.confidence * 100.0);
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use reco_core::gpu::{GpuContext, GpuError};
use reco_io::ffmpeg::calibration_io::{self, CalibrationIoError};

use crate::error::CalibrateError;
use crate::pipeline::{CalibrationPipeline, VideoInfo};
use crate::types::{CalibrationConfig, CalibrationProgress, CalibrationResult, CalibrationStep};

/// Options for [`calibrate_videos`].
///
/// All fields are optional with sensible defaults.
#[derive(Debug, Clone, Default)]
pub struct CalibrateVideosOptions {
    /// Calibration algorithm config. Uses [`CalibrationConfig::default()`] if `None`.
    pub config: Option<CalibrationConfig>,
    /// Path to the left lens profile. Auto-detects from video metadata if `None`.
    pub left_profile: Option<PathBuf>,
    /// Path to the right lens profile. Uses left profile if `None`.
    pub right_profile: Option<PathBuf>,
    /// Manual sync offset in frames. Auto-detects via IMU/audio if `None`.
    pub sync_offset: Option<i64>,
}

/// Errors from [`calibrate_videos`].
#[derive(Debug, thiserror::Error)]
pub enum CalibrateVideosError {
    /// Video I/O error (probe, decode, audio extraction).
    #[error("I/O: {0}")]
    Io(#[from] CalibrationIoError),

    /// GPU initialization error.
    #[error("GPU: {0}")]
    Gpu(#[from] GpuError),

    /// Calibration error.
    #[error("calibration: {0}")]
    Calibrate(#[from] CalibrateError),

    /// No frames could be extracted from the videos.
    #[error("no frames extracted from videos")]
    NoFrames,

    /// Operation was cancelled via the interrupted flag.
    #[error("calibration cancelled")]
    Cancelled,
}

/// Check the interrupted flag and return `Cancelled` if set.
fn check_interrupted(interrupted: &AtomicBool) -> Result<(), CalibrateVideosError> {
    if interrupted.load(Ordering::Relaxed) {
        Err(CalibrateVideosError::Cancelled)
    } else {
        Ok(())
    }
}

/// Emit a progress update for the given step.
fn emit_progress(
    on_progress: &mut dyn FnMut(&CalibrationProgress),
    step: CalibrationStep,
    detail: impl Into<String>,
) {
    on_progress(&CalibrationProgress {
        step,
        detail: detail.into(),
    });
}

/// Calibrate two video files with a single function call.
///
/// Handles the full workflow: video probing, lens profile detection,
/// sync estimation (IMU, then audio fallback), frame extraction, GPU
/// initialization, and calibration.
///
/// The `on_progress` callback is invoked before each major step so
/// callers can display status. The `interrupted` flag is checked
/// before each step and returns [`CalibrateVideosError::Cancelled`]
/// if set.
///
/// For advanced use cases (custom sync, debug output), use
/// [`CalibrationPipeline`] directly. For live/in-memory frames
/// (Jetson cameras, mobile streams), use
/// [`calibrate()`](crate::calibrate) directly.
pub fn calibrate_videos(
    left_video: &Path,
    right_video: &Path,
    options: CalibrateVideosOptions,
    on_progress: &mut dyn FnMut(&CalibrationProgress),
    interrupted: &AtomicBool,
) -> Result<CalibrationResult, CalibrateVideosError> {
    reco_io::init();

    let config = options.config.unwrap_or_default();

    // Probe video metadata
    check_interrupted(interrupted)?;
    emit_progress(
        on_progress,
        CalibrationStep::Probing,
        "Probing video metadata",
    );
    let left_probe = calibration_io::probe_video(left_video)?;
    let right_probe = calibration_io::probe_video(right_video)?;
    let fps = left_probe.fps;

    let left_info = VideoInfo {
        path: left_video.into(),
        width: left_probe.width,
        height: left_probe.height,
        fps: left_probe.fps,
        total_frames: left_probe.total_frames,
    };
    let right_info = VideoInfo {
        path: right_video.into(),
        width: right_probe.width,
        height: right_probe.height,
        fps: right_probe.fps,
        total_frames: right_probe.total_frames,
    };

    let mut pipeline = CalibrationPipeline::new(left_info, right_info, config);

    // Lens profiles
    if let Some(ref lp) = options.left_profile {
        pipeline.load_profiles(lp, options.right_profile.as_deref())?;
    } else {
        pipeline.detect_profiles()?;
    }

    // Sync: manual > IMU > audio > default (0)
    check_interrupted(interrupted)?;
    emit_progress(
        on_progress,
        CalibrationStep::AudioSync,
        "Detecting sync offset",
    );
    if let Some(offset) = options.sync_offset {
        pipeline.set_sync_offset(offset);
    } else {
        let imu_ok = pipeline.imu_sync().ok().flatten().is_some();
        if !imu_ok {
            let sample_rate = 44100;
            let left_ok = calibration_io::extract_audio_pcm(left_video, sample_rate);
            let right_ok = calibration_io::extract_audio_pcm(right_video, sample_rate);
            if let (Ok(left_audio), Ok(right_audio)) = (left_ok, right_ok) {
                let _ = pipeline.audio_sync(&left_audio, &right_audio, sample_rate);
            }
            // If audio sync also fails, offset stays at 0
        }
    }

    // Extract frames at computed indices
    check_interrupted(interrupted)?;
    let (left_indices, right_indices) = pipeline.frame_indices();
    emit_progress(
        on_progress,
        CalibrationStep::ExtractingFrames,
        format!("Extracting {} frame pairs", left_indices.len()),
    );
    log::info!(
        "extracting {} frames (fps: {fps:.1}, sync_offset: {})",
        left_indices.len(),
        pipeline.sync_offset(),
    );

    let left_frames = calibration_io::extract_frames(left_video, &left_indices)?;
    let right_frames = calibration_io::extract_frames(right_video, &right_indices)?;

    let pair_count = left_frames.len().min(right_frames.len());
    if pair_count == 0 {
        return Err(CalibrateVideosError::NoFrames);
    }

    let frame_pairs: Vec<_> = left_frames.into_iter().zip(right_frames).collect();

    // GPU init + calibrate
    check_interrupted(interrupted)?;
    emit_progress(
        on_progress,
        CalibrationStep::FeatureMatching,
        "GPU init and feature matching",
    );
    let gpu = pollster::block_on(GpuContext::new())?;
    log::info!("GPU: {}", gpu.gpu_name());

    let result = pipeline.calibrate(&gpu, &frame_pairs)?;
    Ok(result)
}
