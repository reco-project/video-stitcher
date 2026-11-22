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
//! println!("Assessment: {:?}", result.assessment);
//! ```

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use reco_core::calibration::Lens;
use reco_io::ffmpeg::calibration_io::{self, CalibrationIoError};

use crate::error::CalibrateError;
use crate::pipeline::{CalibrationPipeline, VideoInfo};
use crate::types::{CalibrationConfig, CalibrationProgress, CalibrationResult, CalibrationStep};

/// Options for [`calibrate_videos`]. Missing optics are estimated from footage.
/// Explicit lenses are fixed independently; no profile lookup or optical fallback runs.
#[derive(Debug, Clone, Default)]
pub struct CalibrateVideosOptions {
    /// Sampling config. Uses [`CalibrationConfig::default()`] if `None`.
    pub config: Option<CalibrationConfig>,
    /// Fixed left camera optics; `None` estimates them from image overlap.
    pub left_params: Option<Lens>,
    /// Fixed right camera optics; `None` estimates them from image overlap.
    pub right_params: Option<Lens>,
    /// Manual sync offset in frames. Auto-detects via IMU/audio if `None`.
    pub sync_offset: Option<i64>,
}

/// Errors from [`calibrate_videos`]. `Clone + Send + Sync` so a
/// calibration worker can post the typed result back to the UI
/// thread through an mpsc channel without stringifying.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CalibrateVideosError {
    /// Video I/O error (probe, decode, audio extraction).
    #[error("I/O: {0}")]
    Io(#[from] CalibrationIoError),

    /// Calibration error.
    #[error("calibration: {0}")]
    Calibrate(#[from] CalibrateError),

    /// No calibration frame pairs could be selected after synchronizing the clips.
    #[error("no overlapping calibration frames could be selected from the videos")]
    NoFrames,

    /// Neither IMU nor audio established an offset. Supply a matched pair to resume.
    #[error("Timing needs a check. Match an event in both cameras, then continue.")]
    TimingRequired,

    /// Operation was cancelled via the interrupted flag.
    #[error("calibration cancelled")]
    Cancelled,
}

// Compile-time assertion: `CalibrateVideosError`
// and its transitive inner types are `Clone + Send + Sync`. Regresses
// if a future variant introduces a non-Clone `#[from]` wrap.
const _: fn() = || {
    fn assert_clone_send_sync<T: Clone + Send + Sync + 'static>() {}
    assert_clone_send_sync::<CalibrateVideosError>();
    assert_clone_send_sync::<crate::error::CalibrateError>();
};

/// Check the interrupted flag and return `Cancelled` if set.
fn check_interrupted(interrupted: &AtomicBool) -> Result<(), CalibrateVideosError> {
    if interrupted.load(Ordering::Relaxed) {
        Err(CalibrateVideosError::Cancelled)
    } else {
        Ok(())
    }
}

/// Measure timing from audio when native gyro timing is unavailable.
fn sync_from_audio(
    pipeline: &mut CalibrationPipeline,
    left_video: &Path,
    right_video: &Path,
) -> Result<(), CalibrateVideosError> {
    let sample_rate = 44100;
    let audio = |path: &Path| {
        calibration_io::extract_audio_pcm(path, sample_rate).map_err(|error| {
            log::warn!("Could not read audio from {}: {error}", path.display());
            CalibrateVideosError::TimingRequired
        })
    };
    let left = audio(left_video)?;
    let right = audio(right_video)?;
    pipeline
        .audio_sync(&left, &right, sample_rate)
        .map_err(|error| {
            log::warn!("Audio sync failed: {error}");
            CalibrateVideosError::TimingRequired
        })?;
    Ok(())
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
/// Handles the full workflow: video probing, optional supplied optics,
/// sync estimation (IMU, then audio fallback), frame extraction and native alignment.
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
    config.validate()?;

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

    let mut pipeline = CalibrationPipeline::new(left_info, right_info, config)?;

    pipeline.set_profiles(options.left_params, options.right_params);

    check_interrupted(interrupted)?;
    emit_progress(
        on_progress,
        CalibrationStep::AudioSync,
        "Synchronizing recordings",
    );
    if let Some(offset) = options.sync_offset {
        pipeline.set_sync_offset(offset);
    } else {
        match pipeline.imu_sync() {
            Ok(Some(offset)) => log::info!("IMU sync: {offset} frames"),
            imu => {
                if let Err(error) = imu {
                    log::warn!("IMU sync failed: {error}");
                }
                check_interrupted(interrupted)?;
                let result = sync_from_audio(&mut pipeline, left_video, right_video);
                check_interrupted(interrupted)?;
                result?;
            }
        }
    }

    // Extract frames at computed indices
    check_interrupted(interrupted)?;
    let (left_indices, right_indices) = pipeline.frame_indices();
    if left_indices.is_empty() {
        return Err(CalibrateVideosError::NoFrames);
    }
    emit_progress(
        on_progress,
        CalibrationStep::ExtractingFrames,
        format!("Extracting {} frame pairs", left_indices.len()),
    );
    log::info!(
        "extracting {} frames (fps: {fps:.1}, sync_offset: {})",
        left_indices.len(),
        pipeline
            .sync_offset()
            .ok_or(CalibrateVideosError::TimingRequired)?,
    );

    let left_frames = calibration_io::extract_frames(left_video, &left_indices)?;
    let right_frames = calibration_io::extract_frames(right_video, &right_indices)?;

    let frame_pairs: Vec<_> = left_frames.into_iter().zip(right_frames).collect();

    check_interrupted(interrupted)?;
    emit_progress(
        on_progress,
        CalibrationStep::Optimizing,
        "Fitting and verifying image alignment",
    );
    let result = pipeline.calibrate(interrupted, &frame_pairs)?;
    Ok(result)
}
