//! Calibration-related I/O helpers.
//!
//! Provides video probing, frame extraction, and audio extraction for
//! the calibration workflow. These functions wrap the low-level
//! [`VideoDecoder`] with the specific
//! access patterns that calibration needs (random-access frames at
//! computed indices, short audio segments for sync detection).
//!
//! Previously these lived as private functions in the CLI. They are
//! now public so that any consumer can use them without copying the
//! CLI's code.

use std::path::Path;

use reco_core::source::YuvFrame;
use thiserror::Error;

use super::decoder::{DecodeError, VideoDecoder};

/// Errors from calibration I/O operations.
#[derive(Debug, Error)]
pub enum CalibrationIoError {
    /// Video decode error.
    #[error("decode: {0}")]
    Decode(#[from] DecodeError),

    /// Audio extraction failed.
    #[error("audio extraction failed: {0}")]
    AudioExtraction(String),

    /// No audio found in the video file.
    #[error("no audio in {0}")]
    NoAudio(String),
}

/// Video metadata needed for calibration frame selection.
///
/// Probed from a video file without decoding any frames. This is
/// the I/O-side complement to
/// [`reco_calibrate::pipeline::VideoInfo`](https://docs.rs/reco-calibrate).
#[derive(Debug, Clone)]
pub struct VideoProbe {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frames per second.
    pub fps: f64,
    /// Estimated total frame count (from duration * fps).
    pub total_frames: u64,
}

/// Probe a video file for calibration-relevant metadata.
///
/// Opens the file just long enough to read stream info, then closes
/// it. This replaces the common pattern of opening a `VideoDecoder`,
/// reading width/height/fps/duration, and immediately dropping it.
pub fn probe_video(path: &Path) -> Result<VideoProbe, CalibrationIoError> {
    let decoder = VideoDecoder::open(path)?;
    let fps = decoder.fps();
    let total_frames = decoder
        .duration_secs()
        .map(|d| (d * fps) as u64)
        .unwrap_or((fps * 60.0) as u64);

    Ok(VideoProbe {
        width: decoder.width(),
        height: decoder.height(),
        fps,
        total_frames,
    })
}

/// Extract YUV420P frames from a video at specific frame indices.
///
/// Seeks to each frame index (converted to seconds via the video's
/// frame rate) and decodes a single frame. Returns frames in the
/// order of the given indices.
///
/// This is the I/O complement to
/// [`CalibrationPipeline::frame_indices()`](https://docs.rs/reco-calibrate).
pub fn extract_frames(
    video_path: &Path,
    frame_indices: &[u64],
) -> Result<Vec<YuvFrame>, CalibrationIoError> {
    let mut decoder = VideoDecoder::open(video_path)?;
    let fps = decoder.fps();
    let mut frames = Vec::with_capacity(frame_indices.len());

    for &target_idx in frame_indices {
        let target_secs = target_idx as f64 / fps;
        decoder.seek_to_secs(target_secs)?;

        let mut last_frame = None;
        while let Some(yuv) = decoder.next_frame()? {
            let frame_time = yuv.timestamp_us as f64 / 1_000_000.0;
            last_frame = Some(yuv);
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

/// FFmpeg protocol prefixes that must be rejected when passing paths to the
/// CLI. These could trigger network requests or read from arbitrary sources.
const FORBIDDEN_PATH_PREFIXES: &[&str] = &["http://", "https://", "concat:", "pipe:", "data:"];

/// Extract mono PCM audio samples from a video file.
///
/// Uses the `ffmpeg` CLI to extract up to 60 seconds of mono audio
/// at the given sample rate. Returns signed 16-bit PCM samples.
///
/// Caps extraction at 60 seconds since that is more than enough for
/// cross-correlation sync detection, and avoids slow HDD reads on
/// long recordings.
///
/// This is the I/O complement to
/// [`CalibrationPipeline::audio_sync()`](https://docs.rs/reco-calibrate).
pub fn extract_audio_pcm(
    video_path: &Path,
    sample_rate: u32,
) -> Result<Vec<i16>, CalibrationIoError> {
    let path_str = video_path
        .to_str()
        .ok_or_else(|| CalibrationIoError::AudioExtraction("non-UTF8 path".into()))?;

    // Reject paths that would be interpreted as ffmpeg protocols/URLs.
    let lower = path_str.to_ascii_lowercase();
    for prefix in FORBIDDEN_PATH_PREFIXES {
        if lower.starts_with(prefix) {
            return Err(CalibrationIoError::AudioExtraction(format!(
                "path must be a local file, got forbidden prefix '{prefix}'"
            )));
        }
    }

    let output = std::process::Command::new("ffmpeg")
        .args([
            "-i",
            path_str,
            "-t",
            "60",
            "-vn",
            "-ac",
            "1",
            "-ar",
            &sample_rate.to_string(),
            "-f",
            "s16le",
            "-",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .map_err(|e| CalibrationIoError::AudioExtraction(format!("failed to run ffmpeg: {e}")))?;

    if !output.status.success() {
        return Err(CalibrationIoError::AudioExtraction(format!(
            "ffmpeg exited with {}",
            output.status
        )));
    }

    let samples: Vec<i16> = output
        .stdout
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    if samples.is_empty() {
        return Err(CalibrationIoError::NoAudio(
            video_path.display().to_string(),
        ));
    }

    Ok(samples)
}
