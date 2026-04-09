//! Audio-based temporal synchronization between two video files.
//!
//! Extracts audio tracks from both videos, then uses FFT cross-correlation
//! to find the time offset that best aligns them. Works for any camera
//! type as long as both recordings capture ambient sound.
//!
//! ## Approach
//!
//! 1. Extract mono 16kHz audio from both videos via `ffmpeg` CLI
//! 2. Take a 30-second chunk from the middle of the shorter recording
//! 3. Cross-correlate against the other recording using FFT convolution
//! 4. The peak correlation gives the offset in samples (sub-frame precision)
//!
//! ## Search window
//!
//! Searches up to +-5 minutes by default. Cameras in sports setups can
//! start recording minutes apart (confirmed: XTU had 26s offset).

use std::path::Path;
use std::process::Command;

use realfft::RealFftPlanner;

/// Result of audio synchronization.
#[derive(Debug, Clone, Copy)]
pub struct AudioSyncResult {
    /// Offset in seconds. Positive means right video started later
    /// (right needs to advance more frames to sync).
    pub offset_secs: f64,
    /// Offset in frames at the given fps.
    pub offset_frames: f64,
    /// Peak cross-correlation value (higher = more confident).
    pub confidence: f64,
}

/// Audio sync configuration.
#[derive(Debug, Clone)]
pub struct AudioSyncConfig {
    /// Sample rate for audio extraction (Hz). 16000 is sufficient for sync.
    pub sample_rate: u32,
    /// Duration of the correlation chunk (seconds). Longer = more robust.
    pub chunk_secs: f64,
    /// Maximum search window (seconds). Must be large enough for the
    /// actual offset between recordings.
    pub max_offset_secs: f64,
}

impl Default for AudioSyncConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            chunk_secs: 30.0,
            max_offset_secs: 300.0, // 5 minutes
        }
    }
}

/// Extract mono PCM audio from a video file using ffmpeg CLI.
///
/// Returns raw i16 samples at the configured sample rate.
fn extract_audio(video_path: &Path, sample_rate: u32) -> Result<Vec<i16>, AudioSyncError> {
    let output = Command::new("ffmpeg")
        .args([
            "-i",
            video_path.to_str().ok_or(AudioSyncError::InvalidPath)?,
            "-vn", // no video
            "-ac",
            "1", // mono
            "-ar",
            &sample_rate.to_string(),
            "-f",
            "s16le", // raw PCM signed 16-bit little-endian
            "-",     // output to stdout
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .map_err(|e| AudioSyncError::FfmpegFailed(e.to_string()))?;

    if !output.status.success() {
        return Err(AudioSyncError::FfmpegFailed(format!(
            "ffmpeg exited with {}",
            output.status
        )));
    }

    // Convert bytes to i16 samples
    let samples: Vec<i16> = output
        .stdout
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    if samples.is_empty() {
        return Err(AudioSyncError::NoAudio);
    }

    Ok(samples)
}

/// Compute FFT-based cross-correlation between two signals.
///
/// Uses the convolution theorem: corr(a, b) = ifft(fft(a) * conj(fft(b)))
/// where b is zero-padded and a is reversed for correlation.
fn fft_cross_correlate(signal: &[f64], template: &[f64]) -> Vec<f64> {
    let n = signal.len() + template.len() - 1;
    // Round up to next power of 2 for FFT efficiency
    let fft_len = n.next_power_of_two();

    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    // Prepare signal (zero-padded)
    let mut sig_buf = vec![0.0f64; fft_len];
    sig_buf[..signal.len()].copy_from_slice(signal);
    let mut sig_spec = fft.make_output_vec();
    fft.process(&mut sig_buf, &mut sig_spec).unwrap();

    // Prepare template (reversed for correlation, zero-padded)
    let mut tpl_buf = vec![0.0f64; fft_len];
    for (i, &v) in template.iter().enumerate() {
        tpl_buf[i] = v;
    }
    let mut tpl_spec = fft.make_output_vec();
    fft.process(&mut tpl_buf, &mut tpl_spec).unwrap();

    // Multiply sig_spec * conj(tpl_spec)
    for (s, t) in sig_spec.iter_mut().zip(tpl_spec.iter()) {
        let re = s.re * t.re + s.im * t.im;
        let im = s.im * t.re - s.re * t.im;
        s.re = re;
        s.im = im;
    }

    // Inverse FFT
    let mut result = ifft.make_output_vec();
    ifft.process(&mut sig_spec, &mut result).unwrap();

    // Normalize by FFT length
    let scale = 1.0 / fft_len as f64;
    result.iter_mut().for_each(|v| *v *= scale);

    result.truncate(n);
    result
}

/// Estimate the temporal offset between two video files using audio
/// cross-correlation.
///
/// Returns the offset in seconds. Positive means the right video
/// started later (right needs more frames skipped to sync).
///
/// # Arguments
///
/// * `left_video` - Path to the left camera video file
/// * `right_video` - Path to the right camera video file
/// * `fps` - Video frame rate (for converting offset to frames)
/// * `config` - Sync configuration (search window, sample rate, etc.)
pub fn estimate_sync_offset(
    left_video: &Path,
    right_video: &Path,
    fps: f64,
    config: &AudioSyncConfig,
) -> Result<AudioSyncResult, AudioSyncError> {
    let sr = config.sample_rate;

    log::info!("extracting audio from left video...");
    let left_audio = extract_audio(left_video, sr)?;
    log::info!("extracting audio from right video...");
    let right_audio = extract_audio(right_video, sr)?;

    let left_secs = left_audio.len() as f64 / sr as f64;
    let right_secs = right_audio.len() as f64 / sr as f64;
    log::info!(
        "audio: left={left_secs:.1}s ({} samples), right={right_secs:.1}s ({} samples)",
        left_audio.len(),
        right_audio.len()
    );

    let chunk_samples = (config.chunk_secs * sr as f64) as usize;
    let max_lag_samples = (config.max_offset_secs * sr as f64) as usize;

    // Take a chunk from the middle of the left recording
    let mid_l = left_audio.len() / 2;
    let chunk_start = mid_l.saturating_sub(chunk_samples / 2);
    let chunk_end = (chunk_start + chunk_samples).min(left_audio.len());
    let template: Vec<f64> = left_audio[chunk_start..chunk_end]
        .iter()
        .map(|&s| s as f64)
        .collect();

    // Search window in right recording
    let search_start = mid_l.saturating_sub(chunk_samples / 2 + max_lag_samples);
    let search_end = (mid_l + chunk_samples / 2 + max_lag_samples).min(right_audio.len());
    let signal: Vec<f64> = right_audio[search_start..search_end]
        .iter()
        .map(|&s| s as f64)
        .collect();

    // Normalize
    let normalize = |v: &mut Vec<f64>| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        if std > 1e-10 {
            v.iter_mut().for_each(|x| *x = (*x - mean) / std);
        }
    };

    let mut template_norm = template;
    let mut signal_norm = signal;
    normalize(&mut template_norm);
    normalize(&mut signal_norm);

    log::info!(
        "cross-correlating {:.1}s template against {:.1}s window...",
        template_norm.len() as f64 / sr as f64,
        signal_norm.len() as f64 / sr as f64,
    );

    let corr = fft_cross_correlate(&signal_norm, &template_norm);

    // Find peak
    let (peak_idx, peak_val) = corr
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    // Convert peak index to time offset
    // The correlation peak at index i means: template aligns with signal at position i
    // signal starts at search_start in right audio, template starts at chunk_start in left audio
    let right_match_pos = search_start + peak_idx;
    let offset_samples = right_match_pos as i64 - chunk_start as i64;
    let offset_secs = offset_samples as f64 / sr as f64;
    let offset_frames = offset_secs * fps;

    log::info!(
        "audio sync: offset = {offset_secs:.4}s = {offset_frames:.1} frames (confidence={peak_val:.2})"
    );

    Ok(AudioSyncResult {
        offset_secs,
        offset_frames,
        confidence: *peak_val,
    })
}

/// Errors from audio synchronization.
#[derive(Debug, thiserror::Error)]
pub enum AudioSyncError {
    /// FFmpeg failed to extract audio.
    #[error("ffmpeg audio extraction failed: {0}")]
    FfmpegFailed(String),
    /// Video file has no audio track.
    #[error("no audio track found in video")]
    NoAudio,
    /// Invalid file path.
    #[error("invalid file path")]
    InvalidPath,
}
