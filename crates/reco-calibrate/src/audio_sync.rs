//! Audio-based temporal synchronization.
//!
//! Uses FFT cross-correlation to find the time offset between two
//! audio signals recorded simultaneously by different cameras. Works
//! for any camera type as long as both capture ambient sound.
//!
//! ## API
//!
//! The core function [`correlate`] takes raw PCM samples - it does no
//! file I/O. The app (CLI, GUI, etc.) is responsible for extracting
//! audio from video files using whatever decoder it has.
//!
//! ```ignore
//! // App extracts audio (via reco-io, gstreamer, system API, etc.)
//! let left_samples: Vec<i16> = my_decoder.extract_audio(left_video)?;
//! let right_samples: Vec<i16> = my_decoder.extract_audio(right_video)?;
//!
//! // Crate does the math
//! let result = audio_sync::correlate(&left_samples, &right_samples, 44100, 30.0)?;
//! let sync_frames = result.offset_frames(fps);
//! ```

use realfft::RealFftPlanner;

/// Result of audio cross-correlation.
#[derive(Debug, Clone, Copy)]
pub struct SyncResult {
    /// Offset in seconds. Positive means right recording started later
    /// (right needs to advance more to sync with left).
    pub offset_secs: f64,
    /// Peak cross-correlation value (higher = more confident).
    pub confidence: f64,
}

impl SyncResult {
    /// Convert the offset to frames at the given fps.
    pub fn offset_frames(&self, fps: f64) -> f64 {
        self.offset_secs * fps
    }

    /// Round the offset to the nearest frame.
    pub fn offset_frames_rounded(&self, fps: f64) -> i64 {
        (self.offset_secs * fps).round() as i64
    }
}

/// Find the temporal offset between two audio recordings using
/// FFT cross-correlation.
///
/// Takes raw mono PCM samples from both cameras at the same sample
/// rate. Returns the offset in seconds with sub-frame precision.
///
/// Uses a chunk from the middle of the left recording as a template
/// and correlates it against the full right recording. The chunk
/// approach handles recordings of different lengths and avoids
/// startup noise.
///
/// # Arguments
///
/// * `left_samples` - Mono PCM samples from the left camera
/// * `right_samples` - Mono PCM samples from the right camera
/// * `sample_rate` - Sample rate in Hz (must be the same for both)
/// * `chunk_secs` - Duration of correlation chunk (30.0 recommended)
pub fn correlate(
    left_samples: &[i16],
    right_samples: &[i16],
    sample_rate: u32,
    chunk_secs: f64,
) -> Result<SyncResult, SyncError> {
    if left_samples.is_empty() || right_samples.is_empty() {
        return Err(SyncError::EmptyAudio);
    }

    let sr = sample_rate as f64;

    // Use shorter clip's length to bound the chunk
    let shorter_len = left_samples.len().min(right_samples.len());
    let chunk_samples = ((chunk_secs * sr) as usize).min(shorter_len);

    // Take a chunk from the middle of the left recording
    let mid = left_samples.len() / 2;
    let tpl_start = mid.saturating_sub(chunk_samples / 2);
    let tpl_end = (tpl_start + chunk_samples).min(left_samples.len());

    let mut template: Vec<f64> = left_samples[tpl_start..tpl_end]
        .iter()
        .map(|&s| s as f64)
        .collect();
    let mut signal: Vec<f64> = right_samples.iter().map(|&s| s as f64).collect();

    // Normalize both
    normalize(&mut template);
    normalize(&mut signal);

    log::info!(
        "audio sync: correlating {:.1}s template against {:.1}s signal at {}Hz",
        template.len() as f64 / sr,
        signal.len() as f64 / sr,
        sample_rate,
    );

    let corr = fft_cross_correlate(&signal, &template)?;

    // Find peak
    let (peak_idx, peak_val) = corr
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    // Offset calculation:
    // Template is reversed in fft_cross_correlate (convolution with reversed template = correlation).
    // Peak at index P means right[P - (template_len-1)] aligns with left[tpl_start].
    let match_in_right = peak_idx as i64 - (template.len() as i64 - 1);
    let offset_samples = match_in_right - tpl_start as i64;
    let offset_secs = offset_samples as f64 / sr;

    log::info!(
        "audio sync: offset = {offset_secs:.4}s (confidence={:.0})",
        peak_val,
    );

    Ok(SyncResult {
        offset_secs,
        confidence: *peak_val,
    })
}

/// Errors from audio synchronization.
#[derive(Debug, thiserror::Error)]
pub enum SyncError {
    /// One or both audio signals are empty.
    #[error("empty audio signal")]
    EmptyAudio,
    /// FFT computation failed.
    #[error("FFT error: {0}")]
    FftError(String),
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn normalize(v: &mut [f64]) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    if std > 1e-10 {
        v.iter_mut().for_each(|x| *x = (*x - mean) / std);
    }
}

/// FFT-based cross-correlation (convolution with reversed template).
fn fft_cross_correlate(signal: &[f64], template: &[f64]) -> Result<Vec<f64>, SyncError> {
    // Checked addition to prevent overflow on 32-bit targets
    let n = signal
        .len()
        .checked_add(template.len())
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| SyncError::FftError("signal + template length overflow".to_string()))?;
    let fft_len = n.next_power_of_two();

    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    // Signal (zero-padded)
    let mut sig_buf = vec![0.0f64; fft_len];
    sig_buf[..signal.len()].copy_from_slice(signal);
    let mut sig_spec = fft.make_output_vec();
    fft.process(&mut sig_buf, &mut sig_spec)
        .map_err(|e| SyncError::FftError(e.to_string()))?;

    // Template REVERSED (time-reversal converts convolution to correlation)
    let mut tpl_buf = vec![0.0f64; fft_len];
    for (i, &v) in template.iter().rev().enumerate() {
        tpl_buf[i] = v;
    }
    let mut tpl_spec = fft.make_output_vec();
    fft.process(&mut tpl_buf, &mut tpl_spec)
        .map_err(|e| SyncError::FftError(e.to_string()))?;

    // Multiply spectra (convolution in frequency domain)
    for (s, t) in sig_spec.iter_mut().zip(tpl_spec.iter()) {
        let re = s.re * t.re - s.im * t.im;
        let im = s.re * t.im + s.im * t.re;
        s.re = re;
        s.im = im;
    }

    // Inverse FFT
    let mut result = ifft.make_output_vec();
    ifft.process(&mut sig_spec, &mut result)
        .map_err(|e| SyncError::FftError(e.to_string()))?;

    let scale = 1.0 / fft_len as f64;
    result.iter_mut().for_each(|v| *v *= scale);
    result.truncate(n);
    Ok(result)
}
