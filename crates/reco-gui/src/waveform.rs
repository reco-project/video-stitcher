//! Audio waveform envelope extraction for the audio-sync check panel.
//!
//! Downsamples a short PCM window around the playhead into a coarse
//! peak-amplitude envelope for overlaying left/right camera audio as a
//! visual sync check — a real `sync_offset` error shows up as a shifted
//! transient spike between the two traces. Extraction shells out to
//! `ffmpeg` per call, so callers must not invoke this on the UI thread.

use std::path::Path;

use reco_io::ffmpeg::calibration_io::{self, CalibrationIoError};

/// Sample rate used for envelope extraction. Low enough to keep the
/// `ffmpeg` call fast; far more than enough resolution for a peak
/// envelope over a multi-second window.
const ENVELOPE_SAMPLE_RATE: u32 = 8_000;

/// Downsample PCM samples into `buckets` peak-amplitude values,
/// normalized to `[0.0, 1.0]`. Always returns exactly `buckets` values
/// (zero-filled if `samples` is empty).
pub fn compute_envelope(samples: &[i16], buckets: usize) -> Vec<f32> {
    if buckets == 0 {
        return Vec::new();
    }
    if samples.is_empty() {
        return vec![0.0; buckets];
    }
    let len = samples.len();
    (0..buckets)
        .map(|i| {
            let start = i * len / buckets;
            let end = ((i + 1) * len / buckets).max(start + 1).min(len);
            let peak = samples[start..end]
                .iter()
                .map(|&s| (s as i32).unsigned_abs())
                .max()
                .unwrap_or(0);
            peak as f32 / i16::MAX as f32
        })
        .collect()
}

/// Rescale `left` and `right` in place using one shared peak (the louder
/// of the two), instead of each independently reaching `1.0`.
///
/// Normalizing each channel independently would hide a real amplitude
/// mismatch between the two camera mics - e.g. one camera's audio
/// genuinely quieter, which is itself a sign something's off - by
/// stretching both to look equally loud. A shared peak keeps that
/// difference visible while still auto-scaling the pair away from
/// barely-visible slivers when both are quiet.
pub fn normalize_pair_to_peak(left: &mut [f32], right: &mut [f32]) {
    let peak = left
        .iter()
        .chain(right.iter())
        .copied()
        .fold(0.0f32, f32::max);
    if peak > 1e-6 {
        for v in left.iter_mut().chain(right.iter_mut()) {
            *v /= peak;
        }
    }
}

/// Bucket radius for [`smooth_envelope`]'s moving average.
const SMOOTHING_RADIUS: usize = 3;

/// Smooth `envelope` with a centered moving average of `radius` buckets on
/// each side.
///
/// The raw peak-per-bucket envelope is noisy bucket-to-bucket (each bucket
/// is an independent max, not a continuous signal), which reads as a messy
/// spike train rather than a shape whose peaks are easy to compare by eye.
/// Averaging over a small neighborhood turns it into a smooth curve while
/// still preserving genuine transients wide enough to matter for a
/// frame-scale sync check.
fn smooth_envelope(envelope: &[f32], radius: usize) -> Vec<f32> {
    if radius == 0 || envelope.is_empty() {
        return envelope.to_vec();
    }
    let len = envelope.len();
    (0..len)
        .map(|i| {
            let start = i.saturating_sub(radius);
            let end = (i + radius + 1).min(len);
            let sum: f32 = envelope[start..end].iter().sum();
            sum / (end - start) as f32
        })
        .collect()
}

/// Extract a peak-amplitude envelope for a window of audio centered on
/// `center_secs` in the file at `path`, smoothed (see [`smooth_envelope`]).
///
/// Not normalized - the two channels of a L/R pair should be scaled
/// together afterwards via [`normalize_pair_to_peak`], not independently,
/// so a real amplitude mismatch between the two mics stays visible.
pub fn extract_window_envelope(
    path: &Path,
    center_secs: f64,
    window_secs: f64,
    buckets: usize,
) -> Result<Vec<f32>, CalibrationIoError> {
    let start_secs = (center_secs - window_secs / 2.0).max(0.0);
    let samples = calibration_io::extract_audio_pcm_window(
        path,
        ENVELOPE_SAMPLE_RATE,
        start_secs,
        window_secs,
    )?;
    let envelope = compute_envelope(&samples, buckets);
    Ok(smooth_envelope(&envelope, SMOOTHING_RADIUS))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_envelope_empty_samples_returns_zero_filled() {
        assert_eq!(compute_envelope(&[], 5), vec![0.0; 5]);
    }

    #[test]
    fn compute_envelope_zero_buckets_returns_empty() {
        assert!(compute_envelope(&[1, 2, 3], 0).is_empty());
    }

    #[test]
    fn compute_envelope_returns_exact_bucket_count() {
        let samples: Vec<i16> = (0..997).map(|i| (i % 100) as i16).collect();
        assert_eq!(compute_envelope(&samples, 300).len(), 300);
    }

    #[test]
    fn compute_envelope_finds_peak_per_bucket() {
        // Two buckets: first half has a loud spike, second half is quiet.
        let mut samples = vec![0i16; 200];
        samples[50] = i16::MAX;
        samples[150] = 100;
        let env = compute_envelope(&samples, 2);
        assert!((env[0] - 1.0).abs() < 1e-6, "loud bucket should read ~1.0");
        assert!(env[1] < 0.01, "quiet bucket should read ~0.0");
    }

    #[test]
    fn compute_envelope_handles_min_i16_without_overflow() {
        let samples = vec![i16::MIN; 10];
        let env = compute_envelope(&samples, 1);
        assert!((env[0] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn normalize_pair_to_peak_scales_quiet_pair_up() {
        let mut left = vec![0.0, 0.05, 0.02, 0.1];
        let mut right = vec![0.0; 4];
        normalize_pair_to_peak(&mut left, &mut right);
        assert!(
            (left[3] - 1.0).abs() < 1e-6,
            "the pair's loudest bucket should hit 1.0"
        );
        assert!(
            (left[1] - 0.5).abs() < 1e-6,
            "other buckets scale by the same factor"
        );
    }

    #[test]
    fn normalize_pair_to_peak_leaves_silence_untouched() {
        let mut left = vec![0.0; 4];
        let mut right = vec![0.0; 4];
        normalize_pair_to_peak(&mut left, &mut right);
        assert_eq!(left, vec![0.0; 4], "no division by zero on pure silence");
        assert_eq!(right, vec![0.0; 4]);
    }

    #[test]
    fn normalize_pair_to_peak_preserves_relative_loudness_between_channels() {
        // Right is genuinely quieter than left - a real mic-level
        // mismatch, not just a display artifact - so it must stay
        // quieter after normalization instead of being independently
        // stretched to also reach 1.0.
        let mut left = vec![0.0, 1.0, 0.5];
        let mut right = vec![0.0, 0.4, 0.2];
        normalize_pair_to_peak(&mut left, &mut right);
        assert!((left[1] - 1.0).abs() < 1e-6, "louder channel hits 1.0");
        assert!(
            (right[1] - 0.4).abs() < 1e-6,
            "quieter channel stays proportionally quieter, not re-stretched to 1.0"
        );
    }

    #[test]
    fn smooth_envelope_radius_zero_is_identity() {
        let env = vec![0.0, 1.0, 0.0, 1.0];
        assert_eq!(smooth_envelope(&env, 0), env);
    }

    #[test]
    fn smooth_envelope_empty_stays_empty() {
        assert!(smooth_envelope(&[], 3).is_empty());
    }

    #[test]
    fn smooth_envelope_flattens_a_single_spike() {
        let mut env = vec![0.0; 11];
        env[5] = 1.0;
        let smoothed = smooth_envelope(&env, 2);
        // The spike spreads into its 2-bucket neighborhood (5 buckets averaged into 1.0/5)...
        assert!((smoothed[5] - 1.0 / 5.0).abs() < 1e-6);
        // ...and the peak is no longer an isolated single-bucket outlier.
        assert!(smoothed[5] > smoothed[0]);
        assert!(smoothed[4] > 0.0 && smoothed[6] > 0.0);
    }

    #[test]
    fn smooth_envelope_preserves_length_and_shrinks_window_at_edges() {
        let env = vec![1.0, 1.0, 1.0, 1.0];
        let smoothed = smooth_envelope(&env, 2);
        assert_eq!(smoothed.len(), env.len());
        // Uniform input stays uniform regardless of the clamped edge window.
        for v in smoothed {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }
}
