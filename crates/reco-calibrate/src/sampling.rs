//! Multi-frame sampling and random subset optimization strategy.
//!
//! Inspired by Gyroflow's calibration approach: instead of relying on a
//! single frame pair, we sample multiple frame pairs spread across the
//! video and run many random-subset optimizations to find the most
//! robust calibration parameters.

use rand::Rng;
use rand::seq::SliceRandom;

use crate::types::{CalibrationConfig, GrayFrame, MatchedPoint};

/// Compute frame indices to sample from a video.
///
/// Skips the first and last 5% of the video (often contains setup/teardown),
/// divides the usable 90% range into `num_samples` equal segments, and
/// picks the midpoint of each segment.
///
/// Returns frame indices sorted in ascending order.
pub fn select_frame_indices(total_frames: u64, num_samples: usize) -> Vec<u64> {
    if total_frames == 0 || num_samples == 0 {
        return Vec::new();
    }

    let start = (total_frames as f64 * 0.05) as u64;
    let end = (total_frames as f64 * 0.95) as u64;
    let usable = end.saturating_sub(start);

    if usable == 0 {
        return vec![total_frames / 2];
    }

    let n = num_samples.min(usable as usize);
    let segment_size = usable as f64 / n as f64;

    (0..n)
        .map(|i| {
            let mid = start as f64 + (i as f64 + 0.5) * segment_size;
            (mid as u64).min(end - 1)
        })
        .collect()
}

/// Downscale a grayscale frame by an integer factor using box filtering.
///
/// Used to reduce large 4K frames to ~1920px width for faster feature
/// detection. Returns the original frame if no downscaling is needed.
pub fn downscale_if_needed(frame: &GrayFrame, target_width: u32) -> GrayFrame {
    if frame.width <= target_width {
        return frame.clone();
    }

    let factor = (frame.width / target_width).max(1);
    let new_w = frame.width / factor;
    let new_h = frame.height / factor;

    let mut data = vec![0u8; (new_w * new_h) as usize];
    let factor_sq = factor * factor;

    for out_y in 0..new_h {
        for out_x in 0..new_w {
            let mut sum: u32 = 0;
            for dy in 0..factor {
                for dx in 0..factor {
                    let src_x = out_x * factor + dx;
                    let src_y = out_y * factor + dy;
                    let idx = (src_y * frame.width + src_x) as usize;
                    sum += frame.data[idx] as u32;
                }
            }
            let out_idx = (out_y * new_w + out_x) as usize;
            data[out_idx] = (sum / factor_sq) as u8;
        }
    }

    GrayFrame {
        data,
        width: new_w,
        height: new_h,
    }
}

/// Sample a random subset of matched points for one optimization iteration.
///
/// Selects `ratio` fraction of the total points uniformly at random.
/// Ensures at least `min_matches` points are selected.
pub fn random_subset(
    points: &[MatchedPoint],
    config: &CalibrationConfig,
    rng: &mut impl Rng,
) -> Vec<MatchedPoint> {
    let target = ((points.len() as f64 * config.subset_ratio) as usize)
        .max(config.min_matches)
        .min(points.len());

    let mut indices: Vec<usize> = (0..points.len()).collect();
    indices.shuffle(rng);
    indices.truncate(target);

    indices.iter().map(|&i| points[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_frame_indices_basic() {
        let indices = select_frame_indices(1000, 5);
        assert_eq!(indices.len(), 5);

        // All indices should be in the 5%-95% range
        for &idx in &indices {
            assert!(idx >= 50, "index {idx} below 5% mark");
            assert!(idx < 950, "index {idx} above 95% mark");
        }

        // Should be sorted ascending
        for w in indices.windows(2) {
            assert!(w[0] < w[1], "indices not sorted: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn select_frame_indices_short_video() {
        // Very short video: should still return something
        let indices = select_frame_indices(10, 5);
        assert!(!indices.is_empty());
        assert!(indices.len() <= 5);
    }

    #[test]
    fn select_frame_indices_zero() {
        assert!(select_frame_indices(0, 5).is_empty());
        assert!(select_frame_indices(100, 0).is_empty());
    }

    #[test]
    fn downscale_identity_when_small() {
        let frame = GrayFrame {
            data: vec![128; 100 * 100],
            width: 100,
            height: 100,
        };
        let result = downscale_if_needed(&frame, 1920);
        assert_eq!(result.width, 100);
        assert_eq!(result.height, 100);
    }

    #[test]
    fn downscale_4k_to_1920() {
        let w = 3840u32;
        let h = 2160u32;
        let frame = GrayFrame {
            data: vec![200; (w * h) as usize],
            width: w,
            height: h,
        };
        let result = downscale_if_needed(&frame, 1920);
        // factor = 3840/1920 = 2
        assert_eq!(result.width, 1920);
        assert_eq!(result.height, 1080);
        // All pixels were 200, so downscaled should still be 200
        assert!(result.data.iter().all(|&p| p == 200));
    }

    #[test]
    fn random_subset_respects_ratio() {
        let points: Vec<MatchedPoint> = (0..100)
            .map(|i| MatchedPoint {
                left: [i as f64 * 0.01, 0.0],
                right: [i as f64 * 0.01, 0.0],
            })
            .collect();

        let config = CalibrationConfig {
            subset_ratio: 0.6,
            min_matches: 8,
            ..Default::default()
        };

        let mut rng = rand::rng();
        let subset = random_subset(&points, &config, &mut rng);

        // Should be ~60 points (60% of 100)
        assert!(
            subset.len() >= 50 && subset.len() <= 70,
            "subset size {} not near 60%",
            subset.len()
        );
    }
}
