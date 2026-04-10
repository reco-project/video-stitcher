//! Spatial overlap filtering and RANSAC outlier rejection.
//!
//! Two-stage filtering pipeline:
//! 1. **Spatial filter**: Keeps matches in the expected overlap region
//!    (right side of left image, left side of right image, vertical center)
//! 2. **RANSAC**: Estimates fundamental matrix and rejects geometric outliers
//!    using the `inlier` crate's MAGSAC implementation

use crate::features::{KeyPoint, RawMatch};
use crate::types::CalibrationConfig;

/// Apply spatial overlap filter to raw matches.
///
/// Keeps matches where:
/// - Left keypoint x >= `spatial_x_threshold * width` (right portion of left image)
/// - Right keypoint x <= `spatial_x_threshold * width` (left portion of right image)
/// - Both keypoints y in `[spatial_y_low * height, spatial_y_high * height]`
///
/// Returns the filtered results as-is, even if fewer than `min_matches`.
/// The caller is responsible for deciding how to handle insufficient matches.
pub fn spatial_filter(
    matches: &[RawMatch],
    kp_left: &[KeyPoint],
    kp_right: &[KeyPoint],
    img_w_left: u32,
    img_h_left: u32,
    img_w_right: u32,
    img_h_right: u32,
    config: &CalibrationConfig,
) -> Vec<RawMatch> {
    let x_thresh_left = config.matching.spatial_x_threshold * img_w_left as f64;
    let x_thresh_right = config.matching.spatial_x_threshold * img_w_right as f64;
    // Inner margin: exclude features at extreme fisheye edges
    let x_inner_left = (1.0 - config.matching.spatial_x_inner) * img_w_left as f64;
    let x_inner_right = config.matching.spatial_x_inner * img_w_right as f64;
    let y_low_left = config.matching.spatial_y_low * img_h_left as f64;
    let y_high_left = config.matching.spatial_y_high * img_h_left as f64;
    let y_low_right = config.matching.spatial_y_low * img_h_right as f64;
    let y_high_right = config.matching.spatial_y_high * img_h_right as f64;

    // Max vertical disparity in pixels (average of both image heights).
    // In a side-by-side stereo rig, matched features should have nearly
    // the same y-coordinate. This catches cross-region mismatches like
    // field markings matched to clouds.
    let avg_h = (img_h_left as f64 + img_h_right as f64) / 2.0;
    let max_y_disp = config.matching.max_y_disparity * avg_h;

    let filtered: Vec<RawMatch> = matches
        .iter()
        .filter(|m| {
            let lp = &kp_left[m.left_idx];
            let rp = &kp_right[m.right_idx];

            let left_x = lp.x as f64;
            let left_y = lp.y as f64;
            let right_x = rp.x as f64;
            let right_y = rp.y as f64;

            let y_disp = (left_y - right_y).abs();

            left_x >= x_thresh_left
                && left_x <= x_inner_left
                && right_x >= x_inner_right
                && right_x <= x_thresh_right
                && left_y >= y_low_left
                && left_y <= y_high_left
                && right_y >= y_low_right
                && right_y <= y_high_right
                && y_disp <= max_y_disp
        })
        .copied()
        .collect();

    if filtered.len() < config.matching.min_matches {
        log::warn!(
            "spatial filter yielded {} matches (< {} required)",
            filtered.len(),
            config.matching.min_matches,
        );
    }
    filtered
}

/// Apply RANSAC outlier rejection using fundamental matrix estimation.
///
/// Uses the `inlier` crate's MAGSAC implementation to robustly estimate
/// the fundamental matrix and identify geometric inliers.
///
/// Returns the indices (into the input `matches` slice) of inlier matches.
pub fn ransac_filter(
    matches: &[RawMatch],
    kp_left: &[KeyPoint],
    kp_right: &[KeyPoint],
    config: &CalibrationConfig,
) -> Result<Vec<usize>, crate::error::CalibrateError> {
    if matches.len() < config.matching.min_matches {
        return Err(crate::error::CalibrateError::InsufficientMatches {
            got: matches.len(),
            min: config.matching.min_matches,
        });
    }

    let n = matches.len();

    let pts1: Vec<[f64; 2]> = matches
        .iter()
        .map(|m| {
            let p = &kp_left[m.left_idx];
            [p.x as f64, p.y as f64]
        })
        .collect();
    let pts2: Vec<[f64; 2]> = matches
        .iter()
        .map(|m| {
            let p = &kp_right[m.right_idx];
            [p.x as f64, p.y as f64]
        })
        .collect();

    match crate::ransac::ransac_fundamental(&pts1, &pts2, config.matching.ransac_threshold, 2000) {
        Ok(inliers) => {
            log::debug!("RANSAC: {}/{} inliers", inliers.len(), n);
            Ok(inliers)
        }
        Err(e) => {
            log::warn!("RANSAC failed: {e}");
            Err(crate::error::CalibrateError::RansacFailed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_keypoint(x: f32, y: f32) -> KeyPoint {
        KeyPoint {
            x,
            y,
            response: 1.0,
        }
    }

    #[test]
    fn spatial_filter_keeps_overlap_region() {
        let config = CalibrationConfig {
            matching: crate::types::MatchConfig {
                min_matches: 1, // low threshold so filter doesn't fall back
                spatial_x_threshold: 0.4,
                ..Default::default()
            },
            ..Default::default()
        };

        // With spatial_x_threshold=0.4 and spatial_x_inner=0.15:
        // Left valid range:  x in [768, 1632]  (0.4*1920 to 0.85*1920)
        // Right valid range: x in [288, 768]   (0.15*1920 to 0.4*1920)
        // Y valid range: [216, 864] (0.2*1080 to 0.8*1080)
        let kp_left = vec![
            make_keypoint(100.0, 540.0),  // x < 768 -> reject
            make_keypoint(900.0, 540.0),  // x in [768, 1632] -> OK
            make_keypoint(1700.0, 540.0), // x > 1632 -> reject (extreme edge)
        ];

        let kp_right = vec![
            make_keypoint(100.0, 540.0), // x < 288 -> reject (extreme edge)
            make_keypoint(500.0, 540.0), // x in [288, 768] -> OK
            make_keypoint(900.0, 540.0), // x > 768 -> reject
        ];

        let matches = vec![
            RawMatch {
                left_idx: 0,
                right_idx: 0,
                distance: 10,
            }, // left rejected
            RawMatch {
                left_idx: 1,
                right_idx: 1,
                distance: 20,
            }, // BOTH OK -> keep
            RawMatch {
                left_idx: 2,
                right_idx: 2,
                distance: 30,
            }, // both rejected
        ];

        let result = spatial_filter(
            &matches, &kp_left, &kp_right, 1920, 1080, 1920, 1080, &config,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].left_idx, 1);
    }

    #[test]
    fn spatial_filter_no_fallback_on_few_matches() {
        let config = CalibrationConfig {
            matching: crate::types::MatchConfig {
                min_matches: 100,
                ..Default::default()
            },
            ..Default::default()
        };

        // This keypoint is outside the spatial overlap region, so the
        // filter should reject it. Previously there was a fallback that
        // returned raw unfiltered matches; now the caller gets the
        // filtered (empty) result and decides what to do.
        let kp_left = vec![make_keypoint(100.0, 540.0)];
        let kp_right = vec![make_keypoint(500.0, 540.0)];
        let matches = vec![RawMatch {
            left_idx: 0,
            right_idx: 0,
            distance: 10,
        }];

        let result = spatial_filter(
            &matches, &kp_left, &kp_right, 1920, 1080, 1920, 1080, &config,
        );

        // No fallback: out-of-region match is rejected, result is empty
        assert!(result.is_empty());
    }
}
