//! Default implementations of the calibration pipeline traits.
//!
//! These wrap the built-in algorithms (AKAZE, Hamming matching, seam-weighted
//! reprojection error) behind the trait interfaces, providing zero-config
//! defaults that work well across GoPro, DJI, and XTU cameras.

use crate::features::{self, Descriptor, DetectRegion, KeyPoint, RawMatch};
use crate::geometry::{self, OptParams};
use crate::traits::{CostFunction, FeatureDetector, FeatureMatcher, PointFilter};
use crate::types::MatchedPoint;

// ---------------------------------------------------------------------------
// FeatureDetector: AKAZE
// ---------------------------------------------------------------------------

/// AKAZE-based feature detector with border filtering.
///
/// Uses the vendored AKAZE implementation with bug fixes. Rejects
/// keypoints near the undistortion boundary (black pincushion edges)
/// to prevent false matches on wide-FOV cameras.
///
/// # Example
///
/// ```ignore
/// let detector = AkazeDetector::new(0.0001);
/// let (kps, descs) = detector.detect(rgba, 3840, 2160, None, 2000);
/// ```
pub struct AkazeDetector {
    /// AKAZE response threshold. Lower = more features.
    pub threshold: f64,
    /// Pixel margin for undistortion border filter. Keypoints within this
    /// distance of a black pixel (pincushion edge) are rejected.
    /// Set to 0 to disable. Only applies to images wider than 1000px.
    pub border_margin: i32,
}

impl AkazeDetector {
    /// Create a new AKAZE detector with the given threshold and default
    /// border margin (30px).
    ///
    /// Recommended: 0.0001 for calibration (sensitive), 0.001 for fast detection.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            border_margin: 30,
        }
    }

    /// Create a detector with custom threshold and border margin.
    pub fn with_border_margin(threshold: f64, border_margin: i32) -> Self {
        Self {
            threshold,
            border_margin,
        }
    }
}

impl Default for AkazeDetector {
    fn default() -> Self {
        Self {
            threshold: 0.0001,
            border_margin: 30,
        }
    }
}

impl FeatureDetector for AkazeDetector {
    fn detect(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        region: Option<DetectRegion>,
        max_keypoints: usize,
    ) -> (Vec<KeyPoint>, Vec<Descriptor>) {
        // Run core AKAZE detection (includes border filter at the configured margin)
        let (kps, descs) = features::detect_with_border(
            rgba,
            width,
            height,
            region,
            max_keypoints,
            self.threshold,
            self.border_margin,
        );
        (kps, descs)
    }
}

// ---------------------------------------------------------------------------
// FeatureMatcher: Brute-force Hamming with Lowe's ratio test
// ---------------------------------------------------------------------------

/// Brute-force Hamming distance matcher with Lowe's ratio test.
///
/// Compares every left descriptor to every right descriptor. The ratio
/// test rejects matches where the best match isn't significantly better
/// than the second-best, reducing false positives.
pub struct HammingMatcher {
    /// Lowe's ratio test threshold. Lower = stricter.
    pub lowe_ratio: f64,
}

impl HammingMatcher {
    /// Create a new matcher with the given Lowe's ratio threshold.
    ///
    /// Recommended: 0.75 for calibration, 0.6 for strict matching.
    pub fn new(lowe_ratio: f64) -> Self {
        Self { lowe_ratio }
    }
}

impl Default for HammingMatcher {
    fn default() -> Self {
        Self { lowe_ratio: 0.75 }
    }
}

impl FeatureMatcher for HammingMatcher {
    fn match_features(&self, left: &[Descriptor], right: &[Descriptor]) -> Vec<RawMatch> {
        features::match_descriptors(left, right, self.lowe_ratio)
    }
}

// ---------------------------------------------------------------------------
// PointFilter: No-op (passthrough)
// ---------------------------------------------------------------------------

/// A point filter that passes all points through unchanged.
///
/// Used as the default in [`calibrate()`](crate::calibrate) where the
/// spatial filter and RANSAC already handle outlier rejection. Plug in
/// [`YDisparityFilter`] or a custom implementation when additional
/// post-normalization filtering is needed.
pub struct NoOpFilter;

impl PointFilter for NoOpFilter {
    fn filter(&self, points: &[MatchedPoint]) -> Vec<MatchedPoint> {
        points.to_vec()
    }
}

// ---------------------------------------------------------------------------
// PointFilter: Y-disparity
// ---------------------------------------------------------------------------

/// Filters matched points by vertical disparity.
///
/// In a side-by-side stereo rig, matched features should have nearly
/// the same y-coordinate. Points with large vertical offset are likely
/// mismatches (e.g., field markings matched to clouds).
pub struct YDisparityFilter {
    /// Maximum allowed y-disparity as a fraction of image height.
    pub max_disparity: f64,
}

impl Default for YDisparityFilter {
    fn default() -> Self {
        Self {
            max_disparity: 0.08,
        }
    }
}

impl PointFilter for YDisparityFilter {
    fn filter(&self, points: &[MatchedPoint]) -> Vec<MatchedPoint> {
        points
            .iter()
            .filter(|p| {
                // left[1] and right[1] are plane y-coordinates.
                // For a stereo rig, these should be close.
                let dy = (p.left[1] - p.right[1]).abs();
                dy < self.max_disparity
            })
            .copied()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CostFunction: Seam-weighted reprojection error
// ---------------------------------------------------------------------------

/// Seam-weighted symmetric reprojection error.
///
/// Weights each point by proximity to the stitch seam (horizontal) and
/// image center (vertical). Points near the seam contribute more to the
/// cost since that's where alignment matters most visually.
///
/// Supports optional trimming: the worst `trim_fraction` of points are
/// dropped before summing, making the cost robust to outlier matches.
pub struct SeamWeightedCost {
    /// Gaussian sigma for seam-proximity weighting.
    pub sigma: f64,
    /// Fraction of worst points to drop (0.0 = no trimming).
    pub trim_fraction: f64,
}

impl SeamWeightedCost {
    /// Create a new seam-weighted cost function.
    pub fn new(sigma: f64, trim_fraction: f64) -> Self {
        Self {
            sigma,
            trim_fraction,
        }
    }
}

impl Default for SeamWeightedCost {
    fn default() -> Self {
        Self {
            sigma: 0.08,
            trim_fraction: 0.3,
        }
    }
}

impl CostFunction for SeamWeightedCost {
    fn cost(&self, points: &[MatchedPoint], params: &OptParams) -> f64 {
        if self.trim_fraction > 0.0 {
            geometry::trimmed_seam_weighted_reprojection_error(
                points,
                params,
                self.sigma,
                self.trim_fraction,
            )
        } else {
            geometry::seam_weighted_reprojection_error(points, params, self.sigma)
        }
    }

    fn per_point_cost(&self, points: &[MatchedPoint], params: &OptParams) -> Vec<f64> {
        geometry::per_point_seam_weighted_errors(points, params, self.sigma)
    }
}

/// Raw (unweighted) reprojection error.
///
/// All points contribute equally regardless of position. Useful for
/// diagnostics and comparison with seam-weighted results.
pub struct RawReprojectionCost {
    /// Fraction of worst points to drop (0.0 = no trimming).
    pub trim_fraction: f64,
}

impl Default for RawReprojectionCost {
    fn default() -> Self {
        Self { trim_fraction: 0.3 }
    }
}

impl CostFunction for RawReprojectionCost {
    fn cost(&self, points: &[MatchedPoint], params: &OptParams) -> f64 {
        if self.trim_fraction > 0.0 {
            geometry::trimmed_reprojection_error(points, params, self.trim_fraction)
        } else {
            geometry::reprojection_error(points, params)
        }
    }

    fn per_point_cost(&self, points: &[MatchedPoint], params: &OptParams) -> Vec<f64> {
        geometry::per_point_reprojection_error(points, params)
    }
}
