//! Trait definitions for the calibration pipeline.
//!
//! Each stage of the pipeline is behind a trait, making it easy to
//! swap implementations, test in isolation, and use individual stages
//! independently. Default implementations wrap the built-in algorithms.
//!
//! ## Pipeline stages
//!
//! ```text
//! Images -> FeatureDetector -> FeatureMatcher -> PointFilter -> CostFunction + Optimizer -> PlaneLayout
//! ```
//!
//! ## Standalone usage
//!
//! Each trait can be used independently. For example, use just the
//! feature detector without the full calibration pipeline:
//!
//! ```ignore
//! use reco_calibrate::traits::FeatureDetector;
//! use reco_calibrate::AkazeDetector;
//!
//! let detector = AkazeDetector::new(0.0001);
//! let (keypoints, descriptors) = detector.detect(rgba, width, height, Some(region));
//! ```

use crate::features::{Descriptor, KeyPoint, RawMatch};
use crate::geometry::OptParams;
use crate::types::MatchedPoint;

/// Detects feature keypoints and computes descriptors from an image.
///
/// Implementations should handle:
/// - Converting input format (RGBA) to what the detector needs
/// - Downscaling for performance if appropriate
/// - Region-of-interest filtering
/// - Border filtering (reject features near undistortion edges)
pub trait FeatureDetector: Send + Sync {
    /// Detect keypoints and compute descriptors from RGBA image data.
    ///
    /// Returns matched vectors of keypoints and their descriptors.
    /// `region` restricts detection to a fraction of the image.
    /// `max_keypoints` caps the output by response strength.
    fn detect(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        region: Option<crate::features::DetectRegion>,
        max_keypoints: usize,
    ) -> (Vec<KeyPoint>, Vec<Descriptor>);
}

/// Matches feature descriptors between two images.
///
/// Takes descriptor sets from left and right images and produces
/// a list of raw matches, optionally filtered by ratio test or
/// cross-checking.
pub trait FeatureMatcher: Send + Sync {
    /// Find matches between left and right descriptor sets.
    fn match_features(&self, left: &[Descriptor], right: &[Descriptor]) -> Vec<RawMatch>;
}

/// Filters matched points to remove outliers.
///
/// Applied after feature matching to reject bad correspondences
/// before they reach the optimizer. Multiple filters can be chained.
pub trait PointFilter: Send + Sync {
    /// Filter a set of matched points, returning only the inliers.
    fn filter(&self, points: &[MatchedPoint]) -> Vec<MatchedPoint>;
}

/// Computes calibration error from matched points and parameters.
///
/// The cost function is the objective that the optimizer minimizes.
/// Different implementations weight points differently or use
/// different geometric error metrics.
pub trait CostFunction: Send + Sync {
    /// Compute the total cost for the given parameters.
    fn cost(&self, points: &[MatchedPoint], params: &OptParams) -> f64;

    /// Compute per-point costs (for trimming and diagnostics).
    fn per_point_cost(&self, points: &[MatchedPoint], params: &OptParams) -> Vec<f64>;
}

