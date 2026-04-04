//! Data types for the calibration pipeline.
//!
//! These types carry frame data, matched features, configuration, and
//! calibration results through the pipeline stages.

use reco_core::calibration::MatchCalibration;
use serde::{Deserialize, Serialize};

/// A grayscale image (8-bit, row-major, tightly packed).
#[derive(Clone)]
pub struct GrayFrame {
    /// Pixel data, row-major, one byte per pixel.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// A YUV420P frame (native decoder output).
///
/// Y is full resolution, U and V are half width/height.
#[derive(Clone)]
pub struct YuvFrame {
    /// Luma plane (`width * height` bytes).
    pub y: Vec<u8>,
    /// Cb chroma plane (`width/2 * height/2` bytes).
    pub u: Vec<u8>,
    /// Cr chroma plane (`width/2 * height/2` bytes).
    pub v: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// A pair of corresponding points in normalized plane coordinates.
///
/// Coordinates are in the range `[-0.5, 0.5]` for X and
/// `[-h/(2w), h/(2w)]` for Y, where the plane width is normalized to 1.0.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MatchedPoint {
    /// Point on the left plane (x-plane in optimizer space).
    pub left: [f64; 2],
    /// Point on the right plane (z-plane in optimizer space).
    pub right: [f64; 2],
}

/// Feature matching statistics for a single frame pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMatches {
    /// Matched point pairs surviving all filters.
    pub points: Vec<MatchedPoint>,
    /// Number of keypoints detected in the left image.
    pub keypoints_left: usize,
    /// Number of keypoints detected in the right image.
    pub keypoints_right: usize,
    /// Number of raw descriptor matches before filtering.
    pub raw_matches: usize,
    /// Matches surviving Lowe's ratio test.
    pub post_ratio_test: usize,
    /// Matches surviving the spatial overlap filter.
    pub post_spatial_filter: usize,
    /// Matches surviving RANSAC outlier rejection.
    pub post_ransac: usize,
}

/// Configuration for the calibration pipeline.
///
/// All fields have sensible defaults via [`Default`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Number of frame pairs to sample from the video.
    pub num_frames: usize,
    /// Number of random-subset optimization iterations.
    pub iterations: usize,
    /// Lowe's ratio test threshold (lower = stricter).
    pub lowe_ratio: f64,
    /// Minimum number of matches required per frame pair.
    pub min_matches: usize,
    /// RANSAC confidence level.
    pub ransac_confidence: f64,
    /// RANSAC inlier threshold (Sampson error).
    pub ransac_threshold: f64,
    /// Spatial filter: left image x threshold (keep x >= this fraction of width).
    pub spatial_x_threshold: f64,
    /// Inner overlap margin: exclude features deeper than this fraction from
    /// the image edge. Fisheye images have severe distortion at the extremes,
    /// producing unreliable features. 0.15 = ignore the outermost 15% on the
    /// overlap side.
    pub spatial_x_inner: f64,
    /// Spatial filter: vertical band lower bound (fraction of height).
    pub spatial_y_low: f64,
    /// Spatial filter: vertical band upper bound (fraction of height).
    pub spatial_y_high: f64,
    /// Maximum allowed vertical disparity between matched keypoints,
    /// as a fraction of image height. Rejects matches where features
    /// are at very different y-positions (e.g. field lines vs clouds).
    pub max_y_disparity: f64,
    /// Fraction of the image height to mask from the top (sky removal).
    /// 0.3 = ignore the top 30% of the image for feature detection.
    pub sky_mask_ratio: f64,
    /// Maximum number of keypoints to keep per image after detection,
    /// sorted by response strength. Matches v1's SIFT nfeatures behavior.
    pub max_keypoints: usize,
    /// Enable the 6th parameter (left plane roll) for horizon correction.
    pub enable_sixth_param: bool,
    /// Fraction of total matches used per random subset (0.0-1.0).
    pub subset_ratio: f64,
    /// Maximum optimizer function evaluations per iteration.
    pub max_optimizer_evals: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_frames: 15,
            iterations: 200,
            lowe_ratio: 0.6,
            min_matches: 8,
            ransac_confidence: 0.995,
            ransac_threshold: 1.0,
            spatial_x_threshold: 0.4,
            spatial_x_inner: 0.0,
            spatial_y_low: 0.2,
            spatial_y_high: 0.8,
            max_y_disparity: 0.1,
            sky_mask_ratio: 0.3,
            max_keypoints: 2000,
            enable_sixth_param: true,
            subset_ratio: 0.6,
            max_optimizer_evals: 1000,
        }
    }
}

/// Output of a successful calibration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// The computed calibration (ready to serialize as match.json).
    pub calibration: MatchCalibration,
    /// Total number of matched point pairs across all frames.
    pub total_matches: usize,
    /// Number of frame pairs that produced usable matches.
    pub frames_used: usize,
    /// Residual angular error at the optimum (radians).
    pub residual_error: f64,
    /// Calibration confidence score (0.0-1.0).
    pub confidence: f64,
    /// Per-frame matching statistics.
    pub per_frame: Vec<FrameMatches>,
}
