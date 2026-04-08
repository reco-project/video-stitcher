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
///
/// The `left_pixel_nx` / `right_pixel_nx` fields store the original
/// pixel x-coordinate normalized to `[0, 1]`. These are used for
/// seam-proximity weighting during optimization: points near the stitch
/// seam are weighted more heavily than points far from it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MatchedPoint {
    /// Point on the left plane (x-plane in optimizer space).
    pub left: [f64; 2],
    /// Point on the right plane (z-plane in optimizer space).
    pub right: [f64; 2],
    /// Normalized x-coordinate of the left point in pixel space `[0, 1]`.
    #[serde(default)]
    pub left_pixel_nx: f64,
    /// Normalized x-coordinate of the right point in pixel space `[0, 1]`.
    #[serde(default)]
    pub right_pixel_nx: f64,
}

impl MatchedPoint {
    /// Create a matched point from plane coordinates only.
    ///
    /// Sets pixel coordinates to 0.5 (image center), giving uniform
    /// weight in seam-weighted optimization. Use this for synthetic
    /// test data where pixel position is irrelevant.
    pub fn from_planes(left: [f64; 2], right: [f64; 2]) -> Self {
        Self {
            left,
            right,
            left_pixel_nx: 0.5,
            right_pixel_nx: 0.5,
        }
    }
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
    /// Deprecated: use `detect_y_min` instead.
    pub sky_mask_ratio: f64,
    /// Detection region vertical minimum (fraction of height, 0.0 = top).
    /// Points above this line are excluded from feature detection.
    /// 0.25 = skip top 25% (avoids sky and undistortion edge artifacts).
    pub detect_y_min: f64,
    /// Detection region vertical maximum (fraction of height, 1.0 = bottom).
    /// Points below this line are excluded from feature detection.
    /// 0.85 = skip bottom 15% (avoids close-to-camera ground and undistortion edges).
    pub detect_y_max: f64,
    /// Maximum number of keypoints to keep per image after detection,
    /// sorted by response strength. Matches v1's SIFT nfeatures behavior.
    pub max_keypoints: usize,
    /// Fraction of total matches used per random subset (0.0-1.0).
    pub subset_ratio: f64,
    /// Maximum optimizer function evaluations per iteration.
    pub max_optimizer_evals: usize,
    /// Seconds to skip from the start of the video (setup time).
    pub skip_start_secs: f64,
    /// Seconds to skip from the end of the video (teardown time).
    pub skip_end_secs: f64,

    // --- Optimizer settings ---
    /// Lock cam_d to half_offset (cam_d = 0.5 * (1 - intersect)).
    /// Reduces the optimization to 4 parameters. Based on the finding
    /// that the virtual camera sits at the intersection of the plane normals.
    pub lock_cam_d: bool,

    /// AKAZE detector response threshold. Lower = more features detected.
    /// Default 0.001. Try 0.0005 or 0.0001 for denser detection.
    pub akaze_threshold: f64,

    /// Gaussian sigma for seam-proximity weighting in the objective function.
    ///
    /// Points near the stitch seam are weighted more heavily. A smaller
    /// sigma concentrates weight tighter around the seam. Confirmed
    /// "much better" than unweighted at sigma=0.08 across all test footages.
    pub seam_sigma: f64,

    // --- IMU-derived settings (populated by telemetry module) ---
    /// IMU-derived roll seed for the x_rz initial guess (radians).
    ///
    /// When set, adds an extra optimizer start point seeded with this
    /// value, improving convergence for rigs with measurable roll offset.
    pub imu_xrz_seed: Option<f64>,
    /// Enable the x_rx parameter (right plane pitch).
    ///
    /// Auto-enabled when IMU detects differential pitch > 2 degrees
    /// between cameras. When false, x_rx is fixed at 0.
    pub enable_x_rx: bool,
    /// IMU-derived pitch seed for the x_rx initial guess (radians).
    ///
    /// When set (along with `enable_x_rx`), seeds the x_rx parameter
    /// from the IMU differential pitch, improving convergence.
    pub imu_xrx_seed: Option<f64>,
    /// IMU-derived tilt seed for the z_rx initial guess (radians).
    ///
    /// The left camera's deviation from the rig average tilt, computed
    /// by subtracting the common rig tilt from the left camera's
    /// individual tilt measurement.
    pub imu_zrx_seed: Option<f64>,

    /// Fraction of worst-error points to drop during optimization.
    ///
    /// 0.0 = no trimming (use all points), 0.2 = drop worst 20%.
    /// Makes the optimizer robust to outlier matches that survive RANSAC.
    pub trim_fraction: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_frames: 15,
            iterations: 200,
            lowe_ratio: 0.75,
            min_matches: 6,
            ransac_confidence: 0.995,
            ransac_threshold: 1.0,
            spatial_x_threshold: 0.5,
            spatial_x_inner: 0.0,
            spatial_y_low: 0.2,
            spatial_y_high: 0.8,
            max_y_disparity: 0.08,
            sky_mask_ratio: 0.25,
            detect_y_min: 0.25,
            detect_y_max: 0.85,
            max_keypoints: 2000,
            subset_ratio: 0.6,
            max_optimizer_evals: 1000,
            skip_start_secs: 0.0,
            skip_end_secs: 0.0,
            lock_cam_d: false,
            akaze_threshold: 0.0001,
            seam_sigma: 0.08,
            imu_xrz_seed: None,
            enable_x_rx: false,
            imu_xrx_seed: None,
            imu_zrx_seed: None,
            trim_fraction: 0.3,
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
