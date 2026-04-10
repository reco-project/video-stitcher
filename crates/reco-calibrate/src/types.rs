//! Data types for the calibration pipeline.
//!
//! These types carry frame data, matched features, configuration, and
//! calibration results through the pipeline stages.

use reco_core::calibration::MatchCalibration;
use serde::{Deserialize, Serialize};

/// A grayscale image frame for feature detection (8-bit, row-major, tightly packed).
#[derive(Clone)]
pub struct GrayFrame {
    /// Pixel data, row-major, one byte per pixel.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// Re-export the canonical YUV frame type from reco-core.
///
/// This is the standard YUV420P frame used across all reco crates.
/// Previously reco-calibrate defined its own copy; now it uses the
/// shared definition from [`reco_core::source::YuvFrame`].
pub use reco_core::source::YuvFrame;

/// A matched feature point pair in normalized plane coordinates.
///
/// Each point represents the same physical feature seen by both cameras,
/// mapped onto the two-plane geometric model. The `left`/`right` fields
/// use the optimizer's swap convention (right camera -> left plane, etc.).
///
/// Coordinates are in the range `[-0.5, 0.5]` for X and
/// `[-h/(2w), h/(2w)]` for Y, where the plane width is normalized to 1.0.
/// The `left_pixel_nx` / `right_pixel_nx` fields store the original
/// pixel x-coordinate normalized to `[0, 1]`, used for seam-proximity
/// weighting during optimization.
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

/// Per-frame feature matching statistics and matched points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMatches {
    /// Matched point pairs surviving all filters.
    pub points: Vec<MatchedPoint>,
    /// Number of keypoints detected in the left image.
    pub keypoints_left: usize,
    /// Number of keypoints detected in the right image.
    pub keypoints_right: usize,
    /// Minimum descriptor count across both images (diagnostic baseline).
    #[serde(alias = "raw_matches")]
    pub min_descriptors: usize,
    /// Matches surviving Lowe's ratio test.
    pub post_ratio_test: usize,
    /// Matches surviving the spatial overlap filter.
    pub post_spatial_filter: usize,
    /// Matches surviving RANSAC outlier rejection.
    pub post_ransac: usize,
}

/// AKAZE feature detection settings.
///
/// Controls the AKAZE detector's sensitivity, keypoint cap, and the
/// vertical region of interest used during feature detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AkazeConfig {
    /// AKAZE detector response threshold. Lower = more features detected.
    /// Default 0.0001. Try 0.001 for faster detection with fewer features.
    #[serde(rename = "akaze_threshold")]
    pub threshold: f64,
    /// Maximum number of keypoints to keep per image after detection,
    /// sorted by response strength. Matches v1's SIFT nfeatures behavior.
    pub max_keypoints: usize,
    /// Detection region vertical minimum (fraction of height, 0.0 = top).
    /// Points above this line are excluded from feature detection.
    /// Default 0.05 (skip top 5%). The border filter handles undistortion
    /// edge artifacts, so this only needs to exclude the extreme edges.
    pub detect_y_min: f64,
    /// Detection region vertical maximum (fraction of height, 1.0 = bottom).
    /// Points below this line are excluded from feature detection.
    /// Default 0.95 (skip bottom 5%). The border filter handles edge artifacts.
    pub detect_y_max: f64,
}

impl Default for AkazeConfig {
    fn default() -> Self {
        Self {
            threshold: 0.0001,
            max_keypoints: 2000,
            detect_y_min: 0.05,
            detect_y_max: 0.95,
        }
    }
}

/// Feature matching and spatial filtering settings.
///
/// Controls Lowe's ratio test, spatial overlap filtering, vertical
/// disparity rejection, and RANSAC outlier removal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchConfig {
    /// Lowe's ratio test threshold (lower = stricter).
    pub lowe_ratio: f64,
    /// Minimum number of matches required per frame pair.
    pub min_matches: usize,
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
    /// RANSAC inlier threshold (Sampson error).
    pub ransac_threshold: f64,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            lowe_ratio: 0.75,
            min_matches: 6,
            spatial_x_threshold: 0.5,
            spatial_x_inner: 0.0,
            spatial_y_low: 0.2,
            spatial_y_high: 0.8,
            max_y_disparity: 0.08,
            ransac_threshold: 1.0,
        }
    }
}

/// Nelder-Mead optimizer settings.
///
/// Controls parameter locking, seam weighting, trimming, and
/// iteration limits for the optimization pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Removes `cam_d` from free parameters (derives from `intersect`).
    pub lock_cam_d: bool,
    /// Lock z_rx to 0 (left/z-plane stays static, only translates via intersect).
    /// Reduces the optimization by 1 parameter.
    pub lock_z_rx: bool,
    /// Enable the x_rx parameter (right plane pitch).
    ///
    /// Auto-enabled when IMU detects differential pitch > 2 degrees
    /// between cameras. When false, x_rx is fixed at 0.
    pub enable_x_rx: bool,
    /// Horizontal Gaussian sigma for seam-proximity weighting.
    ///
    /// Controls horizontal (seam-proximity) width; vertical sigma is
    /// fixed at 0.08. Points near the stitch seam are weighted more
    /// heavily. A smaller sigma concentrates weight tighter around the
    /// seam. Confirmed "much better" than unweighted across all test
    /// footages.
    pub seam_sigma: f64,
    /// Fraction of worst-error points to drop during optimization.
    ///
    /// 0.0 = no trimming (use all points), 0.2 = drop worst 20%.
    /// Makes the optimizer robust to outlier matches that survive RANSAC.
    pub trim_fraction: f64,
    /// Maximum optimizer iterations per Nelder-Mead run.
    #[serde(rename = "max_optimizer_iters")]
    pub max_iters: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            lock_cam_d: false,
            lock_z_rx: false,
            enable_x_rx: false,
            seam_sigma: 0.08,
            trim_fraction: 0.3,
            max_iters: 5000,
        }
    }
}

/// Configuration for the calibration pipeline.
///
/// Groups detection, matching, and optimizer settings into sub-structs
/// ([`AkazeConfig`], [`MatchConfig`], [`OptimizerConfig`]) while keeping
/// sampling and IMU fields at the top level.
///
/// Uses `#[serde(flatten)]` so the JSON representation stays flat -
/// existing config files continue to work without changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Number of frame pairs to sample from the video.
    pub num_frames: usize,
    /// Seconds to skip from the start of the video (setup time).
    pub skip_start_secs: f64,
    /// Seconds to skip from the end of the video (teardown time).
    pub skip_end_secs: f64,

    // --- IMU-derived settings (populated by telemetry module) ---
    /// IMU-derived roll seed for the x_rz initial guess (radians).
    ///
    /// When set, adds an extra optimizer start point seeded with this
    /// value, improving convergence for rigs with measurable roll offset.
    pub imu_xrz_seed: Option<f64>,
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

    /// AKAZE feature detection settings.
    #[serde(flatten)]
    pub akaze: AkazeConfig,
    /// Feature matching and filtering settings.
    #[serde(flatten)]
    pub matching: MatchConfig,
    /// Optimizer settings.
    #[serde(flatten)]
    pub optimizer: OptimizerConfig,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_frames: 2,
            skip_start_secs: 0.0,
            skip_end_secs: 0.0,
            imu_xrz_seed: None,
            imu_xrx_seed: None,
            imu_zrx_seed: None,
            akaze: AkazeConfig::default(),
            matching: MatchConfig::default(),
            optimizer: OptimizerConfig::default(),
        }
    }
}

impl CalibrationConfig {
    /// Validate configuration values before starting calibration.
    ///
    /// Returns `Err` with a description of the first invalid field found.
    pub fn validate(&self) -> Result<(), crate::error::CalibrateError> {
        use crate::error::CalibrateError;

        if self.num_frames == 0 {
            return Err(CalibrateError::InvalidConfig(
                "num_frames must be >= 1".into(),
            ));
        }
        if self.matching.lowe_ratio <= 0.0 || self.matching.lowe_ratio > 1.0 {
            return Err(CalibrateError::InvalidConfig(format!(
                "lowe_ratio must be in (0, 1], got {}",
                self.matching.lowe_ratio
            )));
        }
        if self.matching.ransac_threshold <= 0.0 {
            return Err(CalibrateError::InvalidConfig(format!(
                "ransac_threshold must be > 0, got {}",
                self.matching.ransac_threshold
            )));
        }
        if self.akaze.max_keypoints == 0 {
            return Err(CalibrateError::InvalidConfig(
                "max_keypoints must be >= 1".into(),
            ));
        }
        if self.akaze.threshold <= 0.0 {
            return Err(CalibrateError::InvalidConfig(format!(
                "akaze_threshold must be > 0, got {}",
                self.akaze.threshold
            )));
        }
        if !(0.0..=1.0).contains(&self.optimizer.trim_fraction) {
            return Err(CalibrateError::InvalidConfig(format!(
                "trim_fraction must be in [0, 1], got {}",
                self.optimizer.trim_fraction
            )));
        }
        if self.matching.spatial_x_threshold < 0.0 || self.matching.spatial_x_threshold > 1.0 {
            return Err(CalibrateError::InvalidConfig(format!(
                "spatial_x_threshold must be in [0, 1], got {}",
                self.matching.spatial_x_threshold
            )));
        }
        if self.optimizer.seam_sigma <= 0.0 {
            return Err(CalibrateError::InvalidConfig(format!(
                "seam_sigma must be > 0, got {}",
                self.optimizer.seam_sigma
            )));
        }
        Ok(())
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
    /// Residual seam-weighted reprojection error at the optimum (dimensionless).
    pub residual_error: f64,
    /// Calibration confidence score (0.0-1.0).
    pub confidence: f64,
    /// Per-frame matching statistics.
    pub per_frame: Vec<FrameMatches>,
}
