//! Calibration data structures and JSON parsing.
//!
//! Defines the camera intrinsics and plane layout parameters used by the
//! stitching pipeline. These are loaded from calibration JSON files produced
//! by the v1 feature-matching and position-optimization pipeline.
//!
//! ## Data Format
//!
//! The calibration consists of two parts:
//! - [`CameraParams`]: per-camera intrinsics (focal length, principal point, distortion)
//! - [`PlaneLayout`]: relative positioning of the two camera planes in 3D space
//!
//! The distortion model is `fisheye_kb4` (Kannala-Brandt 4-coefficient):
//! ```text
//! θ_d = θ × (1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum allowed dimension (width or height) in pixels.
///
/// Values above this threshold indicate a malformed calibration file and would
/// cause the GPU allocator to request an unreasonably large texture.
pub const MAX_DIM: u32 = 8192;

/// Minimum positive value accepted for focal lengths and the camera axis offset.
///
/// Values at or below this threshold would cause division-by-zero or
/// zero-vector normalization in the stitching shaders.
const EPSILON: f64 = 1e-6;

/// Errors produced by [`MatchCalibration::validate`].
#[derive(Debug, Error)]
pub enum CalibrationError {
    /// A required dimension (width or height) is zero.
    #[error("{camera} camera {field} must be > 0, got {value}")]
    ZeroDimension {
        /// Which camera the error belongs to (`"left"` or `"right"`).
        camera: &'static str,
        /// Field name (`"width"` or `"height"`).
        field: &'static str,
        /// The offending value.
        value: u32,
    },

    /// A dimension exceeds [`MAX_DIM`] and would cause an excessive GPU allocation.
    #[error("{camera} camera {field} exceeds maximum allowed value of {max}, got {value}")]
    DimensionTooLarge {
        /// Which camera the error belongs to.
        camera: &'static str,
        /// Field name.
        field: &'static str,
        /// The offending value.
        value: u32,
        /// The limit that was exceeded.
        max: u32,
    },

    /// A float field contains a non-finite value (NaN or infinity).
    #[error("field '{field}' must be finite, got {value}")]
    NonFiniteFloat {
        /// Dotted field path, e.g. `"left.fx"` or `"params.d[2]"`.
        field: String,
        /// The string representation of the offending value.
        value: String,
    },

    /// A focal length is too small, which would cause division-by-zero in the shader.
    #[error("field '{field}' must be > {epsilon}, got {value}")]
    FocalLengthTooSmall {
        /// Field path.
        field: &'static str,
        /// The offending value.
        value: f64,
        /// The minimum threshold.
        epsilon: f64,
    },

    /// `camera_axis_offset` is too small, which would cause zero-vector normalization.
    #[error("params.cameraAxisOffset must be > {epsilon}, got {value}")]
    AxisOffsetTooSmall {
        /// The offending value.
        value: f64,
        /// The minimum threshold.
        epsilon: f64,
    },

    /// `intersect` is outside the valid `[0.0, 1.0]` range.
    #[error("params.intersect must be in [0.0, 1.0], got {value}")]
    IntersectOutOfRange {
        /// The offending value.
        value: f64,
    },
}

/// Camera intrinsic parameters from a lens profile.
///
/// Contains the focal length, principal point, and distortion coefficients
/// for a single camera. These values are derived from camera calibration
/// (e.g., using a checkerboard pattern) and stored in lens profile JSON files.
///
/// # Coordinate System
///
/// - `fx`, `fy`: focal lengths in pixel units
/// - `cx`, `cy`: principal point (optical center) in pixel coordinates
/// - `d`: four distortion coefficients for the fisheye KB4 model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraParams {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Focal length along the x-axis, in pixels.
    pub fx: f64,
    /// Focal length along the y-axis, in pixels.
    pub fy: f64,
    /// Principal point x-coordinate, in pixels.
    pub cx: f64,
    /// Principal point y-coordinate, in pixels.
    pub cy: f64,
    /// Fisheye KB4 distortion coefficients `[k1, k2, k3, k4]`.
    pub d: [f64; 4],
}

/// Plane layout parameters defining the 3D arrangement of two camera planes.
///
/// These parameters are computed by the position optimization algorithm, which
/// minimizes the angular error between matched feature points across the two
/// camera views.
///
/// # 3D Model
///
/// ```text
///        Z
///        │  ┌──────────┐ Left plane (X-Z)
///        │  │           │
///        │  │           │
///        └──┼───────────┼──── X
///           │           │
///           └──────────┘ Right plane (Y-Z)
///
///   Camera at [camera_axis_offset, 0, camera_axis_offset]
/// ```
///
/// The `intersect` parameter controls how much the planes overlap at the corner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaneLayout {
    /// Distance of the virtual camera from the origin along both X and Z axes.
    ///
    /// The camera is placed at `[camera_axis_offset, 0, camera_axis_offset]`.
    /// Typical range: 0.1–0.35.
    #[serde(rename = "cameraAxisOffset")]
    pub camera_axis_offset: f64,

    /// Overlap ratio between the two planes.
    ///
    /// `0.0` = no overlap, `1.0` = full overlap.
    /// Each plane is translated by `(plane_width / 2) × (1 - intersect)`.
    /// Typical range: 0.2–0.8.
    pub intersect: f64,

    /// Y-axis translation of the right plane, correcting vertical misalignment.
    #[serde(rename = "xTy")]
    pub x_ty: f64,

    /// Z-axis rotation of the right plane (radians), correcting rotational misalignment.
    #[serde(rename = "xRz")]
    pub x_rz: f64,

    /// X-axis rotation of the left plane (radians), correcting tilt misalignment.
    #[serde(rename = "zRx")]
    pub z_rx: f64,

    /// X-axis rotation of the right plane (radians), correcting pitch misalignment.
    ///
    /// Together with xRz (roll), this fully describes the right camera's
    /// orientation. Defaults to 0.0 for backward compatibility with v1.
    #[serde(rename = "xRx", default)]
    pub x_rx: f64,

    /// Z-axis rotation of the left plane (radians), correcting pitch misalignment.
    ///
    /// Together with zRx (roll), this fully describes the left camera's
    /// orientation. Defaults to 0.0 for backward compatibility with v1.
    #[serde(rename = "zRz", default)]
    pub z_rz: f64,
}

/// Complete calibration data for a stereo match.
///
/// Combines per-camera intrinsics with the relative plane layout.
/// This is everything the stitching pipeline needs to render a panorama.
///
/// # JSON Format
///
/// Compatible with the v1 match JSON format:
/// ```json
/// {
///   "left_uniforms": { "width": 3840, "height": 2160, "fx": 1796.32, ... },
///   "right_uniforms": { "width": 3840, "height": 2160, "fx": 1796.32, ... },
///   "params": { "cameraAxisOffset": 0.24, "intersect": 0.54, ... }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchCalibration {
    /// Left camera intrinsic parameters.
    #[serde(rename = "left_uniforms")]
    pub left: CameraParams,

    /// Right camera intrinsic parameters.
    #[serde(rename = "right_uniforms")]
    pub right: CameraParams,

    /// 3D plane layout parameters.
    #[serde(rename = "params")]
    pub layout: PlaneLayout,

    /// Rig tilt in radians (forward lean from vertical).
    ///
    /// Computed from IMU accelerometer during calibration. Applied in the
    /// renderer to straighten vertical lines at the panorama edges.
    /// Defaults to 0.0 for backward compatibility with older calibrations.
    #[serde(default)]
    pub rig_tilt: f64,

    /// Temporal sync offset in frames (positive = right video is ahead).
    ///
    /// Computed from IMU gyro or audio cross-correlation during calibration.
    /// Defaults to 0 for backward compatibility with older calibrations.
    #[serde(default)]
    pub sync_offset: i64,
}

/// Maximum calibration file size (1 MB) to prevent loading unreasonably large files.
const MAX_CALIBRATION_FILE_SIZE: u64 = 1_048_576;

impl MatchCalibration {
    /// Load and validate a calibration from a JSON file.
    ///
    /// Checks file size (max 1 MB), parses JSON, and runs
    /// [`validate`](Self::validate). Returns a descriptive error on any failure.
    pub fn from_file(path: &std::path::Path) -> Result<Self, CalibrationLoadError> {
        let meta = std::fs::metadata(path).map_err(|e| CalibrationLoadError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        if meta.len() > MAX_CALIBRATION_FILE_SIZE {
            return Err(CalibrationLoadError::TooLarge {
                size: meta.len(),
                max: MAX_CALIBRATION_FILE_SIZE,
            });
        }
        let json = std::fs::read_to_string(path).map_err(|e| CalibrationLoadError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        let cal: Self = serde_json::from_str(&json).map_err(CalibrationLoadError::Parse)?;
        cal.validate()?;
        Ok(cal)
    }

    /// Validates all calibration parameters before they are used by the GPU pipeline.
    ///
    /// Catches malformed values that would otherwise cause GPU hangs, shader
    /// division-by-zero, or excessive memory allocations.
    ///
    /// # Errors
    ///
    /// Returns [`CalibrationError`] describing the first invalid field found.
    pub fn validate(&self) -> Result<(), CalibrationError> {
        validate_camera_params(&self.left, "left")?;
        validate_camera_params(&self.right, "right")?;
        validate_layout(&self.layout)?;
        Ok(())
    }
}

/// Errors from loading a calibration file.
#[derive(Debug, Error)]
pub enum CalibrationLoadError {
    /// File I/O error.
    #[error("cannot read calibration file '{path}': {source}")]
    Io {
        /// Path that failed to read.
        path: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// File exceeds the maximum allowed size.
    #[error("calibration file too large ({size} bytes, max {max})")]
    TooLarge {
        /// Actual file size in bytes.
        size: u64,
        /// Maximum allowed size.
        max: u64,
    },
    /// JSON parse error.
    #[error("invalid calibration JSON: {0}")]
    Parse(#[from] serde_json::Error),
    /// Calibration values are invalid.
    #[error("calibration validation failed: {0}")]
    Invalid(#[from] CalibrationError),
}

/// Validates a single camera's intrinsic parameters.
fn validate_camera_params(p: &CameraParams, camera: &'static str) -> Result<(), CalibrationError> {
    // Dimensions: non-zero and within the safe GPU allocation limit
    if p.width == 0 {
        return Err(CalibrationError::ZeroDimension {
            camera,
            field: "width",
            value: p.width,
        });
    }
    if p.height == 0 {
        return Err(CalibrationError::ZeroDimension {
            camera,
            field: "height",
            value: p.height,
        });
    }
    if p.width > MAX_DIM {
        return Err(CalibrationError::DimensionTooLarge {
            camera,
            field: "width",
            value: p.width,
            max: MAX_DIM,
        });
    }
    if p.height > MAX_DIM {
        return Err(CalibrationError::DimensionTooLarge {
            camera,
            field: "height",
            value: p.height,
            max: MAX_DIM,
        });
    }

    // Focal lengths: finite and large enough to avoid division-by-zero
    for (name, val) in [
        (
            if camera == "left" {
                "left.fx"
            } else {
                "right.fx"
            },
            p.fx,
        ),
        (
            if camera == "left" {
                "left.fy"
            } else {
                "right.fy"
            },
            p.fy,
        ),
    ] {
        if !val.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: name.to_owned(),
                value: format!("{val}"),
            });
        }
        if val <= EPSILON {
            return Err(CalibrationError::FocalLengthTooSmall {
                field: name,
                value: val,
                epsilon: EPSILON,
            });
        }
    }

    // Principal point: must be finite (zero is acceptable)
    for (name, val) in [
        (
            if camera == "left" {
                "left.cx"
            } else {
                "right.cx"
            },
            p.cx,
        ),
        (
            if camera == "left" {
                "left.cy"
            } else {
                "right.cy"
            },
            p.cy,
        ),
    ] {
        if !val.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: name.to_owned(),
                value: format!("{val}"),
            });
        }
    }

    // Distortion coefficients: must all be finite
    let d_prefix = if camera == "left" { "left" } else { "right" };
    for (i, coeff) in p.d.iter().enumerate() {
        if !coeff.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: format!("{d_prefix}.d[{i}]"),
                value: format!("{coeff}"),
            });
        }
    }

    Ok(())
}

/// Validates the plane layout parameters.
fn validate_layout(l: &PlaneLayout) -> Result<(), CalibrationError> {
    // camera_axis_offset: must be finite and large enough to avoid zero-vector normalisation
    if !l.camera_axis_offset.is_finite() {
        return Err(CalibrationError::NonFiniteFloat {
            field: "params.cameraAxisOffset".to_owned(),
            value: format!("{}", l.camera_axis_offset),
        });
    }
    if l.camera_axis_offset <= EPSILON {
        return Err(CalibrationError::AxisOffsetTooSmall {
            value: l.camera_axis_offset,
            epsilon: EPSILON,
        });
    }

    // intersect: must be finite and within [0.0, 1.0]
    if !l.intersect.is_finite() {
        return Err(CalibrationError::NonFiniteFloat {
            field: "params.intersect".to_owned(),
            value: format!("{}", l.intersect),
        });
    }
    if !(0.0..=1.0).contains(&l.intersect) {
        return Err(CalibrationError::IntersectOutOfRange { value: l.intersect });
    }

    // Remaining float fields: finite check only (no magnitude constraint)
    for (name, val) in [
        ("params.xTy", l.x_ty),
        ("params.xRz", l.x_rz),
        ("params.xRx", l.x_rx),
        ("params.zRx", l.z_rx),
        ("params.zRz", l.z_rz),
    ] {
        if !val.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: name.to_owned(),
                value: format!("{val}"),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_v1_calibration_json() {
        let json = r#"{
            "left_uniforms": {
                "width": 3840, "height": 2160,
                "fx": 1796.32, "fy": 1797.22,
                "cx": 1919.37, "cy": 1063.17,
                "d": [0.0342, 0.0677, -0.0741, 0.0299]
            },
            "right_uniforms": {
                "width": 3840, "height": 2160,
                "fx": 1796.32, "fy": 1797.22,
                "cx": 1919.37, "cy": 1063.17,
                "d": [0.0342, 0.0677, -0.0741, 0.0299]
            },
            "params": {
                "cameraAxisOffset": 0.2398,
                "intersect": 0.5446,
                "xTy": 0.00476,
                "xRz": 0.00753,
                "zRx": -0.00431
            }
        }"#;

        let cal: MatchCalibration = serde_json::from_str(json).unwrap();

        assert_eq!(cal.left.width, 3840);
        assert_eq!(cal.left.d.len(), 4);
        assert!((cal.layout.camera_axis_offset - 0.2398).abs() < 1e-4);
        assert!((cal.layout.intersect - 0.5446).abs() < 1e-4);
    }

    // --- validation tests ---

    fn valid_cal() -> MatchCalibration {
        MatchCalibration {
            left: CameraParams {
                width: 3840,
                height: 2160,
                fx: 1796.32,
                fy: 1797.22,
                cx: 1919.37,
                cy: 1063.17,
                d: [0.0342, 0.0677, -0.0741, 0.0299],
            },
            right: CameraParams {
                width: 3840,
                height: 2160,
                fx: 1796.32,
                fy: 1797.22,
                cx: 1919.37,
                cy: 1063.17,
                d: [0.0342, 0.0677, -0.0741, 0.0299],
            },
            layout: PlaneLayout {
                camera_axis_offset: 0.2398,
                intersect: 0.5446,
                x_ty: 0.00476,
                x_rz: 0.00753,
                z_rx: -0.00431,
                x_rx: 0.0,
                z_rz: 0.0,
            },
            rig_tilt: 0.0,
            sync_offset: 0,
        }
    }

    #[test]
    fn validate_valid_calibration_passes() {
        assert!(valid_cal().validate().is_ok());
    }

    #[test]
    fn validate_zero_fx_fails() {
        let mut cal = valid_cal();
        cal.left.fx = 0.0;
        let err = cal.validate().unwrap_err();
        assert!(
            matches!(err, CalibrationError::FocalLengthTooSmall { .. }),
            "unexpected error variant: {err}"
        );
    }

    #[test]
    fn validate_negative_fx_fails() {
        let mut cal = valid_cal();
        cal.right.fy = -500.0;
        let err = cal.validate().unwrap_err();
        assert!(
            matches!(err, CalibrationError::FocalLengthTooSmall { .. }),
            "unexpected error variant: {err}"
        );
    }

    #[test]
    fn validate_nan_in_distortion_fails() {
        let mut cal = valid_cal();
        cal.left.d[2] = f64::NAN;
        let err = cal.validate().unwrap_err();
        assert!(
            matches!(err, CalibrationError::NonFiniteFloat { ref field, .. } if field.contains("left.d[2]")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_inf_in_distortion_fails() {
        let mut cal = valid_cal();
        cal.right.d[0] = f64::INFINITY;
        let err = cal.validate().unwrap_err();
        assert!(
            matches!(err, CalibrationError::NonFiniteFloat { .. }),
            "unexpected error variant: {err}"
        );
    }

    #[test]
    fn validate_intersect_above_one_fails() {
        let mut cal = valid_cal();
        cal.layout.intersect = 1.001;
        let err = cal.validate().unwrap_err();
        assert!(
            matches!(err, CalibrationError::IntersectOutOfRange { value } if (value - 1.001).abs() < 1e-9),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_intersect_negative_fails() {
        let mut cal = valid_cal();
        cal.layout.intersect = -0.1;
        let err = cal.validate().unwrap_err();
        assert!(
            matches!(err, CalibrationError::IntersectOutOfRange { .. }),
            "unexpected error variant: {err}"
        );
    }

    #[test]
    fn validate_zero_width_fails() {
        let mut cal = valid_cal();
        cal.right.width = 0;
        let err = cal.validate().unwrap_err();
        assert!(
            matches!(
                err,
                CalibrationError::ZeroDimension {
                    camera: "right",
                    field: "width",
                    ..
                }
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_dimension_too_large_fails() {
        let mut cal = valid_cal();
        cal.left.height = MAX_DIM + 1;
        let err = cal.validate().unwrap_err();
        assert!(
            matches!(
                err,
                CalibrationError::DimensionTooLarge {
                    camera: "left",
                    field: "height",
                    ..
                }
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_zero_camera_axis_offset_fails() {
        let mut cal = valid_cal();
        cal.layout.camera_axis_offset = 0.0;
        let err = cal.validate().unwrap_err();
        assert!(
            matches!(err, CalibrationError::AxisOffsetTooSmall { .. }),
            "unexpected error variant: {err}"
        );
    }

    #[test]
    fn validate_nan_camera_axis_offset_fails() {
        let mut cal = valid_cal();
        cal.layout.camera_axis_offset = f64::NAN;
        let err = cal.validate().unwrap_err();
        assert!(
            matches!(err, CalibrationError::NonFiniteFloat { ref field, .. } if field == "params.cameraAxisOffset"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn roundtrip_serialization() {
        let cal = MatchCalibration {
            left: CameraParams {
                width: 1920,
                height: 1080,
                fx: 1000.0,
                fy: 1000.0,
                cx: 960.0,
                cy: 540.0,
                d: [0.0, 0.0, 0.0, 0.0],
            },
            right: CameraParams {
                width: 1920,
                height: 1080,
                fx: 1000.0,
                fy: 1000.0,
                cx: 960.0,
                cy: 540.0,
                d: [0.0, 0.0, 0.0, 0.0],
            },
            layout: PlaneLayout {
                camera_axis_offset: 0.25,
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
            },
            rig_tilt: 0.3,
            sync_offset: 67,
        };

        let json = serde_json::to_string(&cal).unwrap();
        let parsed: MatchCalibration = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.left.width, cal.left.width);
        assert!((parsed.layout.intersect - cal.layout.intersect).abs() < f64::EPSILON);
    }
}
