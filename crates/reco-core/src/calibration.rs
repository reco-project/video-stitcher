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

    /// `sync_offset` is outside a realistic range.
    ///
    /// Guards against pathological values (e.g. `i64::MIN`) that would
    /// hang the decode pairing loop by trying to skip an astronomical
    /// number of frames. Realistic values are at most a few thousand
    /// frames (a few minutes at 60fps); this limit is deliberately
    /// generous.
    #[error("params.sync_offset must be in [{min}, {max}] frames, got {value}")]
    SyncOffsetOutOfRange {
        /// The offending value.
        value: i64,
        /// The minimum allowed (negative).
        min: i64,
        /// The maximum allowed (positive).
        max: i64,
    },
}

/// Maximum realistic sync_offset in frames.
///
/// Chosen to be comfortably larger than any plausible physical offset
/// (100000 frames is ~28 minutes at 60fps) while rejecting values that
/// would hang the decode pairing loop. See [`CalibrationError::SyncOffsetOutOfRange`].
const MAX_SYNC_OFFSET_FRAMES: i64 = 100_000;

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
///           └──────────┘ Right plane (X-Y)
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

/// Playing field region of interest for per-camera detection filtering.
///
/// Each camera has an optional polygon (normalized `[0,1]` coordinates)
/// defining the visible playing field boundary. Detections outside this
/// polygon are filtered before reaching the director, eliminating false
/// positives from stands, scoreboards, and other non-field areas.
///
/// # JSON Format
///
/// ```json
/// "field_roi": {
///     "left": [[0.49, 0.90], [0.33, 0.73], [0.42, 0.58]],
///     "right": [[0.63, 0.85], [0.78, 0.68], [0.55, 0.60]]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FieldRoi {
    /// Polygon vertices for the left camera, in normalized `[0,1]` coordinates.
    #[serde(default)]
    pub left: Vec<[f64; 2]>,
    /// Polygon vertices for the right camera, in normalized `[0,1]` coordinates.
    #[serde(default)]
    pub right: Vec<[f64; 2]>,
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

    /// Rig roll in radians (rotation around the camera axis).
    ///
    /// Corrects for a rig that is not level side-to-side.
    /// Defaults to 0.0 for backward compatibility.
    #[serde(default)]
    pub rig_roll: f64,

    /// Temporal sync offset in frames (positive = right video is ahead).
    ///
    /// Computed from IMU gyro or audio cross-correlation during calibration.
    /// Defaults to 0 for backward compatibility with older calibrations.
    #[serde(default)]
    pub sync_offset: i64,

    /// Optional playing field ROI polygons for per-camera detection filtering.
    ///
    /// When present, detections outside the polygon for their camera are
    /// discarded before reaching the director. This eliminates false positives
    /// from stands, scoreboards, and other non-field areas.
    #[serde(default)]
    pub field_roi: Option<FieldRoi>,

    /// Lens distortion-correction strength applied in the renderer.
    ///
    /// `1.0` = full correction (the modelled lens profile is applied),
    /// `0.0` = correction off (raw fisheye). Persisted so the GUI restores
    /// the user's choice on reload. Defaults to `1.0` for older
    /// calibrations that predate this field, matching prior behaviour.
    #[serde(default = "default_lens_correction_amount")]
    pub lens_correction_amount: f32,

    /// Seam blend width as a fraction of the panorama overlap.
    ///
    /// Controls how wide the feathered transition between the two cameras
    /// is. Persisted so a hand-tuned seam survives save/reload. Defaults to
    /// `0.05` for older calibrations that predate this field, matching the
    /// previous hard-coded renderer default.
    #[serde(default = "default_blend_width")]
    pub blend_width: f32,
}

/// Backward-compatible default for [`MatchCalibration::lens_correction_amount`]
/// when loading a calibration written before the field existed.
fn default_lens_correction_amount() -> f32 {
    1.0
}

/// Backward-compatible default for [`MatchCalibration::blend_width`] when
/// loading a calibration written before the field existed.
fn default_blend_width() -> f32 {
    0.05
}

/// Maximum calibration file size (1 MB) to prevent loading unreasonably large files.
const MAX_CALIBRATION_FILE_SIZE: u64 = 1_048_576;

impl MatchCalibration {
    /// Load and validate a calibration from a JSON file.
    ///
    /// Checks file size (max 1 MB), parses JSON, and runs
    /// [`validate`](Self::validate). Returns a descriptive error on any failure.
    pub fn from_file(path: &std::path::Path) -> Result<Self, CalibrationLoadError> {
        use std::io::Read;

        let file = std::fs::File::open(path).map_err(|e| CalibrationLoadError::Io {
            path: path.display().to_string(),
            source: e,
        })?;

        // Read up to MAX+1 bytes atomically. If we get more than MAX bytes,
        // the file is too large. This avoids a TOCTOU race between a separate
        // metadata size check and the actual read.
        let mut json = String::new();
        file.take(MAX_CALIBRATION_FILE_SIZE + 1)
            .read_to_string(&mut json)
            .map_err(|e| CalibrationLoadError::Io {
                path: path.display().to_string(),
                source: e,
            })?;

        if json.len() as u64 > MAX_CALIBRATION_FILE_SIZE {
            return Err(CalibrationLoadError::TooLarge {
                size: json.len() as u64,
                max: MAX_CALIBRATION_FILE_SIZE,
            });
        }

        let cal: Self = serde_json::from_str(&json).map_err(CalibrationLoadError::Parse)?;
        cal.validate()?;
        Ok(cal)
    }

    /// Save calibration to a JSON file.
    ///
    /// Uses pretty-printed JSON for human readability.
    pub fn to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let json = self.to_json_pretty();
        std::fs::write(path, json)
    }

    /// Serialize to pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("MatchCalibration is always serializable")
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
        if !self.rig_tilt.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: "rig_tilt".to_string(),
                value: self.rig_tilt.to_string(),
            });
        }
        if !self.rig_roll.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: "rig_roll".to_string(),
                value: self.rig_roll.to_string(),
            });
        }
        // blend_width and lens_correction_amount feed the GPU shader; a
        // NaN/inf slipping in from a hand-edited JSON would corrupt the
        // render. Range is clamped at the GUI setters and re-validated by
        // the viewport, so a finiteness gate here matches rig_tilt/roll.
        if !self.blend_width.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: "blend_width".to_string(),
                value: self.blend_width.to_string(),
            });
        }
        if !self.lens_correction_amount.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: "lens_correction_amount".to_string(),
                value: self.lens_correction_amount.to_string(),
            });
        }
        // B-10: reject pathological sync_offset (e.g. i64::MIN) that would
        // hang the decode pairing loop. Bounds-check directly instead of
        // `.abs()` because `i64::MIN.abs()` itself overflows and panics in
        // debug builds.
        if self.sync_offset < -MAX_SYNC_OFFSET_FRAMES || self.sync_offset > MAX_SYNC_OFFSET_FRAMES {
            return Err(CalibrationError::SyncOffsetOutOfRange {
                value: self.sync_offset,
                min: -MAX_SYNC_OFFSET_FRAMES,
                max: MAX_SYNC_OFFSET_FRAMES,
            });
        }
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
        // v1 JSON has no field_roi, so it should default to None.
        assert!(cal.field_roi.is_none());
    }

    #[test]
    fn parse_calibration_with_field_roi() {
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
            },
            "field_roi": {
                "left": [[0.49, 0.90], [0.33, 0.73], [0.42, 0.58]],
                "right": [[0.63, 0.85], [0.78, 0.68], [0.55, 0.60]]
            }
        }"#;

        let cal: MatchCalibration = serde_json::from_str(json).unwrap();
        let roi = cal.field_roi.as_ref().unwrap();
        assert_eq!(roi.left.len(), 3);
        assert_eq!(roi.right.len(), 3);
        assert!((roi.left[0][0] - 0.49).abs() < 1e-6);
        assert!((roi.right[1][1] - 0.68).abs() < 1e-6);
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
            rig_roll: 0.0,
            sync_offset: 0,
            field_roi: None,
            lens_correction_amount: 1.0,
            blend_width: 0.05,
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
    fn validate_nan_blend_width_fails() {
        let mut cal = valid_cal();
        cal.blend_width = f32::NAN;
        let err = cal.validate().unwrap_err();
        assert!(
            matches!(err, CalibrationError::NonFiniteFloat { ref field, .. } if field == "blend_width"),
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
            field_roi: None,
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
            rig_roll: -0.12,
            sync_offset: 67,
            // Deliberately non-default (correction off, wide seam) so a
            // dropped field would change the round-tripped value.
            lens_correction_amount: 0.0,
            blend_width: 0.123,
        };

        let json = serde_json::to_string(&cal).unwrap();
        let parsed: MatchCalibration = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.left.width, cal.left.width);
        assert!((parsed.layout.intersect - cal.layout.intersect).abs() < f64::EPSILON);
        assert!((parsed.rig_tilt - cal.rig_tilt).abs() < f64::EPSILON);
        assert!((parsed.rig_roll - cal.rig_roll).abs() < f64::EPSILON);
        assert_eq!(parsed.sync_offset, cal.sync_offset);
        assert!((parsed.lens_correction_amount - cal.lens_correction_amount).abs() < f32::EPSILON);
        assert!((parsed.blend_width - cal.blend_width).abs() < f32::EPSILON);
    }

    #[test]
    fn old_calibration_json_without_new_fields_uses_safe_defaults() {
        // A calibration written before lens_correction_amount/blend_width
        // existed must load with the prior behaviour: full correction (1.0)
        // and the 0.05 seam, not 0.0/0.0.
        let mut value = serde_json::to_value(valid_cal()).unwrap();
        let obj = value.as_object_mut().unwrap();
        obj.remove("lens_correction_amount");
        obj.remove("blend_width");

        let parsed: MatchCalibration = serde_json::from_value(value).unwrap();
        assert!((parsed.lens_correction_amount - 1.0).abs() < f32::EPSILON);
        assert!((parsed.blend_width - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn validate_rejects_sync_offset_i64_min() {
        // B-10: i64::MIN used to slip through; the validator must reject it.
        // Do NOT use `.abs()` in the implementation because i64::MIN.abs()
        // itself panics on overflow - test proves the bounds-check path works.
        let mut cal = valid_cal();
        cal.sync_offset = i64::MIN;
        let err = cal.validate().expect_err("i64::MIN must be rejected");
        assert!(matches!(err, CalibrationError::SyncOffsetOutOfRange { .. }));
    }

    #[test]
    fn validate_rejects_sync_offset_i64_max() {
        let mut cal = valid_cal();
        cal.sync_offset = i64::MAX;
        let err = cal.validate().expect_err("i64::MAX must be rejected");
        assert!(matches!(err, CalibrationError::SyncOffsetOutOfRange { .. }));
    }

    #[test]
    fn validate_accepts_plausible_sync_offsets() {
        for v in [-10_000_i64, -60, 0, 60, 10_000] {
            let mut cal = valid_cal();
            cal.sync_offset = v;
            cal.validate()
                .unwrap_or_else(|e| panic!("sync_offset={v} must be accepted: {e:?}"));
        }
    }

    #[test]
    fn validate_rejects_sync_offset_just_past_cap() {
        let mut cal = valid_cal();
        cal.sync_offset = 100_001;
        let err = cal.validate().unwrap_err();
        assert!(matches!(err, CalibrationError::SyncOffsetOutOfRange { .. }));
    }
}
