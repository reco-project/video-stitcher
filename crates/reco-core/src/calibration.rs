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
            },
        };

        let json = serde_json::to_string(&cal).unwrap();
        let parsed: MatchCalibration = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.left.width, cal.left.width);
        assert!((parsed.layout.intersect - cal.layout.intersect).abs() < f64::EPSILON);
    }
}
