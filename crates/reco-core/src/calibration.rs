//! The calibration document: the canonical, serializable source of truth.
//!
//! A [`Calibration`] is the persisted description of how N source feeds become
//! one stitched virtual-camera view. It is plain data: every runtime object
//! (the scene geometry, the virtual-camera basis, GPU pipeline/uniforms, the
//! CPU inverse map) is *derived* from it - the calibration is never built from
//! the runtime, and the derived objects are never serialized.
//!
//! It decomposes into three concerns, one per stitch stage:
//! - [`Lens`] (per source) - undistortion: intrinsics + distortion model + an
//!   `id` naming the lens *model* so a profile is reusable across cameras.
//! - [`Topology`] - 3D placement of the source planes plus the overlap seam.
//! - [`Framing`] - the virtual camera's calibrated coordinate frame; panning
//!   (yaw/pitch) and output framing (fov/size) are runtime, NOT stored here.
//!
//! The distortion model is `fisheye_kb4` (Kannala-Brandt 4-coefficient):
//! `θ_d = θ × (1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum allowed dimension (width or height) in pixels.
///
/// Values above this threshold indicate a malformed calibration and would
/// cause the GPU allocator to request an unreasonably large texture.
pub const MAX_DIM: u32 = 8192;

/// Minimum positive value accepted for focal lengths and the axis offset.
///
/// Values at or below this would cause division-by-zero or zero-vector
/// normalization in the stitching geometry.
const EPSILON: f64 = 1e-6;

/// Current calibration document schema version.
const SCHEMA_VERSION: u32 = 1;

fn default_schema_version() -> u32 {
    SCHEMA_VERSION
}

/// Errors produced by [`Calibration::validate`].
#[derive(Debug, Error)]
pub enum CalibrationError {
    /// A required dimension (width or height) is zero.
    #[error("lens[{index}] {field} must be > 0, got {value}")]
    ZeroDimension {
        /// Index of the offending lens.
        index: usize,
        /// Field name (`"width"` or `"height"`).
        field: &'static str,
        /// The offending value.
        value: u32,
    },

    /// A dimension exceeds [`MAX_DIM`] and would cause an excessive GPU allocation.
    #[error("lens[{index}] {field} exceeds the maximum of {max}, got {value}")]
    DimensionTooLarge {
        /// Index of the offending lens.
        index: usize,
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
        /// Dotted field path, e.g. `"lens[0].fx"` or `"framing.tilt"`.
        field: String,
        /// The string representation of the offending value.
        value: String,
    },

    /// A focal length is too small, which would cause division-by-zero.
    #[error("field '{field}' must be > {epsilon}, got {value}")]
    FocalLengthTooSmall {
        /// Field path.
        field: String,
        /// The offending value.
        value: f64,
        /// The minimum threshold.
        epsilon: f64,
    },

    /// `framing.axis_offset` is too small, which would cause zero-vector normalization.
    #[error("framing.axis_offset must be > {epsilon}, got {value}")]
    AxisOffsetTooSmall {
        /// The offending value.
        value: f64,
        /// The minimum threshold.
        epsilon: f64,
    },

    /// `topology.intersect` is outside the valid `[0.0, 1.0]` range.
    #[error("topology.intersect must be in [0.0, 1.0], got {value}")]
    IntersectOutOfRange {
        /// The offending value.
        value: f64,
    },

    /// The calibration has no lenses.
    #[error("calibration must have at least one lens")]
    NoLenses,

    /// The lens count does not match what the topology can render.
    ///
    /// The L-shape topology (the only one today) indexes exactly two
    /// lenses; any other count would panic at render time. This becomes
    /// topology-aware once projections carry their own arity.
    #[error("L-shape calibration needs exactly 2 lenses, got {found}")]
    ExpectedTwoLenses {
        /// Number of lenses actually present.
        found: usize,
    },

    /// `sync_offset` is outside a realistic range.
    ///
    /// Guards against pathological values (e.g. `i64::MIN`) that would hang the
    /// decode pairing loop by trying to skip an astronomical number of frames.
    #[error("sync_offset must be in [{min}, {max}] frames, got {value}")]
    SyncOffsetOutOfRange {
        /// The offending value.
        value: i64,
        /// The minimum allowed (negative).
        min: i64,
        /// The maximum allowed (positive).
        max: i64,
    },
}

/// Maximum realistic sync_offset in frames (~28 minutes at 60fps).
const MAX_SYNC_OFFSET_FRAMES: i64 = 100_000;

/// One source's optical model: intrinsics + KB4 distortion, plus an `id` that
/// names the lens *model* (e.g. `"gopro-h11-wide-4k"`).
///
/// The `id` identifies a reusable profile - two cameras of the same model share
/// the same `Lens` content (and `id`); a mixed rig has different ones. It is the
/// CPU/GPU-independent record both executors derive their runtime form from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lens {
    /// Lens-model identity (reusable profile key). Empty = unknown.
    #[serde(default)]
    pub id: String,
    /// Calibration frame width in pixels.
    pub width: u32,
    /// Calibration frame height in pixels.
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
    pub distortion: [f64; 4],
    /// How much of the distortion model to apply: `1.0` = full KB4,
    /// `0.0` = pinhole. A rendering choice, persisted per lens.
    #[serde(default = "default_correction")]
    pub correction: f32,
}

fn default_correction() -> f32 {
    1.0
}

impl Lens {
    /// A fisheye (KB4) lens at full correction, with no model id.
    pub fn fisheye(
        width: u32,
        height: u32,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        distortion: [f64; 4],
    ) -> Self {
        Self {
            id: String::new(),
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            distortion,
            correction: 1.0,
        }
    }
}

/// 3D placement of the source planes plus the overlap seam.
///
/// The geometry kind (L-shape today, cylinder/N-camera later) is dispatched by
/// the [`Projection`](crate::projection::Projection) trait; this carries its
/// parameters. The virtual-camera position lives in [`Framing`], not here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topology {
    /// Overlap ratio between the two planes (`0.0` none .. `1.0` full).
    /// Each plane is translated by `(plane_width / 2) × (1 - intersect)`.
    pub intersect: f64,
    /// Y-axis translation of the right plane (vertical misalignment).
    #[serde(default)]
    pub x_ty: f64,
    /// Z-axis rotation of the right plane, radians (roll).
    #[serde(default)]
    pub x_rz: f64,
    /// X-axis rotation of the left plane, radians (tilt).
    #[serde(default)]
    pub z_rx: f64,
    /// X-axis rotation of the right plane, radians (pitch).
    #[serde(default)]
    pub x_rx: f64,
    /// Z-axis rotation of the left plane, radians (pitch).
    #[serde(default)]
    pub z_rz: f64,
    /// Seam blend width as a fraction of the plane overlap. `0.0` = hard seam.
    #[serde(default = "default_blend_width")]
    pub blend_width: f32,
}

fn default_blend_width() -> f32 {
    0.05
}

/// The virtual camera's calibrated coordinate frame: the axis/orientation that
/// panning evolves *within*. Pan (yaw/pitch) and output framing (fov/size) are
/// runtime state and are deliberately NOT stored here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Framing {
    /// Virtual-camera distance from the origin along X and Z; the camera sits
    /// at `[axis_offset, 0, axis_offset]`.
    pub axis_offset: f64,
    /// Rig tilt in radians (forward lean), straightens vertical lines at edges.
    #[serde(default)]
    pub tilt: f64,
    /// Rig roll in radians (lateral lean).
    #[serde(default)]
    pub roll: f64,
}

/// Playing-field region of interest for per-camera detection filtering.
///
/// A detection concern (consumed by `reco-autocam`), kept here transitionally;
/// it will move out of the calibration when detection config is extracted.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FieldRoi {
    /// Polygon vertices for the left camera, normalized `[0,1]`.
    #[serde(default)]
    pub left: Vec<[f64; 2]>,
    /// Polygon vertices for the right camera, normalized `[0,1]`.
    #[serde(default)]
    pub right: Vec<[f64; 2]>,
}

/// The calibration document: canonical, serializable source of truth.
///
/// Everything the stitch needs to turn source frames into a panorama. Plain
/// data - the runtime objects are derived from it (see the module docs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    /// Document schema version, for clean future migrations.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Per-source optical models, one per camera (index 0 = left, 1 = right for
    /// the L-shape). `Vec` so N-camera rigs need no shape change.
    pub lenses: Vec<Lens>,
    /// 3D placement of the source planes + the seam.
    pub topology: Topology,
    /// The virtual camera's calibrated coordinate frame.
    pub framing: Framing,
    /// Temporal sync offset in frames (positive = right video ahead). Consumed
    /// by `reco-io` decode pairing; transitional, moves to the synchronizer.
    #[serde(default)]
    pub sync_offset: i64,
    /// Optional per-camera detection ROI. Transitional (a detection concern).
    #[serde(default)]
    pub field_roi: Option<FieldRoi>,
}

/// Maximum calibration file size (1 MB).
const MAX_CALIBRATION_FILE_SIZE: u64 = 1_048_576;

impl Calibration {
    /// Assemble a calibration from its parts (current schema version, no sync
    /// offset, no ROI).
    pub fn new(lenses: Vec<Lens>, topology: Topology, framing: Framing) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            lenses,
            topology,
            framing,
            sync_offset: 0,
            field_roi: None,
        }
    }

    /// Load and validate a calibration from a JSON file.
    pub fn from_file(path: &std::path::Path) -> Result<Self, CalibrationLoadError> {
        use std::io::Read;

        let file = std::fs::File::open(path).map_err(|e| CalibrationLoadError::Io {
            path: path.display().to_string(),
            source: e,
        })?;

        // Read up to MAX+1 bytes atomically to detect oversize without a TOCTOU race.
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

        let cal: Self = serde_json::from_str(&json).map_err(|e| {
            // Transitional: surface a clear message for old v1 "match"
            // files instead of a raw `missing field lenses`.
            if json.contains("\"left_uniforms\"") || json.contains("\"cameraAxisOffset\"") {
                CalibrationLoadError::LegacyV1
            } else {
                CalibrationLoadError::Parse(e)
            }
        })?;
        cal.validate()?;
        Ok(cal)
    }

    /// Save the calibration to a JSON file (pretty-printed).
    pub fn to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        std::fs::write(path, self.to_json_pretty())
    }

    /// Serialize to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("Calibration is always serializable")
    }

    /// Validate all parameters before they are used to build runtime geometry.
    ///
    /// Catches malformed values that would otherwise cause GPU hangs, shader
    /// division-by-zero, or excessive allocations. Returns the first error found.
    pub fn validate(&self) -> Result<(), CalibrationError> {
        if self.lenses.is_empty() {
            return Err(CalibrationError::NoLenses);
        }
        // The L-shape topology hard-indexes lenses[0] and lenses[1] at
        // render time; reject any other count here with a typed error
        // rather than panicking out-of-bounds downstream.
        if self.lenses.len() != 2 {
            return Err(CalibrationError::ExpectedTwoLenses {
                found: self.lenses.len(),
            });
        }
        for (i, lens) in self.lenses.iter().enumerate() {
            validate_lens(lens, i)?;
        }
        validate_topology(&self.topology)?;
        validate_framing(&self.framing)?;
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
    /// A legacy v1 "match" calibration was detected. No longer supported.
    ///
    /// Transitional: we sniff the old wire shape only to give a clear
    /// message instead of a raw `missing field lenses`. Remove a few
    /// releases after the v1 cutover.
    #[error(
        "legacy v1 calibration ('match' format) is no longer supported; \
         re-run `reco calibrate` to produce a current calibration file"
    )]
    LegacyV1,
    /// Calibration values are invalid.
    #[error(transparent)]
    Invalid(#[from] CalibrationError),
}

/// Validate one lens's intrinsics.
fn validate_lens(lens: &Lens, index: usize) -> Result<(), CalibrationError> {
    for (field, value) in [("width", lens.width), ("height", lens.height)] {
        if value == 0 {
            return Err(CalibrationError::ZeroDimension {
                index,
                field,
                value,
            });
        }
        if value > MAX_DIM {
            return Err(CalibrationError::DimensionTooLarge {
                index,
                field,
                value,
                max: MAX_DIM,
            });
        }
    }

    for (name, val) in [("fx", lens.fx), ("fy", lens.fy)] {
        if !val.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: format!("lens[{index}].{name}"),
                value: format!("{val}"),
            });
        }
        if val <= EPSILON {
            return Err(CalibrationError::FocalLengthTooSmall {
                field: format!("lens[{index}].{name}"),
                value: val,
                epsilon: EPSILON,
            });
        }
    }

    for (name, val) in [("cx", lens.cx), ("cy", lens.cy)] {
        if !val.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: format!("lens[{index}].{name}"),
                value: format!("{val}"),
            });
        }
    }

    for (i, coeff) in lens.distortion.iter().enumerate() {
        if !coeff.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: format!("lens[{index}].distortion[{i}]"),
                value: format!("{coeff}"),
            });
        }
    }

    if !lens.correction.is_finite() {
        return Err(CalibrationError::NonFiniteFloat {
            field: format!("lens[{index}].correction"),
            value: format!("{}", lens.correction),
        });
    }

    Ok(())
}

/// Validate the topology parameters.
fn validate_topology(t: &Topology) -> Result<(), CalibrationError> {
    if !t.intersect.is_finite() {
        return Err(CalibrationError::NonFiniteFloat {
            field: "topology.intersect".to_owned(),
            value: format!("{}", t.intersect),
        });
    }
    if !(0.0..=1.0).contains(&t.intersect) {
        return Err(CalibrationError::IntersectOutOfRange { value: t.intersect });
    }

    for (name, val) in [
        ("topology.x_ty", t.x_ty),
        ("topology.x_rz", t.x_rz),
        ("topology.x_rx", t.x_rx),
        ("topology.z_rx", t.z_rx),
        ("topology.z_rz", t.z_rz),
    ] {
        if !val.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: name.to_owned(),
                value: format!("{val}"),
            });
        }
    }

    if !t.blend_width.is_finite() {
        return Err(CalibrationError::NonFiniteFloat {
            field: "topology.blend_width".to_owned(),
            value: format!("{}", t.blend_width),
        });
    }

    Ok(())
}

/// Validate the framing parameters.
fn validate_framing(f: &Framing) -> Result<(), CalibrationError> {
    if !f.axis_offset.is_finite() {
        return Err(CalibrationError::NonFiniteFloat {
            field: "framing.axis_offset".to_owned(),
            value: format!("{}", f.axis_offset),
        });
    }
    if f.axis_offset <= EPSILON {
        return Err(CalibrationError::AxisOffsetTooSmall {
            value: f.axis_offset,
            epsilon: EPSILON,
        });
    }

    for (name, val) in [("framing.tilt", f.tilt), ("framing.roll", f.roll)] {
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

    fn sample_json() -> &'static str {
        r#"{
            "schema_version": 1,
            "lenses": [
                { "id": "test-cam", "width": 3840, "height": 2160,
                  "fx": 1796.32, "fy": 1797.22, "cx": 1919.37, "cy": 1063.17,
                  "distortion": [0.0342, 0.0677, -0.0741, 0.0299], "correction": 1.0 },
                { "id": "test-cam", "width": 3840, "height": 2160,
                  "fx": 1796.32, "fy": 1797.22, "cx": 1919.37, "cy": 1063.17,
                  "distortion": [0.0342, 0.0677, -0.0741, 0.0299], "correction": 1.0 }
            ],
            "topology": { "intersect": 0.5446, "x_ty": 0.00476, "x_rz": 0.00753,
                          "z_rx": -0.00431, "blend_width": 0.05 },
            "framing": { "axis_offset": 0.2398, "tilt": 0.0, "roll": 0.0 }
        }"#
    }

    #[test]
    fn parse_calibration_json() {
        let cal: Calibration = serde_json::from_str(sample_json()).unwrap();
        assert_eq!(cal.lenses.len(), 2);
        assert_eq!(cal.lenses[0].width, 3840);
        assert_eq!(cal.lenses[0].distortion.len(), 4);
        assert!((cal.framing.axis_offset - 0.2398).abs() < 1e-4);
        assert!((cal.topology.intersect - 0.5446).abs() < 1e-4);
        assert!(cal.field_roi.is_none());
        cal.validate().unwrap();
    }

    #[test]
    fn json_round_trips() {
        // Set every serde-defaulted field to a NON-default value so the
        // round-trip actually catches a dropped or renamed field (a
        // default-valued field survives even if serialization drops it).
        let mut cal = valid_cal();
        cal.lenses[0].correction = 0.0;
        cal.topology.blend_width = 0.123;
        cal.framing.tilt = 0.3;
        cal.framing.roll = -0.12;
        cal.sync_offset = 67;

        let json = cal.to_json_pretty();
        let back: Calibration = serde_json::from_str(&json).unwrap();
        assert_eq!(back.lenses.len(), cal.lenses.len());
        assert!((back.lenses[0].correction - 0.0).abs() < 1e-6);
        assert!((back.topology.blend_width - 0.123).abs() < 1e-6);
        assert!((back.topology.intersect - cal.topology.intersect).abs() < 1e-9);
        assert!((back.framing.axis_offset - cal.framing.axis_offset).abs() < 1e-9);
        assert!((back.framing.tilt - 0.3).abs() < 1e-9);
        assert!((back.framing.roll + 0.12).abs() < 1e-9);
        assert_eq!(back.sync_offset, 67);
    }

    #[test]
    fn parse_calibration_with_field_roi() {
        let mut cal: Calibration = serde_json::from_str(sample_json()).unwrap();
        cal.field_roi = Some(FieldRoi {
            left: vec![[0.49, 0.90], [0.33, 0.73], [0.42, 0.58]],
            right: vec![[0.63, 0.85], [0.78, 0.68], [0.55, 0.60]],
        });
        let json = cal.to_json_pretty();
        let back: Calibration = serde_json::from_str(&json).unwrap();
        let roi = back.field_roi.as_ref().unwrap();
        assert_eq!(roi.left.len(), 3);
        assert!((roi.right[1][1] - 0.68).abs() < 1e-6);
    }

    fn valid_cal() -> Calibration {
        let lens = || Lens {
            id: "test".to_string(),
            width: 1920,
            height: 1080,
            fx: 960.0,
            fy: 960.0,
            cx: 960.0,
            cy: 540.0,
            distortion: [-0.02, 0.004, 0.0, 0.0],
            correction: 1.0,
        };
        Calibration {
            schema_version: SCHEMA_VERSION,
            lenses: vec![lens(), lens()],
            topology: Topology {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            framing: Framing {
                axis_offset: 0.25,
                tilt: 0.0,
                roll: 0.0,
            },
            sync_offset: 0,
            field_roi: None,
        }
    }

    #[test]
    fn valid_calibration_passes() {
        valid_cal().validate().unwrap();
    }

    #[test]
    fn rejects_zero_dimension() {
        let mut c = valid_cal();
        c.lenses[0].width = 0;
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::ZeroDimension { .. })
        ));
    }

    #[test]
    fn rejects_oversized_dimension() {
        let mut c = valid_cal();
        c.lenses[1].height = MAX_DIM + 1;
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::DimensionTooLarge { .. })
        ));
    }

    #[test]
    fn rejects_tiny_focal_length() {
        let mut c = valid_cal();
        c.lenses[0].fx = 0.0;
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::FocalLengthTooSmall { .. })
        ));
    }

    #[test]
    fn rejects_nonfinite() {
        let mut c = valid_cal();
        c.lenses[0].cx = f64::NAN;
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::NonFiniteFloat { .. })
        ));
    }

    #[test]
    fn rejects_axis_offset_too_small() {
        let mut c = valid_cal();
        c.framing.axis_offset = 0.0;
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::AxisOffsetTooSmall { .. })
        ));
    }

    #[test]
    fn rejects_intersect_out_of_range() {
        let mut c = valid_cal();
        c.topology.intersect = 1.5;
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::IntersectOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_no_lenses() {
        let mut c = valid_cal();
        c.lenses.clear();
        assert!(matches!(c.validate(), Err(CalibrationError::NoLenses)));
    }

    #[test]
    fn rejects_one_lens_with_typed_error_not_panic() {
        let mut c = valid_cal();
        c.lenses.pop();
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::ExpectedTwoLenses { found: 1 })
        ));
    }

    #[test]
    fn rejects_three_lenses() {
        let mut c = valid_cal();
        let extra = c.lenses[0].clone();
        c.lenses.push(extra);
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::ExpectedTwoLenses { found: 3 })
        ));
    }

    #[test]
    fn legacy_v1_calibration_gives_clear_error() {
        let path = std::env::temp_dir().join(format!("reco_v1_{}.json", std::process::id()));
        std::fs::write(&path, r#"{"left_uniforms":{"width":100},"params":{}}"#).unwrap();
        let err = Calibration::from_file(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(err, CalibrationLoadError::LegacyV1),
            "expected LegacyV1, got {err:?}"
        );
    }

    #[test]
    fn rejects_pathological_sync_offset() {
        let mut c = valid_cal();
        c.sync_offset = i64::MIN;
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::SyncOffsetOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_sync_offset_just_past_cap() {
        let mut c = valid_cal();
        c.sync_offset = MAX_SYNC_OFFSET_FRAMES + 1;
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::SyncOffsetOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_sync_offset_i64_max() {
        let mut c = valid_cal();
        c.sync_offset = i64::MAX;
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::SyncOffsetOutOfRange { .. })
        ));
    }

    #[test]
    fn accepts_sync_offset_at_range_bounds() {
        let mut c = valid_cal();
        c.sync_offset = MAX_SYNC_OFFSET_FRAMES;
        c.validate().unwrap();
        c.sync_offset = -MAX_SYNC_OFFSET_FRAMES;
        c.validate().unwrap();
    }
}
