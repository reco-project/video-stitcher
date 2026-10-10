//! The calibration document: the canonical, serializable source of truth.
//!
//! A [`Calibration`] is the persisted description of how N source feeds become
//! one stitched virtual-camera view. It is plain data: every runtime object
//! (the scene geometry, the virtual-camera basis, GPU pipeline/uniforms, the
//! CPU inverse map) is *derived* from it - the calibration is never built from
//! the runtime, and the derived objects are never serialized.
//!
//! It decomposes into three concerns, one per stitch stage:
//! - [`Lens`] (per source) - undistortion: intrinsics + distortion model.
//! - [`Projection`] - 3D placement of the source planes plus the overlap seam.
//! - [`Framing`] - the virtual camera's calibrated coordinate frame; panning
//!   (yaw/pitch) and output framing (fov/size) are runtime, NOT stored here.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::projection::{self, ProjectionGeometry};

/// Maximum allowed dimension (width or height) in pixels.
///
/// Values above this threshold indicate a malformed calibration and would
/// cause the GPU allocator to request an unreasonably large texture.
pub const MAX_DIM: u32 = 8192;

/// Current calibration document schema version.
const SCHEMA_VERSION: u32 = 2;

fn default_schema_version() -> u32 {
    SCHEMA_VERSION
}

/// Errors produced by [`Calibration::validate`].
#[derive(Debug, Clone, Error)]
pub enum CalibrationError {
    /// Field observations do not define a finite convex region under this calibration.
    #[error("invalid field region: {0}")]
    InvalidFieldRegion(String),
    /// The document declares a schema version this build does not understand.
    #[error(
        "unsupported calibration schema version {found}; this reco requires version {SCHEMA_VERSION}"
    )]
    UnsupportedSchemaVersion {
        /// The declared document version.
        found: u32,
    },
    /// A value is outside its documented range.
    #[error("{field} must be in [{min}, {max}], got {value}")]
    OutOfRange {
        /// Field path, e.g. `lens[0].correction`.
        field: String,
        /// The offending value.
        value: f64,
        /// Inclusive lower bound.
        min: f64,
        /// Inclusive upper bound.
        max: f64,
    },
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

    /// A value is too small for the math downstream, which divides by
    /// it or normalizes a vector built from it.
    #[error("field '{field}' must be > {epsilon}, got {value}")]
    ValueTooSmall {
        /// Field path, e.g. `lens[0].fx`.
        field: String,
        /// The offending value.
        value: f64,
        /// The minimum threshold.
        epsilon: f64,
    },

    /// The calibration has no lenses.
    #[error("calibration must have at least one lens")]
    NoLenses,

    /// The lens count does not match what the projection consumes.
    ///
    /// Projections index their lenses at render time; any other count
    /// would panic out-of-bounds downstream.
    #[error("{projection} calibration needs exactly {expected} lens(es), got {found}")]
    LensCountMismatch {
        /// Projection name for the message.
        projection: &'static str,
        /// Lens count the projection consumes.
        expected: usize,
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

/// Reject a non-finite (NaN or infinite) float.
pub(crate) fn expect_finite(field: &str, value: f64) -> Result<(), CalibrationError> {
    if !value.is_finite() {
        return Err(CalibrationError::NonFiniteFloat {
            field: field.to_owned(),
            value: format!("{value}"),
        });
    }
    Ok(())
}

/// Reject a value outside the inclusive `[min, max]` range.
pub(crate) fn expect_in_range(
    field: &str,
    value: f64,
    min: f64,
    max: f64,
) -> Result<(), CalibrationError> {
    if !(min..=max).contains(&value) {
        return Err(CalibrationError::OutOfRange {
            field: field.to_owned(),
            value,
            min,
            max,
        });
    }
    Ok(())
}

/// Minimum positive value accepted for focal lengths and camera
/// offsets - anything the geometry divides by or normalizes with.
///
/// Too loose and a near-zero value reaches the geometry, dividing by
/// ~zero or normalizing a ~zero vector into NaN poses. Too tight and
/// legitimate hand-edited calibrations start failing validation for
/// values that render fine.
const VALIDATION_EPSILON: f64 = 1e-6;

/// Reject a value that is not meaningfully positive: at or below
/// [`VALIDATION_EPSILON`]. For any field the math downstream divides
/// by, normalizes with, or that is nonsensical at ~zero. The strict
/// bound is the point: "equal to the threshold" is as degenerate as
/// "below it". Owning the epsilon here keeps the tolerance policy out
/// of projection code.
pub(crate) fn expect_positive(field: &str, value: f64) -> Result<(), CalibrationError> {
    if value <= VALIDATION_EPSILON {
        return Err(CalibrationError::ValueTooSmall {
            field: field.to_owned(),
            value,
            epsilon: VALIDATION_EPSILON,
        });
    }
    Ok(())
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
        let field = format!("lens[{index}].{name}");
        expect_finite(&field, val)?;
        expect_positive(&field, val)?;
    }

    for (name, val) in [("cx", lens.cx), ("cy", lens.cy)] {
        expect_finite(&format!("lens[{index}].{name}"), val)?;
    }

    for (i, coeff) in lens.distortion.iter().enumerate() {
        expect_finite(&format!("lens[{index}].distortion[{i}]"), *coeff)?;
    }

    let correction = format!("lens[{index}].correction");
    expect_finite(&correction, f64::from(lens.correction))?;
    // The shader interprets negative correction as its raw-bypass debug
    // mode and the CPU path would extrapolate the KB4 lerp - reject
    // anything outside the documented [0, 1] blend range.
    expect_in_range(&correction, f64::from(lens.correction), 0.0, 1.0)?;

    Ok(())
}

/// Validate the projection-independent framing parameters. Rules a
/// projection imposes on the framing (the L-shape's minimum axis offset)
/// live in that projection's own validate.
fn validate_framing(f: &Framing) -> Result<(), CalibrationError> {
    expect_finite("framing.axis_offset", f.axis_offset)?;
    expect_finite("framing.tilt", f.tilt)?;
    expect_finite("framing.roll", f.roll)?;
    Ok(())
}

/// One source's optical model: intrinsics + KB4 distortion.
///
/// The distortion model is `fisheye_kb4` (Kannala-Brandt 4-coefficient):
/// `θ_d = θ × (1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)`.
///
/// It is the CPU/GPU-independent record both executors derive their runtime
/// form from. Two cameras of the same model share the same `Lens` content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lens {
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
    /// How much of the distortion polynomial to apply: `1.0` = full KB4, `0.0` = equidistant (zero polynomial, not pinhole). A rendering choice, persisted per lens.
    #[serde(default = "default_correction")]
    pub correction: f32,
}

fn default_correction() -> f32 {
    1.0
}

impl Lens {
    /// Validate a standalone lens's reference dimensions and optical parameters.
    ///
    /// Errors identify this standalone lens as `lens[0]`. A complete [`Calibration`] validates each lens with its actual camera index.
    pub fn validate(&self) -> Result<(), CalibrationError> {
        validate_lens(self, 0)
    }

    /// Rescale intrinsics for a resized image, independently along each axis.
    ///
    /// Distortion and correction are unchanged. This describes resizing, not cropping. Zero source dimensions or invalid resulting lens parameters return an error; the source is never modified. The source profile may exceed [`MAX_DIM`], but the resized lens must fit the calibration limits.
    ///
    /// This is not a geometry-preserving transformation of an entire [`Calibration`]: projections can use the reference image dimensions to size their surfaces. For decoder resizing, retain the calibration and supply the actual source sizes to the executor instead, as exercised by `tests/input_dimensions.rs`.
    pub fn rescaled(&self, width: u32, height: u32) -> Result<Self, CalibrationError> {
        for (field, value) in [("width", self.width), ("height", self.height)] {
            if value == 0 {
                return Err(CalibrationError::ZeroDimension {
                    index: 0,
                    field,
                    value,
                });
            }
        }
        let scale_x = f64::from(width) / f64::from(self.width);
        let scale_y = f64::from(height) / f64::from(self.height);
        let lens = Self {
            width,
            height,
            fx: self.fx * scale_x,
            fy: self.fy * scale_y,
            cx: self.cx * scale_x,
            cy: self.cy * scale_y,
            ..self.clone()
        };
        lens.validate()?;
        Ok(lens)
    }

    /// A fisheye (KB4) lens at full correction.
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

    /// A distortion-free source: a pre-stitched flat panorama frame.
    /// Only the dimensions are load-bearing (cylinder sampling never
    /// undistorts); the intrinsics are centered identity values.
    ///
    /// The programmatic route to a mono lens - consumers building a
    /// cylinder [`Calibration`] in code use this; JSON documents spell
    /// the fields out. Exercised by the mono session and projection
    /// tests.
    pub fn flat(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            fx: width as f64 * 0.5,
            fy: width as f64 * 0.5,
            cx: width as f64 * 0.5,
            cy: height as f64 * 0.5,
            distortion: [0.0; 4],
            correction: 1.0,
        }
    }

    /// Aspect ratio of this lens's calibration frame (width / height).
    ///
    /// Returns 1.0 if height is zero (degenerate, rejected by
    /// validation) - mirrors `ViewportSize::aspect_ratio`.
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            return 1.0;
        }
        self.width as f32 / self.height as f32
    }
}

/// Scene geometry parameters: which shape the sources are painted on.
///
/// Serialized with a mandatory `type` tag (`"l-shape"` / `"cylinder"`).
/// Each variant stores the shape-specific parameters and supplies its geometry implementation. The virtual camera is derived from this projection and [`Framing`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Projection {
    /// Two fisheye cameras on perpendicular planes (the stereo rig).
    LShape(projection::LShape),
    /// One pre-stitched panorama painted on the inside of a cylinder.
    Cylinder(projection::Cylinder),
}

impl Projection {
    /// L-shape parameters, when this is the L-shape projection.
    pub fn l_shape(&self) -> Option<&projection::LShape> {
        match self {
            Projection::LShape(t) => Some(t),
            Projection::Cylinder(_) => None,
        }
    }

    /// Mutable [`Self::l_shape`].
    pub fn l_shape_mut(&mut self) -> Option<&mut projection::LShape> {
        match self {
            Projection::LShape(t) => Some(t),
            Projection::Cylinder(_) => None,
        }
    }

    /// Cylinder parameters, when this is the cylinder projection.
    pub fn cylinder(&self) -> Option<&projection::Cylinder> {
        match self {
            Projection::LShape(_) => None,
            Projection::Cylinder(t) => Some(t),
        }
    }

    /// Borrow the selected shape's geometry implementation without copying its parameters.
    pub(crate) fn geometry(&self) -> &dyn ProjectionGeometry {
        match self {
            Projection::LShape(t) => t,
            Projection::Cylinder(t) => t,
        }
    }

    /// Number of source cameras this projection consumes.
    pub fn camera_count(&self) -> usize {
        self.geometry().camera_count()
    }

    /// Seam blend width. The cylinder has a single surface and no seam,
    /// so it reports `0.0`.
    pub fn blend_width(&self) -> f32 {
        match self {
            Projection::LShape(t) => t.blend_width,
            Projection::Cylinder(_) => 0.0,
        }
    }
}

impl From<projection::LShape> for Projection {
    fn from(t: projection::LShape) -> Self {
        Projection::LShape(t)
    }
}

impl From<projection::Cylinder> for Projection {
    fn from(t: projection::Cylinder) -> Self {
        Projection::Cylinder(t)
    }
}

/// The virtual camera's calibrated coordinate frame: the axis/orientation that
/// panning evolves *within*. Pan (yaw/pitch) and output framing (fov/size) are
/// runtime state and are deliberately NOT stored here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

/// Four field-boundary observations in perimeter order, clockwise or counterclockwise.
///
/// Corners stay attached to their source images. Calibration derives the common region and its curved source-image edges.
/// This is image support for tracking, not a metric ground-plane reconstruction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldRoi {
    /// Exactly four source observations around the boundary. No independently editable per-camera polygons.
    pub corners: [crate::projection::SourcePoint; 4],
}

/// The calibration document: canonical, serializable source of truth.
///
/// Everything the stitch needs to turn source frames into a panorama. Plain
/// data - the runtime objects are derived from it (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    /// Document schema version. [`Self::validate`] rejects versions this build does not understand.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Per-source optical models, one per camera (index 0 = left, 1 = right for
    /// the L-shape). `Vec` so N-camera rigs need no shape change.
    pub lenses: Vec<Lens>,
    /// Source projection model and its shape-specific parameters.
    pub projection: Projection,
    /// The virtual camera's calibrated coordinate frame.
    pub framing: Framing,
    /// Temporal sync offset in frames (positive = right video ahead), consumed by `reco-io` decode pairing.
    #[serde(default)]
    pub sync_offset: i64,
    /// Optional four-corner detection region, interpreted through this calibration.
    #[serde(default)]
    pub field_roi: Option<FieldRoi>,
}

/// Maximum calibration file size (1 MB).
const MAX_CALIBRATION_FILE_SIZE: u64 = 1_048_576;

impl Calibration {
    /// Assemble a calibration from its parts (current schema version, no sync
    /// offset, no ROI).
    pub fn new(lenses: Vec<Lens>, projection: impl Into<Projection>, framing: Framing) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            lenses,
            projection: projection.into(),
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

        let cal: Self = serde_json::from_str(&json)?;
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
        if self.schema_version != SCHEMA_VERSION {
            return Err(CalibrationError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        if self.lenses.is_empty() {
            return Err(CalibrationError::NoLenses);
        }
        // Projections hard-index their lenses at render time; reject a
        // mismatched count here with a typed error rather than
        // panicking out-of-bounds downstream.
        if self.lenses.len() != self.projection.camera_count() {
            return Err(CalibrationError::LensCountMismatch {
                projection: self.projection.geometry().name(),
                expected: self.projection.camera_count(),
                found: self.lenses.len(),
            });
        }
        for (i, lens) in self.lenses.iter().enumerate() {
            validate_lens(lens, i)?;
        }
        // Each projection validates its own parameters (and its
        // requirements on the shared framing).
        match &self.projection {
            Projection::LShape(t) => t.validate(&self.framing)?,
            Projection::Cylinder(t) => t.validate()?,
        }
        validate_framing(&self.framing)?;
        self.field_boundary()?;
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
    #[error(transparent)]
    Invalid(#[from] CalibrationError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_document_round_trips() {
        let calibration: Calibration = serde_json::from_str(sample_json()).unwrap();
        let value = serde_json::to_value(&calibration).unwrap();
        assert_eq!(value["projection"]["type"], "l-shape");
        let decoded: Calibration = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, calibration);
    }

    #[test]
    fn cylinder_document_round_trips_nondefault_parameters() {
        let value = serde_json::json!({"type": "cylinder", "radius": 1357.0, "arc_deg": 245.0, "surface_height": 997.0});
        let projection: Projection = serde_json::from_value(value.clone()).unwrap();
        let cylinder = projection.cylinder().unwrap();
        assert_eq!(cylinder.radius, 1357.0);
        assert_eq!(cylinder.arc_deg, 245.0);
        assert_eq!(cylinder.surface_height, Some(997.0));
        assert_eq!(serde_json::to_value(projection).unwrap(), value);
    }

    #[test]
    fn cylinder_rejects_unknown_keys_instead_of_defaulting() {
        for key in ["focal_length", "sweep_deg", "video_height", "raduis"] {
            let mut value = serde_json::json!({"type": "cylinder"});
            value[key] = serde_json::json!(997.0);
            let err = serde_json::from_value::<Projection>(value).unwrap_err();
            assert!(err.to_string().contains(key), "{err}");
        }
        let default: Projection =
            serde_json::from_value(serde_json::json!({"type": "cylinder"})).unwrap();
        assert_eq!(
            default.cylinder().unwrap(),
            &projection::Cylinder::default()
        );
    }

    #[test]
    fn calibration_rejects_obsolete_topology_even_with_projection_present() {
        let mut value = serde_json::to_value(valid_cal()).unwrap();
        value["topology"] = value["projection"].clone();
        let err = serde_json::from_value::<Calibration>(value).unwrap_err();
        assert!(err.to_string().contains("topology"), "{err}");
    }

    #[test]
    fn calibration_rejects_unknown_nested_fields() {
        let mut calibration = valid_cal();
        calibration.field_roi = Some(FieldRoi {
            corners: [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]].map(|position| {
                crate::projection::SourcePoint {
                    camera: 0,
                    position,
                }
            }),
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("calibration.json");
        for (object, field, misspelled) in [
            ("/lenses/0", "correction", "corection"),
            ("/projection", "blend_width", "blend_wdith"),
            ("/framing", "roll", "rol"),
            ("/field_roi", "corners", "coners"),
        ] {
            let mut value = serde_json::to_value(&calibration).unwrap();
            let fields = value.pointer_mut(object).unwrap().as_object_mut().unwrap();
            let parameter = fields.remove(field).unwrap();
            fields.insert(misspelled.into(), parameter);
            std::fs::write(&path, serde_json::to_string(&value).unwrap()).unwrap();
            let err = Calibration::from_file(&path).unwrap_err();
            assert!(err.to_string().contains(misspelled), "{object}: {err}");
            let err = serde_json::from_value::<Calibration>(value).unwrap_err();
            assert!(err.to_string().contains(misspelled), "{object}: {err}");
        }
    }

    #[test]
    fn calibration_file_rejects_duplicate_fields() {
        let mut calibration = valid_cal();
        calibration.field_roi = Some(FieldRoi {
            corners: [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]].map(|position| {
                crate::projection::SourcePoint {
                    camera: 0,
                    position,
                }
            }),
        });
        let corners =
            serde_json::to_string(&calibration.field_roi.as_ref().unwrap().corners).unwrap();
        let json = serde_json::to_string(&calibration).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("calibration.json");
        for (field, first_value) in [
            ("schema_version", "0"),
            ("correction", "0"),
            ("type", "\"cylinder\""),
            ("intersect", "0"),
            ("roll", "0"),
            ("corners", corners.as_str()),
        ] {
            let duplicate = json.replacen(
                &format!("\"{field}\":"),
                &format!("\"{field}\":{first_value},\"{field}\":"),
                1,
            );
            std::fs::write(&path, duplicate).unwrap();
            let err = Calibration::from_file(&path).unwrap_err();
            assert!(
                err.to_string().contains("duplicate field"),
                "{field}: {err}"
            );
            assert!(err.to_string().contains(field), "{field}: {err}");
        }
    }

    #[test]
    fn unsupported_schema_is_rejected_without_file_loading() {
        for version in [0, 1, SCHEMA_VERSION + 1, u32::MAX] {
            let mut calibration = valid_cal();
            calibration.schema_version = version;
            assert!(matches!(
                calibration.validate(),
                Err(CalibrationError::UnsupportedSchemaVersion { found }) if found == version
            ));
            let source_sizes = vec![(64, 32); calibration.lenses.len()];
            assert!(
                crate::stitch::CpuExecutor::new(
                    calibration,
                    crate::render::viewport::ViewportSize {
                        width: 64,
                        height: 32
                    },
                    source_sizes,
                    false,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn older_schema_version_is_rejected() {
        let mut cal = valid_cal();
        cal.schema_version = 1;
        let path =
            std::env::temp_dir().join(format!("reco-old-schema-{}.json", std::process::id()));
        std::fs::write(&path, cal.to_json_pretty()).unwrap();
        let err = Calibration::from_file(&path).unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert!(matches!(
            err,
            CalibrationLoadError::Invalid(CalibrationError::UnsupportedSchemaVersion { found: 1 })
        ));
        assert!(!err.to_string().contains("newer"));
    }

    #[test]
    fn lens_rescaling_preserves_optics_and_scales_each_image_axis() {
        let mut lens = Lens::fisheye(3840, 2160, 1800.0, 1820.0, 1910.0, 1070.0, [0.03; 4]);
        lens.correction = 0.4;
        let scaled = lens.rescaled(1920, 540).unwrap();
        assert_eq!((scaled.width, scaled.height), (1920, 540));
        assert_eq!((scaled.fx, scaled.fy), (900.0, 455.0));
        assert_eq!((scaled.cx, scaled.cy), (955.0, 267.5));
        assert_eq!(scaled.distortion, lens.distortion);
        assert_eq!(scaled.correction, lens.correction);
        assert_eq!(scaled.rescaled(3840, 2160).unwrap(), lens);
        assert_eq!(lens.rescaled(3840, 2160).unwrap(), lens);
        let large_profile = Lens::flat(MAX_DIM * 2, MAX_DIM);
        assert_eq!(large_profile.rescaled(64, 32).unwrap(), Lens::flat(64, 32));
    }

    #[test]
    fn lens_rescaling_rejects_invalid_source_and_result() {
        let lens = Lens::flat(64, 36);
        for (width, height) in [(0, 36), (64, 0), (MAX_DIM + 1, 36), (64, MAX_DIM + 1)] {
            assert!(lens.rescaled(width, height).is_err());
        }
        let mut invalid = lens.clone();
        invalid.width = 0;
        assert!(invalid.rescaled(64, 36).is_err());
        invalid = lens.clone();
        invalid.fx = f64::NAN;
        assert!(invalid.rescaled(64, 36).is_err());
        invalid = lens.clone();
        invalid.distortion[0] = f64::INFINITY;
        assert!(invalid.rescaled(64, 36).is_err());
        invalid = lens.clone();
        invalid.correction = -1.0;
        assert!(invalid.rescaled(64, 36).is_err());
        let mut extreme = lens.clone();
        extreme.fx = f64::MAX;
        assert!(extreme.rescaled(128, 36).is_err());
        extreme.fx = 2.0 * VALIDATION_EPSILON;
        assert!(extreme.rescaled(1, 36).is_err());
        assert_eq!(lens, Lens::flat(64, 36));
    }

    fn sample_json() -> &'static str {
        r#"{
            "schema_version": 2,
            "lenses": [
                { "width": 3840, "height": 2160,
                  "fx": 1796.32, "fy": 1797.22, "cx": 1919.37, "cy": 1063.17,
                  "distortion": [0.0342, 0.0677, -0.0741, 0.0299], "correction": 1.0 },
                { "width": 3840, "height": 2160,
                  "fx": 1796.32, "fy": 1797.22, "cx": 1919.37, "cy": 1063.17,
                  "distortion": [0.0342, 0.0677, -0.0741, 0.0299], "correction": 1.0 }
            ],
            "projection": { "type": "l-shape",
                          "intersect": 0.5446, "x_ty": 0.00476, "x_rz": 0.00753,
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
        assert!((cal.projection.l_shape().unwrap().intersect - 0.5446).abs() < 1e-4);
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
        cal.projection.l_shape_mut().unwrap().blend_width = 0.123;
        cal.framing.tilt = 0.3;
        cal.framing.roll = -0.12;
        cal.sync_offset = 67;

        let json = cal.to_json_pretty();
        let back: Calibration = serde_json::from_str(&json).unwrap();
        assert_eq!(back.lenses.len(), cal.lenses.len());
        assert!((back.lenses[0].correction - 0.0).abs() < 1e-6);
        assert!((back.projection.l_shape().unwrap().blend_width - 0.123).abs() < 1e-6);
        assert!(
            (back.projection.l_shape().unwrap().intersect
                - cal.projection.l_shape().unwrap().intersect)
                .abs()
                < 1e-9
        );
        assert!((back.framing.axis_offset - cal.framing.axis_offset).abs() < 1e-9);
        assert!((back.framing.tilt - 0.3).abs() < 1e-9);
        assert!((back.framing.roll + 0.12).abs() < 1e-9);
        assert_eq!(back.sync_offset, 67);
    }

    #[test]
    fn parse_calibration_with_field_roi() {
        let mut cal: Calibration = serde_json::from_str(sample_json()).unwrap();
        cal.field_roi = Some(FieldRoi {
            corners: [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]].map(|position| {
                crate::projection::SourcePoint {
                    camera: 0,
                    position,
                }
            }),
        });
        let json = cal.to_json_pretty();
        let back: Calibration = serde_json::from_str(&json).unwrap();
        let roi = back.field_roi.as_ref().unwrap();
        assert_eq!(roi.corners.len(), 4);
        assert_eq!(roi.corners[1].position, [0.7, 0.3]);
    }

    fn valid_cal() -> Calibration {
        let lens = || Lens {
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
            projection: Projection::LShape(projection::LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            }),
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
    fn untagged_projection_document_is_rejected() {
        // The `type` tag is mandatory: a projection object without it
        // must fail to parse rather than silently default to L-shape.
        let mut v: serde_json::Value = serde_json::from_str(&valid_cal().to_json_pretty()).unwrap();
        let topo = v["projection"].as_object_mut().unwrap();
        assert_eq!(topo.remove("type").unwrap(), "l-shape");
        assert!(serde_json::from_value::<Calibration>(v).is_err());
    }

    #[test]
    fn tagged_cylinder_document_round_trips() {
        let cal = Calibration::new(
            vec![Lens::flat(3840, 1080)],
            projection::Cylinder::default(),
            Framing {
                axis_offset: 0.0,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        cal.validate().unwrap();
        let json = cal.to_json_pretty();
        assert!(
            json.contains("\"type\": \"cylinder\""),
            "tagged serialization: {json}"
        );
        let back: Calibration = serde_json::from_str(&json).unwrap();
        let t = back.projection.cylinder().expect("cylinder round-trip");
        assert!((t.radius - 2400.0).abs() < 1e-9);
        assert!((t.arc_deg - 180.0).abs() < 1e-9);
        back.validate().unwrap();
    }

    #[test]
    fn cylinder_lens_count_is_enforced() {
        let mut cal = valid_cal();
        cal.projection = Projection::Cylinder(projection::Cylinder::default());
        assert!(
            matches!(
                cal.validate(),
                Err(CalibrationError::LensCountMismatch {
                    expected: 1,
                    found: 2,
                    ..
                })
            ),
            "two lenses on a mono projection must be rejected"
        );
    }

    #[test]
    fn cylinder_parameters_are_validated() {
        let bad = |f: fn(&mut projection::Cylinder)| {
            let mut t = projection::Cylinder::default();
            f(&mut t);
            let cal = Calibration::new(
                vec![Lens::flat(3840, 1080)],
                t,
                Framing {
                    axis_offset: 0.0,
                    tilt: 0.0,
                    roll: 0.0,
                },
            );
            cal.validate()
        };
        assert!(bad(|t| t.radius = 0.0).is_err());
        assert!(bad(|t| t.surface_height = Some(f64::NAN)).is_err());

        // The messages must state the documented (0, 360] sweep bounds
        // and the validation epsilon, not float internals like
        // MIN_POSITIVE.
        let msg = bad(|t| t.arc_deg = 361.0).unwrap_err().to_string();
        assert!(msg.contains("[0, 360]"), "{msg}");
        let msg = bad(|t| t.arc_deg = 0.0).unwrap_err().to_string();
        assert!(msg.contains("> 0.000001,"), "{msg}");
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
            Err(CalibrationError::ValueTooSmall { ref field, .. }) if field == "lens[0].fx"
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
            Err(CalibrationError::ValueTooSmall { ref field, .. }) if field == "framing.axis_offset"
        ));
    }

    #[test]
    fn rejects_intersect_out_of_range() {
        let mut c = valid_cal();
        c.projection.l_shape_mut().unwrap().intersect = 1.5;
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::OutOfRange { ref field, .. }) if field == "projection.intersect"
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
            Err(CalibrationError::LensCountMismatch { found: 1, .. })
        ));
    }

    #[test]
    fn rejects_three_lenses() {
        let mut c = valid_cal();
        let extra = c.lenses[0].clone();
        c.lenses.push(extra);
        assert!(matches!(
            c.validate(),
            Err(CalibrationError::LensCountMismatch { found: 3, .. })
        ));
    }

    #[test]
    fn legacy_match_format_gives_clear_error() {
        let path = std::env::temp_dir().join(format!("reco_legacy_{}.json", std::process::id()));
        std::fs::write(&path, r#"{"left_uniforms":{"width":100},"params":{}}"#).unwrap();
        let err = Calibration::from_file(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(matches!(err, CalibrationLoadError::Parse(_)));
        assert!(
            err.to_string().contains("unknown field `left_uniforms`"),
            "{err}"
        );
    }

    #[test]
    fn stray_legacy_key_gives_clear_error() {
        // A hand-migrated file keeping old camelCase keys must not load
        // with the seam alignment silently zeroed.
        let mut cal = valid_cal();
        cal.projection.l_shape_mut().unwrap().x_ty = 0.0048;
        let mut v: serde_json::Value = serde_json::from_str(&cal.to_json_pretty()).unwrap();
        let topo = v["projection"].as_object_mut().unwrap();
        topo.remove("x_ty");
        topo.insert("xTy".into(), serde_json::json!(0.0048));
        let path = std::env::temp_dir().join(format!("reco_stray_{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
        let err = Calibration::from_file(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(matches!(err, CalibrationLoadError::Parse(_)));
        assert!(err.to_string().contains("unknown field `xTy`"), "{err}");
    }

    #[test]
    fn newer_schema_version_gives_clear_error() {
        // A file from a future reco must fail loud, not load through the
        // current-schema path with silently misread fields.
        let mut cal = valid_cal();
        cal.schema_version = 99;
        let path = std::env::temp_dir().join(format!("reco_v99_{}.json", std::process::id()));
        std::fs::write(&path, cal.to_json_pretty()).unwrap();
        let err = Calibration::from_file(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(
                err,
                CalibrationLoadError::Invalid(CalibrationError::UnsupportedSchemaVersion {
                    found: 99
                })
            ),
            "expected UnsupportedSchemaVersion, got {err:?}"
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
