//! The cylindrical projection: one pre-stitched panorama painted on
//! the inside of a cylinder, viewed from its axis.
//!
//! This module owns everything cylinder: the calibration parameters
//! (serialized inside the document's `topology` object), their
//! validation, and - as the projection self-ownership refactor
//! progresses - the cylinder inverse map.

use serde::{Deserialize, Serialize};

use crate::calibration::CalibrationError;

// Functions, not constants: serde's `default = "..."` attribute takes
// a function path, never a value expression.
fn default_focal_length() -> f64 {
    2400.0
}

fn default_sweep_deg() -> f64 {
    180.0
}

/// Cylinder placement for a single pre-stitched panorama: the video is
/// painted on the inside of a cylinder and the virtual camera sits on
/// its axis. Defaults match the established 180-degree
/// cylindrical-player convention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cylinder {
    /// Cylinder radius in world units. Larger values = narrower
    /// cylindrical wrap per pixel, so the panorama feels flatter.
    /// Conventional range is 1000-5000.
    #[serde(default = "default_focal_length")]
    pub focal_length: f64,
    /// Full horizontal angular sweep in degrees. 180 is the canonical
    /// pre-stitched action-camera case; 360 is a full cylinder.
    #[serde(default = "default_sweep_deg")]
    pub sweep_deg: f64,
    /// Painted height in the same units as `focal_length` (source
    /// pixels). Omitted = the source video's pixel height, which is
    /// the convention's default.
    #[serde(default)]
    pub video_height: Option<f64>,
}

impl Default for Cylinder {
    fn default() -> Self {
        Self {
            focal_length: default_focal_length(),
            sweep_deg: default_sweep_deg(),
            video_height: None,
        }
    }
}

impl Cylinder {
    /// Validate the cylinder's parameters: positive finite
    /// radius/height, sweep in `(0, 360]` degrees.
    pub(crate) fn validate(&self) -> Result<(), CalibrationError> {
        for (name, val, lo, hi) in [
            (
                "topology.focal_length",
                self.focal_length,
                f64::MIN_POSITIVE,
                f64::MAX,
            ),
            (
                "topology.sweep_deg",
                self.sweep_deg,
                f64::MIN_POSITIVE,
                360.0,
            ),
            (
                "topology.video_height",
                // Omitted = the source pixel height, always valid.
                self.video_height.unwrap_or(1.0),
                f64::MIN_POSITIVE,
                f64::MAX,
            ),
        ] {
            if !val.is_finite() {
                return Err(CalibrationError::NonFiniteFloat {
                    field: name.to_owned(),
                    value: format!("{val}"),
                });
            }
            if val < lo || val > hi {
                return Err(CalibrationError::OutOfRange {
                    field: name.to_owned(),
                    value: val,
                    min: lo,
                    max: hi,
                });
            }
        }
        Ok(())
    }
}
