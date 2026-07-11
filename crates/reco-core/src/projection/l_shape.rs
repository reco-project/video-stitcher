//! The L-shape projection: two fisheye cameras on perpendicular planes.
//!
//! This module owns everything L-shape: the calibration parameters
//! (serialized inside the document's `topology` object), their
//! validation, and - as the projection self-ownership refactor
//! progresses - the plane scene derivation and surface maps.

use serde::{Deserialize, Serialize};

use crate::calibration::{Calibration, CalibrationError, EPSILON, Framing};
use crate::projection::Projection;
use crate::render::viewport::ViewportConfig;
use crate::stitch::{BlendRule, SurfaceMap};

/// Default seam blend width for calibrations that do not specify one.
/// The single source for every constructor and serde default.
pub const DEFAULT_BLEND_WIDTH: f32 = 0.05;

// Functions, not constants: serde's `default = "..."` attribute takes
// a function path, never a value expression.
fn default_blend_width() -> f32 {
    DEFAULT_BLEND_WIDTH
}

/// 3D placement of the two L-shape source planes plus the overlap seam.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LShape {
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

impl LShape {
    /// Validate the L-shape's own parameters and its requirements on
    /// the framing it renders through.
    pub(crate) fn validate(&self, framing: &Framing) -> Result<(), CalibrationError> {
        if !self.intersect.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: "topology.intersect".to_owned(),
                value: format!("{}", self.intersect),
            });
        }
        if !(0.0..=1.0).contains(&self.intersect) {
            return Err(CalibrationError::IntersectOutOfRange {
                value: self.intersect,
            });
        }

        for (name, val) in [
            ("topology.x_ty", self.x_ty),
            ("topology.x_rz", self.x_rz),
            ("topology.x_rx", self.x_rx),
            ("topology.z_rx", self.z_rx),
            ("topology.z_rz", self.z_rz),
        ] {
            if !val.is_finite() {
                return Err(CalibrationError::NonFiniteFloat {
                    field: name.to_owned(),
                    value: format!("{val}"),
                });
            }
        }

        if !self.blend_width.is_finite() {
            return Err(CalibrationError::NonFiniteFloat {
                field: "topology.blend_width".to_owned(),
                value: format!("{}", self.blend_width),
            });
        }
        // The seam smoothstep needs ordered edges; outside [0, 1] the blend
        // is meaningless (the old ViewportConfig::validate enforced this).
        if !(0.0..=1.0).contains(&self.blend_width) {
            return Err(CalibrationError::OutOfRange {
                field: "topology.blend_width".to_owned(),
                value: self.blend_width as f64,
                min: 0.0,
                max: 1.0,
            });
        }

        // The off-axis camera placement is an L-shape concept: the two
        // planes are viewed from `[axis_offset, 0, axis_offset]`, and a
        // zero offset would normalize a zero vector in the view basis.
        if framing.axis_offset <= EPSILON {
            return Err(CalibrationError::AxisOffsetTooSmall {
                value: framing.axis_offset,
                epsilon: EPSILON,
            });
        }

        Ok(())
    }
}

impl Projection for LShape {
    fn name(&self) -> &'static str {
        "l-shape-stereo-2camera"
    }

    fn camera_count(&self) -> usize {
        2
    }

    fn surface_maps(
        &self,
        calibration: &Calibration,
        config: &ViewportConfig,
        yaw: f32,
        pitch: f32,
    ) -> Vec<(Box<dyn SurfaceMap>, BlendRule)> {
        let (left, right) =
            crate::stitch::geometry::l_shape_plane_maps(self, calibration, config, yaw, pitch);
        vec![
            (Box::new(left), BlendRule::Opaque),
            (
                Box::new(right),
                BlendRule::Smoothstep(self.blend_width as f64),
            ),
        ]
    }

    #[cfg(feature = "gpu")]
    fn gpu_program(&self) -> crate::render::GpuProgram {
        crate::render::GpuProgram {
            wgsl: include_str!("../shaders/fisheye.wgsl"),
            vs_entry: "vs_main",
            fs_entry: "fs_main",
            // Seam transition: the right plane's smoothstep alpha blends
            // over the opaque left base (matches BlendRule ordering).
            blend: wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            },
            vertex_layout: crate::render::renderer::Vertex::LAYOUT,
        }
    }
}
