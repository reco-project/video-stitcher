//! The L-shape projection: two fisheye cameras on perpendicular planes.
//!
//! This module owns everything L-shape: the calibration parameters
//! (serialized inside the document's `topology` object), their
//! validation, and - as the projection self-ownership refactor
//! progresses - the plane scene derivation and surface maps.

use nalgebra::{Matrix4, Translation3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::calibration::{Calibration, CalibrationError, Framing};
use crate::precision::VALIDATION_EPSILON;
use crate::projection::{CoverageBoundary, Projection, ProjectionContext};
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
        if framing.axis_offset <= VALIDATION_EPSILON {
            return Err(CalibrationError::AxisOffsetTooSmall {
                value: framing.axis_offset,
                epsilon: VALIDATION_EPSILON,
            });
        }

        Ok(())
    }
}

impl LShape {
    /// Derive the 3D plane placement from these parameters + framing.
    ///
    /// `aspect` is the source frame `width / height`. Mirrors the v1
    /// plane positioning:
    /// - Left plane: `position = [0, 0, (w/2)(1 - intersect)]`, `rotation = [z_rx, π/2, z_rz]`
    /// - Right plane: `position = [(w/2)(1 - intersect), x_ty, 0]`, `rotation = [x_rx, 0, x_rz]`
    /// - Virtual camera at `[axis_offset, 0, axis_offset]`.
    ///
    /// Known limitation: both planes are sized with one aspect (the
    /// callers pass lens 0's). Mixed-aspect rigs are valid
    /// calibrations; per-plane aspects are the planned generalization.
    pub(crate) fn scene(&self, framing: &Framing, aspect: f32) -> PlaneScene {
        // Unit plane width: each plane is a 1.0-wide quad, so the
        // intersect offset is half of what the overlap leaves.
        let half_offset = 0.5 * (1.0 - self.intersect as f32);
        let axis = framing.axis_offset as f32;

        PlaneScene {
            left_position: [0.0, 0.0, half_offset],
            left_rotation: [
                self.z_rx as f32,
                std::f32::consts::FRAC_PI_2,
                self.z_rz as f32,
            ],
            right_position: [half_offset, self.x_ty as f32, 0.0],
            right_rotation: [self.x_rx as f32, 0.0, self.x_rz as f32],
            camera_position: [axis, 0.0, axis],
            plane_aspect: aspect,
        }
    }
}

/// 3D placement of the two L-shape camera planes.
///
/// Derived from [`LShape`] + [`Framing`] by [`LShape::scene`]; consumed
/// by the CPU plane maps, the GPU uniforms, the coverage sampler, and
/// the detection forward map. Plane-less projections have no scene -
/// this type exists only inside the L-shape's own paths.
#[derive(Debug, Clone)]
pub(crate) struct PlaneScene {
    /// Left plane position `[x, y, z]`.
    pub(crate) left_position: [f32; 3],
    /// Left plane rotation `[rx, ry, rz]` in radians.
    pub(crate) left_rotation: [f32; 3],
    /// Right plane position `[x, y, z]`.
    pub(crate) right_position: [f32; 3],
    /// Right plane rotation `[rx, ry, rz]` in radians.
    pub(crate) right_rotation: [f32; 3],
    /// Virtual camera position `[x, y, z]`.
    pub(crate) camera_position: [f32; 3],
    /// Plane aspect ratio (width / height).
    pub(crate) plane_aspect: f32,
}

impl PlaneScene {
    /// Model matrix for the left camera plane.
    ///
    /// The z-plane base rotation is π/2 around Y (faces sideways).
    /// `z_rx` is applied as a post-rotation around X so it acts as
    /// a roll around the plane's final normal. `z_rz` is applied
    /// as a pre-rotation (tilt correction).
    pub(crate) fn model_matrix_left(&self) -> Matrix4<f32> {
        let t = Translation3::new(
            self.left_position[0],
            self.left_position[1],
            self.left_position[2],
        );
        // Base: π/2 Y rotation + z_rz tilt
        let base = UnitQuaternion::from_euler_angles(
            0.0,
            self.left_rotation[1], // π/2
            self.left_rotation[2], // z_rz
        );
        // Post-rotate: z_rx as roll around X (the plane's final normal)
        let roll = UnitQuaternion::from_euler_angles(
            self.left_rotation[0], // z_rx
            0.0,
            0.0,
        );
        let r = roll * base;
        t.to_homogeneous() * r.to_homogeneous()
    }

    /// Model matrix for the right camera plane.
    pub(crate) fn model_matrix_right(&self) -> Matrix4<f32> {
        let t = Translation3::new(
            self.right_position[0],
            self.right_position[1],
            self.right_position[2],
        );
        let r = UnitQuaternion::from_euler_angles(
            self.right_rotation[0],
            self.right_rotation[1],
            self.right_rotation[2],
        );
        t.to_homogeneous() * r.to_homogeneous()
    }
}

impl Projection for LShape {
    fn name(&self) -> &'static str {
        "l-shape-stereo-2camera"
    }

    fn camera_count(&self) -> usize {
        2
    }

    fn camera_position(&self, framing: &Framing) -> [f32; 3] {
        let axis = framing.axis_offset as f32;
        [axis, 0.0, axis]
    }

    /// Dense edge sampling of both planes through the virtual camera -
    /// the L-shape's panorama boundary is irregular, so it is measured,
    /// not computed analytically.
    fn coverage(&self, calibration: &Calibration) -> CoverageBoundary {
        let scene = self.scene(&calibration.framing, calibration.lenses[0].aspect());
        CoverageBoundary::from_l_shape(calibration, &scene)
    }

    fn surface_maps(&self, ctx: &ProjectionContext) -> Vec<(Box<dyn SurfaceMap>, BlendRule)> {
        let (left, right) = crate::stitch::geometry::l_shape_plane_maps(
            self,
            ctx.calibration,
            ctx.viewport,
            ctx.yaw,
            ctx.pitch,
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    fn topo(intersect: f64) -> LShape {
        LShape {
            intersect,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        }
    }

    fn framing(axis_offset: f64) -> Framing {
        Framing {
            axis_offset,
            tilt: 0.0,
            roll: 0.0,
        }
    }

    #[test]
    fn geometry_from_default_layout() {
        let geom = topo(0.5).scene(&framing(0.25), 16.0 / 9.0);

        // Half offset = 0.5 * (1 - 0.5) = 0.25
        assert!((geom.left_position[2] - 0.25).abs() < 1e-5);
        assert!((geom.right_position[0] - 0.25).abs() < 1e-5);
        assert!((geom.camera_position[0] - 0.25).abs() < 1e-5);
        assert!((geom.camera_position[2] - 0.25).abs() < 1e-5);
        assert!((geom.plane_aspect - 16.0 / 9.0).abs() < 1e-5);
    }

    #[test]
    fn geometry_with_corrections() {
        let topology = LShape {
            intersect: 0.55,
            x_ty: 0.005,
            x_rz: 0.008,
            z_rx: -0.004,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        };

        let geom = topology.scene(&framing(0.24), 16.0 / 9.0);

        // Right plane should have the x_ty correction
        assert!((geom.right_position[1] - 0.005).abs() < 1e-5);
        // Rotations should be applied
        assert!((geom.right_rotation[2] - 0.008).abs() < 1e-5);
        assert!((geom.left_rotation[0] - (-0.004)).abs() < 1e-5);
    }

    #[test]
    fn geometry_with_custom_aspect() {
        let aspect_4_3 = 4.0 / 3.0;
        let geom = topo(0.5).scene(&framing(0.25), aspect_4_3);
        assert!((geom.plane_aspect - aspect_4_3).abs() < 1e-5);
    }
}
