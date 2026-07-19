//! The L-shape projection: two fisheye cameras on perpendicular planes.
//!
//! This module owns everything L-shape: the calibration parameters
//! (serialized inside the document's `topology` object), their
//! validation, the plane scene derivation, and the CPU surface maps
//! ([`PlaneMap`]).

use nalgebra::{Matrix3, Matrix4, Perspective3, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

use crate::calibration::{
    Calibration, CalibrationError, Framing, Lens, expect_finite, expect_in_range, expect_positive,
};
use crate::geometry::{
    FAR_PLANE, NEAR_PLANE, Pose, VirtualCamera, opengl_to_wgpu_matrix, view_matrix,
};
use crate::lens::kb4;
use crate::projection::{CoverageBoundary, Projection, ProjectionContext};
use crate::render::viewport::ViewportSize;
use crate::stitch::{BlendRule, SurfaceMap, SurfaceUv};

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
        expect_finite("topology.intersect", self.intersect)?;
        expect_in_range("topology.intersect", self.intersect, 0.0, 1.0)?;

        for (name, val) in [
            ("topology.x_ty", self.x_ty),
            ("topology.x_rz", self.x_rz),
            ("topology.x_rx", self.x_rx),
            ("topology.z_rx", self.z_rx),
            ("topology.z_rz", self.z_rz),
        ] {
            expect_finite(name, val)?;
        }

        expect_finite("topology.blend_width", f64::from(self.blend_width))?;
        // The seam smoothstep needs ordered edges; outside [0, 1] the blend
        // is meaningless (the old ViewportSize::validate enforced this).
        expect_in_range(
            "topology.blend_width",
            f64::from(self.blend_width),
            0.0,
            1.0,
        )?;

        // The off-axis camera placement is an L-shape concept: the two
        // planes are viewed from `[axis_offset, 0, axis_offset]`, and a
        // zero offset would normalize a zero vector in the view basis.
        expect_positive("framing.axis_offset", framing.axis_offset)?;

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

/// Inverse map for one camera plane of the L-shape projection:
/// output pixel -> source-camera UV.
///
/// The GPU rasterizes each camera plane (a textured quad) through the
/// model-view-projection matrix and the fragment shader applies the
/// forward KB4 distortion. For a planar quad that forward transform is
/// a homography, so the CPU dual is the *inverse* of the MVP's drop-z
/// 3x3, built once per frame per plane (reusing [`crate::render`]'s
/// exact view/projection and `crate::lens::kb4`).
///
/// Holds the per-frame inverse rasterization (output NDC -> plane-local)
/// plus the camera's normalised intrinsics and KB4 coefficients, so
/// [`sample_uv`] is a closed-form per-pixel evaluation with no allocation.
///
/// [`sample_uv`]: SurfaceMap::sample_uv
struct PlaneMap {
    /// Inverse of the MVP's drop-z 3x3: maps `[ndc_x, ndc_y, 1]` (up to scale)
    /// to plane-local `[x, y, 1]`. `None` if the MVP is singular (e.g. an
    /// edge-on plane), in which case the plane covers nothing - like the GPU's
    /// zero-area quad.
    m3_inv: Option<Matrix3<f64>>,
    /// The MVP's z-row restricted to the model-z=0 quad: `[m20, m21, m23]`, so
    /// `clip_z = z_row . [local_x, local_y, 1]`. The 3x3 inverse drops z, but the
    /// GPU rasterizer still depth-clips, so the z-row is kept to reconstruct it.
    z_row: (f64, f64, f64),
    /// Output dimensions in pixels.
    out_w: f64,
    out_h: f64,
    /// Camera aspect (`width / height`) baked into the plane quad's half-height.
    plane_aspect: f64,
    /// Normalised intrinsics: `fx/w`, `fy/h`, `cx/w`, `cy/h` (by calibration
    /// resolution, so they are independent of the actual frame size).
    fx_n: f64,
    fy_n: f64,
    cx_n: f64,
    cy_n: f64,
    /// KB4 distortion coefficients `[k1, k2, k3, k4]`.
    d: [f64; 4],
    /// Lens-correction amount in `[0, 1]` (`1` = full KB4, `0` = pinhole).
    correction: f64,
}

impl PlaneMap {
    /// Build a plane map from its model matrix and the shared view-projection.
    fn new(
        model: Matrix4<f32>,
        view_projection: &Matrix4<f32>,
        cam: &Lens,
        out_w: u32,
        out_h: u32,
        plane_aspect: f64,
        correction: f64,
    ) -> Self {
        let mvp = view_projection * model;
        // For a quad at z = 0 in model space, `clip = M3 * [x, y, 1]` where M3
        // takes the x, y and translation columns of the MVP (z dropped).
        let m3 = Matrix3::new(
            mvp[(0, 0)] as f64,
            mvp[(0, 1)] as f64,
            mvp[(0, 3)] as f64,
            mvp[(1, 0)] as f64,
            mvp[(1, 1)] as f64,
            mvp[(1, 3)] as f64,
            mvp[(3, 0)] as f64,
            mvp[(3, 1)] as f64,
            mvp[(3, 3)] as f64,
        );
        let m3_inv = m3.try_inverse();
        // z-row of the MVP for the model-z=0 quad (x, y, translation columns).
        let z_row = (mvp[(2, 0)] as f64, mvp[(2, 1)] as f64, mvp[(2, 3)] as f64);
        let w = cam.width as f64;
        let h = cam.height as f64;
        Self {
            m3_inv,
            z_row,
            out_w: out_w as f64,
            out_h: out_h as f64,
            plane_aspect,
            fx_n: cam.fx / w,
            fy_n: cam.fy / h,
            cx_n: cam.cx / w,
            cy_n: cam.cy / h,
            d: cam.distortion,
            correction,
        }
    }
}

impl SurfaceMap for PlaneMap {
    fn sample_uv(&self, out_x: u32, out_y: u32) -> Option<SurfaceUv> {
        // Output pixel centre -> wgpu NDC (x right, y up).
        let ndc_x = (out_x as f64 + 0.5) / self.out_w * 2.0 - 1.0;
        let ndc_y = 1.0 - (out_y as f64 + 0.5) / self.out_h * 2.0;

        // A singular MVP (edge-on plane) covers nothing.
        let m3_inv = self.m3_inv?;
        // Inverse rasterize to plane-local coordinates (homogeneous divide).
        let p = m3_inv * Vector3::new(ndc_x, ndc_y, 1.0);
        // p.z = 1 / clip_w: reject points at or behind the virtual camera
        // (clip_w <= 0), matching the GPU rasterizer's near/w clip. A plain
        // `p.z.abs()` guard would wrongly admit behind-camera geometry.
        if p.z <= 1e-12 {
            return None;
        }
        let local_x = p.x / p.z;
        let local_y = p.y / p.z;

        // Depth clip: the GPU rasterizer keeps only fragments with
        // `0 <= clip_z <= clip_w` (wgpu NDC z in [0, 1]); geometry within
        // NEAR_PLANE of the virtual camera is clipped. The 3x3 inverse dropped
        // z, so reconstruct `clip_z` from the z-row and `clip_w = 1 / p.z`.
        // Without this, a plane closer than NEAR_PLANE (tiny camera_axis_offset
        // or an extreme FOV) is gathered by the CPU but clipped by the GPU.
        let clip_w = 1.0 / p.z;
        let clip_z = self.z_row.0 * local_x + self.z_row.1 * local_y + self.z_row.2;
        if clip_z < 0.0 || clip_z > clip_w {
            return None;
        }

        // Plane-local -> texture UV (inverse of the quad vertex layout:
        // `local_x = uv_x - 0.5`, `local_y = (0.5 - uv_y) / aspect`).
        let uv_x = local_x + 0.5;
        let uv_y = 0.5 - local_y * self.plane_aspect;

        // The GPU rasterizes a FINITE quad (uv in [0,1]); the extended-UV remap
        // below only widens the sampling domain, not the rasterized footprint.
        // Reject pixels outside the quad so CPU coverage matches the GPU's.
        if !(0.0..=1.0).contains(&uv_x) || !(0.0..=1.0).contains(&uv_y) {
            return None;
        }

        // Shader's extended-UV remap (`uv * 2 - 0.5`) that widens the sampling
        // domain so undistortion can reach past the plane edge.
        let euv_x = uv_x * 2.0 - 0.5;
        let euv_y = uv_y * 2.0 - 0.5;

        // Forward KB4: extended-UV -> distorted (source) camera UV.
        let xn = (euv_x - self.cx_n) / self.fx_n;
        let yn = (euv_y - self.cy_n) / self.fy_n;
        let r = (xn * xn + yn * yn).sqrt();
        // The shader's per-pixel correction lerp between pinhole and full
        // KB4, from the single canonical source in `lens::kb4`.
        let scale = kb4::kb4_forward_scale_with_correction(r, &self.d, self.correction);
        let du = self.fx_n * xn * scale + self.cx_n;
        let dv = self.fy_n * yn * scale + self.cy_n;

        // Outside the source frame -> this surface does not cover the pixel.
        if !(0.0..=1.0).contains(&du) || !(0.0..=1.0).contains(&dv) {
            return None;
        }

        Some(SurfaceUv {
            u: du,
            v: dv,
            edge: euv_x,
        })
    }
}

/// Build the two L-shape plane maps `(left, right)` for one frame.
///
/// Reuses the same scene geometry, view matrix, and perspective projection as
/// the GPU stitch pass, so the CPU and GPU sample the identical source UV for
/// every output pixel (up to f32/f64 precision).
fn l_shape_plane_maps(
    topology: &LShape,
    calib: &Calibration,
    viewport_size: &ViewportSize,
    pose: Pose,
) -> (PlaneMap, PlaneMap) {
    // Known limitation: both planes are sized with lens 0's aspect.
    // Mixed-aspect rigs are valid calibrations; per-plane aspects are
    // the planned generalization.
    let plane_aspect = calib.lenses[0].aspect();
    let scene = topology.scene(&calib.framing, plane_aspect);

    let out_aspect = viewport_size.aspect_ratio();
    let projection = opengl_to_wgpu_matrix()
        * Perspective3::new(
            out_aspect,
            pose.render_fov().to_radians(),
            NEAR_PLANE,
            FAR_PLANE,
        )
        .to_homogeneous();
    let view = view_matrix(
        &scene.camera_position,
        pose.yaw,
        pose.pitch,
        calib.framing.tilt as f32,
        calib.framing.roll as f32,
    );
    let view_projection = projection * view;

    let aspect = plane_aspect as f64;
    let left = PlaneMap::new(
        scene.model_matrix_left(),
        &view_projection,
        &calib.lenses[0],
        viewport_size.width,
        viewport_size.height,
        aspect,
        calib.lenses[0].correction as f64,
    );
    let right = PlaneMap::new(
        scene.model_matrix_right(),
        &view_projection,
        &calib.lenses[1],
        viewport_size.width,
        viewport_size.height,
        aspect,
        calib.lenses[1].correction as f64,
    );
    (left, right)
}

impl Projection for LShape {
    fn name(&self) -> &'static str {
        "l-shape-stereo-2camera"
    }

    fn camera_count(&self) -> usize {
        2
    }

    fn virtual_camera(&self, framing: &Framing) -> VirtualCamera {
        let axis = framing.axis_offset as f32;
        VirtualCamera::new(&[axis, 0.0, axis])
    }

    /// Dense edge sampling of both planes through the virtual camera -
    /// the L-shape's panorama boundary is irregular, so it is measured,
    /// not computed analytically.
    fn coverage(&self, calibration: &Calibration) -> CoverageBoundary {
        let scene = self.scene(&calibration.framing, calibration.lenses[0].aspect());
        CoverageBoundary::from_l_shape(calibration, &scene)
    }

    fn surface_maps(&self, ctx: &ProjectionContext) -> Vec<(Box<dyn SurfaceMap>, BlendRule)> {
        let (left, right) = l_shape_plane_maps(self, ctx.calibration, ctx.viewport_size, ctx.pose);
        vec![
            (Box::new(left), BlendRule::Opaque),
            (
                Box::new(right),
                BlendRule::Smoothstep(self.blend_width as f64),
            ),
        ]
    }

    #[cfg(feature = "gpu")]
    fn gpu_program(&self) -> Option<crate::render::GpuProgram> {
        Some(crate::render::GpuProgram {
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
        })
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

    #[test]
    fn covered_pixels_return_uv_in_range() {
        let cfg = ViewportSize::default();
        let cal = crate::stitch::test_support::calib(1920, 1080);
        let topology = cal.topology.l_shape().unwrap();
        let (left, right) = l_shape_plane_maps(topology, &cal, &cfg, Pose::default());
        // Across the output, every covered pixel must report a UV inside [0,1],
        // and at least one plane must cover a healthy fraction of the frame.
        let mut covered = 0usize;
        for y in (0..cfg.height).step_by(17) {
            for x in (0..cfg.width).step_by(17) {
                for m in [&left, &right] {
                    if let Some(s) = m.sample_uv(x, y) {
                        assert!((0.0..=1.0).contains(&s.u) && (0.0..=1.0).contains(&s.v));
                        covered += 1;
                    }
                }
            }
        }
        assert!(
            covered > 0,
            "expected the planes to cover part of the output"
        );
    }
}
