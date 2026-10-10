//! The L-shape projection: two fisheye cameras on perpendicular planes.
//!
//! This module owns everything L-shape: the calibration parameters
//! (serialized inside the document's `projection` object), their
//! validation, plane scene derivation, detection mapping, and CPU
//! surface maps ([`PlaneMap`]).

use nalgebra::{Matrix3, Matrix4, Perspective3, Point3, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

use crate::calibration::{
    CalibrationError, Framing, Lens, expect_finite, expect_in_range, expect_positive,
};
use crate::geometry::{
    CameraIndex, FAR_PLANE, NEAR_PLANE, RenderPose, VirtualCamera, opengl_to_wgpu_matrix,
    view_matrix,
};
use crate::lens::{inverse_fisheye, kb4};
use crate::projection::{ProjectionContext, ProjectionGeometry, SourceMapError};
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LShape {
    /// Plane placement parameter in `[0, 1]`. Each plane is translated by `(plane_width / 2) × (1 - intersect)` along its placement axis. This is not a measured fraction of rendered overlap.
    pub intersect: f64,
    /// Y-axis translation of the right plane (vertical misalignment).
    #[serde(default)]
    pub x_ty: f64,
    /// Z-axis rotation of the right plane, radians (roll).
    #[serde(default)]
    pub x_rz: f64,
    /// X-axis post-rotation of the left plane, in radians. Composed after its base Y/Z rotation.
    #[serde(default)]
    pub z_rx: f64,
    /// X-axis rotation of the right plane, radians (pitch).
    #[serde(default)]
    pub x_rx: f64,
    /// Z-axis rotation within the left plane's base Y/Z rotation, in radians.
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
        expect_finite("projection.intersect", self.intersect)?;
        expect_in_range("projection.intersect", self.intersect, 0.0, 1.0)?;

        for (name, val) in [
            ("projection.x_ty", self.x_ty),
            ("projection.x_rz", self.x_rz),
            ("projection.x_rx", self.x_rx),
            ("projection.z_rx", self.z_rx),
            ("projection.z_rz", self.z_rz),
        ] {
            expect_finite(name, val)?;
        }

        expect_finite("projection.blend_width", f64::from(self.blend_width))?;
        // The seam smoothstep needs ordered edges; outside [0, 1] the blend
        // is meaningless (the old ViewportSize::validate enforced this).
        expect_in_range(
            "projection.blend_width",
            f64::from(self.blend_width),
            0.0,
            1.0,
        )?;

        // The off-axis camera placement is an L-shape concept: the two
        // planes are viewed from `[axis_offset, 0, axis_offset]`, and a
        // zero offset would normalize a zero vector in the view basis.
        expect_positive("framing.axis_offset", framing.axis_offset)?;
        let axis = framing.axis_offset as f32;
        // The reference basis is normalized in the renderer's f32 precision.
        expect_finite(
            "framing.axis_offset camera distance",
            f64::from(Vector3::new(axis, 0.0, axis).norm()),
        )?;

        Ok(())
    }
}

impl LShape {
    /// Derive unit-square transforms from plane placement and each calibrated image's aspect.
    ///
    /// `aspects` are calibration image `width / height`, not decoded raster sizes. Plane placement:
    /// - Left plane: `position = [0, 0, (w/2)(1 - intersect)]`, `rotation = [z_rx, π/2, z_rz]`
    /// - Right plane: `position = [(w/2)(1 - intersect), x_ty, 0]`, `rotation = [x_rx, 0, x_rz]`
    /// - Virtual camera at `[axis_offset, 0, axis_offset]`.
    pub(crate) fn scene(&self, framing: &Framing, aspects: [f32; 2]) -> PlaneScene {
        PlaneScene {
            models: self
                .plane_models(aspects.map(f64::from))
                .map(|model| model.cast()),
            camera: self.virtual_camera(framing),
        }
    }

    fn plane_models(&self, aspects: [f64; 2]) -> [Matrix4<f64>; 2] {
        // Unit plane width: each plane is a 1.0-wide quad, so the
        // intersect offset is half of what the overlap leaves.
        let half_offset = 0.5 * (1.0 - self.intersect);
        // z_rx rolls around the final plane normal, after the base Y rotation and z_rz tilt.
        let base = UnitQuaternion::from_euler_angles(0.0, std::f64::consts::FRAC_PI_2, self.z_rz);
        let roll = UnitQuaternion::from_euler_angles(self.z_rx, 0.0, 0.0);
        let right_rotation = UnitQuaternion::from_euler_angles(self.x_rx, 0.0, self.x_rz);
        let scale = |aspect| Matrix4::new_nonuniform_scaling(&Vector3::new(1.0, 1.0 / aspect, 1.0));
        [
            Translation3::new(0.0, 0.0, half_offset).to_homogeneous()
                * (roll * base).to_homogeneous()
                * scale(aspects[0]),
            Translation3::new(half_offset, self.x_ty, 0.0).to_homogeneous()
                * right_rotation.to_homogeneous()
                * scale(aspects[1]),
        ]
    }
}

/// L-shape geometry shared by rendering, coverage and source-point mapping.
#[derive(Debug, Clone)]
pub(crate) struct PlaneScene {
    /// Unit-square to world transforms, including each plane's calibrated size.
    pub(crate) models: [Matrix4<f32>; 2],
    /// Virtual camera position and reference orientation.
    pub(crate) camera: VirtualCamera,
}

/// Lens operations shared by raster sampling and source correspondence.
struct NormalizedLens {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    d: [f64; 4],
    correction: f64,
}

impl NormalizedLens {
    fn new(lens: &Lens) -> Self {
        Self {
            fx: lens.fx / lens.width as f64,
            fy: lens.fy / lens.height as f64,
            cx: lens.cx / lens.width as f64,
            cy: lens.cy / lens.height as f64,
            d: lens.distortion,
            correction: lens.correction as f64,
        }
    }

    fn unproject(
        &self,
        point: [f64; 2],
        inverse: &crate::lens::inverse::Inverse,
    ) -> Option<[f64; 2]> {
        let dx = (point[0] - self.cx) / self.fx;
        let dy = (point[1] - self.cy) / self.fy;
        let radius = dx.hypot(dy);
        let theta = inverse.theta(radius)?;
        let scale = if radius < 1e-12 {
            1.0
        } else {
            theta.tan() / radius
        };
        let u = self.fx * dx * scale + self.cx;
        let v = self.fy * dy * scale + self.cy;
        (u.is_finite() && v.is_finite()).then_some([(u - 0.5) / 2.0, (0.5 - v) / 2.0])
    }

    fn sample_plane(&self, [x, y]: [f64; 2]) -> Option<SurfaceUv> {
        let uv_x = x + 0.5;
        let uv_y = 0.5 - y;
        if !(0.0..=1.0).contains(&uv_x) || !(0.0..=1.0).contains(&uv_y) {
            return None;
        }
        let euv_x = uv_x * 2.0 - 0.5;
        let euv_y = uv_y * 2.0 - 0.5;
        let xn = (euv_x - self.cx) / self.fx;
        let yn = (euv_y - self.cy) / self.fy;
        let r = (xn * xn + yn * yn).sqrt();
        let scale = kb4::kb4_forward_scale_with_correction(r, &self.d, self.correction);
        let u = self.fx * xn * scale + self.cx;
        let v = self.fy * yn * scale + self.cy;
        ((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)).then_some(SurfaceUv {
            u,
            v,
            edge: euv_x,
        })
    }
}

/// Convert extended plane UV to a point on the selected L-shape plane.
fn plane_uv_to_world(uv: (f64, f64), camera: CameraIndex, scene: &PlaneScene) -> Point3<f32> {
    let tex_u = ((uv.0 + 0.5) / 2.0) as f32;
    let tex_v = ((uv.1 + 0.5) / 2.0) as f32;
    let local_x = tex_u - 0.5;
    let local_y = 0.5 - tex_v;
    let local_point = nalgebra::Vector4::new(local_x, local_y, 0.0, 1.0);

    // A validated L-shape calibration has exactly two lenses, so resolving
    // `lenses[camera]` before this makes the index a binary choice.
    debug_assert!(camera < 2, "the L-shape mapping is two-camera");
    let world = scene.models[camera] * local_point;
    Point3::new(world.x, world.y, world.z)
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
    clip_depth: bool,
    /// Normalised intrinsics: `fx/w`, `fy/h`, `cx/w`, `cy/h` (by calibration
    /// resolution, so they are independent of the actual frame size).
    optics: NormalizedLens,
    tan_half_h: f64,
    tan_half_v: f64,
}

impl PlaneMap {
    /// Build a plane map from its model matrix and the shared view-projection.
    fn new(
        model: Matrix4<f32>,
        view_projection: &Matrix4<f32>,
        cam: &Lens,
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
        let mut optics = NormalizedLens::new(cam);
        optics.correction = correction;
        Self {
            m3_inv,
            z_row,
            clip_depth: true,
            optics,
            tan_half_h: 1.0
                / (view_projection[(0, 0)] as f64)
                    .hypot(view_projection[(0, 1)] as f64)
                    .hypot(view_projection[(0, 2)] as f64),
            tan_half_v: 1.0
                / (view_projection[(1, 0)] as f64)
                    .hypot(view_projection[(1, 1)] as f64)
                    .hypot(view_projection[(1, 2)] as f64),
        }
    }
}

impl SurfaceMap for PlaneMap {
    fn covers_ndc(&self, min: [f64; 2], max: [f64; 2], rotation_bound: f64) -> bool {
        use super::interval::Interval as I;
        if !rotation_bound.is_finite() || !(0.0..=std::f64::consts::PI).contains(&rotation_bound) {
            return false;
        }
        let Some(m) = self.m3_inv else {
            return false;
        };
        let (x, y) = (I::new(min[0], max[0]), I::new(min[1], max[1]));
        let row = |i| I::point(m[(i, 0)]) * x + I::point(m[(i, 1)]) * y + I::point(m[(i, 2)]);
        let w = row(2);
        if w.lo <= 1e-12 {
            return false;
        }
        let lx = row(0) / w;
        let ly = row(1) / w;
        let depth_coefficient =
            |i| self.z_row.0 * m[(0, i)] + self.z_row.1 * m[(1, i)] + self.z_row.2 * m[(2, i)];
        let depth = I::point(depth_coefficient(0)) * x
            + I::point(depth_coefficient(1)) * y
            + I::point(depth_coefficient(2));
        if !depth.inside(1e-7, 1.0 - 1e-7) {
            return false;
        }
        if rotation_bound > 0.0 {
            let clip_w = I::point(1.0) / w;
            let distance = clip_w.hi
                * (1.0
                    + (x.abs_max() * self.tan_half_h).powi(2)
                    + (y.abs_max() * self.tan_half_v).powi(2))
                .sqrt();
            let margin = 2.0 * (rotation_bound * 0.5).sin() * distance;
            if clip_w.lo - margin < f64::from(NEAR_PLANE) + 1e-6
                || clip_w.hi + margin > f64::from(FAR_PLANE) - 1e-6
            {
                return false;
            }
        }
        let u = lx + I::point(0.5);
        let v = I::point(0.5) - ly;
        if !u.inside(1e-7, 1.0 - 1e-7) || !v.inside(1e-7, 1.0 - 1e-7) {
            return false;
        }
        let xn = (u * I::point(2.0) - I::point(0.5 + self.optics.cx)) / I::point(self.optics.fx);
        let yn = (v * I::point(2.0) - I::point(0.5 + self.optics.cy)) / I::point(self.optics.fy);
        let r = (xn.square() + yn.square()).sqrt();
        if !r.finite() {
            return false;
        }
        // atan(r)/r decreases for nonnegative r. The shader blends only the polynomial coefficients.
        let ratio = |r: f64| if r < 1e-10 { 1.0 } else { r.atan() / r };
        let base = I::new(ratio(r.hi) - 1e-12, ratio(r.lo.max(0.0)) + 1e-12);
        let t2 = I::new(r.lo.max(0.0).atan() - 1e-12, r.hi.atan() + 1e-12).square();
        let mut polynomial = I::point(self.optics.d[3] * self.optics.correction);
        for k in self.optics.d[..3].iter().rev() {
            polynomial = polynomial * t2 + I::point(k * self.optics.correction);
        }
        polynomial = polynomial * t2 + I::point(1.0);
        let scale = base * polynomial;
        let du = I::point(self.optics.fx) * xn * scale + I::point(self.optics.cx);
        let dv = I::point(self.optics.fy) * yn * scale + I::point(self.optics.cy);
        du.inside(1e-7, 1.0 - 1e-7) && dv.inside(1e-7, 1.0 - 1e-7)
    }

    fn sample_uv(&self, [ndc_x, ndc_y]: [f64; 2]) -> Option<SurfaceUv> {
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
        if self.clip_depth && (clip_z < 0.0 || clip_z > clip_w) {
            return None;
        }

        self.optics.sample_plane([local_x, local_y])
    }
}

/// Build the two L-shape plane maps `(left, right)` for one frame.
///
/// Reuses the same scene geometry, view matrix, and perspective projection as
/// the GPU stitch pass, so the CPU and GPU sample the identical source UV for
/// every output pixel (up to f32/f64 precision).
fn l_shape_plane_maps(
    projection: &LShape,
    lenses: &[Lens],
    framing: &Framing,
    viewport_size: &ViewportSize,
    pose: RenderPose,
) -> (PlaneMap, PlaneMap) {
    let scene = projection.scene(
        framing,
        [lenses[0].aspect_ratio(), lenses[1].aspect_ratio()],
    );

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
        &scene.camera,
        pose.yaw,
        pose.pitch,
        framing.tilt as f32,
        framing.roll as f32,
    );
    let view_projection = projection * view;

    let left = PlaneMap::new(
        scene.models[0],
        &view_projection,
        &lenses[0],
        lenses[0].correction as f64,
    );
    let right = PlaneMap::new(
        scene.models[1],
        &view_projection,
        &lenses[1],
        lenses[1].correction as f64,
    );
    (left, right)
}

impl ProjectionGeometry for LShape {
    fn source_rays(
        &self,
        camera: CameraIndex,
        lenses: &[Lens],
        framing: &Framing,
    ) -> Result<super::SourceRays, SourceMapError> {
        let lens = NormalizedLens::new(lenses.get(camera).ok_or(SourceMapError::InvalidCamera)?);
        let inverse = crate::lens::inverse::Inverse::new(&lens.d, lens.correction)
            .ok_or(SourceMapError::NoDirection)?;
        let models = self.plane_models([
            lenses[0].width as f64 / lenses[0].height as f64,
            lenses[1].width as f64 / lenses[1].height as f64,
        ]);
        let model = models[camera];
        let eye = Vector3::new(framing.axis_offset, 0.0, framing.axis_offset);
        Ok(Box::new(move |point| {
            let [x, y] = lens.unproject(point, &inverse)?;
            lens.sample_plane([x, y])?;
            let p = model * nalgebra::Vector4::new(x, y, 0.0, 1.0);
            (p.xyz() - eye).try_normalize(1e-12)
        }))
    }

    fn source_correspondence(
        &self,
        from: CameraIndex,
        to: CameraIndex,
        lenses: &[Lens],
        framing: &Framing,
    ) -> Result<super::SourceCorrespondence, SourceMapError> {
        let models = self.plane_models([
            f64::from(lenses[0].width) / f64::from(lenses[0].height),
            f64::from(lenses[1].width) / f64::from(lenses[1].height),
        ]);
        let target_inverse = models[to]
            .try_inverse()
            .ok_or(SourceMapError::NoDirection)?;
        let transfer = target_inverse * models[from];
        let axis = framing.axis_offset;
        let eye = target_inverse * nalgebra::Vector4::new(axis, 0.0, axis, 1.0);
        if !eye.z.is_finite() || eye.z.abs() < 1e-12 {
            return Err(SourceMapError::NoDirection);
        }
        let source = NormalizedLens::new(&lenses[from]);
        let target = NormalizedLens::new(&lenses[to]);
        let inverse = crate::lens::inverse::Inverse::new(&source.d, source.correction)
            .ok_or(SourceMapError::NoDirection)?;
        Ok(super::SourceCorrespondence::new(move |point| {
            let local = source.unproject(point, &inverse)?;
            source.sample_plane(local)?;
            let p = transfer * nalgebra::Vector4::new(local[0], local[1], 0.0, 1.0);
            let ray = p - eye;
            let distance = -eye.z / ray.z;
            if !distance.is_finite() || distance <= 0.0 {
                return None;
            }
            let uv = target.sample_plane([eye.x + distance * ray.x, eye.y + distance * ray.y])?;
            Some([uv.u, uv.v])
        }))
    }

    fn name(&self) -> &'static str {
        "l-shape-stereo-2camera"
    }

    fn camera_count(&self) -> usize {
        2
    }

    fn virtual_camera(&self, framing: &Framing) -> VirtualCamera {
        let axis = framing.axis_offset as f32;
        VirtualCamera::new(&[axis, 0.0, axis], &[-axis, 0.0, -axis])
    }

    fn map_source_point(
        &self,
        camera: CameraIndex,
        point: [f32; 2],
        lenses: &[Lens],
        framing: &Framing,
    ) -> Result<crate::geometry::YawPitch, SourceMapError> {
        let lens = lenses.get(camera).ok_or(SourceMapError::InvalidCamera)?;
        let scene = self.scene(
            framing,
            [lenses[0].aspect_ratio(), lenses[1].aspect_ratio()],
        );
        let plane_uv = inverse_fisheye(f64::from(point[0]), f64::from(point[1]), lens)
            .ok_or(SourceMapError::NoDirection)?;
        let world_point = plane_uv_to_world(plane_uv, camera, &scene);
        let direction = (world_point - scene.camera.eye).normalize();
        if !direction.iter().all(|v| v.is_finite()) {
            return Err(SourceMapError::NoDirection);
        }
        Ok(scene.camera.direction_to_yaw_pitch(&direction))
    }

    fn surface_maps(&self, ctx: &ProjectionContext) -> Vec<(Box<dyn SurfaceMap>, BlendRule)> {
        let (mut left, mut right) =
            l_shape_plane_maps(self, ctx.lenses, ctx.framing, ctx.viewport_size, ctx.pose);
        left.clip_depth = ctx.clip_depth;
        right.clip_depth = ctx.clip_depth;
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
            wgsl: concat!(
                include_str!("../shaders/video_sampling.wgsl"),
                include_str!("../shaders/fisheye.wgsl")
            ),
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
            pass: crate::render::GpuPass::Planes,
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
    fn correspondence_matches_rendered_overlap_in_both_directions() {
        let mut lenses = vec![
            Lens::fisheye(
                3840,
                2160,
                1796.32,
                1650.0,
                1919.37,
                1063.17,
                [0.0342, 0.0677, -0.0741, 0.0299],
            ),
            Lens::fisheye(
                3840,
                2880,
                1801.0,
                1740.0,
                1900.0,
                1450.0,
                [0.031, 0.061, -0.07, 0.028],
            ),
        ];
        lenses[1].correction = 0.65;
        let mut shape = topo(0.54);
        shape.x_ty = 0.007;
        shape.x_rz = 0.01;
        shape.z_rx = -0.012;
        shape.x_rx = 0.009;
        shape.z_rz = -0.007;
        let mut framing = framing(0.28);
        let viewport = ViewportSize {
            width: 640,
            height: 480,
        };
        let maps = [
            shape
                .source_correspondence(0, 1, &lenses, &framing)
                .unwrap(),
            shape
                .source_correspondence(1, 0, &lenses, &framing)
                .unwrap(),
        ];
        let mut samples = 0;
        for (tilt, roll) in [(0.0, 0.0), (0.17, -0.23)] {
            framing.tilt = tilt;
            framing.roll = roll;
            for (yaw, pitch, fov_degrees) in
                [(-0.3, -0.12, 100.0), (0.0, 0.0, 75.0), (0.3, 0.15, 110.0)]
            {
                let (left, right) = l_shape_plane_maps(
                    &shape,
                    &lenses,
                    &framing,
                    &viewport,
                    RenderPose {
                        yaw,
                        pitch,
                        fov_degrees,
                    },
                );
                for y in (0..viewport.height).step_by(7) {
                    for x in (0..viewport.width).step_by(7) {
                        let ndc = viewport.pixel_ndc(x, y);
                        if let (Some(a), Some(b)) = (left.sample_uv(ndc), right.sample_uv(ndc)) {
                            for (map, source, target) in [(&maps[0], a, b), (&maps[1], b, a)] {
                                let result = map.map([source.u, source.v]).unwrap();
                                let error = (result[0] - target.u).hypot(result[1] - target.v);
                                assert!(
                                    error < 2e-6,
                                    "source={source:?}, target={target:?}, error={error}"
                                );
                                samples += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(samples > 5000, "only {samples} comparisons");
    }

    #[test]
    fn correspondence_preserves_small_placement_changes() {
        let lenses = [
            Lens::fisheye(1920, 1080, 1000.0, 1000.0, 960.0, 540.0, [0.0; 4]),
            Lens::fisheye(1920, 1080, 1000.0, 1000.0, 960.0, 540.0, [0.0; 4]),
        ];
        let shape = topo(0.54);
        let frame = framing(0.28);
        let mut checked = 0;
        for parameter in 0..7 {
            for point in [[0.75, 0.4], [0.75, 0.6], [0.85, 0.5]] {
                let derivative = |step: f64| -> Option<[f64; 2]> {
                    let mut mapped = Vec::new();
                    for sign in [-1.0, 1.0] {
                        let mut shape = shape.clone();
                        let mut frame = frame.clone();
                        let value = match parameter {
                            0 => &mut shape.intersect,
                            1 => &mut shape.x_ty,
                            2 => &mut shape.x_rz,
                            3 => &mut shape.x_rx,
                            4 => &mut shape.z_rx,
                            5 => &mut shape.z_rz,
                            _ => &mut frame.axis_offset,
                        };
                        *value += sign * step;
                        mapped.push(
                            shape
                                .source_correspondence(0, 1, &lenses, &frame)
                                .ok()?
                                .map(point)?,
                        );
                    }
                    Some(std::array::from_fn(|i| {
                        (mapped[1][i] - mapped[0][i]) / (2.0 * step)
                    }))
                };
                let coarse = derivative(1e-5).expect("interior overlap point");
                let fine = derivative(1e-7).expect("interior overlap point");
                for (a, b) in coarse.into_iter().zip(fine) {
                    assert!(
                        (a - b).abs() < 1e-5 * a.abs().max(1.0),
                        "parameter={parameter}, coarse={a}, fine={b}"
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 21);
    }

    #[test]
    fn correspondence_rejects_invalid_points_and_camera_indices() {
        let cal = crate::stitch::test_support::calib(1920, 1080);
        for (from, to) in [(2, 0), (0, 2), (usize::MAX, 0)] {
            assert!(matches!(
                cal.source_correspondence(from, to),
                Err(SourceMapError::InvalidCamera)
            ));
        }
        let map = cal.source_correspondence(0, 0).unwrap();
        for point in [
            [f64::NAN, 0.5],
            [0.5, f64::INFINITY],
            [-0.01, 0.5],
            [0.5, 1.01],
        ] {
            assert!(map.map(point).is_none());
        }
        let source = [0.5, 0.5];
        let result = map.map(source).unwrap();
        assert!((result[0] - source[0]).hypot(result[1] - source[1]) < 1e-9);
    }

    #[test]
    fn correspondence_rejects_geometry_behind_source_ray() {
        let lens = Lens::fisheye(1920, 1080, 1000.0, 1000.0, 960.0, 540.0, [0.0; 4]);
        let shape = topo(0.5);
        let lenses = [lens.clone(), lens];
        // From the left plane centre (z=.25), this ray from (.1,0,.1)
        // travels toward increasing z. The right plane lies behind it at z=0.
        let map = shape
            .source_correspondence(0, 1, &lenses, &framing(0.1))
            .unwrap();
        assert!(map.map([0.5, 0.5]).is_none());
    }

    #[test]
    fn geometry_from_default_layout() {
        let geom = topo(0.5).scene(&framing(0.25), [16.0 / 9.0; 2]);

        // Half offset = 0.5 * (1 - 0.5) = 0.25
        assert!((geom.models[0][(2, 3)] - 0.25).abs() < 1e-5);
        assert!((geom.models[1][(0, 3)] - 0.25).abs() < 1e-5);
        assert!((geom.camera.eye.x - 0.25).abs() < 1e-5);
        assert!((geom.camera.eye.z - 0.25).abs() < 1e-5);
        for model in geom.models {
            assert!((model.column(1).norm() - 9.0 / 16.0).abs() < 1e-5);
        }
    }

    #[test]
    fn geometry_with_corrections() {
        let projection = LShape {
            intersect: 0.55,
            x_ty: 0.005,
            x_rz: 0.008,
            z_rx: -0.004,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        };

        let geom = projection.scene(&framing(0.24), [16.0 / 9.0; 2]);

        // Right plane should have the x_ty correction
        assert!((geom.models[1][(1, 3)] - 0.005).abs() < 1e-5);
        // Rotations should be applied
        assert!((geom.models[1][(1, 0)] - 0.008_f32.sin()).abs() < 1e-5);
        assert!((geom.models[0][(2, 1)] - (-0.004_f32).sin() * 9.0 / 16.0).abs() < 1e-5);
    }

    #[test]
    fn rejects_axis_offsets_that_overflow_the_camera_length() {
        for offset in [1e20, f64::MAX] {
            assert!(matches!(
                topo(0.5).validate(&framing(offset)),
                Err(CalibrationError::NonFiniteFloat { .. })
            ));
        }
    }

    #[test]
    fn geometry_with_independent_aspects() {
        for aspects in [[4.0 / 3.0, 16.0 / 9.0], [16.0 / 9.0, 4.0 / 3.0]] {
            let geom = topo(0.5).scene(&framing(0.25), aspects);
            for (model, aspect) in geom.models.iter().zip(aspects) {
                assert!((model.column(0).norm() - 1.0).abs() < 1e-5);
                assert!((model.column(1).norm() - 1.0 / aspect).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn inverse_fisheye_keeps_optical_center_fixed() {
        let lens = Lens::fisheye(
            3840,
            2160,
            1796.32,
            1797.22,
            1919.37,
            1063.17,
            [0.0342, 0.0677, -0.0741, 0.0299],
        );
        let cx = lens.cx / lens.width as f64;
        let cy = lens.cy / lens.height as f64;
        let actual = inverse_fisheye(cx, cy, &lens).unwrap();
        assert!((actual.0 - cx).abs() < 1e-6);
        assert!((actual.1 - cy).abs() < 1e-6);
    }

    #[test]
    fn inverse_fisheye_is_identity_at_zero_distortion_center() {
        let lens = Lens::fisheye(1920, 1080, 960.0, 540.0, 960.0, 540.0, [0.0; 4]);
        let actual = inverse_fisheye(0.5, 0.5, &lens).unwrap();
        assert!((actual.0 - 0.5).abs() < 1e-6);
        assert!((actual.1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn inverse_fisheye_round_trips_render_correction() {
        let mut lens = Lens::fisheye(
            3840,
            2160,
            1796.32,
            1650.0,
            1919.37,
            1063.17,
            [0.0342, 0.0677, -0.0741, 0.0299],
        );
        let fx = lens.fx / lens.width as f64;
        let fy = lens.fy / lens.height as f64;
        let cx = lens.cx / lens.width as f64;
        let cy = lens.cy / lens.height as f64;
        for correction in [0.0, 0.4, 1.0] {
            lens.correction = correction;
            for ix in 0..=20 {
                for iy in 0..=20 {
                    let source = (ix as f64 / 20.0, iy as f64 / 20.0);
                    let (u, v) = inverse_fisheye(source.0, source.1, &lens).unwrap();
                    let x = (u - cx) / fx;
                    let y = (v - cy) / fy;
                    let scale = kb4::kb4_forward_scale_with_correction(
                        x.hypot(y),
                        &lens.distortion,
                        correction as f64,
                    );
                    let error = ((fx * x * scale + cx - source.0) * lens.width as f64)
                        .hypot((fy * y * scale + cy - source.1) * lens.height as f64);
                    assert!(
                        error < 1e-6,
                        "correction={correction}, source={source:?}, error={error}px"
                    );
                }
            }
        }
    }

    #[test]
    fn inverse_fisheye_rejects_unsolved_and_invalid_points() {
        let mut lens = Lens::fisheye(
            3840,
            2160,
            1796.32,
            1797.22,
            1919.37,
            1063.17,
            [-0.3, 0.0, 0.0, 0.0],
        );
        // This source radius exceeds the polynomial's maximum on [0, pi/2).
        assert!(inverse_fisheye(0.02, 0.308, &lens).is_none());
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(inverse_fisheye(invalid, 0.5, &lens).is_none());
            assert!(inverse_fisheye(0.5, invalid, &lens).is_none());
        }
        lens.correction = 0.0;
        // Zero-polynomial inversion still requires an angle below pi/2.
        assert!(inverse_fisheye(2.0, 0.5, &lens).is_none());
    }

    #[test]
    fn covered_pixels_return_uv_in_range() {
        let cfg = ViewportSize::default();
        let cal = crate::stitch::test_support::calib(1920, 1080);
        let projection = cal.projection.l_shape().unwrap();
        let (left, right) = l_shape_plane_maps(
            projection,
            &cal.lenses,
            &cal.framing,
            &cfg,
            RenderPose::default(),
        );
        // Across the output, every covered pixel must report a UV inside [0,1],
        // and at least one plane must cover a healthy fraction of the frame.
        let mut covered = 0usize;
        for y in (0..cfg.height).step_by(17) {
            for x in (0..cfg.width).step_by(17) {
                for m in [&left, &right] {
                    if let Some(s) = m.sample_uv(cfg.pixel_ndc(x, y)) {
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
