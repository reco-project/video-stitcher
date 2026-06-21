//! L-shape projection geometry: output pixel -> source-camera UV per plane.
//!
//! The GPU rasterizes each camera plane (a textured quad) through the
//! model-view-projection matrix and the fragment shader applies the forward
//! KB4 distortion. For a planar quad that forward transform is a homography,
//! so the CPU dual - output pixel back to plane UV - is the *inverse* of the
//! MVP's drop-z 3x3. This module builds that inverse once per frame per plane
//! (reusing [`crate::render`]'s exact view/projection and [`crate::lens::kb4`])
//! and exposes it as a [`SurfaceMap`].

use nalgebra::{Matrix3, Matrix4, Perspective3, Vector3};

use crate::calibration::{CameraParams, MatchCalibration};
use crate::lens::kb4;
use crate::render::renderer::{FAR_PLANE, NEAR_PLANE, opengl_to_wgpu_matrix, view_matrix};
use crate::render::scene::SceneGeometry;
use crate::render::viewport::ViewportConfig;

use super::{SurfaceMap, SurfaceUv};

/// Inverse map for one camera plane of the L-shape projection.
///
/// Holds the per-frame inverse rasterization (output NDC -> plane-local) plus
/// the camera's normalised intrinsics and KB4 coefficients, so [`sample_uv`]
/// is a closed-form per-pixel evaluation with no allocation.
///
/// [`sample_uv`]: SurfaceMap::sample_uv
pub struct PlaneMap {
    /// Inverse of the MVP's drop-z 3x3: maps `[ndc_x, ndc_y, 1]` (up to scale)
    /// to plane-local `[x, y, 1]`. `None` if the MVP is singular (e.g. an
    /// edge-on plane), in which case the plane covers nothing - like the GPU's
    /// zero-area quad.
    m3_inv: Option<Matrix3<f64>>,
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
        cam: &CameraParams,
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
        let w = cam.width as f64;
        let h = cam.height as f64;
        Self {
            m3_inv,
            out_w: out_w as f64,
            out_h: out_h as f64,
            plane_aspect,
            fx_n: cam.fx / w,
            fy_n: cam.fy / h,
            cx_n: cam.cx / w,
            cy_n: cam.cy / h,
            d: cam.d,
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
        let scale = if r < 1e-9 {
            1.0
        } else {
            // `mix(theta, theta_d, correction) / r`, matching the shader's
            // per-pixel correction lerp between pinhole and full KB4.
            let pinhole = r.atan() / r;
            let full = kb4::kb4_forward_scale(r, &self.d);
            pinhole + (full - pinhole) * self.correction
        };
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
pub fn l_shape_plane_maps(
    calib: &MatchCalibration,
    config: &ViewportConfig,
    yaw: f32,
    pitch: f32,
) -> (PlaneMap, PlaneMap) {
    // Both planes share the camera aspect (a stereo rig's two cameras match).
    let plane_aspect = calib.left.width as f32 / calib.left.height as f32;
    let scene = SceneGeometry::from_layout_with_aspect(&calib.layout, plane_aspect);

    let out_aspect = config.width as f32 / config.height as f32;
    let projection = opengl_to_wgpu_matrix()
        * Perspective3::new(
            out_aspect,
            config.fov_degrees.to_radians(),
            NEAR_PLANE,
            FAR_PLANE,
        )
        .to_homogeneous();
    let view = view_matrix(
        &scene.camera_position,
        yaw,
        pitch,
        config.rig_tilt,
        config.rig_roll,
    );
    let view_projection = projection * view;

    let correction = config.lens_correction_amount as f64;
    let aspect = plane_aspect as f64;
    let left = PlaneMap::new(
        scene.model_matrix_left(),
        &view_projection,
        &calib.left,
        config.width,
        config.height,
        aspect,
        correction,
    );
    let right = PlaneMap::new(
        scene.model_matrix_right(),
        &view_projection,
        &calib.right,
        config.width,
        config.height,
        aspect,
        correction,
    );
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::PlaneLayout;

    fn calib() -> MatchCalibration {
        let cam = CameraParams {
            width: 1920,
            height: 1080,
            fx: 960.0,
            fy: 960.0,
            cx: 960.0,
            cy: 540.0,
            d: [-0.02, 0.004, 0.0, 0.0],
        };
        MatchCalibration {
            left: cam.clone(),
            right: cam,
            layout: PlaneLayout {
                camera_axis_offset: 0.25,
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
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
    fn covered_pixels_return_uv_in_range() {
        let cfg = ViewportConfig::default();
        let (left, right) = l_shape_plane_maps(&calib(), &cfg, 0.0, 0.0);
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
