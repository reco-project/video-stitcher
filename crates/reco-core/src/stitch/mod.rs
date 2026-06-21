//! CPU stitching backend (projection-agnostic).
//!
//! A pure-Rust, no-wgpu software stitcher that mirrors the GPU fisheye shader
//! ([`shaders/fisheye.wgsl`](../render/renderer/index.html)) per output pixel.
//! It serves two roles:
//!
//! 1. **Correctness oracle** for the GPU path. The geometry reuses
//!    [`crate::lens::kb4`] and the same view/projection matrices as
//!    [`crate::render`], so the CPU and GPU outputs agree by construction
//!    rather than by coincidence.
//! 2. **Rendering path for GPU-less targets** (edge SoCs, cloud CPU workers)
//!    where wgpu is unavailable or uneconomical.
//!
//! ## Projection seam
//!
//! [`SurfaceMap`] is the only projection-specific piece: it maps an output
//! pixel to a source-camera UV. The two-plane L-shape provides one
//! [`PlaneMap`] per camera; future projections (cylinder, N-camera) implement
//! the same trait and the gather loop in [`cpu`] is unchanged.
//!
//! ## Phase 1 scope
//!
//! Float reference, NV12 input, RGBA output, L-shape only. The integer /
//! memory-tuned specialisation and NV12-direct output are deliberate later
//! additions, gated on profiling (see the cpu-stitch portability work).

mod cpu;
pub mod geometry;

pub use cpu::stitch_l_shape_rgba;
pub use geometry::{PlaneMap, l_shape_plane_maps};

/// A source-camera sample location produced by a [`SurfaceMap`].
#[derive(Debug, Clone, Copy)]
pub struct SurfaceUv {
    /// Normalised camera UV in `[0, 1]`. Multiply by the frame dimensions for
    /// pixel coordinates.
    pub u: f64,
    /// Normalised camera UV in `[0, 1]`.
    pub v: f64,
    /// Extended-UV x of this surface (the shader's `uv.x * 2 - 0.5`). The
    /// compositor uses it to compute the seam-blend alpha where surfaces
    /// overlap.
    pub edge: f64,
}

/// The projection seam: a per-output-pixel inverse map for one source surface.
///
/// Implemented once per projection. The L-shape supplies two (one per camera
/// plane); the CPU gather loop queries each surface and composites the ones
/// that cover a pixel, so adding a projection means implementing this trait -
/// the loop never changes. This is the CPU dual of the GPU rasterizer's
/// per-fragment plane-UV interpolation.
pub trait SurfaceMap {
    /// Map an output pixel centre to its source-camera UV, or `None` when this
    /// surface does not cover the pixel (the GPU shader's bounds discard).
    fn sample_uv(&self, out_x: u32, out_y: u32) -> Option<SurfaceUv>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{CameraParams, MatchCalibration, PlaneLayout};
    use crate::render::planes::Nv12Planes;
    use crate::render::viewport::ViewportConfig;

    /// A frontal-ish two-camera calibration with a mild fisheye, sized for fast
    /// tests. Both cameras share dimensions (as a real stereo rig does).
    fn test_calib(w: u32, h: u32) -> MatchCalibration {
        let cam = || CameraParams {
            width: w,
            height: h,
            fx: w as f64 * 0.5,
            fy: w as f64 * 0.5,
            cx: w as f64 * 0.5,
            cy: h as f64 * 0.5,
            d: [-0.02, 0.004, 0.0, 0.0],
        };
        MatchCalibration {
            left: cam(),
            right: cam(),
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

    /// Synthetic NV12 frame: a horizontal luma gradient + flat mid-grey chroma.
    fn nv12(w: u32, h: u32, bias: u8) -> (Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0..w * h)
            .map(|i| ((i % w as usize as u32) as usize * 255 / w as usize) as u8)
            .map(|v| v.saturating_add(bias))
            .collect();
        let uv = vec![128u8; (w * (h / 2)) as usize];
        (y, uv)
    }

    #[test]
    fn output_dimensions_and_opaque_alpha() {
        let (w, h) = (96u32, 54u32);
        let calib = test_calib(w, h);
        let cfg = ViewportConfig {
            width: w,
            height: h,
            ..Default::default()
        };
        let (ly, luv) = nv12(w, h, 0);
        let (ry, ruv) = nv12(w, h, 40);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };

        let out = stitch_l_shape_rgba(&left, &right, (w, h), &calib, &cfg, 0.0, 0.0, false);
        assert_eq!(out.len(), (w * h * 4) as usize);
        // Alpha channel is fully opaque.
        assert!(out.iter().skip(3).step_by(4).all(|&a| a == 255));
    }

    #[test]
    fn produces_covered_pixels_and_is_deterministic() {
        let (w, h) = (96u32, 54u32);
        let calib = test_calib(w, h);
        let cfg = ViewportConfig {
            width: w,
            height: h,
            ..Default::default()
        };
        let (ly, luv) = nv12(w, h, 0);
        let (ry, ruv) = nv12(w, h, 40);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };

        let a = stitch_l_shape_rgba(&left, &right, (w, h), &calib, &cfg, 0.0, 0.0, false);
        let b = stitch_l_shape_rgba(&left, &right, (w, h), &calib, &cfg, 0.0, 0.0, false);
        assert_eq!(a, b, "stitch must be deterministic");

        // Some pixels are covered (non-black) - the planes are in view.
        let non_black = a
            .chunks_exact(4)
            .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
            .count();
        assert!(
            non_black > 0,
            "expected some covered (non-black) output pixels"
        );
    }
}
