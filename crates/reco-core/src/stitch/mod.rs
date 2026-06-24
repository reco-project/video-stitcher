//! CPU stitching backend (projection-agnostic).
//!
//! A pure-Rust, no-wgpu software stitcher that mirrors the GPU fisheye shader
//! (`shaders/fisheye.wgsl`, the fragment stage in [`crate::render`]) per output
//! pixel. It serves two roles:
//!
//! 1. **Correctness oracle** for the GPU path. The geometry reuses
//!    `crate::lens::kb4` and the same view/projection matrices as
//!    [`crate::render`], so the CPU and GPU outputs agree by construction
//!    rather than by coincidence.
//! 2. **Rendering path for GPU-less targets** (edge SoCs, cloud CPU workers)
//!    where wgpu is unavailable or uneconomical.
//!
//! ## Projection seam
//!
//! [`SurfaceMap`] is the only projection-specific piece: it maps an output
//! pixel to a source-camera UV. The two-plane L-shape provides one
//! [`PlaneMap`] per camera. The gather loop in `cpu` is currently specialised
//! to the two-surface L-shape composite; the trait is the seam that N-surface
//! projections (cylinder, N-camera) build on.
//!
//! ## Phase 1 scope
//!
//! Float reference, NV12 input, RGBA output, L-shape only. The integer /
//! memory-tuned specialisation and NV12-direct output are deliberate later
//! additions, gated on profiling (see the cpu-stitch portability work).

mod backend;
mod cpu;
mod geometry;

pub use backend::{CpuStitchBackend, GpuStitchBackend, StitchBackend, StitchError};
pub use cpu::{stitch_l_shape_rgba, stitch_l_shape_rgba_yuv420p};
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
/// that cover a pixel. The L-shape composite loop is two-surface today; this
/// trait is the seam future N-surface projections build on. It is the CPU dual
/// of the GPU rasterizer's per-fragment plane-UV interpolation.
pub trait SurfaceMap {
    /// Map an output pixel centre to its source-camera UV, or `None` when this
    /// surface does not cover the pixel (the GPU shader's bounds discard).
    fn sample_uv(&self, out_x: u32, out_y: u32) -> Option<SurfaceUv>;
}

/// Shared test fixtures + GPU acquisition for the stitch test modules.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::calibration::{CameraParams, MatchCalibration, PlaneLayout};
    use crate::gpu::{GpuContext, GpuError};

    /// Two-camera calibration (shared dims, mild fisheye, centred) for tests.
    pub fn calib(w: u32, h: u32) -> MatchCalibration {
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
    pub fn nv12(w: u32, h: u32, bias: u8) -> (Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0..w * h)
            .map(|i| ((i % w) as usize * 255 / w as usize) as u8)
            .map(|val| val.saturating_add(bias))
            .collect();
        let uv = vec![128u8; (w * (h / 2)) as usize];
        (y, uv)
    }

    /// Acquire a GPU context, or `None` to skip the test - unless
    /// `RECO_REQUIRE_GPU` is set, in which case a missing adapter is a hard
    /// failure (so CI with a software adapter cannot silently skip the
    /// agreement tests).
    pub fn gpu_or_skip() -> Option<GpuContext> {
        match pollster::block_on(GpuContext::new()) {
            Ok(g) => Some(g),
            Err(GpuError::NoAdapter | GpuError::AdapterRequest(_)) => {
                assert!(
                    std::env::var("RECO_REQUIRE_GPU").is_err(),
                    "RECO_REQUIRE_GPU is set but no GPU adapter was found"
                );
                None
            }
            Err(e) => panic!("gpu init: {e}"),
        }
    }

    /// Per-channel RGB agreement between a reference (GPU) and a candidate (CPU)
    /// RGBA buffer (alpha ignored). The trustworthy gate is `mean` + `pct_over`
    /// (the fraction of channels diverging past [`DIVERGENCE_LSB`]); `max` is
    /// informational only. A *correct* stitch legitimately produces `max = 255`
    /// at a couple of seam coverage-boundary pixels on some rasterizers (RADV),
    /// so a global-max bound is unsound. Measured floor across llvmpipe / NVIDIA
    /// / RADV: `mean <= 0.31`, `pct_over <= 0.003%`. A real geometry or sampler
    /// divergence gives `mean` in the tens and `pct_over` in the tens of percent.
    pub struct Agreement {
        pub mean: f64,
        /// Percent of RGB channels differing by more than [`DIVERGENCE_LSB`].
        pub pct_over: f64,
        /// Largest single-channel difference (informational; not gated).
        pub max: u32,
    }

    /// Differences larger than this many LSB are structural (coverage / geometry
    /// divergence), never f32-vs-f64 or cross-GPU bilinear rounding: the floor
    /// keeps rounding noise at `<= 3` LSB, and the only larger diffs are full
    /// coverage-XOR flips at individual seam pixels.
    pub const DIVERGENCE_LSB: i32 = 16;

    impl Agreement {
        /// Compare two equal-length RGBA buffers.
        pub fn compare(reference: &[u8], candidate: &[u8]) -> Self {
            assert_eq!(
                reference.len(),
                candidate.len(),
                "agreement buffers differ in length"
            );
            let (mut sum, mut n, mut over, mut max) = (0i64, 0i64, 0i64, 0i32);
            for (r, c) in reference.chunks_exact(4).zip(candidate.chunks_exact(4)) {
                for k in 0..3 {
                    let d = (r[k] as i32 - c[k] as i32).abs();
                    sum += d as i64;
                    n += 1;
                    if d > DIVERGENCE_LSB {
                        over += 1;
                    }
                    max = max.max(d);
                }
            }
            Agreement {
                mean: sum as f64 / n as f64,
                pct_over: 100.0 * over as f64 / n as f64,
                max: max as u32,
            }
        }

        /// Assert the agreement is within `bounds`, printing the stats with
        /// `label` either way. A real divergence blows `mean` and `pct_over`
        /// orders of magnitude past these bounds.
        pub fn assert_within(&self, bounds: AgreementBounds, label: &str) {
            eprintln!(
                "[{label}] mean={:.3} >{DIVERGENCE_LSB}:{:.4}% max={}",
                self.mean, self.pct_over, self.max
            );
            assert!(
                self.mean < bounds.max_mean,
                "{label}: mean RGB diff {:.3} exceeds {:.3} (geometry/sampler divergence?)",
                self.mean,
                bounds.max_mean
            );
            assert!(
                self.pct_over < bounds.max_pct_over,
                "{label}: {:.4}% of channels off by >{DIVERGENCE_LSB} exceeds {:.4}% (coverage divergence?)",
                self.pct_over,
                bounds.max_pct_over
            );
        }
    }

    /// Tolerance for a GPU-vs-CPU agreement assertion. One
    /// [`AgreementBounds::DEFAULT`] covers every scene (smooth ramps and the
    /// high-frequency checker alike): the measured floor across three
    /// rasterizers is `mean <= 0.31` / `pct_over <= 0.003%`, so DEFAULT leaves
    /// ~2.6x / ~16x headroom for cross-GPU rounding while staying ~50-100x below
    /// any real divergence.
    #[derive(Clone, Copy)]
    pub struct AgreementBounds {
        pub max_mean: f64,
        pub max_pct_over: f64,
    }

    impl AgreementBounds {
        pub const DEFAULT: Self = Self {
            max_mean: 0.8,
            max_pct_over: 0.05,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{Agreement, AgreementBounds, calib, gpu_or_skip, nv12};
    use super::*;
    use crate::calibration::MatchCalibration;
    use crate::render::planes::Nv12Planes;
    use crate::render::viewport::ViewportConfig;

    #[test]
    fn output_dimensions_and_opaque_alpha() {
        let (w, h) = (96u32, 54u32);
        let calib = calib(w, h);
        let cfg = ViewportConfig {
            width: w,
            height: h,
            ..Default::default()
        };
        let (ly, luv) = nv12(w, h, 0);
        let (ry, ruv) = nv12(w, h, 40);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };

        let out =
            stitch_l_shape_rgba(&left, &right, (w, h), &calib, &cfg, 0.0, 0.0, false).unwrap();
        assert_eq!(out.len(), (w * h * 4) as usize);
        // Alpha channel is fully opaque.
        assert!(out.iter().skip(3).step_by(4).all(|&a| a == 255));
    }

    #[test]
    fn produces_covered_pixels_and_is_deterministic() {
        let (w, h) = (96u32, 54u32);
        let calib = calib(w, h);
        let cfg = ViewportConfig {
            width: w,
            height: h,
            ..Default::default()
        };
        let (ly, luv) = nv12(w, h, 0);
        let (ry, ruv) = nv12(w, h, 40);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };

        let a = stitch_l_shape_rgba(&left, &right, (w, h), &calib, &cfg, 0.0, 0.0, false).unwrap();
        let b = stitch_l_shape_rgba(&left, &right, (w, h), &calib, &cfg, 0.0, 0.0, false).unwrap();
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

    /// The keystone: the CPU float reference must agree with the GPU shader on
    /// the same scene. Because both share `view_matrix`, the projection, and
    /// `lens::kb4`, the only differences are f32-vs-f64 and hardware-vs-software
    /// bilinear - so the RGB match should be tight. Skips when no GPU adapter.
    #[test]
    fn cpu_matches_gpu_within_tolerance() {
        use crate::render::renderer::InputFormat;

        let Some(gpu) = gpu_or_skip() else {
            return;
        };

        let (cam_w, cam_h) = (256u32, 144u32);
        let (out_w, out_h) = (192u32, 108u32);
        let calib = calib(cam_w, cam_h);
        let config = ViewportConfig {
            width: out_w,
            height: out_h,
            ..Default::default()
        };
        let (ly, luv) = nv12(cam_w, cam_h, 0);
        let (ry, ruv) = nv12(cam_w, cam_h, 30);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };
        let (yaw, pitch) = (0.10f32, -0.05f32);

        // GPU render -> RGBA. The readback is triple-buffered (N-2 latency), so
        // render the same frame three times to drain one result.
        let pipeline = crate::render::pipeline::StitchPipeline::with_gpu(
            gpu,
            calib.clone(),
            config.clone(),
            cam_w,
            cam_h,
            wgpu::TextureFormat::Rgba8Unorm,
            InputFormat::Nv12,
        )
        .expect("pipeline");
        let mut readback =
            crate::gpu::rgba_readback::RgbaReadback::new(pipeline.gpu(), out_w, out_h)
                .expect("readback");
        let mut gpu_rgba: Option<Vec<u8>> = None;
        for _ in 0..3 {
            let cmd = pipeline
                .render_to_target_nv12(&left, &right, yaw, pitch)
                .expect("render");
            let tex = pipeline.render_target();
            if let Some(bytes) = readback
                .readback(pipeline.gpu(), tex, cmd)
                .expect("readback")
            {
                gpu_rgba = Some(bytes.to_vec());
            }
        }
        let gpu_rgba = gpu_rgba.expect("gpu should produce a frame after 3 renders");

        // CPU reference on the same inputs (limited range, matching the GPU default).
        let cpu_rgba = stitch_l_shape_rgba(
            &left,
            &right,
            (cam_w, cam_h),
            &calib,
            &config,
            yaw,
            pitch,
            false,
        )
        .expect("cpu stitch");
        Agreement::compare(&gpu_rgba, &cpu_rgba).assert_within(AgreementBounds::DEFAULT, "nv12");
    }

    /// Synthetic YUV420p: horizontal luma gradient + mild chroma gradients, so
    /// the separate-plane U/V sampler is exercised (not just flat chroma).
    fn yuv420(w: u32, h: u32, bias: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0..w * h)
            .map(|i| ((i % w) as usize * 255 / w as usize) as u8)
            .map(|val| val.saturating_add(bias))
            .collect();
        let (cw, ch2) = (w / 2, h / 2);
        let u: Vec<u8> = (0..cw * ch2)
            .map(|i| ((i % cw) as usize * 255 / cw as usize) as u8)
            .collect();
        let v: Vec<u8> = (0..cw * ch2)
            .map(|i| ((i / cw) as usize * 255 / ch2 as usize) as u8)
            .collect();
        (y, u, v)
    }

    /// YUV420p planar input must agree with the GPU's YUV420p path too,
    /// validating the separate-plane chroma sampler against the shader.
    #[test]
    fn cpu_yuv420p_matches_gpu_within_tolerance() {
        use crate::render::planes::YuvPlanes;
        use crate::render::renderer::InputFormat;

        let Some(gpu) = gpu_or_skip() else {
            return;
        };

        let (cam_w, cam_h) = (256u32, 144u32);
        let (out_w, out_h) = (192u32, 108u32);
        let calib = calib(cam_w, cam_h);
        let config = ViewportConfig {
            width: out_w,
            height: out_h,
            ..Default::default()
        };
        let (ly, lu, lv) = yuv420(cam_w, cam_h, 0);
        let (ry, ru, rv) = yuv420(cam_w, cam_h, 30);
        let left = YuvPlanes {
            y: &ly,
            u: &lu,
            v: &lv,
        };
        let right = YuvPlanes {
            y: &ry,
            u: &ru,
            v: &rv,
        };
        let (yaw, pitch) = (0.10f32, -0.05f32);

        let pipeline = crate::render::pipeline::StitchPipeline::with_gpu(
            gpu,
            calib.clone(),
            config.clone(),
            cam_w,
            cam_h,
            wgpu::TextureFormat::Rgba8Unorm,
            InputFormat::Yuv420p,
        )
        .expect("pipeline");
        let mut readback =
            crate::gpu::rgba_readback::RgbaReadback::new(pipeline.gpu(), out_w, out_h)
                .expect("readback");
        let mut gpu_rgba: Option<Vec<u8>> = None;
        for _ in 0..3 {
            let cmd = pipeline
                .render_to_target(&left, &right, yaw, pitch)
                .expect("render");
            let tex = pipeline.render_target();
            if let Some(bytes) = readback
                .readback(pipeline.gpu(), tex, cmd)
                .expect("readback")
            {
                gpu_rgba = Some(bytes.to_vec());
            }
        }
        let gpu_rgba = gpu_rgba.expect("gpu should produce a frame after 3 renders");

        let cpu_rgba = stitch_l_shape_rgba_yuv420p(
            &left,
            &right,
            (cam_w, cam_h),
            &calib,
            &config,
            yaw,
            pitch,
            false,
        )
        .expect("cpu stitch yuv420p");
        Agreement::compare(&gpu_rgba, &cpu_rgba).assert_within(AgreementBounds::DEFAULT, "yuv420p");
    }

    /// Textured NV12 (rippled luma + gradient interleaved chroma) - non-flat so
    /// geometry divergences surface as large diffs, not LSB noise.
    fn textured_nv12(w: u32, h: u32, phase: f64) -> (Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; (w * h) as usize];
        for j in 0..h {
            for i in 0..w {
                let g = 128.0
                    + 80.0 * ((i as f64) * 0.05 + phase).sin()
                    + 40.0 * ((j as f64) * 0.04).cos();
                y[(j * w + i) as usize] = g.clamp(0.0, 255.0) as u8;
            }
        }
        let mut uv = vec![128u8; (w * (h / 2)) as usize];
        for j in 0..h / 2 {
            for i in (0..w).step_by(2) {
                uv[(j * w + i) as usize] = (i * 255 / w) as u8;
                uv[(j * w + i + 1) as usize] = (j * 255 / (h / 2)) as u8;
            }
        }
        (y, uv)
    }

    /// High-frequency NV12 (fine luma + chroma checker) - "locks" the sampler:
    /// a half-texel / coordinate-convention error is invisible on smooth content
    /// (bilinear reproduces a gradient near-exactly) but shows as a large diff
    /// on a checker.
    fn checker_nv12(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; (w * h) as usize];
        for j in 0..h {
            for i in 0..w {
                let on = ((i / 3) + (j / 3)) % 2 == 0;
                y[(j * w + i) as usize] = if on { 220 } else { 30 };
            }
        }
        let mut uv = vec![128u8; (w * (h / 2)) as usize];
        for j in 0..h / 2 {
            for i in (0..w).step_by(2) {
                let on = ((i / 4) + (j / 4)) % 2 == 0;
                uv[(j * w + i) as usize] = if on { 200 } else { 60 };
                uv[(j * w + i + 1) as usize] = if on { 60 } else { 200 };
            }
        }
        (y, uv)
    }

    /// Render a scene on both the GPU and the CPU and return `(gpu_rgba,
    /// cpu_rgba)`, or `None` if there is no GPU adapter. The shared agreement
    /// primitive: callers diff the buffers however they need (aggregate stats,
    /// coverage/black-region masks).
    fn gpu_cpu_rgba(
        calib: &MatchCalibration,
        config: &ViewportConfig,
        cam: (u32, u32),
        left: &Nv12Planes,
        right: &Nv12Planes,
        yaw: f32,
        pitch: f32,
        full_range: bool,
    ) -> Option<(Vec<u8>, Vec<u8>)> {
        use crate::render::renderer::InputFormat;
        let gpu = gpu_or_skip()?;
        let (cam_w, cam_h) = cam;
        let mut pipeline = crate::render::pipeline::StitchPipeline::with_gpu(
            gpu,
            calib.clone(),
            config.clone(),
            cam_w,
            cam_h,
            wgpu::TextureFormat::Rgba8Unorm,
            InputFormat::Nv12,
        )
        .expect("pipeline");
        pipeline.set_full_range(full_range);
        let mut readback = crate::gpu::rgba_readback::RgbaReadback::new(
            pipeline.gpu(),
            config.width,
            config.height,
        )
        .expect("readback");
        let mut gpu_rgba = None;
        for _ in 0..3 {
            let cmd = pipeline
                .render_to_target_nv12(left, right, yaw, pitch)
                .expect("render");
            let tex = pipeline.render_target();
            if let Some(b) = readback
                .readback(pipeline.gpu(), tex, cmd)
                .expect("readback")
            {
                gpu_rgba = Some(b.to_vec());
            }
        }
        let gpu_rgba = gpu_rgba.expect("gpu frame");
        let cpu_rgba =
            stitch_l_shape_rgba(left, right, cam, calib, config, yaw, pitch, full_range).unwrap();
        Some((gpu_rgba, cpu_rgba))
    }

    /// Regression guard for the regimes that masked the original behind-camera
    /// and quad-footprint divergences: wide pan, wide FOV, off-center principal
    /// point, full-range, and partial lens correction. Each must agree with the
    /// GPU (a geometry divergence blows mean and the >16 fraction far past these
    /// bounds - the original bugs gave 35-68% wrong pixels here).
    #[test]
    fn cpu_matches_gpu_across_regimes() {
        let (cam_w, cam_h) = (256u32, 144u32);
        let cfg = |w: u32, h: u32, fov: f32, corr: f32| ViewportConfig {
            width: w,
            height: h,
            fov_degrees: fov,
            lens_correction_amount: corr,
            ..Default::default()
        };
        let calib_cx = |cx_off: f64| {
            let mut c = calib(cam_w, cam_h);
            c.left.cx = cam_w as f64 * (0.5 + cx_off);
            c.right.cx = cam_w as f64 * (0.5 + cx_off);
            c
        };
        let base = calib(cam_w, cam_h);
        let (ly, luv) = textured_nv12(cam_w, cam_h, 0.0);
        let (ry, ruv) = textured_nv12(cam_w, cam_h, 1.3);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };
        let cases: [(&str, MatchCalibration, ViewportConfig, f32, f32, bool); 6] = [
            (
                "wide-pan",
                base.clone(),
                cfg(192, 108, 75.0, 1.0),
                0.9,
                0.0,
                false,
            ),
            (
                "wide-fov",
                base.clone(),
                cfg(192, 108, 130.0, 1.0),
                0.0,
                0.0,
                false,
            ),
            (
                "off-center-cx",
                calib_cx(0.12),
                cfg(192, 108, 75.0, 1.0),
                0.2,
                0.0,
                false,
            ),
            (
                "full-range",
                base.clone(),
                cfg(192, 108, 75.0, 1.0),
                0.1,
                -0.05,
                true,
            ),
            (
                "lens-corr-0.5",
                base.clone(),
                cfg(192, 108, 75.0, 0.5),
                0.1,
                -0.05,
                false,
            ),
            (
                "non-16:9 (4:3 out)",
                base.clone(),
                cfg(144, 108, 75.0, 1.0),
                0.1,
                -0.05,
                false,
            ),
        ];
        let mut ran = false;
        for (label, calib, config, yaw, pitch, fr) in cases {
            let Some((g, c)) = gpu_cpu_rgba(
                &calib,
                &config,
                (cam_w, cam_h),
                &left,
                &right,
                yaw,
                pitch,
                fr,
            ) else {
                eprintln!("skipping regimes: no GPU adapter");
                return;
            };
            ran = true;
            Agreement::compare(&g, &c).assert_within(AgreementBounds::DEFAULT, label);
        }
        assert!(ran, "regimes test did not run");
    }

    /// The GPU rasterizer depth-clips fragments at NEAR_PLANE; the CPU inverse
    /// map must apply the same `0 <= clip_z <= clip_w` test. A tiny
    /// `camera_axis_offset` (or a FOV near the 179 singularity) places a plane
    /// within the near plane, so without the clip the CPU gathers a near band
    /// the GPU discards - a large coverage divergence (mean in the hundreds,
    /// most channels off by >16). Regression guard for the near-plane clip.
    #[test]
    fn cpu_matches_gpu_near_plane_clip() {
        let (cam_w, cam_h) = (256u32, 144u32);
        let (ly, luv) = textured_nv12(cam_w, cam_h, 0.0);
        let (ry, ruv) = textured_nv12(cam_w, cam_h, 1.3);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };
        let axis = |off: f64| {
            let mut c = calib(cam_w, cam_h);
            c.layout.camera_axis_offset = off;
            c
        };
        let cfg = |fov: f32| ViewportConfig {
            width: 192,
            height: 108,
            fov_degrees: fov,
            ..Default::default()
        };
        // Each case puts a plane within NEAR_PLANE of the virtual camera before
        // the fix: axis offset below ~0.012, or FOV near the projection limit.
        let cases: [(&str, MatchCalibration, ViewportConfig); 3] = [
            ("axis-offset 0.005", axis(0.005), cfg(75.0)),
            ("axis-offset 0.012", axis(0.012), cfg(75.0)),
            ("fov 178", calib(cam_w, cam_h), cfg(178.0)),
        ];
        let mut ran = false;
        for (label, cal, config) in cases {
            let Some((g, c)) = gpu_cpu_rgba(
                &cal,
                &config,
                (cam_w, cam_h),
                &left,
                &right,
                0.0,
                0.0,
                false,
            ) else {
                eprintln!("skipping near-plane: no GPU adapter");
                return;
            };
            ran = true;
            Agreement::compare(&g, &c).assert_within(AgreementBounds::DEFAULT, label);
        }
        assert!(ran, "near-plane test did not run");
    }

    /// High-frequency content locks the sampler: smooth gradients hide a
    /// half-texel / coordinate-convention error (bilinear of a ramp is
    /// near-exact), a checker exposes it. Looser bounds than the smooth regimes
    /// because sharp edges produce isolated f32/f64 + texture-filter-precision
    /// spikes; a real convention bug would blow the mean into the tens.
    #[test]
    fn cpu_matches_gpu_high_frequency() {
        let (cam_w, cam_h) = (256u32, 144u32);
        let cal = calib(cam_w, cam_h);
        let config = ViewportConfig {
            width: 192,
            height: 108,
            ..Default::default()
        };
        let (cy, cuv) = checker_nv12(cam_w, cam_h);
        let left = Nv12Planes { y: &cy, uv: &cuv };
        let right = Nv12Planes { y: &cy, uv: &cuv };
        let Some((g, c)) = gpu_cpu_rgba(
            &cal,
            &config,
            (cam_w, cam_h),
            &left,
            &right,
            0.05,
            -0.03,
            false,
        ) else {
            return;
        };
        Agreement::compare(&g, &c).assert_within(AgreementBounds::DEFAULT, "high-freq");
    }

    /// Consistency: where the GPU leaves the cleared-black background (no plane
    /// covers the pixel), the CPU must also be black - it must NOT "fill in" the
    /// uncovered region with extended plane content. A wide FOV leaves a real
    /// black margin around the L-shape planes. (Edge-extend past the plane is a
    /// deliberate future opt-in for *both* backends, not an accidental CPU-only
    /// divergence.)
    #[test]
    fn cpu_black_region_matches_gpu() {
        let (cam_w, cam_h) = (256u32, 144u32);
        let cal = calib(cam_w, cam_h);
        let config = ViewportConfig {
            width: 192,
            height: 108,
            fov_degrees: 140.0,
            ..Default::default()
        };
        let (ly, luv) = textured_nv12(cam_w, cam_h, 0.0);
        let (ry, ruv) = textured_nv12(cam_w, cam_h, 1.3);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };
        let Some((g, c)) = gpu_cpu_rgba(
            &cal,
            &config,
            (cam_w, cam_h),
            &left,
            &right,
            0.0,
            0.0,
            false,
        ) else {
            return;
        };
        let (mut gpu_black, mut leak) = (0u32, 0u32);
        for (gp, cp) in g.chunks_exact(4).zip(c.chunks_exact(4)) {
            if gp[0] == 0 && gp[1] == 0 && gp[2] == 0 {
                gpu_black += 1;
                // > 8 ignores 1-LSB near-black blend noise at the boundary.
                if cp[0] > 8 || cp[1] > 8 || cp[2] > 8 {
                    leak += 1;
                }
            }
        }
        eprintln!("black-region: gpu_black={gpu_black} cpu-leak={leak}");
        assert!(gpu_black > 0, "expected a black margin at fov=140");
        let leak_pct = 100.0 * leak as f64 / gpu_black as f64;
        assert!(
            leak_pct < 1.0,
            "{leak_pct}% of GPU-black pixels are non-black on CPU (coverage divergence?)"
        );
    }

    /// Trustworthiness self-test for the agreement oracle: it must FAIL on a
    /// real geometry error, not merely pass on a correct one. The correct CPU
    /// stitch agrees with the GPU within [`AgreementBounds::DEFAULT`]; the same
    /// stitch with the virtual pan nudged by ~1px of output must NOT - its mean
    /// blows well past the bound. Without this guard a too-loose tolerance could
    /// silently accept a sub-pixel geometry regression (the class of bug the
    /// near-plane clip divergence was). Uses the offset-sensitive ramp scene; a
    /// flat scene would hide the misregistration.
    #[test]
    fn agreement_oracle_detects_subpixel_offset() {
        let (cam_w, cam_h) = (256u32, 144u32);
        let cal = calib(cam_w, cam_h);
        let config = ViewportConfig {
            width: 192,
            height: 108,
            ..Default::default()
        };
        let (ly, luv) = textured_nv12(cam_w, cam_h, 0.0);
        let (ry, ruv) = textured_nv12(cam_w, cam_h, 1.3);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };
        let (yaw, pitch) = (0.10f32, -0.05f32);
        let Some((gpu_rgba, cpu_rgba)) = gpu_cpu_rgba(
            &cal,
            &config,
            (cam_w, cam_h),
            &left,
            &right,
            yaw,
            pitch,
            false,
        ) else {
            return;
        };

        // The correct CPU stitch must pass the tightened bound.
        let good = Agreement::compare(&gpu_rgba, &cpu_rgba);
        good.assert_within(AgreementBounds::DEFAULT, "probe-correct");

        // ~1px of output (fov 75 over 192px ~= 0.007 rad/px); 0.01 rad is decisive.
        let perturbed = stitch_l_shape_rgba(
            &left,
            &right,
            (cam_w, cam_h),
            &cal,
            &config,
            yaw + 0.01,
            pitch,
            false,
        )
        .expect("cpu stitch");
        let bad = Agreement::compare(&gpu_rgba, &perturbed);
        eprintln!(
            "injection probe: correct mean={:.3}, +1px mean={:.3}",
            good.mean, bad.mean
        );
        assert!(
            bad.mean > AgreementBounds::DEFAULT.max_mean,
            "oracle not trustworthy: a ~1px geometry offset gave mean {:.3}, still within the {:.3} bound",
            bad.mean,
            AgreementBounds::DEFAULT.max_mean
        );
    }
}
