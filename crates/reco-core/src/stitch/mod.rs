//! Stitch executors (L2): the CPU software path and the GPU pipeline owner.
//!
//! `GpuExecutor` (with the `gpu` feature) owns the wgpu render pipeline;
//! [`StitchCore`](crate::core::StitchCore) embeds one as its render
//! substrate. The rest of this module is the pure-Rust, no-wgpu software
//! stitcher that mirrors the GPU fisheye shader (`shaders/fisheye.wgsl`,
//! the fragment stage in [`crate::render`]) per output pixel. The CPU
//! path serves two roles:
//!
//! 1. **Correctness oracle** for the GPU path. The geometry reuses
//!    `crate::lens::kb4` and the same view/projection matrices as
//!    [`crate::render`]. CPU/GPU agreement is checked by the stitch oracle tests.
//! 2. **Rendering path for GPU-less targets** (edge SoCs, cloud CPU workers)
//!    where wgpu is unavailable or uneconomical.
//!
//! ## Projection geometry
//!
//! `SurfaceMap` is the only projection-specific piece: it maps an output
//! pixel to a source-camera UV. The two-plane L-shape provides one
//! `PlaneMap` per camera; cylinder provides one map. The CPU gather loop composites the ordered surface list returned by calibration. External compositors can use the same maps and blend weights; `tests/calibrated_mapping.rs` checks that public path against CPU rendering for both shapes.

mod cpu;
mod executor;
#[cfg(all(feature = "gpu", target_os = "linux"))]
mod nvmm_completion;
#[cfg(feature = "gpu")]
pub(crate) mod residency;

pub use executor::{CpuExecutor, Executor, StitchError, StitchExecutor};
#[cfg(feature = "gpu")]
pub use executor::{GpuExecutor, GpuExecutorConfig};

/// How one projection surface composites over earlier covered surfaces.
///
/// Paired with a [`SurfaceMap`] in the ordered surface list a projection
/// emits: the first valid surface at a pixel is opaque, and later valid
/// surfaces blend over it per their [`BlendRule`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendRule {
    /// Fully replaces whatever is underneath wherever this surface covers.
    Opaque,
    /// Fades in over `[0, width]` of the surface's [`SurfaceUv::edge`]
    /// coordinate (the GPU's seam smoothstep) when an earlier surface already
    /// covers the pixel. A non-positive width renders fully opaque wherever
    /// the surface covers.
    Smoothstep(f64),
}

impl BlendRule {
    /// Nominal blend factor for a covered pixel composited over an earlier
    /// covered surface, using the finite [`SurfaceUv::edge`] coordinate
    /// returned by its map.
    ///
    /// When no earlier surface covers the pixel, composite the first valid
    /// surface opaquely (`1.0`); subsequent valid surfaces composite as
    /// `previous + (source - previous) * alpha` in surface order. See
    /// [`Self::composite_alpha`] for the combined weight.
    pub fn alpha(&self, edge: f64) -> f64 {
        match *self {
            BlendRule::Opaque => 1.0,
            BlendRule::Smoothstep(width) if width > 0.0 => smoothstep(0.0, width, edge),
            BlendRule::Smoothstep(_) => 1.0,
        }
    }

    /// Effective blend factor in surface order: `1.0` when no earlier surface
    /// covers the pixel (`!has_base`), otherwise [`Self::alpha`].
    pub fn composite_alpha(&self, edge: f64, has_base: bool) -> f64 {
        if has_base { self.alpha(edge) } else { 1.0 }
    }
}

/// GLSL-compatible smoothstep (SYNC_WITH `fisheye.wgsl`'s seam blend).
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A source-camera sample location produced by a [`SurfaceMap`].
#[derive(Debug, Clone, Copy)]
pub struct SurfaceUv {
    /// Normalised camera UV in `[0, 1]`. Multiply by the frame dimensions for
    /// pixel coordinates.
    pub u: f64,
    /// Normalised camera UV in `[0, 1]`.
    pub v: f64,
    /// Seam coordinate used by the compositor's blend rule when blending over
    /// an earlier covered surface (normally extended plane UV x).
    pub edge: f64,
}

/// The projection seam: a per-output-pixel inverse map for one source surface.
///
/// Implemented once per projection. The L-shape supplies two (one per camera
/// plane); the CPU gather loop queries each surface and composites the ones
/// that cover a pixel. Maps are obtained through [`Calibration::surface_maps`](crate::calibration::Calibration::surface_maps). It is the CPU dual
/// of the GPU rasterizer's per-fragment plane-UV interpolation.
pub trait SurfaceMap: Send + Sync {
    /// Map a view-plane point in normalized device coordinates to source-camera UV, or `None` outside this surface. The viewport is `[-1, 1]` on each axis, with positive Y pointing up. Positions between pixel centres and beyond the viewport follow the same mapping.
    fn sample_uv(&self, ndc: [f64; 2]) -> Option<SurfaceUv>;

    /// Conservatively certify source support throughout a continuous NDC rectangle. `rotation_bound` additionally reserves depth-clipping room for rotations up to that angle in radians about the same eye. Returning `false` means unknown; blending is excluded.
    fn covers_ndc(&self, min: [f64; 2], max: [f64; 2], rotation_bound: f64) -> bool;
}

/// Shared test fixtures + GPU acquisition for the stitch test modules.
#[cfg(test)]
pub(crate) mod test_support {
    // The agreement types + pattern generators only run under the gpu
    // feature (they gate GPU-vs-CPU comparisons); keep them compiled
    // either way so the fixtures stay in sync.
    #![cfg_attr(not(feature = "gpu"), allow(dead_code))]

    use crate::calibration::{Calibration, Framing, Lens};
    #[cfg(feature = "gpu")]
    use crate::gpu::{GpuContext, GpuError};
    use crate::projection::LShape;

    /// Two-camera calibration (shared dims, mild fisheye, centred) for tests.
    pub fn calib(w: u32, h: u32) -> Calibration {
        let cam = || {
            Lens::fisheye(
                w,
                h,
                w as f64 * 0.5,
                w as f64 * 0.5,
                w as f64 * 0.5,
                h as f64 * 0.5,
                [-0.02, 0.004, 0.0, 0.0],
            )
        };
        Calibration::new(
            vec![cam(), cam()],
            LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            Framing {
                axis_offset: 0.25,
                tilt: 0.0,
                roll: 0.0,
            },
        )
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
    #[cfg(feature = "gpu")]
    pub fn gpu_or_skip() -> Option<GpuContext> {
        gpu_result_or_skip(GpuContext::new_blocking())
    }

    /// Apply the same adapter availability rule to custom test contexts.
    #[cfg(feature = "gpu")]
    pub fn gpu_result_or_skip(result: Result<GpuContext, GpuError>) -> Option<GpuContext> {
        match result {
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
    #![cfg_attr(not(feature = "gpu"), allow(dead_code))]

    use super::cpu::stitch_rgba;
    #[cfg(feature = "gpu")]
    use super::cpu::stitch_rgba_yuv420p;
    #[cfg(feature = "gpu")]
    use super::test_support::gpu_or_skip;
    #[cfg(feature = "gpu")]
    use super::test_support::{Agreement, AgreementBounds};
    use super::test_support::{calib, nv12};
    use crate::calibration::Calibration;
    use crate::geometry::RenderPose;
    use crate::render::planes::Nv12Planes;
    use crate::render::viewport::ViewportSize;

    #[test]
    fn output_dimensions_and_opaque_alpha() {
        let (w, h) = (96u32, 54u32);
        let calib = calib(w, h);
        let cfg = ViewportSize {
            width: w,
            height: h,
        };
        let (ly, luv) = nv12(w, h, 0);
        let (ry, ruv) = nv12(w, h, 40);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };

        let out = stitch_rgba(
            &[left, right],
            &[(w, h); 2],
            &calib,
            &cfg,
            RenderPose::default(),
            Default::default(),
        )
        .unwrap();
        assert_eq!(out.len(), (w * h * 4) as usize);
        // Alpha channel is fully opaque.
        assert!(out.iter().skip(3).step_by(4).all(|&a| a == 255));
    }

    #[test]
    fn produces_covered_pixels_and_is_deterministic() {
        let (w, h) = (96u32, 54u32);
        let calib = calib(w, h);
        let cfg = ViewportSize {
            width: w,
            height: h,
        };
        let (ly, luv) = nv12(w, h, 0);
        let (ry, ruv) = nv12(w, h, 40);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };

        let a = stitch_rgba(
            &[left, right],
            &[(w, h); 2],
            &calib,
            &cfg,
            RenderPose::default(),
            Default::default(),
        )
        .unwrap();
        let b = stitch_rgba(
            &[left, right],
            &[(w, h); 2],
            &calib,
            &cfg,
            RenderPose::default(),
            Default::default(),
        )
        .unwrap();
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

    /// The CPU float reference must agree with the GPU shader on
    /// the same scene. Because both share `view_matrix`, the projection, and
    /// `lens::kb4`, the only differences are f32-vs-f64 and hardware-vs-software
    /// bilinear - so the RGB match should be tight. Skips when no GPU adapter.
    #[test]
    #[cfg(feature = "gpu")]
    fn cpu_matches_gpu_within_tolerance() {
        use crate::render::renderer::InputFormat;

        let Some(gpu) = gpu_or_skip() else {
            return;
        };

        let (cam_w, cam_h) = (256u32, 144u32);
        let (out_w, out_h) = (192u32, 108u32);
        let calib = calib(cam_w, cam_h);
        let viewport_size = ViewportSize {
            width: out_w,
            height: out_h,
        };
        let (ly, luv) = nv12(cam_w, cam_h, 0);
        let (ry, ruv) = nv12(cam_w, cam_h, 30);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };
        let pose = RenderPose {
            yaw: 0.10,
            pitch: -0.05,
            ..Default::default()
        };

        // GPU render -> RGBA. The readback is triple-buffered (N-2 latency), so
        // render the same frame three times to drain one result.
        let mut pipeline = crate::render::pipeline::StitchPipeline::with_gpu(
            gpu,
            &calib
                .projection
                .geometry()
                .gpu_program()
                .expect("L-shape supplies a GPU program"),
            calib.clone(),
            viewport_size.clone(),
            vec![(cam_w, cam_h); 2],
            wgpu::TextureFormat::Rgba8Unorm,
            InputFormat::Nv12,
        )
        .expect("pipeline");
        let mut readback =
            crate::gpu::rgba_readback::RgbaReadback::new(pipeline.gpu(), out_w, out_h)
                .expect("readback");
        let mut gpu_rgba: Option<Vec<u8>> = None;
        for _ in 0..3 {
            let cmd = pipeline.render_nv12(&[left, right], pose).expect("render");
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
        let cpu_rgba = stitch_rgba(
            &[left, right],
            &[(cam_w, cam_h); 2],
            &calib,
            &viewport_size,
            pose,
            Default::default(),
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
    #[cfg(feature = "gpu")]
    fn cpu_yuv420p_matches_gpu_within_tolerance() {
        use crate::render::planes::Yuv420pPlanes;
        use crate::render::renderer::InputFormat;

        let Some(gpu) = gpu_or_skip() else {
            return;
        };

        let (cam_w, cam_h) = (256u32, 144u32);
        let (out_w, out_h) = (192u32, 108u32);
        let calib = calib(cam_w, cam_h);
        let viewport_size = ViewportSize {
            width: out_w,
            height: out_h,
        };
        let (ly, lu, lv) = yuv420(cam_w, cam_h, 0);
        let (ry, ru, rv) = yuv420(cam_w, cam_h, 30);
        let left = Yuv420pPlanes {
            y: &ly,
            u: &lu,
            v: &lv,
        };
        let right = Yuv420pPlanes {
            y: &ry,
            u: &ru,
            v: &rv,
        };
        let pose = RenderPose {
            yaw: 0.10,
            pitch: -0.05,
            ..Default::default()
        };

        let mut pipeline = crate::render::pipeline::StitchPipeline::with_gpu(
            gpu,
            &calib
                .projection
                .geometry()
                .gpu_program()
                .expect("L-shape supplies a GPU program"),
            calib.clone(),
            viewport_size.clone(),
            vec![(cam_w, cam_h); 2],
            wgpu::TextureFormat::Rgba8Unorm,
            InputFormat::Yuv420p,
        )
        .expect("pipeline");
        let mut readback =
            crate::gpu::rgba_readback::RgbaReadback::new(pipeline.gpu(), out_w, out_h)
                .expect("readback");
        let frames = crate::source::FrameSet::Yuv420p(
            [left, right]
                .iter()
                .map(|plane| {
                    crate::source::Yuv420pData {
                        size: (cam_w, cam_h),
                        y: plane.y.to_vec(),
                        u: plane.u.to_vec(),
                        v: plane.v.to_vec(),
                    }
                    .into()
                })
                .collect(),
        );
        let mut gpu_rgba: Option<Vec<u8>> = None;
        for _ in 0..3 {
            let cmd = pipeline.render_frame_set(&frames, pose).expect("render");
            let tex = pipeline.render_target();
            if let Some(bytes) = readback
                .readback(pipeline.gpu(), tex, cmd)
                .expect("readback")
            {
                gpu_rgba = Some(bytes.to_vec());
            }
        }
        let gpu_rgba = gpu_rgba.expect("gpu should produce a frame after 3 renders");

        let cpu_rgba = stitch_rgba_yuv420p(
            &[left, right],
            &[(cam_w, cam_h); 2],
            &calib,
            &viewport_size,
            pose,
            Default::default(),
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
    #[cfg(feature = "gpu")]
    fn gpu_cpu_rgba(
        calib: &Calibration,
        viewport_size: &ViewportSize,
        cam: (u32, u32),
        left: &Nv12Planes,
        right: &Nv12Planes,
        pose: RenderPose,
        full_range: bool,
    ) -> Option<(Vec<u8>, Vec<u8>)> {
        use crate::render::renderer::InputFormat;
        let gpu = gpu_or_skip()?;
        let (cam_w, cam_h) = cam;
        let mut pipeline = crate::render::pipeline::StitchPipeline::with_gpu(
            gpu,
            &calib
                .projection
                .geometry()
                .gpu_program()
                .expect("L-shape supplies a GPU program"),
            calib.clone(),
            viewport_size.clone(),
            vec![(cam_w, cam_h); 2],
            wgpu::TextureFormat::Rgba8Unorm,
            InputFormat::Nv12,
        )
        .expect("pipeline");
        pipeline.set_full_range(full_range);
        let mut readback = crate::gpu::rgba_readback::RgbaReadback::new(
            pipeline.gpu(),
            viewport_size.width,
            viewport_size.height,
        )
        .expect("readback");
        let mut gpu_rgba = None;
        for _ in 0..3 {
            let cmd = pipeline
                .render_nv12(&[*left, *right], pose)
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
        let cpu_rgba = stitch_rgba(
            &[*left, *right],
            &[cam; 2],
            calib,
            viewport_size,
            pose,
            crate::render::color::YuvColor {
                full_range,
                ..Default::default()
            },
        )
        .unwrap();
        Some((gpu_rgba, cpu_rgba))
    }

    /// Regression guard for the regimes that masked the original behind-camera
    /// and quad-footprint divergences: wide pan, wide FOV, off-center principal
    /// point, full-range, and partial lens correction. Each must agree with the
    /// GPU (a geometry divergence blows mean and the >16 fraction far past these
    /// bounds - the original bugs gave 35-68% wrong pixels here).
    #[test]
    #[cfg(feature = "gpu")]
    fn cpu_matches_gpu_across_regimes() {
        let (cam_w, cam_h) = (256u32, 144u32);
        let cfg = |w: u32, h: u32| ViewportSize {
            width: w,
            height: h,
        };
        let pose = |yaw: f32, pitch: f32, fov: f32| RenderPose {
            yaw,
            pitch,
            fov_degrees: fov,
        };
        // Lens correction is now per-lens on the calibration.
        let calib_corr = |c: f32| {
            let mut cal = calib(cam_w, cam_h);
            for lens in &mut cal.lenses {
                lens.correction = c;
            }
            cal
        };
        let calib_cx = |cx_off: f64| {
            let mut c = calib(cam_w, cam_h);
            c.lenses[0].cx = cam_w as f64 * (0.5 + cx_off);
            c.lenses[1].cx = cam_w as f64 * (0.5 + cx_off);
            c
        };
        // Non-zero rig tilt+roll: exercises the view_matrix basis rotation
        // both executors apply - the headline roll-aware render path, which
        // the other regimes (all tilt=roll=0) never touch.
        let calib_tilt_roll = |tilt: f64, roll: f64| {
            let mut c = calib(cam_w, cam_h);
            c.framing.tilt = tilt;
            c.framing.roll = roll;
            c
        };
        let mixed_aspect = |camera: usize| {
            let mut c = calib_tilt_roll(0.26, -0.12);
            c.lenses[camera] = c.lenses[camera].rescaled(cam_w, 192).unwrap();
            c.lenses[camera].cx *= 0.9;
            c.lenses[camera].correction = 0.4;
            c
        };
        let mixed_rotated = |camera: usize, sign: f64| {
            let mut c = mixed_aspect(camera);
            let shape = c.projection.l_shape_mut().unwrap();
            shape.x_rx = sign * 0.12;
            shape.z_rz = sign * 0.1;
            shape.x_rz = sign * 0.08;
            shape.z_rx = -sign * 0.06;
            c
        };
        let base = calib(cam_w, cam_h);
        let (ly, luv) = textured_nv12(cam_w, cam_h, 0.0);
        let (ry, ruv) = textured_nv12(cam_w, cam_h, 1.3);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };
        let cases: [(&str, Calibration, ViewportSize, RenderPose, bool); 12] = [
            (
                "wide-pan",
                base.clone(),
                cfg(192, 108),
                pose(0.9, 0.0, 75.0),
                false,
            ),
            (
                "wide-fov",
                base.clone(),
                cfg(192, 108),
                pose(0.0, 0.0, 130.0),
                false,
            ),
            (
                "off-center-cx",
                calib_cx(0.12),
                cfg(192, 108),
                pose(0.2, 0.0, 75.0),
                false,
            ),
            (
                "full-range",
                base.clone(),
                cfg(192, 108),
                pose(0.1, -0.05, 75.0),
                true,
            ),
            (
                "lens-corr-0",
                calib_corr(0.0),
                cfg(192, 108),
                pose(0.1, -0.05, 75.0),
                false,
            ),
            (
                "lens-corr-0.5",
                calib_corr(0.5),
                cfg(192, 108),
                pose(0.1, -0.05, 75.0),
                false,
            ),
            (
                "non-16:9 (4:3 out)",
                base.clone(),
                cfg(144, 108),
                pose(0.1, -0.05, 75.0),
                false,
            ),
            (
                "tilt+roll-at-yaw",
                calib_tilt_roll(0.26, 0.12),
                cfg(192, 108),
                pose(0.6, 0.0, 75.0),
                false,
            ),
            (
                "mixed-aspect-camera-0",
                mixed_aspect(0),
                cfg(192, 108),
                pose(-0.6, 0.12, 95.0),
                false,
            ),
            (
                "mixed-aspect-camera-1",
                mixed_aspect(1),
                cfg(144, 108),
                pose(0.6, -0.12, 95.0),
                false,
            ),
            (
                "mixed-aspect-rotated-positive",
                mixed_rotated(0, 1.0),
                cfg(192, 108),
                pose(-0.6, 0.12, 95.0),
                false,
            ),
            (
                "mixed-aspect-rotated-negative",
                mixed_rotated(1, -1.0),
                cfg(144, 108),
                pose(0.6, -0.12, 95.0),
                false,
            ),
        ];
        let mut ran = false;
        for (label, calib, viewport_size, case_pose, fr) in cases {
            let Some((g, c)) = gpu_cpu_rgba(
                &calib,
                &viewport_size,
                (cam_w, cam_h),
                &left,
                &right,
                case_pose,
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
    #[cfg(feature = "gpu")]
    fn cpu_matches_gpu_near_plane_clip() {
        let (cam_w, cam_h) = (256u32, 144u32);
        let (ly, luv) = textured_nv12(cam_w, cam_h, 0.0);
        let (ry, ruv) = textured_nv12(cam_w, cam_h, 1.3);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };
        let axis = |off: f64| {
            let mut c = calib(cam_w, cam_h);
            c.framing.axis_offset = off;
            c
        };
        let cfg = |fov: f32| {
            (
                ViewportSize {
                    width: 192,
                    height: 108,
                },
                RenderPose {
                    fov_degrees: fov,
                    ..Default::default()
                },
            )
        };
        // Each case puts a plane within NEAR_PLANE of the virtual camera before
        // the fix: axis offset below ~0.012, or FOV near the projection limit.
        let cases: [(&str, Calibration, (ViewportSize, RenderPose)); 3] = [
            ("axis-offset 0.005", axis(0.005), cfg(75.0)),
            ("axis-offset 0.012", axis(0.012), cfg(75.0)),
            ("fov 178", calib(cam_w, cam_h), cfg(178.0)),
        ];
        let mut ran = false;
        for (label, cal, (viewport_size, case_pose)) in cases {
            let Some((g, c)) = gpu_cpu_rgba(
                &cal,
                &viewport_size,
                (cam_w, cam_h),
                &left,
                &right,
                case_pose,
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
    #[cfg(feature = "gpu")]
    fn cpu_matches_gpu_high_frequency() {
        let (cam_w, cam_h) = (256u32, 144u32);
        let cal = calib(cam_w, cam_h);
        let viewport_size = ViewportSize {
            width: 192,
            height: 108,
        };
        let (cy, cuv) = checker_nv12(cam_w, cam_h);
        let left = Nv12Planes { y: &cy, uv: &cuv };
        let right = Nv12Planes { y: &cy, uv: &cuv };
        let Some((g, c)) = gpu_cpu_rgba(
            &cal,
            &viewport_size,
            (cam_w, cam_h),
            &left,
            &right,
            RenderPose {
                yaw: 0.05,
                pitch: -0.03,
                ..Default::default()
            },
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
    #[cfg(feature = "gpu")]
    fn cpu_black_region_matches_gpu() {
        let (cam_w, cam_h) = (256u32, 144u32);
        let cal = calib(cam_w, cam_h);
        let viewport_size = ViewportSize {
            width: 192,
            height: 108,
        };
        let pose = RenderPose {
            fov_degrees: 140.0,
            ..Default::default()
        };
        let (ly, luv) = textured_nv12(cam_w, cam_h, 0.0);
        let (ry, ruv) = textured_nv12(cam_w, cam_h, 1.3);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };
        let Some((g, c)) = gpu_cpu_rgba(
            &cal,
            &viewport_size,
            (cam_w, cam_h),
            &left,
            &right,
            pose,
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
    #[cfg(feature = "gpu")]
    fn agreement_oracle_detects_subpixel_offset() {
        let (cam_w, cam_h) = (256u32, 144u32);
        let cal = calib(cam_w, cam_h);
        let viewport_size = ViewportSize {
            width: 192,
            height: 108,
        };
        let (ly, luv) = textured_nv12(cam_w, cam_h, 0.0);
        let (ry, ruv) = textured_nv12(cam_w, cam_h, 1.3);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };
        let pose = RenderPose {
            yaw: 0.10,
            pitch: -0.05,
            ..Default::default()
        };
        let Some((gpu_rgba, cpu_rgba)) = gpu_cpu_rgba(
            &cal,
            &viewport_size,
            (cam_w, cam_h),
            &left,
            &right,
            pose,
            false,
        ) else {
            return;
        };

        // The correct CPU stitch must pass the tightened bound.
        let good = Agreement::compare(&gpu_rgba, &cpu_rgba);
        good.assert_within(AgreementBounds::DEFAULT, "probe-correct");

        // ~1px of output (fov 75 over 192px ~= 0.007 rad/px); 0.01 rad is decisive.
        let perturbed = stitch_rgba(
            &[left, right],
            &[(cam_w, cam_h); 2],
            &cal,
            &viewport_size,
            RenderPose {
                yaw: pose.yaw + 0.01,
                ..pose
            },
            Default::default(),
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

    /// Build a mixed-aspect, tilted and rolled L-shape calibration with
    /// blending enabled that exercises left-only, right-only (both zero and
    /// fractional nominal seam weight), blended-overlap, and uncovered pixels.
    fn mixed_aspect_tilted_blend_calib() -> (Calibration, [(u32, u32); 2], RenderPose) {
        use crate::calibration::{Framing, Lens};
        use crate::projection::LShape;

        let left_lens = Lens::fisheye(
            256,
            144,
            120.0,
            120.0,
            128.0,
            72.0,
            [-0.02, 0.004, 0.0, 0.0],
        );
        let mut right_lens = Lens::fisheye(
            256,
            192,
            120.0,
            120.0,
            125.0,
            93.0,
            [-0.02, 0.004, 0.0, 0.0],
        );
        right_lens.correction = 0.85;
        let cal = Calibration::new(
            vec![left_lens, right_lens],
            LShape {
                intersect: 0.5,
                x_ty: 0.04,
                x_rz: 0.06,
                z_rx: -0.05,
                x_rx: 0.07,
                z_rz: 0.05,
                blend_width: 0.15,
            },
            Framing {
                axis_offset: 0.25,
                tilt: 0.12,
                roll: -0.08,
            },
        );
        cal.validate().unwrap();
        let pose = RenderPose {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: 112.0,
        };
        (cal, [(256, 144), (256, 192)], pose)
    }

    /// Calibration where the left plane quad covers the right seam zone,
    /// but the left lens's shifted principal point / narrower FOV rejects
    /// pixels at the lens/source UV boundary (testing lens validity, not just
    /// quad footprint).
    fn lens_boundary_blend_calib() -> (Calibration, [(u32, u32); 2], RenderPose) {
        use crate::calibration::{Framing, Lens};
        use crate::projection::LShape;

        let left_lens = Lens::fisheye(256, 144, 240.0, 240.0, 80.0, 72.0, [0.15, 0.05, 0.0, 0.0]);
        let right_lens = Lens::fisheye(
            256,
            144,
            120.0,
            120.0,
            128.0,
            72.0,
            [-0.02, 0.004, 0.0, 0.0],
        );
        let cal = Calibration::new(
            vec![left_lens, right_lens],
            LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.15,
            },
            Framing {
                axis_offset: 0.25,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        cal.validate().unwrap();
        let pose = RenderPose {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: 95.0,
        };
        (cal, [(256, 144), (256, 144)], pose)
    }

    /// Calibration where the left plane is partially depth-clipped by
    /// `NEAR_PLANE` inside the right seam zone while the right plane remains
    /// valid (testing depth-clip validity in base-support evaluation).
    fn depth_clip_blend_calib() -> (Calibration, [(u32, u32); 2], RenderPose) {
        use crate::calibration::{Framing, Lens};
        use crate::projection::LShape;

        let left_lens = Lens::fisheye(
            256,
            144,
            120.0,
            120.0,
            128.0,
            72.0,
            [-0.02, 0.004, 0.0, 0.0],
        );
        let right_lens = left_lens.clone();
        let cal = Calibration::new(
            vec![left_lens, right_lens],
            LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.12,
                blend_width: 0.2,
            },
            Framing {
                axis_offset: 0.008,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        cal.validate().unwrap();
        let pose = RenderPose {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: 110.0,
        };
        (cal, [(256, 144), (256, 144)], pose)
    }

    fn solid_nv12(w: u32, h: u32, y_val: u8) -> (Vec<u8>, Vec<u8>) {
        (
            vec![y_val; (w * h) as usize],
            vec![128u8; (w * (h / 2)) as usize],
        )
    }

    #[cfg(feature = "gpu")]
    fn render_gpu_nv12(
        cal: &Calibration,
        viewport_size: &ViewportSize,
        source_sizes: [(u32, u32); 2],
        planes: &[Nv12Planes<'_>; 2],
        pose: RenderPose,
        full_range: bool,
    ) -> Option<Vec<u8>> {
        use crate::render::renderer::InputFormat;
        use crate::stitch::{GpuExecutor, GpuExecutorConfig, StitchExecutor};

        let gpu = gpu_or_skip()?;
        let mut config =
            GpuExecutorConfig::new(cal.clone(), source_sizes.to_vec(), InputFormat::Nv12);
        config.viewport_size = viewport_size.clone();
        config.full_range = full_range;
        let mut exec = GpuExecutor::new(gpu, config).expect("gpu executor");
        Some(exec.stitch(planes, pose).expect("gpu stitch"))
    }

    /// Render the combined GPU stitch together with isolated left-only and
    /// right-only GPU draws (using the identical `Renderer`, textures, scene,
    /// and pose) so tests can verify compositing against actual hardware
    /// rasterization without analytic reprojection slack.
    #[cfg(feature = "gpu")]
    fn render_gpu_nv12_with_isolated_oracles(
        cal: &Calibration,
        viewport_size: &ViewportSize,
        source_sizes: [(u32, u32); 2],
        planes: &[Nv12Planes<'_>; 2],
        pose: RenderPose,
        full_range: bool,
    ) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        use crate::gpu::rgba_readback::RgbaReadback;
        use crate::render::renderer::{InputFormat, Renderer};

        let gpu = gpu_or_skip()?;
        let shape = cal.projection.l_shape().expect("L-shape calibration");
        let scene = shape.scene(
            &cal.framing,
            [cal.lenses[0].aspect_ratio(), cal.lenses[1].aspect_ratio()],
        );
        let program = cal
            .projection
            .geometry()
            .gpu_program()
            .expect("L-shape supplies a GPU program");
        let mut renderer = Renderer::new(
            &gpu,
            &program,
            viewport_size.width,
            viewport_size.height,
            &source_sizes,
            wgpu::TextureFormat::Rgba8Unorm,
            InputFormat::Nv12,
        );
        renderer.set_full_range(full_range);
        renderer
            .upload_nv12(&gpu, 0, &planes[0])
            .expect("upload left");
        renderer
            .upload_nv12(&gpu, 1, &planes[1])
            .expect("upload right");

        let mut readback =
            RgbaReadback::new(&gpu, viewport_size.width, viewport_size.height).expect("readback");
        let mut capture = |cmd: wgpu::CommandBuffer| -> Vec<u8> {
            assert!(
                readback
                    .readback(&gpu, renderer.render_target(), cmd)
                    .expect("submit readback")
                    .is_none()
            );
            readback
                .flush_pending(&gpu)
                .expect("flush readback")
                .expect("pending frame")
                .to_vec()
        };

        let combined = capture(renderer.render_to_target(&gpu, Some(&scene), cal, pose));
        let left_only =
            capture(renderer.render_isolated_camera_to_target(&gpu, &scene, cal, pose, 0));
        let right_only =
            capture(renderer.render_isolated_camera_to_target(&gpu, &scene, cal, pose, 1));
        Some((combined, left_only, right_only))
    }

    #[cfg(feature = "gpu")]
    struct GpuOracleCounts {
        right_only_total: usize,
        right_only_fade: usize,
        overlap_fade: usize,
    }

    /// Compare a combined GPU render against isolated left-only and right-only
    /// GPU draws across every pixel of `viewport`.
    ///
    /// Asserts:
    /// - 0 pixels with non-opaque alpha (`a != 255`)
    /// - 0 uncovered pixels with non-black RGB
    /// - 0 left-only pixels differing from isolated left-only GPU draw
    /// - 0 right-only pixels differing from isolated right-only GPU draw by >1 LSB
    /// - 0 overlap pixels bypassing smoothstep blending
    #[cfg(feature = "gpu")]
    fn assert_gpu_matches_isolated_draws(
        label: &str,
        cal: &Calibration,
        viewport: &ViewportSize,
        pose: RenderPose,
        combined: &[u8],
        left_only: &[u8],
        right_only: &[u8],
    ) -> GpuOracleCounts {
        let maps = cal.surface_maps(viewport, pose);
        let mut non_opaque_alpha = 0usize;
        let mut uncovered_mismatch = 0usize;
        let mut left_only_mismatch = 0usize;
        let mut right_only_mismatch = 0usize;
        let mut right_only_darkened = 0usize;
        let mut overlap_mismatch = 0usize;
        let mut right_only_total = 0usize;
        let mut right_only_fade = 0usize;
        let mut overlap_fade = 0usize;

        for py in 0..viewport.height {
            for px in 0..viewport.width {
                let off = ((py * viewport.width + px) * 4) as usize;
                let c = &combined[off..off + 4];
                let l = &left_only[off..off + 4];
                let r = &right_only[off..off + 4];

                if c[3] != 255 {
                    non_opaque_alpha += 1;
                }

                let l_cov = l[3] == 255;
                let r_cov = r[3] == 255;
                match (l_cov, r_cov) {
                    (false, false) => {
                        if c[0] != 0 || c[1] != 0 || c[2] != 0 {
                            uncovered_mismatch += 1;
                        }
                    }
                    (true, false) => {
                        if c[0] != l[0] || c[1] != l[1] || c[2] != l[2] {
                            left_only_mismatch += 1;
                        }
                    }
                    (false, true) => {
                        right_only_total += 1;
                        let ndc = viewport.pixel_ndc(px, py);
                        if let Some(rs) = maps[1].0.sample_uv(ndc)
                            && maps[1].1.alpha(rs.edge) < 1.0
                        {
                            right_only_fade += 1;
                        }
                        if (0..3).any(|ch| c[ch].abs_diff(r[ch]) > 1) {
                            right_only_mismatch += 1;
                        }
                        if (0..3).any(|ch| c[ch].saturating_add(1) < r[ch]) {
                            right_only_darkened += 1;
                        }
                    }
                    (true, true) => {
                        let ndc = viewport.pixel_ndc(px, py);
                        if let Some(rs) = maps[1].0.sample_uv(ndc) {
                            let alpha = maps[1].1.alpha(rs.edge);
                            if alpha < 1.0 {
                                overlap_fade += 1;
                            }
                            for ch in 0..3 {
                                let expected = (f64::from(l[ch])
                                    + (f64::from(r[ch]) - f64::from(l[ch])) * alpha)
                                    .round() as u8;
                                if c[ch].abs_diff(expected) > 2 {
                                    overlap_mismatch += 1;
                                    break;
                                }
                            }
                        } else {
                            // Subpixel GPU/CPU tie-break on the right outer lens
                            // boundary: blended output must remain bounded by
                            // the two source draws (no fade toward black).
                            for ch in 0..3 {
                                let lo = l[ch].min(r[ch]).saturating_sub(1);
                                let hi = l[ch].max(r[ch]).saturating_add(1);
                                if c[ch] < lo || c[ch] > hi {
                                    overlap_mismatch += 1;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(
            non_opaque_alpha, 0,
            "{label}: {non_opaque_alpha} pixels had alpha != 255"
        );
        assert_eq!(
            uncovered_mismatch, 0,
            "{label}: {uncovered_mismatch} uncovered pixels were non-black"
        );
        assert_eq!(
            left_only_mismatch, 0,
            "{label}: {left_only_mismatch} left-only pixels differed from isolated left GPU draw"
        );
        assert_eq!(
            right_only_darkened, 0,
            "{label}: {right_only_darkened} right-only pixels were darkened relative to isolated right GPU draw"
        );
        assert_eq!(
            right_only_mismatch, 0,
            "{label}: {right_only_mismatch} right-only pixels differed by >1 LSB from isolated right GPU draw"
        );
        assert_eq!(
            overlap_mismatch, 0,
            "{label}: {overlap_mismatch} overlap pixels deviated from smoothstep blend"
        );

        GpuOracleCounts {
            right_only_total,
            right_only_fade,
            overlap_fade,
        }
    }

    /// Issue #486 regression: white sources with blending enabled must produce
    /// no black or darkened pixels wherever at least one camera is valid,
    /// across mixed aspects, tilted/rolled planes, lens boundaries, and
    /// near-plane depth clipping, on both CPU and GPU.
    #[test]
    fn white_sources_have_no_dark_pixels_where_any_source_is_valid() {
        let viewport = ViewportSize {
            width: 240,
            height: 135,
        };
        let mut reversed_mixed = mixed_aspect_tilted_blend_calib();
        reversed_mixed.0.lenses.swap(0, 1);
        reversed_mixed.1.swap(0, 1);
        let shape = reversed_mixed.0.projection.l_shape_mut().unwrap();
        shape.x_ty = -0.18;
        shape.x_rz = -0.12;
        shape.x_rx = -0.07;
        shape.z_rz = -0.05;
        reversed_mixed.0.framing.tilt = -0.10;
        reversed_mixed.0.framing.roll = 0.09;

        let cases = [
            ("mixed-aspect-tilted", mixed_aspect_tilted_blend_calib()),
            ("mixed-aspect-reversed", reversed_mixed),
            ("lens-boundary", lens_boundary_blend_calib()),
            ("depth-clip-boundary", depth_clip_blend_calib()),
        ];

        for (label, (cal, sizes, pose)) in cases {
            let (ly, luv) = solid_nv12(sizes[0].0, sizes[0].1, 255);
            let (ry, ruv) = solid_nv12(sizes[1].0, sizes[1].1, 255);
            let planes = [
                Nv12Planes { y: &ly, uv: &luv },
                Nv12Planes { y: &ry, uv: &ruv },
            ];
            let cpu = stitch_rgba(
                &planes,
                &sizes,
                &cal,
                &viewport,
                pose,
                crate::render::color::YuvColor {
                    full_range: true,
                    ..Default::default()
                },
            )
            .unwrap();
            #[cfg(feature = "gpu")]
            let gpu_oracles =
                render_gpu_nv12_with_isolated_oracles(&cal, &viewport, sizes, &planes, pose, true);

            let maps = cal.surface_maps(&viewport, pose);
            let mut neither = 0usize;
            let mut left_only = 0usize;
            let mut right_only_zero = 0usize;
            let mut right_only_frac = 0usize;
            let mut both_blended = 0usize;

            for py in 0..viewport.height {
                for px in 0..viewport.width {
                    let ndc = viewport.pixel_ndc(px, py);
                    let l = maps[0].0.sample_uv(ndc);
                    let r = maps[1].0.sample_uv(ndc);
                    let off = ((py * viewport.width + px) * 4) as usize;
                    let cpu_rgb = &cpu[off..off + 3];
                    assert_eq!(cpu[off + 3], 255, "{label} ({px},{py}) CPU alpha");
                    match (l, r) {
                        (None, None) => {
                            neither += 1;
                            assert_eq!(cpu_rgb, [0, 0, 0], "{label} ({px},{py}) uncovered CPU");
                        }
                        (l_opt, r_opt) => {
                            match (l_opt, r_opt) {
                                (Some(_), None) => left_only += 1,
                                (None, Some(rs)) => {
                                    let nom = maps[1].1.alpha(rs.edge);
                                    if nom == 0.0 {
                                        right_only_zero += 1;
                                    } else if nom < 1.0 {
                                        right_only_frac += 1;
                                    }
                                }
                                (Some(_), Some(rs)) => {
                                    let nom = maps[1].1.alpha(rs.edge);
                                    if nom > 0.0 && nom < 1.0 {
                                        both_blended += 1;
                                    }
                                }
                                (None, None) => unreachable!(),
                            }
                            assert_eq!(
                                cpu_rgb,
                                [255, 255, 255],
                                "{label} ({px},{py}) valid pixel darkened on CPU"
                            );
                        }
                    }
                }
            }

            assert!(left_only > 0, "{label}: expected left-only pixels");
            assert!(
                right_only_zero + right_only_frac > 0,
                "{label}: expected right-only pixels in blend zone (zero={right_only_zero}, frac={right_only_frac})"
            );
            if label == "mixed-aspect-tilted" {
                assert!(neither > 0, "{label}: expected uncovered pixels");
                assert!(
                    right_only_zero > 0,
                    "{label}: expected right-only zero-weight pixels"
                );
                assert!(
                    right_only_frac > 0,
                    "{label}: expected right-only fractional-weight pixels"
                );
                assert!(both_blended > 0, "{label}: expected blended overlap pixels");
            }
            #[cfg(feature = "gpu")]
            if let Some((ref combined, ref gpu_left, ref gpu_right)) = gpu_oracles {
                // Exact isolated-GPU-draw verification: 0 darkened valid pixels,
                // 0 lit uncovered pixels, and opaque alpha on every pixel.
                let mut gpu_dark_on_valid = 0usize;
                let mut gpu_lit_on_uncovered = 0usize;
                for py in 0..viewport.height {
                    for px in 0..viewport.width {
                        let off = ((py * viewport.width + px) * 4) as usize;
                        assert_eq!(combined[off + 3], 255, "{label} ({px},{py}) GPU alpha");
                        let covered = gpu_left[off + 3] == 255 || gpu_right[off + 3] == 255;
                        if covered {
                            if combined[off..off + 3].iter().any(|&c| c < 254) {
                                gpu_dark_on_valid += 1;
                            }
                        } else if combined[off..off + 3] != [0, 0, 0] {
                            gpu_lit_on_uncovered += 1;
                        }
                    }
                }
                assert_eq!(
                    gpu_dark_on_valid, 0,
                    "{label}: {gpu_dark_on_valid} darkened valid pixels on GPU"
                );
                assert_eq!(
                    gpu_lit_on_uncovered, 0,
                    "{label}: {gpu_lit_on_uncovered} lit uncovered pixels on GPU"
                );
                assert_gpu_matches_isolated_draws(
                    label, &cal, &viewport, pose, combined, gpu_left, gpu_right,
                );
                Agreement::compare(combined, &cpu).assert_within(AgreementBounds::DEFAULT, label);
            }
        }
    }

    /// Isolated-GPU-draw oracle regression for the exact counterexample FOVs
    /// where analytic `f32` inverse-homography reprojection diverged from
    /// hardware left-plane rasterization (`93.081` darkened right-only pixel
    /// `(513, 481)`; `91.659` bypassed overlap blend at `(490, 491)`), plus
    /// the default FOV.
    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_dstalpha_oracle_matches_isolated_draws_on_counterexample_fovs() {
        let viewport = ViewportSize {
            width: 960,
            height: 540,
        };
        let (cal, sizes, base_pose) = mixed_aspect_tilted_blend_calib();
        let (ly, luv) = solid_nv12(sizes[0].0, sizes[0].1, 40);
        let (ry, ruv) = solid_nv12(sizes[1].0, sizes[1].1, 255);
        let planes = [
            Nv12Planes { y: &ly, uv: &luv },
            Nv12Planes { y: &ry, uv: &ruv },
        ];

        for (label, fov) in [
            ("mixed-aspect-fov-93.081", 93.081_f32),
            ("mixed-aspect-fov-91.659", 91.659_f32),
            ("mixed-aspect-fov-default", base_pose.fov_degrees),
        ] {
            let pose = RenderPose {
                fov_degrees: fov,
                ..base_pose
            };
            let Some((combined, left_only, right_only)) =
                render_gpu_nv12_with_isolated_oracles(&cal, &viewport, sizes, &planes, pose, true)
            else {
                return;
            };
            let counts = assert_gpu_matches_isolated_draws(
                label,
                &cal,
                &viewport,
                pose,
                &combined,
                &left_only,
                &right_only,
            );
            assert!(
                counts.right_only_fade > 0,
                "{label}: expected right-only pixels inside fade band"
            );
            assert!(
                counts.overlap_fade > 0,
                "{label}: expected blended overlap pixels"
            );
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_p010_blend_retains_ten_bit_precision_and_boundary_ownership() {
        use super::{Executor, GpuExecutor, GpuExecutorConfig};
        use crate::core::StitchCore;
        use crate::render::renderer::InputFormat;
        use crate::sink::PixelFormat;
        use crate::source::{FrameSet, Nv12Data};
        use std::sync::Arc;

        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let viewport = ViewportSize {
            width: 960,
            height: 540,
        };
        let (cal, sizes, base_pose) = mixed_aspect_tilted_blend_calib();
        let (ly, luv) = solid_nv12(sizes[0].0, sizes[0].1, 40);
        let (ry, ruv) = solid_nv12(sizes[1].0, sizes[1].1, 255);
        let planes = [
            Nv12Planes { y: &ly, uv: &luv },
            Nv12Planes { y: &ry, uv: &ruv },
        ];
        let images = FrameSet::Nv12(
            sizes
                .into_iter()
                .zip(planes.iter())
                .map(|(size, planes)| {
                    Arc::new(Nv12Data {
                        size,
                        y: planes.y.to_vec(),
                        uv: planes.uv.to_vec(),
                    })
                })
                .collect(),
        );
        let mut config = GpuExecutorConfig::new(cal.clone(), sizes.to_vec(), InputFormat::Nv12);
        config.viewport_size = viewport.clone();
        config.output_format = wgpu::TextureFormat::Rgb10a2Unorm;
        config.full_range = true;
        let mut core = StitchCore::new(Executor::Gpu(Box::new(
            GpuExecutor::new(gpu, config).unwrap(),
        )))
        .unwrap();
        let luma_code = |gray: f64| -> u16 {
            let rgb10 = (1023.0 * gray).round() / 1023.0;
            (64.0 + 876.0 * rgb10).round() as u16
        };
        let mut eight_bit_codes = [false; 1024];
        for gray8 in 0..=255 {
            eight_bit_codes[usize::from(luma_code(f64::from(gray8) / 255.0))] = true;
        }

        for fov_degrees in [93.081, 91.659] {
            let pose = RenderPose {
                fov_degrees,
                ..base_pose
            };
            let (_, left, right) =
                render_gpu_nv12_with_isolated_oracles(&cal, &viewport, sizes, &planes, pose, true)
                    .unwrap();
            let maps = cal.surface_maps(&viewport, pose);
            let cmd = core.render_frame_set_at_pose(&images, pose).unwrap();
            let gpu = core.executor.gpu_mut().unwrap();
            assert!(gpu.convert_delivery(cmd, None).unwrap().is_none());
            let frame = gpu.flush_delivery().unwrap().unwrap();
            assert_eq!(frame.format, PixelFormat::P010);
            let mut right_only_fade = 0;
            let mut overlap_between_eight_bit_codes = 0;
            for (i, word) in frame.data.chunks_exact(2).take(left.len() / 4).enumerate() {
                let code = u16::from_le_bytes([word[0], word[1]]) >> 6;
                let [x, y] = [i as u32 % viewport.width, i as u32 / viewport.width];
                let offset = i * 4;
                let weight = maps[1]
                    .0
                    .sample_uv(viewport.pixel_ndc(x, y))
                    .map(|sample| maps[1].1.alpha(sample.edge));
                let l = f64::from(left[offset]) / 255.0;
                let r = f64::from(right[offset]) / 255.0;
                let expected = match (left[offset + 3] != 0, right[offset + 3] != 0) {
                    (false, false) => 64,
                    (true, false) => luma_code(l),
                    (false, true) => {
                        right_only_fade += usize::from(weight.is_some_and(|a| a < 1.0));
                        luma_code(r)
                    }
                    (true, true) => {
                        let Some(alpha) = weight else {
                            // The isolated GPU draw owns the outer lens edge.
                            assert!((luma_code(l)..=luma_code(r)).contains(&code));
                            continue;
                        };
                        if alpha > 0.0 && alpha < 1.0 && !eight_bit_codes[usize::from(code)] {
                            overlap_between_eight_bit_codes += 1;
                        }
                        luma_code(l + (r - l) * alpha)
                    }
                };
                assert!(
                    code.abs_diff(expected) <= 2,
                    "FOV {fov_degrees} ({x},{y}): {code} vs {expected}"
                );
            }
            assert!(right_only_fade > 0);
            assert!(overlap_between_eight_bit_codes > 100);
        }
    }

    /// Isolated-GPU-draw oracle regression for a calibration whose left (base)
    /// plane edge crosses the right plane's fade band nearly horizontally.
    /// Analytic reprojection errors amplify along a near-horizontal edge into
    /// long single-row runs of darkened right-only pixels; hardware `DstAlpha`
    /// must produce 0 mismatches across the entire frame.
    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_dstalpha_oracle_nearly_horizontal_base_edge() {
        use crate::calibration::{Framing, Lens};
        use crate::projection::LShape;

        let viewport = ViewportSize {
            width: 960,
            height: 540,
        };
        let left_lens = Lens::fisheye(256, 144, 80.0, 80.0, 128.0, 72.0, [-0.02, 0.004, 0.0, 0.0]);
        let right_lens = Lens::fisheye(256, 192, 80.0, 80.0, 128.0, 96.0, [-0.02, 0.004, 0.0, 0.0]);
        // With axis_offset = 0.8 and intersect = 0.5 (half_offset = 0.25),
        // sin(z_rx) = hh_left / (2 * axis_offset - half_offset) = 0.28125 / 1.35
        // (~0.21 rad) cancels perspective foreshortening so the left plane's
        // projected top edge runs nearly horizontally across the right fade band
        // while staying well inside the 4:3 right plane's vertical footprint.
        let cal = Calibration::new(
            vec![left_lens, right_lens],
            LShape {
                intersect: 0.5,
                x_ty: 0.05,
                x_rz: 0.0,
                z_rx: 0.21,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.25,
            },
            Framing {
                axis_offset: 0.8,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        cal.validate().unwrap();
        let sizes = [(256, 144), (256, 192)];
        let pose = RenderPose {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: 50.0,
        };
        let (ly, luv) = solid_nv12(sizes[0].0, sizes[0].1, 40);
        let (ry, ruv) = solid_nv12(sizes[1].0, sizes[1].1, 255);
        let planes = [
            Nv12Planes { y: &ly, uv: &luv },
            Nv12Planes { y: &ry, uv: &ruv },
        ];

        let Some((combined, left_only, right_only)) =
            render_gpu_nv12_with_isolated_oracles(&cal, &viewport, sizes, &planes, pose, true)
        else {
            return;
        };

        // Confirm the geometry genuinely places a long horizontal base-edge
        // boundary across the right plane's fade band.
        let maps = cal.surface_maps(&viewport, pose);
        let mut max_horizontal_boundary_row = 0usize;
        for py in 1..(viewport.height - 1) {
            let mut row_boundary_px = 0usize;
            for px in 0..viewport.width {
                let off = ((py * viewport.width + px) * 4) as usize;
                let up = (((py - 1) * viewport.width + px) * 4) as usize;
                let dn = (((py + 1) * viewport.width + px) * 4) as usize;
                let right_only_here = left_only[off + 3] == 0 && right_only[off + 3] == 255;
                let adj_left = left_only[up + 3] == 255 || left_only[dn + 3] == 255;
                if right_only_here
                    && adj_left
                    && let Some(rs) = maps[1].0.sample_uv(viewport.pixel_ndc(px, py))
                    && maps[1].1.alpha(rs.edge) < 1.0
                {
                    row_boundary_px += 1;
                }
            }
            max_horizontal_boundary_row = max_horizontal_boundary_row.max(row_boundary_px);
        }
        assert!(
            max_horizontal_boundary_row >= 60,
            "expected a nearly horizontal base edge across the fade band, got max {max_horizontal_boundary_row} px in one row"
        );

        let counts = assert_gpu_matches_isolated_draws(
            "nearly-horizontal-base-edge",
            &cal,
            &viewport,
            pose,
            &combined,
            &left_only,
            &right_only,
        );
        assert!(counts.right_only_total > 0);
        assert!(counts.right_only_fade > 0);
        assert!(counts.overlap_fade > 0);
    }

    /// Hard-seam (`blend_width == 0.0`) fast path must skip the DstAlpha helper
    /// draws, still match isolated draws with 0 mismatches, and output opaque
    /// alpha across the entire frame.
    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_hard_seam_zero_blend_matches_isolated_draws() {
        let viewport = ViewportSize {
            width: 240,
            height: 135,
        };
        let (mut cal, sizes, pose) = mixed_aspect_tilted_blend_calib();
        cal.projection.l_shape_mut().unwrap().blend_width = 0.0;
        let (ly, luv) = solid_nv12(sizes[0].0, sizes[0].1, 40);
        let (ry, ruv) = solid_nv12(sizes[1].0, sizes[1].1, 255);
        let planes = [
            Nv12Planes { y: &ly, uv: &luv },
            Nv12Planes { y: &ry, uv: &ruv },
        ];
        let Some((combined, left_only, right_only)) =
            render_gpu_nv12_with_isolated_oracles(&cal, &viewport, sizes, &planes, pose, true)
        else {
            return;
        };
        assert_gpu_matches_isolated_draws(
            "hard-seam-zero-blend",
            &cal,
            &viewport,
            pose,
            &combined,
            &left_only,
            &right_only,
        );
    }

    /// Planar (L-shape) GPU compositing relies on destination alpha for
    /// hardware-exact seam coverage and must reject alpha-free output formats
    /// with a typed configuration error, even when `blend_width == 0.0` (since
    /// `apply_calibration` can enable blending later on the same renderer).
    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_executor_rejects_alpha_free_output_formats_for_l_shape() {
        use crate::render::{pipeline::PipelineError, renderer::InputFormat};
        use crate::stitch::{GpuExecutor, GpuExecutorConfig, StitchError};

        let Some(gpu) = gpu_or_skip() else {
            return;
        };

        let viewport = ViewportSize {
            width: 128,
            height: 72,
        };
        let (base_cal, sizes, _pose) = mixed_aspect_tilted_blend_calib();

        for blend_width in [0.15_f32, 0.0_f32] {
            let mut cal = base_cal.clone();
            cal.projection.l_shape_mut().unwrap().blend_width = blend_width;

            for format in [
                wgpu::TextureFormat::R8Unorm,
                wgpu::TextureFormat::Rg8Unorm,
                wgpu::TextureFormat::Rg11b10Ufloat,
            ] {
                let config = GpuExecutorConfig {
                    viewport_size: viewport.clone(),
                    output_format: format,
                    ..GpuExecutorConfig::new(cal.clone(), sizes.to_vec(), InputFormat::Nv12)
                };
                match GpuExecutor::new(gpu.clone(), config) {
                    Err(StitchError::Pipeline(PipelineError::InvalidConfig { .. })) => {}
                    Err(other) => panic!(
                        "expected PipelineError::InvalidConfig for {format:?} (blend_width={blend_width}), got {other}"
                    ),
                    Ok(_) => panic!(
                        "alpha-free format {format:?} (blend_width={blend_width}) should be rejected for L-shape GPU compositing"
                    ),
                }
            }
        }

        for format in [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
        ] {
            let config = GpuExecutorConfig {
                viewport_size: viewport.clone(),
                output_format: format,
                ..GpuExecutorConfig::new(base_cal.clone(), sizes.to_vec(), InputFormat::Nv12)
            };
            GpuExecutor::new(gpu.clone(), config)
                .unwrap_or_else(|e| panic!("alpha-bearing format {format:?} should succeed: {e}"));
        }
    }

    /// Verify that overlap seam blending remains unchanged and CPU/GPU agree
    /// with distinct left/right source intensities across mixed-aspect, lens,
    /// and depth-clip boundaries.
    #[test]
    fn overlap_blending_unchanged_and_agrees_across_boundaries() {
        let viewport = ViewportSize {
            width: 240,
            height: 135,
        };
        let cases = [
            ("overlap-mixed-aspect", mixed_aspect_tilted_blend_calib()),
            ("overlap-lens-boundary", lens_boundary_blend_calib()),
            ("overlap-depth-clip", depth_clip_blend_calib()),
        ];

        for (label, (cal, sizes, pose)) in cases {
            let (ly, luv) = solid_nv12(sizes[0].0, sizes[0].1, 40);
            let (ry, ruv) = solid_nv12(sizes[1].0, sizes[1].1, 240);
            let planes = [
                Nv12Planes { y: &ly, uv: &luv },
                Nv12Planes { y: &ry, uv: &ruv },
            ];
            let cpu = stitch_rgba(
                &planes,
                &sizes,
                &cal,
                &viewport,
                pose,
                crate::render::color::YuvColor {
                    full_range: true,
                    ..Default::default()
                },
            )
            .unwrap();
            #[cfg(feature = "gpu")]
            let gpu = render_gpu_nv12(&cal, &viewport, sizes, &planes, pose, true);

            let maps = cal.surface_maps(&viewport, pose);
            let mut checked_overlap = 0usize;
            let mut checked_right_only_fade = 0usize;

            for py in 0..viewport.height {
                for px in 0..viewport.width {
                    let ndc = viewport.pixel_ndc(px, py);
                    let off = ((py * viewport.width + px) * 4) as usize;
                    match (maps[0].0.sample_uv(ndc), maps[1].0.sample_uv(ndc)) {
                        (Some(_), Some(rs)) => {
                            let alpha = maps[1].1.alpha(rs.edge);
                            if alpha > 0.0 && alpha < 1.0 {
                                let expected = (40.0 + (240.0 - 40.0) * alpha).round() as u8;
                                assert!(
                                    cpu[off].abs_diff(expected) <= 1,
                                    "{label} ({px},{py}) CPU overlap={}, expected={expected}, alpha={alpha}",
                                    cpu[off]
                                );
                                #[cfg(feature = "gpu")]
                                if let Some(ref g) = gpu {
                                    assert!(
                                        g[off].abs_diff(expected) <= 2,
                                        "{label} ({px},{py}) GPU overlap={}, expected={expected}, alpha={alpha}",
                                        g[off]
                                    );
                                }
                                checked_overlap += 1;
                            }
                        }
                        (None, Some(rs)) if maps[1].1.alpha(rs.edge) < 1.0 => {
                            assert!(
                                cpu[off].abs_diff(240) <= 1,
                                "{label} ({px},{py}) CPU right-only={}, expected=240",
                                cpu[off]
                            );
                            checked_right_only_fade += 1;
                        }
                        _ => {}
                    }
                }
            }
            assert!(
                checked_right_only_fade > 0,
                "{label}: expected right-only pixels in nominal fade zone"
            );
            if label == "overlap-mixed-aspect" {
                assert!(
                    checked_overlap > 0,
                    "{label}: expected blended overlap pixels"
                );
            }
            #[cfg(feature = "gpu")]
            if let Some(ref g) = gpu {
                Agreement::compare(g, &cpu).assert_within(AgreementBounds::DEFAULT, label);
            }
        }
    }
}
