//! Executor selection: one interface to stitch a frame on the GPU or the CPU.
//!
//! [`StitchExecutor`] is a deliberately narrow, synchronous contract -
//! "NV12 planes + pan -> RGBA bytes" - the common denominator both backends
//! produce naturally. It does NOT try to unify the GPU pipeline's specialised
//! paths (zero-copy import, triple-buffered streaming readback, GUI texture
//! handoff); those are GPU-only by nature and stay inherent to
//! [`GpuExecutor`], reached through [`crate::core::StitchCore`], which owns
//! one executor as its render substrate.
//!
//! - [`CpuExecutor`] reads the document's [`ProjectionGeometry`] and drives the pure-Rust gather.
//! - [`GpuExecutor`] owns the wgpu `StitchPipeline` (there is no other
//!   owner) plus a private blocking-readback ring for the synchronous
//!   contract.
//!
//! [`GpuExecutor`] lives behind the `gpu` feature (default-on);
//! [`CpuExecutor`] is unconditional - it is the render path for
//! wgpu-free builds.

use crate::calibration::Calibration;
use crate::geometry::RenderPose;
use crate::projection::ProjectionGeometry;
use crate::render::planes::Nv12Planes;
use crate::render::viewport::ViewportSize;

// The GPU arm's imports, gated as one block with the code they serve.
#[cfg(feature = "gpu")]
use crate::gpu::rgba_readback::RgbaReadbackError;
#[cfg(feature = "gpu")]
use crate::render::pipeline::PipelineError;
#[cfg(all(test, feature = "gpu"))]
use crate::render::renderer::InputFormat;

/// Errors a stitch executor can return.
///
/// `Clone` so the engine's error type (which wraps this) can stay
/// `Clone + Send + Sync` for worker-thread channels.
#[derive(Debug, Clone, thiserror::Error)]
pub enum StitchError {
    /// The GPU pipeline failed to record or upload a frame.
    #[cfg(feature = "gpu")]
    #[error("gpu pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// The GPU readback failed.
    #[cfg(feature = "gpu")]
    #[error("gpu readback: {0}")]
    Readback(#[from] RgbaReadbackError),
    /// The projection is CPU-only: it has no GPU program to bind.
    #[cfg(feature = "gpu")]
    #[error("projection '{projection}' has no GPU program yet; construct the CPU executor instead")]
    NoGpuProgram {
        /// Name of the CPU-only projection.
        projection: &'static str,
    },
    /// Backend configuration is invalid (e.g. degenerate dimensions).
    #[error("invalid stitch config: {0}")]
    InvalidConfig(String),
    /// A source plane is smaller than the configured frame size.
    #[error("frame size mismatch: plane has {actual} bytes, need at least {expected}")]
    FrameSizeMismatch {
        /// Minimum bytes the plane must contain for the configured dimensions.
        expected: usize,
        /// Bytes the supplied plane actually contains.
        actual: usize,
    },
}

/// One frame's stitch, GPU or CPU, behind a single interface.
///
/// Backends are configured for a fixed source size and output viewport at
/// construction; [`stitch`](Self::stitch) takes only the per-frame planes and
/// pan. Output is `width * height * 4` sRGB-domain RGBA, identical in layout
/// across backends (the GPU and CPU agree to ~1 LSB).
pub trait StitchExecutor {
    /// Stitch one frame set (one NV12 plane pair per camera, in
    /// projection order) to RGBA at the configured output size.
    fn stitch(
        &mut self,
        planes: &[Nv12Planes<'_>],
        pose: RenderPose,
    ) -> Result<Vec<u8>, StitchError>;

    /// Output dimensions `(width, height)` in pixels.
    fn output_dims(&self) -> (u32, u32);

    /// Short backend name for logs and diagnostics.
    fn name(&self) -> &'static str;
}

mod cpu;
#[cfg(feature = "gpu")]
mod gpu;

pub use cpu::CpuExecutor;
#[cfg(feature = "gpu")]
pub use gpu::{GpuExecutor, GpuExecutorConfig};

/// The closed executor set the engine dispatches over (L2): one CPU
/// software path, one GPU pipeline owner.
///
/// [`StitchCore`](crate::core::StitchCore) holds exactly one and
/// resolves every render + live-config operation through it. The
/// GPU-only streaming surface (command-buffer renders, resident-frame
/// imports, readback rings) is reached via `Executor::gpu` /
/// `Executor::gpu_mut` with the `gpu` feature - typed accessors, no downcasting.
///
/// Calibration edits are accepted by [`StitchCore`](crate::core::StitchCore). The GPU arm caches derived scene geometry; the CPU stitch kernel reads the accepted document per call.
pub enum Executor {
    /// Pure-Rust software stitch - the GPU-less path. Boxed (like the
    /// GPU arm) so the enum itself stays pointer-sized inside the
    /// engine.
    Cpu(Box<CpuExecutor>),
    /// wgpu pipeline owner - the streaming path. Boxed: the pipeline
    /// state dwarfs the CPU variant and the enum lives inside every
    /// engine.
    #[cfg(feature = "gpu")]
    Gpu(Box<GpuExecutor>),
}

impl Executor {
    /// The GPU executor, when this is the GPU strategy - the typed
    /// accessor to the streaming/zero-copy surface.
    #[cfg(feature = "gpu")]
    pub fn gpu(&self) -> Option<&GpuExecutor> {
        match self {
            Executor::Gpu(g) => Some(g),
            Executor::Cpu(_) => None,
        }
    }

    /// Mutable [`Self::gpu`].
    #[cfg(feature = "gpu")]
    pub fn gpu_mut(&mut self) -> Option<&mut GpuExecutor> {
        match self {
            Executor::Gpu(g) => Some(g),
            Executor::Cpu(_) => None,
        }
    }

    /// The active calibration document.
    pub fn calibration(&self) -> &Calibration {
        match self {
            Executor::Cpu(c) => &c.calib,
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.calibration(),
        }
    }

    /// The output viewport dimensions.
    pub fn viewport_size(&self) -> &ViewportSize {
        match self {
            Executor::Cpu(c) => &c.viewport_size,
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.viewport_size(),
        }
    }

    /// The projection in effect: a borrow of the calibration
    /// document's projection on either arm.
    pub(crate) fn geometry(&self) -> &dyn ProjectionGeometry {
        match self {
            Executor::Cpu(c) => c.geometry(),
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.geometry(),
        }
    }

    /// Source frame dimensions `(width, height)` per camera.
    pub fn source_sizes(&self) -> &[(u32, u32)] {
        match self {
            Executor::Cpu(c) => &c.source_sizes,
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.source_sizes(),
        }
    }

    /// Resize the output viewport. Returns the accepted `(width,
    /// height)`, or `None` when the request was rejected (zero dim).
    pub fn resize(&mut self, width: u32, height: u32) -> Option<(u32, u32)> {
        match self {
            Executor::Cpu(c) => {
                if width == 0 || height == 0 {
                    log::warn!("resize({width}, {height}) ignored: dimensions must be non-zero");
                    return None;
                }
                c.viewport_size.width = width;
                c.viewport_size.height = height;
                Some((width, height))
            }
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.resize(width, height),
        }
    }

    /// Whether source YUV uses full-range (0-255) quantization rather
    /// than limited range (16-235).
    pub fn set_full_range(&mut self, full_range: bool) {
        match self {
            Executor::Cpu(c) => c.full_range = full_range,
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.set_full_range(full_range),
        }
    }

    /// Flip each camera's source 180 degrees at sample time (rotated
    /// mounts whose streams carry rotation=180 metadata).
    ///
    /// GPU sampling only: the CPU decode path reverses buffers at
    /// extraction, so a CPU engine has nothing to flip here.
    pub fn set_flip_180(&mut self, left: bool, right: bool) {
        match self {
            Executor::Cpu(_) => {
                if left || right {
                    log::warn!(
                        "set_flip_180 ignored on the CPU executor - CPU sources reverse \
                         buffers at decode"
                    );
                }
            }
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.set_flip_180(left, right),
        }
    }

    /// Install a calibration already validated against the active engine's input layout.
    pub(crate) fn apply_calibration(&mut self, calibration: Calibration) {
        match self {
            Executor::Cpu(c) => c.calib = calibration,
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.apply_calibration(calibration),
        }
    }
}

impl StitchExecutor for Executor {
    fn stitch(
        &mut self,
        planes: &[Nv12Planes<'_>],
        pose: RenderPose,
    ) -> Result<Vec<u8>, StitchError> {
        match self {
            Executor::Cpu(c) => c.stitch(planes, pose),
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.stitch(planes, pose),
        }
    }

    fn output_dims(&self) -> (u32, u32) {
        match self {
            Executor::Cpu(c) => c.output_dims(),
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.output_dims(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Executor::Cpu(c) => c.name(),
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.name(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "gpu")]
    use crate::calibration::{Framing, Lens};
    use crate::stitch::test_support::calib;
    #[cfg(feature = "gpu")]
    use crate::stitch::test_support::{Agreement, AgreementBounds, gpu_or_skip, nv12};

    /// Flat input rendered through the shared coverage constraint has no missing pixels, including extreme targets with tilt and roll.
    #[test]
    fn clamped_poses_render_no_black_edges() {
        use crate::core::StitchCore;

        let (cam_w, cam_h) = (256u32, 144u32);
        let (out_w, out_h) = (192u32, 108u32);
        let gray_y = vec![128u8; (cam_w * cam_h) as usize];
        let gray_uv = vec![128u8; (cam_w * (cam_h / 2)) as usize];
        let planes = Nv12Planes {
            y: &gray_y,
            uv: &gray_uv,
        };
        let black_frac = |rgba: &[u8]| {
            let black = rgba
                .chunks_exact(4)
                .filter(|p| p[0] < 2 && p[1] < 2 && p[2] < 2)
                .count();
            black as f64 / (out_w * out_h) as f64
        };

        // (tilt, roll, fov as a fraction of the coverage max): level rig,
        // moderate and gameday tilt, tilt+roll, and the zoomed-in regime
        // where the rotated-corner overhang is proportionally largest.
        for &(tilt, roll, fov_factor) in &[
            (0.0f64, 0.0f64, 0.9f32),
            (0.15, 0.0, 0.9),
            (0.33, 0.0, 0.9),
            (0.33, 0.12, 0.9),
            (0.33, 0.12, 0.5),
        ] {
            let mut cal = calib(cam_w, cam_h);
            cal.framing.tilt = tilt;
            cal.framing.roll = roll;
            let config = ViewportSize {
                width: out_w,
                height: out_h,
            };
            let mut backend = {
                let source_sizes = vec![(cam_w, cam_h); cal.projection.camera_count()];
                CpuExecutor::new(cal.clone(), config.clone(), source_sizes, false)
            }
            .expect("cpu");
            let mut core = StitchCore::new(Executor::Cpu(Box::new(
                {
                    let source_sizes = vec![(cam_w, cam_h); cal.projection.camera_count()];
                    CpuExecutor::new(cal.clone(), config, source_sizes, false)
                }
                .unwrap(),
            )))
            .unwrap();
            let fov = (core.prepare_coverage().unwrap() * fov_factor).min(60.0);

            for &(wy, wp) in &[
                (0.0f32, 0.0f32),
                (-3.0, -1.5),
                (-3.0, 1.5),
                (3.0, -1.5),
                (3.0, 1.5),
                (0.0, -1.5),
                (0.0, 1.5),
                (-3.0, 0.0),
                (3.0, 0.0),
            ] {
                let requested =
                    core.calibration()
                        .world_to_render_pose(crate::geometry::WorldPose {
                            yaw: wy,
                            pitch: wp,
                            fov_degrees: fov,
                        });
                let pose = core.constrain_render_pose(requested, requested).unwrap();
                let frac = black_frac(&backend.stitch(&[planes, planes], pose).unwrap());
                assert!(
                    frac == 0.0,
                    "black fraction {frac:.4} at tilt={tilt} roll={roll} fov={fov:.1} target=({wy},{wp})"
                );
            }
        }
    }

    #[test]
    fn cpu_backend_reports_dims_and_name() {
        let (w, h) = (64u32, 36u32);
        let backend = CpuExecutor::new(
            calib(w, h),
            ViewportSize {
                width: w,
                height: h,
            },
            vec![(w, h); 2],
            false,
        )
        .expect("cpu backend");
        assert_eq!(backend.output_dims(), (w, h));
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn cpu_backend_rejects_undersized_planes() {
        let (w, h) = (64u32, 36u32);
        let mut backend = CpuExecutor::new(
            calib(w, h),
            ViewportSize {
                width: w,
                height: h,
            },
            vec![(w, h); 2],
            false,
        )
        .expect("cpu backend");
        let short = vec![0u8; 10];
        let planes = Nv12Planes {
            y: &short,
            uv: &short,
        };
        // Must return a typed error, not panic (matches the GPU backend).
        let err = backend
            .stitch(&[planes, planes], RenderPose::default())
            .unwrap_err();
        assert!(matches!(err, StitchError::FrameSizeMismatch { .. }));
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn cpu_and_gpu_backends_agree() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };

        let (cam_w, cam_h) = (192u32, 108u32);
        let (out_w, out_h) = (160u32, 90u32);
        let calib = calib(cam_w, cam_h);
        let config = ViewportSize {
            width: out_w,
            height: out_h,
        };
        let (ly, luv) = nv12(cam_w, cam_h, 0);
        let (ry, ruv) = nv12(cam_w, cam_h, 30);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };
        let pose = RenderPose {
            yaw: 0.08,
            pitch: -0.04,
            ..Default::default()
        };

        let mut cpu = {
            let source_sizes = vec![(cam_w, cam_h); calib.projection.camera_count()];
            CpuExecutor::new(calib.clone(), config.clone(), source_sizes, false)
        }
        .expect("cpu backend");
        let mut gpu = GpuExecutor::new(
            gpu,
            GpuExecutorConfig {
                viewport_size: config,
                ..{
                    let source_sizes = vec![(cam_w, cam_h); calib.projection.camera_count()];
                    GpuExecutorConfig::new(calib, source_sizes, InputFormat::Nv12)
                }
            },
        )
        .expect("gpu backend");

        // Drive both through the trait object to prove selection works.
        let backends: [&mut dyn StitchExecutor; 2] = [&mut cpu, &mut gpu];
        let mut outputs = Vec::new();
        for b in backends {
            assert_eq!(b.output_dims(), (out_w, out_h));
            outputs.push(b.stitch(&[left, right], pose).expect("stitch"));
        }
        let (cpu_rgba, gpu_rgba) = (&outputs[0], &outputs[1]);
        assert_eq!(cpu_rgba.len(), (out_w * out_h * 4) as usize);
        Agreement::compare(gpu_rgba, cpu_rgba)
            .assert_within(AgreementBounds::DEFAULT, "backend cpu-vs-gpu");
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn preview_view_matches_batch_target_for_each_projection_and_format() {
        use crate::gpu::rgba_readback::RgbaReadback;
        use crate::source::{FrameSet, Nv12Data, Yuv420pData};

        let Some(gpu) = gpu_or_skip() else { return };
        let (w, h) = (192, 108);
        let cylinder = Calibration::new(
            vec![Lens::flat(w, h)],
            crate::projection::Cylinder {
                radius: 90.0,
                arc_deg: 180.0,
                surface_height: None,
            },
            Framing {
                axis_offset: 0.0,
                tilt: 0.2,
                roll: -0.15,
            },
        );
        for cal in [calib(w, h), cylinder] {
            for format in [InputFormat::Nv12, InputFormat::Yuv420p] {
                let count = cal.projection.camera_count();
                let frames = if format == InputFormat::Nv12 {
                    FrameSet::Nv12(
                        (0..count)
                            .map(|camera| {
                                let (y, uv) = nv12(w, h, camera as u8 * 30);
                                Nv12Data {
                                    size: (w, h),
                                    y,
                                    uv,
                                }
                                .into()
                            })
                            .collect(),
                    )
                } else {
                    FrameSet::Yuv420p(
                        (0..count)
                            .map(|camera| {
                                let (y, uv) = nv12(w, h, camera as u8 * 30);
                                Yuv420pData {
                                    size: (w, h),
                                    y,
                                    u: uv.iter().step_by(2).copied().collect(),
                                    v: uv.iter().skip(1).step_by(2).copied().collect(),
                                }
                                .into()
                            })
                            .collect(),
                    )
                };
                let mut executor = GpuExecutor::new(gpu.clone(), {
                    let source_sizes = vec![(w, h); cal.projection.camera_count()];
                    GpuExecutorConfig::new(cal.clone(), source_sizes, format)
                })
                .unwrap();
                for (width, height) in [(80, 48), (48, 80)] {
                    executor.pipeline.resize(width, height).unwrap();
                    for (invalid_width, invalid_height) in
                        [(0, height), (width, 0), (u32::MAX, height)]
                    {
                        assert!(
                            executor
                                .pipeline
                                .resize(invalid_width, invalid_height)
                                .is_none()
                        );
                        assert_eq!(
                            (
                                executor.pipeline.viewport_size.width,
                                executor.pipeline.viewport_size.height
                            ),
                            (width, height)
                        );
                    }
                    let mut batch_executor = GpuExecutor::new(
                        gpu.clone(),
                        GpuExecutorConfig {
                            viewport_size: ViewportSize { width, height },
                            ..{
                                let source_sizes = vec![(w, h); cal.projection.camera_count()];
                                GpuExecutorConfig::new(cal.clone(), source_sizes, format)
                            }
                        },
                    )
                    .unwrap();
                    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("preview oracle"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    });
                    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
                    for yaw in [-0.4, 0.0, 0.5] {
                        let pose = RenderPose {
                            yaw,
                            pitch: 0.1,
                            fov_degrees: 55.0,
                        };
                        executor
                            .pipeline
                            .render_to_view(&frames, pose, &view)
                            .unwrap();
                        let empty = gpu
                            .device
                            .create_command_encoder(&Default::default())
                            .finish();
                        let mut readback = RgbaReadback::new(&gpu, width, height).unwrap();
                        readback.readback(&gpu, &target, empty).unwrap();
                        let preview = readback.flush_pending(&gpu).unwrap().unwrap().to_vec();
                        let cmd = batch_executor
                            .pipeline
                            .render_frame_set(&frames, pose)
                            .unwrap();
                        readback
                            .readback(&gpu, batch_executor.pipeline.render_target(), cmd)
                            .unwrap();
                        let batch = readback.flush_pending(&gpu).unwrap().unwrap();
                        assert!(
                            preview == batch,
                            "cameras={count}, format={format:?}, {width}x{height}, yaw={yaw}"
                        );
                        assert!(preview.chunks_exact(4).any(|pixel| pixel[0] > 32));
                        let cmd = executor.pipeline.render_frame_set(&frames, pose).unwrap();
                        readback
                            .readback(&gpu, executor.pipeline.render_target(), cmd)
                            .unwrap();
                        let resized = readback.flush_pending(&gpu).unwrap().unwrap();
                        assert_eq!(
                            preview, resized,
                            "resized internal target must match caller output"
                        );
                    }
                    assert!(
                        executor
                            .pipeline
                            .render_to_view(&FrameSet::Nv12(vec![]), RenderPose::default(), &view)
                            .is_err()
                    );
                    let short = FrameSet::Nv12(
                        (0..count)
                            .map(|_| {
                                Nv12Data {
                                    size: (w, h),
                                    y: vec![0],
                                    uv: vec![0],
                                }
                                .into()
                            })
                            .collect(),
                    );
                    assert!(
                        executor
                            .pipeline
                            .render_to_view(&short, RenderPose::default(), &view)
                            .is_err()
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn camera_dimensions_respect_the_consumers_device_limits() {
        use crate::gpu::GpuContext;
        use crate::render::renderer::InputFormat;

        let Some(original) = gpu_or_skip() else {
            return;
        };
        let adapter = original.adapter.as_ref().unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("consumer with smaller texture limits"),
            required_limits: wgpu::Limits {
                max_texture_dimension_2d: 128,
                ..original.device.limits()
            },
            required_features: original.device.features(),
            ..Default::default()
        }))
        .unwrap();
        let gpu = GpuContext::from_device_queue(device, queue, adapter.get_info());
        let config = |sizes, format| GpuExecutorConfig {
            viewport_size: ViewportSize {
                width: 64,
                height: 32,
            },
            ..GpuExecutorConfig::new(calib(64, 32), sizes, format)
        };
        for format in [InputFormat::Nv12, InputFormat::Yuv420p, InputFormat::Rgba] {
            for camera in 0..2 {
                for oversized in [(130, 32), (64, 130)] {
                    let mut sizes = vec![(64, 32); 2];
                    sizes[camera] = oversized;
                    let Err(error) = GpuExecutor::new(gpu.clone(), config(sizes, format)) else {
                        panic!("oversized camera accepted")
                    };
                    assert!(
                        error.to_string().contains("device texture dimension limit"),
                        "{error}"
                    );
                }
            }
            assert!(
                GpuExecutor::new(gpu.clone(), config(vec![(128, 32), (64, 128)], format)).is_ok(),
                "dimensions at the device limit must remain usable"
            );
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn cylinder_gpu_matches_cpu_mapping_and_rejects_invalid_frames() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        eprintln!("cylinder oracle adapter: {}", gpu.gpu_name());
        let (w, h) = (192, 108);
        let (y, uv) = nv12(w, h, 0);
        let frame = Nv12Planes { y: &y, uv: &uv };
        let white_y = vec![200; (w * h) as usize];
        let white_uv = vec![128; (w * h / 2) as usize];
        let white = Nv12Planes {
            y: &white_y,
            uv: &white_uv,
        };
        for (out_w, out_h, surface_height) in [(80, 48, None), (48, 80, Some(80.0))] {
            for arc in [90.0, 180.0, 270.0, 360.0] {
                for full_range in [false, true] {
                    let cal = Calibration::new(
                        vec![Lens::flat(w, h)],
                        crate::projection::Cylinder {
                            radius: 90.0,
                            arc_deg: arc,
                            surface_height,
                        },
                        Framing {
                            axis_offset: 0.0,
                            tilt: 0.21,
                            roll: -0.15,
                        },
                    );
                    let viewport = ViewportSize {
                        width: out_w,
                        height: out_h,
                    };
                    let mut cpu = {
                        let source_sizes = vec![(w, h); cal.projection.camera_count()];
                        CpuExecutor::new(cal.clone(), viewport.clone(), source_sizes, full_range)
                    }
                    .unwrap();
                    let mut render = GpuExecutor::new(
                        gpu.clone(),
                        GpuExecutorConfig {
                            viewport_size: viewport,
                            full_range,
                            ..{
                                let source_sizes = vec![(w, h); cal.projection.camera_count()];
                                GpuExecutorConfig::new(cal, source_sizes, InputFormat::Nv12)
                            }
                        },
                    )
                    .unwrap();
                    for yaw in [0.0, -0.6, 0.7, 3.1] {
                        let pose = RenderPose {
                            yaw,
                            pitch: 0.15,
                            fov_degrees: 65.0,
                        };
                        let actual = render.stitch(&[frame], pose).unwrap();
                        let expected = cpu.stitch(&[frame], pose).unwrap();
                        assert!(
                            actual
                                .iter()
                                .zip(&expected)
                                .all(|(a, b)| a.abs_diff(*b) <= 2),
                            "cylinder pixel mismatch: arc={arc} yaw={yaw} full={full_range}"
                        );
                        Agreement::compare(&actual, &expected).assert_within(
                            AgreementBounds::DEFAULT,
                            &format!("cylinder arc={arc} yaw={yaw} full={full_range}"),
                        );
                        let actual_mask = render.stitch(&[white], pose).unwrap();
                        let expected_mask = cpu.stitch(&[white], pose).unwrap();
                        assert!(
                            actual_mask
                                .chunks_exact(4)
                                .zip(expected_mask.chunks_exact(4))
                                .all(|(a, b)| (a[0] == 0) == (b[0] == 0)),
                            "cylinder validity mask: arc={arc} yaw={yaw}"
                        );
                    }
                    assert!(render.stitch(&[], RenderPose::default()).is_err());
                    assert!(
                        render
                            .stitch(&[frame, frame], RenderPose::default())
                            .is_err()
                    );
                    assert!(
                        render
                            .stitch(&[Nv12Planes { y: &[0], uv: &[0] }], RenderPose::default())
                            .is_err()
                    );
                }
            }
        }
    }
}
