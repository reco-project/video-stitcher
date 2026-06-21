//! Backend selection: one interface to stitch a frame on the GPU or the CPU.
//!
//! [`StitchBackend`] is a deliberately narrow, synchronous contract -
//! "NV12 planes + pan -> RGBA bytes" - the common denominator both backends
//! produce naturally. It does NOT try to unify the GPU pipeline's specialised
//! paths (zero-copy import, triple-buffered streaming readback, GUI texture
//! handoff); those keep their dedicated route through [`crate::core`]. This
//! trait covers the headless "give me stitched RGBA, GPU or CPU" case - cloud
//! encode, edge devices, CLI - where the two backends converge on RGBA.
//!
//! - [`CpuStitchBackend`] wraps the pure-Rust [`stitch_l_shape_rgba`].
//! - [`GpuStitchBackend`] wraps the wgpu [`StitchPipeline`], absorbing its own
//!   render + blocking-readback so it satisfies the synchronous contract.
//!
//! When `wgpu` becomes optional (Phase 3), [`GpuStitchBackend`] moves behind a
//! feature; [`CpuStitchBackend`] stays unconditional.

use crate::calibration::MatchCalibration;
use crate::gpu::GpuContext;
use crate::gpu::rgba_readback::{RgbaReadback, RgbaReadbackError};
use crate::render::pipeline::{PipelineError, StitchPipeline};
use crate::render::planes::Nv12Planes;
use crate::render::renderer::InputFormat;
use crate::render::viewport::ViewportConfig;

use super::stitch_l_shape_rgba;

/// Errors a [`StitchBackend`] can return.
#[derive(Debug, thiserror::Error)]
pub enum StitchError {
    /// The GPU pipeline failed to record or upload a frame.
    #[error("gpu pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// The GPU readback failed.
    #[error("gpu readback: {0}")]
    Readback(#[from] RgbaReadbackError),
    /// The GPU pipeline did not yield a frame when one was expected.
    #[error("gpu produced no frame")]
    NoOutput,
}

/// One frame's stitch, GPU or CPU, behind a single interface.
///
/// Backends are configured for a fixed source size and output viewport at
/// construction; [`stitch`](Self::stitch) takes only the per-frame planes and
/// pan. Output is `width * height * 4` sRGB-domain RGBA, identical in layout
/// across backends (the GPU and CPU agree to ~1 LSB).
pub trait StitchBackend {
    /// Stitch one NV12 frame pair to RGBA at the configured output size.
    fn stitch(
        &mut self,
        left: &Nv12Planes,
        right: &Nv12Planes,
        yaw: f32,
        pitch: f32,
    ) -> Result<Vec<u8>, StitchError>;

    /// Output dimensions `(width, height)` in pixels.
    fn output_dims(&self) -> (u32, u32);

    /// Short backend name for logs and diagnostics.
    fn name(&self) -> &'static str;
}

/// CPU software backend - pure Rust, no GPU. The portable / GPU-less path.
pub struct CpuStitchBackend {
    calib: MatchCalibration,
    config: ViewportConfig,
    cam: (u32, u32),
    full_range: bool,
}

impl CpuStitchBackend {
    /// Configure a CPU backend for a fixed source size and output viewport.
    pub fn new(
        calib: MatchCalibration,
        config: ViewportConfig,
        cam_w: u32,
        cam_h: u32,
        full_range: bool,
    ) -> Self {
        Self {
            calib,
            config,
            cam: (cam_w, cam_h),
            full_range,
        }
    }
}

impl StitchBackend for CpuStitchBackend {
    fn stitch(
        &mut self,
        left: &Nv12Planes,
        right: &Nv12Planes,
        yaw: f32,
        pitch: f32,
    ) -> Result<Vec<u8>, StitchError> {
        Ok(stitch_l_shape_rgba(
            left,
            right,
            self.cam,
            &self.calib,
            &self.config,
            yaw,
            pitch,
            self.full_range,
        ))
    }

    fn output_dims(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

/// GPU backend - wraps the wgpu [`StitchPipeline`]. Renders one frame and
/// blocks on readback so it satisfies the synchronous [`StitchBackend`]
/// contract. Streaming consumers that want pipelined throughput should use
/// [`crate::core`] directly instead.
pub struct GpuStitchBackend {
    pipeline: StitchPipeline,
    readback: RgbaReadback,
    dims: (u32, u32),
}

impl GpuStitchBackend {
    /// Configure a GPU backend. `gpu` is injected so reco-core does not pull an
    /// async runtime into non-test code; callers create it via
    /// [`GpuContext::new`].
    pub fn new(
        gpu: GpuContext,
        calib: MatchCalibration,
        config: ViewportConfig,
        cam_w: u32,
        cam_h: u32,
        full_range: bool,
    ) -> Result<Self, StitchError> {
        let mut pipeline = StitchPipeline::with_gpu(
            gpu,
            calib,
            config.clone(),
            cam_w,
            cam_h,
            wgpu::TextureFormat::Rgba8Unorm,
            InputFormat::Nv12,
        )?;
        pipeline.set_full_range(full_range);
        let readback = RgbaReadback::new(pipeline.gpu(), config.width, config.height)?;
        Ok(Self {
            pipeline,
            readback,
            dims: (config.width, config.height),
        })
    }
}

impl StitchBackend for GpuStitchBackend {
    fn stitch(
        &mut self,
        left: &Nv12Planes,
        right: &Nv12Planes,
        yaw: f32,
        pitch: f32,
    ) -> Result<Vec<u8>, StitchError> {
        // Record the frame, submit it via the readback, then drain it
        // synchronously: one render in, this frame's RGBA out.
        let cmd = self
            .pipeline
            .render_to_target_nv12(left, right, yaw, pitch)?;
        let tex = self.pipeline.render_target();
        self.readback.readback(self.pipeline.gpu(), tex, cmd)?;
        match self.readback.flush_pending(self.pipeline.gpu())? {
            Some(bytes) => Ok(bytes.to_vec()),
            None => Err(StitchError::NoOutput),
        }
    }

    fn output_dims(&self) -> (u32, u32) {
        self.dims
    }

    fn name(&self) -> &'static str {
        "gpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{CameraParams, PlaneLayout};

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

    fn nv12(w: u32, h: u32, bias: u8) -> (Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0..w * h)
            .map(|i| ((i % w) as usize * 255 / w as usize) as u8)
            .map(|val| val.saturating_add(bias))
            .collect();
        let uv = vec![128u8; (w * (h / 2)) as usize];
        (y, uv)
    }

    #[test]
    fn cpu_backend_reports_dims_and_name() {
        let (w, h) = (64u32, 36u32);
        let backend = CpuStitchBackend::new(
            test_calib(w, h),
            ViewportConfig {
                width: w,
                height: h,
                ..Default::default()
            },
            w,
            h,
            false,
        );
        assert_eq!(backend.output_dims(), (w, h));
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn cpu_and_gpu_backends_agree() {
        use crate::gpu::GpuError;

        let gpu = match pollster::block_on(GpuContext::new()) {
            Ok(g) => g,
            Err(GpuError::NoAdapter | GpuError::AdapterRequest(_)) => {
                eprintln!("skipping backend agreement: no adapter");
                return;
            }
            Err(e) => panic!("gpu init: {e}"),
        };

        let (cam_w, cam_h) = (192u32, 108u32);
        let (out_w, out_h) = (160u32, 90u32);
        let calib = test_calib(cam_w, cam_h);
        let config = ViewportConfig {
            width: out_w,
            height: out_h,
            ..Default::default()
        };
        let (ly, luv) = nv12(cam_w, cam_h, 0);
        let (ry, ruv) = nv12(cam_w, cam_h, 30);
        let left = Nv12Planes { y: &ly, uv: &luv };
        let right = Nv12Planes { y: &ry, uv: &ruv };
        let (yaw, pitch) = (0.08f32, -0.04f32);

        let mut cpu = CpuStitchBackend::new(calib.clone(), config.clone(), cam_w, cam_h, false);
        let mut gpu =
            GpuStitchBackend::new(gpu, calib, config, cam_w, cam_h, false).expect("gpu backend");

        // Drive both through the trait object to prove selection works.
        let backends: [&mut dyn StitchBackend; 2] = [&mut cpu, &mut gpu];
        let mut outputs = Vec::new();
        for b in backends {
            assert_eq!(b.output_dims(), (out_w, out_h));
            outputs.push(b.stitch(&left, &right, yaw, pitch).expect("stitch"));
        }
        let (cpu_rgba, gpu_rgba) = (&outputs[0], &outputs[1]);
        assert_eq!(cpu_rgba.len(), (out_w * out_h * 4) as usize);
        assert_eq!(cpu_rgba.len(), gpu_rgba.len());

        let mut max = 0i32;
        for (c, g) in cpu_rgba.chunks_exact(4).zip(gpu_rgba.chunks_exact(4)) {
            for k in 0..3 {
                max = max.max((c[k] as i32 - g[k] as i32).abs());
            }
        }
        eprintln!("backend CPU-vs-GPU max RGB diff: {max}");
        assert!(max <= 4, "backends disagree by {max} (>4)");
    }
}
