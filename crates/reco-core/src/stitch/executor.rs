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
//! - [`CpuExecutor`] binds a [`Projection`] and drives the pure-Rust gather.
//! - [`GpuExecutor`] owns the wgpu [`StitchPipeline`] (there is no other
//!   owner) plus a private blocking-readback ring for the synchronous
//!   contract.
//!
//! [`GpuExecutor`] lives behind the `gpu` feature (default-on);
//! [`CpuExecutor`] is unconditional - it is the render path for
//! wgpu-free builds.

use crate::calibration::{Calibration, Framing, Lens, Topology};
#[cfg(feature = "gpu")]
use crate::gpu::GpuContext;
#[cfg(feature = "gpu")]
use crate::gpu::rgba_readback::{RgbaReadback, RgbaReadbackError};
#[cfg(feature = "gpu")]
use crate::render::pipeline::{PipelineError, StitchPipeline};
use crate::render::planes::{Nv12Planes, YuvPlanes};
#[cfg(feature = "gpu")]
use crate::render::renderer::InputFormat;
use crate::render::scene::SceneGeometry;
use crate::render::viewport::ViewportConfig;

use crate::projection::Projection;

use super::cpu::stitch_rgba;

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
pub struct CpuExecutor {
    /// The bound projection: dispatches the per-frame surface maps.
    pub(crate) projection: Box<dyn Projection>,
    pub(crate) calib: Calibration,
    pub(crate) config: ViewportConfig,
    pub(crate) cam: (u32, u32),
    pub(crate) full_range: bool,
    /// Plane-placement geometry derived from `calib`, cached for the
    /// engine's coverage construction (the stitch kernel re-derives its
    /// own per call). Rebuilt by the [`Executor`] mutation methods.
    pub(crate) scene: SceneGeometry,
}

impl CpuExecutor {
    /// Configure a CPU executor: bind a projection to a fixed source size
    /// and output viewport.
    pub fn new(
        projection: Box<dyn Projection>,
        calib: Calibration,
        config: ViewportConfig,
        cam_w: u32,
        cam_h: u32,
        full_range: bool,
    ) -> Result<Self, StitchError> {
        calib
            .validate()
            .map_err(|e| StitchError::InvalidConfig(e.to_string()))?;
        config.validate().map_err(StitchError::InvalidConfig)?;
        if usize::from(projection.camera_count()) != calib.lenses.len() {
            return Err(StitchError::InvalidConfig(format!(
                "projection '{}' consumes {} cameras but the calibration has {} lenses",
                projection.name(),
                projection.camera_count(),
                calib.lenses.len()
            )));
        }
        if cam_w < 2 || cam_h < 2 {
            return Err(StitchError::InvalidConfig(format!(
                "source dimensions must be >= 2, got {cam_w}x{cam_h}"
            )));
        }
        let scene = derive_scene(&calib);
        Ok(Self {
            projection,
            calib,
            config,
            cam: (cam_w, cam_h),
            full_range,
            scene,
        })
    }

    /// Stitch one NV12 frame pair to RGBA at the configured output
    /// size. `&self` on purpose: the CPU stitch is stateless per call
    /// (the [`StitchExecutor`] trait's `&mut self` accommodates the
    /// GPU arm's readback ring).
    pub fn stitch_nv12(
        &self,
        left: &Nv12Planes<'_>,
        right: &Nv12Planes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<Vec<u8>, StitchError> {
        stitch_rgba(
            self.projection.as_ref(),
            left,
            right,
            self.cam,
            &self.calib,
            &self.config,
            yaw,
            pitch,
            self.full_range,
        )
    }

    /// Stitch one YUV420P frame pair to RGBA at the configured output
    /// size. The CPU kernel is format-flexible per call (unlike the
    /// GPU pipeline, which fixes its input format at construction);
    /// the [`StitchExecutor`] trait covers the NV12 contract, this
    /// inherent entry covers planar YUV sources (file decode).
    pub fn stitch_yuv(
        &self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<Vec<u8>, StitchError> {
        super::cpu::stitch_rgba_yuv420p(
            self.projection.as_ref(),
            left,
            right,
            self.cam,
            &self.calib,
            &self.config,
            yaw,
            pitch,
            self.full_range,
        )
    }
}

/// Plane-placement geometry for a calibration document (both stereo
/// cameras share the lens aspect). One derivation, shared by the CPU
/// executor's cache and its mutation paths - mirrors what
/// `StitchPipeline::update_calibration` does on the GPU side.
fn derive_scene(calib: &Calibration) -> SceneGeometry {
    let aspect = calib.lenses[0].width as f32 / calib.lenses[0].height as f32;
    SceneGeometry::new(&calib.topology, &calib.framing, aspect)
}

impl StitchExecutor for CpuExecutor {
    fn stitch(
        &mut self,
        left: &Nv12Planes,
        right: &Nv12Planes,
        yaw: f32,
        pitch: f32,
    ) -> Result<Vec<u8>, StitchError> {
        // Plane-size + dimension validation lives in stitch_rgba, which
        // returns a typed error instead of panicking on a short/truncated frame.
        self.stitch_nv12(left, right, yaw, pitch)
    }

    fn output_dims(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

/// Configuration for building a [`GpuExecutor`].
///
/// Owns everything the GPU pipeline needs to know about the frames it
/// will stitch: the calibration document, the output viewport, source
/// dimensions and pixel formats, and the projection. Engine-level
/// concerns (detection, trackers, replay) deliberately live on
/// [`StitchCore`](crate::core::StitchCore), not here.
#[cfg(feature = "gpu")]
pub struct GpuExecutorConfig {
    /// Camera calibration document.
    pub calibration: Calibration,
    /// Output viewport (dimensions, FOV).
    pub viewport: ViewportConfig,
    /// Input frame width in pixels (per camera).
    pub input_width: u32,
    /// Input frame height in pixels (per camera).
    pub input_height: u32,
    /// Input pixel format.
    pub input_format: InputFormat,
    /// GPU render-target format. `Rgba8Unorm` suits every compositor
    /// consumer; `Bgra8Unorm` matches native Windows DirectX surfaces
    /// for consumers that prefer to swizzle on upload instead of on
    /// readback.
    pub output_format: wgpu::TextureFormat,
    /// Projection to stitch through. `None` selects the two-camera
    /// L-shape ([`LShapeProjection`](crate::projection::LShapeProjection)).
    pub projection: Option<Box<dyn Projection>>,
    /// Whether source YUV uses full-range (JPEG) quantization.
    pub full_range: bool,
}

#[cfg(feature = "gpu")]
impl GpuExecutorConfig {
    /// New config with required fields only; defaults everywhere else
    /// (1080p viewport, `Rgba8Unorm` output, L-shape projection,
    /// limited-range YUV).
    pub fn new(
        calibration: Calibration,
        input_width: u32,
        input_height: u32,
        input_format: InputFormat,
    ) -> Self {
        Self {
            calibration,
            viewport: ViewportConfig {
                width: 1920,
                height: 1080,
                ..Default::default()
            },
            input_width,
            input_height,
            input_format,
            output_format: wgpu::TextureFormat::Rgba8Unorm,
            projection: None,
            full_range: false,
        }
    }
}

/// GPU executor - the sole owner of the wgpu [`StitchPipeline`] and of
/// the [`Projection`] bound to it.
///
/// [`StitchCore`](crate::core::StitchCore) holds one of these as its
/// render substrate and drives the streaming paths (pipelined readback,
/// zero-copy imports, preview-to-view) through it. The synchronous
/// `stitch()` path (crate-internal until the executor trait goes
/// public) renders one frame and blocks on a private readback ring,
/// for callers that want "planes in, RGBA out" with no pipelining.
#[cfg(feature = "gpu")]
pub struct GpuExecutor {
    pub(crate) pipeline: StitchPipeline,
    /// The bound projection: supplied the pipeline's GPU program at
    /// construction and dispatches coverage construction for the engine.
    pub(crate) projection: Box<dyn Projection>,
    /// Readback ring for the synchronous [`StitchExecutor::stitch`]
    /// path, created on first use so engine-embedded executors (which
    /// read back through the engine's own pipelined ring) never
    /// allocate it. Keyed by the output dims it was built for so a
    /// resize recreates it.
    sync_readback: Option<(RgbaReadback, (u32, u32))>,
}

#[cfg(feature = "gpu")]
impl GpuExecutor {
    /// Build a GPU executor. `gpu` is injected so reco-core does not
    /// pull an async runtime into non-test code; callers create it via
    /// [`GpuContext::new`].
    pub fn new(gpu: GpuContext, config: GpuExecutorConfig) -> Result<Self, StitchError> {
        let projection: Box<dyn Projection> = config
            .projection
            .unwrap_or_else(|| Box::new(crate::projection::LShapeProjection));
        if usize::from(projection.camera_count()) != config.calibration.lenses.len() {
            return Err(StitchError::InvalidConfig(format!(
                "projection '{}' consumes {} cameras but the calibration has {} lenses",
                projection.name(),
                projection.camera_count(),
                config.calibration.lenses.len()
            )));
        }
        log::info!(
            "GpuExecutor: projection '{}' supplies the GPU program and coverage",
            projection.name()
        );
        // Calibration validation happens once, inside with_gpu.
        let mut pipeline = StitchPipeline::with_gpu(
            gpu,
            &projection.gpu_program(),
            config.calibration,
            config.viewport,
            config.input_width,
            config.input_height,
            config.output_format,
            config.input_format,
        )?;
        pipeline.set_full_range(config.full_range);
        Ok(Self {
            pipeline,
            projection,
            sync_readback: None,
        })
    }
}

#[cfg(feature = "gpu")]
impl StitchExecutor for GpuExecutor {
    fn stitch(
        &mut self,
        left: &Nv12Planes,
        right: &Nv12Planes,
        yaw: f32,
        pitch: f32,
    ) -> Result<Vec<u8>, StitchError> {
        // The synchronous contract is NV12-specific; the underlying
        // upload only debug-asserts the format, so guard it here with
        // a typed error instead of corrupting textures in release.
        if self.pipeline.input_format() != InputFormat::Nv12 {
            return Err(StitchError::InvalidConfig(format!(
                "stitch() consumes NV12 planes but the executor was built \
                 for {:?} input",
                self.pipeline.input_format()
            )));
        }
        // (Re)create the private ring on first use or after a resize.
        let dims = self.output_dims();
        if !matches!(&self.sync_readback, Some((_, d)) if *d == dims) {
            let ring = RgbaReadback::new(self.pipeline.gpu(), dims.0, dims.1)?;
            self.sync_readback = Some((ring, dims));
        }
        // Record the frame, submit it via the readback, then drain it
        // synchronously: one render in, this frame's RGBA out.
        let cmd = self
            .pipeline
            .render_to_target_nv12(left, right, yaw, pitch)?;
        let tex = self.pipeline.render_target();
        let (ring, _) = self.sync_readback.as_mut().expect("created above");
        ring.readback(self.pipeline.gpu(), tex, cmd)?;
        // A frame was just submitted, so flush_pending always drains it.
        let frame = ring
            .flush_pending(self.pipeline.gpu())?
            .expect("flush_pending yields the just-submitted frame");
        Ok(frame.to_vec())
    }

    fn output_dims(&self) -> (u32, u32) {
        let v = self.pipeline.viewport();
        (v.width, v.height)
    }

    fn name(&self) -> &'static str {
        "gpu"
    }
}

/// The closed executor set the engine dispatches over (L2): one CPU
/// software path, one GPU pipeline owner.
///
/// [`StitchCore`](crate::core::StitchCore) holds exactly one and
/// resolves every render + live-config operation through it. The
/// GPU-only streaming surface (command-buffer renders, resident-frame
/// imports, readback rings) is reached via [`Executor::gpu`] /
/// [`Executor::gpu_mut`] - a typed accessor, no downcasting.
///
/// The live-config methods dispatch per arm: the GPU arm forwards to
/// the pipeline's update machinery (uniforms, scene rebuild), the CPU
/// arm mutates the executor's own document and rebuilds its cached
/// scene - the stitch kernel reads the document per call, so there is
/// no second copy to drift.
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

    /// The output viewport (dimensions + FOV).
    pub fn viewport(&self) -> &ViewportConfig {
        match self {
            Executor::Cpu(c) => &c.config,
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.viewport(),
        }
    }

    /// The derived plane-placement geometry for the active calibration.
    pub fn scene(&self) -> &SceneGeometry {
        match self {
            Executor::Cpu(c) => &c.scene,
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => &g.pipeline.scene,
        }
    }

    /// The bound projection.
    pub fn projection(&self) -> &dyn Projection {
        match self {
            Executor::Cpu(c) => c.projection.as_ref(),
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.projection.as_ref(),
        }
    }

    /// Source frame dimensions `(width, height)` per camera.
    pub fn source_info(&self) -> (u32, u32) {
        match self {
            Executor::Cpu(c) => c.cam,
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.source_info(),
        }
    }

    /// Current vertical field of view in degrees.
    pub fn fov(&self) -> f32 {
        match self {
            Executor::Cpu(c) => c.config.fov_degrees,
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.fov(),
        }
    }

    /// Set the vertical field of view in degrees, clamped to
    /// `[1.0, 179.0]` on both arms (the CPU projection math degenerates
    /// at 0/180 exactly like the GPU perspective matrix would).
    pub fn set_fov(&mut self, fov_degrees: f32) {
        match self {
            // Mirrors StitchPipeline::set_fov's clamp so the executors
            // cannot diverge on out-of-range input.
            Executor::Cpu(c) => c.config.fov_degrees = fov_degrees.clamp(1.0, 179.0),
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.set_fov(fov_degrees),
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
                c.config.width = width;
                c.config.height = height;
                Some((width, height))
            }
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.resize(width, height),
        }
    }

    /// Set the seam blend width (document field; no geometry rebuild).
    pub fn set_blend_width(&mut self, width: f32) {
        match self {
            Executor::Cpu(c) => c.calib.topology.blend_width = width,
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.set_blend_width(width),
        }
    }

    /// Set the lens-correction strength on every lens, clamped to `[0, 1]`.
    pub fn set_lens_correction_amount(&mut self, amount: f32) {
        match self {
            Executor::Cpu(c) => {
                let amount = amount.clamp(0.0, 1.0);
                for lens in &mut c.calib.lenses {
                    lens.correction = amount;
                }
            }
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.set_lens_correction_amount(amount),
        }
    }

    /// Replace the whole calibration document, rebuilding derived geometry.
    pub fn update_calibration(&mut self, calibration: Calibration) {
        match self {
            Executor::Cpu(c) => {
                c.scene = derive_scene(&calibration);
                c.calib = calibration;
            }
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.update_calibration(calibration),
        }
    }

    /// Replace the topology (plane placement + seam), rebuilding geometry.
    pub fn update_topology(&mut self, topology: Topology) {
        match self {
            Executor::Cpu(c) => {
                c.calib.topology = topology;
                c.scene = derive_scene(&c.calib);
            }
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.update_topology(topology),
        }
    }

    /// Replace the framing (axis offset, tilt, roll), rebuilding geometry.
    pub fn update_framing(&mut self, framing: Framing) {
        match self {
            Executor::Cpu(c) => {
                c.calib.framing = framing;
                c.scene = derive_scene(&c.calib);
            }
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.update_framing(framing),
        }
    }

    /// Replace one or both cameras' intrinsics, rebuilding geometry.
    pub fn update_camera_params(&mut self, left: Option<Lens>, right: Option<Lens>) {
        match self {
            Executor::Cpu(c) => {
                if let Some(l) = left {
                    c.calib.lenses[0] = l;
                }
                if let Some(r) = right {
                    c.calib.lenses[1] = r;
                }
                c.scene = derive_scene(&c.calib);
            }
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.update_camera_params(left, right),
        }
    }
}

impl StitchExecutor for Executor {
    fn stitch(
        &mut self,
        left: &Nv12Planes,
        right: &Nv12Planes,
        yaw: f32,
        pitch: f32,
    ) -> Result<Vec<u8>, StitchError> {
        match self {
            Executor::Cpu(c) => c.stitch(left, right, yaw, pitch),
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.stitch(left, right, yaw, pitch),
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
    use crate::stitch::test_support::calib;
    #[cfg(feature = "gpu")]
    use crate::stitch::test_support::{Agreement, AgreementBounds, gpu_or_skip, nv12};

    /// The no-black guarantee, end to end: flat mid-grey input pushed
    /// through the real render path (`safe_clamp` -> `world_to_render_pose`
    /// -> CPU stitch) must not produce black (uncovered) pixels, even at
    /// extreme clamped targets on tilted/rolled rigs. Before the
    /// roll-aware clamp margins, the axis-aligned margin model leaked
    /// 4-9% black at ~19 deg tilt (worse zoomed in); the residual bound
    /// here covers slice-resolution imprecision only (#334, measured
    /// 0.34% worst-case independent of tilt/roll).
    ///
    /// CPU-only on purpose: the property is about the clamp + pose
    /// geometry, which the GPU shares verbatim via `l_shape_plane_maps`.
    #[test]
    fn clamped_poses_render_no_black_edges() {
        use crate::geometry::VirtualCamera;
        use crate::geometry::resolve_render_pose;
        use crate::projection::CoverageBoundary;
        use crate::render::scene::SceneGeometry;

        let (cam_w, cam_h) = (256u32, 144u32);
        let (out_w, out_h) = (192u32, 108u32);
        let gray_y = vec![128u8; (cam_w * cam_h) as usize];
        let gray_uv = vec![128u8; (cam_w * (cam_h / 2)) as usize];
        let planes = Nv12Planes {
            y: &gray_y,
            uv: &gray_uv,
        };
        let aspect_out = out_w as f32 / out_h as f32;
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
            let plane_aspect = cal.lenses[0].width as f32 / cal.lenses[0].height as f32;
            let scene = SceneGeometry::new(&cal.topology, &cal.framing, plane_aspect);
            let coverage = CoverageBoundary::from_calibration(&cal, &scene);
            let cam = VirtualCamera::new(&scene.camera_position);
            let fov = (coverage.max_fov_degrees() * fov_factor).min(60.0);
            let config = ViewportConfig {
                width: out_w,
                height: out_h,
                fov_degrees: fov,
            };
            let mut backend = CpuExecutor::new(
                Box::new(crate::projection::LShapeProjection),
                cal.clone(),
                config,
                cam_w,
                cam_h,
                false,
            )
            .expect("cpu");

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
                // Through the real authority (clamp + orient in one call),
                // so this test tracks any stage it grows in Steps 6-8.
                let (ry, rp) = resolve_render_pose(
                    &coverage,
                    &cam,
                    tilt as f32,
                    roll as f32,
                    wy,
                    wp,
                    fov,
                    aspect_out,
                );
                let frac = black_frac(&backend.stitch(&planes, &planes, ry, rp).unwrap());
                assert!(
                    frac < 0.01,
                    "black fraction {frac:.4} at tilt={tilt} roll={roll} fov={fov:.1} target=({wy},{wp})"
                );
            }
        }
    }

    #[test]
    fn cpu_backend_reports_dims_and_name() {
        let (w, h) = (64u32, 36u32);
        let backend = CpuExecutor::new(
            Box::new(crate::projection::LShapeProjection),
            calib(w, h),
            ViewportConfig {
                width: w,
                height: h,
                ..Default::default()
            },
            w,
            h,
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
            Box::new(crate::projection::LShapeProjection),
            calib(w, h),
            ViewportConfig {
                width: w,
                height: h,
                ..Default::default()
            },
            w,
            h,
            false,
        )
        .expect("cpu backend");
        let short = vec![0u8; 10];
        let planes = Nv12Planes {
            y: &short,
            uv: &short,
        };
        // Must return a typed error, not panic (matches the GPU backend).
        let err = backend.stitch(&planes, &planes, 0.0, 0.0).unwrap_err();
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

        let mut cpu = CpuExecutor::new(
            Box::new(crate::projection::LShapeProjection),
            calib.clone(),
            config.clone(),
            cam_w,
            cam_h,
            false,
        )
        .expect("cpu backend");
        let mut gpu = GpuExecutor::new(
            gpu,
            GpuExecutorConfig {
                viewport: config,
                ..GpuExecutorConfig::new(calib, cam_w, cam_h, InputFormat::Nv12)
            },
        )
        .expect("gpu backend");

        // Drive both through the trait object to prove selection works.
        let backends: [&mut dyn StitchExecutor; 2] = [&mut cpu, &mut gpu];
        let mut outputs = Vec::new();
        for b in backends {
            assert_eq!(b.output_dims(), (out_w, out_h));
            outputs.push(b.stitch(&left, &right, yaw, pitch).expect("stitch"));
        }
        let (cpu_rgba, gpu_rgba) = (&outputs[0], &outputs[1]);
        assert_eq!(cpu_rgba.len(), (out_w * out_h * 4) as usize);
        Agreement::compare(gpu_rgba, cpu_rgba)
            .assert_within(AgreementBounds::DEFAULT, "backend cpu-vs-gpu");
    }
}
