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
//! - [`CpuExecutor`] reads the document's [`Projection`] and drives the pure-Rust gather.
//! - [`GpuExecutor`] owns the wgpu `StitchPipeline` (there is no other
//!   owner) plus a private blocking-readback ring for the synchronous
//!   contract.
//!
//! [`GpuExecutor`] lives behind the `gpu` feature (default-on);
//! [`CpuExecutor`] is unconditional - it is the render path for
//! wgpu-free builds.

use crate::calibration::{Calibration, Framing, Lens, Topology};
use crate::geometry::Pose;
use crate::projection::{CoverageBoundary, Projection};
use crate::render::planes::{Nv12Planes, YuvPlanes};
use crate::render::viewport::ViewportSize;

use super::cpu::stitch_rgba;

// The GPU arm's imports, gated as one block with the code they serve.
#[cfg(feature = "gpu")]
use crate::gpu::{
    GpuContext,
    rgba_readback::{RgbaReadback, RgbaReadbackError},
};
#[cfg(feature = "gpu")]
use crate::render::{
    pipeline::{PipelineError, StitchPipeline},
    renderer::InputFormat,
};

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
    fn stitch(&mut self, planes: &[Nv12Planes<'_>], pose: Pose) -> Result<Vec<u8>, StitchError>;

    /// Output dimensions `(width, height)` in pixels.
    fn output_dims(&self) -> (u32, u32);

    /// Short backend name for logs and diagnostics.
    fn name(&self) -> &'static str;
}

/// CPU software backend - pure Rust, no GPU. The portable / GPU-less path.
pub struct CpuExecutor {
    pub(crate) calib: Calibration,
    pub(crate) viewport_size: ViewportSize,
    pub(crate) cam: (u32, u32),
    pub(crate) full_range: bool,
}

impl CpuExecutor {
    /// Configure a CPU executor for a fixed source size and output
    /// viewport. The projection is the calibration document's topology
    /// (zero-copy: the document is the engine), so a document edit
    /// switches the projection automatically.
    pub fn new(
        calib: Calibration,
        viewport_size: ViewportSize,
        cam_w: u32,
        cam_h: u32,
        full_range: bool,
    ) -> Result<Self, StitchError> {
        calib
            .validate()
            .map_err(|e| StitchError::InvalidConfig(e.to_string()))?;
        viewport_size
            .validate()
            .map_err(StitchError::InvalidConfig)?;
        if cam_w < 2 || cam_h < 2 {
            return Err(StitchError::InvalidConfig(format!(
                "source dimensions must be >= 2, got {cam_w}x{cam_h}"
            )));
        }
        log::debug!(
            "CpuExecutor: projection '{}' from the calibration topology",
            calib.topology.projection().name()
        );

        Ok(Self {
            calib,
            viewport_size,
            cam: (cam_w, cam_h),
            full_range,
        })
    }

    /// The projection in effect: a borrow of the document's topology.
    pub(crate) fn projection(&self) -> &dyn Projection {
        self.calib.topology.projection()
    }

    /// Stitch one NV12 frame pair to RGBA at the configured output
    /// size. `&self` on purpose: the CPU stitch is stateless per call
    /// (the [`StitchExecutor`] trait's `&mut self` accommodates the
    /// GPU arm's readback ring).
    pub fn stitch_nv12(
        &self,
        planes: &[Nv12Planes<'_>],
        pose: Pose,
    ) -> Result<Vec<u8>, StitchError> {
        stitch_rgba(
            self.projection(),
            planes,
            self.cam,
            &self.calib,
            &self.viewport_size,
            pose,
            self.full_range,
        )
    }

    /// Stitch one YUV420P frame pair to RGBA at the configured output
    /// size. The CPU kernel is format-flexible per call (unlike the
    /// GPU pipeline, which fixes its input format at construction);
    /// the [`StitchExecutor`] trait covers the NV12 contract, this
    /// inherent entry covers planar YUV sources (file decode).
    pub fn stitch_yuv(&self, planes: &[YuvPlanes<'_>], pose: Pose) -> Result<Vec<u8>, StitchError> {
        super::cpu::stitch_rgba_yuv420p(
            self.projection(),
            planes,
            self.cam,
            &self.calib,
            &self.viewport_size,
            pose,
            self.full_range,
        )
    }
}

impl StitchExecutor for CpuExecutor {
    fn stitch(&mut self, planes: &[Nv12Planes<'_>], pose: Pose) -> Result<Vec<u8>, StitchError> {
        // Plane-size + camera-count validation lives in stitch_rgba, which
        // returns a typed error instead of panicking on a short/truncated frame.
        self.stitch_nv12(planes, pose)
    }

    fn output_dims(&self) -> (u32, u32) {
        (self.viewport_size.width, self.viewport_size.height)
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

/// Configuration for building a [`GpuExecutor`].
///
/// Owns everything the GPU pipeline needs to know about the frames it
/// will stitch: the calibration document, the output viewport, and the
/// source dimensions and pixel formats. The projection is the
/// document's topology (zero-copy). Engine-level concerns (detection,
/// trackers, replay) deliberately live on
/// [`StitchCore`](crate::core::StitchCore), not here.
#[cfg(feature = "gpu")]
pub struct GpuExecutorConfig {
    /// Camera calibration document.
    pub calibration: Calibration,
    /// Output viewport dimensions.
    pub viewport_size: ViewportSize,
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
    /// Whether source YUV uses full-range (JPEG) quantization.
    pub full_range: bool,
}

#[cfg(feature = "gpu")]
impl GpuExecutorConfig {
    /// New config with required fields only; defaults everywhere else
    /// (1080p viewport, `Rgba8Unorm` output, limited-range YUV).
    pub fn new(
        calibration: Calibration,
        input_width: u32,
        input_height: u32,
        input_format: InputFormat,
    ) -> Self {
        Self {
            calibration,
            viewport_size: ViewportSize {
                width: 1920,
                height: 1080,
            },
            input_width,
            input_height,
            input_format,
            output_format: wgpu::TextureFormat::Rgba8Unorm,
            full_range: false,
        }
    }
}

/// GPU executor - the sole owner of the wgpu `StitchPipeline` and of
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
    /// Resident-frame machinery: shared decode textures, the VRAM
    /// lookahead pool, decode backpressure. Populated lazily by the
    /// configure/stage methods; empty for pure CPU-frame consumers.
    pub(crate) residency: super::residency::Residency,
    /// Readback ring for the synchronous [`StitchExecutor::stitch`]
    /// path, created on first use so engine-embedded executors (which
    /// read back through the engine's own pipelined ring) never
    /// allocate it. Keyed by the output dims it was built for so a
    /// resize recreates it.
    sync_readback: Option<(RgbaReadback, (u32, u32))>,
    /// NV12 delivery: triple-buffered render-target -> NV12 readback
    /// for encoders and preview taps. Created on first use and keyed
    /// by the dims it was built for so a resize recreates it.
    nv12: Option<(crate::gpu::nv12_converter::Nv12Converter, (u32, u32))>,
}

#[cfg(feature = "gpu")]
impl GpuExecutor {
    /// Build a GPU executor. `gpu` is injected so reco-core does not
    /// pull an async runtime into non-test code; callers create it via
    /// [`GpuContext::new`].
    pub fn new(gpu: GpuContext, config: GpuExecutorConfig) -> Result<Self, StitchError> {
        let projection = config.calibration.topology.projection();
        // Fail fast on CPU-only projections: binding a placeholder
        // program would render garbage frames instead of an error.
        let Some(program) = projection.gpu_program() else {
            log::warn!(
                "GpuExecutor: projection '{}' has no GPU program yet; refusing construction (the CPU executor covers it)",
                projection.name()
            );
            return Err(StitchError::NoGpuProgram {
                projection: projection.name(),
            });
        };
        log::info!(
            "GpuExecutor: projection '{}' from the calibration topology supplies the GPU program and coverage",
            projection.name()
        );
        // Calibration validation happens once, inside with_gpu.
        let mut pipeline = StitchPipeline::with_gpu(
            gpu,
            &program,
            config.calibration,
            config.viewport_size,
            config.input_width,
            config.input_height,
            config.output_format,
            config.input_format,
        )?;
        pipeline.set_full_range(config.full_range);
        Ok(Self {
            pipeline,
            residency: super::residency::Residency::default(),
            sync_readback: None,
            nv12: None,
        })
    }

    // -----------------------------------------------------------------
    // Resident-frame surface (zero-copy sources, lookahead pool)
    // -----------------------------------------------------------------

    /// Wire the shared zero-copy decode textures into the pipeline:
    /// bind groups for rendering, views for detection and replay
    /// packing, texture clones for pool staging, CUDA pointers for
    /// GPU detection, and the decode backpressure channels.
    #[cfg(target_os = "linux")]
    pub(crate) fn configure_shared_textures(&mut self, shared: &crate::interop::SharedTextureSet) {
        let t = &shared.textures;
        let bind_groups = self.pipeline.configure_gpu_source(
            [(&t[0], &t[1]), (&t[2], &t[3])],
            [(&t[4], &t[5]), (&t[6], &t[7])],
        );
        let desc = wgpu::TextureViewDescriptor::default();
        self.residency.bind_groups = Some(bind_groups);
        self.residency.slot_free_tx = Some((
            shared.left_slot_free_tx.clone(),
            shared.right_slot_free_tx.clone(),
        ));
        self.residency.cuda_buf_info = Some((shared.left_buf.clone(), shared.right_buf.clone()));
        self.residency.shared_views = Some([
            t[0].texture.create_view(&desc),
            t[1].texture.create_view(&desc),
            t[2].texture.create_view(&desc),
            t[3].texture.create_view(&desc),
            t[4].texture.create_view(&desc),
            t[5].texture.create_view(&desc),
            t[6].texture.create_view(&desc),
            t[7].texture.create_view(&desc),
        ]);
        self.residency.shared_textures = Some([
            t[0].texture.clone(),
            t[1].texture.clone(),
            t[2].texture.clone(),
            t[3].texture.clone(),
            t[4].texture.clone(),
            t[5].texture.clone(),
            t[6].texture.clone(),
            t[7].texture.clone(),
        ]);
        log::info!("GpuExecutor: shared zero-copy decode textures configured");
    }

    /// CUDA buffer info for GPU detection on the shared decode
    /// textures, cloned so callers can hold it across `&mut` engine
    /// calls. `None` until a zero-copy source is configured.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) fn cuda_buf_info(
        &self,
    ) -> Option<(
        crate::interop::zero_copy::GpuBufInfo,
        crate::interop::zero_copy::GpuBufInfo,
    )> {
        self.residency.cuda_buf_info.clone()
    }

    /// Render from the shared decode textures at the given slots
    /// (immediate zero-copy path).
    #[cfg(target_os = "linux")]
    pub(crate) fn render_shared_slots(
        &mut self,
        left_slot: u8,
        right_slot: u8,
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, StitchError> {
        let bind_groups = self.residency.bind_groups.as_ref().ok_or_else(|| {
            StitchError::InvalidConfig(
                "GPU bind groups not configured - call setup_gpu_source() before run()".into(),
            )
        })?;
        Ok(self
            .pipeline
            .render_gpu_frame(bind_groups, left_slot, right_slot, pose)?)
    }

    /// Render from a VRAM lookahead pool slot (buffered path).
    pub(crate) fn render_pool_slot(
        &mut self,
        slot: usize,
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, StitchError> {
        let pool = self
            .residency
            .pool
            .as_ref()
            .expect("render_pool_slot requires the lookahead pool");
        Ok(self.pipeline.render_with_bind_groups(
            pool.left_bind_group(slot),
            pool.right_bind_group(slot),
            pose,
        )?)
    }

    /// Copy the shared decode slots into a pool slot so the decode
    /// surfaces can recycle while the frame waits in the lookahead
    /// buffer. The copy is awaited before returning. The decode slot
    /// is NOT released here - detection still reads it; the caller
    /// frees it via [`Self::release_decode_slots`] afterwards.
    #[cfg(target_os = "linux")]
    pub(crate) fn stage_shared_to_pool(
        &mut self,
        left_slot: usize,
        right_slot: usize,
    ) -> Result<Option<usize>, StitchError> {
        let residency = &mut self.residency;
        let (Some(pool), Some(shared_tex)) =
            (residency.pool.as_mut(), residency.shared_textures.as_ref())
        else {
            return Ok(None);
        };
        let slot = pool.acquire().ok_or_else(|| {
            StitchError::InvalidConfig(format!(
                "VRAM pool exhausted ({} slots, {} available)",
                pool.capacity(),
                pool.available()
            ))
        })?;
        let gpu = self.pipeline.gpu();
        pool.copy_from_textures(
            gpu,
            slot,
            &shared_tex[left_slot * 2],
            &shared_tex[left_slot * 2 + 1],
            &shared_tex[4 + right_slot * 2],
            &shared_tex[4 + right_slot * 2 + 1],
        );
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        Ok(Some(slot))
    }

    /// Copy four imported NV12 plane textures into a pool slot
    /// (DMA-buf / CVPixelBuffer sources whose import caches live
    /// outside the shared-texture set). The copy is awaited so the
    /// source may recycle its buffer immediately after.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
    pub(crate) fn stage_textures_to_pool(
        &mut self,
        left_y: &wgpu::Texture,
        left_uv: &wgpu::Texture,
        right_y: &wgpu::Texture,
        right_uv: &wgpu::Texture,
    ) -> Result<Option<usize>, StitchError> {
        let Some(pool) = self.residency.pool.as_mut() else {
            return Ok(None);
        };
        let slot = pool.acquire().ok_or_else(|| {
            StitchError::InvalidConfig(format!(
                "VRAM pool exhausted ({} slots, {} available)",
                pool.capacity(),
                pool.available()
            ))
        })?;
        let gpu = self.pipeline.gpu();
        pool.copy_from_textures(gpu, slot, left_y, left_uv, right_y, right_uv);
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        Ok(Some(slot))
    }

    /// Allocate the NVMM detection letterbox surfaces (Jetson).
    ///
    /// `model_size` is the detector's square input dimension (e.g.
    /// 1280); the source dimensions size the letterbox geometry.
    /// Without this the NVMM detection arm no-ops (the director still
    /// advances, just without detections).
    #[cfg(target_os = "linux")]
    pub(crate) fn setup_nvmm_detection(
        &mut self,
        model_size: u32,
        src_width: u32,
        src_height: u32,
    ) -> Result<(), String> {
        let left =
            crate::nvbuf_transform::NvBufDetectionSurface::new(model_size, src_width, src_height)
                .map_err(|e| format!("NVMM left detection surface: {e}"))?;
        let right =
            crate::nvbuf_transform::NvBufDetectionSurface::new(model_size, src_width, src_height)
                .map_err(|e| format!("NVMM right detection surface: {e}"))?;
        self.residency.nvmm_det = Some((left, right));
        log::info!(
            "GpuExecutor: NVMM detection surfaces ready: {model_size}x{model_size} \
             (src {src_width}x{src_height})"
        );
        Ok(())
    }

    /// Letterbox a stereo NVMM frame into the detection surfaces and
    /// wrap the results as per-camera detector frames. Returns `None`
    /// (logged) when the surfaces are not set up or a transform fails.
    #[cfg(target_os = "linux")]
    pub(crate) fn nvmm_detector_frames(
        &mut self,
        left: &crate::source::NvmmPlaneInfo,
        right: &crate::source::NvmmPlaneInfo,
    ) -> Option<
        [(
            crate::geometry::CameraId,
            crate::detect::detector::DetectorFrame<'static>,
        ); 2],
    > {
        use crate::detect::detector::DetectorFrame;
        use crate::geometry::CameraId;

        let (det_left, det_right) = self.residency.nvmm_det.as_mut()?;
        unsafe {
            if let Err(e) = det_left.transform_from_nvmm(left.surface_ptr) {
                log::warn!("NVMM left detection transform failed: {e}");
                return None;
            }
            if let Err(e) = det_right.transform_from_nvmm(right.surface_ptr) {
                log::warn!("NVMM right detection transform failed: {e}");
                return None;
            }
        }
        Some([
            (
                CameraId::Left,
                DetectorFrame::CudaRgbaLetterboxed {
                    ptr: det_left.data_ptr,
                    src_width: left.width,
                    src_height: left.height,
                },
            ),
            (
                CameraId::Right,
                DetectorFrame::CudaRgbaLetterboxed {
                    ptr: det_right.data_ptr,
                    src_width: right.width,
                    src_height: right.height,
                },
            ),
        ])
    }

    /// Import a stereo NVMM frame's DMA-bufs as Vulkan textures
    /// (cached by fd) and hand back Arc-backed clones of the four
    /// Y/UV plane textures `[left_y, left_uv, right_y, right_uv]`.
    #[cfg(target_os = "linux")]
    pub(crate) fn import_nvmm(
        &mut self,
        left: &crate::source::NvmmPlaneInfo,
        right: &crate::source::NvmmPlaneInfo,
    ) -> Result<[wgpu::Texture; 4], String> {
        if self.residency.nvmm_cache.is_none() {
            self.residency.nvmm_cache = Some(crate::interop::dmabuf::DmaBufTextureCache::new());
        }
        let gpu = self.pipeline.gpu();
        let cache = self.residency.nvmm_cache.as_mut().expect("created above");
        cache
            .ensure_imported(
                gpu,
                left.dmabuf_fd,
                left.width,
                left.height,
                left.y_offset,
                left.uv_offset,
                left.total_size,
            )
            .map_err(|e| format!("left NVMM DMA-buf import: {e}"))?;
        cache
            .ensure_imported(
                gpu,
                right.dmabuf_fd,
                right.width,
                right.height,
                right.y_offset,
                right.uv_offset,
                right.total_size,
            )
            .map_err(|e| format!("right NVMM DMA-buf import: {e}"))?;
        let l = cache.get(left.dmabuf_fd);
        let r = cache.get(right.dmabuf_fd);
        Ok([
            l.y_texture.clone(),
            l.uv_texture.clone(),
            r.y_texture.clone(),
            r.uv_texture.clone(),
        ])
    }

    /// Import a stereo NVMM frame and stage it into a pool slot for
    /// buffered rendering. The blit is awaited so the source may
    /// recycle the DMA-buf immediately after.
    #[cfg(target_os = "linux")]
    pub(crate) fn stage_nvmm_to_pool(
        &mut self,
        left: &crate::source::NvmmPlaneInfo,
        right: &crate::source::NvmmPlaneInfo,
    ) -> Result<Option<usize>, String> {
        if self.residency.pool.is_none() {
            return Ok(None);
        }
        let [ly, lu, ry, ru] = self.import_nvmm(left, right)?;
        self.stage_textures_to_pool(&ly, &lu, &ry, &ru)
            .map_err(|e| e.to_string())
    }

    /// Import a stereo CVPixelBuffer pair as Y/UV plane textures
    /// (`[left_y, left_uv, right_y, right_uv]`). The Metal texture
    /// cache is created on first use. Each returned plane keeps its
    /// `CVMetalTextureRef` alive - hold it until the GPU has read the
    /// frame.
    ///
    /// # Safety
    ///
    /// `left` and `right` must be valid, non-null `CVPixelBufferRef`s.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) unsafe fn import_metal(
        &mut self,
        left: crate::interop::metal::CVPixelBufferRef,
        right: crate::interop::metal::CVPixelBufferRef,
    ) -> Result<[crate::interop::metal::ImportedPlaneTexture; 4], String> {
        if self.residency.metal_cache.is_none() {
            let cache = crate::interop::metal::MetalTextureCache::new(self.pipeline.gpu())
                .map_err(|e| e.to_string())?;
            log::info!("Metal zero-copy: texture cache initialized");
            self.residency.metal_cache = Some(cache);
        }
        let gpu = self.pipeline.gpu();
        let cache = self.residency.metal_cache.as_mut().expect("created above");
        let (ly, lu) = unsafe { cache.import_nv12(left, gpu) }.map_err(|e| e.to_string())?;
        let (ry, ru) = unsafe { cache.import_nv12(right, gpu) }.map_err(|e| e.to_string())?;
        Ok([ly, lu, ry, ru])
    }

    /// Hand decode slots back to the decode threads. Call only after
    /// detection has read the slot - releasing earlier lets the decode
    /// thread overwrite the shared memory mid-read.
    #[cfg(target_os = "linux")]
    pub(crate) fn release_decode_slots(&self, left_slot: u8, right_slot: u8) {
        if let Some((ref left_tx, ref right_tx)) = self.residency.slot_free_tx {
            if left_tx.send(left_slot).is_err() {
                log::error!(
                    "failed to release left GPU decode slot {left_slot} - decode thread may have died"
                );
            }
            if right_tx.send(right_slot).is_err() {
                log::error!(
                    "failed to release right GPU decode slot {right_slot} - decode thread may have died"
                );
            }
        }
    }

    /// Drop the decode backpressure senders so decode threads see a
    /// closed channel and exit instead of blocking on `recv()`.
    #[cfg(target_os = "linux")]
    pub(crate) fn drop_decode_channels(&mut self) {
        self.residency.slot_free_tx = None;
    }

    /// Lazily create the D3D11VA staging pool, sized for
    /// `lookahead_frames` of buffering (0 = double-buffered stereo).
    ///
    /// Returns `true` if the pool was created by this call: the first
    /// staged frame performs cross-API warmup (device extraction,
    /// shared-handle imports), so callers skip rendering it.
    #[cfg(target_os = "windows")]
    pub(crate) fn ensure_d3d11_staging(
        &mut self,
        lookahead_frames: usize,
        needs_cuda: bool,
        pixel_format: crate::render::renderer::GpuPixelFormat,
    ) -> Result<bool, String> {
        if self.residency.d3d11_staging.is_some() {
            return Ok(false);
        }
        // For lookahead, size slots to the max frames simultaneously
        // in flight (decoded but not yet rendered), x2 for left+right.
        // Peak occupancy is n + post_smooth_half + 1 (the buffer hits
        // n+1 right after a produce while the pose queue holds
        // post_smooth_half). Slots are assigned by produce_index modulo
        // n_slots with no occupancy check, so the pool must exceed peak
        // occupancy or a producer would overwrite a frame still queued
        // for render. +4 keeps a few frames of slack above the exact
        // fit (the VramPool uses ref-counted acquire/release; this
        // path relies on the sizing margin instead). Without lookahead,
        // 4 slots (double-buffered stereo) suffice.
        let n_slots = if lookahead_frames > 0 {
            let post_smooth_half = (lookahead_frames / 2).max(1);
            (lookahead_frames + post_smooth_half + 4) * 2
        } else {
            4
        };
        let (w, h) = self.pipeline.source_info();
        let pool = crate::interop::d3d11::D3d11StagingPool::new(
            self.pipeline.gpu(),
            w,
            h,
            n_slots,
            needs_cuda,
            pixel_format,
        )
        .map_err(|e| format!("D3D11 staging pool: {e}"))?;
        log::info!("D3D11VA staging pool created: {w}x{h}, {n_slots} {pixel_format:?} slots");
        self.residency.d3d11_staging = Some(pool);
        Ok(true)
    }

    /// Stage a decoded D3D11VA stereo frame into the given pool slots.
    /// The first call extracts FFmpeg's device from the source texture
    /// and builds the staging textures on it.
    #[cfg(target_os = "windows")]
    pub(crate) fn stage_d3d11_frames(
        &mut self,
        left_texture: *mut std::ffi::c_void,
        left_slice: usize,
        right_texture: *mut std::ffi::c_void,
        right_slice: usize,
        left_slot: usize,
        right_slot: usize,
    ) -> Result<(), String> {
        let pool = self
            .residency
            .d3d11_staging
            .as_mut()
            .ok_or_else(|| "D3D11 staging pool not created".to_string())?;
        pool.stage_frame(left_texture, left_slice, left_slot)
            .map_err(|e| e.to_string())?;
        pool.stage_frame(right_texture, right_slice, right_slot)
            .map_err(|e| e.to_string())
    }

    /// Staging slots for a buffered frame, assigned round-robin by
    /// produce index (left in even slots, right in odd). `None` until
    /// the pool exists.
    #[cfg(target_os = "windows")]
    pub(crate) fn d3d11_slots(&self, produce_index: u64) -> Option<(usize, usize)> {
        let pool = self.residency.d3d11_staging.as_ref()?;
        let n = pool.n_slots();
        let i = produce_index as usize * 2;
        Some((i % n, (i + 1) % n))
    }

    /// Render from staged D3D11 pool slots.
    #[cfg(target_os = "windows")]
    pub(crate) fn render_d3d11_slots(
        &mut self,
        left_slot: usize,
        right_slot: usize,
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, StitchError> {
        let pool =
            self.residency.d3d11_staging.as_ref().ok_or_else(|| {
                StitchError::InvalidConfig("D3D11 staging pool not created".into())
            })?;
        Ok(self.pipeline.render_imported_views(
            pool.y_view(left_slot),
            pool.uv_view(left_slot),
            pool.y_view(right_slot),
            pool.uv_view(right_slot),
            pose,
        )?)
    }

    /// Y/UV detection views over two staged slots, Arc-cloned so
    /// callers can hold them across `&mut` engine calls. Layout
    /// `[left_y, left_uv, right_y, right_uv]`. `None` until the pool
    /// exists.
    #[cfg(target_os = "windows")]
    pub(crate) fn d3d11_views(
        &self,
        left_slot: usize,
        right_slot: usize,
    ) -> Option<[wgpu::TextureView; 4]> {
        let pool = self.residency.d3d11_staging.as_ref()?;
        Some([
            pool.y_view(left_slot).clone(),
            pool.uv_view(left_slot).clone(),
            pool.y_view(right_slot).clone(),
            pool.uv_view(right_slot).clone(),
        ])
    }

    /// Allocate the VRAM lookahead pool.
    pub(crate) fn create_lookahead_pool(
        &mut self,
        width: u32,
        height: u32,
        slots: usize,
        pixel_format: crate::render::renderer::GpuPixelFormat,
    ) -> Result<(), String> {
        let pool = crate::gpu::vram_pool::VramPool::new(
            self.pipeline.gpu(),
            &self.pipeline,
            width,
            height,
            slots,
            pixel_format,
        )?;
        self.residency.pool = Some(pool);
        Ok(())
    }

    /// Release a lookahead pool slot after its frame rendered.
    pub(crate) fn release_pool_slot(&mut self, slot: usize) {
        if let Some(pool) = self.residency.pool.as_mut() {
            pool.release(slot);
        }
    }

    // -----------------------------------------------------------------
    // NV12 delivery (encoders, preview taps)
    // -----------------------------------------------------------------

    /// NV12 output dimensions: the viewport rounded down to NV12-safe
    /// values (shared rounding rule with the CPU delivery path).
    pub(crate) fn nv12_dims(&self) -> (u32, u32) {
        let vp = self.pipeline.viewport_size();
        crate::render::nv12_cpu::nv12_dims(vp.width, vp.height)
    }

    /// Submit `render_commands` and convert the render target to NV12.
    ///
    /// Triple-buffered: returns `None` on the first two calls, then
    /// bytes from two frames ago. Drain the tail with
    /// [`Self::flush_nv12`] after the loop. The converter is created
    /// on first use at [`Self::nv12_dims`] and recreated on resize.
    pub(crate) fn convert_nv12(
        &mut self,
        render_commands: wgpu::CommandBuffer,
    ) -> Result<Option<&[u8]>, crate::gpu::nv12_converter::Nv12Error> {
        let dims = self.nv12_dims();
        if self.nv12.as_ref().is_none_or(|(_, built)| *built != dims) {
            let (w, h) = dims;
            let vp = self.pipeline.viewport_size();
            if (vp.width, vp.height) != dims {
                log::info!(
                    "GpuExecutor: NV12 delivery rounds {}x{} viewport to {w}x{h}",
                    vp.width,
                    vp.height
                );
            }
            let converter =
                crate::gpu::nv12_converter::Nv12Converter::new(self.pipeline.gpu(), w, h)?;
            log::info!("GpuExecutor: NV12 delivery initialized ({w}x{h})");
            self.nv12 = Some((converter, dims));
        }
        let (converter, _) = self.nv12.as_mut().expect("created above");
        converter.convert_and_readback(
            self.pipeline.gpu(),
            self.pipeline.render_target(),
            render_commands,
        )
    }

    /// Drain one pending NV12 frame from the triple buffer. `None`
    /// when nothing remains (or NV12 delivery was never used).
    pub(crate) fn flush_nv12(
        &mut self,
    ) -> Result<Option<&[u8]>, crate::gpu::nv12_converter::Nv12Error> {
        match self.nv12.as_mut() {
            Some((converter, _)) => converter.flush_pending(self.pipeline.gpu()),
            None => Ok(None),
        }
    }
}

#[cfg(feature = "gpu")]
impl StitchExecutor for GpuExecutor {
    fn stitch(&mut self, planes: &[Nv12Planes<'_>], pose: Pose) -> Result<Vec<u8>, StitchError> {
        // The GPU upload path is stereo-shaped today; mono programs
        // land with their own bind layout at the cylinder GPU step.
        let [left, right] = planes else {
            return Err(StitchError::InvalidConfig(format!(
                "the GPU executor's synchronous stitch consumes exactly 2                  camera frames, got {}",
                planes.len()
            )));
        };
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
        let cmd = self.pipeline.render_to_target_nv12(left, right, pose)?;
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
        let v = self.pipeline.viewport_size();
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

    /// The output viewport dimensions.
    pub fn viewport_size(&self) -> &ViewportSize {
        match self {
            Executor::Cpu(c) => &c.viewport_size,
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.viewport_size(),
        }
    }

    /// The projection in effect: a borrow of the calibration
    /// document's topology on either arm.
    pub fn projection(&self) -> &dyn Projection {
        match self {
            Executor::Cpu(c) => c.projection(),
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.projection(),
        }
    }

    /// The projection's coverage boundary over this executor's
    /// document: `projection().coverage(calibration())` always pair,
    /// so the executor offers the pairing directly.
    pub fn coverage(&self) -> CoverageBoundary {
        self.projection().coverage(self.calibration())
    }

    /// Source frame dimensions `(width, height)` per camera.
    pub fn source_info(&self) -> (u32, u32) {
        match self {
            Executor::Cpu(c) => c.cam,
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.source_info(),
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
            Executor::Gpu(g) => g.pipeline.resize(width, height),
        }
    }

    /// Set the seam blend width (document field; no geometry rebuild).
    pub fn set_blend_width(&mut self, width: f32) {
        match self {
            Executor::Cpu(c) => {
                if let Some(t) = c.calib.topology.l_shape_mut() {
                    t.blend_width = width;
                } else {
                    log::warn!("set_blend_width({width}) ignored: this topology has no seam");
                }
            }
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.set_blend_width(width),
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
            }
            #[cfg(feature = "gpu")]
            Executor::Gpu(g) => g.pipeline.update_camera_params(left, right),
        }
    }
}

impl StitchExecutor for Executor {
    fn stitch(&mut self, planes: &[Nv12Planes<'_>], pose: Pose) -> Result<Vec<u8>, StitchError> {
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
        use crate::geometry::resolve_render_pose;

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
            let coverage = cal.topology.projection().coverage(&cal);
            let cam = cal.topology.projection().virtual_camera(&cal.framing);
            let fov = (coverage.max_fov_degrees() * fov_factor).min(60.0);
            let config = ViewportSize {
                width: out_w,
                height: out_h,
            };
            let mut backend =
                CpuExecutor::new(cal.clone(), config, cam_w, cam_h, false).expect("cpu");

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
                let pose = Pose {
                    yaw: ry,
                    pitch: rp,
                    fov_degrees: fov,
                };
                let frac = black_frac(&backend.stitch(&[planes, planes], pose).unwrap());
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
            calib(w, h),
            ViewportSize {
                width: w,
                height: h,
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
            calib(w, h),
            ViewportSize {
                width: w,
                height: h,
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
        let err = backend
            .stitch(&[planes, planes], Pose::default())
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
        let pose = Pose {
            yaw: 0.08,
            pitch: -0.04,
            ..Default::default()
        };

        let mut cpu = CpuExecutor::new(calib.clone(), config.clone(), cam_w, cam_h, false)
            .expect("cpu backend");
        let mut gpu = GpuExecutor::new(
            gpu,
            GpuExecutorConfig {
                viewport_size: config,
                ..GpuExecutorConfig::new(calib, cam_w, cam_h, InputFormat::Nv12)
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
    fn gpu_backend_refuses_cpu_only_projections() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };

        // The cylinder has no GPU program yet: construction must fail
        // with the typed error, never bind a placeholder pipeline.
        let (cam_w, cam_h) = (192u32, 108u32);
        let cal = Calibration::new(
            vec![Lens::flat(cam_w, cam_h)],
            crate::projection::Cylinder::default(),
            Framing {
                axis_offset: 0.0,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        let Err(err) = GpuExecutor::new(
            gpu,
            GpuExecutorConfig::new(cal, cam_w, cam_h, InputFormat::Nv12),
        ) else {
            panic!("cylinder+GPU must fail fast");
        };
        assert!(matches!(
            err,
            StitchError::NoGpuProgram {
                projection: "cylindrical-mono-1camera"
            }
        ));
    }
}
