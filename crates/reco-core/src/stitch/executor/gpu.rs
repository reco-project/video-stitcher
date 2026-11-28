//! GPU executor: rendering, resident-frame imports and readback ownership.

use super::{StitchError, StitchExecutor};
use crate::calibration::Calibration;
use crate::geometry::RenderPose;
use crate::gpu::{
    GpuContext,
    rgba_readback::RgbaReadback,
    yuv_converter::{YuvConvertError, YuvConverter},
};
use crate::render::planes::Nv12Planes;
use crate::render::viewport::ViewportSize;
use crate::render::{pipeline::StitchPipeline, renderer::InputFormat};
use crate::sink::{DeliveryFormat, OutputFrame};
use std::collections::VecDeque;

/// Image storage and its associated timestamps have one invalidation lifetime.
struct GpuDelivery {
    converter: YuvConverter,
    pending_pts: VecDeque<Option<i64>>,
}

/// Configuration for building a [`GpuExecutor`].
///
/// Owns everything the GPU pipeline needs to know about the frames it
/// will stitch: the calibration document, the output viewport, and the
/// source dimensions and pixel formats. The projection is the
/// document's projection (zero-copy). Engine-level concerns (detection,
/// trackers, replay) deliberately live on
/// [`StitchCore`](crate::core::StitchCore), not here.
pub struct GpuExecutorConfig {
    /// Camera calibration document.
    pub calibration: Calibration,
    /// Output viewport dimensions.
    pub viewport_size: ViewportSize,
    /// Fixed decoded `(width, height)` for each camera in projection order.
    pub source_sizes: Vec<(u32, u32)>,
    /// Input pixel format.
    pub input_format: InputFormat,
    /// GPU render-target format. `Rgba8Unorm` suits every compositor
    /// consumer; `Bgra8Unorm` matches native Windows DirectX surfaces
    /// for consumers that prefer to swizzle on upload instead of on
    /// readback. `Rgb10a2Unorm` selects P010 delivery and requires even viewport dimensions.
    ///
    /// Planar (L-shape) GPU compositing tracks base-surface coverage in
    /// destination alpha and requires a 4-channel color format with an alpha
    /// channel (such as `Rgba8Unorm`, `Bgra8Unorm`, `Rgb10a2Unorm`, or
    /// `Rgba16Float`). Alpha-free formats (such as `R8Unorm`, `Rg8Unorm`, or
    /// `Rg11b10Ufloat`) are rejected at construction with
    /// [`StitchError::Pipeline`] wrapping
    /// [`PipelineError::InvalidConfig`](crate::render::pipeline::PipelineError::InvalidConfig).
    pub output_format: wgpu::TextureFormat,
    /// Whether source YUV uses full-range (JPEG) quantization.
    pub full_range: bool,
}

impl GpuExecutorConfig {
    /// New config with required fields only; defaults everywhere else
    /// (1080p viewport, `Rgba8Unorm` output, limited-range YUV).
    pub fn new(
        calibration: Calibration,
        source_sizes: Vec<(u32, u32)>,
        input_format: InputFormat,
    ) -> Self {
        Self {
            calibration,
            viewport_size: ViewportSize {
                width: 1920,
                height: 1080,
            },
            source_sizes,
            input_format,
            output_format: wgpu::TextureFormat::Rgba8Unorm,
            full_range: false,
        }
    }
}

/// GPU executor - the sole owner of the wgpu `StitchPipeline` and of
/// the projection geometry bound to it.
///
/// [`StitchCore`](crate::core::StitchCore) holds one of these as its
/// render substrate and drives the streaming paths (pipelined readback,
/// zero-copy imports, preview-to-view) through it. The synchronous
/// [`StitchExecutor::stitch`] path renders one frame and blocks on a private readback ring,
/// for callers that want "planes in, RGBA out" with no pipelining.
pub struct GpuExecutor {
    pub(crate) pipeline: StitchPipeline,
    /// Native imports and lookahead staging. Populated lazily by the
    /// configure/stage methods; empty for pure CPU-frame consumers.
    pub(crate) residency: crate::stitch::residency::Residency,
    /// Readback ring for the synchronous [`StitchExecutor::stitch`]
    /// path, created on first use so engine-embedded executors (which
    /// read back through the engine's own pipelined ring) never
    /// allocate it. Invalidated when output dimensions change.
    sync_readback: Option<RgbaReadback>,
    /// Triple-buffered delivery for encoders and preview taps.
    /// Invalidated on resize, including when rounded NV12 dimensions are unchanged.
    delivery: Option<GpuDelivery>,
    delivery_format: DeliveryFormat,
    delivery_color: crate::render::color::YuvColor,
}

impl GpuExecutor {
    /// Build a GPU executor. `gpu` is injected so reco-core does not
    /// pull an async runtime into non-test code; callers create it via
    /// [`GpuContext::new`].
    pub fn new(gpu: GpuContext, config: GpuExecutorConfig) -> Result<Self, StitchError> {
        let projection = config.calibration.projection.geometry();
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
            "GpuExecutor: projection '{}' from the calibration projection supplies the GPU program and coverage",
            projection.name()
        );
        let delivery_format = if config.output_format == wgpu::TextureFormat::Rgb10a2Unorm {
            YuvConverter::validate_layout(
                &gpu,
                config.viewport_size.width,
                config.viewport_size.height,
                DeliveryFormat::P010,
            )
            .map_err(|error| StitchError::InvalidConfig(error.to_string()))?;
            DeliveryFormat::P010
        } else {
            DeliveryFormat::Nv12
        };
        // Calibration validation happens once, inside with_gpu.
        let mut pipeline = StitchPipeline::with_gpu(
            gpu,
            &program,
            config.calibration,
            config.viewport_size,
            config.source_sizes,
            config.output_format,
            config.input_format,
        )?;
        pipeline.set_full_range(config.full_range);
        Ok(Self {
            pipeline,
            residency: crate::stitch::residency::Residency::default(),
            sync_readback: None,
            delivery: None,
            delivery_format,
            delivery_color: Default::default(),
        })
    }

    pub(super) fn set_source_color(&mut self, color: crate::render::color::YuvColor) {
        self.pipeline.set_source_color(color);
        self.set_delivery_matrix(color.matrix);
    }

    pub(crate) fn set_delivery_matrix(&mut self, matrix: crate::render::color::YuvMatrix) {
        let delivery = crate::render::color::YuvColor::delivery(matrix);
        if self.delivery_color != delivery {
            self.delivery = None;
            self.delivery_color = delivery;
        }
    }

    /// Accept output dimensions and invalidate every delivery queue before another render.
    pub(super) fn resize(&mut self, width: u32, height: u32) -> Option<(u32, u32)> {
        if self.delivery_format == DeliveryFormat::P010 {
            YuvConverter::validate_layout(self.pipeline.gpu(), width, height, self.delivery_format)
                .ok()?;
        }
        let changed = self.output_dims() != (width, height);
        let accepted = self.pipeline.resize(width, height)?;
        if changed {
            self.sync_readback = None;
            self.delivery = None;
        }
        Some(accepted)
    }

    /// Render from a VRAM lookahead pool slot (buffered path).
    #[cfg(target_os = "linux")]
    pub(crate) fn render_pool_slot(
        &mut self,
        slot: usize,
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, StitchError> {
        let pool = self
            .residency
            .pool
            .as_ref()
            .expect("render_pool_slot requires the lookahead pool");
        Ok(self
            .pipeline
            .render_with_bind_groups(pool.bind_groups(slot), pose)?)
    }

    /// Copy native plane textures into the lookahead pool and await completion.
    #[cfg(target_os = "linux")]
    pub(crate) fn stage_textures_to_pool(
        &mut self,
        cameras: &[(&wgpu::Texture, &wgpu::Texture)],
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
        pool.copy_from_textures(gpu, slot, cameras);
        if let Err(error) = crate::gpu::wait_for_reads(&gpu.device) {
            pool.release(slot);
            return Err(StitchError::InvalidConfig(error));
        }
        Ok(Some(slot))
    }

    /// Import a stereo NVMM frame's DMA-bufs as Vulkan textures
    /// (cached by fd) and hand back Arc-backed clones of the four
    /// Y/UV plane textures `[left_y, left_uv, right_y, right_uv]`.
    #[cfg(target_os = "linux")]
    pub(crate) fn import_nvmm(
        &mut self,
        left: &crate::source::NvmmFrame,
        right: &crate::source::NvmmFrame,
    ) -> Result<[wgpu::Texture; 4], String> {
        let (left, right) = (left.info(), right.info());
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
        left: &crate::source::NvmmFrame,
        right: &crate::source::NvmmFrame,
    ) -> Result<Option<usize>, String> {
        if self.residency.pool.is_none() {
            return Ok(None);
        }
        let [ly, lu, ry, ru] = self.import_nvmm(left, right)?;
        self.stage_textures_to_pool(&[(&ly, &lu), (&ry, &ru)])
            .map_err(|e| e.to_string())
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn hold_nvmm_reads(
        &mut self,
        frames: [crate::source::NvmmFrame; 2],
    ) -> Result<super::super::nvmm_completion::NvmmRead, String> {
        if self.residency.nvmm_completion.is_none() {
            self.residency.nvmm_completion = Some(
                super::super::nvmm_completion::NvmmCompletion::new(self.pipeline.gpu())?,
            );
        }
        Ok(self
            .residency
            .nvmm_completion
            .as_ref()
            .expect("created above")
            .hold(frames))
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn finish_nvmm_reads(&mut self) -> Result<(), String> {
        let result = self
            .residency
            .nvmm_completion
            .take()
            .map_or(Ok(()), |mut completion| completion.finish());
        // Do not retain imports from a stopped source across later runs.
        self.residency.nvmm_cache = None;
        result
    }

    /// Import retained cameras using one cache; keep imports until submitted GPU reads finish.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) fn import_metal(
        &mut self,
        cameras: &[crate::interop::metal::RetainedCVPixelBuffer],
    ) -> Result<crate::interop::metal::ImportedFrame, String> {
        if cameras.len() != self.pipeline.source_sizes().len()
            || cameras
                .iter()
                .zip(self.pipeline.source_sizes())
                .any(|(frame, &size)| (frame.width(), frame.height()) != size)
        {
            return Err("Metal camera count or dimensions do not match the executor".into());
        }
        if self.residency.metal_cache.is_none() {
            self.residency.metal_cache = Some(
                crate::interop::metal::MetalTextureCache::new(self.pipeline.gpu())
                    .map_err(|e| e.to_string())?,
            );
        }
        self.residency
            .metal_cache
            .as_mut()
            .expect("created above")
            .import_frame(cameras, self.pipeline.gpu())
            .map_err(|e| e.to_string())
    }

    /// Lazily allocate complete frame slots, including every camera's own extent.
    #[cfg(target_os = "windows")]
    pub(crate) fn ensure_d3d11_staging(
        &mut self,
        lookahead_frames: usize,
        pixel_format: crate::render::renderer::GpuPixelFormat,
    ) -> Result<(), String> {
        let slots = crate::lookahead::pool_slots(lookahead_frames)?;
        let sizes = self.pipeline.source_sizes();
        if let Some(pool) = self.residency.d3d11_staging.as_ref() {
            if !pool.matches(sizes, slots, pixel_format) {
                return Err("D3D11 staging layout changed; create a new session".into());
            }
            return Ok(());
        }
        self.residency.d3d11_staging = Some(
            crate::interop::d3d11::D3d11StagingPool::new(
                self.pipeline.gpu(),
                sizes,
                slots,
                pixel_format,
            )
            .map_err(|e| format!("D3D11 staging pool: {e}"))?,
        );
        Ok(())
    }

    /// Stage all cameras into one frame slot and wait for native copy completion.
    #[cfg(target_os = "windows")]
    pub(crate) fn stage_d3d11_frames(
        &mut self,
        frames: &[crate::source::D3d11CameraFrame],
        slot: usize,
    ) -> Result<(), String> {
        let pool = self
            .residency
            .d3d11_staging
            .as_mut()
            .ok_or_else(|| "D3D11 staging pool not created".to_string())?;
        pool.stage_frames(frames, slot)
            .map_err(|e| format!("D3D11 staging: {e}"))
    }

    /// Complete frame slot assigned round-robin by sequence number.
    #[cfg(target_os = "windows")]
    pub(crate) fn d3d11_slot(&self, sequence: u64) -> Option<usize> {
        Some(self.residency.d3d11_staging.as_ref()?.slot(sequence))
    }

    /// Camera-ordered plane views, cloned for use across mutable engine calls.
    #[cfg(target_os = "windows")]
    pub(crate) fn d3d11_views(
        &self,
        slot: usize,
    ) -> Option<Vec<(wgpu::TextureView, wgpu::TextureView)>> {
        self.residency.d3d11_staging.as_ref()?.views(slot)
    }

    /// Allocate the VRAM lookahead pool.
    #[cfg(target_os = "linux")]
    pub(crate) fn create_lookahead_pool(
        &mut self,
        slots: usize,
        pixel_format: crate::render::renderer::GpuPixelFormat,
    ) -> Result<(), String> {
        let pool = crate::gpu::vram_pool::VramPool::new(
            self.pipeline.gpu(),
            &self.pipeline,
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
    // Packed delivery (encoders, preview taps)
    // -----------------------------------------------------------------

    /// Packed CPU format produced for encoding.
    pub fn delivery_format(&self) -> DeliveryFormat {
        self.delivery_format
    }

    fn delivery_dims(&self) -> (u32, u32) {
        let vp = self.pipeline.viewport_size();
        self.delivery_format.viewport_dims(vp.width, vp.height)
    }

    /// Allocate converter storage before the first submission if needed.
    pub(crate) fn prepare_delivery(&mut self) -> Result<(), YuvConvertError> {
        if self.delivery.is_none() {
            let (w, h) = self.delivery_dims();
            let vp = self.pipeline.viewport_size();
            if (vp.width, vp.height) != (w, h) {
                log::info!(
                    "GpuExecutor: NV12 delivery rounds {}x{} viewport to {w}x{h}",
                    vp.width,
                    vp.height
                );
            }
            let converter = YuvConverter::with_format(
                self.pipeline.gpu(),
                w,
                h,
                self.delivery_format,
                self.delivery_color,
            )?;
            log::info!(
                "GpuExecutor: {:?} delivery initialized ({w}x{h})",
                self.delivery_format
            );
            self.delivery = Some(GpuDelivery {
                converter,
                pending_pts: VecDeque::with_capacity(3),
            });
        }
        Ok(())
    }

    /// Submit the render and return the oldest frame once two frames are pending.
    pub(crate) fn convert_delivery(
        &mut self,
        render_commands: wgpu::CommandBuffer,
        pts_us: Option<i64>,
    ) -> Result<Option<OutputFrame<'_>>, YuvConvertError> {
        self.prepare_delivery()?;
        let dims = self.delivery_dims();
        let format = self.delivery_format;
        let delivery = self.delivery.as_mut().expect("created above");
        let data = delivery.converter.convert_and_readback(
            self.pipeline.gpu(),
            self.pipeline.render_target(),
            render_commands,
        )?;
        delivery.pending_pts.push_back(pts_us);
        Ok(data.map(|data| delivery_frame(data, dims, format, &mut delivery.pending_pts)))
    }

    /// Drain one pending frame, or return `None` when the queue is empty.
    pub(crate) fn flush_delivery(&mut self) -> Result<Option<OutputFrame<'_>>, YuvConvertError> {
        let dims = self.delivery_dims();
        let format = self.delivery_format;
        match self.delivery.as_mut() {
            Some(delivery) => {
                let data = delivery.converter.flush_pending(self.pipeline.gpu())?;
                Ok(data.map(|data| delivery_frame(data, dims, format, &mut delivery.pending_pts)))
            }
            None => Ok(None),
        }
    }
}

fn delivery_frame<'a>(
    data: &'a [u8],
    (width, height): (u32, u32),
    format: DeliveryFormat,
    pending_pts: &mut VecDeque<Option<i64>>,
) -> OutputFrame<'a> {
    OutputFrame {
        data,
        width,
        height,
        format: format.pixel_format(),
        pts_us: pending_pts
            .pop_front()
            .expect("each converted image has a queued timestamp"),
    }
}

impl StitchExecutor for GpuExecutor {
    fn stitch(
        &mut self,
        planes: &[Nv12Planes<'_>],
        pose: RenderPose,
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
        if self.sync_readback.is_none() {
            let ring = RgbaReadback::new(self.pipeline.gpu(), dims.0, dims.1)?;
            self.sync_readback = Some(ring);
        }
        // Record the frame, submit it via the readback, then drain it
        // synchronously: one render in, this frame's RGBA out.
        let cmd = self.pipeline.render_nv12(planes, pose)?;
        let tex = self.pipeline.render_target();
        let ring = self.sync_readback.as_mut().expect("created above");
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
