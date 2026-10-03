//! GPU executor: rendering, resident-frame imports and readback ownership.

use super::{StitchError, StitchExecutor};
use crate::calibration::Calibration;
use crate::geometry::RenderPose;
use crate::gpu::{GpuContext, rgba_readback::RgbaReadback};
use crate::render::planes::Nv12Planes;
use crate::render::viewport::ViewportSize;
use crate::render::{pipeline::StitchPipeline, renderer::InputFormat};

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
    /// readback.
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
    /// Resident-frame machinery: shared decode textures, the VRAM
    /// lookahead pool, decode backpressure. Populated lazily by the
    /// configure/stage methods; empty for pure CPU-frame consumers.
    pub(crate) residency: crate::stitch::residency::Residency,
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
        self.residency.bind_groups = Some(self.pipeline.configure_gpu_source(shared));
        self.residency.slot_free_tx = Some(
            shared
                .cameras
                .iter()
                .map(|c| c.slot_free_tx.clone())
                .collect(),
        );
        self.residency.cuda_buf_info = Some(shared.cameras.iter().map(|c| c.buf.clone()).collect());
        self.residency.shared_views = Some(
            shared
                .cameras
                .iter()
                .map(|c| {
                    std::array::from_fn(|i| c.textures[i].texture.create_view(&Default::default()))
                })
                .collect(),
        );
        self.residency.shared_textures = Some(
            shared
                .cameras
                .iter()
                .map(|c| std::array::from_fn(|i| c.textures[i].texture.clone()))
                .collect(),
        );
        log::info!("GpuExecutor: shared zero-copy decode textures configured");
    }

    /// CUDA buffer info for GPU detection on the shared decode
    /// textures, cloned so callers can hold it across `&mut` engine
    /// calls. `None` until a zero-copy source is configured.
    #[cfg(target_os = "linux")]
    pub(crate) fn cuda_buf_info(&self) -> Option<Vec<crate::interop::zero_copy::GpuBufInfo>> {
        self.residency.cuda_buf_info.clone()
    }

    /// Render from the shared decode textures at the given slots
    /// (immediate zero-copy path).
    #[cfg(target_os = "linux")]
    pub(crate) fn render_shared_slots(
        &mut self,
        slots: &[u8],
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, StitchError> {
        let bind_groups = self.residency.bind_groups.as_ref().ok_or_else(|| {
            StitchError::InvalidConfig(
                "GPU bind groups not configured - call setup_gpu_source() before run()".into(),
            )
        })?;
        Ok(self.pipeline.render_gpu_frame(bind_groups, slots, pose)?)
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

    /// Copy the shared decode slots into a pool slot so the decode
    /// surfaces can recycle while the frame waits in the lookahead
    /// buffer. The copy is awaited before returning. The decode slot
    /// is NOT released here - detection still reads it; the caller
    /// frees it via [`Self::release_decode_slots`] afterwards.
    #[cfg(target_os = "linux")]
    pub(crate) fn stage_shared_to_pool(
        &mut self,
        slots: &[u8],
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
        let sources: Vec<_> = shared_tex
            .iter()
            .zip(slots)
            .map(|(textures, &s)| {
                let s = usize::from(s);
                (&textures[s * 2], &textures[s * 2 + 1])
            })
            .collect();
        pool.copy_from_textures(gpu, slot, &sources);
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        Ok(Some(slot))
    }

    /// Copy imported NVMM plane textures into a pool slot.
    /// The copy is awaited so the
    /// source may recycle its buffer immediately after.
    #[cfg(target_os = "linux")]
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
        pool.copy_from_textures(gpu, slot, &[(left_y, left_uv), (right_y, right_uv)]);
        if let Err(error) = super::super::nvmm_completion::wait_for_reads(&gpu.device) {
            pool.release(slot);
            return Err(StitchError::InvalidConfig(format!(
                "NVMM staging completion: {error}"
            )));
        }
        Ok(Some(slot))
    }

    /// Allocate the NVMM detection letterbox surfaces (Jetson).
    ///
    /// `model_size` is the detector's square input dimension (e.g.
    /// 1280); the source dimensions size the letterbox geometry.
    /// Without this the NVMM detection arm no-ops (the panner still
    /// advances, just without detections).
    #[cfg(target_os = "linux")]
    pub(crate) fn setup_nvmm_detection(
        &mut self,
        model_size: u32,
        src_width: u32,
        src_height: u32,
    ) -> Result<(), String> {
        if self.pipeline.source_sizes() != [(src_width, src_height); 2] {
            return Err(
                "NVMM detection surface dimensions must match both configured cameras".into(),
            );
        }
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
        left: &crate::source::NvmmFrame,
        right: &crate::source::NvmmFrame,
    ) -> Option<
        [(
            crate::geometry::CameraIndex,
            crate::detect::detector::DetectorFrame<'static>,
        ); 2],
    > {
        use crate::detect::detector::DetectorFrame;

        let (left, right) = (left.info(), right.info());

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
                0,
                DetectorFrame::CudaRgbaLetterboxed {
                    ptr: det_left.data_ptr,
                    src_width: left.width,
                    src_height: left.height,
                },
            ),
            (
                1,
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
        self.stage_textures_to_pool(&ly, &lu, &ry, &ru)
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
    pub(crate) fn release_decode_slots(&self, slots: &[u8]) {
        if let Some(senders) = &self.residency.slot_free_tx {
            for (camera, (tx, &slot)) in senders.iter().zip(slots).enumerate() {
                if tx.send(slot).is_err() {
                    log::error!("failed to release camera {camera} GPU decode slot {slot}");
                }
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
    #[cfg(target_os = "windows")]
    pub(crate) fn ensure_d3d11_staging(
        &mut self,
        lookahead_frames: usize,
        pixel_format: crate::render::renderer::GpuPixelFormat,
    ) -> Result<(), String> {
        if self
            .pipeline
            .source_sizes()
            .windows(2)
            .any(|pair| pair[0] != pair[1])
        {
            return Err("D3D11 staging requires equal camera dimensions".into());
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
        let n_slots = crate::lookahead::pool_slots(lookahead_frames)?
            .checked_mul(2)
            .ok_or("D3D11 staging capacity overflow")?;
        if let Some(pool) = self.residency.d3d11_staging.as_ref() {
            if pool.n_slots() != n_slots {
                return Err("D3D11 staging size changed; create a new session".into());
            }
            return Ok(());
        }
        let (w, h) = self.pipeline.source_sizes()[0];
        let pool = crate::interop::d3d11::D3d11StagingPool::new(
            self.pipeline.gpu(),
            w,
            h,
            n_slots,
            pixel_format,
        )
        .map_err(|e| format!("D3D11 staging pool: {e}"))?;
        log::info!("D3D11VA staging pool created: {w}x{h}, {n_slots} {pixel_format:?} slots");
        self.residency.d3d11_staging = Some(pool);
        Ok(())
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
        pool.stage_frames(&[
            (left_texture, left_slice, left_slot),
            (right_texture, right_slice, right_slot),
        ])
        .map_err(|e| format!("D3D11 staging: {e}"))
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
        pose: RenderPose,
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
        if !matches!(&self.sync_readback, Some((_, d)) if *d == dims) {
            let ring = RgbaReadback::new(self.pipeline.gpu(), dims.0, dims.1)?;
            self.sync_readback = Some((ring, dims));
        }
        // Record the frame, submit it via the readback, then drain it
        // synchronously: one render in, this frame's RGBA out.
        let cmd = self.pipeline.render_nv12(planes, pose)?;
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
