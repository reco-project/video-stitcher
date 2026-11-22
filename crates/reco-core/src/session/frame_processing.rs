//! Per-frame render and encode methods for [`StitchSession`].
//!
//! These methods are called once per frame to render a frame set,
//! convert to NV12, and fan out to attached sinks.

use super::StitchSession;
use crate::geometry::RenderPose;
use crate::session::types::{FrameLoopContext, SessionError};
use crate::source::FrameSet;

impl StitchSession {
    /// Resolve the latest world-space camera decision for the next submitted frame.
    ///
    /// Manual source loops call this after detection and panner updates, before an explicit render-at-pose method. `frame_seconds` is the output frame duration, independent of processing time. The batch loop calls this automatically after lookahead smoothing, for every frame residency.
    pub fn advance_render_pose(&mut self, frame_seconds: f32) -> Result<RenderPose, SessionError> {
        Ok(self.core.advance_pose(
            self.core.requested_world_pose,
            self.frame_count,
            frame_seconds,
        )?)
    }

    /// Full per-frame pipeline: detect, pose, render, replay, telemetry.
    ///
    /// Dispatches to the correct detection and render path per
    /// [`FrameSet`] variant. Every variant gets the same five stages;
    /// the dispatch inside each stage takes platform shortcuts (CUDA
    /// shared textures, Metal IOSurface import, D3D11 staging, etc.).
    ///
    /// `timing` carries prior producer work; `frame_t0` starts this processing segment.
    /// Queue residence is excluded from the accumulated host processing total.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_process_frame_any")
    )]
    pub(crate) fn process_frame_any(
        &mut self,
        frame: Option<&FrameSet>,
        elapsed: std::time::Duration,
        pts_us: Option<i64>,
        mut timing: crate::telemetry::FrameTiming,
        frame_t0: std::time::Instant,
        ctx: &FrameLoopContext,
    ) -> Result<(), SessionError> {
        #[allow(unused_mut)]
        let mut upload_time = std::time::Duration::ZERO;
        #[cfg(all(feature = "gpu", target_os = "windows"))]
        if let Some(FrameSet::D3d11Resident(cameras)) = frame
            && self.current_vram_slot.is_none()
        {
            // Finish previous GPU reads before reusing the staging slots. Detection
            // and rendering must both consume the newly staged frame set.
            self.core
                .gpu()
                .device()
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| SessionError::ZeroCopy(e.to_string()))?;
            let staging_t0 = std::time::Instant::now();
            self.stage_d3d11_frames(cameras, self.frame_count)?;
            upload_time = staging_t0.elapsed();
        }

        timing.upload = Some(timing.upload.unwrap_or_default() + upload_time);

        // In the buffered lookahead path, detection already ran during
        // the produce phase. Skip it here and go straight to pose.
        if !ctx.tracking_complete {
            let expiry_t0 = std::time::Instant::now();
            self.core
                .expire_observations(ctx.capture.as_ref(), ctx.frame_seconds)?;
            let expiry_time = expiry_t0.elapsed();
            let capture_times_us = ctx
                .capture
                .as_ref()
                .map_or(&[][..], |capture| capture.times_us.as_slice());
            let frame =
                frame.ok_or_else(|| SessionError::Config("missing input for detection".into()))?;
            let due = self.core.detection_due(self.frame_count);
            let detect_t0 = std::time::Instant::now();
            let ran_detection = match frame {
                #[cfg(all(feature = "gpu", target_os = "linux"))]
                FrameSet::GpuResident { slots } => {
                    if self.core.detector_prefers_cuda_nv12() {
                        if self.frame_count == 0 {
                            log::info!("GpuResident detection: CUDA path (TensorRT/ORT-CUDA)");
                        }
                        if let Some(buffers) = &ctx.gpu_buf_info {
                            if due {
                                crate::profile_scope!("gpu_detect_total");
                                let frames = super::detection_dispatch::cuda_nv12_frames(
                                    buffers,
                                    slots,
                                    [self.left_rotation, self.right_rotation],
                                    self.source_full_range,
                                );
                                self.core.run_detection_frames(&frames, capture_times_us)?;
                            }
                            due
                        } else {
                            if self.frame_count == 0 {
                                log::warn!(
                                    "GpuResident frame but no gpu_buf_info - detection disabled, \
                                 panner advancing without detections"
                                );
                            }
                            false
                        }
                    } else if let Some(views) = self.shared_views() {
                        if self.frame_count == 0 {
                            log::info!(
                                "GpuResident detection: wgpu shared texture views (ORT/wgpu preprocess)"
                            );
                        }
                        if due {
                            crate::profile_scope!("detect_wgpu_nv12");
                            let frames = super::detection_dispatch::shared_nv12_frames(
                                &views,
                                slots,
                                self.core.source_sizes(),
                                [self.left_rotation, self.right_rotation],
                            );
                            self.core.run_detection_frames(&frames, capture_times_us)?;
                        }
                        due
                    } else {
                        if self.frame_count == 0 {
                            log::warn!(
                                "GpuResident frame but no shared views - detection disabled"
                            );
                        }
                        false
                    }
                }
                #[cfg(all(feature = "gpu", target_os = "linux"))]
                FrameSet::NvmmResident(cameras) => {
                    if due {
                        let frames = super::detection_dispatch::nvmm_frames(cameras);
                        self.core.run_detection_frames(&frames, capture_times_us)?;
                    }
                    due
                }
                #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios")))]
                FrameSet::MetalResident(cameras) => {
                    if due {
                        let frames = super::detection_dispatch::metal_frames(cameras);
                        self.core.run_detection_frames(&frames, capture_times_us)?;
                    }
                    due
                }
                #[cfg(all(feature = "gpu", target_os = "windows"))]
                FrameSet::D3d11Resident(..) => {
                    let views = self
                        .gpu_exec_ref()
                        .d3d11_slot(self.frame_count)
                        .and_then(|slot| self.gpu_exec_ref().d3d11_views(slot));
                    if let Some(views) = views {
                        if due {
                            crate::profile_scope!("detect_wgpu_nv12");
                            let frames = super::detection_dispatch::wgpu_nv12_frames(
                                &views,
                                self.core.source_sizes(),
                                [self.left_rotation, self.right_rotation],
                            );
                            self.core.run_detection_frames(&frames, capture_times_us)?;
                        }
                        due
                    } else {
                        false
                    }
                }
                _ => {
                    if due {
                        self.core.run_cpu_detection(frame, capture_times_us)?;
                    }
                    due
                }
            };
            timing.detection = Some(if ran_detection {
                detect_t0.elapsed()
            } else {
                std::time::Duration::ZERO
            });
            let tracking_t0 = std::time::Instant::now();
            self.update_world_pose(elapsed)?;
            timing.tracking = Some(expiry_time + tracking_t0.elapsed());
        }

        // Resolve the presented render-space pose.
        let pose_t0 = std::time::Instant::now();
        let pos = self.advance_render_pose(ctx.frame_seconds)?;
        timing.tracking = Some(timing.tracking.unwrap_or_default() + pose_t0.elapsed());

        // ── 3. Render + replay ─────────────────────────────────────
        let render_t0 = std::time::Instant::now();
        match frame {
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            Some(FrameSet::GpuResident { slots }) => {
                self.render_gpu_resident(slots, pos, pts_us)?;
            }
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            Some(FrameSet::NvmmResident([left, right])) => {
                self.render_nvmm_immediate(left, right, pos, pts_us)?;
            }
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            None if self.current_vram_slot.is_some() && ctx.tracking_complete => {
                self.render_gpu_resident(&[], pos, pts_us)?;
            }
            #[cfg(all(feature = "gpu", target_os = "windows"))]
            frame
                if matches!(frame, Some(FrameSet::D3d11Resident(_)))
                    || (frame.is_none()
                        && self.current_vram_slot.is_some()
                        && ctx.tracking_complete) =>
            {
                let slot = self
                    .current_vram_slot
                    .or_else(|| self.gpu_exec_ref().d3d11_slot(self.frame_count))
                    .ok_or_else(|| {
                        SessionError::ZeroCopy("D3D11 staging pool not created".into())
                    })?;
                let views = self
                    .gpu_exec_ref()
                    .d3d11_views(slot)
                    .ok_or_else(|| SessionError::ZeroCopy("D3D11 frame slot not staged".into()))?;
                let cameras: Vec<_> = views.iter().map(|(y, uv)| (y, uv)).collect();
                self.render_frame_imported_nv12(&cameras, pos, pts_us)?;
            }
            Some(frame) => {
                self.render_frame(frame, pos, pts_us)?;
            }
            None => {
                return Err(SessionError::Config(
                    "missing staged input for rendering".into(),
                ));
            }
        }
        let render_time = render_t0.elapsed();

        // ── 4. Telemetry (uniform for all paths) ───────────────────
        let stitch_time = render_time
            .saturating_sub(self.last_readback_time)
            .saturating_sub(self.last_submit_time);
        timing.stitch = Some(stitch_time);
        timing.readback = Some(self.last_readback_time);
        timing.submit = Some(self.last_submit_time);
        timing.total = Some(timing.total.unwrap_or_default() + frame_t0.elapsed());
        self.telemetry.record_frame(timing);
        self.publish_telemetry(false);

        Ok(())
    }

    /// Render a frame set at the supplied pose, deliver output to sinks and record replay when enabled.
    ///
    /// Handles CPU YUV420P/NV12 and, on Apple platforms, Metal-resident input. Other GPU inputs use the residency-specific render methods. Does not run detection, update trackers or resolve coverage.
    /// `pts_us` follows [`crate::sink::OutputFrame::pts_us`] and stays with this image through delayed readback.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_render_frame")
    )]
    pub fn render_frame(
        &mut self,
        frame: &FrameSet,
        pose: RenderPose,
        pts_us: Option<i64>,
    ) -> Result<(), SessionError> {
        #[cfg(feature = "gpu")]
        if self.core.executor.gpu().is_some() {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            if let FrameSet::MetalResident(cameras) = frame {
                return self.process_metal_frame(cameras, pose, pts_us);
            }
            let render_buf = self.core.render_frame_set_at_pose(frame, pose)?;
            self.submit_render_output(render_buf, pts_us)?;
            self.core.pack_replay_from_pipeline();
            return Ok(());
        }

        let (width, height) = self.nv12_delivery_dims();
        let render_width = self.core.executor.viewport_size().width;
        let crate::core::types::RenderOutcome::Rgba(rgba) =
            self.core.submit_frame_at_pose(frame, pose)?
        else {
            unreachable!("software rendering is synchronous");
        };
        let convert_t0 = std::time::Instant::now();
        crate::render::nv12_cpu::rgba_to_nv12_into(
            rgba,
            render_width,
            width,
            height,
            &mut self.cpu_nv12,
        )?;
        self.last_readback_time = convert_t0.elapsed();
        let submit_t0 = std::time::Instant::now();
        super::sinks::deliver_frame(&mut self.sinks, &self.cpu_nv12, width, height, pts_us)?;
        self.last_submit_time = submit_t0.elapsed();
        self.frame_count += 1;
        Ok(())
    }

    /// Process a frame from GPU-resident RGBA textures (e.g. Bayer demosaic output).
    ///
    /// Copies the RGBA textures into the stitch pipeline's input planes,
    /// renders the stitch, converts to NV12, and delivers to the sinks.
    /// This is the Bayer/GPU-RGBA equivalent of `render_frame` for
    /// YUV/NV12 paths - session features (sink fan-out, replay recording,
    /// frame counting) work automatically.
    /// `pts_us` follows [`crate::sink::OutputFrame::pts_us`].
    #[cfg(feature = "gpu")]
    pub fn render_frame_gpu_rgba(
        &mut self,
        left_rgba: &wgpu::Texture,
        right_rgba: &wgpu::Texture,
        pose: RenderPose,
        pts_us: Option<i64>,
    ) -> Result<(), SessionError> {
        let render_buf = self
            .core
            .render_gpu_rgba_at_pose(left_rgba, right_rgba, pose)?;
        self.submit_render_output(render_buf, pts_us)?;
        self.core.pack_replay_from_pipeline();
        Ok(())
    }

    /// Render camera-ordered imported Y/UV views and deliver the output to sinks.
    ///
    /// Each input supplies full-size Y and half-size UV views on this device in matching NV12 or P010 format. Views must select the correct planes. Keep native owners alive and prevent producer reuse through GPU completion. Stereo replay uses the supplied views directly. `pts_us` follows [`crate::sink::OutputFrame::pts_us`].
    #[cfg(feature = "gpu")]
    pub fn render_frame_imported_nv12(
        &mut self,
        cameras: &[(&wgpu::TextureView, &wgpu::TextureView)],
        pose: RenderPose,
        pts_us: Option<i64>,
    ) -> Result<(), SessionError> {
        let render_buf = self.core.render_imported_nv12_at_pose(cameras, pose)?;
        self.submit_render_output(render_buf, pts_us)?;

        if let [(ly, lu), (ry, ru)] = cameras {
            self.core.pack_gpu_stacked_replay_from_views(
                crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 { y: ly, uv: lu },
                crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 { y: ry, uv: ru },
            );
        }
        Ok(())
    }

    /// Process a MetalResident frame: import CVPixelBuffers as textures, render.
    #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios")))]
    fn process_metal_frame(
        &mut self,
        cameras: &[crate::interop::metal::RetainedCVPixelBuffer],
        pose: RenderPose,
        pts_us: Option<i64>,
    ) -> Result<(), SessionError> {
        let imported = self
            .gpu_exec()
            .import_metal(cameras)
            .map_err(SessionError::ZeroCopy)?;
        let views = imported.views();
        let cameras: Vec<_> = views.iter().map(|(y, uv)| (y, uv)).collect();
        self.render_frame_imported_nv12(&cameras, pose, pts_us)
    }

    /// Render a GpuResident frame: shared CUDA/Vulkan textures.
    ///
    /// Renders from the executor's resident-frame surface, packs
    /// replay from the shared texture views, and releases decode
    /// slots for thread reuse.
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    fn render_gpu_resident(
        &mut self,
        slots: &[u8],
        pose: RenderPose,
        pts_us: Option<i64>,
    ) -> Result<(), SessionError> {
        // VRAM pool path: render from the staged pool slot.
        // Decode slots were already freed during produce.
        if let Some(vram_idx) = self.current_vram_slot {
            let render_buf = self
                .gpu_exec()
                .render_pool_slot(vram_idx, pose)
                .map_err(crate::core::types::StitchCoreError::from)?;
            self.submit_render_output(render_buf, pts_us)?;
            if self.core.stacked_atlas_dims().is_some()
                && let Some([[ly, lu], [ry, ru]]) = self
                    .gpu_exec_ref()
                    .residency
                    .pool
                    .as_ref()
                    .and_then(|p| p.stereo_views(vram_idx))
            {
                use crate::gpu::yuv_stack_packer::StackedPackSource::Nv12;
                self.core.pack_gpu_stacked_replay_from_views(
                    Nv12 { y: &ly, uv: &lu },
                    Nv12 { y: &ry, uv: &ru },
                );
            }
            return Ok(());
        }

        // Shared texture path (non-buffered / immediate mode).
        let render_buf = self
            .gpu_exec()
            .render_shared_slots(slots, pose)
            .map_err(|e| SessionError::ZeroCopy(e.to_string()))?;
        self.submit_render_output(render_buf, pts_us)?;

        if let Some(views) = self.shared_views()
            && let ([left, right], [ls, rs]) = (views.as_slice(), slots)
        {
            let ls = usize::from(*ls);
            let rs = usize::from(*rs);
            self.core.pack_gpu_stacked_replay_from_views(
                crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 {
                    y: &left[ls * 2],
                    uv: &left[ls * 2 + 1],
                },
                crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 {
                    y: &right[rs * 2],
                    uv: &right[rs * 2 + 1],
                },
            );
        }

        // NV12 readback may only wait for an older frame. CUDA may reuse
        // these external textures only after this frame's GPU reads finish.
        self.core
            .gpu()
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| SessionError::ZeroCopy(e.to_string()))?;
        self.gpu_exec().release_decode_slots(slots);

        Ok(())
    }

    /// Clones of the executor's shared texture views (Arc-backed, so
    /// the clone is pointer-sized). Cloned out rather than borrowed
    /// because the callers feed them into `&mut self.core` methods
    /// (detection, replay pack) while the views live on the executor
    /// inside that same core.
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    pub(crate) fn shared_views(&self) -> Option<Vec<[wgpu::TextureView; 4]>> {
        self.core
            .executor
            .gpu()
            .and_then(|g| g.residency.shared_views.clone())
    }

    /// Render an NVMM frame directly from imported DMA-buf textures
    /// (immediate / non-lookahead path).
    ///
    /// Imports both cameras' DMA-bufs into Vulkan textures (cached by fd)
    /// and renders via [`render_frame_imported_nv12`](Self::render_frame_imported_nv12).
    /// The buffered (lookahead) path uses the VRAM pool instead; this is the
    /// fallback when no pool exists (`lookahead == 0`).
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    fn render_nvmm_immediate(
        &mut self,
        left: &crate::source::NvmmFrame,
        right: &crate::source::NvmmFrame,
        pose: RenderPose,
        pts_us: Option<i64>,
    ) -> Result<(), SessionError> {
        let _reads = self
            .gpu_exec()
            .hold_nvmm_reads([left.clone(), right.clone()])
            .map_err(SessionError::ZeroCopy)?;
        let [ly, lu, ry, ru] = self
            .gpu_exec()
            .import_nvmm(left, right)
            .map_err(SessionError::ZeroCopy)?;
        let [ly_view, lu_view, ry_view, ru_view] = [&ly, &lu, &ry, &ru]
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.render_frame_imported_nv12(&[(&ly_view, &lu_view), (&ry_view, &ru_view)], pose, pts_us)
    }

    /// Stage a complete native frame set before detection or rendering reads it.
    #[cfg(all(feature = "gpu", target_os = "windows"))]
    fn stage_d3d11_frames(
        &mut self,
        cameras: &[crate::source::D3d11CameraFrame],
        sequence: u64,
    ) -> Result<usize, SessionError> {
        let lookahead_frames = self.lookahead_frames;
        let pixel_format = self.gpu_pixel_format;
        self.gpu_exec()
            .ensure_d3d11_staging(lookahead_frames, pixel_format)
            .map_err(SessionError::ZeroCopy)?;
        let slot = self
            .gpu_exec_ref()
            .d3d11_slot(sequence)
            .expect("staging pool created above");
        self.gpu_exec()
            .stage_d3d11_frames(cameras, slot)
            .map_err(SessionError::ZeroCopy)?;
        Ok(slot)
    }

    /// Submit a recorded render and fan the NV12 result out to the
    /// attached sinks.
    ///
    /// Used with the zero-copy paths where decode threads write
    /// directly to GPU textures: the executor's render methods produce
    /// the command buffer, and this delivers it.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_submit_render")
    )]
    #[cfg(feature = "gpu")]
    pub(crate) fn submit_render_output(
        &mut self,
        render_commands: wgpu::CommandBuffer,
        pts_us: Option<i64>,
    ) -> Result<(), SessionError> {
        // Field-path borrow: the output borrows the executor inside
        // `core` for the rest of the function, while the sink fan-out
        // below touches only session-owned fields.
        let readback_t0 = std::time::Instant::now();
        let output = self
            .core
            .executor
            .gpu_mut()
            .expect("the streaming session runs on the GPU executor")
            .convert_nv12(render_commands, pts_us)?;
        self.last_readback_time = readback_t0.elapsed();

        // First two calls return None (triple-buffer warmup).
        // From the third call onward, we get data from 2 frames ago.
        let submit_t0 = std::time::Instant::now();
        if let Some(frame) = output {
            super::sinks::deliver_frame(
                &mut self.sinks,
                frame.data,
                frame.width,
                frame.height,
                frame.pts_us,
            )?;
        }
        self.last_submit_time = submit_t0.elapsed();

        self.frame_count += 1;
        Ok(())
    }

    /// Stage borrowed native images for buffered rendering.
    ///
    /// On Linux: copies from shared CUDA/Vulkan textures to VramPool.
    /// On Windows: stages D3D11 frame to an expanded staging pool slot.
    /// Returns the rendering slot, or None for owned CPU/Metal frames.
    #[cfg(feature = "gpu")]
    pub(crate) fn copy_to_vram_pool(
        &mut self,
        frame: &FrameSet,
        produce_index: u64,
    ) -> Result<Option<usize>, SessionError> {
        self.copy_to_vram_pool_platform(frame, produce_index)
    }

    /// Release decode-pool slots back to the decode thread.
    ///
    /// Must be called only AFTER detection has finished reading the
    /// decode slot. In the buffered produce path the frame is copied
    /// into the VramPool, then detection reads the original decode
    /// slot; releasing it earlier lets the decode thread overwrite the
    /// slot mid-read (use-after-free on the shared GPU memory).
    ///
    /// No-op on platforms that do not use the GPU decode slot-free
    /// channel (Windows D3D11 stages into a persistent pool; macOS
    /// imports CVPixelBuffers).
    #[cfg(feature = "gpu")]
    pub(crate) fn release_gpu_decode_slot(&self, frame: &FrameSet) {
        #[cfg(target_os = "linux")]
        {
            if let FrameSet::GpuResident { slots } = frame {
                self.gpu_exec_ref().release_decode_slots(slots);
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = frame;
        }
    }

    #[cfg(all(feature = "gpu", target_os = "linux"))]
    fn copy_to_vram_pool_platform(
        &mut self,
        frame: &FrameSet,
        _produce_index: u64,
    ) -> Result<Option<usize>, SessionError> {
        // NVMM zero-copy: import the DMA-buf as Vulkan textures and copy
        // them into a VRAM pool slot - the same import-then-copy pattern
        // as the macOS Metal arm, just sourced from an NvBufSurface fd
        // instead of a CVPixelBuffer.
        if let FrameSet::NvmmResident([left, right]) = frame {
            return self
                .gpu_exec()
                .stage_nvmm_to_pool(left, right)
                .map_err(SessionError::ZeroCopy);
        }
        let slots = match frame {
            FrameSet::GpuResident { slots } => slots,
            _ => return Ok(None),
        };
        // The decode slot is NOT released here. Detection reads it
        // after this copy (see `produce_one`), so the slot must stay
        // held until `release_gpu_decode_slot` is called
        // post-detection.
        self.gpu_exec()
            .stage_shared_to_pool(slots)
            .map_err(|e| SessionError::Config(e.to_string()))
    }

    #[cfg(all(feature = "gpu", target_os = "windows"))]
    fn copy_to_vram_pool_platform(
        &mut self,
        frame: &FrameSet,
        produce_index: u64,
    ) -> Result<Option<usize>, SessionError> {
        if let FrameSet::D3d11Resident(cameras) = frame {
            return self.stage_d3d11_frames(cameras, produce_index).map(Some);
        }
        Ok(None)
    }

    #[cfg(all(feature = "gpu", not(any(target_os = "linux", target_os = "windows"))))]
    fn copy_to_vram_pool_platform(
        &mut self,
        _frame: &FrameSet,
        _produce_index: u64,
    ) -> Result<Option<usize>, SessionError> {
        Ok(None)
    }
}
