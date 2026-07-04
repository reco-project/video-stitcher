//! Per-frame render and encode methods for [`StitchSession`].
//!
//! These methods are called once per frame to render a stereo pair,
//! convert to NV12, and fan out to attached encoders.

use super::StitchSession;
use crate::geometry::ViewportPosition;
use crate::session::types::{FrameLoopContext, SessionError};
use crate::source::StereoFrame;

impl StitchSession {
    /// Get the current viewport position from the director, or default.
    ///
    /// The engine resolves it: the panner's latest world-space decision
    /// through [`StitchCore::safe_clamp`](crate::core::StitchCore::safe_clamp)
    /// (the single pose-resolution authority - FOV cap, coverage clamp,
    /// roll-aware basis inversion), the `PosePresented` trace, and the
    /// FOV write-back. Panners output unconstrained positions.
    pub fn director_position(&mut self) -> ViewportPosition {
        self.core.presented_clamped_pose(self.frame_count)
    }

    /// Full per-frame pipeline: detect, pose, render, replay, telemetry.
    ///
    /// Dispatches to the correct detection and render path per
    /// [`StereoFrame`] variant. Every variant gets the same five stages;
    /// the dispatch inside each stage takes platform shortcuts (CUDA
    /// shared textures, Metal IOSurface import, D3D11 staging, etc.).
    ///
    /// `decode_time` and `frame_t0` are measured by the caller so that
    /// telemetry captures the full frame timing including source decode.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_process_frame_any")
    )]
    pub(crate) fn process_frame_any(
        &mut self,
        frame: &StereoFrame,
        elapsed: std::time::Duration,
        decode_time: std::time::Duration,
        frame_t0: std::time::Instant,
        ctx: &FrameLoopContext,
    ) -> Result<(), SessionError> {
        let _ = &ctx;

        // In the buffered lookahead path, detection already ran during
        // the produce phase. Skip it here and go straight to pose.
        let (ran_detection, detect_time) = if self.skip_detection {
            (false, std::time::Duration::ZERO)
        } else {
            let due = self.core.detection_due(self.frame_count);
            let detect_t0 = std::time::Instant::now();
            let ran_detection = match frame {
                #[cfg(target_os = "linux")]
                StereoFrame::GpuResident {
                    left_slot,
                    right_slot,
                } => {
                    if self.core.detector_needs_cuda_frames() {
                        if self.frame_count == 0 {
                            log::info!("GpuResident detection: CUDA path (TensorRT/ORT-CUDA)");
                        }
                        if let Some((left_buf, right_buf)) = &ctx.gpu_buf_info {
                            self.detect_and_update_director_gpu(
                                left_buf,
                                right_buf,
                                *left_slot,
                                *right_slot,
                                elapsed,
                            )?;
                            due
                        } else {
                            if self.frame_count == 0 {
                                log::warn!(
                                    "GpuResident frame but no gpu_buf_info - detection disabled, \
                                 director advancing without detections"
                                );
                            }
                            self.update_director(elapsed)?;
                            false
                        }
                    } else if let Some(views) = self.shared_views() {
                        if self.frame_count == 0 {
                            log::info!(
                                "GpuResident detection: wgpu shared texture views (ORT/wgpu preprocess)"
                            );
                        }
                        let ls = *left_slot as usize;
                        let rs = *right_slot as usize;
                        if due {
                            crate::profile_scope!("detect_wgpu_nv12");
                            let (w, h) = self.core.source_info();
                            let frames = super::detection_dispatch::wgpu_nv12_frames(
                                &views[ls * 2],
                                &views[ls * 2 + 1],
                                &views[4 + rs * 2],
                                &views[4 + rs * 2 + 1],
                                w,
                                h,
                                self.left_rotation,
                                self.right_rotation,
                            );
                            self.core.run_detection_frames(&frames);
                        }
                        self.fire_sink_and_update_director(elapsed, due)?;
                        due
                    } else {
                        if self.frame_count == 0 {
                            log::warn!(
                                "GpuResident frame but no shared views - detection disabled"
                            );
                        }
                        self.update_director(elapsed)?;
                        false
                    }
                }
                #[cfg(target_os = "linux")]
                StereoFrame::NvmmResident { left, right } => {
                    // Immediate (non-lookahead) NVMM detection: letterbox via
                    // NvBufSurfTransform, then detect. (The buffered path runs
                    // detection during produce and skips this whole step.)
                    if due {
                        crate::profile_scope!("detect_preletterboxed_total");
                        if let Some(frames) = self.gpu_exec().nvmm_detector_frames(left, right) {
                            self.core.run_detection_frames(&frames);
                        }
                    }
                    self.fire_sink_and_update_director(elapsed, due)?;
                    due
                }
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                StereoFrame::MetalResident { left, right } => {
                    self.detect_and_update_director_metal(
                        left.as_ptr(),
                        right.as_ptr(),
                        left.width(),
                        left.height(),
                        elapsed,
                    )?;
                    due
                }
                #[cfg(target_os = "windows")]
                StereoFrame::D3d11Resident { .. } => {
                    let left_slot = self.frame_count as usize % 2;
                    let right_slot = left_slot + 2;
                    if let Some(views) = self.gpu_exec_ref().d3d11_views(left_slot, right_slot) {
                        if due {
                            crate::profile_scope!("detect_wgpu_nv12");
                            let (w, h) = self.core.source_info();
                            let frames = super::detection_dispatch::wgpu_nv12_frames(
                                &views[0],
                                &views[1],
                                &views[2],
                                &views[3],
                                w,
                                h,
                                self.left_rotation,
                                self.right_rotation,
                            );
                            self.core.run_detection_frames(&frames);
                        }
                        self.fire_sink_and_update_director(elapsed, due)?;
                        due
                    } else {
                        self.update_director(elapsed)?;
                        false
                    }
                }
                _ => {
                    self.detect_and_update_director(frame, elapsed)?;
                    due
                }
            };
            let detect_time = detect_t0.elapsed();
            (ran_detection, detect_time)
        };

        // ── 2. Pose ────────────────────────────────────────────────
        let pos = self.director_position();

        // ── 3. Render + replay ─────────────────────────────────────
        #[allow(unused_mut)]
        let mut upload_time = std::time::Duration::ZERO;
        let render_t0 = std::time::Instant::now();
        match frame {
            #[cfg(target_os = "linux")]
            StereoFrame::GpuResident {
                left_slot,
                right_slot,
            } => {
                self.render_gpu_resident(*left_slot, *right_slot, pos.yaw, pos.pitch)?;
            }
            #[cfg(target_os = "linux")]
            StereoFrame::NvmmResident { left, right } => {
                if self.current_vram_slot.is_some() {
                    // Buffered path: the frame was already imported + copied
                    // into the pool slot during produce; render from it. The
                    // slot args are ignored when current_vram_slot is set.
                    self.render_gpu_resident(0, 0, pos.yaw, pos.pitch)?;
                } else {
                    // Immediate path: import the DMA-buf and render directly.
                    self.render_nvmm_immediate(left, right, pos.yaw, pos.pitch)?;
                }
            }
            #[cfg(target_os = "windows")]
            StereoFrame::D3d11Resident {
                left_texture,
                left_slice,
                right_texture,
                right_slice,
            } => {
                if let Some(staging_slot) = self.current_vram_slot {
                    // Buffered path: staging was done during produce.
                    // Render from the pre-staged slot.
                    self.render_d3d11_from_slot(
                        staging_slot,
                        staging_slot + 1,
                        pos.yaw,
                        pos.pitch,
                    )?;
                } else {
                    // Immediate path: stage and render now.
                    let _ = self
                        .core
                        .gpu()
                        .device()
                        .poll(wgpu::PollType::wait_indefinitely());
                    let staging_t0 = std::time::Instant::now();
                    let first = self.stage_d3d11_frames(
                        *left_texture,
                        *left_slice,
                        *right_texture,
                        *right_slice,
                    )?;
                    upload_time = staging_t0.elapsed();
                    if first {
                        return Ok(());
                    }
                    self.render_d3d11_staged(pos.yaw, pos.pitch)?;
                }
            }
            _ => {
                self.process_frame(frame, pos.yaw, pos.pitch)?;
            }
        }
        let render_time = render_t0.elapsed();

        // ── 4. Telemetry (uniform for all paths) ───────────────────
        let stitch_time = render_time
            .saturating_sub(upload_time)
            .saturating_sub(self.last_readback_time)
            .saturating_sub(self.last_submit_time);
        self.telemetry.record_frame(crate::telemetry::FrameTiming {
            decode: Some(decode_time),
            upload: Some(upload_time),
            detection: if ran_detection {
                Some(detect_time)
            } else {
                None
            },
            stitch: Some(stitch_time),
            readback: Some(self.last_readback_time),
            submit: Some(self.last_submit_time),
            total: Some(frame_t0.elapsed()),
            ..Default::default()
        });

        Ok(())
    }

    /// Render a single CPU-resident stereo frame and submit it to the encoder.
    ///
    /// Handles YUV420P and NV12 input formats. For GPU-resident frames
    /// (zero-copy path), use [`submit_render_output`](Self::submit_render_output)
    /// instead.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_process_frame")
    )]
    pub fn process_frame(
        &mut self,
        frame: &StereoFrame,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), SessionError> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let StereoFrame::MetalResident { left, right } = frame {
            return self.process_metal_frame(left, right, yaw, pitch);
        }

        let render_buf = self.core.render_stereo_frame_at_pose(frame, yaw, pitch)?;
        self.submit_render_output(render_buf)?;
        // GPU stacked-replay pack tap (M7). `render_stereo_frame_at_pose`
        // has just populated the renderer's internal plane textures
        // via `queue.write_texture`, so the packer's pipeline-view
        // path can read them. No-op when the packer isn't enabled.
        // Zero-copy `StereoFrame::GpuResident` goes through
        // `step_gpu_with_bufs` (Linux) which taps the pack with
        // external views instead.
        self.core.pack_replay_from_pipeline();
        Ok(())
    }

    /// Process a frame from GPU-resident RGBA textures (e.g. Bayer demosaic output).
    ///
    /// Copies the RGBA textures into the stitch pipeline's input planes,
    /// renders the stitch, converts to NV12, and submits to encoders.
    /// This is the Bayer/GPU-RGBA equivalent of `process_frame` for
    /// YUV/NV12 paths - session features (encoder fan-out, replay recording,
    /// frame counting) work automatically.
    pub fn process_frame_gpu_rgba(
        &mut self,
        left_rgba: &wgpu::Texture,
        right_rgba: &wgpu::Texture,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), SessionError> {
        let render_buf = self
            .core
            .render_gpu_rgba_at_pose(left_rgba, right_rgba, yaw, pitch);
        self.submit_render_output(render_buf)?;
        self.core.pack_replay_from_pipeline();
        Ok(())
    }

    /// Process a frame from imported NV12 textures (DMA-buf zero-copy path).
    ///
    /// Takes Y and UV textures for both cameras (from DMA-buf Vulkan import),
    /// renders the stitch, converts to NV12, and submits to encoders.
    /// Uses the imported textures directly for replay packing (not the
    /// renderer's internal planes, which aren't written by this path).
    pub fn process_frame_imported_nv12(
        &mut self,
        left_y: &wgpu::Texture,
        left_uv: &wgpu::Texture,
        right_y: &wgpu::Texture,
        right_uv: &wgpu::Texture,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), SessionError> {
        let render_buf = self
            .core
            .render_imported_textures_at_pose(left_y, left_uv, right_y, right_uv, yaw, pitch);
        self.submit_render_output(render_buf)?;

        // Replay pack from the imported views (not internal plane textures,
        // since render_imported_textures doesn't copy into them).
        let ly = left_y.create_view(&wgpu::TextureViewDescriptor::default());
        let lu = left_uv.create_view(&wgpu::TextureViewDescriptor::default());
        let ry = right_y.create_view(&wgpu::TextureViewDescriptor::default());
        let ru = right_uv.create_view(&wgpu::TextureViewDescriptor::default());
        self.core.pack_gpu_stacked_replay_from_views(
            crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 { y: &ly, uv: &lu },
            crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 { y: &ry, uv: &ru },
        );
        Ok(())
    }

    /// Process a MetalResident frame: import CVPixelBuffers as textures, render.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn process_metal_frame(
        &mut self,
        left: &crate::interop::metal::RetainedCVPixelBuffer,
        right: &crate::interop::metal::RetainedCVPixelBuffer,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), SessionError> {
        // SAFETY: RetainedCVPixelBuffer guarantees the pointers are valid.
        let [ly, lu, ry, ru] =
            unsafe { self.gpu_exec().import_metal(left.as_ptr(), right.as_ptr()) }
                .map_err(SessionError::ZeroCopy)?;
        // The imported planes must outlive the render + readback below
        // (each keeps its CVMetalTextureRef alive).
        self.process_frame_imported_nv12(
            &ly.texture,
            &lu.texture,
            &ry.texture,
            &ru.texture,
            yaw,
            pitch,
        )
    }

    /// Render a GpuResident frame: shared CUDA/Vulkan textures.
    ///
    /// Renders from the executor's resident-frame surface, packs
    /// replay from the shared texture views, and releases decode
    /// slots for thread reuse.
    #[cfg(target_os = "linux")]
    fn render_gpu_resident(
        &mut self,
        left_slot: u8,
        right_slot: u8,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), SessionError> {
        // VRAM pool path: render from the staged pool slot.
        // Decode slots were already freed during produce.
        if let Some(vram_idx) = self.current_vram_slot {
            let render_buf = self.gpu_exec().render_pool_slot(vram_idx, yaw, pitch);
            self.submit_render_output(render_buf)?;
            return Ok(());
        }

        // Shared texture path (non-buffered / immediate mode).
        let render_buf = self
            .gpu_exec()
            .render_shared_slots(left_slot, right_slot, yaw, pitch)
            .map_err(|e| SessionError::ZeroCopy(e.to_string()))?;
        self.submit_render_output(render_buf)?;

        if let Some(views) = self.shared_views() {
            let ls = left_slot as usize;
            let rs = right_slot as usize;
            self.core.pack_gpu_stacked_replay_from_views(
                crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 {
                    y: &views[ls * 2],
                    uv: &views[ls * 2 + 1],
                },
                crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 {
                    y: &views[4 + rs * 2],
                    uv: &views[4 + rs * 2 + 1],
                },
            );
        }

        self.gpu_exec().release_decode_slots(left_slot, right_slot);

        Ok(())
    }

    /// Clones of the executor's shared texture views (Arc-backed, so
    /// the clone is pointer-sized). Cloned out rather than borrowed
    /// because the callers feed them into `&mut self.core` methods
    /// (detection, replay pack) while the views live on the executor
    /// inside that same core.
    #[cfg(target_os = "linux")]
    pub(crate) fn shared_views(&self) -> Option<[wgpu::TextureView; 8]> {
        self.core
            .executor
            .gpu()
            .and_then(|g| g.residency.shared_views.clone())
    }

    /// Render an NVMM frame directly from imported DMA-buf textures
    /// (immediate / non-lookahead path).
    ///
    /// Imports both cameras' DMA-bufs into Vulkan textures (cached by fd)
    /// and renders via [`process_frame_imported_nv12`](Self::process_frame_imported_nv12).
    /// The buffered (lookahead) path uses the VRAM pool instead; this is the
    /// fallback when no pool exists (`lookahead == 0`).
    #[cfg(target_os = "linux")]
    fn render_nvmm_immediate(
        &mut self,
        left: &crate::source::NvmmPlaneInfo,
        right: &crate::source::NvmmPlaneInfo,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), SessionError> {
        let [ly, lu, ry, ru] = self
            .gpu_exec()
            .import_nvmm(left, right)
            .map_err(SessionError::ZeroCopy)?;
        self.process_frame_imported_nv12(&ly, &lu, &ry, &ru, yaw, pitch)
    }

    /// Stage D3D11VA decoded frames into the executor's staging pool
    /// (immediate path, double-buffered slots).
    ///
    /// Returns `true` on the first call (pool just created) to signal
    /// that this frame should be skipped (cross-API warmup).
    #[cfg(target_os = "windows")]
    fn stage_d3d11_frames(
        &mut self,
        left_texture: *mut std::ffi::c_void,
        left_slice: usize,
        right_texture: *mut std::ffi::c_void,
        right_slice: usize,
    ) -> Result<bool, SessionError> {
        let needs_cuda = self.core.detector_needs_cuda_frames();
        let lookahead_frames = self.lookahead_frames;
        let pixel_format = self.gpu_pixel_format;
        let first_frame = self
            .gpu_exec()
            .ensure_d3d11_staging(lookahead_frames, needs_cuda, pixel_format)
            .map_err(SessionError::ZeroCopy)?;
        let left_pool_slot = self.frame_count as usize % 2;
        let right_pool_slot = left_pool_slot + 2;
        self.gpu_exec()
            .stage_d3d11_frames(
                left_texture,
                left_slice,
                right_texture,
                right_slice,
                left_pool_slot,
                right_pool_slot,
            )
            .map_err(SessionError::ZeroCopy)?;
        Ok(first_frame)
    }

    /// Render from specific D3D11 staging slots (buffered lookahead path).
    #[cfg(target_os = "windows")]
    fn render_d3d11_from_slot(
        &mut self,
        left_slot: usize,
        right_slot: usize,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), SessionError> {
        let render_buf = self
            .gpu_exec()
            .render_d3d11_slots(left_slot, right_slot, yaw, pitch)
            .map_err(|e| SessionError::ZeroCopy(e.to_string()))?;
        self.submit_render_output(render_buf)
    }

    /// Render from already-staged D3D11VA views (immediate path).
    #[cfg(target_os = "windows")]
    fn render_d3d11_staged(&mut self, yaw: f32, pitch: f32) -> Result<(), SessionError> {
        let left_pool_slot = self.frame_count as usize % 2;
        let right_pool_slot = left_pool_slot + 2;
        self.render_d3d11_from_slot(left_pool_slot, right_pool_slot, yaw, pitch)?;

        if let Some(views) = self
            .gpu_exec_ref()
            .d3d11_views(left_pool_slot, right_pool_slot)
        {
            self.core.pack_gpu_stacked_replay_from_views(
                crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 {
                    y: &views[0],
                    uv: &views[1],
                },
                crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 {
                    y: &views[2],
                    uv: &views[3],
                },
            );
        }

        Ok(())
    }

    /// Submit a recorded render and fan the NV12 result out to the
    /// encoders and the NV12 tap.
    ///
    /// Used with the zero-copy paths where decode threads write
    /// directly to GPU textures: the executor's render methods produce
    /// the command buffer, and this delivers it.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_submit_render")
    )]
    pub fn submit_render_output(
        &mut self,
        render_commands: wgpu::CommandBuffer,
    ) -> Result<(), SessionError> {
        // Field-path borrow: `nv12_data` borrows the executor inside
        // `core` for the rest of the function, while the encode fan-out
        // below touches only session-owned fields.
        let (nv12_width, nv12_height) = self.gpu_exec_ref().nv12_dims();
        let readback_t0 = std::time::Instant::now();
        let nv12_data = self
            .core
            .executor
            .gpu_mut()
            .expect("the streaming session runs on the GPU executor")
            .convert_nv12(render_commands)?;
        self.last_readback_time = readback_t0.elapsed();

        // First two calls return None (triple-buffer warmup).
        // From the third call onward, we get data from 2 frames ago.
        let encode_t0 = std::time::Instant::now();
        if let Some(data) = nv12_data {
            if let Some(ref encoder) = self.encoder {
                encoder.submit(data, self.frame_count as i64)?;
            }
            for enc in &self.extra_encoders {
                enc.submit(data, self.frame_count as i64)?;
            }
            // NV12 tap for snapshot / preview hooks (reco-cli's periodic
            // JPEG writer). Runs after encode submit; the callback is
            // expected to be non-blocking (try_send on a channel).
            if let Some(ref mut tap) = self.nv12_tap {
                tap(data, nv12_width, nv12_height);
            }
        }
        self.last_submit_time = encode_t0.elapsed();

        self.frame_count += 1;
        Ok(())
    }

    /// Convert a pre-rendered frame to NV12 without encoding.
    ///
    /// Returns NV12 data from 2 frames ago (or `None` on the first two calls).
    /// Used by the preview path where the caller displays frames directly
    /// instead of encoding them.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_convert_nv12")
    )]
    pub fn convert_to_nv12(
        &mut self,
        render_commands: wgpu::CommandBuffer,
    ) -> Result<Option<&[u8]>, SessionError> {
        let nv12_data = self
            .core
            .executor
            .gpu_mut()
            .expect("the streaming session runs on the GPU executor")
            .convert_nv12(render_commands)?;
        self.frame_count += 1;
        Ok(nv12_data)
    }

    /// Copy a GPU-resident frame to the VRAM pool if available.
    ///
    /// Returns `Some(slot)` if the frame was copied to the pool (the
    /// decode surface can be freed). Returns `None` for CPU frames
    /// or when no pool is configured.
    /// Copy a GPU-resident frame to a persistent buffer slot.
    ///
    /// On Linux: copies from shared CUDA/Vulkan textures to VramPool.
    /// On Windows: stages D3D11 frame to an expanded staging pool slot.
    /// Returns the slot index for rendering, or None for CPU frames.
    pub(crate) fn copy_to_vram_pool(
        &mut self,
        frame: &StereoFrame,
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
    pub(crate) fn release_gpu_decode_slot(&self, frame: &StereoFrame) {
        #[cfg(target_os = "linux")]
        {
            if let StereoFrame::GpuResident {
                left_slot,
                right_slot,
            } = frame
            {
                self.gpu_exec_ref()
                    .release_decode_slots(*left_slot, *right_slot);
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = frame;
        }
    }

    #[cfg(target_os = "linux")]
    fn copy_to_vram_pool_platform(
        &mut self,
        frame: &StereoFrame,
        _produce_index: u64,
    ) -> Result<Option<usize>, SessionError> {
        // NVMM zero-copy: import the DMA-buf as Vulkan textures and copy
        // them into a VRAM pool slot - the same import-then-copy pattern
        // as the macOS Metal arm, just sourced from an NvBufSurface fd
        // instead of a CVPixelBuffer.
        if let StereoFrame::NvmmResident { left, right } = frame {
            return self
                .gpu_exec()
                .stage_nvmm_to_pool(left, right)
                .map_err(SessionError::ZeroCopy);
        }
        let (ls, rs) = match frame {
            StereoFrame::GpuResident {
                left_slot,
                right_slot,
            } => (*left_slot as usize, *right_slot as usize),
            _ => return Ok(None),
        };
        // The decode slot is NOT released here. Detection reads it
        // after this copy (see `produce_one`), so the slot must stay
        // held until `release_gpu_decode_slot` is called
        // post-detection.
        self.gpu_exec()
            .stage_shared_to_pool(ls, rs)
            .map_err(|e| SessionError::Config(e.to_string()))
    }

    #[cfg(target_os = "windows")]
    fn copy_to_vram_pool_platform(
        &mut self,
        frame: &StereoFrame,
        produce_index: u64,
    ) -> Result<Option<usize>, SessionError> {
        if let StereoFrame::D3d11Resident {
            left_texture,
            left_slice,
            right_texture,
            right_slice,
        } = frame
        {
            let needs_cuda = self.core.detector_needs_cuda_frames();
            let lookahead_frames = self.lookahead_frames;
            let pixel_format = self.gpu_pixel_format;
            self.gpu_exec()
                .ensure_d3d11_staging(lookahead_frames, needs_cuda, pixel_format)
                .map_err(SessionError::ZeroCopy)?;
            let (left_slot, right_slot) = self
                .gpu_exec_ref()
                .d3d11_slots(produce_index)
                .expect("staging pool created above");
            self.gpu_exec()
                .stage_d3d11_frames(
                    *left_texture,
                    *left_slice,
                    *right_texture,
                    *right_slice,
                    left_slot,
                    right_slot,
                )
                .map_err(SessionError::ZeroCopy)?;
            return Ok(Some(left_slot));
        }
        Ok(None)
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn copy_to_vram_pool_platform(
        &mut self,
        frame: &StereoFrame,
        _produce_index: u64,
    ) -> Result<Option<usize>, SessionError> {
        if let StereoFrame::MetalResident { left, right } = frame {
            // SAFETY: RetainedCVPixelBuffer guarantees the pointers are valid.
            let [ly, lu, ry, ru] =
                unsafe { self.gpu_exec().import_metal(left.as_ptr(), right.as_ptr()) }
                    .map_err(SessionError::ZeroCopy)?;
            // The staging copy is awaited before this returns, so the
            // imported planes may drop at the end of this arm.
            return self
                .gpu_exec()
                .stage_textures_to_pool(&ly.texture, &lu.texture, &ry.texture, &ru.texture)
                .map_err(|e| SessionError::Config(e.to_string()));
        }
        Ok(None)
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "windows",
        target_os = "macos",
        target_os = "ios"
    )))]
    fn copy_to_vram_pool_platform(
        &mut self,
        _frame: &StereoFrame,
        _produce_index: u64,
    ) -> Result<Option<usize>, SessionError> {
        Ok(None)
    }
}
