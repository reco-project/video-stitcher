//! Per-frame render and encode methods for [`StitchSession`].
//!
//! These methods are called once per frame to render a stereo pair,
//! convert to NV12, and fan out to attached encoders.

use super::StitchSession;
use crate::detect::director::ViewportPosition;
use crate::session::types::{FrameLoopContext, SessionError};
use crate::source::StereoFrame;

impl StitchSession {
    /// Get the current viewport position from the director, or default.
    ///
    /// Clamps the panner's raw output to the coverage boundary (no-black
    /// region) and applies FOV limits. This keeps all viewport
    /// constraining in the session, so panners can output unconstrained
    /// positions.
    pub fn director_position(&mut self) -> ViewportPosition {
        // Source the raw pre-clamp pose from the panner's most recent
        // decision. When no panner is attached the previous pose stays
        // at its default (identity) value so the viewport centers.
        let mut pos = self.previous_panner_pose;

        // The panner outputs world-space coordinates (from detections
        // mapped via camera_to_panorama). Clamp in world space, then
        // convert to the user-space pitch the renderer expects (the
        // view_matrix applies rig_tilt as a basis rotation, so the
        // render-site pitch must compensate via rig_correction).
        if let Some(coverage) = self.core.coverage() {
            if let Some(ref mut fov) = pos.fov_degrees {
                *fov = fov.min(coverage.max_fov_degrees());
            }
            let fov = pos
                .fov_degrees
                .unwrap_or_else(|| self.core.pipeline().fov());
            let aspect = self.core.pipeline().viewport().aspect_ratio();
            let rig_tilt = self.core.pipeline().viewport().rig_tilt;
            // Clamp in world space (rig_tilt=0 so coverage stays in
            // the panorama's native coordinate system).
            let clamped = coverage.safe_clamp(pos.yaw, pos.pitch, fov, aspect, 0.0);
            pos.yaw = clamped.yaw;
            // Convert world (yaw, pitch) to render-space via exact
            // quaternion inversion of view_matrix's tilt+roll basis.
            // Accounts for roll coupling at non-zero yaw that the
            // closed-form render_pitch misses.
            let cam =
                crate::projection::VirtualCamera::new(&self.core.pipeline().scene.camera_position);
            let rig_roll = self.core.pipeline().viewport().rig_roll;
            let (ry, rp) = crate::lens::rig_correction::world_to_render_pose(
                &cam,
                clamped.yaw,
                clamped.pitch,
                rig_tilt,
                rig_roll,
            );
            pos.yaw = ry;
            pos.pitch = rp;
        }

        // Trace: PosePresented. This is the pose the renderer will
        // actually consume for this frame (post-clamp, post-FOV-cap).
        if let Some(sink) = self.event_sink.as_deref_mut() {
            sink.emit(
                crate::detect::pipeline_event::PipelineEvent::PosePresented {
                    frame_index: self.frame_count,
                    pose: pos,
                },
            );
        }

        if let Some(fov) = pos.fov_degrees {
            self.core.pipeline_mut().set_fov(fov);
        }
        pos
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
            let scheduled_detection =
                self.detection.has_detector() && self.detection.should_detect(self.frame_count);
            let detect_t0 = std::time::Instant::now();
            let ran_detection = match frame {
                #[cfg(target_os = "linux")]
                StereoFrame::GpuResident {
                    left_slot,
                    right_slot,
                } => {
                    if self.detection.needs_cuda_frames() {
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
                            scheduled_detection
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
                    } else if let Some(ref views) = self.gpu_shared_views {
                        if self.frame_count == 0 {
                            log::info!(
                                "GpuResident detection: wgpu shared texture views (ORT/wgpu preprocess)"
                            );
                        }
                        let ls = *left_slot as usize;
                        let rs = *right_slot as usize;
                        let (w, h) = self.core.pipeline().source_info();
                        let lr = self.left_rotation;
                        let rr = self.right_rotation;
                        if scheduled_detection {
                            let detections = self.detection.run_detection_wgpu_nv12(
                                &views[ls * 2],
                                &views[ls * 2 + 1],
                                &views[4 + rs * 2],
                                &views[4 + rs * 2 + 1],
                                w,
                                h,
                                lr,
                                rr,
                            );
                            self.detection.last_detections = self.map_detections(detections);
                        }
                        self.fire_sink_and_update_director(elapsed, scheduled_detection)?;
                        scheduled_detection
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
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                StereoFrame::MetalResident { left, right } => {
                    self.detect_and_update_director_metal(
                        left.as_ptr(),
                        right.as_ptr(),
                        left.width(),
                        left.height(),
                        elapsed,
                    )?;
                    scheduled_detection
                }
                #[cfg(target_os = "windows")]
                StereoFrame::D3d11Resident { .. } => {
                    if self.d3d11_staging_pool.is_some() {
                        let left_slot = self.frame_count as usize % 2;
                        let right_slot = left_slot + 2;
                        let (w, h) = self.core.pipeline().source_info();
                        let lr = self.left_rotation;
                        let rr = self.right_rotation;
                        let should_detect = self.detection.has_detector()
                            && self.detection.should_detect(self.frame_count);
                        if should_detect {
                            let pool = self.d3d11_staging_pool.as_ref().unwrap();
                            let detections = self.detection.run_detection_wgpu_nv12(
                                pool.y_view(left_slot),
                                pool.uv_view(left_slot),
                                pool.y_view(right_slot),
                                pool.uv_view(right_slot),
                                w,
                                h,
                                lr,
                                rr,
                            );
                            self.detection.last_detections = self.map_detections(detections);
                        }
                        self.fire_sink_and_update_director(elapsed, should_detect)?;
                        scheduled_detection
                    } else {
                        self.update_director(elapsed)?;
                        false
                    }
                }
                _ => {
                    self.detect_and_update_director(frame, elapsed)?;
                    scheduled_detection
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
        // Lazily create the texture cache on first MetalResident frame.
        if self.metal_texture_cache.is_none() {
            self.metal_texture_cache = Some(crate::interop::metal::MetalTextureCache::new(
                self.core.gpu(),
            )?);
            log::info!("Metal zero-copy: texture cache initialized");
        }
        let cache = self.metal_texture_cache.as_mut().unwrap();

        // SAFETY: RetainedCVPixelBuffer guarantees the pointer is valid.
        let (left_y, left_uv) = unsafe { cache.import_nv12(left.as_ptr(), self.core.gpu())? };
        let (right_y, right_uv) = unsafe { cache.import_nv12(right.as_ptr(), self.core.gpu())? };

        let render_buf = self.core.render_imported_textures_at_pose(
            &left_y.texture,
            &left_uv.texture,
            &right_y.texture,
            &right_uv.texture,
            yaw,
            pitch,
        );
        self.submit_render_output(render_buf)?;

        let desc = wgpu::TextureViewDescriptor::default();
        let ly = left_y.texture.create_view(&desc);
        let lu = left_uv.texture.create_view(&desc);
        let ry = right_y.texture.create_view(&desc);
        let ru = right_uv.texture.create_view(&desc);
        self.core.pack_gpu_stacked_replay_from_views(
            crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 { y: &ly, uv: &lu },
            crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 { y: &ry, uv: &ru },
        );
        Ok(())
    }

    /// Render a GpuResident frame: shared CUDA/Vulkan textures.
    ///
    /// Renders from pre-built bind groups, packs replay from shared
    /// texture views, and releases decode slots for thread reuse.
    #[cfg(target_os = "linux")]
    fn render_gpu_resident(
        &mut self,
        left_slot: u8,
        right_slot: u8,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), SessionError> {
        // VRAM pool path: render from pool bind groups.
        // Decode slots were already freed during produce.
        if let Some(vram_idx) = self.current_vram_slot {
            let pool = self
                .vram_pool
                .as_ref()
                .expect("vram_pool must exist when current_vram_slot is set");
            let left_bg = pool.left_bind_group(vram_idx);
            let right_bg = pool.right_bind_group(vram_idx);
            let render_buf = self
                .core
                .pipeline_mut()
                .render_with_bind_groups(left_bg, right_bg, yaw, pitch);
            self.submit_render_output(render_buf)?;
            return Ok(());
        }

        // Shared texture path (non-buffered / immediate mode).
        let bind_groups = self.gpu_bind_groups.as_ref().ok_or_else(|| {
            SessionError::ZeroCopy(
                "GPU bind groups not configured - call setup_gpu_source() before run()".into(),
            )
        })?;
        let render_buf =
            self.core
                .render_gpu_frame_at_pose(bind_groups, left_slot, right_slot, yaw, pitch);
        self.submit_render_output(render_buf)?;

        if let Some(ref views) = self.gpu_shared_views {
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

        if let Some((ref left_tx, ref right_tx)) = self.gpu_slot_free_tx {
            if left_tx.send(left_slot).is_err() {
                log::error!(
                    "Failed to release left GPU slot {left_slot} - decode thread may have died"
                );
            }
            if right_tx.send(right_slot).is_err() {
                log::error!(
                    "Failed to release right GPU slot {right_slot} - decode thread may have died"
                );
            }
        }

        Ok(())
    }

    /// Stage D3D11VA decoded frames into the shared staging pool.
    ///
    /// Lazily creates the pool on first call. Performs `CopySubresourceRegion`
    /// from FFmpeg's decode textures to our SHARED_NTHANDLE staging textures.
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
        let first_frame = self.d3d11_staging_pool.is_none();
        if first_frame {
            let (w, h) = self.core.pipeline().source_info();
            let needs_cuda = self.detection.needs_cuda_frames();
            // For lookahead, size slots to the max frames simultaneously
            // in flight (decoded but not yet rendered), x2 for left+right.
            // Peak occupancy is n + post_smooth_half + 1 (buffer hits n+1
            // right after a produce while the pose queue holds
            // post_smooth_half). Slots are assigned by produce_index modulo
            // n_slots with no occupancy check, so the pool must exceed peak
            // occupancy or a producer would overwrite a frame still queued
            // for render. +4 keeps a few frames of slack above the exact
            // fit (the Linux VramPool uses ref-counted acquire/release; this
            // path relies on the sizing margin instead). Without lookahead,
            // 4 slots (double-buffered stereo) suffice.
            let n_slots = if self.lookahead_frames > 0 {
                let post_smooth_half = (self.lookahead_frames / 2).max(1);
                (self.lookahead_frames + post_smooth_half + 4) * 2
            } else {
                4
            };
            match crate::interop::d3d11::D3d11StagingPool::new(
                self.core.gpu(),
                w,
                h,
                n_slots,
                needs_cuda,
                self.gpu_pixel_format,
            ) {
                Ok(pool) => {
                    log::info!(
                        "D3D11VA staging pool created: {}x{}, {n_slots} {:?} slots",
                        w,
                        h,
                        self.gpu_pixel_format
                    );
                    self.d3d11_staging_pool = Some(pool);
                }
                Err(e) => {
                    return Err(SessionError::ZeroCopy(format!("D3D11 staging pool: {e}")));
                }
            }
        }
        let left_pool_slot = self.frame_count as usize % 2;
        let right_pool_slot = left_pool_slot + 2;

        let pool = self.d3d11_staging_pool.as_mut().unwrap();
        pool.stage_frame(left_texture, left_slice, left_pool_slot)?;
        pool.stage_frame(right_texture, right_slice, right_pool_slot)?;
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
        let pool = self.d3d11_staging_pool.as_ref().unwrap();
        let render_buf = self.core.render_imported_views_at_pose(
            pool.y_view(left_slot),
            pool.uv_view(left_slot),
            pool.y_view(right_slot),
            pool.uv_view(right_slot),
            yaw,
            pitch,
        );
        self.submit_render_output(render_buf)
    }

    /// Render from already-staged D3D11VA views (immediate path).
    #[cfg(target_os = "windows")]
    fn render_d3d11_staged(&mut self, yaw: f32, pitch: f32) -> Result<(), SessionError> {
        let left_pool_slot = self.frame_count as usize % 2;
        let right_pool_slot = left_pool_slot + 2;

        let pool = self.d3d11_staging_pool.as_ref().unwrap();
        let render_buf = self.core.render_imported_views_at_pose(
            pool.y_view(left_pool_slot),
            pool.uv_view(left_pool_slot),
            pool.y_view(right_pool_slot),
            pool.uv_view(right_pool_slot),
            yaw,
            pitch,
        );
        self.submit_render_output(render_buf)?;

        let pool = self.d3d11_staging_pool.as_ref().unwrap();
        self.core.pack_gpu_stacked_replay_from_views(
            crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 {
                y: pool.y_view(left_pool_slot),
                uv: pool.uv_view(left_pool_slot),
            },
            crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 {
                y: pool.y_view(right_pool_slot),
                uv: pool.uv_view(right_pool_slot),
            },
        );

        Ok(())
    }

    /// Render from GPU-resident textures and submit to the async encoder.
    ///
    /// Used with the zero-copy path where decode threads write directly
    /// to shared GPU textures. The caller must configure bind groups via
    /// [`pipeline_mut()`](Self::pipeline_mut) and call
    /// `StitchPipeline::render_gpu_frame` to get the command buffer,
    /// then pass it here.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_submit_render")
    )]
    pub fn submit_render_output(
        &mut self,
        render_commands: wgpu::CommandBuffer,
    ) -> Result<(), SessionError> {
        let readback_t0 = std::time::Instant::now();
        let nv12_data = self.nv12_converter.convert_and_readback(
            self.core.gpu(),
            self.core.pipeline().render_target(),
            render_commands,
        )?;
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
        let nv12_data = self.nv12_converter.convert_and_readback(
            self.core.gpu(),
            self.core.pipeline().render_target(),
            render_commands,
        )?;
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
                && let Some((ref left_tx, ref right_tx)) = self.gpu_slot_free_tx
            {
                let _ = left_tx.send(*left_slot);
                let _ = right_tx.send(*right_slot);
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
        let (ls, rs) = match frame {
            StereoFrame::GpuResident {
                left_slot,
                right_slot,
            } => (*left_slot as usize, *right_slot as usize),
            _ => return Ok(None),
        };
        {
            if let (Some(pool), Some(shared_tex)) =
                (self.vram_pool.as_mut(), self.gpu_shared_textures.as_ref())
            {
                let slot = pool.acquire().ok_or_else(|| {
                    SessionError::Config(format!(
                        "VRAM pool exhausted ({} slots, {} available)",
                        pool.capacity(),
                        pool.available()
                    ))
                })?;
                let gpu = self.core.pipeline().gpu();
                pool.copy_from_textures(
                    gpu,
                    slot,
                    &shared_tex[ls * 2],
                    &shared_tex[ls * 2 + 1],
                    &shared_tex[4 + rs * 2],
                    &shared_tex[4 + rs * 2 + 1],
                );
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                // NOTE: the decode slot is NOT released here. Detection
                // reads it after this copy (see `produce_one`), so the
                // slot must stay held until `release_gpu_decode_slot` is
                // called post-detection. Releasing it now would let the
                // decode thread overwrite the slot mid-detection.
                return Ok(Some(slot));
            }
        }
        Ok(None)
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
            // Ensure the D3D11 staging pool is initialized (lazy init
            // needs the first source texture to extract the D3D11 device).
            if self.d3d11_staging_pool.is_none() {
                self.stage_d3d11_frames(*left_texture, *left_slice, *right_texture, *right_slice)?;
            }
            let pool = self.d3d11_staging_pool.as_mut().unwrap();
            let left_slot = (produce_index as usize * 2) % pool.n_slots();
            let right_slot = (produce_index as usize * 2 + 1) % pool.n_slots();
            pool.stage_frame(*left_texture, *left_slice, left_slot)?;
            pool.stage_frame(*right_texture, *right_slice, right_slot)?;
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
            // Import CVPixelBuffers as wgpu textures (separate Y + UV).
            if self.metal_texture_cache.is_none() {
                self.metal_texture_cache = Some(crate::interop::metal::MetalTextureCache::new(
                    self.core.gpu(),
                )?);
            }
            let cache = self.metal_texture_cache.as_mut().unwrap();
            let (left_y, left_uv) = unsafe { cache.import_nv12(left.as_ptr(), self.core.gpu())? };
            let (right_y, right_uv) =
                unsafe { cache.import_nv12(right.as_ptr(), self.core.gpu())? };

            if let Some(pool) = self.vram_pool.as_mut() {
                let slot = pool.acquire().ok_or_else(|| {
                    SessionError::Config(format!(
                        "VRAM pool exhausted ({} slots, {} available)",
                        pool.capacity(),
                        pool.available()
                    ))
                })?;
                let gpu = self.core.pipeline().gpu();
                pool.copy_from_textures(
                    gpu,
                    slot,
                    &left_y.texture,
                    &left_uv.texture,
                    &right_y.texture,
                    &right_uv.texture,
                );
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                return Ok(Some(slot));
            }
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
