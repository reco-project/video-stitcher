//! Per-frame render and encode methods for [`StitchSession`].
//!
//! These methods are called once per frame to render a stereo pair,
//! convert to NV12, and fan out to attached encoders.

use super::StitchSession;
use crate::director::ViewportPosition;
use crate::session::types::{SessionError, StepResult};
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
            let (ry, rp) = crate::rig_correction::world_to_render_pose(
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
            sink.emit(crate::pipeline_event::PipelineEvent::PosePresented {
                frame_index: self.frame_count,
                pose: pos,
            });
        }

        if let Some(fov) = pos.fov_degrees {
            self.core.pipeline_mut().set_fov(fov);
        }
        pos
    }

    /// Process one frame with full session features: detection, director,
    /// coverage clamping, and encoding.
    ///
    /// This is the recommended API for interactive consumers (GUI apps, OBS
    /// plugins) that control their own frame loop. It combines
    /// `detect_and_update_director()`, `director_position()`, and
    /// `process_frame()` into a single call and returns what happened.
    ///
    /// Pass `override_position` to bypass the director (e.g. when the user
    /// grabs the viewport with their mouse). The director still updates
    /// internally so it stays warm.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_step")
    )]
    pub fn step(
        &mut self,
        frame: &StereoFrame,
        elapsed: std::time::Duration,
        override_position: Option<ViewportPosition>,
    ) -> Result<StepResult, SessionError> {
        // Run detection and update director.
        self.detect_and_update_director(frame, elapsed)?;

        // Get viewport position (from director or override).
        let pos = if let Some(ovr) = override_position {
            if let Some(fov) = ovr.fov_degrees {
                self.core.pipeline_mut().set_fov(fov);
            }
            ovr
        } else {
            self.director_position()
        };

        let frame_index = self.frame_count;

        // Render + encode.
        self.process_frame(frame, pos.yaw, pos.pitch)?;

        Ok(StepResult {
            viewport: pos,
            frame_index,
        })
    }

    /// Render a single CPU-resident stereo frame and submit it to the encoder.
    ///
    /// Handles YUV420P and NV12 input formats. For GPU-resident frames
    /// (zero-copy path), use [`submit_render_output`](Self::submit_render_output)
    /// instead.
    ///
    /// For interactive consumers that want detection + director + encoding in
    /// one call, use [`step()`](Self::step) instead.
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
        self.core.drive_gpu_stacked_pack();
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
        self.core.drive_gpu_stacked_pack();
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
            crate::yuv_stack_packer::StackedPackSource::Nv12 { y: &ly, uv: &lu },
            crate::yuv_stack_packer::StackedPackSource::Nv12 { y: &ry, uv: &ru },
        );
        Ok(())
    }

    /// Process a MetalResident frame: import CVPixelBuffers as textures, render.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn process_metal_frame(
        &mut self,
        left: &crate::metal_interop::RetainedCVPixelBuffer,
        right: &crate::metal_interop::RetainedCVPixelBuffer,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), SessionError> {
        // Lazily create the texture cache on first MetalResident frame.
        if self.metal_texture_cache.is_none() {
            self.metal_texture_cache = Some(crate::metal_interop::MetalTextureCache::new(
                self.core.gpu(),
            )?);
            log::info!("Metal zero-copy: texture cache initialized");
        }
        let cache = self.metal_texture_cache.as_ref().unwrap();

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
        let nv12_data = self.nv12_converter.convert_and_readback(
            self.core.gpu(),
            self.core.pipeline().render_target(),
            render_commands,
        )?;

        // First two calls return None (triple-buffer warmup).
        // From the third call onward, we get data from 2 frames ago.
        if let Some(data) = nv12_data {
            if let Some(ref encoder) = self.encoder {
                encoder.submit(data, self.frame_count as i64)?;
            }
            // Fan out to extra encoders (multi-output).
            for enc in &self.extra_encoders {
                enc.submit(data, self.frame_count as i64)?;
            }
        }

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
}
