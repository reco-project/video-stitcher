//! Render and submit methods for [`StitchCore`](super::StitchCore).
//!
//! Contains the `submit_frame_*` push methods (which run detection,
//! tick the panner, and read back RGBA) and the low-level
//! `render_*_at_pose` methods (GPU-only, no readback).

use crate::geometry::RenderPose;
use crate::render::planes::Yuv420pPlanes;
use crate::stitch::Executor;

use super::types::{RenderOutcome, ReplayFrame, StitchCoreError};

impl super::StitchCore {
    // -----------------------------------------------------------------
    // Submit / render
    // -----------------------------------------------------------------

    /// Submit a CPU-resident frame set and render the current pose.
    ///
    /// The panner-driven push entry for every CPU-resident format:
    /// runs detection on schedule, ticks the panner and coverage
    /// clamping to pick the viewport, renders, and reads back RGBA.
    /// On the GPU executor the first two calls produce
    /// [`RenderOutcome::Warmup`] while the triple-buffered staging ring
    /// fills; the CPU executor stitches synchronously.
    ///
    /// The set length is validated against the projection's camera
    /// count by the executor. Detection consumes every
    /// camera; the stacked replay recorder consumes two-camera sets only.
    ///
    /// GPU-resident variants (shared-texture slots, D3D11, NVMM,
    /// Metal) are session-managed zero-copy paths, not push submits;
    /// they get a typed error here.
    pub fn submit_frame(
        &mut self,
        frames: &crate::source::FrameSet,
        frame_seconds: f32,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        if !matches!(
            frames,
            crate::source::FrameSet::Yuv420p(_) | crate::source::FrameSet::Nv12(_)
        ) {
            return Err(StitchCoreError::Config(
                "submit_frame requires CPU-resident YUV420P or NV12 frames".into(),
            ));
        }
        frames
            .validate_host_images(self.source_sizes())
            .map_err(StitchCoreError::Config)?;
        self.anchor_session_start();
        if self.detection_due(self.frame_count) {
            self.run_cpu_detection(frames, &[])?;
        }
        let requested = self.request_current_pose();
        let pose = self.advance_pose(requested, self.frame_count, frame_seconds)?;
        self.submit_frame_at_pose(frames, pose)
    }

    /// Render CPU-resident inputs without advancing detection or camera control.
    pub(crate) fn submit_frame_at_pose(
        &mut self,
        frames: &crate::source::FrameSet,
        pose: RenderPose,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        use crate::source::FrameSet;
        frames
            .validate_host_images(self.source_sizes())
            .map_err(StitchCoreError::Config)?;
        match frames {
            FrameSet::Yuv420p(cams) => {
                self.anchor_session_start();
                // Feed the stacked-video replay recorder before render
                // so the recording captures the exact planes the
                // pipeline will see. Errors inside the recorder are
                // logged by the impl; never propagate them - a failing
                // recorder must not break the live stitch output. The
                // stacked format is 2-tile, so only pair sets record.
                if let [left, right] = cams.as_slice()
                    && let Some(ref mut recorder) = self.stacked_recorder
                {
                    let (src_w, src_h) = left.size;
                    recorder.record_yuv(&left.as_planes(), &right.as_planes(), src_w, src_h);
                }
                match &self.executor {
                    Executor::Cpu(_) => {
                        let planes: Vec<Yuv420pPlanes<'_>> =
                            cams.iter().map(|c| c.as_planes()).collect();
                        self.submit_cpu_yuv(&planes, pose)
                    }
                    #[cfg(feature = "gpu")]
                    Executor::Gpu(_) => self.submit_gpu_frames(frames, pose),
                }
            }
            FrameSet::Nv12(cams) => {
                self.anchor_session_start();
                // The stacked replay recorder is YUV420P-native and
                // does not tap NV12 submits today.
                match &self.executor {
                    Executor::Cpu(_) => {
                        let planes: Vec<crate::render::planes::Nv12Planes<'_>> =
                            cams.iter().map(|c| c.as_planes()).collect();
                        self.submit_cpu_nv12(&planes, pose)
                    }
                    #[cfg(feature = "gpu")]
                    Executor::Gpu(_) => self.submit_gpu_frames(frames, pose),
                }
            }
            _ => Err(StitchCoreError::Config(
                "GPU-resident frame sets are session-managed zero-copy paths; \
                 submit_frame takes CPU-resident sets (YUV420P / NV12)"
                    .into(),
            )),
        }
    }

    #[cfg(feature = "gpu")]
    fn submit_gpu_frames(
        &mut self,
        frames: &crate::source::FrameSet,
        pose: RenderPose,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        let cmd = self
            .executor
            .gpu_mut()
            .expect("GPU submit")
            .pipeline
            .render_frame_set(frames, pose)?;
        self.deliver_gpu_render(cmd, pose)
    }

    /// CPU arm of the YUV submits: synchronous software stitch - RGBA
    /// immediately, no staging ring, no warmup.
    fn submit_cpu_yuv(
        &mut self,
        planes: &[Yuv420pPlanes<'_>],
        pose: RenderPose,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        // Infallible on wgpu-free builds (single-variant enum); the gpu
        // build adds the second arm.
        #[allow(clippy::infallible_destructuring_match)]
        let cpu = match &self.executor {
            Executor::Cpu(cpu) => cpu,
            #[cfg(feature = "gpu")]
            Executor::Gpu(_) => unreachable!("routed from the CPU arm"),
        };
        let rgba = cpu.stitch_yuv(planes, pose)?;
        Ok(self.deliver_cpu_frame(rgba, pose))
    }

    /// NV12 counterpart of [`Self::submit_cpu_yuv`].
    fn submit_cpu_nv12(
        &mut self,
        planes: &[crate::render::planes::Nv12Planes<'_>],
        pose: RenderPose,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        // See submit_cpu_yuv for the wgpu-free lint note.
        #[allow(clippy::infallible_destructuring_match)]
        let cpu = match &self.executor {
            Executor::Cpu(cpu) => cpu,
            #[cfg(feature = "gpu")]
            Executor::Gpu(_) => unreachable!("routed from the CPU arm"),
        };
        let rgba = cpu.stitch_nv12(planes, pose)?;
        Ok(self.deliver_cpu_frame(rgba, pose))
    }

    #[cfg(feature = "gpu")]
    fn deliver_gpu_render(
        &mut self,
        cmd: wgpu::CommandBuffer,
        pose: RenderPose,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        // GPU stacked-replay pack runs before the readback so the
        // borrow checker sees `self.readback` free while the pack
        // runs. Queue ordering: `queue.write_texture` inside
        // `render_frame_set` is already enqueued; the pack submit
        // processes the writes before its compute pass reads the
        // textures, and the subsequent stitch submit reads the
        // same textures into the render target. No-op when packer
        // is not enabled.
        self.pack_replay_from_pipeline();
        // Split-borrow: push_replay only accesses self.replay +
        // self.session_start; self.readback keeps the rgba slice
        // alive. Inlining the replay push (instead of going through
        // `&mut self` on a helper) lets the borrow checker see the
        // fields are disjoint.
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let Executor::Gpu(gpu) = &self.executor else {
            unreachable!("routed from the GPU arm");
        };
        if self.readback.is_none() {
            self.readback = Some(crate::gpu::rgba_readback::RgbaReadback::new(
                gpu.pipeline.gpu(),
                self.output_width,
                self.output_height,
            )?);
        }
        let rgba = self
            .readback
            .as_mut()
            .expect("gpu engine owns the readback ring")
            .readback(gpu.pipeline.gpu(), gpu.pipeline.render_target(), cmd)?;
        self.frame_count += 1;
        if let (Some(replay), Some(bytes)) = (self.replay.as_mut(), rgba) {
            replay.push(ReplayFrame {
                rgba: bytes.to_vec(),
                captured_at,
                pose,
            });
        }
        Ok(match rgba {
            Some(bytes) => RenderOutcome::Rgba(bytes),
            None => RenderOutcome::Warmup,
        })
    }

    /// Store a CPU-stitched frame and hand out the borrowed outcome -
    /// the synchronous dual of the GPU readback tail (replay push +
    /// frame accounting).
    fn deliver_cpu_frame(&mut self, rgba: Vec<u8>, pose: RenderPose) -> RenderOutcome<'_> {
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        self.frame_count += 1;
        if let Some(replay) = self.replay.as_mut() {
            replay.push(ReplayFrame {
                rgba: rgba.clone(),
                captured_at,
                pose,
            });
        }
        self.cpu_frame = rgba;
        RenderOutcome::Rgba(&self.cpu_frame)
    }

    /// Drain one pending readback slot without submitting a new frame.
    ///
    /// Useful at shutdown to collect the 1-2 frames still in-flight in
    /// the triple-buffered staging pipeline.
    #[cfg(feature = "gpu")]
    pub fn flush(&mut self) -> Result<Option<&[u8]>, StitchCoreError> {
        let Executor::Gpu(gpu) = &self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        match &mut self.readback {
            Some(readback) => Ok(readback.flush_pending(gpu.pipeline.gpu())?),
            None => Ok(None),
        }
    }

    // -----------------------------------------------------------------
    // Preview mode (engine renders straight to a caller-supplied view)
    // -----------------------------------------------------------------

    /// Prepare a complete candidate for panorama, lens and recording draws.
    ///
    /// Owns the decoded inputs through upload or native staging. Provide the
    /// source's matrix and range and the clockwise rotations of native camera
    /// images in projection order. Host images already have their decoded
    /// orientation, and CUDA leases carry their own color and rotation.
    /// Source owners may be dropped after this call. D3D11 preparation waits
    /// for its staging query before returning a drawable frame.
    ///
    /// Keep the current frame until this returns successfully and release it
    /// when accepting its replacement. At most two frames from this engine
    /// may be held at once. See [`super::PreviewFrame`] for residency and
    /// completion guarantees.
    #[cfg(feature = "gpu")]
    pub fn prepare_preview_frame(
        &mut self,
        frames: crate::source::FrameSet,
        color: crate::render::color::YuvColor,
        rotations: [i32; 2],
    ) -> Result<super::PreviewFrame, StitchCoreError> {
        let gpu = self
            .executor
            .gpu_mut()
            .ok_or(StitchCoreError::RequiresGpu)?;
        Ok(gpu
            .pipeline
            .prepare_preview_frame(frames, color, rotations)?)
    }

    /// Submit a prepared frame at the supplied pose, without detection or readback.
    ///
    /// The caller owns the target and controls the number of draws in flight.
    /// The view must be full, single-sampled and match the configured viewport,
    /// format and device. Keep any external target owner alive until the returned
    /// render submission completes.
    #[cfg(feature = "gpu")]
    pub fn render_to_view(
        &self,
        frame: &super::PreviewFrame,
        pose: RenderPose,
        view: &wgpu::TextureView,
    ) -> Result<wgpu::SubmissionIndex, StitchCoreError> {
        let gpu = self.executor.gpu().ok_or(StitchCoreError::RequiresGpu)?;
        let cmd = gpu.pipeline.draw_preview(frame, pose, Some(view))?;
        Ok(gpu.pipeline.gpu().queue().submit(Some(cmd)))
    }

    /// Render and queue NV12 readback from the same prepared preview planes.
    ///
    /// Encodes limited-range NV12 using the first camera's YUV matrix.
    /// Requires an NV12 delivery target. Returns `None` during readback
    /// warmup, then the oldest completed frame. Drain the tail with
    /// [`Self::flush_nv12`].
    #[cfg(feature = "gpu")]
    pub fn render_and_readback_nv12(
        &mut self,
        frame: &super::PreviewFrame,
        pose: RenderPose,
    ) -> Result<Option<&[u8]>, StitchCoreError> {
        if self.delivery_format() != crate::sink::DeliveryFormat::Nv12 {
            return Err(StitchCoreError::Config(
                "preview recording requires NV12 delivery".into(),
            ));
        }
        let gpu = self
            .executor
            .gpu_mut()
            .ok_or(StitchCoreError::RequiresGpu)?;
        let cmd = gpu.pipeline.draw_preview(frame, pose, None)?;
        gpu.set_delivery_matrix(frame.camera(0).1.color.matrix);
        Ok(gpu.convert_delivery(cmd, None)?.map(|frame| frame.data))
    }

    /// Drain one pending NV12 frame from the preview recording tap.
    #[cfg(feature = "gpu")]
    pub fn flush_nv12(&mut self) -> Result<Option<&[u8]>, StitchCoreError> {
        if self.delivery_format() != crate::sink::DeliveryFormat::Nv12 {
            return Err(StitchCoreError::Config(
                "preview recording requires NV12 delivery".into(),
            ));
        }
        let gpu = self
            .executor
            .gpu_mut()
            .ok_or(StitchCoreError::RequiresGpu)?;
        Ok(gpu.flush_delivery()?.map(|frame| frame.data))
    }

    // Session encoding combines a single frame's rendering and conversion in one submission.

    /// Render from GPU-resident RGBA textures (e.g. Bayer demosaic output).
    ///
    /// Copies the demosaiced textures into the stitch pipeline's input
    /// planes (GPU-to-GPU blit), then renders the stitch. Returns the
    /// render command buffer for `submit_render_output`.
    #[cfg(feature = "gpu")]
    pub(crate) fn render_gpu_rgba_at_pose(
        &mut self,
        left_rgba: &wgpu::Texture,
        right_rgba: &wgpu::Texture,
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        let Executor::Gpu(gpu) = &mut self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        Ok(gpu
            .pipeline
            .render_from_gpu_rgba(left_rgba, right_rgba, pose)?)
    }

    /// Encode host YUV420P/NV12 input at an explicit pose. Submit the returned commands before another render changes the shared uniforms or upload textures.
    #[cfg(feature = "gpu")]
    pub(crate) fn render_frame_set_at_pose(
        &mut self,
        frames: &crate::source::FrameSet,
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        let Executor::Gpu(gpu) = &mut self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        Ok(gpu.pipeline.render_frame_set(frames, pose)?)
    }

    /// Encode imported NV12/P010 camera planes. Submit before another render changes the shared uniforms. Encoded commands retain wgpu bindings, not native producer-reuse protection.
    #[cfg(feature = "gpu")]
    pub(crate) fn render_imported_nv12_at_pose(
        &mut self,
        cameras: &[(&wgpu::TextureView, &wgpu::TextureView)],
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        let Executor::Gpu(gpu) = &mut self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        Ok(gpu.pipeline.render_imported_views(cameras, pose)?)
    }
}
