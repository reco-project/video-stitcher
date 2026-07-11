//! Render and submit methods for [`StitchCore`](super::StitchCore).
//!
//! Contains the `submit_frame_*` push methods (which run detection,
//! tick the director, and read back RGBA) and the low-level
//! `render_*_at_pose` methods (GPU-only, no readback).

use crate::geometry::ViewportPosition;
#[cfg(feature = "gpu")]
use crate::render::planes::BgraPlanes;
use crate::render::planes::YuvPlanes;
use crate::stitch::Executor;

use super::types::{RenderOutcome, ReplayFrame, StitchCoreError};

impl super::StitchCore {
    // -----------------------------------------------------------------
    // Submit / render
    // -----------------------------------------------------------------

    /// Submit a stereo YUV420P frame pair and render the current pose.
    ///
    /// Uses the director (if attached) and coverage clamping to pick
    /// the viewport, renders, and reads back RGBA. The first two calls
    /// produce [`RenderOutcome::Warmup`] while the triple-buffered
    /// staging ring fills; from the third call onward every submit
    /// yields RGBA bytes from two frames ago.
    pub fn submit_frame_yuv(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();

        // Feed the stacked-video replay recorder before render so
        // the recording captures the exact planes the pipeline will
        // see. Errors inside the recorder are logged by the impl;
        // never propagate them - a failing recorder must not break
        // the live stitch output.
        if let Some(ref mut recorder) = self.stacked_recorder {
            let (src_w, src_h) = self.executor.source_info();
            recorder.record_yuv(left, right, src_w, src_h);
        }

        // Detection first, so the director's `update` tick in
        // resolve_current_pose sees the latest tracked objects. Skipped
        // frames reuse last_detections so the director still has context.
        let ran_detection = self.detection_due(self.frame_count);
        if ran_detection {
            let (src_w, src_h) = self.executor.source_info();
            self.run_yuv_detection(left, right, src_w, src_h);
        }

        let pose = self.resolve_current_pose(ran_detection);
        match &self.executor {
            Executor::Cpu(_) => self.submit_cpu_yuv(left, right, pose),
            #[cfg(feature = "gpu")]
            Executor::Gpu(_) => self.submit_gpu_yuv(left, right, pose),
        }
    }

    /// CPU arm of the YUV submits: synchronous software stitch - RGBA
    /// immediately, no staging ring, no warmup.
    fn submit_cpu_yuv(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        pose: ViewportPosition,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        // Infallible on wgpu-free builds (single-variant enum); the gpu
        // build adds the second arm.
        #[allow(clippy::infallible_destructuring_match)]
        let cpu = match &self.executor {
            Executor::Cpu(cpu) => cpu,
            #[cfg(feature = "gpu")]
            Executor::Gpu(_) => unreachable!("routed from the CPU arm"),
        };
        let rgba = cpu.stitch_yuv(&[*left, *right], pose.yaw, pose.pitch)?;
        Ok(self.deliver_cpu_frame(rgba, pose))
    }

    /// GPU arm of the YUV submits: pipelined render + triple-buffered
    /// readback.
    #[cfg(feature = "gpu")]
    fn submit_gpu_yuv(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        pose: ViewportPosition,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        let Executor::Gpu(gpu) = &self.executor else {
            unreachable!("routed from the GPU arm");
        };
        let cmd = gpu
            .pipeline
            .render_to_target(left, right, pose.yaw, pose.pitch)?;
        // GPU stacked-replay pack runs before the readback so the
        // borrow checker sees `self.readback` free while the pack
        // runs. Queue ordering: `queue.write_texture` inside
        // `render_to_target` is already enqueued; the pack submit
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

    /// Submit a stereo YUV420P frame pair at an explicit pose.
    ///
    /// Same full loop as [`Self::submit_frame_yuv`] - anchors the
    /// session-start clock, runs detection when
    /// `frame_count % detection_interval == 0`, renders, reads back
    /// RGBA, pushes into the replay buffer, increments frame_count -
    /// but bypasses the director and uses the caller-supplied
    /// `(yaw, pitch)` directly. The FOV stays at whatever the
    /// pipeline currently has (set via [`Self::set_fov`] or
    /// `update_calibration`).
    ///
    /// This is the canonical submit path for interactive UIs (OBS
    /// pan/zoom sliders, mouse-drag preview) where pose comes from
    /// user input rather than a director.
    pub fn submit_frame_yuv_at_pose(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();

        // Replay recording tap - see `submit_frame_yuv` for the
        // rationale (record-before-render so the file exactly
        // matches what the pipeline consumed).
        if let Some(ref mut recorder) = self.stacked_recorder {
            let (src_w, src_h) = self.executor.source_info();
            recorder.record_yuv(left, right, src_w, src_h);
        }

        // `submit_frame_yuv_at_pose` bypasses resolve_current_pose (caller
        // provides the pose directly), but detection still runs on the
        // schedule so directors stay populated for a later `current_pose()`
        // peek or a regular `submit_frame_yuv` submit.
        if self.detection_due(self.frame_count) {
            let (src_w, src_h) = self.executor.source_info();
            self.run_yuv_detection(left, right, src_w, src_h);
        }

        let pose = ViewportPosition {
            yaw,
            pitch,
            fov_degrees: None,
        };
        match &self.executor {
            Executor::Cpu(_) => self.submit_cpu_yuv(left, right, pose),
            #[cfg(feature = "gpu")]
            Executor::Gpu(_) => self.submit_gpu_yuv(left, right, pose),
        }
    }

    /// Submit a stereo BGRA frame pair at an explicit pose. See
    /// [`Self::submit_frame_yuv_at_pose`] for semantics.
    ///
    /// Does not run detection (BGRA backends are not yet supported;
    /// see [`Self::submit_frame_bgra`] for the rationale).
    #[cfg(feature = "gpu")]
    pub fn submit_frame_bgra_at_pose(
        &mut self,
        left: &BgraPlanes<'_>,
        right: &BgraPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();
        let Executor::Gpu(gpu) = &self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        let cmd = gpu
            .pipeline
            .render_to_target_bgra(left, right, yaw, pitch)?;
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let Executor::Gpu(gpu) = &self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
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
                pose: ViewportPosition {
                    yaw,
                    pitch,
                    fov_degrees: None,
                },
            });
        }
        Ok(match rgba {
            Some(bytes) => RenderOutcome::Rgba(bytes),
            None => RenderOutcome::Warmup,
        })
    }

    /// Submit a stereo packed-RGBA/BGRA frame pair and render the
    /// current pose.
    ///
    /// Requires the core to have been built with `InputFormat::Bgra`.
    /// See [`Self::submit_frame_yuv`] for return semantics.
    #[cfg(feature = "gpu")]
    pub fn submit_frame_bgra(
        &mut self,
        left: &BgraPlanes<'_>,
        right: &BgraPlanes<'_>,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();

        // BGRA detection path: YOLO backends today consume YUV or
        // NV12 `RawFrame` variants. Wrapping BGRA bytes as a YUV
        // frame would require a color-space conversion we're not
        // paying for yet - consumers that want detection on BGRA
        // sources (OBS Browser Source, screen capture) attach a
        // detector that understands BGRA once such a backend exists.
        // For now, BGRA submits tick the director with the last
        // detections (potentially from earlier YUV submits) but do
        // not run detection themselves.

        // `fresh_detection = false`: BGRA submits never run detection by
        // design (see comment above). Directors must see this frame as
        // "reusing cached detections" even on interval ticks, otherwise
        // hysteresis counters over-fire.
        let pose = self.resolve_current_pose(false);
        let Executor::Gpu(gpu) = &self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        let cmd = gpu
            .pipeline
            .render_to_target_bgra(left, right, pose.yaw, pose.pitch)?;
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let Executor::Gpu(gpu) = &self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
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

    /// Submit a stereo NV12 frame pair and render the current pose.
    ///
    /// Works on both executors. The CPU arm stitches synchronously
    /// (RGBA available immediately, no warmup); the GPU arm requires a
    /// pipeline built with `InputFormat::Nv12` and follows the
    /// triple-buffered readback semantics of
    /// [`Self::submit_frame_yuv`]. NV12 is the native camera / NVDEC /
    /// X5 format, so this is the day-1 submit path for live sources.
    ///
    /// The stacked replay recorder is YUV420P-native and does not tap
    /// NV12 submits today.
    pub fn submit_frame_nv12(
        &mut self,
        left: &crate::render::planes::Nv12Planes<'_>,
        right: &crate::render::planes::Nv12Planes<'_>,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();

        let ran_detection = self.detection_due(self.frame_count);
        if ran_detection {
            let (src_w, src_h) = self.executor.source_info();
            self.run_nv12_detection(left, right, src_w, src_h);
        }

        let pose = self.resolve_current_pose(ran_detection);
        match &self.executor {
            Executor::Cpu(_) => {
                // See submit_cpu_yuv for the wgpu-free lint note.
                #[allow(clippy::infallible_destructuring_match)]
                let cpu = match &self.executor {
                    Executor::Cpu(cpu) => cpu,
                    #[cfg(feature = "gpu")]
                    Executor::Gpu(_) => unreachable!("routed from the CPU arm"),
                };
                let rgba = cpu.stitch_nv12(&[*left, *right], pose.yaw, pose.pitch)?;
                Ok(self.deliver_cpu_frame(rgba, pose))
            }
            #[cfg(feature = "gpu")]
            Executor::Gpu(_) => self.submit_gpu_nv12(left, right, pose),
        }
    }

    /// GPU arm of [`Self::submit_frame_nv12`].
    #[cfg(feature = "gpu")]
    fn submit_gpu_nv12(
        &mut self,
        left: &crate::render::planes::Nv12Planes<'_>,
        right: &crate::render::planes::Nv12Planes<'_>,
        pose: ViewportPosition,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        let Executor::Gpu(gpu) = &self.executor else {
            unreachable!("routed from the GPU arm");
        };
        let cmd = gpu
            .pipeline
            .render_to_target_nv12(left, right, pose.yaw, pose.pitch)?;
        self.pack_replay_from_pipeline();
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let Executor::Gpu(gpu) = &self.executor else {
            unreachable!("routed from the GPU arm");
        };
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

    /// Submit a mono frame - a single pre-stitched panorama - and
    /// render the current pose. The single-camera counterpart of
    /// [`Self::submit_frame_yuv`] for mono topologies (the cylinder).
    ///
    /// Detection is not wired for mono topologies yet (the
    /// detection-to-panorama mapping is L-shape-only); an attached
    /// detector logs once and idles, and the panner runs without
    /// detections. The stacked replay recorder (2-tile format) is
    /// likewise skipped.
    // TODO: wire mono detection + replay (Step 13 PR B).
    pub fn submit_frame_mono_yuv(
        &mut self,
        frame: &YuvPlanes<'_>,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();

        if self.detection_due(self.frame_count) && !self.mono_detection_warned {
            self.mono_detection_warned = true;
            log::warn!("detection is not supported on mono topologies yet; the detector idles");
        }

        let pose = self.resolve_current_pose(false);
        match &self.executor {
            Executor::Cpu(_) => {
                // See submit_cpu_yuv for the wgpu-free lint note.
                #[allow(clippy::infallible_destructuring_match)]
                let cpu = match &self.executor {
                    Executor::Cpu(cpu) => cpu,
                    #[cfg(feature = "gpu")]
                    Executor::Gpu(_) => unreachable!("routed from the CPU arm"),
                };
                let rgba = cpu.stitch_yuv(&[*frame], pose.yaw, pose.pitch)?;
                Ok(self.deliver_cpu_frame(rgba, pose))
            }
            #[cfg(feature = "gpu")]
            Executor::Gpu(_) => Err(StitchCoreError::Executor(
                crate::stitch::StitchError::InvalidConfig(
                    "the mono GPU pass is not wired yet; run mono topologies on the \
                     CPU executor"
                        .into(),
                ),
            )),
        }
    }

    /// Store a CPU-stitched frame and hand out the borrowed outcome -
    /// the synchronous dual of the GPU readback tail (replay push +
    /// frame accounting).
    fn deliver_cpu_frame(&mut self, rgba: Vec<u8>, pose: ViewportPosition) -> RenderOutcome<'_> {
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
        Ok(self
            .readback
            .as_mut()
            .expect("gpu engine owns the readback ring")
            .flush_pending(gpu.pipeline.gpu())?)
    }

    // -----------------------------------------------------------------
    // Preview mode (engine renders straight to a caller-supplied view)
    // -----------------------------------------------------------------

    /// Render a stereo YUV420P frame directly to a surface view - the
    /// interactive preview path (GUI/CLI). No detection, no director, no
    /// readback; the caller supplies the (already oriented) pose.
    ///
    /// The full pose is the render parameter: when `pose.fov_degrees` is
    /// set it applies for this frame, so an out-of-tick FOV clamp can
    /// never leave the view rendering a stale cached value.
    #[cfg(feature = "gpu")]
    pub fn render_to_view(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        pose: ViewportPosition,
        view: &wgpu::TextureView,
    ) -> Result<(), StitchCoreError> {
        if let Some(fov) = pose.fov_degrees {
            self.executor.set_fov(fov);
        }
        let Executor::Gpu(gpu) = &self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        Ok(gpu
            .pipeline
            .render_to_view(left, right, pose.yaw, pose.pitch, view)?)
    }

    /// Render a stereo frame and read back NV12 bytes for encoding - the
    /// preview-mode recording tap. Triple-buffered: `None` on the first
    /// two calls, then data from two frames ago. Drain the tail with
    /// [`Self::flush_nv12`] after the loop. The converter is created
    /// lazily on first use (dimensions rounded to NV12-safe values).
    #[cfg(feature = "gpu")]
    pub fn render_and_readback_nv12(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        pose: ViewportPosition,
    ) -> Result<Option<&[u8]>, StitchCoreError> {
        if let Some(fov) = pose.fov_degrees {
            self.executor.set_fov(fov);
        }
        let (yaw, pitch) = (pose.yaw, pose.pitch);
        let Executor::Gpu(gpu) = &mut self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        let cmd = gpu.pipeline.render_to_target(left, right, yaw, pitch)?;
        gpu.convert_nv12(cmd)
            .map_err(|e| StitchCoreError::Config(format!("NV12 preview readback: {e}")))
    }

    /// Drain one pending NV12 frame from the preview recording tap.
    /// Returns `None` when nothing remains (or the tap was never used).
    #[cfg(feature = "gpu")]
    pub fn flush_nv12(&mut self) -> Result<Option<&[u8]>, StitchCoreError> {
        let Executor::Gpu(gpu) = &mut self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        gpu.flush_nv12()
            .map_err(|e| StitchCoreError::Config(format!("NV12 preview flush: {e}")))
    }

    // -----------------------------------------------------------------
    // Low-level render-at-pose methods
    //
    // These produce a `wgpu::CommandBuffer` at a **caller-supplied pose**
    // without running detection, without ticking the director, and
    // without performing RGBA readback. They exist so consumers that
    // need the rendered GPU texture as input to further GPU work
    // (NV12 conversion for encoding, compositor texture import) can
    // drive the core without paying for readback.
    //
    // The `StitchSession::run` pull-adapter routes its encode loop
    // through these, passing the pose it resolved through the engine's
    // dispatch. They are also the render primitives for multi-output
    // consumers (record + stream, zero-copy compositor).
    // -----------------------------------------------------------------

    /// Render a stereo YUV420P frame at an explicit pose.
    ///
    /// Does not run detection, does not tick the director, does not
    /// read back RGBA. Consumers that want the full `submit_*` loop
    /// (detection + director + readback) should call
    /// [`Self::submit_frame_yuv`] instead.
    ///
    /// The caller is responsible for consuming the render by
    /// submitting the returned command buffer (directly or chained
    /// into further GPU work); the engine's readback and NV12
    /// delivery paths do this internally.
    #[cfg(feature = "gpu")]
    pub fn render_yuv_at_pose(
        &self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        let Executor::Gpu(gpu) = &self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        Ok(gpu.pipeline.render_to_target(left, right, yaw, pitch)?)
    }

    /// Render a stereo packed-RGBA/BGRA frame at an explicit pose.
    /// See [`Self::render_yuv_at_pose`] for semantics.
    #[cfg(feature = "gpu")]
    pub fn render_bgra_at_pose(
        &self,
        left: &BgraPlanes<'_>,
        right: &BgraPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        let Executor::Gpu(gpu) = &self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        Ok(gpu
            .pipeline
            .render_to_target_bgra(left, right, yaw, pitch)?)
    }

    /// Render from GPU-resident RGBA textures (e.g. Bayer demosaic output).
    ///
    /// Copies the demosaiced textures into the stitch pipeline's input
    /// planes (GPU-to-GPU blit), then renders the stitch. Returns the
    /// render command buffer for `submit_render_output`.
    #[cfg(feature = "gpu")]
    pub fn render_gpu_rgba_at_pose(
        &self,
        left_rgba: &wgpu::Texture,
        right_rgba: &wgpu::Texture,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        Ok(self
            .executor
            .gpu()
            .expect("zero-copy render paths require the GPU executor")
            .pipeline
            .render_from_gpu_rgba(left_rgba, right_rgba, yaw, pitch)
            .map_err(crate::stitch::StitchError::from)?)
    }

    /// Render any [`StereoFrame`](crate::source::StereoFrame) variant
    /// (YUV / NV12 / GpuResident) at an explicit pose.
    ///
    /// Thin wrapper over the pipeline's stereo-frame render that
    /// converts the pipeline error into a `StitchCoreError`. The
    /// `MetalResident` variant is NOT handled here; use
    /// [`Self::render_imported_textures_at_pose`] after importing the
    /// `CVPixelBuffer` via `MetalTextureCache`.
    #[cfg(feature = "gpu")]
    pub fn render_stereo_frame_at_pose(
        &self,
        frame: &crate::source::StereoFrame,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        let Executor::Gpu(gpu) = &self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        Ok(gpu.pipeline.render_stereo_frame(frame, yaw, pitch)?)
    }

    /// Render from four pre-imported textures at an explicit pose.
    ///
    /// Used by the macOS zero-copy path where `CVPixelBuffer` Y/UV
    /// planes are imported as wgpu textures via `MetalTextureCache`
    /// (in `interop::metal`), and the Linux zero-copy path that shares
    /// textures through the bind-group variant below.
    #[cfg(feature = "gpu")]
    pub fn render_imported_textures_at_pose(
        &mut self,
        left_y: &wgpu::Texture,
        left_uv: &wgpu::Texture,
        right_y: &wgpu::Texture,
        right_uv: &wgpu::Texture,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        Ok(self
            .executor
            .gpu_mut()
            .expect("zero-copy render paths require the GPU executor")
            .pipeline
            .render_imported_textures(left_y, left_uv, right_y, right_uv, yaw, pitch)
            .map_err(crate::stitch::StitchError::from)?)
    }
}
