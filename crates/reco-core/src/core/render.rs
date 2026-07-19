//! Render and submit methods for [`StitchCore`](super::StitchCore).
//!
//! Contains the `submit_frame_*` push methods (which run detection,
//! tick the director, and read back RGBA) and the low-level
//! `render_*_at_pose` methods (GPU-only, no readback).

use crate::geometry::Pose;
#[cfg(feature = "gpu")]
use crate::render::planes::BgraPlanes;
use crate::render::planes::YuvPlanes;
use crate::stitch::Executor;

use super::types::{RenderOutcome, ReplayFrame, StitchCoreError};

impl super::StitchCore {
    // -----------------------------------------------------------------
    // Submit / render
    // -----------------------------------------------------------------

    /// Submit a CPU-resident frame set and render the current pose.
    ///
    /// The director-driven push entry for every CPU-resident format:
    /// runs detection on schedule, ticks the director and coverage
    /// clamping to pick the viewport, renders, and reads back RGBA.
    /// On the GPU executor the first two calls produce
    /// [`RenderOutcome::Warmup`] while the triple-buffered staging ring
    /// fills; the CPU executor stitches synchronously.
    ///
    /// The set length is validated against the projection's camera
    /// count by the executor (`check_camera_count` on the CPU arm; the
    /// GPU render program is two-camera until the mono GPU pass lands,
    /// so other lengths get a typed error there). Detection and the
    /// stacked replay recorder consume two-camera sets only today -
    /// other sets render without them (the detector warns once and
    /// idles).
    ///
    /// GPU-resident variants (shared-texture slots, D3D11, NVMM,
    /// Metal) are session-managed zero-copy paths, not push submits;
    /// they get a typed error here.
    pub fn submit_frame(
        &mut self,
        frames: &crate::source::FrameSet,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        use crate::source::FrameSet;
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
                    let (src_w, src_h) = self.executor.source_info();
                    recorder.record_yuv(&left.as_planes(), &right.as_planes(), src_w, src_h);
                }
                let ran_detection = self.detect_yuv_set(cams);
                let pose = self.resolve_current_pose(ran_detection);
                match &self.executor {
                    Executor::Cpu(_) => {
                        let planes: Vec<YuvPlanes<'_>> =
                            cams.iter().map(|c| c.as_planes()).collect();
                        self.submit_cpu_yuv(&planes, pose)
                    }
                    #[cfg(feature = "gpu")]
                    Executor::Gpu(_) => match cams.as_slice() {
                        [left, right] => {
                            self.submit_gpu_yuv(&left.as_planes(), &right.as_planes(), pose)
                        }
                        _ => Err(Self::mono_gpu_not_wired()),
                    },
                }
            }
            FrameSet::Nv12(cams) => {
                self.anchor_session_start();
                // The stacked replay recorder is YUV420P-native and
                // does not tap NV12 submits today.
                let ran_detection = self.detect_nv12_set(cams);
                let pose = self.resolve_current_pose(ran_detection);
                match &self.executor {
                    Executor::Cpu(_) => {
                        let planes: Vec<crate::render::planes::Nv12Planes<'_>> =
                            cams.iter().map(|c| c.as_planes()).collect();
                        self.submit_cpu_nv12(&planes, pose)
                    }
                    #[cfg(feature = "gpu")]
                    Executor::Gpu(_) => match cams.as_slice() {
                        [left, right] => {
                            self.submit_gpu_nv12(&left.as_planes(), &right.as_planes(), pose)
                        }
                        _ => Err(Self::mono_gpu_not_wired()),
                    },
                }
            }
            _ => Err(StitchCoreError::Config(
                "GPU-resident frame sets are session-managed zero-copy paths; \
                 submit_frame takes CPU-resident sets (YUV420P / NV12)"
                    .into(),
            )),
        }
    }

    /// Run scheduled detection on a YUV420P set. Detection mapping is
    /// two-camera only today; other set lengths warn once and idle.
    /// Returns whether detection ran (feeds the director's
    /// fresh-detection flag in `resolve_current_pose`).
    fn detect_yuv_set(&mut self, cams: &[crate::source::YuvData]) -> bool {
        if !self.detection_due(self.frame_count) {
            return false;
        }
        match cams {
            [left, right] => {
                let (src_w, src_h) = self.executor.source_info();
                self.run_yuv_detection(&left.as_planes(), &right.as_planes(), src_w, src_h);
                true
            }
            _ => {
                self.warn_detection_unmapped();
                false
            }
        }
    }

    /// NV12 counterpart of [`Self::detect_yuv_set`].
    fn detect_nv12_set(&mut self, cams: &[crate::source::Nv12Data]) -> bool {
        if !self.detection_due(self.frame_count) {
            return false;
        }
        match cams {
            [left, right] => {
                let (src_w, src_h) = self.executor.source_info();
                self.run_nv12_detection(&left.as_planes(), &right.as_planes(), src_w, src_h);
                true
            }
            _ => {
                self.warn_detection_unmapped();
                false
            }
        }
    }

    /// One-time warning when a detector is attached but the set shape
    /// has no detection mapping (mono topologies).
    fn warn_detection_unmapped(&mut self) {
        if !self.mono_detection_warned {
            self.mono_detection_warned = true;
            log::warn!("detection is not supported on mono topologies yet; the detector idles");
        }
    }

    /// Typed error for GPU submits of set shapes the two-camera GPU
    /// program cannot render yet.
    #[cfg(feature = "gpu")]
    fn mono_gpu_not_wired() -> StitchCoreError {
        StitchCoreError::Executor(crate::stitch::StitchError::InvalidConfig(
            "the mono GPU pass is not wired yet; run mono topologies on the \
             CPU executor"
                .into(),
        ))
    }

    /// CPU arm of the YUV submits: synchronous software stitch - RGBA
    /// immediately, no staging ring, no warmup.
    fn submit_cpu_yuv(
        &mut self,
        planes: &[YuvPlanes<'_>],
        pose: Pose,
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
        pose: Pose,
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

    /// GPU arm of the YUV submits: pipelined render + triple-buffered
    /// readback.
    #[cfg(feature = "gpu")]
    fn submit_gpu_yuv(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        pose: Pose,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        let Executor::Gpu(gpu) = &self.executor else {
            unreachable!("routed from the GPU arm");
        };
        let cmd = gpu.pipeline.render_to_target(left, right, pose)?;
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
    /// Same full loop as [`Self::submit_frame`] - anchors the
    /// session-start clock, runs detection when
    /// `frame_count % detection_interval == 0`, renders, reads back
    /// RGBA, pushes into the replay buffer, increments frame_count -
    /// but bypasses the director and uses the caller-supplied pose
    /// directly - fov included, like every render path.
    ///
    /// This is the canonical submit path for interactive UIs (OBS
    /// pan/zoom sliders, mouse-drag preview) where pose comes from
    /// user input rather than a director.
    pub fn submit_frame_yuv_at_pose(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        pose: Pose,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();

        // Replay recording tap - see `submit_frame` for the
        // rationale (record-before-render so the file exactly
        // matches what the pipeline consumed).
        if let Some(ref mut recorder) = self.stacked_recorder {
            let (src_w, src_h) = self.executor.source_info();
            recorder.record_yuv(left, right, src_w, src_h);
        }

        // `submit_frame_yuv_at_pose` bypasses resolve_current_pose (caller
        // provides the pose directly), but detection still runs on the
        // schedule so directors stay populated for a later `current_pose()`
        // peek or a regular `submit_frame` submit.
        if self.detection_due(self.frame_count) {
            let (src_w, src_h) = self.executor.source_info();
            self.run_yuv_detection(left, right, src_w, src_h);
        }
        match &self.executor {
            Executor::Cpu(_) => self.submit_cpu_yuv(&[*left, *right], pose),
            #[cfg(feature = "gpu")]
            Executor::Gpu(_) => self.submit_gpu_yuv(left, right, pose),
        }
    }

    /// Submit a stereo BGRA frame pair at an explicit pose. See
    /// [`Self::submit_frame_yuv_at_pose`] for semantics.
    ///
    /// Does not run detection: YOLO backends today consume YUV or
    /// NV12 `RawFrame` variants, and wrapping BGRA bytes as a YUV
    /// frame would need a color-space conversion we are not paying
    /// for. Consumers that want detection on BGRA sources (OBS
    /// Browser Source, screen capture) attach a detector that
    /// understands BGRA once such a backend exists; until then BGRA
    /// submits tick the director with the last detections from any
    /// earlier YUV submits.
    #[cfg(feature = "gpu")]
    pub fn submit_frame_bgra_at_pose(
        &mut self,
        left: &BgraPlanes<'_>,
        right: &BgraPlanes<'_>,
        pose: Pose,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();
        let Executor::Gpu(gpu) = &self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        let cmd = gpu.pipeline.render_to_target_bgra(left, right, pose)?;
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

    /// GPU arm of the NV12 submits.
    #[cfg(feature = "gpu")]
    fn submit_gpu_nv12(
        &mut self,
        left: &crate::render::planes::Nv12Planes<'_>,
        right: &crate::render::planes::Nv12Planes<'_>,
        pose: Pose,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        let Executor::Gpu(gpu) = &self.executor else {
            unreachable!("routed from the GPU arm");
        };
        let cmd = gpu.pipeline.render_to_target_nv12(left, right, pose)?;
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

    /// Store a CPU-stitched frame and hand out the borrowed outcome -
    /// the synchronous dual of the GPU readback tail (replay push +
    /// frame accounting).
    fn deliver_cpu_frame(&mut self, rgba: Vec<u8>, pose: Pose) -> RenderOutcome<'_> {
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
    /// The full pose is the render parameter - fov included, so there
    /// is no cached zoom state to go stale between calls.
    #[cfg(feature = "gpu")]
    pub fn render_to_view(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        pose: Pose,
        view: &wgpu::TextureView,
    ) -> Result<(), StitchCoreError> {
        let Executor::Gpu(gpu) = &self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        Ok(gpu.pipeline.render_to_view(left, right, pose, view)?)
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
        pose: Pose,
    ) -> Result<Option<&[u8]>, StitchCoreError> {
        let Executor::Gpu(gpu) = &mut self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        let cmd = gpu.pipeline.render_to_target(left, right, pose)?;
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
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        Ok(self
            .executor
            .gpu()
            .expect("zero-copy render paths require the GPU executor")
            .pipeline
            .render_from_gpu_rgba(left_rgba, right_rgba, pose)
            .map_err(crate::stitch::StitchError::from)?)
    }

    /// Render any [`FrameSet`](crate::source::FrameSet) variant
    /// (YUV / NV12 / GpuResident) at an explicit pose.
    ///
    /// Thin wrapper over the pipeline's frame-set render that
    /// converts the pipeline error into a `StitchCoreError`. The
    /// `MetalResident` variant is NOT handled here; use
    /// [`Self::render_imported_textures_at_pose`] after importing the
    /// `CVPixelBuffer` via `MetalTextureCache`.
    #[cfg(feature = "gpu")]
    pub fn render_frame_set_at_pose(
        &self,
        frames: &crate::source::FrameSet,
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        let Executor::Gpu(gpu) = &self.executor else {
            return Err(StitchCoreError::RequiresGpu);
        };
        Ok(gpu.pipeline.render_frame_set(frames, pose)?)
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
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        Ok(self
            .executor
            .gpu_mut()
            .expect("zero-copy render paths require the GPU executor")
            .pipeline
            .render_imported_textures(left_y, left_uv, right_y, right_uv, pose)
            .map_err(crate::stitch::StitchError::from)?)
    }
}
