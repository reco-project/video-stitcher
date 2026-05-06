//! Render and submit methods for [`StitchCore`](super::StitchCore).
//!
//! Contains the `submit_frame_*` push methods (which run detection,
//! tick the director, and read back RGBA) and the low-level
//! `render_*_at_pose` methods (GPU-only, no readback).

use crate::detect::director::ViewportPosition;
use crate::render::pipeline::{BgraPlanes, YuvPlanes};

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
            let (src_w, src_h) = self.pipeline.source_info();
            recorder.record_yuv(left, right, src_w, src_h);
        }

        // Detection first, so the director's `update` tick in
        // resolve_current_pose sees the latest tracked objects. Skipped
        // frames reuse last_detections so the director still has context.
        let ran_detection = self.detector.is_some() && self.should_run_detection();
        if ran_detection {
            let (src_w, src_h) = self.pipeline.source_info();
            let dets = self.run_yuv_detection(left, right, src_w, src_h);
            self.last_detections = self.map_detections_to_panorama(dets);
        }

        let pose = self.resolve_current_pose(ran_detection);
        let cmd = self
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
        self.drive_gpu_stacked_pack();
        // Split-borrow: push_replay only accesses self.replay +
        // self.session_start; self.readback keeps the rgba slice
        // alive. Inlining the replay push (instead of going through
        // `&mut self` on a helper) lets the borrow checker see the
        // fields are disjoint.
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let rgba =
            self.readback
                .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)?;
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
    /// pipeline currently has (set via [`Self::pipeline_mut`] or
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
            let (src_w, src_h) = self.pipeline.source_info();
            recorder.record_yuv(left, right, src_w, src_h);
        }

        // `submit_frame_yuv_at_pose` bypasses resolve_current_pose (caller
        // provides the pose directly), but detection still runs on the
        // schedule so directors stay populated for a later `current_pose()`
        // peek or a regular `submit_frame_yuv` submit.
        if self.detector.is_some() && self.should_run_detection() {
            let (src_w, src_h) = self.pipeline.source_info();
            let dets = self.run_yuv_detection(left, right, src_w, src_h);
            self.last_detections = self.map_detections_to_panorama(dets);
        }

        let cmd = self.pipeline.render_to_target(left, right, yaw, pitch)?;
        // GPU stacked-replay pack - see `submit_frame_yuv` for
        // ordering rationale. No-op when not enabled.
        self.drive_gpu_stacked_pack();
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let rgba =
            self.readback
                .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)?;
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

    /// Submit a stereo BGRA frame pair at an explicit pose. See
    /// [`Self::submit_frame_yuv_at_pose`] for semantics.
    ///
    /// Does not run detection (BGRA backends are not yet supported;
    /// see [`Self::submit_frame_bgra`] for the rationale).
    pub fn submit_frame_bgra_at_pose(
        &mut self,
        left: &BgraPlanes<'_>,
        right: &BgraPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();
        let cmd = self
            .pipeline
            .render_to_target_bgra(left, right, yaw, pitch)?;
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let rgba =
            self.readback
                .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)?;
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
        let cmd = self
            .pipeline
            .render_to_target_bgra(left, right, pose.yaw, pose.pitch)?;
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let rgba =
            self.readback
                .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)?;
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

    /// Drain one pending readback slot without submitting a new frame.
    ///
    /// Useful at shutdown to collect the 1-2 frames still in-flight in
    /// the triple-buffered staging pipeline.
    pub fn flush(&mut self) -> Result<Option<&[u8]>, StitchCoreError> {
        Ok(self.readback.flush_pending(self.pipeline.gpu())?)
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
    // The M3 `StitchSession::run` pull-adapter (plan step 2) uses these
    // to route its encode loop through `StitchCore`: session owns its
    // own director + detection pipeline during the transition and
    // passes the resolved pose explicitly here. Once the session
    // migration completes, these remain as the "render primitives" for
    // multi-output consumers (record + stream, zero-copy compositor).
    // -----------------------------------------------------------------

    /// Render a stereo YUV420P frame at an explicit pose.
    ///
    /// Does not run detection, does not tick the director, does not
    /// read back RGBA. Consumers that want the full `submit_*` loop
    /// (detection + director + readback) should call
    /// [`Self::submit_frame_yuv`] instead.
    ///
    /// The caller is responsible for subsequently consuming the
    /// rendered texture (via [`Self::pipeline`] + `render_target()`)
    /// or submitting the returned command buffer to chain further
    /// GPU work.
    pub fn render_yuv_at_pose(
        &self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        Ok(self.pipeline.render_to_target(left, right, yaw, pitch)?)
    }

    /// Render a stereo packed-RGBA/BGRA frame at an explicit pose.
    /// See [`Self::render_yuv_at_pose`] for semantics.
    pub fn render_bgra_at_pose(
        &self,
        left: &BgraPlanes<'_>,
        right: &BgraPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        Ok(self
            .pipeline
            .render_to_target_bgra(left, right, yaw, pitch)?)
    }

    /// Render from GPU-resident RGBA textures (e.g. Bayer demosaic output).
    ///
    /// Copies the demosaiced textures into the stitch pipeline's input
    /// planes (GPU-to-GPU blit), then renders the stitch. Returns the
    /// render command buffer for `submit_render_output`.
    pub fn render_gpu_rgba_at_pose(
        &self,
        left_rgba: &wgpu::Texture,
        right_rgba: &wgpu::Texture,
        yaw: f32,
        pitch: f32,
    ) -> wgpu::CommandBuffer {
        self.pipeline
            .render_from_gpu_rgba(left_rgba, right_rgba, yaw, pitch)
    }

    /// Render any [`StereoFrame`](crate::source::StereoFrame) variant
    /// (YUV / NV12 / GpuResident) at an explicit pose.
    ///
    /// Thin wrapper over
    /// [`StitchPipeline::render_stereo_frame`](crate::render::pipeline::StitchPipeline::render_stereo_frame)
    /// that converts the pipeline error into a `StitchCoreError`. The
    /// `MetalResident` variant is NOT handled here; use
    /// [`Self::render_imported_textures_at_pose`] after importing the
    /// `CVPixelBuffer` via `MetalTextureCache`.
    pub fn render_stereo_frame_at_pose(
        &self,
        frame: &crate::source::StereoFrame,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        Ok(self.pipeline.render_stereo_frame(frame, yaw, pitch)?)
    }

    /// Render from four pre-imported textures at an explicit pose.
    ///
    /// Used by the macOS zero-copy path where `CVPixelBuffer` Y/UV
    /// planes are imported as wgpu textures via `MetalTextureCache`
    /// (in `interop::metal`), and the Linux zero-copy path that shares
    /// textures through the bind-group variant below.
    pub fn render_imported_textures_at_pose(
        &mut self,
        left_y: &wgpu::Texture,
        left_uv: &wgpu::Texture,
        right_y: &wgpu::Texture,
        right_uv: &wgpu::Texture,
        yaw: f32,
        pitch: f32,
    ) -> wgpu::CommandBuffer {
        self.pipeline
            .render_imported_textures(left_y, left_uv, right_y, right_uv, yaw, pitch)
    }

    /// Render from pre-built GPU texture views at an explicit pose.
    ///
    /// Used by the D3D11VA zero-copy path where NV12 plane views are
    /// created with `TextureAspect::Plane0` / `Plane1`.
    pub fn render_imported_views_at_pose(
        &mut self,
        left_y: &wgpu::TextureView,
        left_uv: &wgpu::TextureView,
        right_y: &wgpu::TextureView,
        right_uv: &wgpu::TextureView,
        yaw: f32,
        pitch: f32,
    ) -> wgpu::CommandBuffer {
        self.pipeline
            .render_imported_views(left_y, left_uv, right_y, right_uv, yaw, pitch)
    }

    /// Render from pre-configured GPU bind groups and decode slots at
    /// an explicit pose (Linux zero-copy path).
    ///
    /// Thin wrapper over
    /// `StitchPipeline::render_gpu_frame`.
    /// Consumers must have already called
    /// `StitchPipeline::configure_gpu_source` via [`Self::pipeline_mut`].
    #[cfg(target_os = "linux")]
    pub fn render_gpu_frame_at_pose(
        &mut self,
        bind_groups: &crate::render::pipeline::GpuSourceBindGroups,
        left_slot: u8,
        right_slot: u8,
        yaw: f32,
        pitch: f32,
    ) -> wgpu::CommandBuffer {
        self.pipeline
            .render_gpu_frame(bind_groups, left_slot, right_slot, yaw, pitch)
    }
}
