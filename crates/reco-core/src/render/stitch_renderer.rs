//! Surface-oriented renderer for interactive panoramic preview.
//!
//! [`StitchRenderer`] wraps [`StitchPipeline`]
//! with precomputed coverage boundary and surface format handling. Created once
//! per window/surface, then fed frames and viewport positions each tick.
//!
//! For batch encoding (no surface), use [`StitchSession`](crate::session::StitchSession) instead.
//!
//! # Example
//!
//! ```rust,ignore
//! use reco_core::render::stitch_renderer::StitchRenderer;
//!
//! let renderer = StitchRenderer::new(
//!     calibration, gpu, viewport, input_width, input_height,
//!     surface_format, input_format,
//! )?;
//!
//! // In the render loop:
//! renderer.render_yuv(&left_planes, &right_planes, yaw, pitch, &view)?;
//! ```

use super::pipeline::{Nv12Planes, PipelineError, StitchPipeline, YuvPlanes};
use super::renderer::InputFormat;
use super::scene::SceneGeometry;
use super::viewport::ViewportConfig;
use crate::calibration::Calibration;
use crate::gpu::GpuContext;
use crate::gpu::nv12_converter::Nv12Converter;
use crate::gpu::rgba_readback::RgbaReadback;
use crate::projection::CoverageBoundary;

/// Where to render the stitched panorama.
///
/// Pass to [`StitchRenderer::render`] to choose between surface display
/// and readback without calling different methods.
pub enum RenderTarget<'a> {
    /// Present directly to a window surface.
    Surface(&'a wgpu::TextureView),
    /// Render to the internal RGBA texture for readback.
    Texture,
}

/// What happened after a [`StitchRenderer::render`] call.
pub enum SurfaceRenderOutcome {
    /// Surface render submitted to the GPU queue (nothing to do).
    Submitted,
    /// Texture render returned a command buffer for readback.
    Commands(wgpu::CommandBuffer),
}

/// Surface-oriented stitch renderer.
///
/// Combines a [`StitchPipeline`] with a precomputed [`CoverageBoundary`]
/// for interactive preview rendering. The renderer does not own the window
/// or surface - callers provide a [`wgpu::TextureView`] each frame.
pub struct StitchRenderer {
    /// The underlying GPU stitch pipeline.
    pipeline: StitchPipeline,
    /// Precomputed coverage boundary for no-black-edge clamping.
    coverage: CoverageBoundary,
    /// NV12 converter for encode readback (lazy-initialized on first call).
    nv12: Option<Nv12Converter>,
    /// RGBA readback helper for display in GUI frameworks (lazy-initialized).
    rgba: Option<RgbaReadback>,
}

impl StitchRenderer {
    /// Create a renderer for surface-based preview.
    ///
    /// Strips sRGB from the surface format to avoid double-gamma encoding,
    /// builds scene geometry from calibration, and precomputes the coverage
    /// boundary.
    ///
    /// # Arguments
    ///
    /// * `calibration` - Camera calibration with intrinsics and layout.
    /// * `gpu` - GPU context (must be compatible with the target surface).
    /// * `viewport` - Output viewport dimensions and blend settings.
    /// * `input_width` - Width of each input camera frame in pixels.
    /// * `input_height` - Height of each input camera frame in pixels.
    /// * `surface_format` - The surface's texture format (sRGB is stripped automatically).
    /// * `input_format` - Pixel format of the input frames
    ///   ([`Yuv420p`](InputFormat::Yuv420p) for file decode,
    ///   [`Nv12`](InputFormat::Nv12) for Jetson/NVDEC live input).
    pub fn new(
        calibration: Calibration,
        gpu: GpuContext,
        viewport: ViewportConfig,
        input_width: u32,
        input_height: u32,
        surface_format: wgpu::TextureFormat,
        input_format: InputFormat,
    ) -> Result<Self, PipelineError> {
        let render_format = Self::strip_srgb(surface_format);

        let aspect = calibration.lenses[0].width as f32 / calibration.lenses[0].height as f32;
        let scene = SceneGeometry::new(&calibration.topology, &calibration.framing, aspect);
        let coverage = CoverageBoundary::from_calibration(&calibration, &scene);

        let pipeline = StitchPipeline::with_gpu(
            gpu,
            calibration,
            viewport,
            input_width,
            input_height,
            render_format,
            input_format,
        )?;

        Ok(Self {
            pipeline,
            coverage,
            nv12: None,
            rgba: None,
        })
    }

    /// Render YUV420P frames to a texture view.
    ///
    /// Uploads the planes, composites the panorama at the given yaw/pitch,
    /// and writes the result to `view`. The view should use the stripped
    /// (non-sRGB) format returned by [`Self::strip_srgb`].
    pub fn render_yuv(
        &self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
        view: &wgpu::TextureView,
    ) -> Result<(), PipelineError> {
        self.pipeline.render_to_view(left, right, yaw, pitch, view)
    }

    /// Render NV12 frames to a texture view.
    ///
    /// Like [`Self::render_yuv`] but for NV12 input (Y + interleaved UV).
    /// The pipeline must have been created with NV12-compatible textures
    /// for this to produce correct results.
    pub fn render_nv12(
        &self,
        left: &Nv12Planes<'_>,
        right: &Nv12Planes<'_>,
        yaw: f32,
        pitch: f32,
        view: &wgpu::TextureView,
    ) -> Result<(), PipelineError> {
        self.pipeline
            .render_nv12_to_view(left, right, yaw, pitch, view)
    }

    /// Render YUV420P frames to either a surface view or the internal
    /// texture target, depending on the [`RenderTarget`] variant.
    ///
    /// Replaces the need to choose between `render_yuv` (surface) and
    /// `pipeline().render_to_target` (readback). The outcome tells the
    /// caller what happened:
    ///
    /// - [`SurfaceRenderOutcome::Submitted`] — GPU work was submitted for a
    ///   surface present; nothing left to do.
    /// - [`SurfaceRenderOutcome::Commands`] — a command buffer is ready for
    ///   readback (pass to [`RgbaReadback::readback`](crate::gpu::rgba_readback::RgbaReadback::readback)
    ///   or [`Nv12Converter::convert_and_readback`](crate::gpu::nv12_converter::Nv12Converter::convert_and_readback)).
    pub fn render(
        &self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
        target: RenderTarget<'_>,
    ) -> Result<SurfaceRenderOutcome, PipelineError> {
        match target {
            RenderTarget::Surface(view) => {
                self.pipeline
                    .render_to_view(left, right, yaw, pitch, view)?;
                Ok(SurfaceRenderOutcome::Submitted)
            }
            RenderTarget::Texture => {
                let cmd = self.pipeline.render_to_target(left, right, yaw, pitch)?;
                Ok(SurfaceRenderOutcome::Commands(cmd))
            }
        }
    }

    // ── Render + Readback API ──

    /// Render a stereo frame and get NV12 data for encoding.
    ///
    /// Renders to the internal target, runs NV12 conversion, and returns
    /// NV12 data. Uses triple-buffered readback (returns `None` on first
    /// two calls, data from 2 frames ago afterward).
    ///
    /// Use alongside `render_yuv()`/`render_nv12()` for combined display + recording:
    /// ```rust,ignore
    /// renderer.render_yuv(&left, &right, yaw, pitch, &surface_view)?;
    /// if recording {
    ///     if let Some(nv12) = renderer.render_and_readback_nv12(&left, &right, yaw, pitch)? {
    ///         encoder.submit(nv12_to_frame(nv12))?;
    ///     }
    /// }
    /// ```
    pub fn render_and_readback_nv12(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<Option<&[u8]>, PipelineError> {
        // Lazy-init NV12 converter. Round to NV12-safe dimensions
        // (width divisible by 4, height even) since window sizes can be odd.
        if self.nv12.is_none() {
            let w = self.pipeline.viewport().width & !3;
            let h = self.pipeline.viewport().height & !1;
            self.nv12 = Some(Nv12Converter::new(self.pipeline.gpu(), w, h).map_err(|e| {
                PipelineError::InvalidConfig {
                    reason: format!("NV12 converter init: {e}"),
                }
            })?);
            log::info!("StitchRenderer: NV12 readback initialized ({w}x{h})");
        }

        // Render to internal target.
        let cmd = self.pipeline.render_to_target(left, right, yaw, pitch)?;

        // NV12 convert + readback.
        let nv12 = self.nv12.as_mut().unwrap();
        let data = nv12
            .convert_and_readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)
            .map_err(|e| PipelineError::InvalidConfig {
                reason: format!("NV12 readback: {e}"),
            })?;

        Ok(data)
    }

    /// Flush remaining NV12 frames from the triple-buffer pipeline.
    ///
    /// Call after the last frame to get the remaining 1-2 frames.
    pub fn flush_nv12(&mut self) -> Result<Option<&[u8]>, PipelineError> {
        if let Some(ref mut nv12) = self.nv12 {
            nv12.flush_pending(self.pipeline.gpu())
                .map_err(|e| PipelineError::InvalidConfig {
                    reason: format!("NV12 flush: {e}"),
                })
        } else {
            Ok(None)
        }
    }

    /// Render a stereo frame and read back tightly-packed RGBA pixels.
    ///
    /// Intended for GUI frameworks and plugin consumers (OBS, egui) that
    /// need CPU-side pixels for display. Mirrors
    /// [`render_and_readback_nv12`](Self::render_and_readback_nv12) but
    /// outputs `width * height * 4` bytes of RGBA in the render target's
    /// texture format (`Rgba8Unorm` or `Bgra8Unorm` — see
    /// [`strip_srgb`](Self::strip_srgb)).
    ///
    /// Uses [`RgbaReadback`]'s triple-buffer staging so the returned slice
    /// is always from 2 frames ago. `None` on the first two calls during
    /// warmup; `Some(&[u8])` from the third call onward. Call
    /// [`flush_rgba`](Self::flush_rgba) in a loop after the frame loop to
    /// drain the last two frames.
    ///
    /// Use alongside [`render_yuv`](Self::render_yuv) when the GPU drives a
    /// surface and you only need readback for a secondary display path:
    /// ```rust,ignore
    /// renderer.render_yuv(&left, &right, yaw, pitch, &surface_view)?;
    /// if let Some(rgba) = renderer.render_and_readback_rgba(&left, &right, yaw, pitch)? {
    ///     shared_display_buffer.copy_from_slice(rgba);
    /// }
    /// ```
    ///
    /// Note that this submits its own render command buffer internally
    /// (the double-render is cheap — shader cost is negligible next to
    /// display compositing), so callers that only need readback should
    /// use this method alone rather than combining with `render_yuv`.
    pub fn render_and_readback_rgba(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<Option<&[u8]>, PipelineError> {
        if self.rgba.is_none() {
            let w = self.pipeline.viewport().width;
            let h = self.pipeline.viewport().height;
            self.rgba = Some(RgbaReadback::new(self.pipeline.gpu(), w, h).map_err(|e| {
                PipelineError::InvalidConfig {
                    reason: format!("RGBA readback init: {e}"),
                }
            })?);
            log::info!("StitchRenderer: RGBA readback initialized ({w}x{h})");
        }

        let cmd = self.pipeline.render_to_target(left, right, yaw, pitch)?;

        let rgba = self.rgba.as_mut().unwrap();
        let data = rgba
            .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)
            .map_err(|e| PipelineError::InvalidConfig {
                reason: format!("RGBA readback: {e}"),
            })?;

        Ok(data)
    }

    /// Flush one pending RGBA frame from the triple-buffer pipeline.
    ///
    /// Call in a loop after the frame loop ends to drain the final 1-2
    /// frames. Returns `None` when no frames remain.
    pub fn flush_rgba(&mut self) -> Result<Option<&[u8]>, PipelineError> {
        if let Some(ref mut rgba) = self.rgba {
            rgba.flush_pending(self.pipeline.gpu())
                .map_err(|e| PipelineError::InvalidConfig {
                    reason: format!("RGBA flush: {e}"),
                })
        } else {
            Ok(None)
        }
    }

    /// Access the GPU render target texture directly.
    ///
    /// Contains the rendered RGBA panorama (after any render call).
    /// Useful for custom compositing, GPU texture sharing, or screenshots.
    pub fn render_target(&self) -> &wgpu::Texture {
        self.pipeline.render_target()
    }

    pub fn coverage(&self) -> &CoverageBoundary {
        &self.coverage
    }

    /// Clamp a viewport pose to the coverage boundary, accounting for
    /// the current rig tilt. Consumers should call this instead of
    /// accessing coverage + rig_tilt separately.
    pub fn clamp_pose(
        &self,
        yaw: f32,
        pitch: f32,
        fov_degrees: f32,
        aspect: f32,
    ) -> crate::projection::ClampedPosition {
        self.coverage.safe_clamp(
            yaw,
            pitch,
            fov_degrees,
            aspect,
            self.pipeline.calibration().framing.tilt as f32,
        )
    }

    /// Maximum vertical FOV (degrees) that fits within the coverage area.
    pub fn max_fov_degrees(&self) -> f32 {
        self.coverage.max_fov_degrees()
    }

    /// Shared reference to the underlying stitch pipeline.
    pub fn pipeline(&self) -> &StitchPipeline {
        &self.pipeline
    }

    /// Mutable reference to the underlying stitch pipeline.
    ///
    /// Use this for operations like [`StitchPipeline::resize`] or
    /// [`StitchPipeline::set_fov`].
    pub fn pipeline_mut(&mut self) -> &mut StitchPipeline {
        &mut self.pipeline
    }

    /// Shared reference to the GPU context.
    pub fn gpu(&self) -> &GpuContext {
        self.pipeline.gpu()
    }

    /// Update calibration parameters and recompute the coverage boundary.
    ///
    /// Takes effect on the next render call. Useful for interactive
    /// calibration preview where the user adjusts sliders.
    pub fn update_calibration(&mut self, calibration: crate::calibration::Calibration) {
        self.pipeline.update_calibration(calibration);
        self.coverage =
            CoverageBoundary::from_calibration(self.pipeline.calibration(), &self.pipeline.scene);
    }

    /// Replace the topology (plane placement + seam) and recompute coverage.
    pub fn update_topology(&mut self, topology: crate::calibration::Topology) {
        let mut cal = self.pipeline.calibration().clone();
        cal.topology = topology;
        self.update_calibration(cal);
    }

    /// Replace the framing (axis offset, tilt, roll) and recompute coverage.
    pub fn update_framing(&mut self, framing: crate::calibration::Framing) {
        let mut cal = self.pipeline.calibration().clone();
        cal.framing = framing;
        self.update_calibration(cal);
    }

    /// Replace one or both cameras' intrinsics (focal, principal point,
    /// distortion) without rebuilding the pipeline or touching the layout.
    ///
    /// Intended for interactive lens-parameter tweaking in a GUI. See
    /// [`StitchPipeline::update_camera_params`] for the full contract.
    /// Coverage boundary is not recomputed because the layout (and thus
    /// the panorama extent) is unchanged - only the per-camera undistort
    /// uniforms shift, which affects what each camera "sees" through its
    /// lens but not how the stitched planes are arranged in world space.
    pub fn update_camera_params(
        &mut self,
        left: Option<crate::calibration::Lens>,
        right: Option<crate::calibration::Lens>,
    ) {
        self.pipeline.update_camera_params(left, right);
    }

    /// Set the seam blend width (0.0 = hard edge, 0.15 = default smooth blend).
    pub fn set_blend_width(&mut self, w: f32) {
        self.pipeline.set_blend_width(w);
    }

    pub fn set_rig_tilt(&mut self, radians: f32) {
        self.pipeline.set_rig_tilt(radians);
    }

    pub fn set_rig_roll(&mut self, radians: f32) {
        self.pipeline.set_rig_roll(radians);
    }

    /// Access the current calibration (for saving after adjustments).
    pub fn calibration(&self) -> &crate::calibration::Calibration {
        self.pipeline.calibration()
    }

    /// Strip sRGB encoding from a texture format.
    ///
    /// The stitch shader outputs sRGB-encoded values directly (BT.709
    /// YCbCr to R'G'B'). Rendering to an sRGB-format surface would apply
    /// sRGB encoding again, causing double-gamma (faded colors). This
    /// returns the equivalent linear format.
    pub fn strip_srgb(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
        match format {
            wgpu::TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8Unorm,
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_srgb_converts_known_formats() {
        assert_eq!(
            StitchRenderer::strip_srgb(wgpu::TextureFormat::Rgba8UnormSrgb),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        assert_eq!(
            StitchRenderer::strip_srgb(wgpu::TextureFormat::Bgra8UnormSrgb),
            wgpu::TextureFormat::Bgra8Unorm,
        );
    }

    #[test]
    fn strip_srgb_passes_through_non_srgb() {
        assert_eq!(
            StitchRenderer::strip_srgb(wgpu::TextureFormat::Rgba8Unorm),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        assert_eq!(
            StitchRenderer::strip_srgb(wgpu::TextureFormat::Bgra8Unorm),
            wgpu::TextureFormat::Bgra8Unorm,
        );
        assert_eq!(
            StitchRenderer::strip_srgb(wgpu::TextureFormat::Rgba16Float),
            wgpu::TextureFormat::Rgba16Float,
        );
    }
}
