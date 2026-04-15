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
//! use reco_core::stitch_renderer::StitchRenderer;
//!
//! let renderer = StitchRenderer::new(
//!     calibration, gpu, viewport, input_width, input_height, surface_format,
//! )?;
//!
//! // In the render loop:
//! renderer.render_yuv(&left_planes, &right_planes, yaw, pitch, &view)?;
//! ```

use crate::calibration::MatchCalibration;
use crate::gpu::GpuContext;
use crate::nv12_converter::Nv12Converter;
use crate::pipeline::{Nv12Planes, PipelineError, StitchPipeline, YuvPlanes};
use crate::projection::CoverageBoundary;
use crate::renderer::InputFormat;
use crate::scene::SceneGeometry;
use crate::viewport::ViewportConfig;

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
    /// NV12 converter for readback (lazy-initialized on first readback call).
    nv12: Option<Nv12Converter>,
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
    pub fn new(
        calibration: MatchCalibration,
        gpu: GpuContext,
        viewport: ViewportConfig,
        input_width: u32,
        input_height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Result<Self, PipelineError> {
        let render_format = Self::strip_srgb(surface_format);

        let aspect = calibration.left.width as f32 / calibration.left.height as f32;
        let scene = SceneGeometry::from_layout_with_aspect(&calibration.layout, aspect);
        let coverage = CoverageBoundary::from_calibration(&calibration, &scene);

        let pipeline = StitchPipeline::with_gpu(
            gpu,
            calibration,
            viewport,
            input_width,
            input_height,
            render_format,
            InputFormat::Yuv420p,
        )?;

        Ok(Self {
            pipeline,
            coverage,
            nv12: None,
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

    /// Access the GPU render target texture directly.
    ///
    /// Contains the rendered RGBA panorama (after any render call).
    /// Useful for custom compositing, GPU texture sharing, or screenshots.
    pub fn render_target(&self) -> &wgpu::Texture {
        self.pipeline.render_target()
    }

    /// The precomputed coverage boundary.
    pub fn coverage(&self) -> &CoverageBoundary {
        &self.coverage
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
    pub fn update_calibration(&mut self, calibration: crate::calibration::MatchCalibration) {
        self.pipeline.update_calibration(calibration);
        self.coverage =
            CoverageBoundary::from_calibration(self.pipeline.calibration(), &self.pipeline.scene);
    }

    /// Update only the plane layout and recompute coverage.
    pub fn update_layout(&mut self, layout: crate::calibration::PlaneLayout) {
        self.pipeline.update_layout(layout);
        self.coverage =
            CoverageBoundary::from_calibration(self.pipeline.calibration(), &self.pipeline.scene);
    }

    /// Set the seam blend width (0.0 = hard edge, 0.15 = default smooth blend).
    pub fn set_blend_width(&mut self, w: f32) {
        self.pipeline.viewport.blend_width = w;
    }

    /// Set rig tilt correction in radians.
    pub fn set_rig_tilt(&mut self, radians: f32) {
        self.pipeline.viewport.rig_tilt = radians;
    }

    /// Access the current calibration (for saving after adjustments).
    pub fn calibration(&self) -> &crate::calibration::MatchCalibration {
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
