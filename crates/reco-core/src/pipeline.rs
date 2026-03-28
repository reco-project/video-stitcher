//! Stitch pipeline orchestration.
//!
//! The [`StitchPipeline`] coordinates all stages: GPU setup, frame ingestion,
//! rendering, viewport cropping, and output encoding. It is the primary
//! entry point for consumers of `reco-core`.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use reco_core::pipeline::StitchPipeline;
//! use reco_core::calibration::MatchCalibration;
//! use reco_core::viewport::ViewportConfig;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let calibration: MatchCalibration = todo!("load from JSON");
//! let viewport = ViewportConfig::default();
//!
//! let pipeline = StitchPipeline::new(calibration, viewport, 1920, 1080).await?;
//! # Ok(())
//! # }
//! ```

use crate::calibration::MatchCalibration;
use crate::director::ViewportPosition;
use crate::gpu::{GpuContext, GpuError};
use crate::renderer::{InputFormat, RenderError, Renderer};
use crate::scene::SceneGeometry;
use crate::viewport::{ResolvedViewport, ViewportConfig};

use thiserror::Error;

/// Borrowed YUV420P plane references for pipeline input.
///
/// Tightly packed (no stride padding):
/// - `y`: `width × height` bytes
/// - `u`: `(width/2) × (height/2)` bytes
/// - `v`: `(width/2) × (height/2)` bytes
pub struct YuvPlanes<'a> {
    /// Y (luma) plane, full resolution.
    pub y: &'a [u8],
    /// U (Cb) plane, half resolution.
    pub u: &'a [u8],
    /// V (Cr) plane, half resolution.
    pub v: &'a [u8],
}

/// Errors from the stitch pipeline.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// GPU initialization failed.
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    /// Render error.
    #[error("render error: {0}")]
    Render(#[from] RenderError),
}

/// The main stitching pipeline.
///
/// Owns the GPU context, scene geometry, and renderer. Consumers provide
/// YUV420P frames and receive stitched RGBA output via [`Self::process_frame`].
pub struct StitchPipeline {
    /// GPU device and queue.
    pub gpu: GpuContext,
    /// 3D scene layout computed from calibration.
    pub scene: SceneGeometry,
    /// Calibration data (camera intrinsics + layout).
    pub calibration: MatchCalibration,
    /// Output viewport configuration.
    pub viewport: ViewportConfig,
    /// GPU renderer (textures, pipelines, bind groups).
    renderer: Renderer,
}

impl StitchPipeline {
    /// Create a new stitch pipeline.
    ///
    /// Initializes the GPU, computes the scene geometry from the
    /// calibration data, and creates the render pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Gpu`] if no compatible GPU is found.
    pub async fn new(
        calibration: MatchCalibration,
        viewport: ViewportConfig,
        input_width: u32,
        input_height: u32,
    ) -> Result<Self, PipelineError> {
        Self::with_gpu(
            GpuContext::new().await?,
            calibration,
            viewport,
            input_width,
            input_height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            InputFormat::Yuv420p,
        )
    }

    /// Create a pipeline with an existing GPU context and custom output format.
    ///
    /// Used by the preview window which needs a specific surface format
    /// and provides its own GPU context (selected with surface compatibility).
    pub fn with_gpu(
        gpu: GpuContext,
        calibration: MatchCalibration,
        viewport: ViewportConfig,
        input_width: u32,
        input_height: u32,
        output_format: wgpu::TextureFormat,
        input_format: InputFormat,
    ) -> Result<Self, PipelineError> {
        let scene = SceneGeometry::from_layout(&calibration.layout);
        let renderer = Renderer::new(
            &gpu,
            viewport.width,
            viewport.height,
            input_width,
            input_height,
            output_format,
            input_format,
        );

        log::info!(
            "Pipeline initialized: {}x{} output, GPU: {}",
            viewport.width,
            viewport.height,
            gpu.adapter_info.name
        );

        Ok(Self {
            gpu,
            scene,
            calibration,
            viewport,
            renderer,
        })
    }

    /// Render a frame directly to a texture view (for window display).
    ///
    /// Unlike [`Self::process_frame`], this does NOT read back to CPU — the result
    /// stays on the GPU and is presented to the surface.
    pub fn render_to_view(
        &self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
        target_view: &wgpu::TextureView,
    ) {
        self.renderer
            .upload_left_yuv(&self.gpu, left.y, left.u, left.v);
        self.renderer
            .upload_right_yuv(&self.gpu, right.y, right.u, right.v);

        let viewport = ResolvedViewport {
            config: self.viewport.clone(),
            position: ViewportPosition { yaw, pitch },
        };

        self.renderer.render_to_view(
            &self.gpu,
            &self.scene,
            &self.calibration,
            &viewport,
            self.viewport.blend_width,
            target_view,
        );
    }

    /// Resize the depth buffer for the preview window.
    pub fn resize_depth(&mut self, width: u32, height: u32) {
        self.renderer.resize_depth(&self.gpu, width, height);
    }

    /// Process a single frame through the GPU pipeline.
    ///
    /// Uploads left and right YUV420P planes to the GPU, renders the stitched
    /// panorama at the given viewport position, and reads back the result.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "process_frame")
    )]
    pub fn process_frame(
        &self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<Vec<u8>, PipelineError> {
        self.renderer
            .upload_left_yuv(&self.gpu, left.y, left.u, left.v);
        self.renderer
            .upload_right_yuv(&self.gpu, right.y, right.u, right.v);

        let viewport = ResolvedViewport {
            config: self.viewport.clone(),
            position: ViewportPosition { yaw, pitch },
        };

        Ok(self.renderer.render_frame(
            &self.gpu,
            &self.scene,
            &self.calibration,
            &viewport,
            self.viewport.blend_width,
        )?)
    }

    /// Render a frame assuming textures are already populated (zero-copy path).
    ///
    /// Used with CUDA/Vulkan shared textures where the decode thread writes
    /// frame data directly to GPU memory via `cuMemcpy2D`. No CPU upload needed.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "process_frame_gpu")
    )]
    pub fn process_frame_gpu(&self, yaw: f32, pitch: f32) -> Result<Vec<u8>, PipelineError> {
        let viewport = ResolvedViewport {
            config: self.viewport.clone(),
            position: ViewportPosition { yaw, pitch },
        };

        Ok(self.renderer.render_frame(
            &self.gpu,
            &self.scene,
            &self.calibration,
            &viewport,
            self.viewport.blend_width,
        )?)
    }

    /// Render a frame to the internal render target without CPU readback.
    ///
    /// Uploads YUV planes and returns the render `CommandBuffer` without
    /// submitting. The caller must submit it (typically together with NV12
    /// conversion commands via [`Nv12Converter`](crate::nv12_converter::Nv12Converter)).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "render_to_target")
    )]
    pub fn render_to_target(
        &self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> wgpu::CommandBuffer {
        self.renderer
            .upload_left_yuv(&self.gpu, left.y, left.u, left.v);
        self.renderer
            .upload_right_yuv(&self.gpu, right.y, right.u, right.v);

        let viewport = ResolvedViewport {
            config: self.viewport.clone(),
            position: ViewportPosition { yaw, pitch },
        };

        self.renderer.render_to_target(
            &self.gpu,
            &self.scene,
            &self.calibration,
            &viewport,
            self.viewport.blend_width,
        )
    }

    /// Render to the internal target without upload or readback (zero-copy path).
    ///
    /// Returns the render `CommandBuffer` without submitting. Assumes textures
    /// are already populated via CUDA/Vulkan shared memory.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "render_to_target_gpu")
    )]
    pub fn render_to_target_gpu(&self, yaw: f32, pitch: f32) -> wgpu::CommandBuffer {
        let viewport = ResolvedViewport {
            config: self.viewport.clone(),
            position: ViewportPosition { yaw, pitch },
        };

        self.renderer.render_to_target(
            &self.gpu,
            &self.scene,
            &self.calibration,
            &viewport,
            self.viewport.blend_width,
        )
    }

    /// Access the rendered RGBA texture for NV12 conversion.
    pub fn render_target(&self) -> &wgpu::Texture {
        self.renderer.render_target()
    }

    /// Mutable access to the renderer (for swapping shared textures).
    pub fn renderer_mut(&mut self) -> &mut Renderer {
        &mut self.renderer
    }
}
