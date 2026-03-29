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

/// Borrowed NV12 plane references for pipeline input.
///
/// Tightly packed (no stride padding):
/// - `y`: `width × height` bytes
/// - `uv`: `width × (height/2)` bytes (interleaved U,V)
pub struct Nv12Planes<'a> {
    /// Y (luma) plane, full resolution.
    pub y: &'a [u8],
    /// Interleaved UV (CbCr) plane, half resolution in each dimension.
    pub uv: &'a [u8],
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
/// YUV420P or NV12 frames and receive stitched RGBA output via
/// [`Self::process_frame`] or [`Self::render_to_target_nv12`].
pub struct StitchPipeline {
    /// GPU device and queue.
    pub(crate) gpu: GpuContext,
    /// 3D scene layout computed from calibration.
    pub(crate) scene: SceneGeometry,
    /// Calibration data (camera intrinsics + layout).
    pub(crate) calibration: MatchCalibration,
    /// Output viewport configuration.
    pub(crate) viewport: ViewportConfig,
    /// GPU renderer (textures, pipelines, bind groups).
    renderer: Renderer,
}

/// Pre-built bind groups for GPU-resident zero-copy sources.
///
/// Created by [`StitchPipeline::configure_gpu_source`]. Each slot
/// corresponds to a double-buffer index used by the decode thread.
#[cfg(target_os = "linux")]
pub struct GpuSourceBindGroups {
    left: [wgpu::BindGroup; 2],
    right: [wgpu::BindGroup; 2],
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
        output_format: impl Into<wgpu::TextureFormat>,
        input_format: InputFormat,
    ) -> Result<Self, PipelineError> {
        let output_format = output_format.into();
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

    /// The name of the GPU this pipeline is running on.
    pub fn gpu_name(&self) -> &str {
        self.gpu.gpu_name()
    }

    /// Shared reference to the GPU context.
    ///
    /// Needed by consumers that create their own wgpu resources
    /// (e.g. surface configuration for a preview window).
    pub fn gpu(&self) -> &GpuContext {
        &self.gpu
    }

    /// Current viewport configuration.
    pub fn viewport(&self) -> &ViewportConfig {
        &self.viewport
    }

    /// Resize the output viewport.
    ///
    /// Call this when the window or output dimensions change.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.viewport.width = width;
        self.viewport.height = height;
    }

    /// Set the horizontal field of view in degrees.
    pub fn set_fov(&mut self, fov_degrees: f32) {
        self.viewport.fov_degrees = fov_degrees;
    }

    /// Get the current field of view in degrees.
    pub fn fov(&self) -> f32 {
        self.viewport.fov_degrees
    }

    /// Set up bind groups for GPU-resident zero-copy input.
    ///
    /// Creates bind groups for the provided shared textures (Y + UV per slot
    /// per camera). Call once during setup, then pass the result to
    /// [`Self::render_gpu_frame`] each frame.
    #[cfg(target_os = "linux")]
    pub fn configure_gpu_source(
        &mut self,
        left_textures: [(
            &crate::vulkan_interop::SharedTexture,
            &crate::vulkan_interop::SharedTexture,
        ); 2],
        right_textures: [(
            &crate::vulkan_interop::SharedTexture,
            &crate::vulkan_interop::SharedTexture,
        ); 2],
    ) -> GpuSourceBindGroups {
        let left_bg_0 = self.renderer.create_texture_bind_group(
            &left_textures[0].0.texture,
            &left_textures[0].1.texture,
            "left_slot0",
        );
        let left_bg_1 = self.renderer.create_texture_bind_group(
            &left_textures[1].0.texture,
            &left_textures[1].1.texture,
            "left_slot1",
        );
        let right_bg_0 = self.renderer.create_texture_bind_group(
            &right_textures[0].0.texture,
            &right_textures[0].1.texture,
            "right_slot0",
        );
        let right_bg_1 = self.renderer.create_texture_bind_group(
            &right_textures[1].0.texture,
            &right_textures[1].1.texture,
            "right_slot1",
        );
        GpuSourceBindGroups {
            left: [left_bg_0, left_bg_1],
            right: [right_bg_0, right_bg_1],
        }
    }

    /// Select bind groups for a GPU-resident frame and render.
    ///
    /// Call this instead of manually setting bind groups on the renderer.
    #[cfg(target_os = "linux")]
    pub fn render_gpu_frame(
        &mut self,
        bind_groups: &GpuSourceBindGroups,
        left_slot: u8,
        right_slot: u8,
        yaw: f32,
        pitch: f32,
    ) -> wgpu::CommandBuffer {
        self.renderer
            .set_left_bind_group(bind_groups.left[left_slot as usize].clone());
        self.renderer
            .set_right_bind_group(bind_groups.right[right_slot as usize].clone());
        self.render_to_target_gpu(yaw, pitch)
    }

    /// Render from imported GPU textures (e.g. Metal/VideoToolbox zero-copy).
    ///
    /// Takes raw Y + UV texture references for each camera, creates bind groups,
    /// and renders. Unlike [`Self::render_gpu_frame`] which uses pre-built
    /// double-buffered bind groups, this creates them per-frame (the overhead
    /// is negligible compared to decode time).
    pub fn render_imported_textures(
        &mut self,
        left_y: &wgpu::Texture,
        left_uv: &wgpu::Texture,
        right_y: &wgpu::Texture,
        right_uv: &wgpu::Texture,
        yaw: f32,
        pitch: f32,
    ) -> wgpu::CommandBuffer {
        let left_bg =
            self.renderer
                .create_texture_bind_group(left_y, left_uv, "metal_left");
        let right_bg =
            self.renderer
                .create_texture_bind_group(right_y, right_uv, "metal_right");
        self.renderer.set_left_bind_group(left_bg);
        self.renderer.set_right_bind_group(right_bg);
        self.render_to_target_gpu(yaw, pitch)
    }

    /// Process a CPU-resident stereo frame and return the render command buffer.
    ///
    /// Handles YUV420P vs NV12 format differences internally.
    /// For GPU-resident frames, use [`Self::render_gpu_frame`] instead.
    pub fn render_stereo_frame(
        &self,
        frame: &crate::source::StereoFrame,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        use crate::source::StereoFrame;
        match frame {
            StereoFrame::Yuv420p(pair) => {
                let left = YuvPlanes {
                    y: &pair.left.y,
                    u: &pair.left.u,
                    v: &pair.left.v,
                };
                let right = YuvPlanes {
                    y: &pair.right.y,
                    u: &pair.right.u,
                    v: &pair.right.v,
                };
                self.render_to_target(&left, &right, yaw, pitch)
            }
            StereoFrame::Nv12(pair) => {
                let left = Nv12Planes {
                    y: &pair.left.y,
                    uv: &pair.left.uv,
                };
                let right = Nv12Planes {
                    y: &pair.right.y,
                    uv: &pair.right.uv,
                };
                self.render_to_target_nv12(&left, &right, yaw, pitch)
            }
            StereoFrame::GpuResident { .. } => {
                panic!("GpuResident frames must use render_gpu_frame()")
            }
        }
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
    ) -> Result<(), PipelineError> {
        self.renderer
            .upload_left_yuv(&self.gpu, left.y, left.u, left.v)?;
        self.renderer
            .upload_right_yuv(&self.gpu, right.y, right.u, right.v)?;

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
        Ok(())
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
            .upload_left_yuv(&self.gpu, left.y, left.u, left.v)?;
        self.renderer
            .upload_right_yuv(&self.gpu, right.y, right.u, right.v)?;

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
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        self.renderer
            .upload_left_yuv(&self.gpu, left.y, left.u, left.v)?;
        self.renderer
            .upload_right_yuv(&self.gpu, right.y, right.u, right.v)?;

        let viewport = ResolvedViewport {
            config: self.viewport.clone(),
            position: ViewportPosition { yaw, pitch },
        };

        Ok(self.renderer.render_to_target(
            &self.gpu,
            &self.scene,
            &self.calibration,
            &viewport,
            self.viewport.blend_width,
        ))
    }

    /// Upload NV12 frames and render to the internal target.
    ///
    /// Like `render_to_target` but accepts NV12 input (Y + interleaved UV)
    /// instead of YUV420P. Requires the pipeline to be initialized with
    /// `InputFormat::Nv12`.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "render_to_target_nv12")
    )]
    pub fn render_to_target_nv12(
        &self,
        left: &Nv12Planes<'_>,
        right: &Nv12Planes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        self.renderer.upload_left_nv12(&self.gpu, left.y, left.uv)?;
        self.renderer
            .upload_right_nv12(&self.gpu, right.y, right.uv)?;

        let viewport = ResolvedViewport {
            config: self.viewport.clone(),
            position: ViewportPosition { yaw, pitch },
        };

        Ok(self.renderer.render_to_target(
            &self.gpu,
            &self.scene,
            &self.calibration,
            &viewport,
            self.viewport.blend_width,
        ))
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
}
