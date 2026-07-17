//! Stitch pipeline orchestration.
//!
//! `StitchPipeline` coordinates all stages: GPU setup, frame ingestion,
//! rendering, and viewport cropping. It is crate-internal - the
//! [`GpuExecutor`](crate::stitch::GpuExecutor) is its sole owner, and
//! consumers reach rendering through the engine or a
//! [`StitchSession`](crate::session::StitchSession). This module also
//! re-exports the public frame-plane value types
//! ([`YuvPlanes`], [`Nv12Planes`], [`BgraPlanes`], ...) and
//! [`PipelineError`], the pipeline half of the engine's error type.

use super::renderer::{InputFormat, RenderError, Renderer};
use super::viewport::ViewportSize;
use crate::calibration::Calibration;
use crate::geometry::Pose;
use crate::gpu::{GpuContext, GpuError};
use crate::projection::l_shape::PlaneScene;

use thiserror::Error;

pub use super::planes::{BgraPlanes, FramePlaneView, Nv12Planes, StridedYuvPlanes, YuvPlanes};

/// Errors from the stitch pipeline. `Clone + Send + Sync` so consumers
/// posting results to worker threads can carry the typed error.
#[derive(Debug, Clone, Error)]
pub enum PipelineError {
    /// GPU initialization failed.
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    /// The calibration document is invalid.
    #[error("invalid calibration: {0}")]
    Calibration(#[from] crate::calibration::CalibrationError),

    /// Render error.
    #[error("render error: {0}")]
    Render(#[from] RenderError),

    /// The active topology has no GPU plane pass (the mono cylinder,
    /// until its projection-owned fullscreen pass lands).
    #[error("this topology has no GPU plane scene; the mono GPU pass is not wired yet")]
    NoPlaneScene,

    /// Wrong StereoFrame variant for this render method.
    #[error("unsupported frame variant: {reason}")]
    UnsupportedFrameVariant {
        /// Description of the mismatch.
        reason: &'static str,
    },

    /// Invalid configuration.
    #[error("invalid config: {reason}")]
    InvalidConfig {
        /// What is wrong.
        reason: String,
    },
}

/// The main stitching pipeline.
///
/// Owns the GPU context, scene geometry, and renderer. The
/// [`GpuExecutor`](crate::stitch::GpuExecutor) is its sole owner;
/// everything else reaches rendering through the engine.
pub(crate) struct StitchPipeline {
    /// GPU device and queue.
    pub(crate) gpu: GpuContext,
    /// L-shape plane placement. `None` for plane-less topologies -
    /// their GPU pass carries its own geometry when it lands.
    pub(crate) scene: Option<PlaneScene>,
    /// Calibration data (camera intrinsics + layout).
    pub(crate) calibration: Calibration,
    /// Output viewport configuration.
    pub(crate) viewport_size: ViewportSize,
    /// GPU renderer (textures, pipelines, bind groups).
    renderer: Renderer,
    /// Input frame dimensions.
    input_width: u32,
    input_height: u32,
}

/// Pre-built bind groups for GPU-resident zero-copy sources.
///
/// Created by [`StitchPipeline::configure_gpu_source`]. Each slot
/// corresponds to a double-buffer index used by the decode thread.
#[cfg(target_os = "linux")]
pub(crate) struct GpuSourceBindGroups {
    left: [wgpu::BindGroup; 2],
    right: [wgpu::BindGroup; 2],
}

impl StitchPipeline {
    /// The L-shape's plane placement for the GPU uniforms, when the
    /// topology has planes. Both planes share lens 0's aspect (the
    /// documented limitation, kept identical to the CPU maps).
    fn derive_plane_scene(calibration: &Calibration) -> Option<PlaneScene> {
        calibration
            .topology
            .l_shape()
            .map(|t| t.scene(&calibration.framing, calibration.lenses[0].aspect()))
    }

    /// Create a pipeline with an existing GPU context and custom output format.
    ///
    /// Used by the preview window which needs a specific surface format
    /// and provides its own GPU context (selected with surface compatibility).
    pub fn with_gpu(
        gpu: GpuContext,
        program: &crate::render::GpuProgram,
        calibration: Calibration,
        viewport_size: ViewportSize,
        input_width: u32,
        input_height: u32,
        output_format: impl Into<wgpu::TextureFormat>,
        input_format: InputFormat,
    ) -> Result<Self, PipelineError> {
        // Validate inputs before GPU resource creation. This is THE
        // enforcement boundary for in-memory calibrations: every
        // constructor (StitchCore, StitchSession, the preview bridge,
        // StitchJob) funnels through here, so a wrong lens count or a
        // NaN surfaces as a typed error instead of an index panic or a
        // GPU hang further down.
        calibration.validate()?;
        if let Err(e) = viewport_size.validate() {
            return Err(PipelineError::InvalidConfig { reason: e });
        }
        if input_width == 0 || input_height == 0 {
            return Err(PipelineError::InvalidConfig {
                reason: format!("input dimensions must be > 0, got {input_width}x{input_height}"),
            });
        }
        if input_width > crate::calibration::MAX_DIM || input_height > crate::calibration::MAX_DIM {
            return Err(PipelineError::InvalidConfig {
                reason: format!(
                    "input dimensions {input_width}x{input_height} exceed MAX_DIM ({})",
                    crate::calibration::MAX_DIM
                ),
            });
        }

        let output_format = output_format.into();
        let scene = Self::derive_plane_scene(&calibration);
        let renderer = Renderer::new(
            &gpu,
            program,
            viewport_size.width,
            viewport_size.height,
            input_width,
            input_height,
            output_format,
            input_format,
            // Quad geometry: the source aspect, independent of whether
            // the topology has planes (harmless placeholder otherwise).
            calibration.lenses[0].aspect(),
        );

        log::info!(
            "Pipeline initialized: {}x{} output, GPU: {}",
            viewport_size.width,
            viewport_size.height,
            gpu.adapter_info.name
        );

        Ok(Self {
            gpu,
            scene,
            calibration,
            viewport_size,
            renderer,
            input_width,
            input_height,
        })
    }

    /// Shared reference to the GPU context.
    ///
    /// Needed by consumers that create their own wgpu resources
    /// (e.g. surface configuration for a preview window).
    pub fn gpu(&self) -> &GpuContext {
        &self.gpu
    }

    /// The calibration data this pipeline was created with.
    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    /// The projection in effect: a borrow of the document's topology.
    pub(crate) fn projection(&self) -> &dyn crate::projection::Projection {
        self.calibration.topology.projection()
    }

    /// The current output viewport configuration.
    pub fn viewport_size(&self) -> &ViewportSize {
        &self.viewport_size
    }

    /// Input frame dimensions as `(width, height)`.
    pub fn source_info(&self) -> (u32, u32) {
        (self.input_width, self.input_height)
    }

    /// Input pixel format the pipeline was built for. Needed by the
    /// stacked-video GPU packer so it can pick the matching shader
    /// kernel variant (separate R8 planes for YUV420P vs interleaved
    /// Rg8 UV for NV12) without the consumer passing the format
    /// through a second time.
    pub(crate) fn input_format(&self) -> super::renderer::InputFormat {
        self.renderer.input_format()
    }

    /// Left-side source plane views (Y/U/V texture views). Used by
    /// the stacked-video GPU packer to read the same uploaded
    /// source data the stitch shader samples; the pack runs in
    /// parallel with the panorama render into its own atlas buffer.
    /// For NV12 inputs the `U` view is the interleaved UV texture
    /// and the `V` view is a 1×1 dummy.
    pub(crate) fn left_plane_views(
        &self,
    ) -> (wgpu::TextureView, wgpu::TextureView, wgpu::TextureView) {
        self.renderer.left_plane_views()
    }

    /// Right-side counterpart to [`Self::left_plane_views`].
    pub(crate) fn right_plane_views(
        &self,
    ) -> (wgpu::TextureView, wgpu::TextureView, wgpu::TextureView) {
        self.renderer.right_plane_views()
    }

    /// Update the viewport metadata (aspect ratio, projection matrix).
    ///
    /// **Important:** this does NOT recreate GPU textures or the render
    /// target. Use this for viewport-metadata changes (e.g. surface
    /// reconfigure in a preview window). For actual output resolution
    /// changes, rebuild the pipeline with [`Self::with_gpu`].
    /// Returns `Some((width, height))` on success, or `None` if the
    /// dimensions were zero (ignored). Consumers that own external
    /// staging buffers (e.g.
    /// [`RgbaReadback`](crate::gpu::rgba_readback::RgbaReadback)) should
    /// recreate them when the returned size differs from the previous.
    pub fn resize(&mut self, width: u32, height: u32) -> Option<(u32, u32)> {
        if width == 0 || height == 0 {
            log::warn!("resize({width}, {height}) ignored: dimensions must be non-zero");
            return None;
        }
        self.viewport_size.width = width;
        self.viewport_size.height = height;
        Some((width, height))
    }

    /// Set the lens distortion correction amount for every lens (per-frame
    /// uniform; no scene rebuild).
    pub fn set_lens_correction_amount(&mut self, amount: f32) {
        let c = amount.clamp(0.0, 1.0);
        for lens in &mut self.calibration.lenses {
            lens.correction = c;
        }
    }

    /// Set the seam blend width (per-frame uniform; no scene rebuild).
    pub fn set_blend_width(&mut self, width: f32) {
        if let Some(t) = self.calibration.topology.l_shape_mut() {
            t.blend_width = width;
        } else {
            log::warn!("set_blend_width({width}) ignored: this topology has no seam");
        }
    }

    /// Update calibration parameters. Recomputes the plane scene from the
    /// new layout. Takes effect on the next render call (uniforms are rebuilt
    /// each frame from the stored calibration and scene).
    ///
    /// No GPU pipeline recreation needed - only the uniform data changes.
    pub fn update_calibration(&mut self, calibration: Calibration) {
        self.scene = Self::derive_plane_scene(&calibration);
        self.calibration = calibration;
        log::debug!("Pipeline calibration updated");
    }

    /// Replace the topology (plane placement + seam), rebuilding the scene.
    pub fn update_topology(&mut self, topology: crate::calibration::Topology) {
        let mut cal = self.calibration.clone();
        cal.topology = topology;
        self.update_calibration(cal);
    }

    /// Replace the framing (axis offset, tilt, roll), rebuilding the scene.
    pub fn update_framing(&mut self, framing: crate::calibration::Framing) {
        let mut cal = self.calibration.clone();
        cal.framing = framing;
        self.update_calibration(cal);
    }

    /// Update per-camera intrinsics (focal, principal point, distortion)
    /// for one or both cameras without touching the plane layout or rig
    /// orientation.
    ///
    /// Intended for interactive lens tweaking in a GUI: each `Lens`
    /// change is written into the shader's per-frame uniform buffer, so the
    /// next render call reflects the new values. No GPU pipeline or scene
    /// recreation is needed - cheap enough (~microseconds) to call on
    /// every slider drag.
    ///
    /// `left`/`right` are `None` to leave that side untouched. If both are
    /// `None` this is a no-op. Passing `Some` for a side replaces that
    /// side's `Lens` on the stored calibration; the next render
    /// picks it up automatically.
    ///
    /// Does not recompute the plane scene because the plane layout is
    /// unchanged; only the camera intrinsics (which live on the stored
    /// calibration and are re-read each frame) need updating.
    pub fn update_camera_params(
        &mut self,
        left: Option<crate::calibration::Lens>,
        right: Option<crate::calibration::Lens>,
    ) {
        if left.is_none() && right.is_none() {
            return;
        }
        if let Some(l) = left {
            self.calibration.lenses[0] = l;
        }
        if let Some(r) = right {
            self.calibration.lenses[1] = r;
        }
        log::debug!("Pipeline camera params updated");
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
            &crate::interop::vulkan::SharedTexture,
            &crate::interop::vulkan::SharedTexture,
        ); 2],
        right_textures: [(
            &crate::interop::vulkan::SharedTexture,
            &crate::interop::vulkan::SharedTexture,
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
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        self.renderer
            .set_left_bind_group(bind_groups.left[left_slot as usize].clone());
        self.renderer
            .set_right_bind_group(bind_groups.right[right_slot as usize].clone());
        self.render_to_target_gpu(pose)
    }

    /// Create a texture bind group from Y + UV textures.
    pub fn create_texture_bind_group(
        &self,
        y_texture: &wgpu::Texture,
        uv_texture: &wgpu::Texture,
        label: &str,
    ) -> wgpu::BindGroup {
        self.renderer
            .create_texture_bind_group(y_texture, uv_texture, label)
    }

    /// Render from pre-built bind groups (VRAM pool path).
    pub fn render_with_bind_groups(
        &mut self,
        left_bg: &wgpu::BindGroup,
        right_bg: &wgpu::BindGroup,
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        self.renderer.set_left_bind_group(left_bg.clone());
        self.renderer.set_right_bind_group(right_bg.clone());
        self.render_to_target_gpu(pose)
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
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        let left_bg = self
            .renderer
            .create_texture_bind_group(left_y, left_uv, "metal_left");
        let right_bg = self
            .renderer
            .create_texture_bind_group(right_y, right_uv, "metal_right");
        self.renderer.set_left_bind_group(left_bg);
        self.renderer.set_right_bind_group(right_bg);
        self.render_to_target_gpu(pose)
    }

    /// Render from pre-built GPU texture views.
    ///
    /// Used by the D3D11VA zero-copy path where NV12 plane views are
    /// created from `TextureAspect::Plane0` / `Plane1`.
    #[cfg(target_os = "windows")]
    pub fn render_imported_views(
        &mut self,
        left_y: &wgpu::TextureView,
        left_uv: &wgpu::TextureView,
        right_y: &wgpu::TextureView,
        right_uv: &wgpu::TextureView,
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        let left_bg = self
            .renderer
            .create_bind_group_from_views(left_y, left_uv, "d3d11_left");
        let right_bg = self
            .renderer
            .create_bind_group_from_views(right_y, right_uv, "d3d11_right");
        self.renderer.set_left_bind_group(left_bg);
        self.renderer.set_right_bind_group(right_bg);
        self.render_to_target_gpu(pose)
    }

    /// Process a CPU-resident stereo frame and return the render command buffer.
    ///
    /// Handles YUV420P vs NV12 format differences internally.
    /// For GPU-resident frames, use [`Self::render_gpu_frame`] instead.
    pub fn render_stereo_frame(
        &self,
        frame: &crate::source::StereoFrame,
        pose: Pose,
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
                self.render_to_target(&left, &right, pose)
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
                self.render_to_target_nv12(&left, &right, pose)
            }
            StereoFrame::GpuResident { .. } => Err(PipelineError::UnsupportedFrameVariant {
                reason: "GpuResident frames must use render_gpu_frame()",
            }),
            #[allow(unreachable_patterns)]
            _ => Err(PipelineError::UnsupportedFrameVariant {
                reason: "unsupported StereoFrame variant for CPU render path",
            }),
        }
    }

    /// Render a frame directly to a texture view (for window display).
    ///
    /// Unlike the encode path, this does NOT read back to CPU — the result
    /// stays on the GPU and is presented to the surface.
    pub fn render_to_view(
        &self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        pose: Pose,
        target_view: &wgpu::TextureView,
    ) -> Result<(), PipelineError> {
        let Some(scene) = &self.scene else {
            return Err(PipelineError::NoPlaneScene);
        };
        self.renderer
            .upload_left_yuv(&self.gpu, left.y, left.u, left.v)?;
        self.renderer
            .upload_right_yuv(&self.gpu, right.y, right.u, right.v)?;

        self.renderer.render_to_view(
            &self.gpu,
            scene,
            &self.calibration,
            pose,
            self.viewport_size.aspect_ratio(),
            self.calibration.topology.blend_width(),
            target_view,
        );
        Ok(())
    }

    /// Render a frame to the internal render target without CPU readback.
    ///
    /// Uploads YUV planes and returns the render `CommandBuffer` without
    /// submitting. The caller must submit it (typically together with NV12
    /// conversion commands via the NV12 converter).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "render_to_target")
    )]
    pub fn render_to_target(
        &self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        let Some(scene) = &self.scene else {
            return Err(PipelineError::NoPlaneScene);
        };
        self.renderer
            .upload_left_yuv(&self.gpu, left.y, left.u, left.v)?;
        self.renderer
            .upload_right_yuv(&self.gpu, right.y, right.u, right.v)?;

        Ok(self.renderer.render_to_target(
            &self.gpu,
            scene,
            &self.calibration,
            pose,
            self.calibration.topology.blend_width(),
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
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        let Some(scene) = &self.scene else {
            return Err(PipelineError::NoPlaneScene);
        };
        self.renderer.upload_left_nv12(&self.gpu, left.y, left.uv)?;
        self.renderer
            .upload_right_nv12(&self.gpu, right.y, right.uv)?;

        Ok(self.renderer.render_to_target(
            &self.gpu,
            scene,
            &self.calibration,
            pose,
            self.calibration.topology.blend_width(),
        ))
    }

    /// Upload packed BGRA/RGBA frames and render to the internal target.
    ///
    /// Expects each plane as `width * height * 4` bytes in (R, G, B, A) byte
    /// order. Use [`BgraPlanes::from_bgra_swizzle_into`] when the source
    /// is BGRA. Requires the pipeline to be initialized with
    /// [`InputFormat::Bgra`](crate::render::renderer::InputFormat#variant.Bgra).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "render_to_target_bgra")
    )]
    pub fn render_to_target_bgra(
        &self,
        left: &BgraPlanes<'_>,
        right: &BgraPlanes<'_>,
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        let Some(scene) = &self.scene else {
            return Err(PipelineError::NoPlaneScene);
        };
        self.renderer.upload_left_bgra(&self.gpu, left.rgba)?;
        self.renderer.upload_right_bgra(&self.gpu, right.rgba)?;

        Ok(self.renderer.render_to_target(
            &self.gpu,
            scene,
            &self.calibration,
            pose,
            self.calibration.topology.blend_width(),
        ))
    }

    /// Render from GPU-resident RGBA textures (e.g. Bayer demosaic output).
    ///
    /// Copies source textures into the input planes, then renders the
    /// stitch to the internal target. Returns the complete command buffer.
    /// The caller submits the demosaic encoder first, then this one.
    /// Requires `InputFormat::Bgra`.
    pub fn render_from_gpu_rgba(
        &self,
        left_rgba: &wgpu::Texture,
        right_rgba: &wgpu::Texture,
        pose: Pose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        // Copy demosaiced textures into stitch pipeline input planes
        let mut copy_encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bayer_copy"),
                });
        self.renderer
            .copy_texture_to_left(&mut copy_encoder, left_rgba);
        self.renderer
            .copy_texture_to_right(&mut copy_encoder, right_rgba);
        self.gpu
            .queue
            .submit(std::iter::once(copy_encoder.finish()));

        // Render stitch (reads from the just-populated input textures)
        self.render_to_target_gpu(pose)
    }

    /// Render to the internal target without upload or readback (zero-copy path).
    ///
    /// Returns the render `CommandBuffer` without submitting. Assumes textures
    /// are already populated via CUDA/Vulkan shared memory.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "render_to_target_gpu")
    )]
    /// Render to the internal target using whatever textures are currently
    /// bound. Call [`Self::render_imported_textures`] once to set up
    /// bind groups, then use this for subsequent frames with the same
    /// textures to avoid per-frame bind group allocation.
    pub fn render_to_target_gpu(&self, pose: Pose) -> Result<wgpu::CommandBuffer, PipelineError> {
        let Some(scene) = &self.scene else {
            return Err(PipelineError::NoPlaneScene);
        };
        Ok(self.renderer.render_to_target(
            &self.gpu,
            scene,
            &self.calibration,
            pose,
            self.calibration.topology.blend_width(),
        ))
    }

    /// Enable 180-degree UV flip for the GPU zero-copy path.
    ///
    /// When set, the shader flips texture coordinates before sampling,
    /// equivalent to the CPU path's buffer reversal for rotated video
    /// (e.g., DJI cameras with rotation=180 metadata).
    pub fn set_flip_180(&mut self, left: bool, right: bool) {
        self.renderer.set_flip_180(left, right);
    }

    pub fn set_full_range(&mut self, full_range: bool) {
        self.renderer.set_full_range(full_range);
    }

    /// Access the rendered RGBA texture for NV12 conversion.
    pub fn render_target(&self) -> &wgpu::Texture {
        self.renderer.render_target()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::planes::copy_plane_tight;

    /// Build a test plane where row `r` contains byte value `r` for the first
    /// `width` bytes, followed by `0xFF` padding up to `stride`.
    fn padded_plane(width: u32, height: u32, stride: u32) -> Vec<u8> {
        let mut buf = vec![0xFF; (stride * height) as usize];
        for r in 0..height {
            for c in 0..width {
                buf[(r * stride + c) as usize] = r as u8;
            }
        }
        buf
    }

    #[test]
    fn copy_into_strips_row_padding() {
        // 4-pixel wide plane padded to 8-byte rows (typical OBS alignment).
        let y_data = padded_plane(4, 3, 8);
        let u_data = padded_plane(2, 2, 4);
        let v_data = padded_plane(2, 2, 4);
        let strided = StridedYuvPlanes {
            y: FramePlaneView {
                data: &y_data,
                stride: 8,
                width: 4,
                height: 3,
            },
            u: FramePlaneView {
                data: &u_data,
                stride: 4,
                width: 2,
                height: 2,
            },
            v: FramePlaneView {
                data: &v_data,
                stride: 4,
                width: 2,
                height: 2,
            },
        };

        let mut buffer = Vec::new();
        let tight = strided.copy_into(&mut buffer);

        assert_eq!(tight.y.len(), 12);
        assert_eq!(tight.u.len(), 4);
        assert_eq!(tight.v.len(), 4);
        // Row 0 should be [0,0,0,0], row 1 [1,1,1,1], etc - no 0xFF padding.
        assert_eq!(tight.y, &[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
        assert_eq!(tight.u, &[0, 0, 1, 1]);
        assert_eq!(tight.v, &[0, 0, 1, 1]);
    }

    #[test]
    fn copy_into_fast_path_when_tight() {
        // stride == width means no padding - fast path takes a single memcpy.
        let y_data: Vec<u8> = (0..12).collect();
        let u_data: Vec<u8> = (0..4).collect();
        let v_data: Vec<u8> = (4..8).collect();
        let strided = StridedYuvPlanes {
            y: FramePlaneView {
                data: &y_data,
                stride: 4,
                width: 4,
                height: 3,
            },
            u: FramePlaneView {
                data: &u_data,
                stride: 2,
                width: 2,
                height: 2,
            },
            v: FramePlaneView {
                data: &v_data,
                stride: 2,
                width: 2,
                height: 2,
            },
        };
        let mut buffer = Vec::new();
        let tight = strided.copy_into(&mut buffer);
        assert_eq!(tight.y, y_data.as_slice());
        assert_eq!(tight.u, u_data.as_slice());
        assert_eq!(tight.v, v_data.as_slice());
    }

    #[test]
    fn copy_into_reuses_buffer_without_realloc() {
        let plane = padded_plane(4, 3, 8);
        let strided = StridedYuvPlanes {
            y: FramePlaneView {
                data: &plane,
                stride: 8,
                width: 4,
                height: 3,
            },
            u: FramePlaneView {
                data: &plane,
                stride: 8,
                width: 2,
                height: 2,
            },
            v: FramePlaneView {
                data: &plane,
                stride: 8,
                width: 2,
                height: 2,
            },
        };

        let mut buffer = Vec::with_capacity(64);
        let cap_before = buffer.capacity();
        let _tight = strided.copy_into(&mut buffer);
        // 12 + 4 + 4 = 20 bytes needed, 64 capacity, no realloc.
        assert_eq!(buffer.capacity(), cap_before);

        // Second call with same dims: still no realloc.
        let _tight2 = strided.copy_into(&mut buffer);
        assert_eq!(buffer.capacity(), cap_before);
    }

    // ── B-24 regression: copy_plane_tight must not panic on malformed input

    #[test]
    fn copy_plane_tight_handles_stride_less_than_width() {
        // Pathological: caller declares width=8 but stride=4.
        // Before B-24 this would overlap rows and panic on slice
        // index. Now it zero-fills and logs.
        let data = vec![0xAA_u8; 16]; // 4 rows * 4 stride
        let src = FramePlaneView {
            data: &data,
            stride: 4,
            width: 8,
            height: 4,
        };
        let mut dst = vec![0xFF_u8; 32]; // 8*4
        copy_plane_tight(&src, &mut dst);
        assert!(
            dst.iter().all(|&b| b == 0),
            "zero-fill expected on stride<width"
        );
    }

    #[test]
    fn copy_plane_tight_handles_short_source_buffer() {
        let data = vec![0x77_u8; 4]; // Way too small for 8*4 claim.
        let src = FramePlaneView {
            data: &data,
            stride: 8,
            width: 8,
            height: 4,
        };
        let mut dst = vec![0xFF_u8; 32];
        copy_plane_tight(&src, &mut dst);
        assert!(dst.iter().all(|&b| b == 0));
    }

    #[test]
    fn copy_plane_tight_handles_dst_size_mismatch() {
        let data = vec![0xAB_u8; 32];
        let src = FramePlaneView {
            data: &data,
            stride: 8,
            width: 8,
            height: 4,
        };
        let mut dst = vec![0xFF_u8; 16]; // half of what's claimed
        copy_plane_tight(&src, &mut dst);
        assert!(dst.iter().all(|&b| b == 0));
    }

    #[test]
    fn copy_plane_tight_still_fast_path_when_tight() {
        let data: Vec<u8> = (0..32).collect();
        let src = FramePlaneView {
            data: &data,
            stride: 8,
            width: 8,
            height: 4,
        };
        let mut dst = vec![0; 32];
        copy_plane_tight(&src, &mut dst);
        assert_eq!(dst.as_slice(), data.as_slice());
    }
}
