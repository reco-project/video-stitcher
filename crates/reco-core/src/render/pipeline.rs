//! Stitch pipeline orchestration.
//!
//! `StitchPipeline` coordinates all stages: GPU setup, frame ingestion,
//! rendering, and viewport cropping. It is crate-internal - the
//! [`GpuExecutor`](crate::stitch::GpuExecutor) is its sole owner, and
//! consumers reach rendering through the engine or a
//! [`StitchSession`](crate::session::StitchSession). This module also
//! re-exports the public frame-plane value types
//! ([`Yuv420pPlanes`], [`Nv12Planes`], [`RgbaPlanes`], ...) and
//! [`PipelineError`], the pipeline half of the engine's error type.

use super::renderer::{InputFormat, RenderError, Renderer};
use super::viewport::ViewportSize;
use crate::calibration::Calibration;
use crate::geometry::RenderPose;
use crate::gpu::{GpuContext, GpuError};
use crate::projection::l_shape::PlaneScene;
use crate::source::{Nv12Data, Yuv420pData};
use std::sync::{Arc, Weak};

use thiserror::Error;

#[cfg(test)]
#[path = "pipeline/upload_tests.rs"]
mod upload_tests;

pub use super::planes::{
    FramePlaneView, Nv12Planes, RgbaPlanes, StridedYuv420pPlanes, Yuv420pPlanes,
};

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

    /// A stereo-plane entry point was used with a non-planar projection.
    #[error("this entry point requires a stereo plane projection; use the frame-set render path")]
    NoPlaneScene,

    /// Wrong FrameSet variant for this render method.
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
    /// Cached L-shape plane placement; fullscreen projections do not use it.
    pub(crate) scene: Option<PlaneScene>,
    /// Calibration data (camera intrinsics + layout).
    pub(crate) calibration: Calibration,
    /// Output viewport configuration.
    pub(crate) viewport_size: ViewportSize,
    /// GPU renderer (textures, pipelines, bind groups).
    renderer: Renderer,
    /// Input frame dimensions, in projection order.
    source_sizes: Vec<(u32, u32)>,
    /// Weak identity keeps allocations distinct without retaining decoded pixels. `Arc::make_mut` dissociates weak references, so edited images upload again.
    uploaded: Vec<Option<UploadedImage>>,
}

#[derive(Clone)]
enum UploadedImage {
    Yuv420p(Weak<Yuv420pData>),
    Nv12(Weak<Nv12Data>),
}

/// Pre-built bind groups for GPU-resident zero-copy sources.
///
/// Created by [`StitchPipeline::configure_gpu_source`]. Each slot
/// corresponds to a double-buffer index used by the decode thread.
#[cfg(target_os = "linux")]
pub(crate) struct GpuSourceBindGroups {
    cameras: Vec<[wgpu::BindGroup; 2]>,
}

impl StitchPipeline {
    /// The L-shape's plane placement for the GPU uniforms, when the
    /// projection has planes, including each calibrated image's proportions.
    fn derive_plane_scene(calibration: &Calibration) -> Option<PlaneScene> {
        calibration.projection.l_shape().map(|t| {
            t.scene(
                &calibration.framing,
                [
                    calibration.lenses[0].aspect_ratio(),
                    calibration.lenses[1].aspect_ratio(),
                ],
            )
        })
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
        source_sizes: Vec<(u32, u32)>,
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
        let limit = gpu.device.limits().max_texture_dimension_2d;
        if viewport_size.width > limit || viewport_size.height > limit {
            return Err(PipelineError::InvalidConfig {
                reason: format!("viewport exceeds the device texture dimension limit {limit}"),
            });
        }
        crate::source::validate_camera_sizes(
            &source_sizes,
            calibration.lenses.len(),
            input_format != InputFormat::Rgba,
        )
        .map_err(|reason| PipelineError::InvalidConfig { reason })?;

        for (camera, &(width, height)) in source_sizes.iter().enumerate() {
            if width > limit || height > limit {
                return Err(PipelineError::InvalidConfig {
                    reason: format!(
                        "camera {camera} dimensions {width}x{height} exceed the device texture dimension limit {limit}"
                    ),
                });
            }
        }

        let output_format = output_format.into();
        let scene = Self::derive_plane_scene(&calibration);
        let renderer = Renderer::new(
            &gpu,
            program,
            viewport_size.width,
            viewport_size.height,
            &source_sizes,
            output_format,
            input_format,
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
            uploaded: vec![None; source_sizes.len()],
            source_sizes,
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

    /// The projection in effect: a borrow of the document's projection.
    pub(crate) fn geometry(&self) -> &dyn crate::projection::ProjectionGeometry {
        self.calibration.projection.geometry()
    }

    /// The current output viewport configuration.
    pub fn viewport_size(&self) -> &ViewportSize {
        &self.viewport_size
    }

    /// Input `(width, height)` for each camera, in projection order.
    pub fn source_sizes(&self) -> &[(u32, u32)] {
        &self.source_sizes
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

    /// Update viewport geometry and invalidate the internal output texture.
    ///
    /// Shaders and camera resources remain valid. Output storage is recreated only when used; caller-owned preview targets need none. Rejects zero or device-exceeding dimensions without mutation. Owners of readback storage must resize it too.
    pub fn resize(&mut self, width: u32, height: u32) -> Option<(u32, u32)> {
        let limit = self.gpu.device.limits().max_texture_dimension_2d;
        if width == 0 || height == 0 || width > limit || height > limit {
            log::warn!("resize({width}, {height}) ignored: dimensions must be in 1..={limit}");
            return None;
        }
        self.renderer.resize(width, height);
        self.viewport_size.width = width;
        self.viewport_size.height = height;
        Some((width, height))
    }

    /// Install a validated calibration with unchanged projection kind and image dimensions.
    pub(crate) fn apply_calibration(&mut self, calibration: Calibration) {
        let scene = Self::derive_plane_scene(&calibration);
        self.scene = scene;
        self.calibration = calibration;
    }

    /// Set up bind groups for GPU-resident zero-copy input.
    ///
    /// Creates bind groups for the provided shared textures (Y + UV per slot
    /// per camera). Call once during setup, then pass the result to
    /// [`Self::render_gpu_frame`] each frame.
    #[cfg(target_os = "linux")]
    pub fn configure_gpu_source(
        &mut self,
        shared: &crate::interop::SharedTextureSet,
    ) -> GpuSourceBindGroups {
        GpuSourceBindGroups {
            cameras: shared
                .cameras
                .iter()
                .map(|camera| {
                    std::array::from_fn(|slot| {
                        self.renderer.create_texture_bind_group(
                            &camera.textures[slot * 2].texture,
                            &camera.textures[slot * 2 + 1].texture,
                            "decode_slot",
                        )
                    })
                })
                .collect(),
        }
    }

    /// Select each camera's shared decode slot and render.
    #[cfg(target_os = "linux")]
    pub fn render_gpu_frame(
        &mut self,
        bind_groups: &GpuSourceBindGroups,
        slots: &[u8],
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        self.check_inputs(slots.len(), InputFormat::Nv12)?;
        if bind_groups.cameras.len() != slots.len() || slots.iter().any(|&slot| slot > 1) {
            return Err(PipelineError::InvalidConfig {
                reason: "invalid shared decode slots".into(),
            });
        }
        for (camera, &slot) in slots.iter().enumerate() {
            self.renderer.set_camera_bind_group(
                camera,
                bind_groups.cameras[camera][usize::from(slot)].clone(),
            );
        }
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
    #[cfg(target_os = "linux")]
    pub fn render_with_bind_groups(
        &mut self,
        groups: &[wgpu::BindGroup],
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        self.check_inputs(groups.len(), InputFormat::Nv12)?;
        for (camera, group) in groups.iter().enumerate() {
            self.renderer.set_camera_bind_group(camera, group.clone());
        }
        self.render_to_target_gpu(pose)
    }

    /// Release imported textures before returning to host-backed input.
    pub(crate) fn clear_imported_bind_groups(&mut self) {
        self.renderer.clear_imported_bind_groups();
    }

    /// Render camera-ordered Y/UV views without uploading or retaining input bindings for later frames.
    pub fn render_imported_views(
        &mut self,
        cameras: &[(&wgpu::TextureView, &wgpu::TextureView)],
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        self.bind_imported_views(cameras)?;
        let result = self.render_to_target_gpu(pose);
        self.clear_imported_bind_groups();
        result
    }

    fn bind_imported_views(
        &mut self,
        cameras: &[(&wgpu::TextureView, &wgpu::TextureView)],
    ) -> Result<(), PipelineError> {
        self.check_inputs(cameras.len(), InputFormat::Nv12)?;
        // Validate every backing allocation before changing any camera binding.
        for (camera, (&(y, uv), &(width, height))) in
            cameras.iter().zip(&self.source_sizes).enumerate()
        {
            let (y, uv) = (y.texture(), uv.texture());
            use wgpu::TextureFormat as F;
            let uv_size = match (y.format(), uv.format()) {
                (F::R8Unorm, F::Rg8Unorm) | (F::R16Unorm, F::Rg16Unorm) => {
                    Some((width / 2, height / 2))
                }
                (F::NV12, F::NV12) | (F::P010, F::P010) if y == uv => Some((width, height)),
                _ => None,
            };
            if (y.width(), y.height()) != (width, height)
                || uv_size != Some((uv.width(), uv.height()))
                || y.format() != cameras[0].0.texture().format()
                || [y, uv].iter().any(|t| {
                    t.dimension() != wgpu::TextureDimension::D2
                        || t.depth_or_array_layers() != 1
                        || t.mip_level_count() != 1
                        || t.sample_count() != 1
                        || !t.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
                })
            {
                return Err(PipelineError::InvalidConfig {
                    reason: format!(
                        "camera {camera} requires matching {width}x{height} NV12/P010 plane views"
                    ),
                });
            }
        }
        for (camera, &(y, uv)) in cameras.iter().enumerate() {
            let group = self
                .renderer
                .create_bind_group_from_views(y, uv, "imported_camera");
            self.renderer.set_camera_bind_group(camera, group);
        }
        Ok(())
    }

    /// Process a CPU-resident frame set and return the render command buffer.
    ///
    /// Handles YUV420P vs NV12 format differences internally.
    /// For GPU-resident frames, use [`Self::render_gpu_frame`] instead.
    ///
    /// Validates camera count and pixel format before uploading.
    pub fn render_frame_set(
        &mut self,
        frames: &crate::source::FrameSet,
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        self.upload_frame_set(frames)?;
        self.render_to_target_gpu(pose)
    }

    fn upload_frame_set(
        &mut self,
        frames: &crate::source::FrameSet,
    ) -> Result<usize, PipelineError> {
        frames
            .validate_host_images(self.source_sizes())
            .map_err(|reason| PipelineError::InvalidConfig { reason })?;
        use crate::source::FrameSet;
        let mut uploads = 0;
        match frames {
            FrameSet::Yuv420p(cams) => {
                self.check_inputs(cams.len(), InputFormat::Yuv420p)?;
                self.renderer.clear_imported_bind_groups();
                for (camera, data) in cams.iter().enumerate() {
                    let identity = Arc::downgrade(data);
                    if matches!(&self.uploaded[camera], Some(UploadedImage::Yuv420p(old)) if old.ptr_eq(&identity))
                    {
                        continue;
                    }
                    self.uploaded[camera] = None;
                    self.renderer
                        .upload_yuv(&self.gpu, camera, &data.as_planes())?;
                    self.uploaded[camera] = Some(UploadedImage::Yuv420p(identity));
                    uploads += 1;
                }
                Ok(uploads)
            }
            FrameSet::Nv12(cams) => {
                self.check_inputs(cams.len(), InputFormat::Nv12)?;
                self.renderer.clear_imported_bind_groups();
                for (camera, data) in cams.iter().enumerate() {
                    let identity = Arc::downgrade(data);
                    if matches!(&self.uploaded[camera], Some(UploadedImage::Nv12(old)) if old.ptr_eq(&identity))
                    {
                        continue;
                    }
                    self.uploaded[camera] = None;
                    self.renderer
                        .upload_nv12(&self.gpu, camera, &data.as_planes())?;
                    self.uploaded[camera] = Some(UploadedImage::Nv12(identity));
                    uploads += 1;
                }
                Ok(uploads)
            }
            _ => Err(PipelineError::UnsupportedFrameVariant {
                reason: "resident frames require their platform render path",
            }),
        }
    }

    fn check_inputs(&self, count: usize, format: InputFormat) -> Result<(), PipelineError> {
        let expected = self.calibration.projection.camera_count();
        if count != expected || format != self.input_format() {
            return Err(PipelineError::InvalidConfig {
                reason: format!(
                    "expected {expected} {:?} cameras, got {count} {format:?}",
                    self.input_format()
                ),
            });
        }
        Ok(())
    }

    pub(crate) fn render_nv12(
        &mut self,
        cameras: &[Nv12Planes<'_>],
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        self.check_inputs(cameras.len(), InputFormat::Nv12)?;
        self.uploaded.fill(None);
        self.renderer.clear_imported_bind_groups();
        for (camera, data) in cameras.iter().enumerate() {
            self.renderer.upload_nv12(&self.gpu, camera, data)?;
        }
        self.render_to_target_gpu(pose)
    }

    /// Render a frame directly to a texture view (for window display).
    ///
    /// Unlike the encode path, this does NOT read back to CPU — the result
    /// stays on the GPU and is presented to the surface.
    pub fn render_to_view(
        &mut self,
        frames: &crate::source::FrameSet,
        pose: RenderPose,
        target_view: &wgpu::TextureView,
    ) -> Result<wgpu::SubmissionIndex, PipelineError> {
        self.validate_target(target_view)?;
        self.upload_frame_set(frames)?;
        Ok(self.render_bound_to_view(pose, target_view))
    }

    fn validate_target(&self, view: &wgpu::TextureView) -> Result<(), PipelineError> {
        let texture = view.texture();
        if texture.width() != self.viewport_size.width
            || texture.height() != self.viewport_size.height
            || texture.dimension() != wgpu::TextureDimension::D2
            || texture.depth_or_array_layers() != 1
            || texture.mip_level_count() != 1
            || texture.sample_count() != 1
            || !texture
                .usage()
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
            || crate::render::strip_srgb(texture.format())
                != crate::render::strip_srgb(self.renderer.output_format())
        {
            return Err(PipelineError::InvalidConfig {
                reason: "render target must match the configured viewport and output format".into(),
            });
        }
        Ok(())
    }

    fn render_bound_to_view(
        &self,
        pose: RenderPose,
        target_view: &wgpu::TextureView,
    ) -> wgpu::SubmissionIndex {
        self.renderer.render_to_view(
            &self.gpu,
            self.scene.as_ref(),
            &self.calibration,
            pose,
            self.viewport_size.aspect_ratio(),
            self.calibration.projection.blend_width(),
            target_view,
        )
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
        &mut self,
        left: &Yuv420pPlanes<'_>,
        right: &Yuv420pPlanes<'_>,
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        let Some(scene) = &self.scene else {
            return Err(PipelineError::NoPlaneScene);
        };
        self.uploaded.fill(None);
        self.renderer.clear_imported_bind_groups();
        self.renderer
            .upload_left_yuv(&self.gpu, left.y, left.u, left.v)?;
        self.renderer
            .upload_right_yuv(&self.gpu, right.y, right.u, right.v)?;

        Ok(self.renderer.render_to_target(
            &self.gpu,
            Some(scene),
            &self.calibration,
            pose,
            self.calibration.projection.blend_width(),
        ))
    }

    /// Upload packed RGBA frames and render to the internal target.
    ///
    /// Expects each plane as `width * height * 4` bytes in (R, G, B, A) byte
    /// order. Use [`RgbaPlanes::from_bgra_swizzle_into`] when the source
    /// is BGRA. Requires the pipeline to be initialized with
    /// [`InputFormat::Rgba`](crate::render::renderer::InputFormat#variant.Rgba).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "render_to_target_rgba")
    )]
    pub fn render_to_target_rgba(
        &mut self,
        left: &RgbaPlanes<'_>,
        right: &RgbaPlanes<'_>,
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        let Some(scene) = &self.scene else {
            return Err(PipelineError::NoPlaneScene);
        };
        self.uploaded.fill(None);
        self.renderer.clear_imported_bind_groups();
        self.renderer.upload_left_rgba(&self.gpu, left.rgba)?;
        self.renderer.upload_right_rgba(&self.gpu, right.rgba)?;

        Ok(self.renderer.render_to_target(
            &self.gpu,
            Some(scene),
            &self.calibration,
            pose,
            self.calibration.projection.blend_width(),
        ))
    }

    /// Render from GPU-resident RGBA textures (e.g. Bayer demosaic output).
    ///
    /// Copies source textures into the input planes, then renders the
    /// stitch to the internal target. Returns the complete command buffer.
    /// The caller submits the demosaic encoder first, then this one.
    /// Requires `InputFormat::Rgba`.
    pub fn render_from_gpu_rgba(
        &mut self,
        left_rgba: &wgpu::Texture,
        right_rgba: &wgpu::Texture,
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        if self.scene.is_none() {
            return Err(PipelineError::NoPlaneScene);
        }
        self.uploaded.fill(None);
        self.renderer.clear_imported_bind_groups();

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
    /// Returns the render `CommandBuffer` without submitting. Samples the current camera bindings, which must already contain the input pixels.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "render_to_target_gpu")
    )]
    pub fn render_to_target_gpu(
        &self,
        pose: RenderPose,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        Ok(self.renderer.render_to_target(
            &self.gpu,
            self.scene.as_ref(),
            &self.calibration,
            pose,
            self.calibration.projection.blend_width(),
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
        let strided = StridedYuv420pPlanes {
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
        let strided = StridedYuv420pPlanes {
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
        let strided = StridedYuv420pPlanes {
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
