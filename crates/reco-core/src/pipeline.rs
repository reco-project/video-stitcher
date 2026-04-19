//! Stitch pipeline orchestration.
//!
//! The [`StitchPipeline`] coordinates all stages: GPU setup, frame ingestion,
//! rendering, viewport cropping, and output encoding. It is the primary
//! entry point for consumers of `reco-core`.
//!
//! ## Usage
//!
//! Most consumers should use [`StitchSession`](crate::session::StitchSession)
//! instead of `StitchPipeline` directly. The pipeline is exposed for advanced
//! use cases like preview windows that need direct surface rendering.
//!
//! ```rust,no_run,compile_fail
//! use reco_core::pipeline::StitchPipeline;
//! use reco_core::gpu::GpuContext;
//!
//! let gpu = pollster::block_on(GpuContext::new())?;
//! let pipeline = StitchPipeline::with_gpu(
//!     gpu, calibration, viewport, 1920, 1080,
//!     wgpu::TextureFormat::Rgba8UnormSrgb,
//!     reco_core::renderer::InputFormat::Yuv420p,
//! )?;
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

/// A single stride-aware plane view.
///
/// External frameworks (OBS `obs_source_frame`, V4L2, GStreamer buffers)
/// typically expose planes as a pointer plus a row stride that may exceed
/// the plane width (padding). [`YuvPlanes`] assumes tight packing, so a
/// consumer that receives padded data has to repack before calling
/// [`StitchPipeline::render_to_target`]. [`StridedYuvPlanes::copy_into`]
/// does the repack into a caller-owned buffer without reallocating on
/// every frame.
pub struct FramePlaneView<'a> {
    /// Plane pixel bytes. Must have length at least `stride * height`.
    pub data: &'a [u8],
    /// Bytes per row, including any trailing padding.
    pub stride: u32,
    /// Plane width in pixels (bytes per row of usable data).
    pub width: u32,
    /// Plane height in rows.
    pub height: u32,
}

/// Stride-aware YUV420P planes, suitable for hardware-decoded frames
/// or framework callbacks that expose raw pointers + strides.
///
/// Use [`Self::copy_into`] to produce a tightly packed [`YuvPlanes`]
/// view ready for [`StitchPipeline::render_to_target`].
pub struct StridedYuvPlanes<'a> {
    /// Y (luma) plane, full resolution.
    pub y: FramePlaneView<'a>,
    /// U (Cb) plane, half resolution per dimension.
    pub u: FramePlaneView<'a>,
    /// V (Cr) plane, half resolution per dimension.
    pub v: FramePlaneView<'a>,
}

impl StridedYuvPlanes<'_> {
    /// Repack the strided planes into a caller-owned tight buffer and
    /// return a [`YuvPlanes`] view into it.
    ///
    /// The `buffer` is resized to `y_len + u_len + v_len` bytes. Cache
    /// the buffer across frames to avoid per-frame allocation.
    pub fn copy_into<'b>(&self, buffer: &'b mut Vec<u8>) -> YuvPlanes<'b> {
        let y_len = (self.y.width as usize) * (self.y.height as usize);
        let u_len = (self.u.width as usize) * (self.u.height as usize);
        let v_len = (self.v.width as usize) * (self.v.height as usize);
        buffer.resize(y_len + u_len + v_len, 0);
        {
            let (y_dst, rest) = buffer.split_at_mut(y_len);
            let (u_dst, v_dst) = rest.split_at_mut(u_len);
            copy_plane_tight(&self.y, y_dst);
            copy_plane_tight(&self.u, u_dst);
            copy_plane_tight(&self.v, v_dst);
        }
        let (y, rest) = buffer.split_at(y_len);
        let (u, v) = rest.split_at(u_len);
        YuvPlanes { y, u, v }
    }
}

/// Copy `src` rows into `dst`, skipping stride padding.
///
/// `dst` must be exactly `src.width * src.height` bytes long.
/// Rows where `stride == width` are copied in a single `copy_from_slice`.
///
/// B-24 (deep-review-2026-04-18): the loop below slices
/// `src.data[row*stride..row*stride + width]`, which panics if
/// `stride < width` (overlaps next row) or if the source data is
/// shorter than `stride*height` (out of bounds). Previously these
/// preconditions were only `debug_assert!`, so release builds
/// produced a panic deep in the render path instead of a safe
/// fallback. Now the preconditions are runtime-checked and a
/// malformed plane falls back to zero-filling the destination plus a
/// warn log so the pipeline continues with a black plane rather than
/// aborting.
fn copy_plane_tight(src: &FramePlaneView<'_>, dst: &mut [u8]) {
    let width = src.width as usize;
    let height = src.height as usize;
    let stride = src.stride as usize;

    if dst.len() != width.saturating_mul(height) {
        log::warn!(
            "copy_plane_tight: dst {} bytes != width*height {} bytes; zero-filling",
            dst.len(),
            width.saturating_mul(height),
        );
        dst.fill(0);
        return;
    }
    if stride < width {
        log::warn!(
            "copy_plane_tight: stride {stride} < width {width}; malformed FramePlaneView, \
             zero-filling plane",
        );
        dst.fill(0);
        return;
    }
    if src.data.len() < stride.saturating_mul(height) {
        log::warn!(
            "copy_plane_tight: source buffer {} bytes < stride*height {} bytes; \
             zero-filling plane",
            src.data.len(),
            stride.saturating_mul(height),
        );
        dst.fill(0);
        return;
    }

    if stride == width {
        // Fast path: source is already tight; one memcpy.
        dst.copy_from_slice(&src.data[..width * height]);
        return;
    }

    for row in 0..height {
        let src_start = row * stride;
        let dst_start = row * width;
        dst[dst_start..dst_start + width].copy_from_slice(&src.data[src_start..src_start + width]);
    }
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

/// Borrowed packed BGRA/RGBA plane for pipeline input.
///
/// Tightly packed, 4 bytes per pixel in (R, G, B, A) byte order - the
/// shader samples `rgba.rgb` directly, so callers with BGRA data need
/// to swizzle bytes at upload time (see
/// [`BgraPlanes::from_bgra_swizzle_into`]). Used for OBS Browser
/// Source, screen capture, WebRTC ingest, and any other consumer whose
/// source frames are already sRGB-domain.
pub struct BgraPlanes<'a> {
    /// Packed RGBA bytes, length `width * height * 4`.
    pub rgba: &'a [u8],
}

impl<'a> BgraPlanes<'a> {
    /// Wrap an already-RGBA byte slice without copying.
    ///
    /// Length must be exactly `width * height * 4`; callers are expected
    /// to match the dimensions the pipeline was constructed with.
    pub fn from_rgba(rgba: &'a [u8]) -> Self {
        Self { rgba }
    }

    /// Swizzle a BGRA-ordered source into a caller-owned `Vec<u8>` and
    /// return a [`BgraPlanes`] borrowing from it.
    ///
    /// OBS and most desktop compositors deliver BGRA; the shader expects
    /// RGBA, so this helper performs the cheap byte-reorder once per
    /// frame. Cache the buffer across frames to avoid per-frame
    /// allocation (the helper `resize`s in place).
    pub fn from_bgra_swizzle_into(bgra: &[u8], buffer: &'a mut Vec<u8>) -> Self {
        buffer.resize(bgra.len(), 0);
        for (src, dst) in bgra.chunks_exact(4).zip(buffer.chunks_exact_mut(4)) {
            dst[0] = src[2]; // R <- B
            dst[1] = src[1]; // G
            dst[2] = src[0]; // B <- R
            dst[3] = src[3]; // A
        }
        Self { rgba: buffer }
    }
}

/// Errors from the stitch pipeline. `Clone + Send + Sync` so consumers
/// posting results to worker threads can carry the typed error.
#[derive(Debug, Clone, Error)]
pub enum PipelineError {
    /// GPU initialization failed.
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    /// Render error.
    #[error("render error: {0}")]
    Render(#[from] RenderError),

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
/// Owns the GPU context, scene geometry, and renderer. Consumers provide
/// YUV420P or NV12 frames and receive stitched RGBA output via
/// [`Self::render_to_target`] or [`Self::render_to_target_nv12`].
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
    /// Input frame dimensions.
    input_width: u32,
    input_height: u32,
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
        // Validate inputs before GPU resource creation.
        if let Err(e) = viewport.validate() {
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
        let aspect = calibration.left.width as f32 / calibration.left.height as f32;
        let scene = SceneGeometry::from_layout_with_aspect(&calibration.layout, aspect);
        let renderer = Renderer::new(
            &gpu,
            viewport.width,
            viewport.height,
            input_width,
            input_height,
            output_format,
            input_format,
            &scene,
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
            input_width,
            input_height,
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

    /// The calibration data this pipeline was created with.
    pub fn calibration(&self) -> &MatchCalibration {
        &self.calibration
    }

    /// The current output viewport configuration.
    pub fn viewport(&self) -> &ViewportConfig {
        &self.viewport
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
    #[allow(
        dead_code,
        reason = "consumed by StitchCore's GPU stacked-replay wiring in the next commit of the M7 sprint"
    )]
    pub(crate) fn input_format(&self) -> crate::renderer::InputFormat {
        self.renderer.input_format()
    }

    /// Left-side source plane views (Y/U/V texture views). Used by
    /// the stacked-video GPU packer to read the same uploaded
    /// source data the stitch shader samples; the pack runs in
    /// parallel with the panorama render into its own atlas buffer.
    /// For NV12 inputs the `U` view is the interleaved UV texture
    /// and the `V` view is a 1×1 dummy.
    #[allow(
        dead_code,
        reason = "consumed by StitchCore's GPU stacked-replay wiring in the next commit of the M7 sprint"
    )]
    pub(crate) fn left_plane_views(
        &self,
    ) -> (wgpu::TextureView, wgpu::TextureView, wgpu::TextureView) {
        self.renderer.left_plane_views()
    }

    /// Right-side counterpart to [`Self::left_plane_views`].
    #[allow(
        dead_code,
        reason = "consumed by StitchCore's GPU stacked-replay wiring in the next commit of the M7 sprint"
    )]
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
    /// [`RgbaReadback`](crate::rgba_readback::RgbaReadback)) should
    /// recreate them when the returned size differs from the previous.
    pub fn resize(&mut self, width: u32, height: u32) -> Option<(u32, u32)> {
        if width == 0 || height == 0 {
            log::warn!("resize({width}, {height}) ignored: dimensions must be non-zero");
            return None;
        }
        self.viewport.width = width;
        self.viewport.height = height;
        Some((width, height))
    }

    /// Set the vertical field of view in degrees.
    ///
    /// Values are clamped to `[1.0, 179.0]` to prevent degenerate
    /// projection matrices (0 or 180 would produce NaN/Inf).
    pub fn set_fov(&mut self, fov_degrees: f32) {
        self.viewport.fov_degrees = fov_degrees.clamp(1.0, 179.0);
    }

    /// Get the current field of view in degrees.
    pub fn fov(&self) -> f32 {
        self.viewport.fov_degrees
    }

    /// Update calibration parameters. Recomputes [`SceneGeometry`] from the
    /// new layout. Takes effect on the next render call (uniforms are rebuilt
    /// each frame from the stored calibration and scene).
    ///
    /// No GPU pipeline recreation needed - only the uniform data changes.
    pub fn update_calibration(&mut self, calibration: MatchCalibration) {
        let aspect = calibration.left.width as f32 / calibration.left.height as f32;
        self.scene = SceneGeometry::from_layout_with_aspect(&calibration.layout, aspect);
        self.calibration = calibration;
        log::debug!("Pipeline calibration updated");
    }

    /// Update only the plane layout (convenience for slider adjustments).
    ///
    /// Equivalent to cloning the current calibration, replacing its layout,
    /// and calling [`update_calibration`](Self::update_calibration).
    pub fn update_layout(&mut self, layout: crate::calibration::PlaneLayout) {
        let mut cal = self.calibration.clone();
        cal.layout = layout;
        self.update_calibration(cal);
    }

    /// Update per-camera intrinsics (focal, principal point, distortion)
    /// for one or both cameras without touching the plane layout or rig
    /// orientation.
    ///
    /// Intended for interactive lens tweaking in a GUI: each `CameraParams`
    /// change is written into the shader's per-frame uniform buffer, so the
    /// next render call reflects the new values. No GPU pipeline or scene
    /// recreation is needed - cheap enough (~microseconds) to call on
    /// every slider drag.
    ///
    /// `left`/`right` are `None` to leave that side untouched. If both are
    /// `None` this is a no-op. Passing `Some` for a side replaces that
    /// side's `CameraParams` on the stored calibration; the next render
    /// picks it up automatically.
    ///
    /// Does not recompute `SceneGeometry` because the plane layout is
    /// unchanged; only the camera intrinsics (which live on the stored
    /// calibration and are re-read each frame) need updating.
    pub fn update_camera_params(
        &mut self,
        left: Option<crate::calibration::CameraParams>,
        right: Option<crate::calibration::CameraParams>,
    ) {
        if left.is_none() && right.is_none() {
            return;
        }
        if let Some(l) = left {
            self.calibration.left = l;
        }
        if let Some(r) = right {
            self.calibration.right = r;
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
        let left_bg = self
            .renderer
            .create_texture_bind_group(left_y, left_uv, "metal_left");
        let right_bg = self
            .renderer
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
            position: ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            },
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

    /// Render NV12 frames directly to a texture view (for window display).
    ///
    /// Like [`Self::render_to_view`] but accepts NV12 input (Y + interleaved
    /// UV) instead of YUV420P. Requires the pipeline to be initialized with
    /// `InputFormat::Nv12`.
    pub fn render_nv12_to_view(
        &self,
        left: &Nv12Planes<'_>,
        right: &Nv12Planes<'_>,
        yaw: f32,
        pitch: f32,
        target_view: &wgpu::TextureView,
    ) -> Result<(), PipelineError> {
        self.renderer.upload_left_nv12(&self.gpu, left.y, left.uv)?;
        self.renderer
            .upload_right_nv12(&self.gpu, right.y, right.uv)?;

        let viewport = ResolvedViewport {
            config: self.viewport.clone(),
            position: ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            },
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
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        self.renderer
            .upload_left_yuv(&self.gpu, left.y, left.u, left.v)?;
        self.renderer
            .upload_right_yuv(&self.gpu, right.y, right.u, right.v)?;

        let viewport = ResolvedViewport {
            config: self.viewport.clone(),
            position: ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            },
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
            position: ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            },
        };

        Ok(self.renderer.render_to_target(
            &self.gpu,
            &self.scene,
            &self.calibration,
            &viewport,
            self.viewport.blend_width,
        ))
    }

    /// Upload packed BGRA/RGBA frames and render to the internal target.
    ///
    /// Expects each plane as `width * height * 4` bytes in (R, G, B, A) byte
    /// order. Use [`BgraPlanes::from_bgra_swizzle_into`] when the source
    /// is BGRA. Requires the pipeline to be initialized with
    /// [`InputFormat::Bgra`](crate::renderer::InputFormat#variant.Bgra).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "render_to_target_bgra")
    )]
    pub fn render_to_target_bgra(
        &self,
        left: &BgraPlanes<'_>,
        right: &BgraPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        self.renderer.upload_left_bgra(&self.gpu, left.rgba)?;
        self.renderer.upload_right_bgra(&self.gpu, right.rgba)?;

        let viewport = ResolvedViewport {
            config: self.viewport.clone(),
            position: ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            },
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
    pub(crate) fn render_to_target_gpu(&self, yaw: f32, pitch: f32) -> wgpu::CommandBuffer {
        let viewport = ResolvedViewport {
            config: self.viewport.clone(),
            position: ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            },
        };

        self.renderer.render_to_target(
            &self.gpu,
            &self.scene,
            &self.calibration,
            &viewport,
            self.viewport.blend_width,
        )
    }

    /// Enable 180-degree UV flip for the GPU zero-copy path.
    ///
    /// When set, the shader flips texture coordinates before sampling,
    /// equivalent to the CPU path's buffer reversal for rotated video
    /// (e.g., DJI cameras with rotation=180 metadata).
    pub fn set_flip_180(&mut self, left: bool, right: bool) {
        self.renderer.set_flip_180(left, right);
    }

    /// Access the rendered RGBA texture for NV12 conversion.
    pub fn render_target(&self) -> &wgpu::Texture {
        self.renderer.render_target()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
