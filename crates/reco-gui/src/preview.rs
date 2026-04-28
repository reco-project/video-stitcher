//! Zero-copy GPU preview bridge: reco-core StitchRenderer -> Slint Image.
//!
//! Shares the wgpu device and queue with Slint via its `unstable-wgpu-28`
//! feature. Renders directly into a `wgpu::Texture` allocated on Slint's
//! device; the texture is then wrapped in a `slint::Image` and handed to
//! the UI — no CPU readback, no staging buffers, no readback latency.
//!
//! ## Wiring
//!
//! 1. `main()` calls `slint::BackendSelector::require_wgpu_28()` before
//!    creating the UI so Slint initializes its renderer on wgpu 28.
//! 2. A `set_rendering_notifier` callback fires with
//!    `GraphicsAPI::WGPU28 { device, queue, .. }` on `RenderingSetup`.
//! 3. Those handles are passed to `PreviewBridge::new`, which builds a
//!    `GpuContext::from_device_queue` and a `StitchRenderer`.
//! 4. Each frame allocates a fresh `wgpu::Texture` on the shared device
//!    with `RENDER_ATTACHMENT | TEXTURE_BINDING`, renders into it, and
//!    moves it into `slint::Image::try_from`.
//!
//! ## Why a fresh texture per frame?
//!
//! `slint::Image::try_from(wgpu::Texture)` takes ownership: Slint holds
//! the texture for as long as the `Image` is referenced by any UI
//! property. Reusing a texture would mean the old `Image` (still
//! displayed until the UI swaps the property) and the new render target
//! alias the same storage. Slint's texture pool / compositor makes
//! per-frame allocation inexpensive; the driver reuses VRAM slabs.

use reco_core::calibration::{CameraParams, MatchCalibration};
use reco_core::gpu::GpuContext;
use reco_core::lens_preview::LensPreviewRenderer;
use reco_core::pipeline::{PipelineError, YuvPlanes};
use reco_core::stitch_renderer::StitchRenderer;
use reco_core::viewport::ViewportConfig;
use reco_core::wgpu;

/// Bridges reco-core GPU rendering to Slint via a shared wgpu device.
///
/// Each `render_frame` call produces a fresh `slint::Image` backed by a
/// GPU texture on Slint's own device - the UI displays it with zero
/// copies.
pub struct PreviewBridge {
    renderer: StitchRenderer,
    viewport_width: u32,
    viewport_height: u32,
    texture_format: wgpu::TextureFormat,
    lens_preview: Option<LensPreviewRenderer>,
    input_width: u32,
    input_height: u32,
}

impl PreviewBridge {
    /// Create a new preview bridge sharing Slint's wgpu 28 device.
    ///
    /// The `device`, `queue`, and `adapter_info` must come from Slint's
    /// `GraphicsAPI::WGPU28` rendering notifier — using anything else
    /// will not produce a zero-copy path.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
        calibration: MatchCalibration,
        input_width: u32,
        input_height: u32,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Result<Self, PipelineError> {
        let gpu = GpuContext::from_device_queue(device, queue, adapter_info);

        log::info!(
            "PreviewBridge (zero-copy): GPU={} ({})",
            gpu.gpu_name(),
            gpu.backend_name(),
        );

        let viewport = ViewportConfig {
            width: viewport_width,
            height: viewport_height,
            fov_degrees: 75.0,
            blend_width: 0.05,
            rig_tilt: calibration.rig_tilt as f32,
            rig_roll: calibration.rig_roll as f32,
            ..ViewportConfig::default()
        };

        // Slint expects textures in a format it can sample. Rgba8Unorm is
        // the safe common denominator across backends (Vulkan/Metal/DX12).
        let texture_format = wgpu::TextureFormat::Rgba8Unorm;

        let renderer = StitchRenderer::new(
            calibration,
            gpu,
            viewport,
            input_width,
            input_height,
            texture_format,
            reco_core::renderer::InputFormat::Yuv420p,
        )?;

        Ok(Self {
            renderer,
            viewport_width,
            viewport_height,
            texture_format,
            lens_preview: None,
            input_width,
            input_height,
        })
    }

    /// Render a YUV420P stereo pair, return a Slint image backed by a
    /// GPU texture on the shared device.
    pub fn render_frame(
        &self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<slint::Image, PipelineError> {
        let device = self.renderer.gpu().device();

        // Allocate a fresh texture on the shared device. RENDER_ATTACHMENT
        // lets reco-core write into it; TEXTURE_BINDING lets Slint sample
        // it in the compositor.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("reco-gui preview frame"),
            size: wgpu::Extent3d {
                width: self.viewport_width,
                height: self.viewport_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.texture_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Render into Slint's own device — commands submit on the shared
        // queue, no copies, no synchronization round-trip.
        self.renderer.render_yuv(left, right, yaw, pitch, &view)?;

        // Hand the texture to Slint. ownership transfers; Slint releases
        // it when the Image is no longer referenced by any UI property.
        slint::Image::try_from(texture).map_err(|_| PipelineError::InvalidConfig {
            reason: "slint::Image::try_from(wgpu::Texture) failed".into(),
        })
    }

    /// Access the underlying renderer for viewport adjustments.
    #[allow(dead_code)]
    pub fn renderer(&self) -> &StitchRenderer {
        &self.renderer
    }

    /// Mutable access for resize, FOV, calibration updates.
    #[allow(dead_code)]
    pub fn renderer_mut(&mut self) -> &mut StitchRenderer {
        &mut self.renderer
    }

    /// Current viewport dimensions.
    pub fn viewport_size(&self) -> (u32, u32) {
        (self.viewport_width, self.viewport_height)
    }

    /// Resize the render target. Updates both the pipeline viewport and
    /// the texture dimensions used for per-frame allocation.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.viewport_width = width;
        self.viewport_height = height;
        self.renderer.pipeline_mut().resize(width, height);
    }

    /// Render a single camera through orthographic projection with
    /// optional lens correction. Returns a Slint image for display.
    ///
    /// `correction_amount`: 0.0 = raw input, 1.0 = full KB4 correction.
    pub fn render_lens_preview(
        &mut self,
        planes: &YuvPlanes<'_>,
        params: &CameraParams,
        correction_amount: f32,
    ) -> Result<slint::Image, PipelineError> {
        let lp = self.lens_preview.get_or_insert_with(|| {
            let aspect = self.input_width as f32 / self.input_height as f32;
            LensPreviewRenderer::new(
                self.renderer.gpu(),
                self.input_width,
                self.input_height,
                aspect,
                self.texture_format,
            )
        });

        let texture = lp.render_yuv(self.renderer.gpu(), planes, params, correction_amount);

        slint::Image::try_from(texture).map_err(|_| PipelineError::InvalidConfig {
            reason: "slint::Image::try_from(wgpu::Texture) failed".into(),
        })
    }
}
