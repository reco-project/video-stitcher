//! GPU preview bridge: reco-core StitchRenderer -> Slint Image.
//!
//! Uses a headless `GpuContext` with reco-core's wgpu 29 pipeline, renders
//! to an internal RGBA target, reads back via double-buffered staging, and
//! produces a `slint::Image` from the pixel data.
//!
//! ## Performance
//!
//! Double-buffered readback pipelines GPU work: while frame N is being
//! read back, frame N+1 is rendering. This overlaps GPU render+copy with
//! CPU readback, eliminating the blocking `device.poll(Wait)` on the
//! render path. The result is always one frame behind, but at 30fps
//! that's 33ms of latency - imperceptible for interactive preview.
//!
//! ## Why CPU readback?
//!
//! Slint 1.15 supports wgpu 27/28 via `unstable-wgpu-{27,28}` features,
//! but reco-core uses wgpu 29 (custom fork). Since wgpu types from
//! different major versions are incompatible Rust types, we cannot share
//! the GPU device between Slint and reco-core. Once Slint adds wgpu 29
//! support (tracked in slint-ui/slint#11378), this module should be
//! replaced with zero-copy texture import via `GpuContext::from_device_queue`.

use reco_core::calibration::MatchCalibration;
use reco_core::gpu::GpuContext;
use reco_core::pipeline::{PipelineError, YuvPlanes};
use reco_core::stitch_renderer::StitchRenderer;
use reco_core::viewport::ViewportConfig;
use reco_core::wgpu;

/// RGBA readback staging buffer for GPU -> CPU transfer.
struct StagingBuffer {
    buffer: wgpu::Buffer,
    padded_bytes_per_row: u32,
    unpadded_bytes_per_row: u32,
    width: u32,
    height: u32,
}

impl StagingBuffer {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let unpadded_bytes_per_row = 4 * width;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = padded_bytes_per_row as u64 * height as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reco-gui staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            padded_bytes_per_row,
            unpadded_bytes_per_row,
            width,
            height,
        }
    }

    /// Encode a copy from texture to this staging buffer.
    fn copy_from_texture(&self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Try to read back pixels. Returns true if data was available.
    /// Writes directly into the provided SharedPixelBuffer to avoid
    /// an intermediate allocation.
    fn try_read_into(
        &self,
        device: &wgpu::Device,
        dest: &mut slint::SharedPixelBuffer<slint::Rgba8Pixel>,
    ) -> bool {
        let slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });

        // Non-blocking poll first: the GPU work was submitted on the
        // previous frame, so it should be done by now.
        if device.poll(wgpu::PollType::Poll).is_err() {
            return false;
        }

        match rx.try_recv() {
            Ok(Ok(())) => {}
            Ok(Err(_)) => return false,
            Err(_) => {
                // Not ready yet. Fall back to blocking wait as safety net.
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("GPU poll failed");
                match rx.recv() {
                    Ok(Ok(())) => {}
                    _ => return false,
                }
            }
        }

        let mapped = slice.get_mapped_range();
        let out = dest.make_mut_bytes();
        for row in 0..self.height as usize {
            let src_start = row * self.padded_bytes_per_row as usize;
            let src_end = src_start + self.unpadded_bytes_per_row as usize;
            let dst_start = row * self.unpadded_bytes_per_row as usize;
            let dst_end = dst_start + self.unpadded_bytes_per_row as usize;
            out[dst_start..dst_end].copy_from_slice(&mapped[src_start..src_end]);
        }
        drop(mapped);
        self.buffer.unmap();
        true
    }
}

/// Bridges reco-core GPU rendering to Slint pixel buffers.
///
/// Uses double-buffered readback: while one staging buffer is being
/// read by the CPU, the other receives the next frame's GPU output.
pub struct PreviewBridge {
    renderer: StitchRenderer,
    staging: [StagingBuffer; 2],
    /// Which staging buffer to write to next.
    write_idx: usize,
    /// Whether the previous staging buffer has a pending readback.
    has_pending: bool,
    /// Reusable pixel buffer for building Slint images.
    pixel_buf: slint::SharedPixelBuffer<slint::Rgba8Pixel>,
    viewport_width: u32,
    viewport_height: u32,
}

impl PreviewBridge {
    /// Create a new preview bridge with headless GPU rendering.
    pub fn new(
        calibration: MatchCalibration,
        input_width: u32,
        input_height: u32,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Result<Self, PipelineError> {
        let gpu =
            pollster::block_on(GpuContext::new()).map_err(|e| PipelineError::InvalidConfig {
                reason: format!("GPU init failed: {e}"),
            })?;

        log::info!(
            "PreviewBridge: GPU={} ({})",
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
        };

        let surface_format = wgpu::TextureFormat::Rgba8Unorm;

        let renderer = StitchRenderer::new(
            calibration,
            gpu,
            viewport,
            input_width,
            input_height,
            surface_format,
        )?;

        let device = renderer.gpu().device();
        let staging = [
            StagingBuffer::new(device, viewport_width, viewport_height),
            StagingBuffer::new(device, viewport_width, viewport_height),
        ];

        let pixel_buf =
            slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(viewport_width, viewport_height);

        Ok(Self {
            renderer,
            staging,
            write_idx: 0,
            has_pending: false,
            pixel_buf,
            viewport_width,
            viewport_height,
        })
    }

    /// Render a YUV420P stereo pair and return the previous frame's image.
    ///
    /// Uses double-buffered readback: submits this frame's render+copy
    /// without waiting, then reads back the PREVIOUS frame's pixels.
    /// Returns `None` on the very first call (no previous frame yet).
    pub fn render_frame(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<Option<slint::Image>, PipelineError> {
        // 1. Submit this frame's render + copy (non-blocking).
        let render_cmd = self
            .renderer
            .pipeline()
            .render_to_target(left, right, yaw, pitch)?;

        let device = self.renderer.gpu().device();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reco-gui readback copy"),
        });
        self.staging[self.write_idx].copy_from_texture(&mut encoder, self.renderer.render_target());

        self.renderer
            .gpu()
            .queue()
            .submit([render_cmd, encoder.finish()]);

        // 2. Read back the PREVIOUS frame from the other buffer.
        let image = if self.has_pending {
            let read_idx = 1 - self.write_idx;
            if self.staging[read_idx].try_read_into(device, &mut self.pixel_buf) {
                Some(slint::Image::from_rgba8(self.pixel_buf.clone()))
            } else {
                None
            }
        } else {
            None
        };

        // 3. Swap buffers.
        self.write_idx = 1 - self.write_idx;
        self.has_pending = true;

        Ok(image)
    }

    /// Flush the last pending readback (blocking).
    /// Call once after the final render to get the last frame.
    #[allow(dead_code)]
    pub fn flush(&mut self) -> Option<slint::Image> {
        if !self.has_pending {
            return None;
        }
        let read_idx = 1 - self.write_idx;
        let device = self.renderer.gpu().device();
        if self.staging[read_idx].try_read_into(device, &mut self.pixel_buf) {
            self.has_pending = false;
            Some(slint::Image::from_rgba8(self.pixel_buf.clone()))
        } else {
            None
        }
    }

    /// Render a single frame with blocking readback (for seek/step).
    /// This is slower but returns the exact frame requested.
    pub fn render_frame_sync(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<slint::Image, PipelineError> {
        // Flush any pending readback first.
        self.has_pending = false;

        let render_cmd = self
            .renderer
            .pipeline()
            .render_to_target(left, right, yaw, pitch)?;

        let device = self.renderer.gpu().device();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reco-gui sync readback"),
        });
        self.staging[0].copy_from_texture(&mut encoder, self.renderer.render_target());

        self.renderer
            .gpu()
            .queue()
            .submit([render_cmd, encoder.finish()]);

        // Blocking readback for immediate result.
        self.staging[0].try_read_into(device, &mut self.pixel_buf);
        Ok(slint::Image::from_rgba8(self.pixel_buf.clone()))
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
    #[allow(dead_code)]
    pub fn viewport_size(&self) -> (u32, u32) {
        (self.viewport_width, self.viewport_height)
    }
}
