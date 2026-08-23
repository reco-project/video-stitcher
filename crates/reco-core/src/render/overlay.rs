//! Generic RGBA overlay composition for post-camera graphics.
//!
//! The compositor deliberately knows nothing about sports, HTML, or browser
//! engines. Consumers upload an [`OverlayFrame`] and the compositor blends it
//! over the final camera image after stitching and viewport/autocam rendering.

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::gpu::GpuContext;

const STRAIGHT_ALPHA_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

/// A complete straight-alpha RGBA8 overlay surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverlayFrame {
    /// Pixel width of the reference canvas.
    pub width: u32,
    /// Pixel height of the reference canvas.
    pub height: u32,
    /// Tightly packed RGBA8 pixels in row-major order.
    pub rgba: Vec<u8>,
}

impl OverlayFrame {
    /// Validate dimensions and byte length.
    pub fn validate(&self) -> Result<(), OverlayError> {
        if self.width == 0 || self.height == 0 {
            return Err(OverlayError::InvalidFrame(
                "overlay dimensions must be non-zero".into(),
            ));
        }
        let expected = self
            .width
            .checked_mul(self.height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| OverlayError::InvalidFrame("overlay dimensions overflow".into()))?
            as usize;
        if self.rgba.len() != expected {
            return Err(OverlayError::InvalidFrame(format!(
                "expected {expected} RGBA bytes for {}x{}, got {}",
                self.width,
                self.height,
                self.rgba.len()
            )));
        }
        Ok(())
    }
}

/// Non-blocking source of independently rendered overlay frames.
///
/// Implementations may run a browser, network client, or another renderer on
/// their own thread. [`try_frame`](Self::try_frame) must never wait for a new
/// frame: video processing reuses the previous GPU texture when it returns
/// `Ok(None)`.
pub trait OverlayFrameSource: Send {
    /// Return the newest available frame, or `None` when nothing changed.
    fn try_frame(&mut self) -> Result<Option<OverlayFrame>, String>;
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OverlayParams {
    reference_size: [f32; 2],
    output_size: [f32; 2],
}

/// Cached GPU resources for blending one RGBA surface over render targets.
pub(crate) struct RgbaOverlayCompositor {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    reference_size: (u32, u32),
    output_size: (u32, u32),
}

impl RgbaOverlayCompositor {
    pub(crate) fn new(
        gpu: &GpuContext,
        output_format: wgpu::TextureFormat,
        output_size: (u32, u32),
        frame: &OverlayFrame,
    ) -> Result<Self, OverlayError> {
        frame.validate()?;
        let device = gpu.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reco rgba overlay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/rgba_overlay.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reco rgba overlay bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<OverlayParams>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reco rgba overlay pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("reco rgba overlay pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(STRAIGHT_ALPHA_BLEND),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("reco rgba overlay sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reco rgba overlay params"),
            contents: bytemuck::bytes_of(&OverlayParams {
                reference_size: [frame.width as f32, frame.height as f32],
                output_size: [output_size.0 as f32, output_size.1 as f32],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let texture = Self::create_texture(device, frame.width, frame.height);
        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &texture,
            &sampler,
            &params_buffer,
        );
        let mut compositor = Self {
            pipeline,
            bind_group_layout,
            sampler,
            texture,
            bind_group,
            params_buffer,
            reference_size: (frame.width, frame.height),
            output_size,
        };
        compositor.upload(gpu, frame)?;
        Ok(compositor)
    }

    fn create_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("reco cached rgba overlay"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Browser bytes are already in the same display-encoded space as
            // the stitcher's Rgba8Unorm output. Do not apply an sRGB decode.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture: &wgpu::Texture,
        sampler: &wgpu::Sampler,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reco rgba overlay bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    pub(crate) fn upload(
        &mut self,
        gpu: &GpuContext,
        frame: &OverlayFrame,
    ) -> Result<(), OverlayError> {
        frame.validate()?;
        if self.reference_size != (frame.width, frame.height) {
            self.texture = Self::create_texture(gpu.device(), frame.width, frame.height);
            self.bind_group = Self::create_bind_group(
                gpu.device(),
                &self.bind_group_layout,
                &self.texture,
                &self.sampler,
                &self.params_buffer,
            );
            self.reference_size = (frame.width, frame.height);
            self.write_params(gpu);
        }
        gpu.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &frame.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(frame.width * 4),
                rows_per_image: Some(frame.height),
            },
            wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    pub(crate) fn resize(&mut self, gpu: &GpuContext, output_size: (u32, u32)) {
        self.output_size = output_size;
        self.write_params(gpu);
    }

    fn write_params(&self, gpu: &GpuContext) {
        gpu.queue().write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&OverlayParams {
                reference_size: [self.reference_size.0 as f32, self.reference_size.1 as f32],
                output_size: [self.output_size.0 as f32, self.output_size.1 as f32],
            }),
        );
    }

    pub(crate) fn encode(
        &self,
        gpu: &GpuContext,
        target: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reco rgba overlay encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("reco rgba overlay pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.finish()
    }
}

/// Errors validating or preparing an overlay surface.
#[derive(Debug, Clone, Error)]
pub enum OverlayError {
    /// The supplied RGBA surface is malformed.
    #[error("invalid overlay frame: {0}")]
    InvalidFrame(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_tightly_packed_rgba() {
        let valid = OverlayFrame {
            width: 2,
            height: 3,
            rgba: vec![0; 24],
        };
        assert!(valid.validate().is_ok());
        let invalid = OverlayFrame {
            rgba: vec![0; 23],
            ..valid
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn rejects_zero_and_overflowing_dimensions() {
        assert!(
            OverlayFrame {
                width: 0,
                height: 1080,
                rgba: Vec::new(),
            }
            .validate()
            .is_err()
        );
        assert!(
            OverlayFrame {
                width: u32::MAX,
                height: u32::MAX,
                rgba: Vec::new(),
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn uses_straight_alpha_color_and_porter_duff_alpha() {
        assert_eq!(
            STRAIGHT_ALPHA_BLEND.color.src_factor,
            wgpu::BlendFactor::SrcAlpha
        );
        assert_eq!(
            STRAIGHT_ALPHA_BLEND.color.dst_factor,
            wgpu::BlendFactor::OneMinusSrcAlpha
        );
        assert_eq!(
            STRAIGHT_ALPHA_BLEND.alpha.src_factor,
            wgpu::BlendFactor::One
        );
        assert_eq!(
            STRAIGHT_ALPHA_BLEND.alpha.dst_factor,
            wgpu::BlendFactor::OneMinusSrcAlpha
        );
    }

    #[test]
    fn gpu_compositor_writes_visible_overlay_pixels() {
        let Ok(gpu) = GpuContext::new_blocking() else {
            eprintln!("GPU unavailable; skipping overlay compositor integration test");
            return;
        };
        let size = 64;
        let target = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay compositor test target"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut clear_encoder =
            gpu.device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("overlay compositor test clear"),
                });
        {
            let _pass = clear_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay compositor test clear pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        gpu.queue().submit(Some(clear_encoder.finish()));

        let frame = OverlayFrame {
            width: size,
            height: size,
            rgba: [255_u8, 0, 255, 255].repeat((size * size) as usize),
        };
        let compositor =
            RgbaOverlayCompositor::new(&gpu, wgpu::TextureFormat::Rgba8Unorm, (size, size), &frame)
                .unwrap();
        let mut readback = crate::gpu::rgba_readback::RgbaReadback::new(&gpu, size, size).unwrap();
        assert!(
            readback
                .readback(&gpu, &target, compositor.encode(&gpu, &view))
                .unwrap()
                .is_none()
        );
        let pixels = readback.flush_pending(&gpu).unwrap().unwrap();
        let center = ((size / 2 * size + size / 2) * 4) as usize;
        assert_eq!(&pixels[center..center + 4], &[255, 0, 255, 255]);
    }
}
