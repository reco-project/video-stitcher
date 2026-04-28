//! Single-camera lens preview renderer.
//!
//! Renders one camera's input through orthographic projection with the
//! KB4 fisheye shader. Unlike [`GpuUndistort`](crate::undistort::GpuUndistort)
//! (which does blocking GPU readback), this produces a fresh
//! `wgpu::Texture` per frame with `RENDER_ATTACHMENT | TEXTURE_BINDING`
//! so the result can be handed to a UI framework as a zero-copy image.
//!
//! The `lens_correction_amount` parameter (0.0 to 1.0) lets users
//! visually evaluate how much the lens profile contributes: 1.0 is
//! full KB4 correction, 0.0 is raw (pinhole projection).

use crate::calibration::CameraParams;
use crate::gpu::GpuContext;
use crate::pipeline::YuvPlanes;
use crate::renderer::{InputFormat, build_gpu_uniforms, opengl_to_wgpu_matrix};

use bytemuck::Pod;
use nalgebra::Orthographic3;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
}

impl Vertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 12,
                shader_location: 1,
            },
        ],
    };
}

fn quad_vertices(aspect: f32) -> [Vertex; 6] {
    let hw = 0.5;
    let hh = 0.5 / aspect;
    [
        Vertex {
            position: [-hw, -hh, 0.0],
            uv: [0.0, 1.0],
        },
        Vertex {
            position: [hw, -hh, 0.0],
            uv: [1.0, 1.0],
        },
        Vertex {
            position: [hw, hh, 0.0],
            uv: [1.0, 0.0],
        },
        Vertex {
            position: [-hw, -hh, 0.0],
            uv: [0.0, 1.0],
        },
        Vertex {
            position: [hw, hh, 0.0],
            uv: [1.0, 0.0],
        },
        Vertex {
            position: [-hw, hh, 0.0],
            uv: [0.0, 0.0],
        },
    ]
}

/// GPU resources for single-camera lens preview.
///
/// Reuse across frames. Each `render_yuv` call allocates a fresh
/// output texture (ownership transfers to the caller).
pub struct LensPreviewRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    y_texture: wgpu::Texture,
    u_texture: wgpu::Texture,
    v_texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    texture_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    plane_aspect: f32,
    output_format: wgpu::TextureFormat,
}

impl LensPreviewRenderer {
    /// Create a renderer for single-camera preview.
    ///
    /// `width`/`height` are the input frame dimensions.
    /// `output_format` should match the UI's expected texture format
    /// (e.g. `Rgba8Unorm` for Slint).
    pub fn new(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        plane_aspect: f32,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lens_preview"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fisheye.wgsl").into()),
        });

        let vertices = quad_vertices(plane_aspect);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lens_preview_quad"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lens_preview_tex_layout"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                texture_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lens_preview_uniform_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lens_preview_layout"),
            bind_group_layouts: &[&texture_layout, &uniform_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lens_preview_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex::LAYOUT],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lens_preview_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let create_tex = |label: &str, w: u32, h: u32| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage,
                view_formats: &[],
            })
        };

        let y_texture = create_tex("lens_preview_y", width, height);
        let u_texture = create_tex("lens_preview_u", width / 2, height / 2);
        let v_texture = create_tex("lens_preview_v", width / 2, height / 2);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lens_preview_uniforms"),
            size: std::mem::size_of::<crate::renderer::GpuUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lens_preview_uniform_bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            vertex_buffer,
            y_texture,
            u_texture,
            v_texture,
            sampler,
            texture_layout,
            uniform_buffer,
            uniform_bind_group,
            width,
            height,
            plane_aspect,
            output_format,
        }
    }

    /// Render one camera frame and return the output texture.
    ///
    /// The returned texture has `RENDER_ATTACHMENT | TEXTURE_BINDING` and
    /// can be passed to `slint::Image::try_from(wgpu::Texture)`.
    ///
    /// `correction_amount`: 0.0 = raw input, 1.0 = full KB4 correction.
    pub fn render_yuv(
        &self,
        gpu: &GpuContext,
        planes: &YuvPlanes<'_>,
        params: &CameraParams,
        correction_amount: f32,
    ) -> wgpu::Texture {
        upload_plane(
            &gpu.queue,
            &self.y_texture,
            planes.y,
            self.width,
            self.height,
        );
        upload_plane(
            &gpu.queue,
            &self.u_texture,
            planes.u,
            self.width / 2,
            self.height / 2,
        );
        upload_plane(
            &gpu.queue,
            &self.v_texture,
            planes.v,
            self.width / 2,
            self.height / 2,
        );

        let hh = 0.5 / self.plane_aspect;
        let ortho = Orthographic3::new(-0.5, 0.5, -hh, hh, -1.0, 1.0);
        let mvp = opengl_to_wgpu_matrix() * ortho.to_homogeneous();

        let mut uniforms =
            build_gpu_uniforms(&mvp, params, false, 0.0, InputFormat::Yuv420p, false);
        uniforms.lens_preview[0] = correction_amount.clamp(0.0, 1.0);

        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let texture_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lens_preview_tex_bg"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self.y_texture.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &self.u_texture.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &self.v_texture.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let output = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lens_preview_output"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.output_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output.create_view(&Default::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lens_preview_encode"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("lens_preview_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_bind_group(0, &texture_bind_group, &[]);
            pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }

        gpu.queue.submit(Some(encoder.finish()));
        output
    }
}

fn upload_plane(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    data: &[u8],
    width: u32,
    height: u32,
) {
    queue.write_texture(
        texture.as_image_copy(),
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}
