//! Single-camera lens preview renderer.
//!
//! Renders one camera's input through orthographic projection with the
//! KB4 fisheye shader. Unlike [`GpuUndistort`](super::undistort::GpuUndistort)
//! (which does blocking GPU readback), this produces a fresh
//! `wgpu::Texture` per frame with `RENDER_ATTACHMENT | TEXTURE_BINDING`
//! so the result can be handed to a UI framework as a zero-copy image.
//!
//! The correction amount (0.0 to 1.0, the per-lens `Lens.correction`)
//! lets users visually evaluate how much the lens profile contributes:
//! 1.0 is full KB4 correction, 0.0 is equidistant (zero polynomial, not raw).

use crate::calibration::Lens;
use crate::geometry::opengl_to_wgpu_matrix;
use crate::gpu::GpuContext;
use crate::render::color::YuvColor;
use crate::render::pipeline::Yuv420pPlanes;
use crate::render::renderer::{InputFormat, build_gpu_uniforms, video_compilation_options};

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
    extended_pipeline: std::sync::OnceLock<wgpu::RenderPipeline>,
    pipeline_layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
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
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lens_preview"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../shaders/video_sampling.wgsl"),
                    include_str!("../shaders/lens_mapping.wgsl"),
                    include_str!("../shaders/fisheye.wgsl")
                )
                .into(),
            ),
        });

        let vertices = quad_vertices(width as f32 / height as f32);
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

        let pipeline = create_pipeline(device, &pipeline_layout, &shader, output_format, false);

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
            size: std::mem::size_of::<crate::render::renderer::GpuUniforms>() as u64,
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
            extended_pipeline: std::sync::OnceLock::new(),
            pipeline_layout,
            shader,
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
            output_format,
        }
    }

    /// Render one camera frame and return the output texture.
    ///
    /// The returned texture has `RENDER_ATTACHMENT | TEXTURE_BINDING` and
    /// can be passed to `slint::Image::try_from(wgpu::Texture)`.
    ///
    /// Uses the lens's correction strength: zero is equidistant, not raw.
    /// `color` describes the matrix and range of the supplied 8-bit YUV planes.
    pub fn render_yuv(
        &self,
        gpu: &GpuContext,
        planes: &Yuv420pPlanes<'_>,
        params: &Lens,
        color: YuvColor,
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

        let hh = 0.5 * self.height as f32 / self.width as f32;
        let ortho = Orthographic3::new(-0.5, 0.5, -hh, hh, -1.0, 1.0);
        let mvp = opengl_to_wgpu_matrix() * ortho.to_homogeneous();

        let color = YuvColor {
            samples: Default::default(),
            ..color
        };
        let mut uniforms =
            build_gpu_uniforms(&mvp, params, false, 0.0, InputFormat::Yuv420p, false, color);
        uniforms.lens_preview[0] = params.correction;

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

        let pipeline = if color.flags() != 0 {
            self.extended_pipeline.get_or_init(|| {
                create_pipeline(
                    &gpu.device,
                    &self.pipeline_layout,
                    &self.shader,
                    self.output_format,
                    true,
                )
            })
        } else {
            &self.pipeline
        };
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
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_bind_group(0, &texture_bind_group, &[]);
            pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }

        gpu.queue.submit(Some(encoder.finish()));
        output
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    output_format: wgpu::TextureFormat,
    extended: bool,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("lens_preview_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Vertex::LAYOUT],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: video_compilation_options(extended),
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
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::yuv_converter::YuvConverter;

    #[test]
    fn source_points_follow_rendered_landmarks_at_each_correction_strength() {
        let Some(gpu) = crate::stitch::test_support::gpu_or_skip() else {
            return;
        };
        let (width, height) = (256_u32, 144_u32);
        let preview =
            LensPreviewRenderer::new(&gpu, width, height, wgpu::TextureFormat::Rgba8Unorm);
        let mut converter = YuvConverter::new(&gpu, width, height).unwrap();
        let chroma = vec![128; (width * height / 4) as usize];
        let lens = Lens::fisheye(
            1024,
            576,
            580.0,
            550.0,
            490.0,
            300.0,
            [0.2, -0.06, 0.015, -0.001],
        );
        let folded = Lens::fisheye(
            3840,
            2880,
            1500.0,
            1500.0,
            1920.0,
            1440.0,
            [0.5, 0.0, -0.08, 0.0],
        );
        for (mut lens, corrections, points) in [
            (
                lens,
                &[0.0, 0.35, 1.0][..],
                [(48_i32, 32_i32), (130, 80), (208, 114)],
            ),
            (folded, &[1.0][..], [(16, 12), (130, 80), (240, 132)]),
        ] {
            for &correction in corrections {
                lens.correction = correction;
                for (source_x, source_y) in points {
                    let point = [
                        (f64::from(source_x) + 0.5) / f64::from(width),
                        (f64::from(source_y) + 0.5) / f64::from(height),
                    ];
                    let (u, v) = crate::lens::inverse_fisheye(point[0], point[1], &lens).unwrap();
                    let mapped = [(u + 0.5) * 0.5, (v + 0.5) * 0.5];
                    let mut y = vec![16; (width * height) as usize];
                    for dy in -3_i32..=3 {
                        for dx in -3_i32..=3 {
                            if dx * dx + dy * dy <= 9 {
                                y[((source_y + dy) as u32 * width + (source_x + dx) as u32)
                                    as usize] = 235;
                            }
                        }
                    }
                    let output = preview.render_yuv(
                        &gpu,
                        &Yuv420pPlanes {
                            y: &y,
                            u: &chroma,
                            v: &chroma,
                        },
                        &lens,
                        Default::default(),
                    );
                    let empty = gpu
                        .device
                        .create_command_encoder(&Default::default())
                        .finish();
                    assert!(
                        converter
                            .convert_and_readback(&gpu, &output, empty)
                            .unwrap()
                            .is_none()
                    );
                    let pixels = converter.flush_pending(&gpu).unwrap().unwrap();
                    let mut weight = 0.0;
                    let mut center = [0.0, 0.0];
                    for (index, &value) in pixels[..(width * height) as usize].iter().enumerate() {
                        let contribution = f64::from(value.saturating_sub(32));
                        weight += contribution;
                        center[0] += (index % width as usize) as f64 * contribution;
                        center[1] += (index / width as usize) as f64 * contribution;
                    }
                    assert!(weight > 100.0, "rendered landmark disappeared");
                    for axis in 0..2 {
                        let extent = if axis == 0 { width } else { height };
                        let expected = mapped[axis] * f64::from(extent) - 0.5;
                        let actual = center[axis] / weight;
                        assert!(
                            (expected - actual).abs() < 0.75,
                            "correction={correction}, source={point:?}, axis={axis}: mapped {expected}, rendered {actual}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn lens_preview_initializes_extended_pipeline_only_for_extended_color() {
        let Some(gpu) = crate::stitch::test_support::gpu_or_skip() else {
            return;
        };
        let preview = LensPreviewRenderer::new(&gpu, 8, 8, wgpu::TextureFormat::Rgba8Unorm);
        let planes = Yuv420pPlanes {
            y: &[160; 64],
            u: &[128; 16],
            v: &[128; 16],
        };
        let lens = Lens::flat(8, 8);
        assert!(preview.extended_pipeline.get().is_none());
        preview.render_yuv(&gpu, &planes, &lens, YuvColor::default());
        assert!(preview.extended_pipeline.get().is_none());
        let color = YuvColor::delivery(crate::render::color::YuvMatrix::Bt2020Ncl);
        preview.render_yuv(&gpu, &planes, &lens, color);
        assert!(preview.extended_pipeline.get().is_some());
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }
}
