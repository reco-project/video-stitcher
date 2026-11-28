//! GPU compositing for calibrated projections.
//!
//! Source textures, sampling, colour conversion and output targets are shared.
//! L-shape draws two fisheye-corrected planes; cylinder uses a single fullscreen inverse map.

use crate::calibration::{Calibration, Lens};
use crate::geometry::{
    FAR_PLANE, NEAR_PLANE, RenderPose, matrix4_to_columns, opengl_to_wgpu_matrix, view_matrix,
};
use crate::gpu::GpuContext;
use crate::projection::l_shape::PlaneScene;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix4, Perspective3};
use std::sync::OnceLock;
use thiserror::Error;
use wgpu::util::DeviceExt;

// ---- Constants ----

/// Errors from the renderer.
#[derive(Debug, Clone, Error)]
pub enum RenderError {
    /// Frame data has wrong size.
    #[error("frame data size mismatch: expected {expected} bytes, got {actual}")]
    FrameSizeMismatch { expected: usize, actual: usize },
}

// ---- GPU-side structs ----

/// Uniform buffer layout (must match `Uniforms` in fisheye.wgsl exactly).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct GpuUniforms {
    mvp: [[f32; 4]; 4],
    intrinsics: [f32; 4],
    dist: [f32; 4],
    color_scale: [f32; 4],
    color_offset_blend: [f32; 4],
    flags: [u32; 4],
    pub(crate) lens_preview: [f32; 4],
}

/// Vertex with 3D position and UV coordinates.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
}

impl Vertex {
    pub(crate) const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
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

/// Generate unit-square vertices; model transforms supply each plane's size and placement.
///
/// The quad lies in the XY plane, centered at origin. The model matrix
/// positions and rotates it to match the v1 Three.js `PlaneGeometry`.
fn quad_vertices() -> [Vertex; 6] {
    let hw = 0.5; // half width
    let hh = 0.5; // half height
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

// ---- Renderer ----

/// Per-plane GPU resources (YUV textures + uniform buffer + bind groups).
struct PlaneResources {
    y_texture: wgpu::Texture,
    u_texture: wgpu::Texture,
    v_texture: wgpu::Texture,
    texture_bind_group: wgpu::BindGroup,
    imported_bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

/// Input pixel format for the renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// YUV420P: three separate R8 textures (Y full-res, U half-res, V half-res).
    /// Used with software decode or CPU-side conversion.
    Yuv420p,
    /// NV12: Y as R8 (full-res) + interleaved UV as Rg8 (half-res).
    /// NVDEC native output format. V texture is a 1x1 dummy.
    Nv12,
    /// Packed RGBA: a single `Rgba8Unorm` texture at full resolution
    /// holding pre-composited sRGB-domain RGB. Used by OBS Browser Source,
    /// screen capture, WebRTC ingest, and other non-camera sources whose
    /// native format is already RGB. The shader samples the single plane
    /// and writes the RGB triple straight out (no YUV conversion).
    /// `u_texture` / `v_texture` are 1x1 dummies in this mode.
    Rgba,
}

/// GPU-side pixel format for NV12-family zero-copy decode output.
///
/// Determines texture formats and byte widths for CUDA/Vulkan shared
/// texture creation. The shader works unchanged for all variants because
/// wgpu's Unorm normalization maps both 8-bit `[0, 255]` and 16-bit
/// `[0, 65535]` values to `[0.0, 1.0]` in the fragment shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GpuPixelFormat {
    /// 8-bit NV12 (standard H.264/HEVC decode output).
    /// Y plane: `R8Unorm`, UV plane: `Rg8Unorm`, 1 byte per sample.
    Nv12,
    /// 10-bit P010 (e.g. DJI Action 4 HEVC 10-bit).
    /// Y plane: `R16Unorm`, UV plane: `Rg16Unorm`, 2 bytes per sample.
    /// NVDEC stores 10-bit values in the upper bits of each `u16`.
    P010,
}

impl GpuPixelFormat {
    /// wgpu texture format for the Y (luma) plane.
    pub fn y_format(self) -> wgpu::TextureFormat {
        match self {
            Self::Nv12 => wgpu::TextureFormat::R8Unorm,
            Self::P010 => wgpu::TextureFormat::R16Unorm,
        }
    }

    /// wgpu texture format for the UV (chroma) plane.
    pub fn uv_format(self) -> wgpu::TextureFormat {
        match self {
            Self::Nv12 => wgpu::TextureFormat::Rg8Unorm,
            Self::P010 => wgpu::TextureFormat::Rg16Unorm,
        }
    }

    /// Bytes per luma/chroma sample (1 for 8-bit, 2 for 10-bit).
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Nv12 => 1,
            Self::P010 => 2,
        }
    }

    /// wgpu texture format for the full NV12/P010 texture (used for D3D11 import).
    pub fn wgpu_format(self) -> wgpu::TextureFormat {
        match self {
            Self::Nv12 => wgpu::TextureFormat::NV12,
            Self::P010 => wgpu::TextureFormat::P010,
        }
    }
}

struct VideoPipelines {
    main: wgpu::RenderPipeline,
    right_uncovered: Option<wgpu::RenderPipeline>,
}

pub(crate) fn video_compilation_options(
    extended: bool,
) -> wgpu::PipelineCompilationOptions<'static> {
    wgpu::PipelineCompilationOptions {
        constants: if extended {
            &[("extended_video_color", 1.0)]
        } else {
            &[]
        },
        ..Default::default()
    }
}

/// The GPU renderer for panoramic stitching.
///
/// Holds all wgpu resources: pipelines, textures, bind groups, and buffers.
/// Created once per pipeline and reused for every frame.
pub(crate) struct Renderer {
    video: VideoPipelines,
    extended_video: OnceLock<VideoPipelines>,
    video_program: crate::render::GpuProgram,
    video_shader: wgpu::ShaderModule,
    video_layout: wgpu::PipelineLayout,
    /// Planes-only helper: restores opaque alpha (1.0) across the full target.
    opaque_alpha_pipeline: Option<wgpu::RenderPipeline>,
    vertex_buffer: Option<wgpu::Buffer>,
    cameras: Vec<PlaneResources>,
    /// Caller-owned previews do not allocate an unused internal output texture.
    render_target: OnceLock<(wgpu::Texture, wgpu::TextureView)>,
    output_format: wgpu::TextureFormat,
    output_width: u32,
    output_height: u32,
    /// Input pixel format (YUV420P or NV12).
    input_format: InputFormat,
    color: super::color::YuvColor,
    /// Stored for creating bind groups from external textures (zero-copy).
    texture_layout: wgpu::BindGroupLayout,
    /// Shared sampler, stored for bind group creation.
    sampler: wgpu::Sampler,
    /// Device handle for creating bind groups (Arc-based, cheap to clone).
    device: wgpu::Device,
    /// Whether to flip UV coordinates for 180-degree rotation per camera [left, right].
    /// Set by the zero-copy path when the source video has rotation metadata.
    /// The CPU decode path handles rotation by reversing buffers instead.
    flip_180: [bool; 2],
}

impl Renderer {
    pub(crate) fn upload_yuv(
        &self,
        gpu: &GpuContext,
        camera: usize,
        data: &super::planes::Yuv420pPlanes<'_>,
    ) -> Result<(), RenderError> {
        upload_yuv(gpu, &self.cameras[camera], data.y, data.u, data.v)
    }

    pub(crate) fn upload_nv12(
        &self,
        gpu: &GpuContext,
        camera: usize,
        data: &super::planes::Nv12Planes<'_>,
    ) -> Result<(), RenderError> {
        upload_nv12(gpu, &self.cameras[camera], data.y, data.uv)
    }

    /// Create a new renderer with all GPU resources.
    ///
    /// Allocates textures, buffers, and compiles the shader pipeline.
    /// This is called once during pipeline initialization.
    ///
    /// `input_format` selects between YUV420P (3 separate planes) and
    /// NV12 (Y + interleaved UV). NV12 is the native NVDEC output format.
    pub fn new(
        gpu: &GpuContext,
        program: &crate::render::GpuProgram,
        output_width: u32,
        output_height: u32,
        source_sizes: &[(u32, u32)],
        output_format: wgpu::TextureFormat,
        input_format: InputFormat,
    ) -> Self {
        debug_assert!(
            program.pass != super::GpuPass::Planes || super::has_alpha_channel(output_format),
            "planar GPU compositing requires a 4-channel color render target format with an alpha channel, got {output_format:?}"
        );
        let device = &gpu.device;

        // Shader: compiled from the projection's GPU program descriptor -
        // the render pipeline builds exactly what the projection declares.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("projection_composite"),
            source: wgpu::ShaderSource::Wgsl(program.wgsl.into()),
        });

        // Vertex buffer (quad for both planes — same shape, different model matrices)
        let vertices = quad_vertices();
        let vertex_buffer = (program.pass == super::GpuPass::Planes).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quad_vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });

        // Bind group layouts — YUV420P: 3 plane textures + 1 sampler
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
            label: Some("texture_layout"),
            entries: &[
                texture_entry(0), // Y plane
                texture_entry(1), // U plane
                texture_entry(2), // V plane
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_layout"),
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

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stitch_pipeline_layout"),
            bind_group_layouts: &[&texture_layout, &uniform_layout],
            immediate_size: 0,
        });

        let video = Self::create_video_pipelines(
            device,
            program,
            &shader,
            &pipeline_layout,
            output_format,
            false,
        );
        let opaque_alpha_pipeline = if program.pass == super::GpuPass::Planes {
            let empty_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("stitch_opaque_alpha_layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
            let opaque_alpha = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("stitch_opaque_alpha_pipeline"),
                layout: Some(&empty_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_opaque_alpha"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_opaque_alpha"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: output_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALPHA,
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

            Some(opaque_alpha)
        } else {
            None
        };

        // Sampler (shared by both planes)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("video_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let (camera_count, uniform_size) = match program.pass {
            super::GpuPass::Planes => (2, std::mem::size_of::<GpuUniforms>()),
            super::GpuPass::Panorama => (
                2,
                std::mem::size_of::<crate::projection::panorama::PanoramaUniforms>(),
            ),
            super::GpuPass::Cylinder => (
                1,
                std::mem::size_of::<crate::projection::cylinder::CylinderUniforms>(),
            ),
        };
        let cameras = (0..camera_count)
            .map(|camera| {
                Self::create_plane_resources(
                    device,
                    &texture_layout,
                    &uniform_layout,
                    &sampler,
                    source_sizes[camera],
                    input_format,
                    &format!("camera_{camera}"),
                    uniform_size as u64,
                )
            })
            .collect();

        Self {
            video,
            extended_video: OnceLock::new(),
            video_program: *program,
            video_shader: shader,
            video_layout: pipeline_layout,
            opaque_alpha_pipeline,
            vertex_buffer,
            cameras,
            render_target: OnceLock::new(),
            output_format,
            output_width,
            output_height,
            input_format,
            color: super::color::YuvColor::default(),
            texture_layout,
            sampler,
            device: device.clone(),
            flip_180: [false, false],
        }
    }

    fn create_video_pipelines(
        device: &wgpu::Device,
        program: &crate::render::GpuProgram,
        shader: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        output_format: wgpu::TextureFormat,
        extended: bool,
    ) -> VideoPipelines {
        // Render pipeline with alpha blending for seam transition
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stitch_render_pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(program.vs_entry),
                compilation_options: Default::default(),
                buffers: if program.pass == super::GpuPass::Planes {
                    &[Vertex::LAYOUT]
                } else {
                    &[]
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(program.fs_entry),
                compilation_options: video_compilation_options(extended),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(program.blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Both sides visible
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let right_uncovered = if program.pass == super::GpuPass::Planes {
            let right_uncovered = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("stitch_right_uncovered_pipeline"),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some(program.vs_entry),
                    compilation_options: Default::default(),
                    buffers: &[Vertex::LAYOUT],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_right_uncovered"),
                    compilation_options: video_compilation_options(extended),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: output_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                                dst_factor: wgpu::BlendFactor::DstAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::COLOR,
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

            Some(right_uncovered)
        } else {
            None
        };
        VideoPipelines {
            main: pipeline,
            right_uncovered,
        }
    }

    fn video_pipelines(&self) -> &VideoPipelines {
        if self.color.flags() == 0 || self.input_format == InputFormat::Rgba {
            return &self.video;
        }
        self.extended_video.get_or_init(|| {
            Self::create_video_pipelines(
                &self.device,
                &self.video_program,
                &self.video_shader,
                &self.video_layout,
                self.output_format,
                true,
            )
        })
    }

    fn create_plane_resources(
        device: &wgpu::Device,
        texture_layout: &wgpu::BindGroupLayout,
        uniform_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        (width, height): (u32, u32),
        input_format: InputFormat,
        label: &str,
        uniform_size: u64,
    ) -> PlaneResources {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;

        let create_texture = |name: &str, w: u32, h: u32, format: wgpu::TextureFormat| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("{label}_{name}")),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };

        // Y plane format + u/v plane selection depend on input format.
        // For Rgba the "Y" slot actually holds a full-res Rgba8Unorm texture;
        // u/v are 1x1 dummies that the shader never samples.
        let y_texture = match input_format {
            InputFormat::Yuv420p | InputFormat::Nv12 => {
                create_texture("y", width, height, wgpu::TextureFormat::R8Unorm)
            }
            InputFormat::Rgba => {
                create_texture("rgba", width, height, wgpu::TextureFormat::Rgba8Unorm)
            }
        };

        let (u_texture, v_texture) = match input_format {
            InputFormat::Yuv420p => {
                // YUV420P: separate R8 U and V at half resolution
                let u = create_texture("u", width / 2, height / 2, wgpu::TextureFormat::R8Unorm);
                let v = create_texture("v", width / 2, height / 2, wgpu::TextureFormat::R8Unorm);
                (u, v)
            }
            InputFormat::Nv12 => {
                // NV12: interleaved UV as Rg8Unorm at half resolution
                let uv = create_texture("uv", width / 2, height / 2, wgpu::TextureFormat::Rg8Unorm);
                // Dummy V texture - shader won't sample it in NV12 mode
                let v_dummy = create_texture("v_dummy", 1, 1, wgpu::TextureFormat::R8Unorm);
                (uv, v_dummy)
            }
            InputFormat::Rgba => {
                // Shader skips u/v sampling on the Rgba path; 1x1 dummies
                // satisfy the bind group layout without wasting memory.
                let u_dummy = create_texture("u_dummy", 1, 1, wgpu::TextureFormat::R8Unorm);
                let v_dummy = create_texture("v_dummy", 1, 1, wgpu::TextureFormat::R8Unorm);
                (u_dummy, v_dummy)
            }
        };

        let y_view = y_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let u_view = u_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let v_view = v_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}_texture_bg")),
            layout: texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&u_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&v_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}_uniforms")),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}_uniform_bg")),
            layout: uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        PlaneResources {
            y_texture,
            u_texture,
            v_texture,
            texture_bind_group,
            imported_bind_group: None,
            uniform_buffer,
            uniform_bind_group,
            width,
            height,
        }
    }

    /// Upload YUV420P planes to the left camera textures.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "gpu_upload")
    )]
    pub fn upload_left_yuv(
        &self,
        gpu: &GpuContext,
        y: &[u8],
        u: &[u8],
        v: &[u8],
    ) -> Result<(), RenderError> {
        upload_yuv(gpu, &self.cameras[0], y, u, v)
    }

    /// Upload YUV420P planes to the right camera textures.
    pub fn upload_right_yuv(
        &self,
        gpu: &GpuContext,
        y: &[u8],
        u: &[u8],
        v: &[u8],
    ) -> Result<(), RenderError> {
        upload_yuv(gpu, &self.cameras[1], y, u, v)
    }

    /// Upload a packed RGBA plane to the left camera texture.
    ///
    /// Data must be `width * height * 4` bytes in (R, G, B, A) byte order.
    /// Upload-side swizzling translates BGRA sources before calling this -
    /// see [`pipeline::RgbaPlanes`](crate::render::pipeline::RgbaPlanes).
    pub fn upload_left_rgba(&self, gpu: &GpuContext, rgba: &[u8]) -> Result<(), RenderError> {
        debug_assert_eq!(
            self.input_format,
            InputFormat::Rgba,
            "upload_left_rgba requires InputFormat::Rgba"
        );
        upload_rgba(gpu, &self.cameras[0], rgba)
    }

    /// Upload a packed RGBA plane to the right camera texture.
    /// See [`Self::upload_left_rgba`].
    pub fn upload_right_rgba(&self, gpu: &GpuContext, rgba: &[u8]) -> Result<(), RenderError> {
        debug_assert_eq!(
            self.input_format,
            InputFormat::Rgba,
            "upload_right_rgba requires InputFormat::Rgba"
        );
        upload_rgba(gpu, &self.cameras[1], rgba)
    }

    /// Copy a GPU texture into the left input plane (RGBA mode).
    ///
    /// Zero-copy alternative to [`upload_left_rgba`]: the source texture
    /// must be `Rgba8Unorm` with matching dimensions. The copy is
    /// appended to `encoder` as a GPU-side blit with no CPU involvement.
    pub fn copy_texture_to_left(&self, encoder: &mut wgpu::CommandEncoder, source: &wgpu::Texture) {
        copy_texture_to_plane(encoder, source, &self.cameras[0]);
    }

    /// Copy a GPU texture into the right input plane (RGBA mode).
    pub fn copy_texture_to_right(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
    ) {
        copy_texture_to_plane(encoder, source, &self.cameras[1]);
    }

    /// Create a texture bind group from external textures.
    ///
    /// Used for CUDA/Vulkan zero-copy: pre-build one bind group per
    /// double-buffer slot (before the render loop), then select the active
    /// slot each frame by cloning the appropriate pre-built group (cheap
    /// Arc refcount increment) and passing it to
    /// [`Self::set_camera_bind_group`].
    /// `wgpu::BindGroup` implements `Clone`, so no GPU allocation occurs on
    /// the per-frame path.
    pub fn create_texture_bind_group(
        &self,
        y_texture: &wgpu::Texture,
        uv_texture: &wgpu::Texture,
        label: &str,
    ) -> wgpu::BindGroup {
        let y_view = y_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uv_view = uv_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.create_bind_group_from_views(&y_view, &uv_view, label)
    }

    /// Create a texture bind group from pre-built texture views.
    ///
    /// Used by the D3D11VA zero-copy path where NV12 plane views require
    /// explicit `TextureAspect::Plane0` / `Plane1` and cannot be obtained
    /// from `TextureViewDescriptor::default()`.
    pub fn create_bind_group_from_views(
        &self,
        y_view: &wgpu::TextureView,
        uv_view: &wgpu::TextureView,
        label: &str,
    ) -> wgpu::BindGroup {
        let v_view = self.cameras[0]
            .v_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(uv_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&v_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Set a camera's texture bind group for the next render.
    pub fn set_camera_bind_group(&mut self, camera: usize, bind_group: wgpu::BindGroup) {
        self.cameras[camera].imported_bind_group = Some(bind_group);
    }

    /// Release imported bindings and sample the owned upload textures again.
    pub(crate) fn clear_imported_bind_groups(&mut self) {
        for camera in &mut self.cameras {
            camera.imported_bind_group = None;
        }
    }

    /// Enable 180-degree UV flip for the GPU zero-copy path.
    ///
    /// When set, the shader flips texture coordinates before sampling,
    /// equivalent to the CPU path's buffer reversal for rotated video.
    pub fn set_flip_180(&mut self, left: bool, right: bool) {
        self.flip_180 = [left, right];
    }

    /// Set full-range YUV mode for the shader.
    pub fn set_full_range(&mut self, full_range: bool) {
        self.color.full_range = full_range;
    }

    pub(crate) fn set_source_color(&mut self, color: super::color::YuvColor) {
        self.color = color;
    }

    pub(crate) fn set_sample_format(
        &mut self,
        samples: super::color::YuvSampleFormat,
    ) -> super::color::YuvSampleFormat {
        std::mem::replace(&mut self.color.samples, samples)
    }

    /// Create fresh `TextureView`s for the left plane's Y/U/V
    /// textures. Needed by [`crate::gpu::yuv_stack_packer::YuvStackPacker`]
    /// when replay recording is enabled: the packer samples the same
    /// uploaded source data the stitch shader reads, producing a
    /// tiled atlas in parallel with the panorama render.
    ///
    /// For NV12 inputs the returned `U` view is the interleaved UV
    /// texture (Rg8Unorm) and the `V` view is the 1×1 dummy; the
    /// packer's NV12 kernel ignores the V binding.
    pub(crate) fn left_plane_views(
        &self,
    ) -> (wgpu::TextureView, wgpu::TextureView, wgpu::TextureView) {
        Self::plane_views(&self.cameras[0])
    }

    /// Right-side counterpart to [`Self::left_plane_views`].
    pub(crate) fn right_plane_views(
        &self,
    ) -> (wgpu::TextureView, wgpu::TextureView, wgpu::TextureView) {
        Self::plane_views(&self.cameras[1])
    }

    fn plane_views(
        plane: &PlaneResources,
    ) -> (wgpu::TextureView, wgpu::TextureView, wgpu::TextureView) {
        (
            plane
                .y_texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            plane
                .u_texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            plane
                .v_texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
        )
    }

    /// Input format the renderer was built for. Surfaced so the
    /// stacked-video packer can pick the right shader variant
    /// (YUV420P vs NV12) at session setup.
    pub(crate) fn input_format(&self) -> InputFormat {
        self.input_format
    }

    /// Encode the shared stitch render pass: projection, uniforms, and draw calls.
    ///
    /// Returns the command encoder with the render pass already recorded.
    /// Callers handle submission, readback, or further encoding as needed.
    #[allow(clippy::too_many_arguments)]
    fn encode_stitch_pass(
        &self,
        gpu: &GpuContext,
        scene: Option<&PlaneScene>,
        calibration: &Calibration,
        pose: RenderPose,
        blend_width: f32,
        target_view: &wgpu::TextureView,
        aspect: f32,
        encoder_label: &str,
    ) -> wgpu::CommandEncoder {
        if let Some(scene) = scene {
            let projection = opengl_to_wgpu_matrix()
                * Perspective3::new(
                    aspect,
                    pose.render_fov().to_radians(),
                    NEAR_PLANE,
                    FAR_PLANE,
                )
                .to_homogeneous();
            let view = view_matrix(
                &scene.camera,
                pose.yaw,
                pose.pitch,
                calibration.framing.tilt as f32,
                calibration.framing.roll as f32,
            );

            let left_mvp = projection * view * scene.models[0];
            let mut left_uniforms = build_gpu_uniforms(
                &left_mvp,
                &calibration.lenses[0],
                false,
                blend_width,
                self.input_format,
                self.flip_180[0],
                self.color,
            );
            left_uniforms.lens_preview[0] = calibration.lenses[0].correction;

            let right_mvp = projection * view * scene.models[1];
            let mut right_uniforms = build_gpu_uniforms(
                &right_mvp,
                &calibration.lenses[1],
                true,
                blend_width,
                self.input_format,
                self.flip_180[1],
                self.color,
            );
            right_uniforms.lens_preview[0] = calibration.lenses[1].correction;

            gpu.queue.write_buffer(
                &self.cameras[0].uniform_buffer,
                0,
                bytemuck::bytes_of(&left_uniforms),
            );
            gpu.queue.write_buffer(
                &self.cameras[1].uniform_buffer,
                0,
                bytemuck::bytes_of(&right_uniforms),
            );
        } else if let crate::calibration::Projection::Panorama(panorama) = &calibration.projection {
            let uniforms = panorama.gpu_uniforms(
                &calibration.lenses,
                &calibration.framing,
                std::array::from_fn(|i| {
                    video_flags(i as u32, self.input_format, self.flip_180[i], self.color)
                }),
            );
            for (camera, uniform) in self.cameras.iter().zip(&uniforms) {
                gpu.queue
                    .write_buffer(&camera.uniform_buffer, 0, bytemuck::bytes_of(uniform));
            }
        } else if let crate::calibration::Projection::Cylinder(cylinder) = &calibration.projection {
            let uniforms = cylinder.gpu_uniforms(
                &calibration.framing,
                f64::from(calibration.lenses[0].height),
                aspect,
                pose,
                video_flags(0, self.input_format, self.flip_180[0], self.color),
            );
            gpu.queue.write_buffer(
                &self.cameras[0].uniform_buffer,
                0,
                bytemuck::bytes_of(&uniforms),
            );
        }
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(encoder_label),
            });

        let pipelines = self.video_pipelines();
        let planes_blend_pipelines = if scene.is_some() && blend_width > 0.0 {
            pipelines
                .right_uncovered
                .as_ref()
                .zip(self.opaque_alpha_pipeline.as_ref())
        } else {
            None
        };
        let clear_color = if planes_blend_pipelines.is_some() {
            wgpu::Color::TRANSPARENT
        } else {
            wgpu::Color::BLACK
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stitch_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            let bind_camera = |pass: &mut wgpu::RenderPass<'_>, camera: &PlaneResources| {
                pass.set_bind_group(
                    0,
                    camera
                        .imported_bind_group
                        .as_ref()
                        .unwrap_or(&camera.texture_bind_group),
                    &[],
                );
                pass.set_bind_group(1, &camera.uniform_bind_group, &[]);
            };

            if let Some((right_uncovered_pipeline, opaque_alpha_pipeline)) = planes_blend_pipelines
            {
                let vertices = self
                    .vertex_buffer
                    .as_ref()
                    .expect("GpuPass::Planes always has a vertex buffer");
                pass.set_vertex_buffer(0, vertices.slice(..));

                // Draw 1: Left camera (base). Covered pixels become (L, 1.0);
                // uncovered pixels stay (0, 0, 0, 0).
                pass.set_pipeline(&pipelines.main);
                bind_camera(&mut pass, &self.cameras[0]);
                pass.draw(0..6, 0..1);

                // Draw 2: Right camera uncovered fill inside the fade band.
                // Where dst.a == 0.0, writes R without modifying dst.a; where
                // dst.a == 1.0, keeps L untouched.
                pass.set_pipeline(right_uncovered_pipeline);
                bind_camera(&mut pass, &self.cameras[1]);
                pass.draw(0..6, 0..1);

                // Draw 3: Right camera standard smoothstep blend.
                // Over left-covered pixels (L): a * R + (1 - a) * L.
                // Over right-only fade-band pixels (R): a * R + (1 - a) * R = R.
                // Outside fade band (a = 1): R.
                pass.set_pipeline(&pipelines.main);
                pass.draw(0..6, 0..1);

                // Draw 4: Restore dst.a = 1.0 across the full target without
                // touching RGB so covered and uncovered pixels stay opaque.
                pass.set_pipeline(opaque_alpha_pipeline);
                pass.draw(0..3, 0..1);
            } else {
                pass.set_pipeline(&pipelines.main);
                if let Some(vertices) = &self.vertex_buffer {
                    pass.set_vertex_buffer(0, vertices.slice(..));
                }
                for camera in &self.cameras {
                    bind_camera(&mut pass, camera);
                    pass.draw(0..if self.vertex_buffer.is_some() { 6 } else { 3 }, 0..1);
                }
            }
        }

        encoder
    }

    /// Render a stitched frame to the internal render target, without readback.
    ///
    /// Returns the recorded `CommandBuffer` without submitting it.
    /// The caller should submit it (typically together with NV12 conversion
    /// commands) to ensure proper GPU synchronization.
    /// Use [`Self::render_target`] to get a reference to the output texture.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "gpu_render_to_target")
    )]
    pub fn render_to_target(
        &self,
        gpu: &GpuContext,
        scene: Option<&PlaneScene>,
        calibration: &Calibration,
        pose: RenderPose,
        blend_width: f32,
    ) -> wgpu::CommandBuffer {
        let aspect = self.output_width as f32 / self.output_height as f32;
        let encoder = self.encode_stitch_pass(
            gpu,
            scene,
            calibration,
            pose,
            blend_width,
            &self.output_target().1,
            aspect,
            "stitch_to_target",
        );
        encoder.finish()
    }

    /// Access the internal render target texture.
    ///
    /// Used by [`Nv12Converter`](crate::gpu::nv12_converter::Nv12Converter) to read
    /// the RGBA output without an intermediate CPU copy.
    pub fn render_target(&self) -> &wgpu::Texture {
        &self.output_target().0
    }

    /// Configured output format, without allocating internal image storage.
    pub fn output_format(&self) -> wgpu::TextureFormat {
        self.output_format
    }

    /// Invalidate only output storage, retaining shaders and camera resources.
    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) != (self.output_width, self.output_height) {
            self.render_target.take();
            self.output_width = width;
            self.output_height = height;
        }
    }

    fn output_target(&self) -> &(wgpu::Texture, wgpu::TextureView) {
        self.render_target.get_or_init(|| {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("render_target"),
                size: wgpu::Extent3d {
                    width: self.output_width,
                    height: self.output_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.output_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            (texture, view)
        })
    }

    /// Render a stitched frame directly to a texture view (e.g., a window surface).
    ///
    /// Unlike [`Self::render_to_target`], this does NOT read back the result to CPU.
    /// Used for interactive preview windows.
    pub fn render_to_view(
        &self,
        gpu: &GpuContext,
        scene: Option<&PlaneScene>,
        calibration: &Calibration,
        pose: RenderPose,
        aspect: f32,
        blend_width: f32,
        target_view: &wgpu::TextureView,
    ) -> wgpu::SubmissionIndex {
        let encoder = self.encode_stitch_pass(
            gpu,
            scene,
            calibration,
            pose,
            blend_width,
            target_view,
            aspect,
            "preview_frame",
        );
        gpu.queue.submit(Some(encoder.finish()))
    }

    /// Test-only isolated single-camera GPU draw (cleared to transparent,
    /// `blend_width = 0.0`) so oracle tests can compare combined compositing
    /// directly against actual hardware left-only and right-only rasterization.
    #[cfg(test)]
    pub(crate) fn render_isolated_camera_to_target(
        &self,
        gpu: &GpuContext,
        scene: &PlaneScene,
        calibration: &Calibration,
        pose: RenderPose,
        camera: usize,
    ) -> wgpu::CommandBuffer {
        let aspect = self.output_width as f32 / self.output_height as f32;
        let projection = opengl_to_wgpu_matrix()
            * Perspective3::new(
                aspect,
                pose.render_fov().to_radians(),
                NEAR_PLANE,
                FAR_PLANE,
            )
            .to_homogeneous();
        let view = view_matrix(
            &scene.camera,
            pose.yaw,
            pose.pitch,
            calibration.framing.tilt as f32,
            calibration.framing.roll as f32,
        );
        let mvp = projection * view * scene.models[camera];
        let mut uniforms = build_gpu_uniforms(
            &mvp,
            &calibration.lenses[camera],
            camera == 1,
            0.0,
            self.input_format,
            self.flip_180[camera],
            self.color,
        );
        uniforms.lens_preview[0] = calibration.lenses[camera].correction;
        gpu.queue.write_buffer(
            &self.cameras[camera].uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("isolated_camera_to_target"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("isolated_camera_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.output_target().1,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            let vertices = self
                .vertex_buffer
                .as_ref()
                .expect("GpuPass::Planes always has a vertex buffer");
            let plane = &self.cameras[camera];
            pass.set_pipeline(&self.video_pipelines().main);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_bind_group(
                0,
                plane
                    .imported_bind_group
                    .as_ref()
                    .unwrap_or(&plane.texture_bind_group),
                &[],
            );
            pass.set_bind_group(1, &plane.uniform_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        encoder.finish()
    }
}

// ---- Helper functions ----

/// Upload a packed RGBA plane (4 bytes per pixel) to a GPU texture.
///
/// Expects `width * height * 4` bytes in (R, G, B, A) order.
/// Callers with BGRA source data need to swizzle to RGBA before
/// this call - the shader samples `rgba.rgb` directly.
fn upload_rgba(gpu: &GpuContext, plane: &PlaneResources, rgba: &[u8]) -> Result<(), RenderError> {
    let w = plane.width;
    let h = plane.height;
    let expected = (w * h * 4) as usize;
    if rgba.len() != expected {
        return Err(RenderError::FrameSizeMismatch {
            expected,
            actual: rgba.len(),
        });
    }
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &plane.y_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    Ok(())
}

/// GPU-to-GPU texture copy into a plane's y_texture.
fn copy_texture_to_plane(
    encoder: &mut wgpu::CommandEncoder,
    source: &wgpu::Texture,
    plane: &PlaneResources,
) {
    let size = wgpu::Extent3d {
        width: plane.width,
        height: plane.height,
        depth_or_array_layers: 1,
    };
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: source,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &plane.y_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        size,
    );
}

/// Upload a single R8Unorm plane to a GPU texture.
fn upload_plane(gpu: &GpuContext, texture: &wgpu::Texture, data: &[u8], width: u32, height: u32) {
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
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

/// Upload YUV420P planes (Y full-res, U/V half-res) to GPU textures.
fn upload_yuv(
    gpu: &GpuContext,
    plane: &PlaneResources,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Result<(), RenderError> {
    let w = plane.width;
    let h = plane.height;
    let uv_w = w / 2;
    let uv_h = h / 2;

    if y.len() != (w * h) as usize {
        return Err(RenderError::FrameSizeMismatch {
            expected: (w * h) as usize,
            actual: y.len(),
        });
    }
    if u.len() != (uv_w * uv_h) as usize {
        return Err(RenderError::FrameSizeMismatch {
            expected: (uv_w * uv_h) as usize,
            actual: u.len(),
        });
    }
    if v.len() != (uv_w * uv_h) as usize {
        return Err(RenderError::FrameSizeMismatch {
            expected: (uv_w * uv_h) as usize,
            actual: v.len(),
        });
    }

    upload_plane(gpu, &plane.y_texture, y, w, h);
    upload_plane(gpu, &plane.u_texture, u, uv_w, uv_h);
    upload_plane(gpu, &plane.v_texture, v, uv_w, uv_h);
    Ok(())
}

/// Upload NV12 planes (Y full-res, interleaved UV half-res) to GPU textures.
///
/// UV plane is `Rg8Unorm` at half resolution in each dimension.
/// Each texel contains (U, V) as two bytes.
fn upload_nv12(
    gpu: &GpuContext,
    plane: &PlaneResources,
    y: &[u8],
    uv: &[u8],
) -> Result<(), RenderError> {
    let w = plane.width;
    let h = plane.height;
    let uv_w = w / 2;
    let uv_h = h / 2;

    if y.len() != (w * h) as usize {
        return Err(RenderError::FrameSizeMismatch {
            expected: (w * h) as usize,
            actual: y.len(),
        });
    }
    if uv.len() != (uv_w * uv_h * 2) as usize {
        return Err(RenderError::FrameSizeMismatch {
            expected: (uv_w * uv_h * 2) as usize,
            actual: uv.len(),
        });
    }

    upload_plane(gpu, &plane.y_texture, y, w, h);
    // UV plane is Rg8Unorm: 2 bytes per texel, so bytes_per_row = uv_w * 2
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &plane.u_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        uv,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(uv_w * 2),
            rows_per_image: Some(uv_h),
        },
        wgpu::Extent3d {
            width: uv_w,
            height: uv_h,
            depth_or_array_layers: 1,
        },
    );
    Ok(())
}

fn video_flags(
    camera: u32,
    input: InputFormat,
    flip_180: bool,
    color: super::color::YuvColor,
) -> [u32; 4] {
    [camera, input as u32, u32::from(flip_180), color.flags()]
}

/// Build the GPU uniform struct for one plane.
///
/// `flip_180`: when true, the shader flips UV coordinates to apply
/// 180-degree rotation. Used by the GPU zero-copy path where the CPU
/// buffer-reversal trick from the software decode path is not possible.
pub(crate) fn build_gpu_uniforms(
    mvp: &Matrix4<f32>,
    camera: &Lens,
    is_right: bool,
    blend_width: f32,
    input_format: InputFormat,
    flip_180: bool,
    color: super::color::YuvColor,
) -> GpuUniforms {
    let w = camera.width as f32;
    let h = camera.height as f32;
    GpuUniforms {
        mvp: matrix4_to_columns(mvp),
        intrinsics: [
            camera.fx as f32 / w,
            camera.fy as f32 / h,
            camera.cx as f32 / w,
            camera.cy as f32 / h,
        ],
        dist: [
            camera.distortion[0] as f32,
            camera.distortion[1] as f32,
            camera.distortion[2] as f32,
            camera.distortion[3] as f32,
        ],
        color_scale: [1.0, 1.0, 1.0, 0.0],
        color_offset_blend: [0.0, 0.0, 0.0, blend_width],
        flags: video_flags(u32::from(is_right), input_format, flip_180, color),
        // Full correction for normal stitching. LensPreviewRenderer
        // overrides this field for the single-camera preview mode.
        lens_preview: [1.0, 0.0, 0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_output_is_lazy_and_only_size_changes_invalidate_it() {
        use crate::stitch::test_support::{calib, gpu_or_skip};
        let Some(gpu) = gpu_or_skip() else { return };
        let calibration = calib(64, 32);
        let program = calibration.projection.geometry().gpu_program().unwrap();
        let mut renderer = Renderer::new(
            &gpu,
            &program,
            80,
            48,
            &[(64, 32); 2],
            wgpu::TextureFormat::Rgba8Unorm,
            InputFormat::Nv12,
        );
        assert!(renderer.render_target.get().is_none());
        assert_eq!(renderer.output_format(), wgpu::TextureFormat::Rgba8Unorm);
        renderer.resize(48, 80);
        assert!(renderer.render_target.get().is_none());
        assert_eq!(
            (
                renderer.render_target().width(),
                renderer.render_target().height()
            ),
            (48, 80)
        );
        let retained = renderer.render_target().clone();
        renderer.resize(48, 80);
        assert_eq!(renderer.render_target(), &retained);
        renderer.resize(128, 72);
        assert!(renderer.render_target.get().is_none());
        assert_ne!(renderer.render_target(), &retained);
        assert_eq!(
            (
                renderer.render_target().width(),
                renderer.render_target().height()
            ),
            (128, 72)
        );
    }

    #[test]
    fn uniforms_are_normalized() {
        assert_eq!(std::mem::size_of::<GpuUniforms>(), 160);
        assert_eq!(
            std::mem::size_of::<crate::projection::panorama::PanoramaUniforms>(),
            224
        );
        assert_eq!(
            std::mem::size_of::<crate::projection::cylinder::CylinderUniforms>(),
            80
        );
        let camera = Lens::fisheye(
            3840,
            2160,
            1796.32,
            1797.22,
            1919.37,
            1063.17,
            [0.0342, 0.0677, -0.0741, 0.0299],
        );
        let mvp = Matrix4::identity();
        let u = build_gpu_uniforms(
            &mvp,
            &camera,
            false,
            0.0,
            InputFormat::Yuv420p,
            false,
            Default::default(),
        );

        // fx/width ≈ 0.4678
        assert!((u.intrinsics[0] - 1796.32 / 3840.0).abs() < 1e-4);
        // cy/height ≈ 0.4922
        assert!((u.intrinsics[3] - 1063.17 / 2160.0).abs() < 1e-4);
        // is_right = 0, use_nv12 = 0
        assert_eq!(u.flags[0], 0);
        assert_eq!(u.flags[1], 0);
    }
}
