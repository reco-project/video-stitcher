//! GPU renderer for the panoramic stitching pipeline.
//!
//! Manages wgpu render pipelines, textures, and bind groups for rendering
//! two fisheye-corrected camera planes into a stitched panoramic output.
//!
//! ## Pipeline
//!
//! ```text
//! YUV420P path:
//!   Left Y/U/V planes ──► 3 textures ──┐
//!                                       ├──► Render pass (YUV→RGB + fisheye) ──► RGBA output
//!   Right Y/U/V planes ──► 3 textures ──┘
//!
//! NV12 path:
//!   Left Y + UV planes ──► 2 textures ──┐
//!                                        ├──► Render pass (NV12→RGB + fisheye) ──► RGBA output
//!   Right Y + UV planes ──► 2 textures ──┘
//! ```
//!
//! Each plane is a textured quad positioned in 3D space (L-shape geometry).
//! YUV/NV12 to RGB conversion (BT.709), fisheye undistortion, and color
//! correction all happen in the fragment shader. Uploading YUV directly
//! reduces CPU-GPU transfer from 8.3 MB to 3.1 MB per frame (62% less
//! bandwidth) and eliminates CPU-side swscale color conversion entirely.

use crate::calibration::{CameraParams, MatchCalibration};
use crate::gpu::GpuContext;
use crate::scene::SceneGeometry;
use crate::viewport::ResolvedViewport;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix4, Perspective3, Point3, UnitQuaternion, Vector3};
use thiserror::Error;
use wgpu::util::DeviceExt;

// ---- Constants ----

/// Near clipping plane for the perspective projection.
const NEAR_PLANE: f32 = 0.01;
/// Far clipping plane for the perspective projection.
const FAR_PLANE: f32 = 5.0;
/// Aspect ratio of scene planes (matches GoPro 16:9 capture).
///
/// Deprecated: derive the aspect ratio from camera parameters instead.
/// Use [`SceneGeometry::from_layout_with_aspect`](crate::scene::SceneGeometry::from_layout_with_aspect)
/// with `camera.width as f32 / camera.height as f32`.
#[deprecated(
    since = "0.1.0",
    note = "derive aspect ratio from camera parameters (width/height) instead"
)]
pub const PLANE_ASPECT: f32 = 16.0 / 9.0;

/// Errors from the renderer.
#[derive(Debug, Error)]
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
}

/// Vertex with 3D position and UV coordinates.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
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

/// Generate quad vertices for a plane (1.0 wide, given aspect ratio).
///
/// The quad lies in the XY plane, centered at origin. The model matrix
/// positions and rotates it to match the v1 Three.js `PlaneGeometry`.
fn quad_vertices(aspect: f32) -> [Vertex; 6] {
    let hw = 0.5; // half width
    let hh = 0.5 / aspect; // half height
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
    /// NVDEC native output format. V texture is a 1×1 dummy.
    Nv12,
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
}

/// The GPU renderer for panoramic stitching.
///
/// Holds all wgpu resources: pipelines, textures, bind groups, and buffers.
/// Created once per pipeline and reused for every frame.
pub(crate) struct Renderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    left: PlaneResources,
    right: PlaneResources,
    render_target: wgpu::Texture,
    render_target_view: wgpu::TextureView,
    output_width: u32,
    output_height: u32,
    /// Input pixel format (YUV420P or NV12).
    input_format: InputFormat,
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
    /// Create a new renderer with all GPU resources.
    ///
    /// Allocates textures, buffers, and compiles the shader pipeline.
    /// This is called once during pipeline initialization.
    ///
    /// `input_format` selects between YUV420P (3 separate planes) and
    /// NV12 (Y + interleaved UV). NV12 is the native NVDEC output format.
    pub fn new(
        gpu: &GpuContext,
        output_width: u32,
        output_height: u32,
        input_width: u32,
        input_height: u32,
        output_format: wgpu::TextureFormat,
        input_format: InputFormat,
        scene: &SceneGeometry,
    ) -> Self {
        let device = &gpu.device;

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fisheye"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fisheye.wgsl").into()),
        });

        // Vertex buffer (quad for both planes — same shape, different model matrices)
        let vertices = quad_vertices(scene.plane_aspect);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
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
            bind_group_layouts: &[Some(&texture_layout), Some(&uniform_layout)],
            immediate_size: 0,
        });

        // Render pipeline with alpha blending for seam transition
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stitch_render_pipeline"),
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
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
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

        // Sampler (shared by both planes)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("video_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Per-plane resources
        let left = Self::create_plane_resources(
            device,
            &texture_layout,
            &uniform_layout,
            &sampler,
            input_width,
            input_height,
            input_format,
            "left",
        );
        let right = Self::create_plane_resources(
            device,
            &texture_layout,
            &uniform_layout,
            &sampler,
            input_width,
            input_height,
            input_format,
            "right",
        );

        // Render target (output-sized, RGBA)
        let render_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render_target"),
            size: wgpu::Extent3d {
                width: output_width,
                height: output_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: output_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            pipeline,
            vertex_buffer,
            left,
            right,
            render_target,
            render_target_view,
            output_width,
            output_height,
            input_format,
            texture_layout,
            sampler,
            device: device.clone(),
            flip_180: [false, false],
        }
    }

    fn create_plane_resources(
        device: &wgpu::Device,
        texture_layout: &wgpu::BindGroupLayout,
        uniform_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
        input_format: InputFormat,
        label: &str,
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

        // Y plane is always R8Unorm at full resolution
        let y_texture = create_texture("y", width, height, wgpu::TextureFormat::R8Unorm);

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
                // Dummy V texture — shader won't sample it in NV12 mode
                let v_dummy = create_texture("v_dummy", 1, 1, wgpu::TextureFormat::R8Unorm);
                (uv, v_dummy)
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
            size: std::mem::size_of::<GpuUniforms>() as u64,
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
        upload_yuv(gpu, &self.left, y, u, v)
    }

    /// Upload YUV420P planes to the right camera textures.
    pub fn upload_right_yuv(
        &self,
        gpu: &GpuContext,
        y: &[u8],
        u: &[u8],
        v: &[u8],
    ) -> Result<(), RenderError> {
        upload_yuv(gpu, &self.right, y, u, v)
    }

    /// Upload NV12 planes to the left camera textures.
    ///
    /// Y is R8Unorm at full resolution, UV is Rg8Unorm at half resolution.
    /// Requires the renderer to be initialized with `InputFormat::Nv12`.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "gpu_upload_nv12")
    )]
    pub fn upload_left_nv12(
        &self,
        gpu: &GpuContext,
        y: &[u8],
        uv: &[u8],
    ) -> Result<(), RenderError> {
        debug_assert_eq!(
            self.input_format,
            InputFormat::Nv12,
            "upload_left_nv12 requires InputFormat::Nv12"
        );
        upload_nv12(gpu, &self.left, y, uv)
    }

    /// Upload NV12 planes to the right camera textures.
    pub fn upload_right_nv12(
        &self,
        gpu: &GpuContext,
        y: &[u8],
        uv: &[u8],
    ) -> Result<(), RenderError> {
        debug_assert_eq!(
            self.input_format,
            InputFormat::Nv12,
            "upload_right_nv12 requires InputFormat::Nv12"
        );
        upload_nv12(gpu, &self.right, y, uv)
    }

    /// Create a texture bind group from external textures.
    ///
    /// Used for CUDA/Vulkan zero-copy: pre-build one bind group per
    /// double-buffer slot (before the render loop), then select the active
    /// slot each frame by cloning the appropriate pre-built group (cheap
    /// Arc refcount increment) and passing it to
    /// [`Self::set_left_bind_group`] / [`Self::set_right_bind_group`].
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
        let v_view = self
            .left
            .v_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&uv_view),
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

    /// Set the left plane's texture bind group for the next render.
    pub fn set_left_bind_group(&mut self, bind_group: wgpu::BindGroup) {
        self.left.texture_bind_group = bind_group;
    }

    /// Set the right plane's texture bind group for the next render.
    pub fn set_right_bind_group(&mut self, bind_group: wgpu::BindGroup) {
        self.right.texture_bind_group = bind_group;
    }

    /// Enable 180-degree UV flip for the GPU zero-copy path.
    ///
    /// When set, the shader flips texture coordinates before sampling,
    /// equivalent to the CPU path's buffer reversal for rotated video.
    pub fn set_flip_180(&mut self, left: bool, right: bool) {
        self.flip_180 = [left, right];
    }

    /// Encode the shared stitch render pass: projection, uniforms, and draw calls.
    ///
    /// Returns the command encoder with the render pass already recorded.
    /// Callers handle submission, readback, or further encoding as needed.
    #[allow(clippy::too_many_arguments)]
    fn encode_stitch_pass(
        &self,
        gpu: &GpuContext,
        scene: &SceneGeometry,
        calibration: &MatchCalibration,
        viewport: &ResolvedViewport,
        blend_width: f32,
        target_view: &wgpu::TextureView,
        aspect: f32,
        encoder_label: &str,
    ) -> wgpu::CommandEncoder {
        let projection = opengl_to_wgpu_matrix()
            * Perspective3::new(
                aspect,
                viewport.config.fov_degrees.to_radians(),
                NEAR_PLANE,
                FAR_PLANE,
            )
            .to_homogeneous();
        let view = view_matrix(
            &scene.camera_position,
            viewport.position.yaw,
            viewport.position.pitch,
            viewport.config.rig_tilt,
        );

        let left_mvp = projection * view * scene.model_matrix_left();
        let left_uniforms = build_gpu_uniforms(
            &left_mvp,
            &calibration.left,
            false,
            blend_width,
            self.input_format,
            self.flip_180[0],
        );

        let right_mvp = projection * view * scene.model_matrix_right();
        let right_uniforms = build_gpu_uniforms(
            &right_mvp,
            &calibration.right,
            true,
            blend_width,
            self.input_format,
            self.flip_180[1],
        );

        gpu.queue.write_buffer(
            &self.left.uniform_buffer,
            0,
            bytemuck::bytes_of(&left_uniforms),
        );
        gpu.queue.write_buffer(
            &self.right.uniform_buffer,
            0,
            bytemuck::bytes_of(&right_uniforms),
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(encoder_label),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stitch_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
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

            pass.set_bind_group(0, &self.left.texture_bind_group, &[]);
            pass.set_bind_group(1, &self.left.uniform_bind_group, &[]);
            pass.draw(0..6, 0..1);

            pass.set_bind_group(0, &self.right.texture_bind_group, &[]);
            pass.set_bind_group(1, &self.right.uniform_bind_group, &[]);
            pass.draw(0..6, 0..1);
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
        scene: &SceneGeometry,
        calibration: &MatchCalibration,
        viewport: &ResolvedViewport,
        blend_width: f32,
    ) -> wgpu::CommandBuffer {
        let aspect = self.output_width as f32 / self.output_height as f32;
        let encoder = self.encode_stitch_pass(
            gpu,
            scene,
            calibration,
            viewport,
            blend_width,
            &self.render_target_view,
            aspect,
            "stitch_to_target",
        );
        encoder.finish()
    }

    /// Access the internal render target texture.
    ///
    /// Used by [`Nv12Converter`](crate::nv12_converter::Nv12Converter) to read
    /// the RGBA output without an intermediate CPU copy.
    pub fn render_target(&self) -> &wgpu::Texture {
        &self.render_target
    }

    /// Render a stitched frame directly to a texture view (e.g., a window surface).
    ///
    /// Unlike [`Self::render_to_target`], this does NOT read back the result to CPU.
    /// Used for interactive preview windows.
    pub fn render_to_view(
        &self,
        gpu: &GpuContext,
        scene: &SceneGeometry,
        calibration: &MatchCalibration,
        viewport: &ResolvedViewport,
        blend_width: f32,
        target_view: &wgpu::TextureView,
    ) {
        let aspect = viewport.config.width as f32 / viewport.config.height as f32;
        let encoder = self.encode_stitch_pass(
            gpu,
            scene,
            calibration,
            viewport,
            blend_width,
            target_view,
            aspect,
            "preview_frame",
        );
        gpu.queue.submit(Some(encoder.finish()));
    }
}

// ---- Helper functions ----

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

/// Build the view matrix for the virtual camera.
///
/// Camera sits at `position` and looks at the origin (corner where the two
/// planes meet) by default. This matches v1 Three.js where the OrbitControls
/// target is `[0, 0, 0]`. `yaw` rotates around Y (left/right from center),
/// `pitch` rotates around X (up/down).
fn view_matrix(position: &[f32; 3], yaw: f32, pitch: f32, rig_tilt: f32) -> Matrix4<f32> {
    let eye = Point3::new(position[0], position[1], position[2]);
    // Base direction: eye → origin (the L-shape corner)
    let mut base_forward = -eye.coords.normalize();
    let mut world_up = Vector3::new(0.0, 1.0, 0.0);

    // Camera's base right axis — perpendicular to view in the horizontal plane.
    // This accounts for the camera being at 45° in the XZ plane.
    let base_right = base_forward.cross(&world_up).normalize();

    // Rig tilt: rotate the entire reference frame around the base right axis.
    // This tilts "up" and "forward" so that yaw/pitch operate in the tilted
    // coordinate system. Panning in this tilted frame naturally introduces
    // roll that compensates for edge distortion from a tilted camera rig.
    if rig_tilt.abs() > 1e-6 {
        let tilt_q =
            UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(base_right), rig_tilt);
        base_forward = tilt_q * base_forward;
        world_up = tilt_q * world_up;
    }

    // Yaw: rotate around the (possibly tilted) up axis
    let up_axis = nalgebra::Unit::new_normalize(world_up);
    let yaw_q = UnitQuaternion::from_axis_angle(&up_axis, yaw);
    // Pitch: rotate around the yaw-rotated right axis
    let right = yaw_q * base_right;
    let pitch_q = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(right), pitch);
    let rotation = pitch_q * yaw_q;
    let forward = rotation * base_forward;
    let up = rotation * world_up;
    let target = Point3::from(eye.coords + forward);
    nalgebra::Isometry3::look_at_rh(&eye, &target, &up).to_homogeneous()
}

/// Build the GPU uniform struct for one plane.
///
/// `flip_180`: when true, the shader flips UV coordinates to apply
/// 180-degree rotation. Used by the GPU zero-copy path where the CPU
/// buffer-reversal trick from the software decode path is not possible.
pub(crate) fn build_gpu_uniforms(
    mvp: &Matrix4<f32>,
    camera: &CameraParams,
    is_right: bool,
    blend_width: f32,
    input_format: InputFormat,
    flip_180: bool,
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
            camera.d[0] as f32,
            camera.d[1] as f32,
            camera.d[2] as f32,
            camera.d[3] as f32,
        ],
        color_scale: [1.0, 1.0, 1.0, 0.0], // identity (no color correction yet)
        color_offset_blend: [0.0, 0.0, 0.0, blend_width],
        flags: [
            is_right as u32,
            (input_format == InputFormat::Nv12) as u32,
            flip_180 as u32,
            0,
        ],
    }
}

/// Convert a nalgebra `Matrix4` to column-major `[[f32; 4]; 4]` for wgpu.
pub(crate) fn matrix4_to_columns(m: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let s = m.as_slice();
    [
        [s[0], s[1], s[2], s[3]],
        [s[4], s[5], s[6], s[7]],
        [s[8], s[9], s[10], s[11]],
        [s[12], s[13], s[14], s[15]],
    ]
}

/// OpenGL to wgpu clip space correction: Z from \[-1,1\] to \[0,1\].
///
/// nalgebra's `Perspective3` uses OpenGL conventions. wgpu expects
/// clip space Z in [0, 1], so we apply this correction.
#[rustfmt::skip]
pub(crate) fn opengl_to_wgpu_matrix() -> Matrix4<f32> {
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_are_normalized() {
        let camera = CameraParams {
            width: 3840,
            height: 2160,
            fx: 1796.32,
            fy: 1797.22,
            cx: 1919.37,
            cy: 1063.17,
            d: [0.0342, 0.0677, -0.0741, 0.0299],
        };
        let mvp = Matrix4::identity();
        let u = build_gpu_uniforms(&mvp, &camera, false, 0.0, InputFormat::Yuv420p, false);

        // fx/width ≈ 0.4678
        assert!((u.intrinsics[0] - 1796.32 / 3840.0).abs() < 1e-4);
        // cy/height ≈ 0.4922
        assert!((u.intrinsics[3] - 1063.17 / 2160.0).abs() < 1e-4);
        // is_right = 0, use_nv12 = 0
        assert_eq!(u.flags[0], 0);
        assert_eq!(u.flags[1], 0);
    }

    #[test]
    fn opengl_to_wgpu_maps_z() {
        let m = opengl_to_wgpu_matrix();
        // Point at Z = -1 (OpenGL near) should map to Z = 0 (wgpu near)
        let p = m * nalgebra::Vector4::new(0.0, 0.0, -1.0, 1.0);
        assert!((p.z - (-0.5 + 0.5)).abs() < 1e-5); // -0.5 + 0.5 = 0
        // Point at Z = 1 (OpenGL far) should map to Z = 1 (wgpu far)
        let p = m * nalgebra::Vector4::new(0.0, 0.0, 1.0, 1.0);
        assert!((p.z - 1.0).abs() < 1e-5);
    }
}
