//! GPU renderer for the panoramic stitching pipeline.
//!
//! Manages wgpu render pipelines, textures, and bind groups for rendering
//! two fisheye-corrected camera planes into a stitched panoramic output.
//!
//! ## Pipeline
//!
//! ```text
//! Left RGBA frame  ──► GPU texture ──┐
//!                                    ├──► Render pass ──► RGBA output
//! Right RGBA frame ──► GPU texture ──┘
//! ```
//!
//! Each plane is a textured quad positioned in 3D space (L-shape geometry).
//! The fisheye undistortion and color correction happen in the fragment shader.

use crate::calibration::{CameraParams, MatchCalibration};
use crate::gpu::GpuContext;
use crate::scene::SceneGeometry;
use crate::viewport::ResolvedViewport;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix4, Perspective3, Point3, UnitQuaternion, Vector3};
use thiserror::Error;
use wgpu::util::DeviceExt;

/// Errors from the renderer.
#[derive(Debug, Error)]
pub enum RenderError {
    /// Frame data has wrong size.
    #[error("frame data size mismatch: expected {expected} bytes, got {actual}")]
    FrameSizeMismatch { expected: usize, actual: usize },

    /// Buffer mapping failed.
    #[error("GPU buffer mapping failed")]
    BufferMapFailed,
}

// ---- GPU-side structs ----

/// Uniform buffer layout (must match `Uniforms` in fisheye.wgsl exactly).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuUniforms {
    mvp: [[f32; 4]; 4],
    intrinsics: [f32; 4],
    dist: [f32; 4],
    lab_scale: [f32; 4],
    lab_offset_blend: [f32; 4],
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

/// Generate quad vertices for a plane (1.0 wide, 16:9 aspect).
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

/// Per-plane GPU resources (texture + uniform buffer + bind groups).
struct PlaneResources {
    texture: wgpu::Texture,
    texture_bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

/// The GPU renderer for panoramic stitching.
///
/// Holds all wgpu resources: pipelines, textures, bind groups, and buffers.
/// Created once per pipeline and reused for every frame.
pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    left: PlaneResources,
    right: PlaneResources,
    render_target: wgpu::Texture,
    render_target_view: wgpu::TextureView,
    depth_texture_view: wgpu::TextureView,
    output_buffer: wgpu::Buffer,
    output_width: u32,
    output_height: u32,
}

impl Renderer {
    /// Create a new renderer with all GPU resources.
    ///
    /// Allocates textures, buffers, and compiles the shader pipeline.
    /// This is called once during pipeline initialization.
    pub fn new(
        gpu: &GpuContext,
        output_width: u32,
        output_height: u32,
        input_width: u32,
        input_height: u32,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let device = &gpu.device;

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fisheye"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fisheye.wgsl").into()),
        });

        // Vertex buffer (quad for both planes — same shape, different model matrices)
        let vertices = quad_vertices(16.0 / 9.0);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Bind group layouts
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_layout"),
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
            push_constant_ranges: &[],
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
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
            "left",
        );
        let right = Self::create_plane_resources(
            device,
            &texture_layout,
            &uniform_layout,
            &sampler,
            input_width,
            input_height,
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

        // Depth buffer for correct L-shape occlusion
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width: output_width,
                height: output_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Staging buffer for CPU readback
        let bytes_per_row = align_to_256(output_width * 4);
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_staging"),
            size: (bytes_per_row * output_height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            vertex_buffer,
            left,
            right,
            render_target,
            render_target_view,
            depth_texture_view,
            output_buffer,
            output_width,
            output_height,
        }
    }

    fn create_plane_resources(
        device: &wgpu::Device,
        texture_layout: &wgpu::BindGroupLayout,
        uniform_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
        label: &str,
    ) -> PlaneResources {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("{label}_video")),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}_texture_bg")),
            layout: texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
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
            texture,
            texture_bind_group,
            uniform_buffer,
            uniform_bind_group,
            width,
            height,
        }
    }

    /// Upload an RGBA frame to the left camera texture.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "gpu_upload")
    )]
    pub fn upload_left_frame(&self, gpu: &GpuContext, rgba_data: &[u8]) {
        upload_frame(gpu, &self.left, rgba_data);
    }

    /// Upload an RGBA frame to the right camera texture.
    pub fn upload_right_frame(&self, gpu: &GpuContext, rgba_data: &[u8]) {
        upload_frame(gpu, &self.right, rgba_data);
    }

    /// Render a stitched frame and read back the RGBA result.
    ///
    /// Both camera textures must be uploaded before calling this.
    /// Returns a tightly-packed RGBA buffer of `output_width * output_height * 4` bytes.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "gpu_render")
    )]
    pub fn render_frame(
        &self,
        gpu: &GpuContext,
        scene: &SceneGeometry,
        calibration: &MatchCalibration,
        viewport: &ResolvedViewport,
        blend_width: f32,
    ) -> Result<Vec<u8>, RenderError> {
        let aspect = self.output_width as f32 / self.output_height as f32;
        let projection = opengl_to_wgpu_matrix()
            * Perspective3::new(aspect, viewport.config.fov_degrees.to_radians(), 0.01, 5.0)
                .to_homogeneous();
        let view = view_matrix(
            &scene.camera_position,
            viewport.position.yaw,
            viewport.position.pitch,
        );

        // Build uniforms for both planes
        let left_mvp = projection * view * scene.model_matrix_left();
        let left_uniforms = build_gpu_uniforms(&left_mvp, &calibration.left, false, blend_width);

        let right_mvp = projection * view * scene.model_matrix_right();
        let right_uniforms = build_gpu_uniforms(&right_mvp, &calibration.right, true, blend_width);

        // Write uniform buffers (staged before submission)
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

        // Encode render pass
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stitch_frame"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stitch_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            // Draw left plane (fully opaque base layer)
            pass.set_bind_group(0, &self.left.texture_bind_group, &[]);
            pass.set_bind_group(1, &self.left.uniform_bind_group, &[]);
            pass.draw(0..6, 0..1);

            // Draw right plane (blends over left at seam)
            pass.set_bind_group(0, &self.right.texture_bind_group, &[]);
            pass.set_bind_group(1, &self.right.uniform_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }

        // Copy render target to staging buffer
        let bytes_per_row = align_to_256(self.output_width * 4);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.render_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(self.output_height),
                },
            },
            wgpu::Extent3d {
                width: self.output_width,
                height: self.output_height,
                depth_or_array_layers: 1,
            },
        );

        {
            crate::profile_scope!("gpu_submit");
            gpu.queue.submit(Some(encoder.finish()));
        }

        // Readback: map the staging buffer and copy to CPU
        let output = {
            crate::profile_scope!("gpu_readback");
            let buffer_slice = self.output_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            gpu.device
                .poll(wgpu::PollType::wait())
                .map_err(|_| RenderError::BufferMapFailed)?;
            rx.recv()
                .map_err(|_| RenderError::BufferMapFailed)?
                .map_err(|_| RenderError::BufferMapFailed)?;

            let mapped = buffer_slice.get_mapped_range();

            // Remove row padding (bytes_per_row is aligned to 256)
            let tight_row = self.output_width as usize * 4;
            let padded_row = bytes_per_row as usize;
            let mut output = Vec::with_capacity(tight_row * self.output_height as usize);
            for row in 0..self.output_height as usize {
                let start = row * padded_row;
                output.extend_from_slice(&mapped[start..start + tight_row]);
            }

            drop(mapped);
            self.output_buffer.unmap();
            output
        };

        Ok(output)
    }

    /// Render a stitched frame directly to a texture view (e.g., a window surface).
    ///
    /// Unlike [`Self::render_frame`], this does NOT read back the result to CPU.
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
        let projection = opengl_to_wgpu_matrix()
            * Perspective3::new(aspect, viewport.config.fov_degrees.to_radians(), 0.01, 5.0)
                .to_homogeneous();
        let view = view_matrix(
            &scene.camera_position,
            viewport.position.yaw,
            viewport.position.pitch,
        );

        let left_mvp = projection * view * scene.model_matrix_left();
        let left_uniforms = build_gpu_uniforms(&left_mvp, &calibration.left, false, blend_width);

        let right_mvp = projection * view * scene.model_matrix_right();
        let right_uniforms = build_gpu_uniforms(&right_mvp, &calibration.right, true, blend_width);

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
                label: Some("preview_frame"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("preview_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
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

        gpu.queue.submit(Some(encoder.finish()));
    }

    /// Resize the output dimensions (recreates depth texture).
    ///
    /// Called when the preview window is resized. The render target and
    /// output buffer are only used for CPU readback, so they are left as-is.
    pub fn resize_depth(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        let depth_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_texture_view =
            depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
    }

    /// Width of the output render target.
    pub fn output_width(&self) -> u32 {
        self.output_width
    }

    /// Height of the output render target.
    pub fn output_height(&self) -> u32 {
        self.output_height
    }
}

// ---- Helper functions ----

fn upload_frame(gpu: &GpuContext, plane: &PlaneResources, rgba_data: &[u8]) {
    let expected = (plane.width * plane.height * 4) as usize;
    assert_eq!(
        rgba_data.len(),
        expected,
        "frame data size mismatch: expected {expected} bytes ({}x{}x4), got {}",
        plane.width,
        plane.height,
        rgba_data.len()
    );
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &plane.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(plane.width * 4),
            rows_per_image: Some(plane.height),
        },
        wgpu::Extent3d {
            width: plane.width,
            height: plane.height,
            depth_or_array_layers: 1,
        },
    );
}

/// Build the view matrix for the virtual camera.
///
/// Camera sits at `position` and looks at the origin (corner where the two
/// planes meet) by default. This matches v1 Three.js where the OrbitControls
/// target is `[0, 0, 0]`. `yaw` rotates around Y (left/right from center),
/// `pitch` rotates around X (up/down).
fn view_matrix(position: &[f32; 3], yaw: f32, pitch: f32) -> Matrix4<f32> {
    let eye = Point3::new(position[0], position[1], position[2]);
    // Base direction: eye → origin (the L-shape corner)
    let base_forward = -eye.coords.normalize();
    let world_up = Vector3::new(0.0, 1.0, 0.0);
    // Camera's base right axis — perpendicular to view in the horizontal plane.
    // This accounts for the camera being at 45° in the XZ plane.
    let base_right = base_forward.cross(&world_up).normalize();
    // Yaw: rotate around world Y (horizontal pan)
    let yaw_q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw);
    // Pitch: rotate around the yaw-rotated camera right axis (head nod)
    let right = yaw_q * base_right;
    let pitch_q = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(right), pitch);
    let rotation = pitch_q * yaw_q;
    let forward = rotation * base_forward;
    let up = rotation * world_up;
    let target = Point3::from(eye.coords + forward);
    nalgebra::Isometry3::look_at_rh(&eye, &target, &up).to_homogeneous()
}

/// Build the GPU uniform struct for one plane.
fn build_gpu_uniforms(
    mvp: &Matrix4<f32>,
    camera: &CameraParams,
    is_right: bool,
    blend_width: f32,
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
        lab_scale: [1.0, 1.0, 1.0, 0.0], // identity (no color correction yet)
        lab_offset_blend: [0.0, 0.0, 0.0, blend_width],
        flags: [is_right as u32, 0, 0, 0],
    }
}

/// Convert a nalgebra `Matrix4` to column-major `[[f32; 4]; 4]` for wgpu.
fn matrix4_to_columns(m: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let s = m.as_slice();
    [
        [s[0], s[1], s[2], s[3]],
        [s[4], s[5], s[6], s[7]],
        [s[8], s[9], s[10], s[11]],
        [s[12], s[13], s[14], s[15]],
    ]
}

/// OpenGL→wgpu clip space correction: Z from [-1,1] to [0,1].
///
/// nalgebra's `Perspective3` uses OpenGL conventions. wgpu expects
/// clip space Z in [0, 1], so we apply this correction.
#[rustfmt::skip]
fn opengl_to_wgpu_matrix() -> Matrix4<f32> {
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0,
    )
}

/// Align a value up to the next multiple of 256.
///
/// wgpu requires `bytes_per_row` to be a multiple of 256 for texture↔buffer copies.
fn align_to_256(value: u32) -> u32 {
    (value + 255) & !255
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_to_256_works() {
        assert_eq!(align_to_256(0), 0);
        assert_eq!(align_to_256(1), 256);
        assert_eq!(align_to_256(256), 256);
        assert_eq!(align_to_256(257), 512);
        // 1920 * 4 = 7680, already a multiple of 256
        assert_eq!(align_to_256(7680), 7680);
    }

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
        let u = build_gpu_uniforms(&mvp, &camera, false, 0.0);

        // fx/width ≈ 0.4678
        assert!((u.intrinsics[0] - 1796.32 / 3840.0).abs() < 1e-4);
        // cy/height ≈ 0.4922
        assert!((u.intrinsics[3] - 1063.17 / 2160.0).abs() < 1e-4);
        // is_right = 0
        assert_eq!(u.flags[0], 0);
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
