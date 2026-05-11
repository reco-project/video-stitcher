//! wgpu compute shader for detection preprocessing.
//!
//! Converts NV12 texture views into a float32 CHW tensor ready for
//! ORT inference. Runs entirely on the GPU via a single compute
//! dispatch, replacing the CPU scalar bilinear resize (~300ms at
//! 1280x1280) with a GPU pass (~1ms).
//!
//! Works on any wgpu backend (DX12, Vulkan, Metal) making it the
//! universal preprocessing path for DirectML, CUDA EP, and CoreML.

use std::borrow::Cow;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    pad_x: f32,
    pad_y: f32,
    scale: f32,
    flip180: u32,
}

const SHADER: &str = r#"
struct Uniforms {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    pad_x: f32,
    pad_y: f32,
    scale: f32,
    flip180: u32,
}

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var uv_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;
@group(0) @binding(4) var bilinear_sampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    if (ox >= u.dst_w || oy >= u.dst_h) {
        return;
    }

    let hw = u.dst_w * u.dst_h;
    let idx = oy * u.dst_w + ox;

    // Letterbox padding region
    let pad_x_u = u32(u.pad_x);
    let pad_y_u = u32(u.pad_y);
    let content_w = u.dst_w - 2u * pad_x_u;
    let content_h = u.dst_h - 2u * pad_y_u;

    if (ox < pad_x_u || ox >= pad_x_u + content_w ||
        oy < pad_y_u || oy >= pad_y_u + content_h) {
        let grey = 0.44705883;
        output[idx] = grey;
        output[idx + hw] = grey;
        output[idx + 2u * hw] = grey;
        return;
    }

    // Map to source coordinates
    var src_x = f32(ox - pad_x_u) / u.scale;
    var src_y = f32(oy - pad_y_u) / u.scale;

    if (u.flip180 != 0u) {
        src_x = f32(u.src_w - 1u) - src_x;
        src_y = f32(u.src_h - 1u) - src_y;
    }

    // Normalized coords for bilinear sampling
    let tex_x = (src_x + 0.5) / f32(u.src_w);
    let tex_y = (src_y + 0.5) / f32(u.src_h);
    let uv_coord = vec2<f32>(tex_x, tex_y);

    // Sample NV12 planes
    let y_val = textureSampleLevel(y_tex, bilinear_sampler, uv_coord, 0.0).r;
    let uv_val = textureSampleLevel(uv_tex, bilinear_sampler, uv_coord, 0.0).rg;

    // BT.709 full-range YUV to RGB
    let cb = uv_val.r - 0.5;
    let cr = uv_val.g - 0.5;
    let r = clamp(y_val + 1.5748 * cr, 0.0, 1.0);
    let g = clamp(y_val - 0.1873 * cb - 0.4681 * cr, 0.0, 1.0);
    let b = clamp(y_val + 1.8556 * cb, 0.0, 1.0);

    // CHW layout, already [0,1] normalized
    output[idx] = r;
    output[idx + hw] = g;
    output[idx + 2u * hw] = b;
}
"#;

/// GPU preprocessor for NV12 → float32 CHW tensor conversion.
pub struct WgpuPreprocessor {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    input_size: u32,
    tensor_bytes: usize,
}

impl WgpuPreprocessor {
    /// Create a preprocessor for the given model input size and frame dimensions.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_size: u32,
        frame_width: u32,
        frame_height: u32,
    ) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("detect_preprocess"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("detect_preprocess_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("detect_preprocess_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("detect_preprocess_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("detect_bilinear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let tensor_bytes = (input_size * input_size * 3) as usize * 4;

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("detect_output"),
            size: tensor_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("detect_staging"),
            size: tensor_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fw = frame_width as f32;
        let fh = frame_height as f32;
        let is = input_size as f32;
        let scale = (is / fw).min(is / fh);
        let new_w = (fw * scale).round();
        let new_h = (fh * scale).round();

        let uniforms = Uniforms {
            src_w: frame_width,
            src_h: frame_height,
            dst_w: input_size,
            dst_h: input_size,
            pad_x: (is - new_w) / 2.0,
            pad_y: (is - new_h) / 2.0,
            scale,
            flip180: 0,
        };

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("detect_uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        log::info!(
            "WgpuPreprocessor: {}x{} -> {}x{}, scale={:.3}, pad=({:.1},{:.1}), tensor={:.1}MB",
            frame_width,
            frame_height,
            input_size,
            input_size,
            scale,
            uniforms.pad_x,
            uniforms.pad_y,
            tensor_bytes as f64 / 1024.0 / 1024.0,
        );

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            output_buffer,
            staging_buffer,
            uniform_buffer,
            input_size,
            tensor_bytes,
        }
    }

    /// Preprocess NV12 views into a float32 CHW tensor on CPU.
    ///
    /// Takes Y (R8Unorm) and UV (Rg8Unorm) plane views, runs the compute
    /// shader, reads back the CHW float32 data.
    pub fn preprocess(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        y_view: &wgpu::TextureView,
        uv_view: &wgpu::TextureView,
        rotation: i32,
    ) -> Vec<f32> {
        let flip: u32 = if rotation == 180 { 1 } else { 0 };
        queue.write_buffer(
            &self.uniform_buffer,
            std::mem::offset_of!(Uniforms, flip180) as u64,
            bytemuck::bytes_of(&flip),
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("detect_preprocess_bg"),
            layout: &self.bind_group_layout,
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
                    resource: self.output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("detect_preprocess"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                self.input_size.div_ceil(16),
                self.input_size.div_ceil(16),
                1,
            );
        }
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.tensor_bytes as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let floats: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();
        floats
    }

    /// Model input size (square dimension).
    pub fn input_size(&self) -> u32 {
        self.input_size
    }
}
