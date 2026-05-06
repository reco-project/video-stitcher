//! Universal GPU color grading pass.
//!
//! Applies brightness, saturation, and gamma to any RGBA texture.
//! Designed to slot into any input pipeline (Bayer demosaic, NV12
//! decode, file playback, OBS BGRA). Future: 1D LUT, tone mapping.

use super::GpuContext;
use wgpu::util::DeviceExt;

/// Parameters for the color grade compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorGradeParams {
    pub brightness: f32,
    pub saturation: f32,
    /// Gamma exponent (0.5 = sqrt, 1.0 = linear, 0.4545 = sRGB approx).
    pub gamma: f32,
    _pad: f32,
}

impl Default for ColorGradeParams {
    fn default() -> Self {
        Self {
            brightness: 1.0,
            saturation: 1.0,
            gamma: 1.0,
            _pad: 0.0,
        }
    }
}

impl ColorGradeParams {
    /// Check if all parameters are identity (no-op).
    /// When true, the grade pass can be skipped entirely.
    pub fn is_identity(&self) -> bool {
        (self.brightness - 1.0).abs() < 1e-6
            && (self.saturation - 1.0).abs() < 1e-6
            && (self.gamma - 1.0).abs() < 1e-6
    }
}

/// GPU compute pass for universal color grading.
///
/// Operates on caller-provided input/output textures (no owned
/// framebuffers). When [`ColorGradeParams::is_identity`] returns
/// true, [`encode`](Self::encode) is a no-op - the caller should
/// use the input texture directly instead of the output.
pub struct ColorGradePass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    identity: bool,
}

impl ColorGradePass {
    /// Create a new color grade pipeline.
    pub fn new(gpu: &GpuContext, params: &ColorGradeParams) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("color_grade"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/color_grade.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("color_grade_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("color_grade_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("color_grade_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("color_grade_params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            bind_group_layout,
            params_buffer,
            identity: params.is_identity(),
        }
    }

    /// Update parameters without rebuilding the pipeline.
    pub fn update_params(&mut self, gpu: &GpuContext, params: &ColorGradeParams) {
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
        self.identity = params.is_identity();
    }

    /// Whether the current parameters are identity (no-op).
    pub fn is_identity(&self) -> bool {
        self.identity
    }

    /// Encode the color grade dispatch into `encoder`.
    ///
    /// `input` must be `Rgba8Unorm` with `TEXTURE_BINDING` usage.
    /// `output` must be `Rgba8Unorm` with `STORAGE_BINDING` usage.
    /// When [`is_identity`](Self::is_identity) returns true, this is
    /// a no-op and the caller should use `input` directly.
    pub fn encode(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
    ) {
        if self.identity {
            return;
        }

        let input_view = input.create_view(&wgpu::TextureViewDescriptor::default());
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("color_grade_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let size = input.size();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("color_grade"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(size.width.div_ceil(16), size.height.div_ceil(16), 1);
    }
}
