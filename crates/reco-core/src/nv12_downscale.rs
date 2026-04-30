//! GPU-side NV12 downscale for detection readback.
//!
//! Downscales a full-resolution NV12 staging texture to detection
//! resolution using a compute shader with hardware bilinear filtering.
//! Cuts readback bandwidth by ~17x (5K: 24MB -> 1.4MB per camera).

use crate::gpu::GpuContext;

/// Detection-resolution NV12 downscaler.
///
/// Created once per session. Call [`downscale_and_readback`] on detection
/// frames to get downscaled NV12 data for the detector.
pub struct Nv12Downscaler {
    pipeline: wgpu::ComputePipeline,
    sampler: wgpu::Sampler,
    params_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    out_width: u32,
    out_height: u32,
    output_size_bytes: u64,
}

impl Nv12Downscaler {
    /// Create a downscaler that produces `out_width x out_height` NV12 output.
    pub fn new(gpu: &GpuContext, out_width: u32, out_height: u32) -> Self {
        let shader = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("nv12_downscale"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/nv12_downscale.wgsl").into(),
                ),
            });

        let bind_group_layout =
            gpu.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("nv12_downscale_bgl"),
                    entries: &[
                        // Y texture
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
                        // UV texture
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
                        // Sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // Output buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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

        let pipeline_layout =
            gpu.device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("nv12_downscale_pl"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = gpu
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("nv12_downscale"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nv12_downscale_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // NV12 size: Y = w*h, UV = w*(h/2)
        let y_size = (out_width * out_height) as u64;
        let uv_size = (out_width * out_height / 2) as u64;
        let output_size_bytes = y_size + uv_size;
        // Round up to 4-byte alignment for the u32 array
        let buffer_size = output_size_bytes.div_ceil(4) * 4;

        let output_buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("nv12_downscale_output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("nv12_downscale_readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("nv12_downscale_params"),
            size: 8, // 2x u32
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        gpu.queue().write_buffer(
            &params_buffer,
            0,
            bytemuck::cast_slice(&[out_width, out_height]),
        );

        log::info!(
            "NV12 downscaler: {}x{} output ({:.1} KB NV12)",
            out_width,
            out_height,
            output_size_bytes as f64 / 1024.0,
        );

        Self {
            pipeline,
            sampler,
            params_buffer,
            output_buffer,
            readback_buffer,
            bind_group_layout,
            out_width,
            out_height,
            output_size_bytes,
        }
    }

    /// Downscale an NV12 staging texture and read back the result.
    ///
    /// `y_view` and `uv_view` are the Y (R8Unorm) and UV (Rg8Unorm)
    /// plane views of the full-resolution NV12 staging texture.
    ///
    /// Returns `(y_data, uv_data)` at the downscaled resolution.
    pub fn downscale_and_readback(
        &self,
        gpu: &GpuContext,
        y_view: &wgpu::TextureView,
        uv_view: &wgpu::TextureView,
    ) -> (Vec<u8>, Vec<u8>) {
        let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nv12_downscale_bg"),
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
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nv12_downscale"),
            });

        // Clear the output buffer (threads only write their lane, rest must be 0).
        encoder.clear_buffer(&self.output_buffer, 0, None);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nv12_downscale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = self.out_width.div_ceil(4);
            pass.dispatch_workgroups(groups_x.div_ceil(64), self.out_height, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.readback_buffer,
            0,
            self.output_size_bytes.div_ceil(4) * 4,
        );

        gpu.queue().submit(std::iter::once(encoder.finish()));

        // Map and read back.
        let slice = self.readback_buffer.slice(..self.output_size_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = gpu.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let w = self.out_width as usize;
        let h = self.out_height as usize;
        let y_size = w * h;
        let uv_size = w * h / 2;

        let y_data = data[..y_size].to_vec();
        let uv_data = data[y_size..y_size + uv_size].to_vec();

        drop(data);
        self.readback_buffer.unmap();

        (y_data, uv_data)
    }

    /// Output width.
    pub fn width(&self) -> u32 {
        self.out_width
    }

    /// Output height.
    pub fn height(&self) -> u32 {
        self.out_height
    }
}
