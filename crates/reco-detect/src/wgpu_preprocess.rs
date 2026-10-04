//! wgpu compute shader for detection preprocessing.
//!
//! Converts NV12 texture views into a float32 CHW tensor ready for
//! host-input inference. A single GPU dispatch converts and letterboxes the source; only the model-sized tensor is read back. Source dimensions may change between calls without rebuilding the pipeline or tensor buffers.

use crate::letterbox::Letterbox;
use reco_core::detect::detector::DetectorError;
use std::borrow::Cow;

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    pad_x: u32,
    pad_y: u32,
    content_w: u32,
    content_h: u32,
    scale: f32,
    flip180: u32,
}

const SHADER: &str = r#"
struct Uniforms {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    pad_x: u32,
    pad_y: u32,
    content_w: u32,
    content_h: u32,
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
    if (ox < u.pad_x || ox >= u.pad_x + u.content_w ||
        oy < u.pad_y || oy >= u.pad_y + u.content_h) {
        let grey = 0.44705883;
        output[idx] = grey;
        output[idx + hw] = grey;
        output[idx + 2u * hw] = grey;
        return;
    }

    // Map to source coordinates
    var src_x = f32(ox - u.pad_x) / u.scale;
    var src_y = f32(oy - u.pad_y) / u.scale;

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
    tensor: Vec<f32>,
}

impl WgpuPreprocessor {
    /// Create reusable GPU and host tensor storage for a square model input.
    /// Returns an error for zero or device-incompatible tensor dimensions.
    pub fn new(device: &wgpu::Device, input_size: u32) -> Result<Self, DetectorError> {
        let tensor_bytes = u64::from(input_size)
            .checked_mul(u64::from(input_size))
            .and_then(|pixels| pixels.checked_mul(12))
            .filter(|&bytes| {
                bytes != 0
                    && bytes <= u64::from(device.limits().max_storage_buffer_binding_size)
                    && bytes <= device.limits().max_buffer_size
                    && input_size.div_ceil(16)
                        <= device.limits().max_compute_workgroups_per_dimension
            })
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| {
                DetectorError::InferenceFailed(
                    "model tensor exceeds device limits or is empty".into(),
                )
            })?;
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

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("detect_uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            pipeline,
            bind_group_layout,
            sampler,
            output_buffer,
            staging_buffer,
            uniform_buffer,
            input_size,
            tensor_bytes,
            tensor: vec![0.0; tensor_bytes / 4],
        })
    }

    /// Preprocess NV12 views into a float32 CHW tensor on CPU.
    ///
    /// Accepts full Y/UV views of NV12 or P010 planes on `device`, using the supplied source size and 0/180-degree rotation. Plane aspects and device identity are the caller's responsibility. The returned slice borrows reusable storage until the next call. Layout and readback failures return an error before inference.
    pub fn preprocess(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (y_view, uv_view): (&wgpu::TextureView, &wgpu::TextureView),
        size: (u32, u32),
        rotation: i32,
    ) -> Result<&[f32], DetectorError> {
        let layout = Letterbox::new(size, self.input_size)?;
        let (y, uv) = (y_view.texture(), uv_view.texture());
        use wgpu::TextureFormat as F;
        let uv_size = match (y.format(), uv.format()) {
            (F::R8Unorm, F::Rg8Unorm) | (F::R16Unorm, F::Rg16Unorm) => {
                Some((size.0 / 2, size.1 / 2))
            }
            (F::NV12, F::NV12) | (F::P010, F::P010) if y == uv => Some(size),
            _ => None,
        };
        if !size.0.is_multiple_of(2)
            || !size.1.is_multiple_of(2)
            || !matches!(rotation, 0 | 180)
            || (y.width(), y.height()) != size
            || uv_size != Some((uv.width(), uv.height()))
            || [y, uv].iter().any(|t| {
                t.dimension() != wgpu::TextureDimension::D2
                    || t.depth_or_array_layers() != 1
                    || t.mip_level_count() != 1
                    || t.sample_count() != 1
                    || !t.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
            })
        {
            return Err(DetectorError::InferenceFailed(
                "invalid NV12/P010 source layout or rotation".into(),
            ));
        }
        let uniforms = Uniforms {
            src_w: size.0,
            src_h: size.1,
            dst_w: self.input_size,
            dst_h: self.input_size,
            pad_x: layout.pad_x,
            pad_y: layout.pad_y,
            content_w: layout.width,
            content_h: layout.height,
            scale: layout.scale,
            flip180: u32::from(rotation == 180),
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

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
        let mapped = device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| e.to_string())
            .and_then(|_| rx.recv().map_err(|e| e.to_string()))
            .and_then(|result| result.map_err(|e| e.to_string()));
        if let Err(error) = mapped {
            self.staging_buffer.unmap();
            return Err(DetectorError::InferenceFailed(format!(
                "wgpu tensor readback: {error}"
            )));
        }

        let data = slice.get_mapped_range();
        self.tensor.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        self.staging_buffer.unmap();
        Ok(&self.tensor)
    }

    /// Model input size (square dimension).
    pub fn input_size(&self) -> u32 {
        self.input_size
    }
}
