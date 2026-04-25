//! GPU Bayer demosaic + ISP pipeline for raw camera sensors.
//!
//! Converts raw 10-bit RGGB Bayer data (e.g. IMX477 via direct V4L2)
//! into an RGBA GPU texture that can be copied directly into the stitch
//! pipeline's input plane via [`StitchRenderer::copy_texture_to_left`].
//! No CPU readback in the hot path.

use crate::gpu::GpuContext;
use wgpu::util::DeviceExt;

/// ISP tuning parameters passed to the demosaic compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IspParams {
    pub width: u32,
    pub height: u32,
    pub black_level: f32,
    pub lsc_strength: f32,
    pub wb_r: f32,
    pub wb_g: f32,
    pub wb_b: f32,
    pub white_level: f32,
    pub ccm_row0: [f32; 4],
    pub ccm_row1: [f32; 4],
    pub ccm_row2: [f32; 4],
    pub saturation: f32,
    /// Overall brightness scale (applied in linear space before gamma).
    /// 1.0 = no change. <1.0 darkens, >1.0 brightens.
    /// Replaces the vault's adaptive p99.5 tonemap with a manual knob.
    pub brightness: f32,
    _pad: [f32; 2],
}

impl IspParams {
    /// Default ISP parameters matching the fieldkit preview server recipe.
    ///
    /// Raw values are 16-bit (10-bit left-shifted by 6). The shader
    /// right-shifts internally, so black_level and white_level are in
    /// 10-bit space (0-1023). white_level=400 matches the fieldkit's
    /// proven tonemap white point (not the full sensor range).
    pub fn imx477_default(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            black_level: 64.0,
            lsc_strength: 0.35,
            wb_r: 1.7,
            wb_g: 1.0,
            wb_b: 2.0,
            white_level: 300.0,
            ccm_row0: [1.0, 0.0, 0.0, 0.0],
            ccm_row1: [0.0, 1.0, 0.0, 0.0],
            ccm_row2: [0.0, 0.0, 1.0, 0.0],
            saturation: 1.6,
            brightness: 0.5,
            _pad: [0.0; 2],
        }
    }

    /// Indoor warm LED preset (from the vault's verified pipeline).
    pub fn imx477_indoor(width: u32, height: u32) -> Self {
        Self {
            wb_r: 1.18,
            wb_b: 3.3,
            white_level: 500.0,
            ccm_row0: [1.6, -0.5, -0.1, 0.0],
            ccm_row1: [-0.15, 1.4, -0.25, 0.0],
            ccm_row2: [-0.1, -0.35, 1.45, 0.0],
            saturation: 1.35,
            ..Self::imx477_default(width, height)
        }
    }
}

/// Compute grey-world AWB gains from raw Bayer bytes (RGGB pattern).
///
/// Samples every `stride`-th pixel for speed. Returns `(wb_r, wb_b)`
/// gains that make the average scene neutral grey. Call once per frame
/// (or every N frames) and feed into `IspParams::wb_r / wb_b`.
pub fn compute_awb(raw_bytes: &[u8], width: u32, height: u32, stride: u32) -> (f32, f32) {
    let (mut sum_r, mut sum_g, mut sum_b) = (0u64, 0u64, 0u64);
    let (mut count_r, mut count_g, mut count_b) = (0u32, 0u32, 0u32);

    let w = width as usize;
    for y in (0..height).step_by(stride as usize) {
        for x in (0..width).step_by(stride as usize) {
            let idx = (y as usize * w + x as usize) * 2;
            if idx + 1 >= raw_bytes.len() {
                continue;
            }
            let val = u16::from_le_bytes([raw_bytes[idx], raw_bytes[idx + 1]]) as u64;
            // RGGB: (even_x, even_y)=R, (odd_x, even_y)=Gr, (even_x, odd_y)=Gb, (odd_x, odd_y)=B
            match (x % 2, y % 2) {
                (0, 0) => { sum_r += val; count_r += 1; }
                (1, 1) => { sum_b += val; count_b += 1; }
                _ => { sum_g += val; count_g += 1; }
            }
        }
    }

    if count_r == 0 || count_g == 0 || count_b == 0 {
        return (1.0, 1.0);
    }

    let mean_r = sum_r as f64 / count_r as f64;
    let mean_g = sum_g as f64 / count_g as f64;
    let mean_b = sum_b as f64 / count_b as f64;

    let wb_r = (mean_g / mean_r) as f32;
    let wb_b = (mean_g / mean_b) as f32;

    (wb_r, wb_b)
}

/// GPU pipeline for Bayer demosaic + ISP processing.
///
/// The hot-path method [`process_gpu`](Self::process_gpu) uploads raw
/// Bayer data, dispatches the compute shader, and returns a reference
/// to the GPU-resident RGBA output texture. The caller copies this
/// texture into the stitch pipeline's input plane via
/// [`StitchRenderer::copy_texture_to_left`] - no CPU readback needed.
pub struct BayerDemosaic {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    input_texture: wgpu::Texture,
    output_texture: wgpu::Texture,
    width: u32,
    height: u32,
}

impl BayerDemosaic {
    /// Create a new Bayer demosaic pipeline for the given frame dimensions.
    pub fn new(gpu: &GpuContext, width: u32, height: u32, params: &IspParams) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bayer_demosaic"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/bayer_demosaic.wgsl").into(),
            ),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bayer_demosaic_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
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

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bayer_demosaic_layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bayer_demosaic_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("isp_params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let input_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bayer_input"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("demosaic_output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        Self {
            pipeline,
            bind_group_layout,
            params_buffer,
            input_texture,
            output_texture,
            width,
            height,
        }
    }

    /// Update ISP parameters (WB, CCM, etc.) without rebuilding the pipeline.
    pub fn update_params(&self, gpu: &GpuContext, params: &IspParams) {
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload raw Bayer data and encode the demosaic compute pass.
    ///
    /// `raw_bytes` is the raw MMAP buffer: `width * height * 2` bytes
    /// of little-endian u16 (R16Uint). Uploaded directly with no
    /// per-pixel conversion.
    ///
    /// Does NOT submit - the caller batches this with the stitch render
    /// into a single GPU submission for maximum throughput.
    pub fn encode_demosaic(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        raw_bytes: &[u8],
    ) {
        debug_assert_eq!(
            raw_bytes.len(),
            (self.width * self.height * 2) as usize,
        );

        {
            crate::profile_scope!("bayer_upload");
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.input_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                raw_bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.width * 2),
                    rows_per_image: Some(self.height),
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        crate::profile_scope!("bayer_compute");
        let input_view = self
            .input_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let output_view = self
            .output_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bayer_demosaic_bg"),
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

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bayer_demosaic"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                (self.width + 15) / 16,
                (self.height + 15) / 16,
                1,
            );
        }
    }

    /// Output texture reference (for use with renderer copy methods).
    pub fn output_texture(&self) -> &wgpu::Texture {
        &self.output_texture
    }
}
