//! GPU RGBA→NV12 format converter.
//!
//! Converts the RGBA render target to NV12 on the GPU using a compute shader,
//! then reads back the smaller NV12 buffer to the CPU. This eliminates
//! CPU-side swscale (which was the main encode bottleneck) and reduces
//! GPU→CPU readback bandwidth by 2.7× (NV12 is 1.5 bpp vs RGBA 4 bpp).
//!
//! ## Why not convert in the fragment shader?
//!
//! The render target must remain RGBA for post-stitch plugins (overlays,
//! color grading). The NV12 conversion happens *after* plugins, at the
//! encode boundary — invisible to plugin authors.
//!
//! ## Performance
//!
//! On an RTX 5070 at 1080p, this replaces:
//! - CPU swscale RGBA→NV12: ~2.5ms
//! - GPU readback of RGBA (8.3 MB): ~0.78ms
//!
//! With:
//! - GPU compute shader: ~0.1ms
//! - GPU readback of NV12 (3.1 MB): ~0.29ms

use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Uniform parameters for the RGBA→NV12 compute shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Nv12Params {
    width: u32,
    height: u32,
}

/// GPU-accelerated RGBA → NV12 converter.
///
/// Created once per pipeline alongside the [`Renderer`](crate::renderer::Renderer).
/// Call [`convert_and_readback`](Self::convert_and_readback) after rendering
/// each frame to get NV12 data ready for the encoder.
pub struct Nv12Converter {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    nv12_gpu_buffer: wgpu::Buffer,
    nv12_staging_buffer: wgpu::Buffer,
    /// Cached bind group for the current render target. Avoids per-frame
    /// descriptor pool allocation which causes OOM on Vulkan (wgpu#7525).
    /// Stores a raw pointer to the texture for identity comparison (never dereferenced).
    cached_bind_group: Option<(*const wgpu::Texture, wgpu::BindGroup)>,
    /// Reusable readback buffer (avoids 3 MB allocation per frame at 1080p).
    readback_buffer: Vec<u8>,
    /// Reusable channel for map_async signaling (avoids per-frame channel alloc).
    map_tx: std::sync::mpsc::SyncSender<Result<(), wgpu::BufferAsyncError>>,
    map_rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    width: u32,
    height: u32,
    /// Dispatch dimensions for the compute shader.
    dispatch_x: u32,
    dispatch_y: u32,
}

impl Nv12Converter {
    /// Create a new NV12 converter for the given output dimensions.
    ///
    /// Returns an error if `width` is not divisible by 4 or `height` is not even.
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Result<Self, Nv12Error> {
        if !width.is_multiple_of(4) {
            return Err(Nv12Error::InvalidDimensions(format!(
                "width must be divisible by 4, got {width}"
            )));
        }
        if !height.is_multiple_of(2) {
            return Err(Nv12Error::InvalidDimensions(format!(
                "height must be even, got {height}"
            )));
        }

        let device = &gpu.device;

        // Compile compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rgba_to_nv12"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rgba_to_nv12.wgsl").into()),
        });

        // Bind group layout: input texture + output buffer + params
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nv12_bind_group_layout"),
            entries: &[
                // @binding(0): input RGBA texture
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
                // @binding(1): output NV12 storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2): params uniform (width, height)
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
            label: Some("nv12_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nv12_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // NV12 buffer size: Y plane (w*h) + UV plane (w*h/2) = w*h*3/2 bytes
        let nv12_bytes = width as u64 * height as u64 * 3 / 2;
        // GPU storage buffer for compute shader output
        let nv12_gpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nv12_gpu"),
            size: nv12_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // CPU-readable staging buffer for readback
        let nv12_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nv12_staging"),
            size: nv12_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Uniform buffer with width/height
        let params = Nv12Params { width, height };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nv12_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Dispatch: one thread per 4 horizontal pixels, one thread per row
        // Workgroup size is (16, 4), so dispatch groups are:
        let dispatch_x = (width / 4).div_ceil(16);
        let dispatch_y = height.div_ceil(4);

        log::info!(
            "NV12 converter: {}x{} → {} bytes ({:.1} MB), dispatch ({}, {}, 1)",
            width,
            height,
            nv12_bytes,
            nv12_bytes as f64 / 1_048_576.0,
            dispatch_x,
            dispatch_y,
        );

        let (map_tx, map_rx) = std::sync::mpsc::sync_channel(1);

        Ok(Self {
            pipeline,
            bind_group_layout,
            params_buffer,
            nv12_gpu_buffer,
            nv12_staging_buffer,
            cached_bind_group: None,
            readback_buffer: vec![0u8; nv12_bytes as usize],
            map_tx,
            map_rx,
            width,
            height,
            dispatch_x,
            dispatch_y,
        })
    }

    /// Convert the RGBA render target to NV12 and read back to CPU.
    ///
    /// `render_commands` is the command buffer from the preceding render pass.
    /// It is submitted together with the compute shader in a single
    /// `queue.submit` call to guarantee correct GPU synchronization.
    ///
    /// Returns a borrowed slice of tightly-packed NV12 data:
    /// `[Y plane: width*height bytes] [UV plane: width*height/2 bytes]`
    ///
    /// The slice is valid until the next call to this method.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "nv12_convert_readback")
    )]
    pub fn convert_and_readback(
        &mut self,
        gpu: &GpuContext,
        render_target: &wgpu::Texture,
        render_commands: wgpu::CommandBuffer,
    ) -> Result<&[u8], Nv12Error> {
        // Cache the bind group to avoid per-frame descriptor pool allocation,
        // which causes OOM on the Vulkan backend (wgpu#7525). Rebuild only
        // if the render target texture changes.
        let texture_ptr: *const wgpu::Texture = render_target;
        let needs_rebuild = self
            .cached_bind_group
            .as_ref()
            .is_none_or(|(ptr, _)| *ptr != texture_ptr);

        if needs_rebuild {
            let render_target_view =
                render_target.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nv12_bind_group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&render_target_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.nv12_gpu_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });
            self.cached_bind_group = Some((texture_ptr, bind_group));
        }

        let (_, bind_group) = self.cached_bind_group.as_ref().unwrap();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nv12_encoder"),
            });

        // Dispatch compute shader
        {
            crate::profile_scope!("nv12_compute");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nv12_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(self.dispatch_x, self.dispatch_y, 1);
        }

        // Copy GPU buffer to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.nv12_gpu_buffer,
            0,
            &self.nv12_staging_buffer,
            0,
            self.nv12_gpu_buffer.size(),
        );

        {
            crate::profile_scope!("nv12_submit");
            // Submit render + compute together for correct synchronization
            gpu.queue.submit([render_commands, encoder.finish()]);
        }

        // Readback from staging buffer into reusable buffer
        {
            crate::profile_scope!("nv12_readback");
            let buffer_slice = self.nv12_staging_buffer.slice(..);
            let tx = self.map_tx.clone();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            gpu.device
                .poll(wgpu::PollType::wait())
                .map_err(|_| Nv12Error::BufferMapFailed)?;
            self.map_rx
                .recv()
                .map_err(|_| Nv12Error::BufferMapFailed)?
                .map_err(|_| Nv12Error::BufferMapFailed)?;

            let mapped = buffer_slice.get_mapped_range();
            self.readback_buffer.copy_from_slice(&mapped);
            drop(mapped);
            self.nv12_staging_buffer.unmap();
        }

        Ok(&self.readback_buffer)
    }

    /// Output width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Output height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Errors from the NV12 converter.
#[derive(Debug, thiserror::Error)]
pub enum Nv12Error {
    /// GPU buffer mapping failed.
    #[error("NV12 buffer mapping failed")]
    BufferMapFailed,

    /// Invalid output dimensions.
    #[error("invalid NV12 dimensions: {0}")]
    InvalidDimensions(String),
}
