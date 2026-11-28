//! GPU conversion from the stitch render target to NV12 or P010.
//!
//! Both formats share one triple-buffered readback ring. Readback overlaps
//! the current submission with the frame from two submissions ago, and
//! waits if that older GPU work has yet to finish.

use super::GpuContext;
use crate::{render::color::YuvColor, sink::DeliveryFormat};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Uniform dimensions shared by both compute shaders.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ConvertParams {
    width: u32,
    height: u32,
}

/// GPU-accelerated NV12 or P010 converter with triple-buffered readback.
///
/// Created once per pipeline alongside the internal renderer.
/// Call [`convert_and_readback`](Self::convert_and_readback) after rendering
/// each frame to get YUV data ready for the encoder.
///
/// Uses three staging buffers so the CPU reads from two frames ago.
/// Readback can still block when GPU work has not completed.
///
/// The write slot cycles 0 -> 1 -> 2 -> 0. The read slot is always
/// `(write_slot + 1) % 3`, which was written 2 frames ago.
///
/// Note: returns `None` on the first two calls (2-frame latency). Call
/// [`flush_pending`](Self::flush_pending) after the frame loop to get the
/// last two frames.
pub struct YuvConverter {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    /// Triple-buffered staging buffers for pipelined readback.
    staging_buffers: [wgpu::Buffer; 3],
    /// Which staging buffer to write to next (0, 1, or 2).
    current_slot: usize,
    /// Number of frames pending readback (0, 1, or 2).
    pending_count: u8,
    /// Cached bind group for the current render target. Avoids per-frame
    /// descriptor pool allocation which causes OOM on Vulkan (wgpu#7525).
    ///
    /// Retain the texture handle and compare GPU resource identity, not the
    /// address of a caller-owned handle that can be moved or replaced.
    cached_bind_group: Option<(wgpu::Texture, wgpu::BindGroup)>,
    /// Reuse host storage when staging buffers are mapped.
    readback_buffers: [Vec<u8>; 3],
    /// Reusable channel for map_async signaling (avoids per-frame channel alloc).
    map_tx: std::sync::mpsc::SyncSender<Result<(), wgpu::BufferAsyncError>>,
    map_rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    width: u32,
    height: u32,
    format: DeliveryFormat,
    /// Dispatch dimensions for the compute shader.
    dispatch_x: u32,
    dispatch_y: u32,
}

impl YuvConverter {
    /// Create a new NV12 converter for the given output dimensions.
    ///
    /// Dimensions must be nonzero, width divisible by four and height even. Image and buffer sizes must fit the device's texture, allocation and storage-binding limits.
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Result<Self, YuvConvertError> {
        Self::with_color(gpu, width, height, Default::default())
    }

    /// Create an 8-bit NV12 converter with the specified matrix and range.
    pub fn with_color(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        color: crate::render::color::YuvColor,
    ) -> Result<Self, YuvConvertError> {
        Self::with_format(gpu, width, height, DeliveryFormat::Nv12, color)
    }

    /// Create a converter with the given delivery format, color matrix and range.
    ///
    /// P010 requires an even, nonzero viewport and an `Rgb10a2Unorm` source.
    /// NV12 supports cropping an 8-bit source to a width divisible by four and an even height.
    pub fn with_format(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        format: DeliveryFormat,
        color: YuvColor,
    ) -> Result<Self, YuvConvertError> {
        let frame_len = Self::validate_layout(gpu, width, height, format)?;
        let frame_bytes = frame_len as u64;
        let device = &gpu.device;

        let (shader_name, source) = match format {
            DeliveryFormat::Nv12 => ("rgba_to_nv12", include_str!("../shaders/rgba_to_nv12.wgsl")),
            DeliveryFormat::P010 => ("rgba_to_p010", include_str!("../shaders/rgba_to_p010.wgsl")),
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_name),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        // Bind group layout: input texture + output buffer + params
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yuv_bind_group_layout"),
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
                // @binding(1): output YUV storage buffer
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
            label: Some("yuv_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let ([kr, kg, kb], [cb_r, cb_g, _], [_, cr_g, cr_b]) = color.encode_coefficients();
        let (y_offset, y_scale, c_scale) = match format {
            DeliveryFormat::Nv12 => color.encode_range(),
            DeliveryFormat::P010 if color.full_range => (0.0, 1023.0, 1023.0),
            DeliveryFormat::P010 => (64.0, 876.0, 896.0),
        };
        let constants = [
            ("kr", f64::from(kr)),
            ("kg", f64::from(kg)),
            ("kb", f64::from(kb)),
            ("cb_r", f64::from(-cb_r)),
            ("cb_g", f64::from(-cb_g)),
            ("cr_g", f64::from(-cr_g)),
            ("cr_b", f64::from(-cr_b)),
            ("y_offset", f64::from(y_offset)),
            ("y_scale", f64::from(y_scale)),
            ("c_scale", f64::from(c_scale)),
        ];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yuv_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });

        // GPU storage buffer for compute shader output
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yuv_gpu"),
            size: frame_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Triple-buffered CPU-readable staging buffers for pipelined readback.
        // Older submissions can finish while newer frames are being rendered.
        let staging_buffers = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("yuv_staging_0"),
                size: frame_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("yuv_staging_1"),
                size: frame_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("yuv_staging_2"),
                size: frame_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        ];

        // Uniform buffer with width/height
        let params = ConvertParams { width, height };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yuv_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let (blocks_x, blocks_y) = match format {
            DeliveryFormat::Nv12 => (width / 4, height),
            DeliveryFormat::P010 => (width / 2, height / 2),
        };
        let dispatch_x = blocks_x.div_ceil(16);
        let dispatch_y = blocks_y.div_ceil(4);

        log::debug!(
            "{format:?} converter: {}x{} → {} bytes ({:.1} MB), dispatch ({}, {}, 1)",
            width,
            height,
            frame_bytes,
            frame_bytes as f64 / 1_048_576.0,
            dispatch_x,
            dispatch_y,
        );

        let (map_tx, map_rx) = std::sync::mpsc::sync_channel(1);

        Ok(Self {
            pipeline,
            bind_group_layout,
            params_buffer,
            output_buffer,
            staging_buffers,
            current_slot: 0,
            pending_count: 0,
            cached_bind_group: None,
            readback_buffers: [
                vec![0u8; frame_len],
                vec![0u8; frame_len],
                vec![0u8; frame_len],
            ],
            map_tx,
            map_rx,
            width,
            height,
            format,
            dispatch_x,
            dispatch_y,
        })
    }

    pub(crate) fn validate_layout(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        format: DeliveryFormat,
    ) -> Result<usize, YuvConvertError> {
        let limits = gpu.device.limits();
        format.frame_bytes(width, height)
            .filter(|_| format != DeliveryFormat::Nv12 || width.is_multiple_of(4))
            .filter(|&size| {
                width <= limits.max_texture_dimension_2d
                    && height <= limits.max_texture_dimension_2d
                    && size as u64 <= limits.max_buffer_size
                    && size as u64 <= u64::from(limits.max_storage_buffer_binding_size)
            })
            .ok_or_else(|| YuvConvertError::InvalidDimensions(format!(
                "{format:?} layout {width}x{height} must have nonzero, even dimensions and fit device and host limits (NV12 width must be divisible by 4)"
            )))
    }

    /// Convert the RGB render target and read back packed NV12 or P010 bytes.
    ///
    /// Uses triple-buffered staging: this frame's GPU work writes to the
    /// current slot, while the CPU polls the slot from two frames ago.
    /// It waits if that earlier submission is still in flight.
    ///
    /// `render_commands` is the command buffer from the preceding render pass.
    /// It is submitted together with the compute shader in a single
    /// `queue.submit` call to guarantee correct GPU synchronization.
    ///
    /// Returns `None` on the first two calls (GPU work is submitted but no
    /// frame from 2 frames ago is available yet). From the third call onward,
    /// returns data from 2 frames ago as a borrowed slice.
    ///
    /// Call [`flush_pending`](Self::flush_pending) after the frame loop
    /// to get the last frames' data.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "yuv_convert_readback")
    )]
    pub fn convert_and_readback(
        &mut self,
        gpu: &GpuContext,
        render_target: &wgpu::Texture,
        render_commands: wgpu::CommandBuffer,
    ) -> Result<Option<&[u8]>, YuvConvertError> {
        let write_slot = self.current_slot;

        // --- Submit this frame's GPU work FIRST ---
        // By submitting before readback, the GPU starts working on the
        // current frame while the CPU reads back data from 2 frames ago.
        self.submit_gpu_work(gpu, render_target, render_commands, write_slot)?;

        // --- Read back from 2 frames ago (if available) ---
        // read_slot = (write_slot + 1) % 3 is the oldest pending slot.
        // When write_slot cycles 0->1->2->0:
        //   write 0, read 1 (was written 2 frames ago)
        //   write 1, read 2 (was written 2 frames ago)
        //   write 2, read 0 (was written 2 frames ago)
        let has_result = if self.pending_count >= 2 {
            let read_slot = (write_slot + 1) % 3;
            self.readback_staging(gpu, read_slot)?;
            true
        } else {
            false
        };

        self.pending_count = (self.pending_count + 1).min(2);
        self.current_slot = (write_slot + 1) % 3;

        if has_result {
            let read_slot = (write_slot + 1) % 3;
            Ok(Some(&self.readback_buffers[read_slot]))
        } else {
            Ok(None)
        }
    }

    /// Flush one pending frame from the triple-buffer pipeline.
    ///
    /// Call this in a loop after the frame loop ends to drain remaining
    /// frames. Returns `None` when no frames are pending.
    ///
    /// Uses blocking poll since no new GPU work follows to overlap with.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "yuv_flush")
    )]
    pub fn flush_pending(&mut self, gpu: &GpuContext) -> Result<Option<&[u8]>, YuvConvertError> {
        if self.pending_count == 0 {
            return Ok(None);
        }
        // Oldest pending slot: current_slot was already advanced, so walk
        // back by pending_count to find the oldest unread buffer.
        let read_slot = (self.current_slot + 3 - self.pending_count as usize) % 3;
        self.readback_staging_blocking(gpu, read_slot)?;
        self.pending_count -= 1;
        Ok(Some(&self.readback_buffers[read_slot]))
    }

    /// Submit GPU render + YUV compute + copy to a specific staging slot.
    fn submit_gpu_work(
        &mut self,
        gpu: &GpuContext,
        render_target: &wgpu::Texture,
        render_commands: wgpu::CommandBuffer,
        slot: usize,
    ) -> Result<(), YuvConvertError> {
        // Cache the bind group to avoid per-frame descriptor pool allocation,
        // which causes OOM on the Vulkan backend (wgpu#7525). Rebuild only
        // if the render target texture changes.
        let needs_rebuild = self
            .cached_bind_group
            .as_ref()
            .is_none_or(|(texture, _)| texture != render_target);

        if needs_rebuild {
            let input = render_target.format();
            let is_p010 = self.format == DeliveryFormat::P010;
            if (input == wgpu::TextureFormat::Rgb10a2Unorm) != is_p010 {
                return Err(YuvConvertError::InputFormat {
                    input,
                    output: self.format,
                });
            }
            if is_p010
                && (render_target.width(), render_target.height()) != (self.width, self.height)
            {
                return Err(YuvConvertError::InvalidDimensions(format!(
                    "P010 source is {}x{}, expected {}x{}",
                    render_target.width(),
                    render_target.height(),
                    self.width,
                    self.height,
                )));
            }
            let render_target_view =
                render_target.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("yuv_bind_group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&render_target_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });
            self.cached_bind_group = Some((render_target.clone(), bind_group));
        }

        let (_, bind_group) = self
            .cached_bind_group
            .as_ref()
            .expect("bind group was just built above");

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("yuv_encoder"),
            });

        // Dispatch compute shader
        {
            crate::profile_scope!("yuv_compute");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yuv_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(self.dispatch_x, self.dispatch_y, 1);
        }

        // Copy GPU buffer to the target staging buffer
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffers[slot],
            0,
            self.output_buffer.size(),
        );

        {
            crate::profile_scope!("yuv_submit");
            gpu.queue.submit([render_commands, encoder.finish()]);
        }

        Ok(())
    }

    /// Read back data from a staging buffer using non-blocking poll.
    ///
    /// The buffer was submitted 2 frames ago, so it should already be complete.
    /// Falls back to blocking poll as a safety net if the GPU hasn't finished.
    fn readback_staging(&mut self, gpu: &GpuContext, slot: usize) -> Result<(), YuvConvertError> {
        crate::profile_scope!("yuv_readback");
        let buffer_slice = self.staging_buffers[slot].slice(..);
        let tx = self.map_tx.clone();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Non-blocking poll: the GPU work for this buffer was submitted 2
        // frames ago, so it should be complete.
        gpu.device
            .poll(wgpu::PollType::Poll)
            .map_err(|_| YuvConvertError::BufferMapFailed)?;

        // Check if map completed. Fall back to blocking if needed.
        match self.map_rx.try_recv() {
            Ok(result) => result.map_err(|_| YuvConvertError::BufferMapFailed)?,
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Safety net: block if the GPU somehow hasn't finished.
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|_| YuvConvertError::BufferMapFailed)?;
                self.map_rx
                    .recv()
                    .map_err(|_| YuvConvertError::BufferMapFailed)?
                    .map_err(|_| YuvConvertError::BufferMapFailed)?;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(YuvConvertError::BufferMapFailed);
            }
        }

        let mapped = buffer_slice.get_mapped_range();
        self.readback_buffers[slot].copy_from_slice(&mapped);
        drop(mapped);
        self.staging_buffers[slot].unmap();
        Ok(())
    }

    /// Read back data from a staging buffer using blocking poll.
    ///
    /// Used by [`flush_pending`](Self::flush_pending) where no new GPU work
    /// follows to overlap with, so we must wait for completion.
    fn readback_staging_blocking(
        &mut self,
        gpu: &GpuContext,
        slot: usize,
    ) -> Result<(), YuvConvertError> {
        crate::profile_scope!("yuv_readback_blocking");
        let buffer_slice = self.staging_buffers[slot].slice(..);
        let tx = self.map_tx.clone();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| YuvConvertError::BufferMapFailed)?;
        self.map_rx
            .recv()
            .map_err(|_| YuvConvertError::BufferMapFailed)?
            .map_err(|_| YuvConvertError::BufferMapFailed)?;

        let mapped = buffer_slice.get_mapped_range();
        self.readback_buffers[slot].copy_from_slice(&mapped);
        drop(mapped);
        self.staging_buffers[slot].unmap();
        Ok(())
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

/// Errors from the GPU delivery converter.
#[derive(Debug, Clone, thiserror::Error)]
pub enum YuvConvertError {
    /// GPU buffer mapping failed.
    #[error("YUV buffer mapping failed")]
    BufferMapFailed,

    /// Invalid output dimensions.
    #[error("invalid delivery dimensions: {0}")]
    InvalidDimensions(String),

    /// The source format does not match the requested delivery format.
    #[error("cannot convert {input:?} to {output:?}")]
    InputFormat {
        /// Input texture format.
        input: wgpu::TextureFormat,
        /// Requested delivery format.
        output: DeliveryFormat,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacing_a_texture_handle_preserves_frame_identity_and_drain_order() {
        let Some(gpu) = crate::stitch::test_support::gpu_or_skip() else {
            return;
        };
        let make_texture = |value: u8| {
            let size = wgpu::Extent3d {
                width: 4,
                height: 2,
                depth_or_array_layers: 1,
            };
            let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("nv12-frame-identity"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            gpu.queue().write_texture(
                texture.as_image_copy(),
                &[value; 32],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16),
                    rows_per_image: Some(2),
                },
                size,
            );
            texture
        };
        let mut converter = YuvConverter::new(&gpu, 4, 2).unwrap();
        // Replacing the handle in-place must not reuse its former texture's binding.
        let mut texture = Box::new(make_texture(0));
        for values in [&[0u8][..], &[255, 0][..], &[0, 255, 0, 255, 0][..]] {
            let mut output = Vec::new();
            for &value in values {
                *texture = make_texture(value);
                let commands = gpu
                    .device()
                    .create_command_encoder(&Default::default())
                    .finish();
                if let Some(bytes) = converter
                    .convert_and_readback(&gpu, &texture, commands)
                    .unwrap()
                {
                    output.push(bytes[0]);
                }
            }
            while let Some(bytes) = converter.flush_pending(&gpu).unwrap() {
                output.push(bytes[0]);
            }
            let expected: Vec<_> = values
                .iter()
                .map(|&v| if v == 0 { 16 } else { 235 })
                .collect();
            assert_eq!(output, expected);
            assert!(converter.flush_pending(&gpu).unwrap().is_none());
        }
    }
}

#[cfg(test)]
#[path = "yuv_converter_tests.rs"]
mod p010_tests;
