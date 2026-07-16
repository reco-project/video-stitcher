//! GPU-resident bit-depth downconversion for the lookahead pool.
//!
//! See `shaders/lookahead_downconvert.wgsl` for the "why" (lookahead pool
//! VRAM cost scales with source bit depth) and the shader itself. This
//! module owns the two small render pipelines (one per plane, since the Y
//! and UV planes differ in channel count and resolution) and the single
//! entry point, [`LookaheadDownconverter::convert_plane`], that runs one.
//!
//! Opt-in, off by default - see [`crate::session::vram_pool::LookaheadBitDepth`].
//! Currently wired into [`crate::session::vram_pool::VramPool`] (Linux/
//! macOS). Not yet wired into the Windows `D3d11StagingPool` path - see
//! `FRICTION.md` "Lookahead pool VRAM cost scales with source bit depth"
//! for why that side needs separate, careful follow-up (the pool there
//! also feeds CUDA-resident AI detection via D3D11 shared-handle import,
//! which this pass's plain wgpu-native destination textures don't
//! support without additional interop work).

use crate::gpu::GpuContext;

/// Fullscreen-triangle render pipelines that copy one plane's pixel values
/// from a source texture into a same-resolution, lower-bit-depth
/// destination texture (e.g. `R16Unorm` -> `R8Unorm` for a Y plane, or
/// `Rg16Unorm` -> `Rg8Unorm` for a UV plane).
///
/// Two pipelines are needed (not one) because a wgpu render pipeline is
/// tied to a fixed output format at creation time, and the Y/UV planes use
/// different formats (single- vs dual-channel). Both pipelines share the
/// same shader module and bind group layout - only the color target format
/// differs.
pub struct LookaheadDownconverter {
    bind_group_layout: wgpu::BindGroupLayout,
    y_pipeline: wgpu::RenderPipeline,
    uv_pipeline: wgpu::RenderPipeline,
}

impl LookaheadDownconverter {
    /// Build the downconversion pipelines.
    ///
    /// Cheap one-time setup (two small pipeline objects); construct once
    /// per `GpuContext` and reuse across frames, mirroring how
    /// [`crate::render::renderer::Renderer`] owns its pipelines.
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lookahead_downconvert"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/lookahead_downconvert.wgsl").into(),
            ),
        });

        // Non-filterable: the shader uses textureLoad (exact texel copy,
        // same resolution in and out), not textureSample, so no sampler is
        // bound and the source format does not need to support linear
        // filtering (16-bit Unorm formats are not guaranteed filterable on
        // every wgpu backend).
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lookahead_downconvert_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lookahead_downconvert_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let make_pipeline = |label: &str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let y_pipeline = make_pipeline("lookahead_downconvert_y", wgpu::TextureFormat::R8Unorm);
        let uv_pipeline = make_pipeline("lookahead_downconvert_uv", wgpu::TextureFormat::Rg8Unorm);

        Self {
            bind_group_layout,
            y_pipeline,
            uv_pipeline,
        }
    }

    /// Downconvert one plane: `src` is sampled with `textureLoad` and its
    /// values written into `dst` at `dst`'s (lower) bit depth.
    ///
    /// `dst` must be `R8Unorm` when `is_uv` is `false` (Y plane) or
    /// `Rg8Unorm` when `is_uv` is `true` (UV plane), with
    /// `RENDER_ATTACHMENT` usage, and the same width/height as `src`. `src`
    /// must have `TEXTURE_BINDING` usage. Appends one render pass to
    /// `encoder`; does not submit.
    pub fn convert_plane(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        is_uv: bool,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lookahead_downconvert_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            }],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lookahead_downconvert_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
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
        pass.set_pipeline(if is_uv {
            &self.uv_pipeline
        } else {
            &self.y_pipeline
        });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end correctness check on a real (or software) adapter: known
    /// 16-bit values in a source texture must round-trip through the
    /// downconvert pass to the expected 8-bit values, within the rounding
    /// wgpu's own hardware Unorm normalization introduces at each end
    /// (1 LSB in the 8-bit output, i.e. +/-1/255).
    ///
    /// Runs against whatever adapter `GpuContext::new` selects in this
    /// environment (falls back to software rendering under CI/headless
    /// conditions per wgpu's normal adapter selection) - skips instead of
    /// failing if no adapter is available at all, matching this crate's
    /// other GPU-dependent tests (see `interop::cuda::tests`).
    #[test]
    fn y_plane_downconvert_matches_expected_8bit_values() {
        let gpu = match pollster::block_on(GpuContext::new()) {
            Ok(gpu) => gpu,
            Err(e) => {
                eprintln!("skipping: no GPU adapter available ({e})");
                return;
            }
        };
        let downconverter = LookaheadDownconverter::new(&gpu);

        const W: u32 = 4;
        const H: u32 = 1;
        // R16Unorm source values chosen to land on exact 8-bit boundaries
        // after normalization: 0x0000 -> 0, 0x8080 -> 128, 0xFFFF -> 255,
        // 0x4040 -> 64 (0x4040 / 0xFFFF * 255 = 64.0 exactly).
        let src_u16: [u16; W as usize] = [0x0000, 0x8080, 0xFFFF, 0x4040];

        let src_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_src_y"),
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&src_u16),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 2),
                rows_per_image: Some(H),
            },
            wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
        );

        let dst_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_dst_y"),
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let src_view = src_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = dst_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        downconverter.convert_plane(&gpu.device, &mut encoder, &src_view, &dst_view, false);

        // Readback: copy to a mapped buffer.
        let bytes_per_row = 256; // wgpu COPY_BYTES_PER_ROW_ALIGNMENT
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_readback"),
            size: (bytes_per_row * H) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &dst_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(H),
                },
            },
            wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll failed");
        rx.recv()
            .expect("map_async never signaled")
            .expect("buffer map failed");

        let data = slice.get_mapped_range();
        let got = [data[0], data[1], data[2], data[3]];
        drop(data);
        readback.unmap();

        assert_eq!(
            got,
            [0, 128, 255, 64],
            "unexpected downconverted Y values: {got:?}"
        );
    }
}
