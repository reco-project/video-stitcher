//! GPU YUV420P stacked-video packer (M7 pivot item).
//!
//! Produces a grid-packed YUV420P atlas from N source tiles that are
//! already GPU-resident, skipping the CPU memcpy that
//! [`reco_io::stacked_video::pack_yuv420p`] would otherwise do on
//! inputs that flow CPU-side (~8 ms/frame at 4K). Optional linear
//! downscale is free because the sampler handles it — so this also
//! subsumes FRICTION reco-obs A19 (replay resolution dropdown).
//!
//! # When to use this vs the CPU pack
//!
//! The packer only helps when the source tiles are already on the
//! GPU. For FFmpeg software decode → CPU `YuvPlanes<'_>` inputs,
//! copying them to the GPU just to pack would lose to the existing
//! CPU pack. [`StitchPipeline`] chooses the path explicitly and
//! logs the decision once per session (see the integration site).
//!
//! # Threading model
//!
//! Single-thread, mirrors [`Nv12Converter`]. The pipeline's owning
//! thread dispatches the compute kernels and polls readbacks.
//! Triple-buffered staging so the CPU reads from 2-frames-old
//! storage — guaranteed complete on the GPU and does not need a
//! blocking device poll.

use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};
use thiserror::Error;

/// Grid layout describing how source tiles map into the atlas.
///
/// Mirrors the shape of
/// [`reco_io::stacked_video::GridLayout`] intentionally: consumers
/// who opt into GPU packing should be able to pass the same layout
/// value they would pass to the CPU `pack_yuv420p` helper. The
/// duplicate type lives in reco-core so this module has no reco-io
/// dependency (reco-core is the "no I/O" layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackGridLayout {
    tile_width: u32,
    tile_height: u32,
    rows: u32,
    cols: u32,
}

impl StackGridLayout {
    /// Vertical stack of `n` tiles, each `width × height`.
    ///
    /// Returns `None` for odd `width`/`height` (YUV420P requires
    /// even) or `n == 0`.
    pub fn vstack(width: u32, height: u32, n: u32) -> Option<Self> {
        Self::grid(width, height, n, 1)
    }

    /// Horizontal stack of `n` tiles.
    pub fn hstack(width: u32, height: u32, n: u32) -> Option<Self> {
        Self::grid(width, height, 1, n)
    }

    /// Generic `rows × cols` grid. All tiles share dims.
    pub fn grid(width: u32, height: u32, rows: u32, cols: u32) -> Option<Self> {
        if width == 0 || height == 0 || rows == 0 || cols == 0 {
            return None;
        }
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return None;
        }
        Some(Self {
            tile_width: width,
            tile_height: height,
            rows,
            cols,
        })
    }

    /// Tile slot capacity (`rows * cols`).
    pub fn capacity(&self) -> u32 {
        self.rows * self.cols
    }

    /// Output atlas width in Y-plane pixels.
    pub fn atlas_width(&self) -> u32 {
        self.cols * self.tile_width
    }

    /// Output atlas height in Y-plane rows.
    pub fn atlas_height(&self) -> u32 {
        self.rows * self.tile_height
    }

    /// Row index of the slot that holds `tile_index`, counting in
    /// Y-plane tile-height rows. Row-major: tile 0 is top-left,
    /// tile cols-1 is top-right, tile cols is leftmost of row 2, etc.
    fn tile_row(&self, tile_index: u32) -> u32 {
        tile_index / self.cols
    }

    /// Column index of the slot that holds `tile_index`.
    fn tile_col(&self, tile_index: u32) -> u32 {
        tile_index % self.cols
    }
}

/// Packer errors.
#[derive(Debug, Clone, Error)]
pub enum PackerError {
    /// Width or height failed YUV420P even-alignment checks or the
    /// 4-column packing shader requires width divisible by 4.
    #[error("invalid dimensions for packer: {0}")]
    InvalidDimensions(String),
}

/// Shader-side uniform layout. 10 × u32 = 40 bytes; padded to 48
/// bytes for std140 alignment on backends that require 16-byte
/// struct boundaries. Kept here in Rust rather than relying on
/// automatic bindgen so the byte layout is obviously correct when
/// reading the shader source.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PackParams {
    tile_y_row_offset: u32,
    tile_y_col_offset: u32,
    out_tile_width: u32,
    out_tile_height: u32,
    atlas_y_stride: u32,
    atlas_uv_stride: u32,
    y_plane_u32_offset: u32,
    u_plane_u32_offset: u32,
    v_plane_u32_offset: u32,
    _pad: u32,
}

/// Staging ring depth. Same 3-slot pipelining as [`Nv12Converter`]:
/// write slot 0 → 1 → 2 → 0; read always (write + 1) % 3 which is
/// guaranteed-done on the GPU two frames later.
const STAGING_SLOTS: usize = 3;

/// Output tile dimensions after optional downscale.
///
/// When `output == source_dims` the sampler returns exact byte
/// values (R8Unorm 1:1 reads). When smaller, the sampler's linear
/// filter handles the downscale for free — matching what a
/// bilinear-filtered CPU downscale would produce.
#[derive(Debug, Clone, Copy)]
pub struct OutputTileSize {
    /// Y-plane width per tile (atlas columns per tile).
    pub width: u32,
    /// Y-plane rows per tile.
    pub height: u32,
}

impl OutputTileSize {
    /// Match source dims — no scale, byte-for-byte equivalent to the
    /// CPU pack on aligned reads.
    pub fn unscaled(tile_width: u32, tile_height: u32) -> Self {
        Self {
            width: tile_width,
            height: tile_height,
        }
    }

    /// Downscale to `(w, h)`. Both dims must be even for YUV420P and
    /// `w` must be divisible by 4 for the packing shader.
    pub fn scaled(w: u32, h: u32) -> Self {
        Self {
            width: w,
            height: h,
        }
    }
}

/// A fully-packed readback: atlas bytes split into Y/U/V planes in
/// the exact layout
/// [`reco_io::stacked_video::pack_yuv420p`] produces on the CPU
/// path. Consumers pass these straight to
/// `VideoEncoder::write_yuv420p_planes`.
pub struct StackedAtlas {
    /// Y plane, `atlas_width * atlas_height` bytes.
    pub y: Vec<u8>,
    /// U plane, `(atlas_width / 2) * (atlas_height / 2)` bytes.
    pub u: Vec<u8>,
    /// V plane, same size as U.
    pub v: Vec<u8>,
    /// Atlas Y-plane width.
    pub width: u32,
    /// Atlas Y-plane height.
    pub height: u32,
}

/// GPU-accelerated stacked-video packer with triple-buffered readback.
///
/// # Usage
///
/// ```rust,ignore
/// let layout = StackGridLayout::vstack(1920, 1080, 2).unwrap();
/// let out   = OutputTileSize::unscaled(1920, 1080);
/// let mut packer = YuvStackPacker::new(&gpu, layout, out)?;
///
/// // Per frame, inside the command encoder that drives the stitch:
/// packer.pack_tile(&mut encoder, 0, &left_y, &left_u, &left_v);
/// packer.pack_tile(&mut encoder, 1, &right_y, &right_u, &right_v);
/// packer.copy_to_staging(&mut encoder);
///
/// // Later (after queue.submit), poll for the frame that's been
/// // in flight for two submits:
/// if let Some(atlas) = packer.poll_ready(&gpu) { /* encode it */ }
/// ```
pub struct YuvStackPacker {
    /// Pipelines for the three plane-packing kernels.
    pipeline_y: wgpu::ComputePipeline,
    pipeline_u: wgpu::ComputePipeline,
    pipeline_v: wgpu::ComputePipeline,
    /// Shared bind group layout — same for all three kernels.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Linear-filter sampler for the downscale path. Also used for
    /// unscaled pack; in the 1:1 case sampling at pixel centers
    /// returns the exact source byte.
    sampler: wgpu::Sampler,
    /// Per-tile uniform buffers. One per tile slot so successive
    /// `pack_tile` calls don't stomp on each other's uniforms
    /// before the GPU consumes them.
    params_buffers: Vec<wgpu::Buffer>,
    /// Atlas storage buffer (`STORAGE | COPY_SRC`). Contains the
    /// full YUV420P atlas: Y bytes, then U bytes, then V bytes,
    /// tight packed. Size computed from `layout` + `output`.
    atlas_buffer: wgpu::Buffer,
    /// Triple-buffered `COPY_DST | MAP_READ` staging buffers.
    staging_buffers: [wgpu::Buffer; STAGING_SLOTS],
    current_slot: usize,
    pending_count: u8,
    /// Readback buffers, one per staging slot, reused to avoid
    /// per-frame allocation. Sized at `atlas_bytes` (Y + UV + UV).
    readback_buffers: [Vec<u8>; STAGING_SLOTS],
    /// Channel for `map_async` signaling. Reuses a SyncSender pair
    /// the same way `Nv12Converter` does — avoids per-frame channel
    /// alloc which shows up on Mac M4 profiles.
    map_tx: std::sync::mpsc::SyncSender<Result<(), wgpu::BufferAsyncError>>,
    map_rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    /// Captured config for validation + log lines.
    layout: StackGridLayout,
    output: OutputTileSize,
    /// Plane byte sizes.
    y_plane_bytes: u32,
    uv_plane_bytes: u32,
    atlas_bytes: u32,
}

impl YuvStackPacker {
    /// Create a packer for a given layout and output tile size.
    pub fn new(
        gpu: &GpuContext,
        layout: StackGridLayout,
        output: OutputTileSize,
    ) -> Result<Self, PackerError> {
        if !output.width.is_multiple_of(4) {
            return Err(PackerError::InvalidDimensions(format!(
                "output tile width must be divisible by 4 (packed 4-to-u32), got {}",
                output.width
            )));
        }
        if !output.height.is_multiple_of(2) {
            return Err(PackerError::InvalidDimensions(format!(
                "output tile height must be even (YUV420P subsampling), got {}",
                output.height
            )));
        }

        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yuv420p_stack_pack"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/yuv420p_stack_pack.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yuv_stack_pack_bgl"),
            entries: &[
                // @binding(0,1,2): three R8Unorm source textures (Y, U, V)
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(3): linear-filter sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // @binding(4): atlas storage buffer (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(5): uniform params
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
            label: Some("yuv_stack_pack_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline_y = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yuv_stack_pack_y"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("pack_y"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_u = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yuv_stack_pack_u"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("pack_u"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_v = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yuv_stack_pack_v"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("pack_v"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("yuv_stack_pack_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let atlas_w = output.width * layout.cols;
        let atlas_h = output.height * layout.rows;
        let y_plane_bytes = atlas_w * atlas_h;
        let uv_plane_bytes = (atlas_w / 2) * (atlas_h / 2);
        let atlas_bytes = y_plane_bytes + 2 * uv_plane_bytes;

        let atlas_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yuv_stack_pack_atlas"),
            size: atlas_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let make_staging = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: atlas_bytes as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let staging_buffers = [
            make_staging("yuv_stack_pack_staging_0"),
            make_staging("yuv_stack_pack_staging_1"),
            make_staging("yuv_stack_pack_staging_2"),
        ];

        let params_buffers = (0..layout.capacity())
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("yuv_stack_pack_params_{i}")),
                    size: std::mem::size_of::<PackParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let (map_tx, map_rx) = std::sync::mpsc::sync_channel(STAGING_SLOTS);

        Ok(Self {
            pipeline_y,
            pipeline_u,
            pipeline_v,
            bind_group_layout,
            sampler,
            params_buffers,
            atlas_buffer,
            staging_buffers,
            current_slot: 0,
            pending_count: 0,
            readback_buffers: [
                vec![0u8; atlas_bytes as usize],
                vec![0u8; atlas_bytes as usize],
                vec![0u8; atlas_bytes as usize],
            ],
            map_tx,
            map_rx,
            layout,
            output,
            y_plane_bytes,
            uv_plane_bytes,
            atlas_bytes,
        })
    }

    /// Layout the packer was built for.
    pub fn layout(&self) -> &StackGridLayout {
        &self.layout
    }

    /// Output tile size (may be smaller than the source tiles if
    /// downscale is in effect).
    pub fn output(&self) -> OutputTileSize {
        self.output
    }

    /// Atlas dimensions in Y-plane pixels.
    pub fn atlas_dims(&self) -> (u32, u32) {
        (
            self.output.width * self.layout.cols,
            self.output.height * self.layout.rows,
        )
    }

    /// Dispatch the three plane kernels for one tile.
    ///
    /// `tile_index` is the row-major slot the tile occupies in the
    /// atlas. `src_y` / `src_u` / `src_v` are the source plane
    /// textures — typically the renderer's internal plane textures
    /// for that camera side. The texture sizes don't have to match
    /// `self.output`: the sampler's linear filter handles the
    /// downscale.
    ///
    /// Call once per tile per frame, then
    /// [`Self::copy_to_staging`] once, then `queue.submit(...)`.
    pub fn pack_tile(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        tile_index: u32,
        src_y: &wgpu::TextureView,
        src_u: &wgpu::TextureView,
        src_v: &wgpu::TextureView,
    ) {
        assert!(
            tile_index < self.layout.capacity(),
            "tile_index {} out of range (capacity {})",
            tile_index,
            self.layout.capacity()
        );

        let atlas_w = self.output.width * self.layout.cols;
        let params = PackParams {
            tile_y_row_offset: self.layout.tile_row(tile_index) * self.output.height,
            tile_y_col_offset: self.layout.tile_col(tile_index) * self.output.width,
            out_tile_width: self.output.width,
            out_tile_height: self.output.height,
            atlas_y_stride: atlas_w,
            atlas_uv_stride: atlas_w / 2,
            y_plane_u32_offset: 0,
            u_plane_u32_offset: self.y_plane_bytes / 4,
            v_plane_u32_offset: (self.y_plane_bytes + self.uv_plane_bytes) / 4,
            _pad: 0,
        };
        gpu.queue.write_buffer(
            &self.params_buffers[tile_index as usize],
            0,
            bytemuck::bytes_of(&params),
        );

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("yuv_stack_pack_bg_tile_{tile_index}")),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_y),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src_u),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(src_v),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.atlas_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.params_buffers[tile_index as usize].as_entire_binding(),
                },
            ],
        });

        // Workgroup size 8×8. Y dispatch covers (tile_w / 4, tile_h).
        // U/V dispatches cover ((tile_w / 2) / 4, tile_h / 2).
        let y_groups_x = self.output.width.div_ceil(32); // /4 for quad, /8 for wg
        let y_groups_y = self.output.height.div_ceil(8);
        let uv_groups_x = (self.output.width / 2).div_ceil(32);
        let uv_groups_y = (self.output.height / 2).div_ceil(8);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("yuv_stack_pack_pass_tile_{tile_index}")),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_pipeline(&self.pipeline_y);
        pass.dispatch_workgroups(y_groups_x, y_groups_y, 1);
        pass.set_pipeline(&self.pipeline_u);
        pass.dispatch_workgroups(uv_groups_x, uv_groups_y, 1);
        pass.set_pipeline(&self.pipeline_v);
        pass.dispatch_workgroups(uv_groups_x, uv_groups_y, 1);
    }

    /// Queue a GPU copy from the atlas buffer into the current
    /// staging slot, then advance the slot pointer. Call once per
    /// frame after all `pack_tile` calls.
    pub fn copy_to_staging(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.atlas_buffer,
            0,
            &self.staging_buffers[self.current_slot],
            0,
            self.atlas_bytes as u64,
        );
        self.current_slot = (self.current_slot + 1) % STAGING_SLOTS;
        self.pending_count = (self.pending_count + 1).min(STAGING_SLOTS as u8);
    }

    /// Non-blocking poll for the frame two submits ago. Returns
    /// `None` for the first two calls (warmup) and whenever the
    /// slot isn't ready yet.
    pub fn poll_ready(&mut self, gpu: &GpuContext) -> Option<StackedAtlas> {
        // Read slot lags the write slot by 2 (current_slot already
        // advanced post-copy). Equivalent to `(current_slot + 1) % 3`.
        if self.pending_count < STAGING_SLOTS as u8 {
            return None;
        }
        let read_slot = (self.current_slot + 1) % STAGING_SLOTS;
        let buffer = &self.staging_buffers[read_slot];
        let slice = buffer.slice(..);
        let tx = self.map_tx.clone();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        // Non-blocking poll. If the GPU hasn't finished yet we'll
        // try again next frame.
        let poll = gpu.device.poll(wgpu::PollType::Poll).ok()?;
        let _ = poll; // intentionally ignore PollStatus details

        match self.map_rx.try_recv() {
            Ok(Ok(())) => {
                // Snapshot dims + plane bounds before borrowing
                // readback_buffers mutably — keeps the borrow
                // checker happy without a second pass.
                let (atlas_w, atlas_h) = self.atlas_dims();
                let y_end = self.y_plane_bytes as usize;
                let u_end = y_end + self.uv_plane_bytes as usize;
                let v_end = u_end + self.uv_plane_bytes as usize;

                let mapped = slice.get_mapped_range();
                let dst = &mut self.readback_buffers[read_slot];
                dst.clear();
                dst.extend_from_slice(&mapped);
                drop(mapped);
                buffer.unmap();

                Some(StackedAtlas {
                    y: dst[..y_end].to_vec(),
                    u: dst[y_end..u_end].to_vec(),
                    v: dst[u_end..v_end].to_vec(),
                    width: atlas_w,
                    height: atlas_h,
                })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;

    /// Fill an R8Unorm texture with a single byte value across all
    /// pixels and return the view. Helper for the smoke tests below.
    fn make_filled_r8(gpu: &GpuContext, w: u32, h: u32, fill: u8) -> wgpu::TextureView {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_filled_r8"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let data = vec![fill; (w * h) as usize];
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Pump the triple-buffered readback until a frame comes out.
    /// Normal operation has a two-frame warmup; tests need to
    /// re-submit the same tiles a few times to flush through.
    fn pump_packer(
        packer: &mut YuvStackPacker,
        gpu: &GpuContext,
        dispatch: impl Fn(&mut wgpu::CommandEncoder, &YuvStackPacker),
    ) -> StackedAtlas {
        for _ in 0..8 {
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch(&mut enc, packer);
            packer.copy_to_staging(&mut enc);
            gpu.queue.submit(std::iter::once(enc.finish()));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            if let Some(atlas) = packer.poll_ready(gpu) {
                return atlas;
            }
        }
        panic!("packer never produced a frame after 8 submits");
    }

    /// 2-tile vstack, unscaled 8x8 tiles with constant fills.
    /// Requires a live GPU so the test is `#[ignore]`d; run manually
    /// with `cargo test -p reco-core -- --ignored yuv_stack`.
    #[test]
    #[ignore = "requires a GPU (wgpu adapter init)"]
    fn vstack_constants_round_trip() {
        let gpu = GpuContext::new_blocking().expect("GPU init");
        let layout = StackGridLayout::vstack(8, 8, 2).expect("layout");
        let out = OutputTileSize::unscaled(8, 8);
        let mut packer = YuvStackPacker::new(&gpu, layout, out).expect("packer");

        let left_y = make_filled_r8(&gpu, 8, 8, 100);
        let left_u = make_filled_r8(&gpu, 4, 4, 110);
        let left_v = make_filled_r8(&gpu, 4, 4, 120);
        let right_y = make_filled_r8(&gpu, 8, 8, 200);
        let right_u = make_filled_r8(&gpu, 4, 4, 210);
        let right_v = make_filled_r8(&gpu, 4, 4, 220);

        let atlas = pump_packer(&mut packer, &gpu, |enc, p| {
            p.pack_tile(&gpu, enc, 0, &left_y, &left_u, &left_v);
            p.pack_tile(&gpu, enc, 1, &right_y, &right_u, &right_v);
        });

        // Atlas Y plane: 8 wide × 16 tall. Top half = 100, bottom = 200.
        assert_eq!(atlas.width, 8);
        assert_eq!(atlas.height, 16);
        assert_eq!(atlas.y.len(), 8 * 16);
        // First 64 bytes = left tile fill (100)
        assert!(
            atlas.y[..64].iter().all(|&b| b == 100),
            "left Y tile should be all 100, got {:?}...",
            &atlas.y[..8]
        );
        // Next 64 bytes = right tile fill (200)
        assert!(
            atlas.y[64..].iter().all(|&b| b == 200),
            "right Y tile should be all 200"
        );

        // U plane: 4 wide × 8 tall. Top 16 bytes = 110, bottom 16 = 210.
        assert_eq!(atlas.u.len(), 4 * 8);
        assert!(atlas.u[..16].iter().all(|&b| b == 110));
        assert!(atlas.u[16..].iter().all(|&b| b == 210));
        // V plane same shape.
        assert_eq!(atlas.v.len(), 4 * 8);
        assert!(atlas.v[..16].iter().all(|&b| b == 120));
        assert!(atlas.v[16..].iter().all(|&b| b == 220));
    }

    /// Downscale path: 16x16 source tiles → 8x8 output tiles in a
    /// 2-tile vstack. Constant fills survive downscaling (a
    /// linear-filtered constant region is still that constant).
    /// This exercises the sampler's built-in bilinear when output
    /// < source.
    #[test]
    #[ignore = "requires a GPU (wgpu adapter init)"]
    fn vstack_downscale_constants() {
        let gpu = GpuContext::new_blocking().expect("GPU init");
        // Layout describes OUTPUT tile dims, since atlas dims follow
        // output × grid. Source textures can be arbitrary larger dims.
        let layout = StackGridLayout::vstack(8, 8, 2).expect("layout");
        let out = OutputTileSize::scaled(8, 8);
        let mut packer = YuvStackPacker::new(&gpu, layout, out).expect("packer");

        // Source textures at 16x16 (double the output tile).
        let left_y = make_filled_r8(&gpu, 16, 16, 77);
        let left_u = make_filled_r8(&gpu, 8, 8, 88);
        let left_v = make_filled_r8(&gpu, 8, 8, 99);
        let right_y = make_filled_r8(&gpu, 16, 16, 177);
        let right_u = make_filled_r8(&gpu, 8, 8, 188);
        let right_v = make_filled_r8(&gpu, 8, 8, 199);

        let atlas = pump_packer(&mut packer, &gpu, |enc, p| {
            p.pack_tile(&gpu, enc, 0, &left_y, &left_u, &left_v);
            p.pack_tile(&gpu, enc, 1, &right_y, &right_u, &right_v);
        });

        // Atlas same shape as unscaled 8x8 × 2 vstack.
        assert_eq!((atlas.width, atlas.height), (8, 16));
        assert!(atlas.y[..64].iter().all(|&b| b == 77));
        assert!(atlas.y[64..].iter().all(|&b| b == 177));
        assert!(atlas.u[..16].iter().all(|&b| b == 88));
        assert!(atlas.u[16..].iter().all(|&b| b == 188));
        assert!(atlas.v[..16].iter().all(|&b| b == 99));
        assert!(atlas.v[16..].iter().all(|&b| b == 199));
    }
}
