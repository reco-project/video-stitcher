//! GPU → CPU RGBA readback with triple-buffered staging.
//!
//! [`RgbaReadback`] reads pixels from an RGBA wgpu texture (typically the
//! stitch pipeline's render target) into a tightly-packed
//! `Vec<u8>` of `[R, G, B, A, R, G, B, A, ...]` bytes. It strips the
//! 256-byte row padding wgpu requires for `copy_texture_to_buffer` so
//! callers receive exactly `width * height * 4` bytes.
//!
//! ## Triple-buffered readback
//!
//! Uses the same pattern as [`Nv12Converter`](crate::nv12_converter::Nv12Converter):
//! three staging buffers cycle so the CPU always reads from 2 frames ago,
//! which is guaranteed to have completed on the GPU. This avoids the
//! blocking `poll(Wait)` stall (3-8ms at 1080p) that a single-buffered
//! reader incurs.
//!
//! ## Why a separate type
//!
//! [`Nv12Converter`](crate::nv12_converter::Nv12Converter) also runs a
//! compute shader to convert RGBA → NV12 before the readback. GUI and
//! OBS consumers want the raw RGBA (or BGRA) pixels for display, not an
//! encoded NV12 stream. Sharing the NV12 converter would force every
//! consumer to pay for the GPU conversion work; instead this module
//! handles the staging-buffer + map-async + row-strip plumbing directly.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reco_core::rgba_readback::RgbaReadback;
//!
//! let mut readback = RgbaReadback::new(&gpu, width, height)?;
//! // Inside the render loop:
//! let cmd = pipeline.render_to_target(&left, &right, yaw, pitch)?;
//! if let Some(rgba) = readback.readback(&gpu, pipeline.render_target(), cmd)? {
//!     // rgba is &[u8] of length width * height * 4
//!     display.update(rgba);
//! }
//! ```

use crate::gpu::GpuContext;

/// GPU → CPU RGBA readback with triple-buffered staging.
///
/// Call [`readback`](Self::readback) once per rendered frame with the
/// pipeline's command buffer. Returns the RGBA bytes from 2 frames ago
/// (or `None` on the first two calls during pipeline warm-up).
///
/// Call [`flush_pending`](Self::flush_pending) in a loop after the main
/// frame loop to drain the remaining buffered frames.
pub struct RgbaReadback {
    /// Triple-buffered staging buffers (CPU-readable, GPU-writable).
    staging: [wgpu::Buffer; 3],
    /// Triple-buffered tightly-packed output buffers (no row padding).
    output: [Vec<u8>; 3],
    /// Which staging buffer to write to next (0, 1, or 2).
    current_slot: usize,
    /// Number of frames pending readback (0, 1, or 2).
    pending_count: u8,
    width: u32,
    height: u32,
    /// Bytes per row on the CPU side (tightly packed): `width * 4`.
    bytes_per_row: u32,
    /// Bytes per row in the staging buffer (rounded up to 256 for wgpu).
    padded_bytes_per_row: u32,
    /// Reusable channel for map_async signaling (avoids per-frame alloc).
    map_tx: std::sync::mpsc::SyncSender<Result<(), wgpu::BufferAsyncError>>,
    map_rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl RgbaReadback {
    /// Create a readback helper for a texture of the given dimensions.
    ///
    /// Allocates three staging buffers plus three tightly-packed output
    /// `Vec<u8>`s (no per-frame allocation during the frame loop).
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Result<Self, RgbaReadbackError> {
        if width == 0 || height == 0 {
            return Err(RgbaReadbackError::InvalidDimensions(format!(
                "width and height must be > 0, got {width}x{height}"
            )));
        }

        let bytes_per_row = width * 4;
        // wgpu requires `copy_texture_to_buffer` rows aligned to 256 bytes
        // (D3D12 / Vulkan requirement, enforced as
        // `COPY_BYTES_PER_ROW_ALIGNMENT`).
        let padded_bytes_per_row = bytes_per_row.div_ceil(256) * 256;
        let staging_size = (padded_bytes_per_row as u64) * (height as u64);
        let output_len = (bytes_per_row as usize) * (height as usize);

        let device = &gpu.device;
        let staging = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rgba_staging_0"),
                size: staging_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rgba_staging_1"),
                size: staging_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rgba_staging_2"),
                size: staging_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        ];

        let (map_tx, map_rx) = std::sync::mpsc::sync_channel(1);

        log::info!(
            "RgbaReadback: {width}x{height} → {staging_size} bytes staging \
             ({} MB, row padding {padded_bytes_per_row}/{bytes_per_row})",
            staging_size / 1_048_576,
        );

        Ok(Self {
            staging,
            output: [
                vec![0u8; output_len],
                vec![0u8; output_len],
                vec![0u8; output_len],
            ],
            current_slot: 0,
            pending_count: 0,
            width,
            height,
            bytes_per_row,
            padded_bytes_per_row,
            map_tx,
            map_rx,
        })
    }

    /// Submit the render command buffer and read back RGBA pixels.
    ///
    /// Enqueues `render_commands` plus a `copy_texture_to_buffer` into the
    /// current staging slot, then maps and strips row padding from the
    /// staging buffer written 2 frames ago.
    ///
    /// Returns `None` on the first two calls (GPU warmup), and
    /// `Some(&[u8])` of length `width * height * 4` afterward. The slice
    /// is tightly packed `R, G, B, A` in the order wgpu wrote it (matches
    /// the render target's texture format — `Rgba8Unorm` or `Bgra8Unorm`
    /// depending on how the pipeline was created).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "rgba_readback")
    )]
    pub fn readback(
        &mut self,
        gpu: &GpuContext,
        source: &wgpu::Texture,
        render_commands: wgpu::CommandBuffer,
    ) -> Result<Option<&[u8]>, RgbaReadbackError> {
        let write_slot = self.current_slot;

        self.submit_copy(gpu, source, render_commands, write_slot)?;

        // Read back from 2 frames ago (pending >= 2).
        let has_result = if self.pending_count >= 2 {
            let read_slot = (write_slot + 1) % 3;
            self.map_and_strip(gpu, read_slot)?;
            true
        } else {
            false
        };

        self.pending_count = (self.pending_count + 1).min(2);
        self.current_slot = (write_slot + 1) % 3;

        if has_result {
            let read_slot = (write_slot + 1) % 3;
            Ok(Some(&self.output[read_slot]))
        } else {
            Ok(None)
        }
    }

    /// Flush one pending frame from the triple-buffer pipeline.
    ///
    /// Call this in a loop after the frame loop ends to drain remaining
    /// frames. Returns `None` when no frames are pending. Uses blocking
    /// poll since no new GPU work follows to overlap with.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "rgba_flush")
    )]
    pub fn flush_pending(&mut self, gpu: &GpuContext) -> Result<Option<&[u8]>, RgbaReadbackError> {
        if self.pending_count == 0 {
            return Ok(None);
        }
        // Oldest pending slot: walk back by pending_count from current_slot.
        let read_slot = (self.current_slot + 3 - self.pending_count as usize) % 3;
        self.map_and_strip_blocking(gpu, read_slot)?;
        self.pending_count -= 1;
        Ok(Some(&self.output[read_slot]))
    }

    /// Output width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Output height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    fn submit_copy(
        &self,
        gpu: &GpuContext,
        source: &wgpu::Texture,
        render_commands: wgpu::CommandBuffer,
        slot: usize,
    ) -> Result<(), RgbaReadbackError> {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rgba_readback_encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging[slot],
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        gpu.queue.submit([render_commands, encoder.finish()]);
        Ok(())
    }

    /// Map the staging buffer at `slot` and copy its contents into the
    /// corresponding output buffer, stripping wgpu's per-row padding.
    ///
    /// Uses a non-blocking `PollType::Poll` first since the GPU work is
    /// 2 frames old and should already be complete; falls back to a
    /// blocking wait if the poll reports no progress.
    fn map_and_strip(&mut self, gpu: &GpuContext, slot: usize) -> Result<(), RgbaReadbackError> {
        crate::profile_scope!("rgba_map_async");
        let buffer_slice = self.staging[slot].slice(..);
        let tx = self.map_tx.clone();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        gpu.device
            .poll(wgpu::PollType::Poll)
            .map_err(|_| RgbaReadbackError::BufferMapFailed)?;

        match self.map_rx.try_recv() {
            Ok(result) => result.map_err(|_| RgbaReadbackError::BufferMapFailed)?,
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|_| RgbaReadbackError::BufferMapFailed)?;
                self.map_rx
                    .recv()
                    .map_err(|_| RgbaReadbackError::BufferMapFailed)?
                    .map_err(|_| RgbaReadbackError::BufferMapFailed)?;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(RgbaReadbackError::BufferMapFailed);
            }
        }

        self.copy_mapped_to_output(slot);
        Ok(())
    }

    fn map_and_strip_blocking(
        &mut self,
        gpu: &GpuContext,
        slot: usize,
    ) -> Result<(), RgbaReadbackError> {
        crate::profile_scope!("rgba_flush_blocking");
        let buffer_slice = self.staging[slot].slice(..);
        let tx = self.map_tx.clone();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| RgbaReadbackError::BufferMapFailed)?;
        self.map_rx
            .recv()
            .map_err(|_| RgbaReadbackError::BufferMapFailed)?
            .map_err(|_| RgbaReadbackError::BufferMapFailed)?;
        self.copy_mapped_to_output(slot);
        Ok(())
    }

    /// Copy the mapped staging buffer into `output[slot]`, stripping
    /// wgpu's per-row padding so the output is tightly packed.
    fn copy_mapped_to_output(&mut self, slot: usize) {
        let mapped = self.staging[slot].slice(..).get_mapped_range();
        let row_stride = self.padded_bytes_per_row as usize;
        let row_bytes = self.bytes_per_row as usize;
        let height = self.height as usize;
        let output = &mut self.output[slot];

        if row_stride == row_bytes {
            // No padding - single copy.
            output.copy_from_slice(&mapped[..row_bytes * height]);
        } else {
            for row in 0..height {
                let src = row * row_stride;
                let dst = row * row_bytes;
                output[dst..dst + row_bytes].copy_from_slice(&mapped[src..src + row_bytes]);
            }
        }
        drop(mapped);
        self.staging[slot].unmap();
    }
}

/// Errors from [`RgbaReadback`]. `Clone + Send + Sync`.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RgbaReadbackError {
    /// GPU buffer mapping failed (device lost, timeout, ...).
    #[error("RGBA buffer mapping failed")]
    BufferMapFailed,

    /// Invalid output dimensions.
    #[error("invalid RGBA dimensions: {0}")]
    InvalidDimensions(String),
}

#[cfg(test)]
mod tests {
    #[test]
    fn padded_row_stride_matches_width_when_aligned() {
        // 1920 * 4 = 7680, which is a multiple of 256.
        let padded = (1920u32 * 4).div_ceil(256) * 256;
        assert_eq!(padded, 7680);
    }

    #[test]
    fn padded_row_stride_rounds_up_unaligned_widths() {
        // 100 * 4 = 400, which rounds up to 512.
        let padded = (100u32 * 4).div_ceil(256) * 256;
        assert_eq!(padded, 512);

        // 64 * 4 = 256, already aligned.
        let padded = (64u32 * 4).div_ceil(256) * 256;
        assert_eq!(padded, 256);
    }
}
