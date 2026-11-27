//! VRAM texture pool for GPU-resident frame buffering.
//!
//! Holds N per-input NV12 frames in regular wgpu textures (not shared
//! CUDA/Vulkan memory). Frames are copied from the 2-slot decode
//! path into this pool, then the decode slot is freed immediately.
//! This decouples the decode surface count from the buffer depth
//! on the Linux CUDA/Vulkan and NVMM paths.

#![allow(dead_code)]

use std::collections::VecDeque;

/// One frame set in VRAM.
struct VramSlot {
    textures: Vec<[wgpu::Texture; 2]>,
    bind_groups: Vec<wgpu::BindGroup>,
}

/// Pool of VRAM-resident per-input NV12 textures for frame buffering.
pub(crate) struct VramPool {
    pub(crate) pixel_format: crate::render::renderer::GpuPixelFormat,
    slots: Vec<VramSlot>,
    free: VecDeque<usize>,
}

impl VramPool {
    /// Allocate N per-input NV12/P010 slots in VRAM.
    ///
    /// Allocation is the LAST line of defense: callers should run a
    /// pre-flight budget check first (see [`crate::gpu::GpuContext::available_vram`]),
    /// which fails fast with a clear message when the requested lookahead
    /// would not fit. This still guards the slip-through cases the
    /// pre-flight check cannot prevent - a budget that shifts between the
    /// check and allocation, or a backend where `available_vram` returns
    /// `None` and the check is skipped. A per-slot `catch_unwind` converts
    /// the wgpu OOM panic into a descriptive error. (Error scopes are
    /// avoided here: they deadlock once the driver is in a bad post-OOM
    /// state.)
    pub fn new(
        gpu: &crate::gpu::GpuContext,
        pipeline: &crate::render::pipeline::StitchPipeline,
        n_slots: usize,
        pixel_format: crate::render::renderer::GpuPixelFormat,
    ) -> Result<Self, String> {
        let vram_bytes = crate::lookahead::image_bytes(
            pipeline.source_sizes(),
            pixel_format.bytes_per_sample(),
        )?
        .checked_mul(n_slots)
        .ok_or("lookahead pool byte size overflow")?;
        let vram_mb = vram_bytes as f64 / (1024.0 * 1024.0);

        let y_format = pixel_format.y_format();
        let uv_format = pixel_format.uv_format();

        let mut slots = Vec::with_capacity(n_slots);
        let mut free = VecDeque::with_capacity(n_slots);

        // wgpu panics on OOM in create_texture. We use catch_unwind
        // to convert the panic into a clean error. Error scopes deadlock
        // when the driver is in a bad state from failed allocations.
        for i in 0..n_slots {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let create_tex = |label: &str, fmt, w, h| {
                    gpu.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d {
                            width: w,
                            height: h,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: fmt,
                        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                };
                let textures: Vec<_> = pipeline
                    .source_sizes()
                    .iter()
                    .map(|&(width, height)| {
                        [
                            create_tex("vram_Y", y_format, width, height),
                            create_tex("vram_UV", uv_format, width / 2, height / 2),
                        ]
                    })
                    .collect();
                let bind_groups = textures
                    .iter()
                    .map(|t| pipeline.create_texture_bind_group(&t[0], &t[1], "vram_camera"))
                    .collect();
                VramSlot {
                    textures,
                    bind_groups,
                }
            }));

            match result {
                Ok(slot) => {
                    slots.push(slot);
                    free.push_back(i);
                }
                Err(_) => {
                    return Err(format!(
                        "VRAM allocation failed at slot {i}/{n_slots} (~{vram_mb:.0} MB). \
                         Reduce --lookahead or use --no-zero-copy."
                    ));
                }
            }
        }

        log::info!("VramPool: {n_slots} per-input NV12 slots, ~{vram_mb:.0} MB VRAM");

        Ok(Self {
            slots,
            free,
            pixel_format,
        })
    }

    /// Take a free slot. Returns None if the pool is exhausted.
    pub fn acquire(&mut self) -> Option<usize> {
        self.free.pop_front()
    }

    /// Return a slot to the free list after rendering.
    pub fn release(&mut self, slot: usize) {
        debug_assert!(slot < self.slots.len(), "slot index out of bounds");
        self.free.push_back(slot);
    }

    /// Copy from source textures (Y + UV per camera) into a pool slot.
    pub fn copy_from_textures(
        &self,
        gpu: &crate::gpu::GpuContext,
        slot: usize,
        sources: &[(&wgpu::Texture, &wgpu::Texture)],
    ) {
        let dst = &self.slots[slot];
        assert_eq!(sources.len(), dst.textures.len());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vram_pool_copy"),
            });

        let copy_tex = |enc: &mut wgpu::CommandEncoder,
                        src: &wgpu::Texture,
                        dst: &wgpu::Texture,
                        w: u32,
                        h: u32| {
            enc.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: src,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: dst,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        };

        for ((y, uv), textures) in sources.iter().zip(&dst.textures) {
            let (width, height) = (textures[0].width(), textures[0].height());
            assert_eq!((y.width(), y.height()), (width, height));
            assert_eq!((uv.width(), uv.height()), (width / 2, height / 2));
            copy_tex(&mut encoder, y, &textures[0], width, height);
            copy_tex(&mut encoder, uv, &textures[1], width / 2, height / 2);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    pub(crate) fn stereo_views(&self, slot: usize) -> Option<[[wgpu::TextureView; 2]; 2]> {
        let [left, right] = self.slots[slot].textures.as_slice() else {
            return None;
        };
        Some([left, right].map(|planes| {
            planes
                .each_ref()
                .map(|t| t.create_view(&Default::default()))
        }))
    }

    /// Per-input bind groups for rendering a pool slot.
    pub fn bind_groups(&self, slot: usize) -> &[wgpu::BindGroup] {
        &self.slots[slot].bind_groups
    }

    /// Number of free slots available.
    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// Total slot count.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
}

/// VRAM budget (bytes) available to the lookahead pool on a discrete GPU,
/// from the driver's `free` figure when it is trustworthy, else a fraction of
/// `total`.
///
/// `free` is the honest signal: it already excludes the OS compositor and the
/// rest of the pipeline, so on small cards it avoids the over-estimate a flat
/// `total * fraction` makes (a fixed baseline is a large share of an 8 GB card).
/// But some drivers report a bogus ~0 free on a multi-GB card mid-session, which
/// would block a viable export, so `free` is trusted only when it is a
/// believable fraction of total; otherwise the total-based estimate is used and
/// over-allocation is caught at pool creation. Capped at 70% of `total` either
/// way.
///
/// For unified-memory GPUs (such as Apple Silicon), use
/// [`unified_lookahead_budget_bytes`] or
/// [`crate::gpu::GpuContext::lookahead_memory_budget`].
pub fn lookahead_budget_bytes(free_vram: u64, total_vram: u64) -> usize {
    crate::lookahead::discrete_vram_budget_bytes(free_vram, total_vram)
}

/// Lookahead byte budget for a unified-memory GPU (such as Apple Silicon),
/// constrained by both Metal working-set headroom and actual host memory.
///
/// Unlike [`lookahead_budget_bytes`], low host availability is real shared-RAM
/// pressure and never falls back to a high fraction of `gpu_total`.
pub fn unified_lookahead_budget_bytes(
    gpu_free: u64,
    gpu_total: u64,
    host_available: u64,
    host_total: u64,
) -> usize {
    crate::lookahead::unified_memory_budget_bytes(gpu_free, gpu_total, host_available, host_total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bogus_free_falls_back_to_total() {
        // zzz's case: 5.52 GB total, driver reports a bogus free=0. The budget
        // must not collapse to ~0; it falls back to the total-based estimate.
        let total = 5_520_000_000u64;
        let budget = lookahead_budget_bytes(0, total);
        assert!(budget > 0);
        assert!(budget < total as usize); // reserves headroom
        // The 2.49 GB pool from the forum report still fits.
        assert!(budget >= 2_490_000_000);
        // Falls back to exactly the total-based estimate.
        assert_eq!(budget, (total as f64 * 0.70) as usize);
    }

    #[test]
    fn clean_small_card_trusts_free_no_false_green() {
        // Problem 2: a clean 8 GB card. DXGI budget ~7.6 GB total, but a fixed
        // baseline (compositor + pipeline) leaves ~4.5 GB actually free. The old
        // total*0.70 = 5.33 GB over-estimated and let a 4.71 GB pool pass
        // pre-flight, then OOM. The free-trusting budget must stay below the
        // real free so that pool is correctly rejected.
        let total = 7_600_000_000u64;
        let free = 4_500_000_000u64;
        let budget = lookahead_budget_bytes(free, total);
        // Below total*0.70 - free constrained it.
        assert!(budget < (total as f64 * 0.70) as usize);
        // Below the actual free (reserve held back), so it never promises more
        // than the card has.
        assert!(budget < free as usize);
        // The 4.71 GB default pool does NOT fit -> no false green.
        assert!(budget < 4_710_000_000);
    }

    #[test]
    fn big_card_capped_at_total_fraction() {
        // 24 GB card, 22 GB free. free - reserve (21.5 GB) exceeds the total
        // ceiling, so the pool is capped at total*0.70, not the huge free.
        let total = 24_000_000_000u64;
        let free = 22_000_000_000u64;
        let budget = lookahead_budget_bytes(free, total);
        assert_eq!(budget, (total as f64 * 0.70) as usize);
    }
}
