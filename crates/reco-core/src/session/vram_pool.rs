//! VRAM texture pool for GPU-resident frame buffering.
//!
//! Holds N stereo NV12 frames in regular wgpu textures (not shared
//! CUDA/Vulkan memory). Frames are copied from the 2-slot decode
//! path into this pool, then the decode slot is freed immediately.
//! This decouples the decode surface count from the buffer depth
//! and works on any wgpu backend (Vulkan, Metal, DirectX).

#![allow(dead_code)]

use std::collections::VecDeque;

/// A single stereo NV12 frame in VRAM.
struct VramSlot {
    left_y: wgpu::Texture,
    left_uv: wgpu::Texture,
    right_y: wgpu::Texture,
    right_uv: wgpu::Texture,
    left_bind_group: wgpu::BindGroup,
    right_bind_group: wgpu::BindGroup,
}

/// Pool of VRAM-resident stereo NV12 textures for frame buffering.
pub(crate) struct VramPool {
    slots: Vec<VramSlot>,
    free: VecDeque<usize>,
    width: u32,
    height: u32,
}

impl VramPool {
    /// Allocate N stereo NV12/P010 slots in VRAM.
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
        width: u32,
        height: u32,
        n_slots: usize,
        pixel_format: crate::render::renderer::GpuPixelFormat,
    ) -> Result<Self, String> {
        let vram_bytes = estimate_vram(width, height, n_slots, pixel_format.bytes_per_sample());
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
                let left_y = create_tex(&format!("vram_L_Y_{i}"), y_format, width, height);
                let left_uv =
                    create_tex(&format!("vram_L_UV_{i}"), uv_format, width / 2, height / 2);
                let right_y = create_tex(&format!("vram_R_Y_{i}"), y_format, width, height);
                let right_uv =
                    create_tex(&format!("vram_R_UV_{i}"), uv_format, width / 2, height / 2);
                let left_bind_group =
                    pipeline.create_texture_bind_group(&left_y, &left_uv, &format!("vram_L_{i}"));
                let right_bind_group =
                    pipeline.create_texture_bind_group(&right_y, &right_uv, &format!("vram_R_{i}"));
                VramSlot {
                    left_y,
                    left_uv,
                    right_y,
                    right_uv,
                    left_bind_group,
                    right_bind_group,
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

        log::info!(
            "VramPool: {n_slots} stereo NV12 slots at {width}x{height}, ~{vram_mb:.0} MB VRAM"
        );

        Ok(Self {
            slots,
            free,
            width,
            height,
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
        src_left_y: &wgpu::Texture,
        src_left_uv: &wgpu::Texture,
        src_right_y: &wgpu::Texture,
        src_right_uv: &wgpu::Texture,
    ) {
        let dst = &self.slots[slot];
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

        copy_tex(
            &mut encoder,
            src_left_y,
            &dst.left_y,
            self.width,
            self.height,
        );
        copy_tex(
            &mut encoder,
            src_left_uv,
            &dst.left_uv,
            self.width / 2,
            self.height / 2,
        );
        copy_tex(
            &mut encoder,
            src_right_y,
            &dst.right_y,
            self.width,
            self.height,
        );
        copy_tex(
            &mut encoder,
            src_right_uv,
            &dst.right_uv,
            self.width / 2,
            self.height / 2,
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Left bind group for rendering a pool slot.
    pub fn left_bind_group(&self, slot: usize) -> &wgpu::BindGroup {
        &self.slots[slot].left_bind_group
    }

    /// Right bind group for rendering a pool slot.
    pub fn right_bind_group(&self, slot: usize) -> &wgpu::BindGroup {
        &self.slots[slot].right_bind_group
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

/// Estimate VRAM usage for N stereo slots.
///
/// `bytes_per_sample` is 1 for 8-bit NV12, 2 for 10-bit P010, so the
/// estimate is correct for both pixel formats (P010 is twice NV12).
pub fn estimate_vram(width: u32, height: u32, n_slots: usize, bytes_per_sample: usize) -> usize {
    let y_bytes = width as usize * height as usize * bytes_per_sample;
    let uv_bytes = y_bytes / 2;
    let per_frame = (y_bytes + uv_bytes) * 2; // 2 cameras
    per_frame * n_slots
}

/// Number of stereo pool slots required for `n` lookahead frames.
///
/// Mirrors the buffered-loop sizing (lookahead window + post-smoothing tail +
/// slack). Keep in sync with `run_loop`'s `pool_size` and the D3D11 staging
/// pool so the slider, the pre-flight check, and the actual allocation agree.
pub fn lookahead_pool_slots(n: usize) -> usize {
    let post_smooth_half = (n / 2).max(1);
    n + post_smooth_half + 4
}

/// Largest lookahead frame count whose pool fits within `budget_bytes`.
///
/// Inverts [`lookahead_pool_slots`] against the per-slot cost from
/// [`estimate_vram`]. Returns 0 when not even a minimal pool fits.
pub fn max_lookahead_frames(per_slot_bytes: usize, budget_bytes: usize) -> usize {
    let max_slots = budget_bytes / per_slot_bytes.max(1);
    // lookahead_pool_slots(n) ~= 1.5*n + 4 for n >= 2; invert for n.
    max_slots.saturating_sub(4) * 2 / 3
}

/// Lookahead VRAM fit thresholds, in seconds, for a source resolution and
/// VRAM budget.
///
/// `safe_secs` is a comfortable ceiling (green zone upper bound); `max_secs`
/// is the hard ceiling that still fits `budget_bytes` (red zone lower bound);
/// the band between is "tight" (yellow). The lookahead pool stores
/// source-resolution frames (they are re-rendered into the export), so the
/// cost scales with source `width`x`height`, not output resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LookaheadFit {
    /// Comfortable ceiling with headroom (green zone upper bound), seconds.
    pub safe_secs: f64,
    /// Hard ceiling that still fits the budget (red zone lower bound), seconds.
    pub max_secs: f64,
}

/// Headroom fraction for the comfortable (`safe`/green) ceiling.
const LOOKAHEAD_SAFE_FRACTION: f64 = 0.75;

/// Compute lookahead fit thresholds for a source resolution and VRAM budget.
pub fn lookahead_fit(
    width: u32,
    height: u32,
    bytes_per_sample: usize,
    budget_bytes: usize,
    fps: f64,
) -> LookaheadFit {
    let per_slot = estimate_vram(width, height, 1, bytes_per_sample);
    let fps = fps.max(1.0);
    let max_frames = max_lookahead_frames(per_slot, budget_bytes);
    let safe_frames = max_lookahead_frames(
        per_slot,
        (budget_bytes as f64 * LOOKAHEAD_SAFE_FRACTION) as usize,
    );
    LookaheadFit {
        safe_secs: safe_frames as f64 / fps,
        max_secs: max_frames as f64 / fps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_slots_match_buffered_sizing() {
        assert_eq!(lookahead_pool_slots(0), 1 + 4); // post_smooth_half clamps to 1
        assert_eq!(lookahead_pool_slots(10), 10 + 5 + 4);
        assert_eq!(lookahead_pool_slots(44), 44 + 22 + 4);
    }

    #[test]
    fn max_frames_zero_when_budget_too_small() {
        let per_slot = estimate_vram(5312, 4648, 1, 1); // ~74 MB
        assert_eq!(max_lookahead_frames(per_slot, 0), 0);
        assert_eq!(max_lookahead_frames(per_slot, per_slot), 0); // < 4 slots
    }

    #[test]
    fn max_frames_monotonic_in_budget() {
        let per_slot = estimate_vram(1920, 1080, 1, 1);
        let small = max_lookahead_frames(per_slot, 500_000_000);
        let big = max_lookahead_frames(per_slot, 4_000_000_000);
        assert!(big > small);
    }

    #[test]
    fn fit_safe_below_max() {
        // chefboyrd86's case: 5.3K (5312x4648), ~1.7 GB budget, 30 fps.
        let fit = lookahead_fit(5312, 4648, 1, 1_700_000_000, 30.0);
        assert!(fit.safe_secs <= fit.max_secs);
        // His 1.5s default does NOT fit on this budget -> lands in the red zone.
        assert!(fit.max_secs < 1.5);
    }

    #[test]
    fn fit_high_budget_allows_default() {
        // Plenty of VRAM: 1080p source, 8 GB budget -> default 1.5s is safe.
        let fit = lookahead_fit(1920, 1080, 1, 8_000_000_000, 30.0);
        assert!(fit.safe_secs >= 1.5);
    }
}
