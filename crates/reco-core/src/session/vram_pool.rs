//! VRAM texture pool for GPU-resident frame buffering.
//!
//! Holds N stereo NV12 frames in regular wgpu textures (not shared
//! CUDA/Vulkan memory). Frames are copied from the 2-slot decode
//! path into this pool, then the decode slot is freed immediately.
//! This decouples the decode surface count from the buffer depth
//! and works on any wgpu backend (Vulkan, Metal, DirectX).

#![allow(dead_code)]

use std::collections::VecDeque;

/// Whether the lookahead pool buffers frames at the source's native pixel
/// format or downconverts to 8-bit NV12 first.
///
/// The lookahead pool's job is to hold N future decoded stereo frames so
/// the AI panner/tracker can smooth camera trajectory using future world-
/// state - but the *same* buffered frames are also what the final stitch
/// render consumes once it catches up (this is not an AI-only side
/// buffer). For 10-bit sources (`GpuPixelFormat::P010`, e.g. DJI Action 4
/// HEVC), each pool slot costs 2x the bytes of an 8-bit source, which is
/// what makes deep lookahead windows VRAM-expensive on lower-VRAM cards.
/// `Reduced8Bit` roughly halves that cost by discarding the source's extra
/// bit depth before it enters the pool - a genuine (if subtle) whole-
/// export quality tradeoff (more banding risk in smooth gradients: sky,
/// pitch grass, floodlit surfaces), not a free lunch. See FRICTION.md
/// "Lookahead pool VRAM cost scales with source bit depth".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LookaheadBitDepth {
    /// Buffer frames at the source's native bit depth (current/default
    /// behavior, unchanged from before this option existed).
    #[default]
    Native,
    /// Downconvert to 8-bit NV12 before buffering. No effect when the
    /// source is already 8-bit.
    Reduced8Bit,
}

impl LookaheadBitDepth {
    /// The pixel format the pool's own textures should be allocated in,
    /// given the source's actual format.
    pub fn pool_format(
        self,
        source_format: crate::render::renderer::GpuPixelFormat,
    ) -> crate::render::renderer::GpuPixelFormat {
        match self {
            Self::Native => source_format,
            Self::Reduced8Bit => crate::render::renderer::GpuPixelFormat::Nv12,
        }
    }
}

/// A single stereo frame in VRAM, at the pool's `pool_format`.
struct VramSlot {
    left_y: wgpu::Texture,
    left_uv: wgpu::Texture,
    right_y: wgpu::Texture,
    right_uv: wgpu::Texture,
    left_bind_group: wgpu::BindGroup,
    right_bind_group: wgpu::BindGroup,
}

/// Pool of VRAM-resident stereo NV12/P010 textures for frame buffering.
pub(crate) struct VramPool {
    slots: Vec<VramSlot>,
    free: VecDeque<usize>,
    width: u32,
    height: u32,
    /// Pixel format of the source textures passed to `copy_from_textures`.
    source_format: crate::render::renderer::GpuPixelFormat,
    /// Pixel format the pool's own slot textures are allocated in. Equal to
    /// `source_format` unless `LookaheadBitDepth::Reduced8Bit` was
    /// requested for a 10-bit source, in which case `copy_from_textures`
    /// runs a GPU downconvert pass (see `downconverter`) instead of a raw
    /// texture-to-texture copy.
    pool_format: crate::render::renderer::GpuPixelFormat,
    /// Built only when `pool_format != source_format` - a plain same-
    /// format copy needs no shader pass.
    downconverter: Option<crate::render::lookahead_downconvert::LookaheadDownconverter>,
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
    ///
    /// `source_format` is the pixel format of the textures that will be
    /// passed to `copy_from_textures` (the decode/import format). `bit_depth`
    /// selects whether the pool stores frames at that format unchanged
    /// (`Native`) or downconverts to 8-bit NV12 first (`Reduced8Bit`) - see
    /// [`LookaheadBitDepth`].
    pub fn new(
        gpu: &crate::gpu::GpuContext,
        pipeline: &crate::render::pipeline::StitchPipeline,
        width: u32,
        height: u32,
        n_slots: usize,
        source_format: crate::render::renderer::GpuPixelFormat,
        bit_depth: LookaheadBitDepth,
    ) -> Result<Self, String> {
        let pool_format = bit_depth.pool_format(source_format);
        let vram_bytes = estimate_vram(width, height, n_slots, pool_format.bytes_per_sample());
        let vram_mb = vram_bytes as f64 / (1024.0 * 1024.0);

        let y_format = pool_format.y_format();
        let uv_format = pool_format.uv_format();
        let downconverter = if pool_format != source_format {
            Some(crate::render::lookahead_downconvert::LookaheadDownconverter::new(gpu))
        } else {
            None
        };

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
                        // RENDER_ATTACHMENT is only exercised when
                        // `downconverter` is Some (the downconvert render
                        // pass writes into these textures); harmless to
                        // request unconditionally otherwise.
                        usage: wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::RENDER_ATTACHMENT,
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
            "VramPool: {n_slots} stereo {pool_format:?} slots at {width}x{height}, \
             ~{vram_mb:.0} MB VRAM{}",
            if downconverter.is_some() {
                format!(" (downconverted from source {source_format:?})")
            } else {
                String::new()
            }
        );

        Ok(Self {
            slots,
            free,
            width,
            height,
            source_format,
            pool_format,
            downconverter,
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
    ///
    /// `src_*` must be in `self.source_format`. When the pool was built
    /// with `LookaheadBitDepth::Native` (the common case), this is a plain
    /// GPU-to-GPU `copy_texture_to_texture` - bit-exact, no shader
    /// involved, identical to this method's behavior before
    /// `LookaheadBitDepth` existed. When built with `Reduced8Bit` for a
    /// 10-bit source, it instead runs the `LookaheadDownconverter` render
    /// pass per plane (see `render::lookahead_downconvert`).
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

        match &self.downconverter {
            None => {
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
            }
            Some(downconverter) => {
                let view =
                    |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
                let (src_ly, src_luv, src_ry, src_ruv) = (
                    view(src_left_y),
                    view(src_left_uv),
                    view(src_right_y),
                    view(src_right_uv),
                );
                let (dst_ly, dst_luv, dst_ry, dst_ruv) = (
                    view(&dst.left_y),
                    view(&dst.left_uv),
                    view(&dst.right_y),
                    view(&dst.right_uv),
                );
                downconverter.convert_plane(&gpu.device, &mut encoder, &src_ly, &dst_ly, false);
                downconverter.convert_plane(&gpu.device, &mut encoder, &src_luv, &dst_luv, true);
                downconverter.convert_plane(&gpu.device, &mut encoder, &src_ry, &dst_ry, false);
                downconverter.convert_plane(&gpu.device, &mut encoder, &src_ruv, &dst_ruv, true);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Copy a stereo frame from D3D11-imported multi-planar textures
    /// (Windows only) into a pool slot.
    ///
    /// Unlike [`Self::copy_from_textures`] (Linux/macOS, where the source
    /// is already two separate single-plane textures per camera), the
    /// D3D11 source is one combined multi-planar texture per camera with
    /// `TextureAspect::Plane0`/`Plane1` selecting Y/UV - see
    /// [`crate::interop::d3d11::D3d11StagingPool::plane_source`]. When the
    /// pool's format matches the source's, this does a plane-aspect-
    /// selected `copy_texture_to_texture` (bit-exact, no shader - a raw
    /// copy needs the source `texture` + aspect, not a view, which is why
    /// this is a separate method from `copy_from_textures` rather than a
    /// shared helper). When built with `LookaheadBitDepth::Reduced8Bit`,
    /// runs the same `LookaheadDownconverter` render pass
    /// `copy_from_textures` uses, via the pre-built plane views (a render
    /// pass only needs a view).
    pub fn copy_from_d3d11(
        &self,
        gpu: &crate::gpu::GpuContext,
        slot: usize,
        src_left: crate::interop::d3d11::D3d11PlaneSource<'_>,
        src_right: crate::interop::d3d11::D3d11PlaneSource<'_>,
    ) {
        let dst = &self.slots[slot];
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vram_pool_copy_from_d3d11"),
            });

        match &self.downconverter {
            None => {
                let copy = |enc: &mut wgpu::CommandEncoder,
                            src_tex: &wgpu::Texture,
                            aspect: wgpu::TextureAspect,
                            dst_tex: &wgpu::Texture,
                            w: u32,
                            h: u32| {
                    enc.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: src_tex,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: dst_tex,
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
                copy(
                    &mut encoder,
                    src_left.texture,
                    wgpu::TextureAspect::Plane0,
                    &dst.left_y,
                    self.width,
                    self.height,
                );
                copy(
                    &mut encoder,
                    src_left.texture,
                    wgpu::TextureAspect::Plane1,
                    &dst.left_uv,
                    self.width / 2,
                    self.height / 2,
                );
                copy(
                    &mut encoder,
                    src_right.texture,
                    wgpu::TextureAspect::Plane0,
                    &dst.right_y,
                    self.width,
                    self.height,
                );
                copy(
                    &mut encoder,
                    src_right.texture,
                    wgpu::TextureAspect::Plane1,
                    &dst.right_uv,
                    self.width / 2,
                    self.height / 2,
                );
            }
            Some(downconverter) => {
                let view =
                    |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
                let (dst_ly, dst_luv, dst_ry, dst_ruv) = (
                    view(&dst.left_y),
                    view(&dst.left_uv),
                    view(&dst.right_y),
                    view(&dst.right_uv),
                );
                downconverter.convert_plane(
                    &gpu.device,
                    &mut encoder,
                    src_left.y_view,
                    &dst_ly,
                    false,
                );
                downconverter.convert_plane(
                    &gpu.device,
                    &mut encoder,
                    src_left.uv_view,
                    &dst_luv,
                    true,
                );
                downconverter.convert_plane(
                    &gpu.device,
                    &mut encoder,
                    src_right.y_view,
                    &dst_ry,
                    false,
                );
                downconverter.convert_plane(
                    &gpu.device,
                    &mut encoder,
                    src_right.uv_view,
                    &dst_ruv,
                    true,
                );
            }
        }

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

/// Ceiling fraction of *total* VRAM the lookahead pool may use.
///
/// Acts as an upper bound (and the fallback when `free` is unusable). The pool
/// competes with decode surfaces, the stitch pipeline, the encoder, AI tensors,
/// and the OS compositor, so the pool is never allowed past this fraction of
/// total even when the driver reports more free.
const LOOKAHEAD_BUDGET_FRACTION: f64 = 0.70;

/// Runtime margin held back below the driver's "free" figure.
///
/// `free` is sampled just before the pool allocates, but the encoder output
/// surfaces, extra decode buffering, and AI tensors still grow during the run.
/// This fixed margin leaves room for that growth so the pool does not consume
/// the last byte of free VRAM.
const LOOKAHEAD_FREE_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

/// VRAM budget (bytes) available to the lookahead pool, from the driver's
/// `free` figure when it is trustworthy, else a fraction of `total`.
///
/// `free` is the honest signal: it already excludes the OS compositor and the
/// rest of the pipeline, so on small cards it avoids the over-estimate a flat
/// `total * fraction` makes (a fixed baseline is a large share of an 8 GB card).
/// But some drivers report a bogus ~0 free on a multi-GB card mid-session, which
/// would block a viable export, so `free` is trusted only when it is a
/// believable fraction of total; otherwise the total-based estimate is used and
/// over-allocation is caught gracefully at pool creation. Capped at
/// `total * LOOKAHEAD_BUDGET_FRACTION` either way. Shared by the export
/// pre-flight check and the GUI risk slider so the two stay consistent.
pub fn lookahead_budget_bytes(free_vram: u64, total_vram: u64) -> usize {
    let total_based = (total_vram as f64 * LOOKAHEAD_BUDGET_FRACTION) as u64;
    let budget = if free_vram >= total_vram / 8 {
        free_vram
            .saturating_sub(LOOKAHEAD_FREE_RESERVE_BYTES)
            .min(total_based)
    } else {
        total_based
    };
    budget as usize
}

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
    fn bit_depth_native_keeps_source_format() {
        use crate::render::renderer::GpuPixelFormat;
        assert_eq!(
            LookaheadBitDepth::Native.pool_format(GpuPixelFormat::P010),
            GpuPixelFormat::P010
        );
        assert_eq!(
            LookaheadBitDepth::Native.pool_format(GpuPixelFormat::Nv12),
            GpuPixelFormat::Nv12
        );
    }

    #[test]
    fn bit_depth_reduced_always_yields_nv12() {
        use crate::render::renderer::GpuPixelFormat;
        assert_eq!(
            LookaheadBitDepth::Reduced8Bit.pool_format(GpuPixelFormat::P010),
            GpuPixelFormat::Nv12
        );
        // No-op for sources that are already 8-bit.
        assert_eq!(
            LookaheadBitDepth::Reduced8Bit.pool_format(GpuPixelFormat::Nv12),
            GpuPixelFormat::Nv12
        );
    }

    #[test]
    fn bit_depth_default_is_native() {
        assert_eq!(LookaheadBitDepth::default(), LookaheadBitDepth::Native);
    }

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
