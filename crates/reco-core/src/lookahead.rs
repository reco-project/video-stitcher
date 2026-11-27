//! Checked storage accounting for lookahead scheduling.

#[cfg(any(
    all(feature = "gpu", any(target_os = "linux", target_os = "windows")),
    test
))]
pub(crate) fn pool_slots(frames: usize) -> Result<usize, String> {
    // Immediate D3D11 submission uses two alternating slots. Buffered runs
    // retain the frame being presented plus exactly the requested future frames.
    if frames == 0 {
        return Ok(2);
    }
    frames
        .checked_add(1)
        .ok_or_else(|| "lookahead pool capacity overflow".into())
}

#[cfg(any(feature = "gpu", test))]
pub(crate) fn image_bytes(sizes: &[(u32, u32)], bytes_per_sample: usize) -> Result<usize, String> {
    sizes.iter().try_fold(0usize, |total, &(width, height)| {
        let y = (width as usize).checked_mul(height as usize);
        let chroma = (width as usize / 2)
            .checked_mul(height as usize / 2)
            .and_then(|n| n.checked_mul(2));
        y.zip(chroma)
            .and_then(|(y, uv)| y.checked_add(uv))
            .and_then(|n| n.checked_mul(bytes_per_sample))
            .and_then(|n| total.checked_add(n))
            .ok_or_else(|| "lookahead image byte size overflow".into())
    })
}

pub(crate) fn max_frames(per_slot: usize, budget: usize) -> usize {
    if per_slot == 0 {
        return 0;
    }
    (budget / per_slot).saturating_sub(1)
}

/// Ceiling fraction of *total* dedicated VRAM the lookahead pool may use.
///
/// Acts as an upper bound (and the fallback when `free` is unusable). The pool
/// competes with decode surfaces, the stitch pipeline, the encoder, AI tensors,
/// and the OS compositor, so the pool is never allowed past this fraction of
/// total even when the driver reports more free.
#[cfg(any(feature = "gpu", test))]
pub(crate) const DISCRETE_BUDGET_FRACTION: f64 = 0.70;

/// Runtime margin held back below a discrete GPU driver's reported "free" VRAM.
///
/// `free` is sampled just before the pool allocates, but encoder output
/// surfaces, extra decode buffering, and AI tensors still grow during the run.
#[cfg(any(feature = "gpu", test))]
pub(crate) const DISCRETE_FREE_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

/// Ceiling fraction of *total* physical RAM a unified-memory lookahead window
/// may occupy (e.g. Apple Silicon).
///
/// Lookahead frames share physical RAM with the OS, WindowServer, hardware
/// decoders/encoders, AI inference buffers, and the Metal render pipeline.
/// Capping at 25% of physical RAM keeps even an idle 16 GB machine from
/// allocating half its RAM to buffered frames before decoders and encoders
/// reach steady state.
#[cfg(any(feature = "gpu", test))]
pub(crate) const UNIFIED_TOTAL_BUDGET_FRACTION: f64 = 0.25;

/// Maximum share of currently available host memory a unified-memory lookahead
/// window may claim, leaving at least half of spare RAM for the rest of the
/// pipeline and OS.
#[cfg(any(feature = "gpu", test))]
pub(crate) const UNIFIED_AVAILABLE_BUDGET_FRACTION: f64 = 0.50;

/// Fixed host-memory headroom held back on unified-memory systems for runtime
/// decoder/encoder surface growth, AI runtime scratch buffers, and VM overhead
/// so lookahead retention does not push macOS into memory-compressor thrashing.
#[cfg(any(feature = "gpu", test))]
pub(crate) const UNIFIED_HOST_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Lookahead byte budget for a discrete GPU from driver-reported `(free, total)` VRAM.
///
/// `free` is the honest signal: it already excludes the OS compositor and the
/// rest of the pipeline, so on small cards it avoids the over-estimate a flat
/// `total * fraction` makes. Some discrete drivers report a bogus ~0 free on a
/// multi-GB card mid-session, so `free` is trusted only when it is at least
/// `total / 8`; otherwise the total-based ceiling is used and over-allocation
/// is caught at pool creation.
#[cfg(any(feature = "gpu", test))]
pub(crate) fn discrete_vram_budget_bytes(free_vram: u64, total_vram: u64) -> usize {
    let total_based = (total_vram as f64 * DISCRETE_BUDGET_FRACTION) as u64;
    let budget = if free_vram >= total_vram / 8 {
        free_vram
            .saturating_sub(DISCRETE_FREE_RESERVE_BYTES)
            .min(total_based)
    } else {
        total_based
    };
    usize::try_from(budget).unwrap_or(usize::MAX)
}

/// Lookahead byte budget for a unified-memory GPU (such as Apple Silicon)
/// bounded by both Metal's working-set headroom and actual host memory.
///
/// Unlike discrete VRAM, low `host_available` reflects real shared-RAM
/// pressure and must never fall back to a high fraction of Metal's
/// `recommendedMaxWorkingSetSize`.
#[cfg(any(feature = "gpu", test))]
pub(crate) fn unified_memory_budget_bytes(
    gpu_free: u64,
    gpu_total: u64,
    host_available: u64,
    host_total: u64,
) -> usize {
    let gpu_headroom = gpu_free
        .saturating_sub(DISCRETE_FREE_RESERVE_BYTES)
        .min((gpu_total as f64 * DISCRETE_BUDGET_FRACTION) as u64);
    let host_total_cap = (host_total as f64 * UNIFIED_TOTAL_BUDGET_FRACTION) as u64;
    let host_avail_cap = (host_available as f64 * UNIFIED_AVAILABLE_BUDGET_FRACTION) as u64;
    let host_reserve_cap = host_available.saturating_sub(UNIFIED_HOST_RESERVE_BYTES);
    let host_budget = host_total_cap.min(host_avail_cap).min(host_reserve_cap);
    let budget = gpu_headroom.min(host_budget);
    usize::try_from(budget).unwrap_or(usize::MAX)
}

/// Snapshot of macOS Mach `vm_statistics64` page counters and physical RAM size.
#[cfg(any(
    all(feature = "gpu", any(target_os = "macos", target_os = "ios")),
    test
))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MacosVmSample {
    pub free_count: u32,
    pub speculative_count: u32,
    pub external_page_count: u32,
    pub purgeable_count: u32,
    pub internal_page_count: u32,
    pub wire_count: u32,
    pub compressor_page_count: u32,
    pub page_size: u64,
    pub host_total: u64,
}

/// Compute conservatively reclaimable host memory in bytes from a macOS
/// `vm_statistics64` sample.
///
/// Counts truly free pages (`free_count - speculative_count`, since XNU nests
/// speculative readahead inside `free_count`) plus clean file-backed cache
/// (`external_page_count`) and volatile discardable pages (`purgeable_count`),
/// capped at `host_total - (app + wired + compressed)`. Active/inactive
/// anonymous app memory, wired kernel/driver memory, and compressed pages are
/// excluded so lookahead retention does not induce compressor thrashing.
#[cfg(any(
    all(feature = "gpu", any(target_os = "macos", target_os = "ios")),
    test
))]
pub(crate) fn macos_vm_available_bytes(sample: MacosVmSample) -> u64 {
    let free_pages =
        u64::from(sample.free_count).saturating_sub(u64::from(sample.speculative_count));
    let direct_avail = free_pages
        .saturating_add(u64::from(sample.external_page_count))
        .saturating_add(u64::from(sample.purgeable_count))
        .saturating_mul(sample.page_size);

    let app_pages =
        u64::from(sample.internal_page_count).saturating_sub(u64::from(sample.purgeable_count));
    let used_bytes = app_pages
        .saturating_add(u64::from(sample.wire_count))
        .saturating_add(u64::from(sample.compressor_page_count))
        .saturating_mul(sample.page_size);
    let unused_ram = sample.host_total.saturating_sub(used_bytes);

    direct_avail.min(unused_ram)
}

/// Tracks owned image allocations in the lookahead buffer.
pub(crate) struct ImageBudget {
    retained: usize,
    limit: usize,
}

impl ImageBudget {
    pub(crate) fn new(limit: usize) -> Result<Self, String> {
        if limit == 0 {
            return Err("enabled lookahead requires a nonzero image-memory budget".into());
        }
        Ok(Self { retained: 0, limit })
    }

    pub(crate) fn retain(&mut self, bytes: usize) -> Result<(), String> {
        let required = self
            .retained
            .checked_add(bytes)
            .ok_or_else(|| "lookahead retained byte size overflow".to_string())?;
        if required > self.limit {
            return Err(format!(
                "lookahead image storage needs {required} bytes, exceeding its {} byte budget; reduce lookahead or source resolution, or explicitly increase the image-memory budget",
                self.limit
            ));
        }
        self.retained = required;
        Ok(())
    }

    pub(crate) fn release(&mut self, bytes: usize) {
        self.retained = self
            .retained
            .checked_sub(bytes)
            .expect("released unaccounted lookahead bytes");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn checked_sizes_and_exact_fit() {
        assert_eq!(pool_slots(0).unwrap(), 2);
        assert_eq!(pool_slots(1).unwrap(), 2);
        assert!(pool_slots(usize::MAX).is_err());
        assert!(image_bytes(&[(u32::MAX, u32::MAX)], 2).is_err());
        assert_eq!(
            image_bytes(&[(128, 96), (256, 192)], 1).unwrap(),
            (128 * 96 + 256 * 192) * 3 / 2
        );
        assert_eq!(
            image_bytes(&[(128, 96), (256, 192)], 2).unwrap(),
            (128 * 96 + 256 * 192) * 3
        );
        for cameras in [1, 2, 3] {
            assert_eq!(
                image_bytes(&vec![(128, 96); cameras], 2).unwrap(),
                128 * 96 * 3 * cameras
            );
        }
        for n in 1..100 {
            let bytes = pool_slots(n).unwrap() * 7;
            assert_eq!(max_frames(7, bytes), n);
            assert_eq!(max_frames(7, bytes - 1), n - 1);
        }
    }

    #[test]
    fn exact_limit_rejection_does_not_change_accounting() {
        assert!(ImageBudget::new(0).is_err());
        let mut b = ImageBudget::new(20).unwrap();
        b.retain(12).unwrap();
        assert!(b.retain(9).is_err());
        b.retain(8).unwrap();
        b.release(12);
        b.retain(12).unwrap();
        assert!(b.retain(usize::MAX).is_err());
        b.release(20);
        assert_eq!(b.retained, 0);
    }

    #[test]
    fn unified_memory_16gb_bounds_dual_8k_10bit_lookahead() {
        // Issue #469: 16 GiB Apple Silicon (M5), Metal reports ~12.7 GB
        // recommendedMaxWorkingSetSize and ~12.5 GB "free", while dual 8K
        // (7680x4320) 10-bit P010 frames are ~199 MB per slot.
        let per_slot = image_bytes(&[(7680, 4320), (7680, 4320)], 2).unwrap();
        assert_eq!(per_slot, 199_065_600);

        let host_total = 16 * GIB;
        let metal_total = 12_700_000_000u64;
        let metal_free = 12_500_000_000u64;

        // Typical active 16 GB desktop (~6 GiB free+reclaimable):
        // Metal-only discrete formula would allow ~8.89 GB (oversubscribing RAM);
        // unified formula clamps to 3 GiB (~3.22 GB).
        let metal_only = discrete_vram_budget_bytes(metal_free, metal_total);
        assert!(metal_only > 8_000_000_000);

        let typical_budget =
            unified_memory_budget_bytes(metal_free, metal_total, 6 * GIB, host_total);
        assert_eq!(typical_budget, (3 * GIB) as usize);
        assert!(typical_budget < metal_only);

        // Single bounded window (N + 1 slots):
        // - 0.1s @ 60fps (6 frames -> 7 slots = 1.39 GB) fits comfortably
        // - 0.2s @ 60fps (12 frames -> 13 slots = 2.59 GB, measured ~52.6 fps) fits
        // - 0.4s @ 60fps (24 frames -> 25 slots = 4.98 GB) is rejected
        assert!(pool_slots(6).unwrap() * per_slot <= typical_budget);
        assert!(pool_slots(12).unwrap() * per_slot <= typical_budget);
        assert!(pool_slots(24).unwrap() * per_slot > typical_budget);
        assert_eq!(max_frames(per_slot, typical_budget), 15);

        // Even on an idle 16 GB machine (10 GiB available), the 25% physical-RAM
        // ceiling caps the budget at 4 GiB so 0.4s dual-8K (4.98 GB) still cannot
        // oversubscribe unified memory once decoders/encoders spin up.
        let idle_budget =
            unified_memory_budget_bytes(metal_free, metal_total, 10 * GIB, host_total);
        assert_eq!(idle_budget, (4 * GIB) as usize);
        assert!(pool_slots(24).unwrap() * per_slot > idle_budget);
        assert_eq!(max_frames(per_slot, idle_budget), 20);

        // Under memory pressure (<= 2 GiB available), the reserve clamps budget
        // to zero instead of falling back to Metal's working-set limit.
        assert_eq!(
            unified_memory_budget_bytes(metal_free, metal_total, 2 * GIB, host_total),
            0
        );
        assert_eq!(
            unified_memory_budget_bytes(metal_free, metal_total, GIB, host_total),
            0
        );
        assert_eq!(
            unified_memory_budget_bytes(metal_free, metal_total, 0, host_total),
            0
        );
    }

    #[test]
    fn unified_memory_scales_across_24gb_32gb_64gb_and_respects_metal_headroom() {
        let per_slot_8k_10bit = image_bytes(&[(7680, 4320), (7680, 4320)], 2).unwrap();

        // 24 GiB unified memory, 14 GiB host available -> 25% cap = 6 GiB (31 frames = 0.52s)
        let b24 = unified_memory_budget_bytes(17 * GIB, 18 * GIB, 14 * GIB, 24 * GIB);
        assert_eq!(b24, (6 * GIB) as usize);
        assert!(pool_slots(24).unwrap() * per_slot_8k_10bit <= b24);
        assert_eq!(max_frames(per_slot_8k_10bit, b24), 31);

        // 32 GiB unified memory, 20 GiB host available -> 25% cap = 8 GiB (42 frames = 0.70s)
        let b32 = unified_memory_budget_bytes(23 * GIB, 24 * GIB, 20 * GIB, 32 * GIB);
        assert_eq!(b32, (8 * GIB) as usize);
        assert_eq!(max_frames(per_slot_8k_10bit, b32), 42);

        // 64 GiB unified memory, 44 GiB host available -> 25% cap = 16 GiB (85 frames = 1.42s)
        let b64 = unified_memory_budget_bytes(47 * GIB, 48 * GIB, 44 * GIB, 64 * GIB);
        assert_eq!(b64, (16 * GIB) as usize);
        assert_eq!(max_frames(per_slot_8k_10bit, b64), 85);

        // Strictly monotonic with machine RAM when host availability is ample.
        assert!(b24 < b32 && b32 < b64);

        // When Metal's own working-set headroom is tighter than host RAM, Metal caps it.
        let metal_tight = unified_memory_budget_bytes(2 * GIB, 24 * GIB, 20 * GIB, 32 * GIB);
        assert_eq!(
            metal_tight,
            (2 * GIB - DISCRETE_FREE_RESERVE_BYTES) as usize
        );

        // 8 GiB unified memory with dual 4K 8-bit NV12 (~24.9 MB/slot) still allows >1s @ 60fps.
        let per_slot_4k_8bit = image_bytes(&[(3840, 2160), (3840, 2160)], 1).unwrap();
        let b8 = unified_memory_budget_bytes(5 * GIB, 5 * GIB + GIB / 3, 7 * GIB / 2, 8 * GIB);
        assert_eq!(b8, (3 * GIB / 2) as usize); // 3.5 GiB - 2 GiB reserve = 1.5 GiB
        assert!(max_frames(per_slot_4k_8bit, b8) >= 60);
    }

    #[test]
    fn discrete_gpu_budget_preserves_dedicated_vram_policy() {
        // 16 GiB discrete GPU with 14 GiB free -> capped at 70% of dedicated VRAM (11.2 GiB),
        // substantially higher than a 16 GiB unified-memory Mac (<= 4 GiB).
        let discrete_16 = discrete_vram_budget_bytes(14 * GIB, 16 * GIB);
        let unified_16 = unified_memory_budget_bytes(12 * GIB, 12 * GIB, 10 * GIB, 16 * GIB);
        assert_eq!(discrete_16, (16.0 * GIB as f64 * 0.70) as usize);
        assert!(discrete_16 > unified_16 * 2);

        // Small 8 GB discrete card: trusts free minus 512 MiB reserve.
        let discrete_8 = discrete_vram_budget_bytes(4_500_000_000, 7_600_000_000);
        assert_eq!(
            discrete_8,
            (4_500_000_000 - DISCRETE_FREE_RESERVE_BYTES) as usize
        );

        // Bogus free=0 on a discrete driver still falls back to 70% of total.
        let bogus_free = discrete_vram_budget_bytes(0, 5_520_000_000);
        assert_eq!(bogus_free, (5_520_000_000f64 * 0.70) as usize);
    }

    #[test]
    fn macos_vm_available_excludes_app_wired_compressed_and_speculative() {
        let page = 16_384u64; // Apple Silicon 16 KiB pages
        let total = 16 * GIB; // 1,048,576 pages

        // Typical 16 GiB Apple Silicon state:
        // free=180k (includes speculative=10k -> 170k truly free),
        // external=250k, purgeable=20k (subset of internal=320k -> 300k app),
        // wired=240k, compressor=60k.
        let typical = macos_vm_available_bytes(MacosVmSample {
            free_count: 180_000,
            speculative_count: 10_000,
            external_page_count: 250_000,
            purgeable_count: 20_000,
            internal_page_count: 320_000,
            wire_count: 240_000,
            compressor_page_count: 60_000,
            page_size: page,
            host_total: total,
        });
        assert_eq!(typical, (170_000 + 250_000 + 20_000) * page);

        // Compressor thrashing (issue #469 vm_stat): ~5k free, heavy compressor/wired.
        let thrashing = macos_vm_available_bytes(MacosVmSample {
            free_count: 5_000,
            speculative_count: 500,
            external_page_count: 10_000,
            purgeable_count: 1_000,
            internal_page_count: 500_000,
            wire_count: 250_000,
            compressor_page_count: 280_000,
            page_size: page,
            host_total: total,
        });
        assert_eq!(thrashing, (4_500 + 10_000 + 1_000) * page);
        assert!(thrashing < 512 * 1024 * 1024);

        // Non-atomic counter race: speculative_count > free_count and
        // purgeable_count > internal_page_count must saturate cleanly without underflow.
        let raced = macos_vm_available_bytes(MacosVmSample {
            free_count: 1_000,
            speculative_count: 5_000,
            external_page_count: 20_000,
            purgeable_count: 10_000,
            internal_page_count: 5_000,
            wire_count: 200_000,
            compressor_page_count: 0,
            page_size: page,
            host_total: total,
        });
        assert_eq!(raced, (20_000 + 10_000) * page);

        // When used pages exceed host_total, availability clamps to 0.
        let oversubscribed = macos_vm_available_bytes(MacosVmSample {
            free_count: 100_000,
            speculative_count: 0,
            external_page_count: 100_000,
            purgeable_count: 0,
            internal_page_count: 600_000,
            wire_count: 400_000,
            compressor_page_count: 200_000,
            page_size: page,
            host_total: total,
        });
        assert_eq!(oversubscribed, 0);
    }
}
