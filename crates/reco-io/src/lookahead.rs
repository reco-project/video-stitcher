//! Admission policy for lookahead image storage, sampled once per job.

#[cfg(target_os = "linux")]
mod cgroup;

/// Choose an image allowance, preserving an explicit caller override.
///
/// Host storage (including GPU upload paths) uses half the currently available
/// RAM, limited by the process's visible Linux cgroup ancestors. Unresolved
/// Linux hierarchies require an explicit allowance. This leaves room
/// for decoders, output and inference but is not a process-memory guarantee.
/// On macOS, available RAM is capped at total minus used memory to exclude
/// active application memory that the OS could reclaim through compression.
/// Unknown availability is an error, never an unlimited fallback. Explicit
/// allowances do not bypass the session's native driver admission check.
pub fn image_budget(explicit_bytes: Option<usize>) -> Result<usize, String> {
    if let Some(bytes) = explicit_bytes {
        return nonzero(bytes);
    }
    let mut system = sysinfo::System::new();
    system.refresh_memory_specifics(sysinfo::MemoryRefreshKind::nothing().with_ram());
    let available = system.available_memory();
    #[cfg(target_os = "macos")]
    let available = macos_available_memory(available, system.total_memory(), system.used_memory());
    #[cfg(target_os = "linux")]
    let available = cgroup::available_memory(available).map_err(|error| {
        format!("cannot resolve lookahead cgroup memory allowance: {error}; provide an explicit byte budget")
    })?;
    let bytes = usize::try_from(available / 2).unwrap_or(usize::MAX);
    nonzero(bytes)
}

#[cfg(any(target_os = "macos", test))]
fn macos_available_memory(available: u64, total: u64, used: u64) -> u64 {
    // Apple's pressure metric includes active anonymous pages. Image retention
    // must not budget those as spare RAM; sysinfo's used count includes them,
    // wired memory and compressed storage, while excluding reclaimable caches.
    available.min(total.saturating_sub(used))
}

/// Choose a native image allowance from driver memory, falling back to host policy when unavailable. An explicit override still applies and does not bypass native admission checks.
#[cfg(feature = "gpu")]
pub fn native_image_budget(
    gpu: &reco_core::gpu::GpuContext,
    explicit_bytes: Option<usize>,
) -> Result<usize, String> {
    if let Some(bytes) = explicit_bytes {
        return nonzero(bytes);
    }
    match gpu.available_vram() {
        Some((free, total)) => nonzero(reco_core::gpu::vram_pool::lookahead_budget_bytes(
            free, total,
        )),
        None => image_budget(None),
    }
}

fn nonzero(bytes: usize) -> Result<usize, String> {
    if bytes == 0 {
        Err("no lookahead image-memory allowance is available; provide an explicit nonzero byte budget or disable lookahead".into())
    } else {
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_budget_does_not_probe_or_change_the_request() {
        assert_eq!(image_budget(Some(123)).unwrap(), 123);
        assert!(image_budget(Some(0)).is_err());
    }

    #[test]
    fn macos_budget_excludes_used_memory_without_increasing_availability() {
        assert_eq!(macos_available_memory(8_000, 16_000, 14_000), 2_000);
        assert_eq!(macos_available_memory(1_000, 16_000, 14_000), 1_000);
        assert_eq!(macos_available_memory(0, 16_000, 14_000), 0);
        assert_eq!(macos_available_memory(8_000, 16_000, 16_000), 0);
        assert_eq!(macos_available_memory(8_000, 16_000, 17_000), 0);
        assert_eq!(macos_available_memory(8_000, 0, 0), 0);
        assert_eq!(macos_available_memory(u64::MAX, u64::MAX, 1), u64::MAX - 1);
    }
}
