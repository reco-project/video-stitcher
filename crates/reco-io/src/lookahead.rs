//! Admission policy for lookahead image storage, sampled once per job.

#[cfg(target_os = "linux")]
mod cgroup;

/// Choose an image allowance, preserving an explicit caller override.
///
/// Host storage (including GPU upload paths) uses half the currently available
/// RAM, limited by the process's visible Linux cgroup ancestors. Unresolved
/// Linux hierarchies require an explicit allowance. This leaves room
/// for decoders, output and inference but is not a process-memory guarantee.
/// Unknown availability is an error, never an unlimited fallback. Explicit
/// allowances do not bypass the session's native driver admission check.
pub fn image_budget(explicit_bytes: Option<usize>) -> Result<usize, String> {
    if let Some(bytes) = explicit_bytes {
        return nonzero(bytes);
    }
    let mut system = sysinfo::System::new();
    system.refresh_memory_specifics(sysinfo::MemoryRefreshKind::nothing().with_ram());
    let available = system.available_memory();
    #[cfg(target_os = "linux")]
    let available = cgroup::available_memory(available).map_err(|error| {
        format!("cannot resolve lookahead cgroup memory allowance: {error}; provide an explicit byte budget")
    })?;
    let bytes = usize::try_from(available / 2).unwrap_or(usize::MAX);
    nonzero(bytes)
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
}
