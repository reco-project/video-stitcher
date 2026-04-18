//! OBS Studio source plugin for Reco panoramic video stitching.
//!
//! This crate builds as a shared library (`.so` / `.dll` / `.dylib`) that OBS
//! loads as a plugin. It registers a video source called "Reco Panorama
//! Stitcher" that uses [`reco_core::pipeline::StitchPipeline`] to stitch two
//! camera inputs into a single panoramic output.
//!
//! ## Installation
//!
//! Copy the built library to OBS's plugin directory:
//! - **Linux**: `~/.config/obs-studio/plugins/reco-obs/bin/64bit/libreco_obs.so`
//! - **Windows**: `%APPDATA%/obs-studio/plugins/reco-obs/bin/64bit/reco_obs.dll`
//! - **macOS**: `~/Library/Application Support/obs-studio/plugins/reco-obs.plugin/Contents/MacOS/libreco_obs.dylib`
//!
//! ## Limitations
//!
//! - Camera frame input is not yet wired up (renders test pattern).
//!   The next step is to pull frames from upstream OBS sources.
//! - RGBA readback from wgpu to CPU then re-upload to OBS texture incurs
//!   an extra GPU-CPU-GPU copy. Platform-specific zero-copy interop
//!   (DMA-BUF, shared handles) is a future optimization.

mod ffi;
mod source;

use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

/// OBS API version this plugin targets.
///
/// Format: `(major << 24) | (minor << 16) | patch`.
/// We target OBS 30.0.0 as a baseline; the struct layout is forward-compatible
/// as long as OBS checks `sizeof(obs_source_info)` during registration.
const LIBOBS_API_VER: u32 = 30 << 24;

/// Storage for the module pointer OBS passes us.
static MODULE_PTR: AtomicPtr<ffi::obs_module_t> = AtomicPtr::new(ptr::null_mut());

/// Called by OBS to give us our module handle.
///
/// # Safety
///
/// Called by OBS during module loading. `module` is valid for the module lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn obs_module_set_pointer(module: *mut ffi::obs_module_t) {
    MODULE_PTR.store(module, Ordering::Release);
}

/// Return our stored module handle.
///
/// # Safety
///
/// Called by OBS; returns the pointer from `obs_module_set_pointer`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn obs_current_module() -> *mut ffi::obs_module_t {
    MODULE_PTR.load(Ordering::Acquire)
}

/// Return the OBS API version this plugin was built against.
#[unsafe(no_mangle)]
pub extern "C" fn obs_module_ver() -> u32 {
    LIBOBS_API_VER
}

/// Install the standard tracing subscriber + log bridge for the OBS
/// plugin.
///
/// Post-deployment this is the user's one shot at diagnosing a bug.
/// OBS captures the plugin's stderr into its own log file, so every
/// event from our pipeline ends up in a place the user can zip up
/// and mail. Bridges `log::*` calls from reco-core / reco-io to the
/// tracing pipeline so there is one structured source of truth.
///
/// Called once from [`obs_module_load`]. Uses `try_init` so repeated
/// plugin loads (OBS devtools reload) do not panic.
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let _ = tracing_log::LogTracer::init();
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(true).with_level(true))
        .try_init();
}

/// Install a panic hook that emits the panic location + payload as a
/// `tracing::error!` event before the default hook runs.
///
/// Critical for the OBS plugin: if any of our `unsafe extern "C"`
/// callbacks panics, the default hook unwinds across the C ABI which
/// is undefined behavior (Rust Reference; downgraded from UB to
/// abort under `panic = "unwind"`, but still aborts the OBS host).
/// The `catch_unwind` wrappers at each FFI boundary (separate commit)
/// rely on this hook to produce a diagnosable record before the catch
/// converts the panic into a safe early-return.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".into());
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".into()
        };
        tracing::error!(
            target: "panic",
            location = %location,
            payload = %payload,
            "reco-obs: panic caught by tracing panic hook"
        );
        default_hook(info);
    }));
}

/// Called by OBS to load the module. Register our source here.
///
/// # Safety
///
/// Called by OBS during startup. We register our source info struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn obs_module_load() -> bool {
    // M2 migration: tracing_subscriber replaces env_logger. OBS captures
    // the plugin's stderr so structured tracing events land in OBS's
    // log files alongside other plugin output.
    init_tracing();
    install_panic_hook();

    log::info!("reco-obs: module loading (API version {LIBOBS_API_VER:#010x})");

    let info = source::source_info();
    unsafe {
        ffi::obs_register_source_s(&info, std::mem::size_of::<ffi::obs_source_info>());
    }

    log::info!("reco-obs: source registered as 'reco_stitcher'");
    true
}

/// Called by OBS when the module is unloaded.
#[unsafe(no_mangle)]
pub extern "C" fn obs_module_unload() {
    log::info!("reco-obs: module unloaded");
}

/// Return the module display name.
#[unsafe(no_mangle)]
pub extern "C" fn obs_module_name() -> *const std::os::raw::c_char {
    c"reco-obs".as_ptr()
}

/// Return the module description.
#[unsafe(no_mangle)]
pub extern "C" fn obs_module_description() -> *const std::os::raw::c_char {
    c"GPU-accelerated panoramic video stitcher powered by Reco".as_ptr()
}
