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

/// Called by OBS to load the module. Register our source here.
///
/// # Safety
///
/// Called by OBS during startup. We register our source info struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn obs_module_load() -> bool {
    // Initialize logging with an Info default so plugin messages
    // surface in OBS's captured stderr (and via OBS's log files too)
    // without the user needing to set RUST_LOG manually. If the env
    // already sets RUST_LOG, env_logger's parser takes over.
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();

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
