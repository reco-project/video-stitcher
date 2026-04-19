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
mod obs_log;
mod source;

use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

/// Global state shared between the frontend-event callback and the
/// source instance. `true` whenever OBS is actively recording or
/// streaming. Consulted by the replay recorder when the source is
/// configured to "follow OBS record/stream" mode (FRICTION A20).
///
/// Using atomics (no mutex) because writers come from a single OBS
/// thread (the frontend event loop) and readers are source ticks;
/// relaxed ordering is fine since the worst-case miss is one
/// recorder tick of stale state.
#[cfg(all(feature = "replay", have_frontend_api))]
pub(crate) static OBS_RECORDING_OR_STREAMING: AtomicBool = AtomicBool::new(false);

/// Stub constant for builds without the frontend-api. In that case
/// the replay recorder always falls through to its "independent"
/// mode regardless of what the source thinks.
#[cfg(all(feature = "replay", not(have_frontend_api)))]
pub(crate) static OBS_RECORDING_OR_STREAMING: AtomicBool = AtomicBool::new(true);

/// Wrap the body of an `unsafe extern "C"` callback with
/// `std::panic::catch_unwind` so a panic inside the callback cannot
/// unwind across the C ABI boundary into the OBS host.
///
/// Pairs with the crate-private `install_panic_hook` helper: on a
/// panic, the hook emits a structured `tracing::error!` event
/// (location + payload), then this macro's `catch_unwind` converts
/// the would-be abort into a safe early-return with the caller-
/// supplied default value. Together they complete the T-1 (severity-
/// adjusted High-DoS) mitigation from the 2026-04-18 deep review.
///
/// Usage:
/// ```ignore
/// unsafe extern "C" fn source_get_width(data: *mut c_void) -> u32 {
///     ffi_catch!(0u32, {
///         // original body that may panic
///     })
/// }
/// ```
///
/// The macro uses `AssertUnwindSafe` because the callback's captured
/// state is typically a `&mut SourceInstance` whose invariants may
/// already be broken by the panic. Consumers inside the body must not
/// rely on re-entrant state being consistent after a caught panic;
/// the safe default value is the only contract.
#[macro_export]
macro_rules! ffi_catch {
    ($default:expr, $body:block) => {{
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(v) => v,
            Err(_) => {
                // Panic already logged via the panic hook installed in
                // `obs_module_load`. Return a safe default so OBS can
                // continue rather than aborting the host.
                $default
            }
        }
    }};
}

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
    ffi_catch!((), {
        MODULE_PTR.store(module, Ordering::Release);
    })
}

/// Return our stored module handle.
///
/// # Safety
///
/// Called by OBS; returns the pointer from `obs_module_set_pointer`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn obs_current_module() -> *mut ffi::obs_module_t {
    ffi_catch!(ptr::null_mut(), { MODULE_PTR.load(Ordering::Acquire) })
}

/// Return the OBS API version this plugin was built against.
#[unsafe(no_mangle)]
pub extern "C" fn obs_module_ver() -> u32 {
    ffi_catch!(0u32, { LIBOBS_API_VER })
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
        // Stderr layer: captured by RUST_LOG for external tooling
        // and developer runs. Still useful alongside the OBS layer
        // because OBS doesn't duplicate stderr into its log file.
        .with(fmt::layer().with_target(true).with_level(true))
        // OBS log layer: surfaces each event in OBS's own log pane
        // (Help → Log Files → View Current Log). This is the pane
        // end users actually open when something is wrong; without
        // it our diagnostics were invisible to them.
        .with(obs_log::ObsLogLayer)
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
    ffi_catch!(false, {
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
    })
}

/// Called by OBS after all modules have finished loading. This is
/// the documented hook point for registering frontend event
/// callbacks (the frontend-api is only safe to call once every
/// module's `obs_module_load` has returned).
///
/// Only compiled when bindgen found `obs-frontend-api.h`; builds
/// against bare libobs skip the auto-follow and leave
/// `OBS_RECORDING_OR_STREAMING` pinned `true` so the replay
/// recorder always runs when enabled (fallback behavior preserves
/// the pre-A20 UX).
#[cfg(all(feature = "replay", have_frontend_api))]
#[unsafe(no_mangle)]
pub extern "C" fn obs_module_post_load() {
    ffi_catch!((), {
        unsafe {
            ffi::obs_frontend_add_event_callback(Some(on_frontend_event), ptr::null_mut());
        }
        log::info!(
            "reco-obs: frontend event callback registered \
             (replay will auto-follow OBS Record/Stream)"
        );
    })
}

#[cfg(all(feature = "replay", have_frontend_api))]
unsafe extern "C" fn on_frontend_event(
    event: ffi::obs_frontend_event,
    _private_data: *mut std::os::raw::c_void,
) {
    ffi_catch!((), {
        // Only four events of interest: the two start/stop pairs
        // for Record and Stream. Everything else ignored.
        match event {
            ffi::obs_frontend_event_OBS_FRONTEND_EVENT_RECORDING_STARTED
            | ffi::obs_frontend_event_OBS_FRONTEND_EVENT_STREAMING_STARTED => {
                OBS_RECORDING_OR_STREAMING.store(true, Ordering::Relaxed);
                log::info!("reco-obs: OBS Record/Stream STARTED — replay follow active");
            }
            ffi::obs_frontend_event_OBS_FRONTEND_EVENT_RECORDING_STOPPED
            | ffi::obs_frontend_event_OBS_FRONTEND_EVENT_STREAMING_STOPPED => {
                // Only flip off when BOTH are idle; streaming could
                // be stopping while recording continues, and the
                // replay should keep going in that case.
                let still_recording = unsafe { ffi::obs_frontend_recording_active() };
                let still_streaming = unsafe { ffi::obs_frontend_streaming_active() };
                if !still_recording && !still_streaming {
                    OBS_RECORDING_OR_STREAMING.store(false, Ordering::Relaxed);
                    log::info!("reco-obs: OBS Record/Stream STOPPED — replay follow idle");
                }
            }
            _ => {}
        }
    })
}

/// Called by OBS when the module is unloaded.
#[unsafe(no_mangle)]
pub extern "C" fn obs_module_unload() {
    ffi_catch!((), {
        log::info!("reco-obs: module unloaded");
    })
}

/// Return the module display name.
#[unsafe(no_mangle)]
pub extern "C" fn obs_module_name() -> *const std::os::raw::c_char {
    ffi_catch!(ptr::null(), { c"reco-obs".as_ptr() })
}

/// Return the module description.
#[unsafe(no_mangle)]
pub extern "C" fn obs_module_description() -> *const std::os::raw::c_char {
    ffi_catch!(ptr::null(), {
        c"GPU-accelerated panoramic video stitcher powered by Reco".as_ptr()
    })
}

#[cfg(test)]
mod macro_tests {
    //! Regression tests for the `ffi_catch!` macro. Completes the T-1
    //! (deep-review-2026-04-18) mitigation: panic hook + catch_unwind
    //! wrappers together prevent a panic in any of the 22 `extern "C"`
    //! callbacks from propagating across the C ABI.

    #[test]
    fn ffi_catch_returns_value_on_success() {
        let out: u32 = crate::ffi_catch!(0u32, { 7 + 35 });
        assert_eq!(out, 42);
    }

    #[test]
    fn ffi_catch_returns_default_on_panic() {
        let out: u32 = crate::ffi_catch!(42u32, {
            panic!("synthetic panic for test");
        });
        assert_eq!(out, 42);
    }

    #[test]
    fn ffi_catch_handles_string_panic_payload() {
        let out: i64 = crate::ffi_catch!(-1i64, {
            panic!("{}", String::from("string payload"));
        });
        assert_eq!(out, -1);
    }

    #[test]
    fn ffi_catch_handles_void_callback() {
        let mut counter = 0;
        crate::ffi_catch!((), {
            counter += 1;
            if counter > 100 {
                panic!("unreachable");
            }
        });
        assert_eq!(counter, 1);
    }

    #[test]
    fn ffi_catch_returns_default_pointer_on_panic() {
        let out: *mut u8 = crate::ffi_catch!(std::ptr::null_mut(), {
            panic!("ptr callback panic");
        });
        assert!(out.is_null());
    }
}
