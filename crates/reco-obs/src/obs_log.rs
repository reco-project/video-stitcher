//! OBS log bridge.
//!
//! Plugin diagnostic messages (the `log::warn!` / `log::info!` calls
//! scattered through `source.rs`) used to land only in the process
//! stderr, which OBS does not capture into its log file. Users who
//! opened `Help → Log Files → View Current Log` to diagnose a problem
//! saw OBS's own output but nothing from reco-obs, making it hard to
//! tell whether the plugin was even loaded.
//!
//! This module installs a `tracing` layer that calls OBS's `blog(...)`
//! via the `blog_shim.c` C stub so each tracing event also surfaces
//! in OBS's log pane. Preserves the stderr output too (both layers
//! run) so out-of-OBS tooling (`RUST_LOG=trace`, tests) still sees
//! everything.

use std::ffi::{CString, c_char, c_int};

use tracing::{Level, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

unsafe extern "C" {
    fn reco_obs_blog(level: c_int, message: *const c_char);
}

/// OBS log-level constants mirroring `util/base.h`. Kept as raw ints
/// so we don't need to pull the bindgen-generated enum name across
/// module boundaries (and its numeric values are stable OBS API).
const LOG_ERROR: c_int = 100;
const LOG_WARNING: c_int = 200;
const LOG_INFO: c_int = 300;
const LOG_DEBUG: c_int = 400;

fn tracing_level_to_obs(level: &Level) -> c_int {
    match *level {
        Level::ERROR => LOG_ERROR,
        Level::WARN => LOG_WARNING,
        Level::INFO => LOG_INFO,
        Level::DEBUG | Level::TRACE => LOG_DEBUG,
    }
}

/// tracing-subscriber Layer that mirrors each event to OBS's log.
///
/// The `message` field on a tracing event carries the formatted
/// message; we extract it via a minimal Visit impl. Spans are
/// ignored — OBS's log is a flat stream so nested span context would
/// be noisy. Users needing span-aware output can still consult
/// stderr / the captured tracing file.
pub struct ObsLogLayer;

impl<S> Layer<S> for ObsLogLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = MessageVisitor {
            message: String::new(),
        };
        event.record(&mut visitor);
        if visitor.message.is_empty() {
            return;
        }
        let obs_level = tracing_level_to_obs(event.metadata().level());
        let target = event.metadata().target();
        // Prefix with the tracing target so OBS log readers can
        // filter our lines ("[reco-obs]") from OBS-core output.
        // `CString::new` rejects interior NULs; on the rare chance a
        // user message contains one we drop the event rather than
        // pass a malformed string to C.
        let formatted = format!("[{target}] {}", visitor.message);
        if let Ok(cstr) = CString::new(formatted) {
            // SAFETY: `reco_obs_blog` only reads the NUL-terminated
            // buffer; no pointer escapes. `cstr` owns the storage
            // for the duration of the call.
            unsafe {
                reco_obs_blog(obs_level, cstr.as_ptr());
            }
        }
    }
}

/// Visitor that captures the `message` field of a tracing event.
struct MessageVisitor {
    message: String,
}

impl tracing::field::Visit for MessageVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        // The default tracing fmt subscriber hits `record_debug` for
        // `log::*` macro calls (they go through the tracing-log
        // bridge as `message = <debug>`). Keep the Debug formatting
        // so we preserve the user-facing message text.
        if field.name() == "message" {
            use std::fmt::Write;
            let _ = write!(&mut self.message, "{:?}", value);
        }
    }
}
