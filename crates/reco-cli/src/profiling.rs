//! Own the process-wide profiler for the lifetime of a CLI command.

use anyhow::Context;
use tracing_chrome::{ChromeLayer, FlushGuard};
use tracing_subscriber::{Registry, prelude::*, reload};

/// Detaches the trace layer before its writer is flushed and joined.
#[must_use]
pub(super) struct Guard {
    layer: reload::Handle<Option<ChromeLayer<Registry>>, Registry>,
    _flush: FlushGuard,
}

impl Drop for Guard {
    fn drop(&mut self) {
        // Native destructors may log after Rust thread-local storage is gone.
        // They must not enter the profiler after the command has finished.
        let _ = self.layer.reload(None);
    }
}

/// Start tracing after argument parsing, retaining worker-thread events.
pub(super) fn init() -> anyhow::Result<Guard> {
    let writer = std::fs::File::create("reco-trace.json")
        .context("create profiling trace reco-trace.json")?;
    let (layer, flush) = tracing_chrome::ChromeLayerBuilder::new()
        .writer(writer)
        .include_args(true)
        .build();
    let (layer, handle) = reload::Layer::new(Some(layer));
    tracing_subscriber::registry().with(layer).try_init()?;
    eprintln!("Profiling enabled - trace will be written to reco-trace.json");
    Ok(Guard {
        layer: handle,
        _flush: flush,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    static CHECK_NATIVE_EXIT: AtomicBool = AtomicBool::new(false);

    struct LateLog;

    impl Drop for LateLog {
        fn drop(&mut self) {
            tracing::trace!("thread-local destructor");
        }
    }

    thread_local! {
        static LATE_LOG: LateLog = const { LateLog };
    }

    #[cfg(target_os = "linux")]
    #[used]
    #[unsafe(link_section = ".fini_array")]
    static NATIVE_EXIT: extern "C" fn() = late_native_log;

    #[cfg(target_os = "linux")]
    extern "C" fn late_native_log() {
        if CHECK_NATIVE_EXIT.load(Ordering::Relaxed) {
            tracing::trace!("native destructor");
            tracing::trace_span!("native destructor span").in_scope(|| {});
        }
    }

    #[test]
    fn command_lifetime_process() {
        let Ok(directory) = std::env::var("RECO_PROFILING_TEST_DIRECTORY") else {
            return;
        };
        // This test runs alone in its own process, including native destructors.
        std::env::set_current_dir(directory).unwrap();
        // Initialize before the profiler so this destructor runs after its TLS.
        LATE_LOG.with(|_| {});
        let guard = super::init().unwrap();
        tracing::info!("main event");
        std::thread::spawn(|| {
            tracing::info_span!("worker span").in_scope(|| tracing::info!("worker event"));
        })
        .join()
        .unwrap();
        let late_span = tracing::info_span!("retained span");
        if std::env::var_os("RECO_PROFILING_TEST_UNWIND").is_some() {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _guard = guard;
                    panic!("command failed");
                }))
                .is_err()
            );
        } else {
            drop(guard);
        }
        late_span.in_scope(|| tracing::info!("after shutdown"));
        drop(late_span);
        std::thread::spawn(|| tracing::info!("late worker"))
            .join()
            .unwrap();
        CHECK_NATIVE_EXIT.store(true, Ordering::Relaxed);
    }

    #[test]
    fn finishes_trace_before_late_logs_on_return_and_unwind() {
        for unwind in [false, true] {
            let directory = std::env::temp_dir()
                .join(format!("reco-profiling-{}-{unwind}", std::process::id()));
            std::fs::create_dir(&directory).unwrap();
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            child.args([
                "--exact",
                "profiling::tests::command_lifetime_process",
                "--nocapture",
            ]);
            child.env("RECO_PROFILING_TEST_DIRECTORY", &directory);
            child.env_remove("RECO_PROFILING_TEST_UNWIND");
            if unwind {
                child.env("RECO_PROFILING_TEST_UNWIND", "1");
            }
            let output = child.output().unwrap();
            assert!(output.status.success(), "{output:?}");
            let trace = std::fs::read_to_string(directory.join("reco-trace.json")).unwrap();
            let _: serde_json::Value = serde_json::from_str(&trace).unwrap();
            for event in ["main event", "worker event", "worker span"] {
                assert!(trace.contains(event), "missing {event}");
            }
            for event in [
                "after shutdown",
                "late worker",
                "native destructor",
                "thread-local destructor",
            ] {
                assert!(!trace.contains(event), "unexpected {event}");
            }
            std::fs::remove_file(directory.join("reco-trace.json")).unwrap();
            std::fs::remove_dir(directory).unwrap();
        }
    }
}
