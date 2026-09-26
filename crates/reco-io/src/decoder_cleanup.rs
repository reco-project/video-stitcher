//! Scoped ownership of retired file decoders.

use std::sync::{Arc, Mutex, Weak};
use std::thread::JoinHandle;

/// Defers file-source cleanup until this scope is dropped.
///
/// Pass to [`crate::adapters::FfmpegFileSource::defer_cleanup`] to keep source
/// replacement and cancellation from waiting on native decoder teardown.
/// Drop sources before this scope: its destructor waits for their retired
/// workers. Sources that outlive the scope instead wait in their own destructor.
/// Native I/O has no shutdown deadline.
#[derive(Default)]
pub struct DecoderCleanup {
    retired: Arc<Retired>,
}

#[derive(Default)]
struct Retired(Mutex<Vec<JoinHandle<()>>>);

impl Retired {
    fn push(&self, worker: JoinHandle<()>) {
        let mut workers = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let mut i = 0;
        while i < workers.len() {
            if workers[i].is_finished() {
                join(workers.swap_remove(i));
            } else {
                i += 1;
            }
        }
        workers.push(worker);
    }
}

impl Drop for Retired {
    fn drop(&mut self) {
        for worker in self
            .0
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
        {
            join(worker);
        }
    }
}

/// Must be dropped after disconnecting the worker's input/output queues.
#[derive(Default)]
pub(crate) struct DecoderTask {
    worker: Option<JoinHandle<()>>,
    retired: Weak<Retired>,
}

impl DecoderTask {
    pub(crate) fn new(worker: JoinHandle<()>) -> Self {
        Self {
            worker: Some(worker),
            retired: Weak::new(),
        }
    }

    pub(crate) fn defer_to(&mut self, cleanup: &DecoderCleanup) {
        self.retired = Arc::downgrade(&cleanup.retired);
    }
}

impl Drop for DecoderTask {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            if let Some(retired) = self.retired.upgrade() {
                retired.push(worker);
            } else {
                join(worker);
            }
        }
    }
}

fn join(worker: JoinHandle<()>) {
    if worker.join().is_err() {
        log::error!("decode coordinator panicked during cleanup");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_stream::Restartable;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{TryRecvError, channel};
    use std::time::{Duration, Instant};

    struct Source {
        stream: Restartable<u8>,
        task: DecoderTask,
    }

    fn source(
        run: impl FnOnce(
            &std::sync::mpsc::SyncSender<crate::decode_stream::Message<u8>>,
            &Arc<AtomicBool>,
        ) -> Result<(), reco_core::source::SourceError>
        + Send
        + 'static,
    ) -> Source {
        let (stream, worker) = Restartable::new(run).unwrap();
        Source {
            stream,
            task: DecoderTask::new(worker),
        }
    }

    #[test]
    fn default_drop_disconnects_backpressure_and_joins() {
        let cleaned = Arc::new(AtomicBool::new(false));
        let done = cleaned.clone();
        let source = source(move |tx, _| {
            while tx.send(Ok(Some(1))).is_ok() {}
            done.store(true, Ordering::Relaxed);
            Ok(())
        });
        assert_eq!(source.stream.receive().unwrap(), Some(1));
        drop(source);
        assert!(cleaned.load(Ordering::Relaxed));
    }

    #[test]
    fn retired_sources_do_not_block_but_scope_waits_for_all() {
        let cleanup = DecoderCleanup::default();
        let mut releases = Vec::new();
        let cleaned = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for _ in 0..2 {
            let (release, wait) = channel();
            let (started, ready) = channel();
            let done = cleaned.clone();
            let mut source = source(move |_, stop| {
                started.send(stop.clone()).unwrap();
                wait.recv().unwrap();
                done.fetch_add(1, Ordering::Relaxed);
                Ok(())
            });
            source.task.defer_to(&cleanup);
            let stop = ready.recv_timeout(Duration::from_secs(5)).unwrap();
            drop(source);
            assert!(stop.load(Ordering::Relaxed));
            releases.push(release);
        }
        assert_eq!(cleanup.retired.0.lock().unwrap().len(), 2);
        let (done, completed) = channel();
        let shutdown = std::thread::spawn(move || {
            drop(cleanup);
            done.send(()).unwrap();
        });
        assert!(matches!(completed.try_recv(), Err(TryRecvError::Empty)));
        releases.pop().unwrap().send(()).unwrap();
        assert!(matches!(completed.try_recv(), Err(TryRecvError::Empty)));
        releases.pop().unwrap().send(()).unwrap();
        completed.recv_timeout(Duration::from_secs(5)).unwrap();
        shutdown.join().unwrap();
        assert_eq!(cleaned.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn source_outliving_scope_falls_back_to_join() {
        let cleanup = DecoderCleanup::default();
        let cleaned = Arc::new(AtomicBool::new(false));
        let done = cleaned.clone();
        let mut source = source(move |tx, _| {
            while tx.send(Ok(Some(1))).is_ok() {}
            done.store(true, Ordering::Relaxed);
            Ok(())
        });
        source.task.defer_to(&cleanup);
        assert_eq!(source.stream.receive().unwrap(), Some(1));
        drop(cleanup);
        drop(source);
        assert!(cleaned.load(Ordering::Relaxed));
    }

    #[test]
    fn completed_handles_are_pruned_on_retirement() {
        let cleanup = DecoderCleanup::default();
        for _ in 0..100 {
            let worker = std::thread::spawn(|| {});
            let deadline = Instant::now() + Duration::from_secs(5);
            while !worker.is_finished() {
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            cleanup.retired.push(worker);
            assert_eq!(cleanup.retired.0.lock().unwrap().len(), 1);
        }
    }

    #[test]
    fn unwinding_cancels_sources_before_waiting_for_cleanup() {
        let cleaned = Arc::new(AtomicBool::new(false));
        let done = cleaned.clone();
        let result = std::panic::catch_unwind(|| {
            let cleanup = DecoderCleanup::default();
            let mut source = source(move |tx, _| {
                while tx.send(Ok(Some(1))).is_ok() {}
                done.store(true, Ordering::Relaxed);
                Ok(())
            });
            source.task.defer_to(&cleanup);
            assert_eq!(source.stream.receive().unwrap(), Some(1));
            panic!("consumer failed");
        });
        assert!(result.is_err());
        assert!(cleaned.load(Ordering::Relaxed));
    }
}
