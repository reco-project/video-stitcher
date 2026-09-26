//! One coordinator and one pending replacement per file source.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use super::*;

type Run<T> =
    Box<dyn FnOnce(&SyncSender<Message<T>>, &Arc<AtomicBool>) -> Result<(), SourceError> + Send>;

struct Request<T> {
    run: Run<T>,
    tx: SyncSender<Message<T>>,
    stop: Arc<AtomicBool>,
}

struct Pending<T> {
    request: Option<Request<T>>,
    closed: bool,
}

/// Replacements cancel the current generation without waiting on decoder I/O.
/// The coordinator completes its cleanup before starting another generation.
pub(crate) struct Restartable<T> {
    rx: Receiver<Message<T>>,
    stop: Arc<AtomicBool>,
    pending: Arc<(Mutex<Pending<T>>, Condvar)>,
}

impl<T: Send + 'static> Restartable<T> {
    pub(crate) fn new(
        run: impl FnOnce(&SyncSender<Message<T>>, &Arc<AtomicBool>) -> Result<(), SourceError>
        + Send
        + 'static,
    ) -> Result<(Self, JoinHandle<()>), SourceError> {
        let (request, rx) = Self::request(run);
        let stop = request.stop.clone();
        let pending = Arc::new((
            Mutex::new(Pending {
                request: Some(request),
                closed: false,
            }),
            Condvar::new(),
        ));
        let work = pending.clone();
        let worker = std::thread::Builder::new()
            .name("decode_zip".into())
            .spawn(move || {
                loop {
                    let request = {
                        let (lock, wake) = &*work;
                        let mut pending = lock.lock().unwrap();
                        while pending.request.is_none() && !pending.closed {
                            pending = wake.wait(pending).unwrap();
                        }
                        if pending.closed {
                            break;
                        }
                        pending.request.take().unwrap()
                    };
                    if request.stop.load(Ordering::Relaxed) {
                        continue;
                    }
                    // A failed generation must not strand a pending receiver.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        (request.run)(&request.tx, &request.stop)
                    }))
                    .unwrap_or_else(|panic| {
                        let reason = panic
                            .downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| panic.downcast_ref::<&str>().copied())
                            .unwrap_or("non-string panic payload");
                        Err(SourceError::Read {
                            reason: format!("decode coordinator panicked: {reason}"),
                        })
                    });
                    let _ = request.tx.send(result.map(|()| None));
                }
            })
            .map_err(|error| SourceError::Read {
                reason: format!("spawn decode coordinator: {error}"),
            })?;
        Ok((Self { rx, stop, pending }, worker))
    }

    pub(crate) fn receive(&self) -> Message<T> {
        receive(&self.rx)
    }

    pub(crate) fn try_receive(&self) -> Result<Message<T>, std::sync::mpsc::TryRecvError> {
        self.rx.try_recv()
    }

    #[cfg(test)]
    pub(crate) fn from_receiver(rx: Receiver<Message<T>>) -> Self {
        Self {
            rx,
            stop: Arc::new(AtomicBool::new(false)),
            pending: Arc::new((
                Mutex::new(Pending {
                    request: None,
                    closed: false,
                }),
                Condvar::new(),
            )),
        }
    }

    fn request(
        run: impl FnOnce(&SyncSender<Message<T>>, &Arc<AtomicBool>) -> Result<(), SourceError>
        + Send
        + 'static,
    ) -> (Request<T>, Receiver<Message<T>>) {
        let (tx, rx) = sync_channel(4);
        (
            Request {
                run: Box::new(run),
                tx,
                stop: Arc::new(AtomicBool::new(false)),
            },
            rx,
        )
    }

    pub(crate) fn replace(
        &mut self,
        run: impl FnOnce(&SyncSender<Message<T>>, &Arc<AtomicBool>) -> Result<(), SourceError>
        + Send
        + 'static,
    ) {
        self.stop.store(true, Ordering::Relaxed);
        let (request, rx) = Self::request(run);
        self.rx = rx;
        self.stop = request.stop.clone();
        let (lock, wake) = &*self.pending;
        lock.lock().unwrap().request = Some(request);
        wake.notify_one();
    }
}

impl<T> Drop for Restartable<T> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let (lock, wake) = &*self.pending;
        let mut pending = lock.lock().unwrap();
        pending.closed = true;
        pending.request = None;
        wake.notify_one();
        // Dropping rx releases a blocked output send. The coordinator owns
        // child cleanup; joining it here would block the caller on native I/O.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;
    use std::time::Duration;

    #[test]
    fn replacements_coalesce_and_wait_for_cleanup_off_caller() {
        let obsolete_runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (started_tx, started_rx) = channel();
        let (release_tx, release_rx) = channel();
        let (cleaned_tx, cleaned_rx) = channel();
        let (mut stream, worker) = Restartable::new(move |_, stop| {
            started_tx.send(stop.clone()).unwrap();
            release_rx.recv().unwrap();
            cleaned_tx.send(()).unwrap();
            Ok(())
        })
        .unwrap();
        let stop = started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        for _ in 0..100 {
            let runs = obsolete_runs.clone();
            stream.replace(move |_, _| {
                runs.fetch_add(1, Ordering::Relaxed);
                Ok(())
            });
        }
        stream.replace(move |tx, _| {
            cleaned_rx
                .try_recv()
                .expect("previous generation cleaned up");
            tx.send(Ok(Some(42u8))).unwrap();
            Ok(())
        });
        assert!(stop.load(Ordering::Relaxed));
        assert!(matches!(
            stream.rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        release_tx.send(()).unwrap();
        assert_eq!(receive(&stream.rx).unwrap(), Some(42));
        assert_eq!(receive(&stream.rx).unwrap(), None);
        drop(stream);
        worker.join().unwrap();
        assert_eq!(obsolete_runs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn drop_does_not_wait_for_inflight_work_and_discards_pending() {
        let pending_runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (started_tx, started_rx) = channel();
        let (release_tx, release_rx) = channel();
        let (mut stream, worker) = Restartable::<u8>::new(move |_, stop| {
            started_tx.send(stop.clone()).unwrap();
            release_rx.recv().unwrap();
            Ok(())
        })
        .unwrap();
        let stop = started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let runs = pending_runs.clone();
        stream.replace(move |_, _| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });
        drop(stream);
        assert!(stop.load(Ordering::Relaxed));
        release_tx.send(()).unwrap();
        worker.join().unwrap();
        assert_eq!(pending_runs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn panic_and_error_are_terminal_but_allow_restart() {
        let (mut stream, worker) = Restartable::<u8>::new(|_, _| panic!("decoder panic")).unwrap();
        assert!(
            receive(&stream.rx)
                .unwrap_err()
                .to_string()
                .contains("decoder panic")
        );
        stream.replace(|_, _| Err(disconnected()));
        assert!(receive(&stream.rx).is_err());
        stream.replace(|_, _| Ok(()));
        assert!(receive(&stream.rx).unwrap().is_none());
        drop(stream);
        worker.join().unwrap();
    }

    #[test]
    fn replacement_releases_a_full_output_queue() {
        let (full_tx, full_rx) = channel();
        let (mut stream, worker) = Restartable::new(move |tx, _| {
            for _ in 0..4 {
                tx.send(Ok(Some(1u8))).unwrap();
            }
            full_tx.send(()).unwrap();
            let _ = tx.send(Ok(Some(2)));
            Ok(())
        })
        .unwrap();
        full_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        stream.replace(|tx, _| {
            tx.send(Ok(Some(3))).unwrap();
            Ok(())
        });
        assert_eq!(
            stream
                .rx
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap(),
            Some(3)
        );
        drop(stream);
        worker.join().unwrap();
    }
}
