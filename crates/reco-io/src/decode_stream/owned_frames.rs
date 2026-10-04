//! Cancellation and ordered cleanup for camera-ordered owned frames.

use super::*;
use crate::decoder_cleanup::DecoderTask;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct OwnedFrames<T> {
    // Disconnect output before joining the collector, which releases its inputs.
    rx: Receiver<Message<Vec<T>>>,
    _workers: Vec<DecoderTask>,
    stop: Arc<AtomicBool>,
}

impl<T: Send + 'static> OwnedFrames<T> {
    pub(crate) fn new(
        inputs: Vec<(Receiver<Message<T>>, DecoderTask)>,
        offset: i64,
        stop: Arc<AtomicBool>,
    ) -> Self {
        let (receivers, mut workers): (Vec<_>, Vec<_>) = inputs.into_iter().unzip();
        let (rx, collector) = collect(receivers, offset, 4);
        workers.insert(0, DecoderTask::new(collector));
        Self {
            rx,
            _workers: workers,
            stop,
        }
    }

    pub(crate) fn receive(&self) -> Message<Vec<T>> {
        receive(&self.rx)
    }
}

impl<T> Drop for OwnedFrames<T> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;
    use std::time::Duration;

    #[test]
    fn mono_delivery_retains_the_frame_until_the_consumer_releases_it() {
        let frame = Arc::new(7u8);
        let weak = Arc::downgrade(&frame);
        let (rx, worker) = spawn("owned_mono_test".into(), 4, move |tx| {
            let _ = tx.send(Ok(Some(frame)));
            Ok::<_, SourceError>(())
        });
        let stream = OwnedFrames::new(
            vec![(rx, DecoderTask::new(worker))],
            0,
            Arc::new(AtomicBool::new(false)),
        );
        let frames = stream.receive().unwrap().unwrap();
        assert_eq!(frames.len(), 1);
        assert!(stream.receive().unwrap().is_none());
        drop(stream);
        assert!(weak.upgrade().is_some());
        drop(frames);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn drop_releases_backpressured_queues_before_joining_every_worker() {
        let stop = Arc::new(AtomicBool::new(false));
        let (full_tx, full_rx) = channel();
        let (done_tx, done_rx) = channel();
        let worker = || {
            let stop = stop.clone();
            let full = full_tx.clone();
            let done = done_tx.clone();
            let (rx, handle) = spawn("owned_pair_test".into(), 4, move |tx| {
                let mut count = 0;
                while !stop.load(Ordering::Relaxed) {
                    if tx.send(Ok(Some(1u8))).is_err() {
                        break;
                    }
                    count += 1;
                    if count == 9 {
                        full.send(()).unwrap();
                    }
                }
                done.send(()).unwrap();
                Ok::<_, SourceError>(())
            });
            (rx, DecoderTask::new(handle))
        };
        let stream = OwnedFrames::new(vec![worker(), worker()], 0, stop.clone());
        for _ in 0..2 {
            full_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        let (closed_tx, closed_rx) = channel();
        let closer = std::thread::spawn(move || {
            drop(stream);
            closed_tx.send(()).unwrap();
        });
        closed_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        closer.join().unwrap();
        assert!(stop.load(Ordering::Relaxed));
        for _ in 0..2 {
            done_rx.try_recv().unwrap();
        }
    }

    #[test]
    fn cancellation_interrupts_extreme_sync_offsets_in_both_directions() {
        for offset in [i64::MIN, i64::MAX] {
            let stop = Arc::new(AtomicBool::new(false));
            let (started_tx, started_rx) = channel();
            let worker = || {
                let stop = stop.clone();
                let started = started_tx.clone();
                let (rx, handle) = spawn("sync_cancel_test".into(), 4, move |tx| {
                    started.send(()).unwrap();
                    while !stop.load(Ordering::Relaxed) {
                        if tx.send(Ok(Some(1u8))).is_err() {
                            break;
                        }
                    }
                    Ok::<_, SourceError>(())
                });
                (rx, DecoderTask::new(handle))
            };
            let stream = OwnedFrames::new(vec![worker(), worker()], offset, stop.clone());
            for _ in 0..2 {
                started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            }
            let (done_tx, done_rx) = channel();
            let closer = std::thread::spawn(move || {
                drop(stream);
                done_tx.send(()).unwrap();
            });
            done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            closer.join().unwrap();
        }
    }

    #[test]
    fn owned_frames_preserve_frames_eof_and_errors() {
        for fail in [false, true] {
            let worker = || {
                let (rx, handle) = spawn("terminal_test".into(), 4, move |tx| {
                    tx.send(Ok(Some(7u8))).unwrap();
                    if fail { Err(disconnected()) } else { Ok(()) }
                });
                (rx, DecoderTask::new(handle))
            };
            let stream = OwnedFrames::new(
                vec![worker(), worker()],
                0,
                Arc::new(AtomicBool::new(false)),
            );
            assert_eq!(stream.receive().unwrap(), Some(vec![7, 7]));
            let terminal = stream.receive();
            if fail {
                assert!(terminal.is_err());
            } else {
                assert!(terminal.unwrap().is_none());
            }
        }
    }
}
