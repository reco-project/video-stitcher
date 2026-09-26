//! Bounded decode streams with explicit completion and failure.

use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

use reco_core::source::SourceError;

mod restart;
pub(crate) use restart::Restartable;

pub(crate) type Message<T> = Result<Option<T>, SourceError>;

pub(crate) fn disconnected() -> SourceError {
    SourceError::Read {
        reason: "decode worker disconnected without a terminal result".into(),
    }
}

pub(crate) fn receive<T>(rx: &Receiver<Message<T>>) -> Message<T> {
    rx.recv().map_err(|_| disconnected())?
}

#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos", test))]
fn receive_pair<T>(left: &Receiver<Message<T>>, right: &Receiver<Message<T>>) -> Message<(T, T)> {
    let left = receive(left)?;
    let right = receive(right)?;
    Ok(left.zip(right))
}

#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos", test))]
pub(crate) fn pair<T: Send + 'static, U: Send + 'static>(
    left: Receiver<Message<T>>,
    right: Receiver<Message<T>>,
    sync_offset: i64,
    capacity: usize,
    map: impl Fn((T, T)) -> U + Send + 'static,
) -> (Receiver<Message<U>>, JoinHandle<()>) {
    spawn(
        "decode_pair".into(),
        capacity,
        move |tx| -> Result<(), SourceError> {
            let skip_rx = if sync_offset > 0 { &right } else { &left };
            if !skip(skip_rx, sync_offset.unsigned_abs())? {
                return Ok(());
            }
            while let Some(frames) = receive_pair(&left, &right)? {
                if tx.send(Ok(Some(map(frames)))).is_err() {
                    break;
                }
            }
            Ok(())
        },
    )
}

pub(crate) fn skip<T>(rx: &Receiver<Message<T>>, count: u64) -> Result<bool, SourceError> {
    for _ in 0..count {
        if receive(rx)?.is_none() {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn spawn<T, E>(
    name: String,
    capacity: usize,
    run: impl FnOnce(&SyncSender<Message<T>>) -> Result<(), E> + Send + 'static,
) -> (Receiver<Message<T>>, JoinHandle<()>)
where
    T: Send + 'static,
    E: std::fmt::Display,
{
    let (tx, rx) = sync_channel(capacity);
    let handle = std::thread::Builder::new()
        .name(name.clone())
        .spawn(move || {
            let terminal = run(&tx).map(|()| None).map_err(|error| SourceError::Read {
                reason: format!("{name}: {error}"),
            });
            // Receiver drop cancels a stream, including a blocked frame send.
            let _ = tx.send(terminal);
        })
        .expect("spawn decode worker");
    (rx, handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scripted(messages: Vec<Message<u8>>) -> Receiver<Message<u8>> {
        let (tx, rx) = sync_channel(messages.len());
        for message in messages {
            tx.send(message).unwrap();
        }
        rx
    }

    fn failure() -> Message<u8> {
        Err(SourceError::Read {
            reason: "broken input".into(),
        })
    }

    #[test]
    fn pairs_all_frames_then_explicit_eof() {
        let (rx, worker) = pair(
            scripted(vec![Ok(Some(1)), Ok(Some(2)), Ok(None)]),
            scripted(vec![Ok(Some(3)), Ok(Some(4)), Ok(None)]),
            0,
            1,
            |frames| frames,
        );
        assert_eq!(receive(&rx).unwrap(), Some((1, 3)));
        assert_eq!(receive(&rx).unwrap(), Some((2, 4)));
        assert_eq!(receive(&rx).unwrap(), None);
        worker.join().unwrap();
    }

    #[test]
    fn shorter_input_is_clean_eof_in_either_order() {
        for left_shorter in [true, false] {
            let short = scripted(vec![Ok(None)]);
            let long = scripted(vec![Ok(Some(1)), Ok(Some(2)), Ok(None)]);
            let (left, right) = if left_shorter {
                (short, long)
            } else {
                (long, short)
            };
            let (rx, worker) = pair(left, right, 0, 1, |frames| frames);
            assert_eq!(receive(&rx).unwrap(), None);
            worker.join().unwrap();
        }
    }

    #[test]
    fn failure_after_frames_is_not_eof_in_either_order() {
        for left_fails in [true, false] {
            let broken = scripted(vec![Ok(Some(1)), failure()]);
            let short = scripted(vec![Ok(Some(2)), Ok(None)]);
            let (left, right) = if left_fails {
                (broken, short)
            } else {
                (short, broken)
            };
            let (rx, worker) = pair(left, right, 0, 1, |frames| frames);
            assert!(receive(&rx).unwrap().is_some());
            assert!(
                receive(&rx)
                    .unwrap_err()
                    .to_string()
                    .contains("broken input")
            );
            worker.join().unwrap();
        }
    }

    #[test]
    fn offset_skips_preserve_failure_and_eof() {
        for offset in [-2, 2] {
            for terminal in [Ok(None), failure()] {
                let expect_error = terminal.is_err();
                let skipped = scripted(vec![Ok(Some(1)), terminal]);
                let other = scripted(vec![Ok(Some(2))]);
                let (left, right) = if offset < 0 {
                    (skipped, other)
                } else {
                    (other, skipped)
                };
                let (rx, worker) = pair(left, right, offset, 1, |frames| frames);
                if expect_error {
                    assert!(receive(&rx).is_err());
                } else {
                    assert_eq!(receive(&rx).unwrap(), None);
                }
                worker.join().unwrap();
            }
        }
    }

    #[test]
    fn unexpected_disconnect_is_not_eof() {
        assert!(
            receive(&scripted(vec![]))
                .unwrap_err()
                .to_string()
                .contains("disconnected")
        );
    }

    #[test]
    fn worker_reports_failure_after_partial_output() {
        let (rx, worker) = spawn("test".into(), 1, |tx| -> Result<(), &str> {
            tx.send(Ok(Some(7))).unwrap();
            Err("seek failed")
        });
        assert_eq!(receive(&rx).unwrap(), Some(7));
        assert!(
            receive(&rx)
                .unwrap_err()
                .to_string()
                .contains("seek failed")
        );
        worker.join().unwrap();
    }

    #[test]
    fn dropping_receiver_unblocks_full_pipeline() {
        let produce = || {
            spawn("test".into(), 1, |tx| -> Result<(), &str> {
                while tx.send(Ok(Some(7))).is_ok() {}
                Ok(())
            })
        };
        let (left, left_worker) = produce();
        let (right, right_worker) = produce();
        let (rx, pair_worker) = pair(left, right, 0, 1, |frames| frames);
        assert!(receive(&rx).unwrap().is_some());
        drop(rx);
        let (done_tx, done_rx) = sync_channel(1);
        std::thread::spawn(move || {
            pair_worker.join().unwrap();
            left_worker.join().unwrap();
            right_worker.join().unwrap();
            done_tx.send(()).unwrap();
        });
        done_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
    }
}
