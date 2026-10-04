//! Bounded decode streams with explicit completion and failure.

use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

use reco_core::source::SourceError;

mod restart;
pub(crate) use restart::Restartable;

#[cfg(any(
    all(feature = "gpu", any(target_os = "windows", target_os = "macos")),
    test
))]
mod owned_frames;
#[cfg(all(feature = "gpu", any(target_os = "windows", target_os = "macos")))]
pub(crate) use owned_frames::OwnedFrames;

pub(crate) type Message<T> = Result<Option<T>, SourceError>;

pub(crate) fn disconnected() -> SourceError {
    SourceError::Read {
        reason: "decode worker disconnected without a terminal result".into(),
    }
}

pub(crate) fn receive<T>(rx: &Receiver<Message<T>>) -> Message<T> {
    rx.recv().map_err(|_| disconnected())?
}

/// Collect one frame from every input, stopping at the shortest input.
/// Read all terminal messages before accepting EOF, so another input's error is not hidden.
/// For stereo, positive offsets skip right frames and negative offsets skip left frames.
#[cfg(any(
    test,
    all(
        feature = "gpu",
        any(target_os = "linux", target_os = "windows", target_os = "macos")
    )
))]
pub(crate) fn collect<T: Send + 'static>(
    inputs: Vec<Receiver<Message<T>>>,
    sync_offset: i64,
    capacity: usize,
) -> (Receiver<Message<Vec<T>>>, JoinHandle<()>) {
    spawn(
        "decode_collect".into(),
        capacity,
        move |tx| -> Result<(), SourceError> {
            if inputs.is_empty() {
                return Ok(());
            }
            if inputs.len() == 2 {
                let shifted = usize::from(sync_offset > 0);
                if !skip(&inputs[shifted], sync_offset.unsigned_abs())? {
                    // EOF during alignment must not hide the other input's failure.
                    receive(&inputs[1 - shifted])?;
                    return Ok(());
                }
            }
            loop {
                let mut frames = Vec::with_capacity(inputs.len());
                for input in &inputs {
                    if let Some(frame) = receive(input)? {
                        frames.push(frame);
                    }
                }
                if frames.len() != inputs.len() || tx.send(Ok(Some(frames))).is_err() {
                    return Ok(());
                }
            }
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
    fn collects_mono_and_multiple_inputs_in_order() {
        for cameras in [1, 2, 3] {
            let inputs = (0..cameras)
                .map(|camera| scripted(vec![Ok(Some(camera)), Ok(Some(camera + 10)), Ok(None)]))
                .collect();
            let (rx, worker) = collect(inputs, 0, 1);
            assert_eq!(receive(&rx).unwrap(), Some((0..cameras).collect()));
            assert_eq!(receive(&rx).unwrap(), Some((10..10 + cameras).collect()));
            assert_eq!(receive(&rx).unwrap(), None);
            worker.join().unwrap();
        }
    }

    #[test]
    fn collection_applies_stereo_offsets_and_ignores_mono_offset() {
        for offset in [-1, 0, 1] {
            let (rx, worker) = collect(
                vec![
                    scripted(vec![Ok(Some(1)), Ok(Some(2)), Ok(None)]),
                    scripted(vec![Ok(Some(11)), Ok(Some(12)), Ok(None)]),
                ],
                offset,
                4,
            );
            let expected = match offset {
                -1 => vec![vec![2, 11]],
                0 => vec![vec![1, 11], vec![2, 12]],
                1 => vec![vec![1, 12]],
                _ => unreachable!(),
            };
            for frames in expected {
                assert_eq!(receive(&rx).unwrap(), Some(frames));
            }
            assert!(receive(&rx).unwrap().is_none());
            worker.join().unwrap();
        }
        let (rx, worker) = collect(vec![scripted(vec![Ok(Some(9)), Ok(None)])], i64::MIN, 1);
        assert_eq!(receive(&rx).unwrap(), Some(vec![9]));
        assert!(receive(&rx).unwrap().is_none());
        worker.join().unwrap();
        for offset in [-1, 1] {
            for terminal in [Ok(None), failure()] {
                let fails = terminal.is_err();
                let selected = usize::from(offset > 0);
                let mut inputs = vec![scripted(vec![Ok(None)]), scripted(vec![Ok(None)])];
                inputs[selected] = scripted(vec![terminal]);
                let (rx, worker) = collect(inputs, offset, 1);
                assert_eq!(receive(&rx).is_err(), fails);
                worker.join().unwrap();
            }
        }
    }

    #[test]
    fn offset_past_eof_preserves_other_camera_failure() {
        for offset in [-1, 1] {
            let shifted = usize::from(offset > 0);
            let mut inputs = vec![scripted(vec![Ok(None)]), scripted(vec![Ok(None)])];
            inputs[1 - shifted] = scripted(vec![failure()]);
            let (rx, worker) = collect(inputs, offset, 1);
            let result = receive(&rx);
            worker.join().unwrap();
            assert!(
                result.is_err(),
                "offset {offset} hid the other camera's error"
            );
        }
    }

    #[test]
    fn collection_preserves_errors_at_eof_and_cancels_blocked_sends() {
        for failed in 0..3 {
            let inputs = (0..3)
                .map(|i| {
                    scripted(if i == failed {
                        vec![failure()]
                    } else {
                        vec![Ok(None)]
                    })
                })
                .collect();
            let (rx, worker) = collect(inputs, 0, 1);
            assert!(receive(&rx).is_err());
            worker.join().unwrap();
        }
        let (rx, worker) = collect(
            vec![scripted(vec![Ok(Some(1)), Ok(Some(2)), Ok(None)])],
            0,
            0,
        );
        drop(rx);
        worker.join().unwrap();
    }

    #[test]
    fn pairs_all_frames_then_explicit_eof() {
        let (rx, worker) = collect(
            vec![
                scripted(vec![Ok(Some(1)), Ok(Some(2)), Ok(None)]),
                scripted(vec![Ok(Some(3)), Ok(Some(4)), Ok(None)]),
            ],
            0,
            1,
        );
        assert_eq!(receive(&rx).unwrap(), Some(vec![1, 3]));
        assert_eq!(receive(&rx).unwrap(), Some(vec![2, 4]));
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
            let (rx, worker) = collect(vec![left, right], 0, 1);
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
            let (rx, worker) = collect(vec![left, right], 0, 1);
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
                let (rx, worker) = collect(vec![left, right], offset, 1);
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
        let (rx, collector) = collect(vec![left, right], 0, 1);
        assert!(receive(&rx).unwrap().is_some());
        drop(rx);
        let (done_tx, done_rx) = sync_channel(1);
        std::thread::spawn(move || {
            collector.join().unwrap();
            left_worker.join().unwrap();
            right_worker.join().unwrap();
            done_tx.send(()).unwrap();
        });
        done_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
    }
}
