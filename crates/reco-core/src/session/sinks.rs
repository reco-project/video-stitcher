//! Sink attachment and delivery fan-out for
//! [`StitchSession`](crate::session::StitchSession).
//!
//! The session owns a list of attached
//! [`OutputSink`](crate::sink::OutputSink)s and fans every rendered
//! frame out to all of them. Attach-time
//! [`SinkOptions`](crate::session::SinkOptions) decide two things the
//! sink itself cannot know:
//!
//! - **Delivery**: lossless sinks (encoders) run on a dedicated
//!   thread with a bounded queue that blocks the render loop when
//!   full (backpressure); lossy sinks (snapshot taps, previews) are
//!   called inline on the render thread and must never block -
//!   threading them would cost a full-frame memcpy per frame even
//!   when the sink keeps one frame in thirty.
//! - **Error policy**: whether a failing sink aborts the session
//!   (a batch export losing its only encoder) or is detached with a
//!   warning while the others continue (a live stream dropping while
//!   the recording carries on).

use crate::sink::{OutputFrame, OutputSink, PixelFormat, SinkError, SinkInput};
use crate::sink_thread::SinkThread;

/// How the session hands frames to a sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkDelivery {
    /// The sink runs on a dedicated thread behind a bounded queue of
    /// `queue_depth` frames. Each frame is copied into a pooled
    /// buffer; a full queue blocks the render loop (backpressure).
    /// For lossless consumers: encoders, file writers.
    Threaded {
        /// Frames in flight between the render thread and the sink
        /// thread (typically 2).
        queue_depth: usize,
    },
    /// The sink is called on the render thread with the frame
    /// borrowed straight from the readback buffer - zero copies from
    /// the session's side. `consume` must not block; a sink that
    /// needs real work off-loads internally (channel + own thread)
    /// and drops frames when it falls behind.
    Inline,
}

/// What happens when a sink returns an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkErrorPolicy {
    /// Propagate the error; the session run fails.
    Abort,
    /// Log a warning, detach the sink, and keep delivering to the
    /// rest. The failure surfaces once; the run continues.
    Detach,
}

/// Attach-time configuration for a sink.
///
/// Delivery and error policy are deployment decisions, not sink
/// properties: the same RTMP sink can be fatal in a stream-only run
/// and disposable next to a recording.
#[derive(Debug, Clone, Copy)]
pub struct SinkOptions {
    /// How frames reach the sink.
    pub delivery: SinkDelivery,
    /// What a sink error does to the session.
    pub on_error: SinkErrorPolicy,
}

impl SinkOptions {
    /// Lossless sink on its own thread, aborting the session on
    /// failure. The encoder archetype.
    pub fn threaded(queue_depth: usize) -> Self {
        Self {
            delivery: SinkDelivery::Threaded { queue_depth },
            on_error: SinkErrorPolicy::Abort,
        }
    }

    /// Non-blocking sink called on the render thread, detached on
    /// failure. The snapshot/preview-tap archetype.
    pub fn inline_lossy() -> Self {
        Self {
            delivery: SinkDelivery::Inline,
            on_error: SinkErrorPolicy::Detach,
        }
    }
}

/// A sink attached to the session, wrapped per its delivery mode.
pub(crate) struct AttachedSink {
    runner: SinkRunner,
    on_error: SinkErrorPolicy,
    /// Captured at attach time; threaded sinks move to their thread.
    name: String,
}

enum SinkRunner {
    Threaded(SinkThread),
    Inline(Box<dyn OutputSink>),
}

impl AttachedSink {
    pub(crate) fn new(
        sink: Box<dyn OutputSink>,
        options: SinkOptions,
        width: u32,
        height: u32,
    ) -> Self {
        let name = sink.name().to_string();
        let runner = match options.delivery {
            SinkDelivery::Threaded { queue_depth } => {
                log::info!("Sink '{name}': threaded delivery, queue depth {queue_depth}");
                SinkRunner::Threaded(SinkThread::new(sink, width, height, queue_depth))
            }
            SinkDelivery::Inline => {
                log::info!("Sink '{name}': inline delivery on the render thread");
                SinkRunner::Inline(sink)
            }
        };
        Self {
            runner,
            on_error: options.on_error,
            name,
        }
    }

    /// Thread counters for telemetry: (frames, avg consume ms,
    /// backpressure stalls, backpressure ms). `None` for inline sinks.
    pub(crate) fn thread_stats(&self) -> Option<(u64, f32, u64, f32)> {
        match &self.runner {
            SinkRunner::Threaded(t) => Some(t.stats()),
            SinkRunner::Inline(_) => None,
        }
    }

    fn deliver(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        pts_us: i64,
    ) -> Result<(), SinkError> {
        match &mut self.runner {
            SinkRunner::Threaded(t) => t.submit(data, pts_us),
            SinkRunner::Inline(sink) => sink.consume(OutputFrame {
                data,
                width,
                height,
                format: PixelFormat::Nv12,
                pts_us,
            }),
        }
    }

    fn finish(&mut self) -> Result<(), SinkError> {
        match &mut self.runner {
            SinkRunner::Threaded(t) => t.finish(),
            SinkRunner::Inline(sink) => sink.finish(),
        }
    }
}

/// Reject sinks the GPU session cannot feed. The streaming session
/// delivers NV12 CPU bytes only; RGBA delivery arrives with the CPU
/// session loop.
pub(crate) fn validate_sink_input(wants: SinkInput, name: &str) -> Result<(), String> {
    match wants {
        SinkInput::CpuBytes(PixelFormat::Nv12) => Ok(()),
        other => Err(format!(
            "sink '{name}' wants {other:?}, but the session delivers \
             NV12 CPU bytes only"
        )),
    }
}

/// Fan one NV12 frame out to every attached sink, in attach order.
///
/// `pts_us` carries the session frame index (the file encoder tracks
/// its own timestamps). A sink error either propagates (Abort) or
/// detaches that sink with a warning and keeps going (Detach).
pub(crate) fn deliver_frame(
    sinks: &mut Vec<AttachedSink>,
    data: &[u8],
    width: u32,
    height: u32,
    pts_us: i64,
) -> Result<(), SinkError> {
    let mut i = 0;
    while i < sinks.len() {
        match sinks[i].deliver(data, width, height, pts_us) {
            Ok(()) => i += 1,
            Err(e) => match sinks[i].on_error {
                SinkErrorPolicy::Abort => return Err(e),
                SinkErrorPolicy::Detach => {
                    log::warn!(
                        "Sink '{}' failed ({e}); detaching, {} sink(s) continue",
                        sinks[i].name,
                        sinks.len() - 1
                    );
                    sinks.remove(i);
                }
            },
        }
    }
    Ok(())
}

/// Finish every sink in attach order.
///
/// All sinks get their `finish` call even when an earlier one fails,
/// so every output file is finalized; the first Abort-policy error is
/// returned afterwards, Detach-policy errors are logged.
pub(crate) fn finish_all(sinks: &mut Vec<AttachedSink>) -> Result<(), SinkError> {
    let mut first_abort: Option<SinkError> = None;
    for sink in sinks.iter_mut() {
        if let Err(e) = sink.finish() {
            match sink.on_error {
                SinkErrorPolicy::Abort => {
                    if first_abort.is_none() {
                        first_abort = Some(e);
                    } else {
                        log::warn!("Sink '{}' also failed to finish: {e}", sink.name);
                    }
                }
                SinkErrorPolicy::Detach => {
                    log::warn!("Sink '{}' failed to finish: {e}", sink.name);
                }
            }
        }
    }
    sinks.clear();
    match first_abort {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    /// Records every frame it consumes and its finish call.
    struct RecordingSink {
        label: &'static str,
        frames: Arc<Mutex<Vec<Vec<u8>>>>,
        finished: Arc<AtomicU64>,
        /// Fail consume from this frame count on (u64::MAX = never).
        fail_after: u64,
        seen: u64,
    }

    impl RecordingSink {
        fn new(
            label: &'static str,
            frames: Arc<Mutex<Vec<Vec<u8>>>>,
            finished: Arc<AtomicU64>,
        ) -> Self {
            Self {
                label,
                frames,
                finished,
                fail_after: u64::MAX,
                seen: 0,
            }
        }
    }

    impl OutputSink for RecordingSink {
        fn name(&self) -> &str {
            self.label
        }
        fn wants(&self) -> SinkInput {
            SinkInput::CpuBytes(PixelFormat::Nv12)
        }
        fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
            if self.seen >= self.fail_after {
                return Err(SinkError::Consume {
                    frame_index: Some(self.seen),
                    reason: "scripted failure".into(),
                });
            }
            self.seen += 1;
            self.frames.lock().unwrap().push(frame.data.to_vec());
            Ok(())
        }
        fn finish(&mut self) -> Result<(), SinkError> {
            self.finished.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn attach(sink: RecordingSink, options: SinkOptions) -> AttachedSink {
        AttachedSink::new(Box::new(sink), options, 16, 16)
    }

    fn frame(byte: u8) -> Vec<u8> {
        vec![byte; 16 * 16 * 3 / 2]
    }

    #[test]
    fn all_sinks_see_identical_frames() {
        let (f1, f2) = (
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Mutex::new(Vec::new())),
        );
        let done = Arc::new(AtomicU64::new(0));
        let mut sinks = vec![
            attach(
                RecordingSink::new("threaded", f1.clone(), done.clone()),
                SinkOptions::threaded(2),
            ),
            attach(
                RecordingSink::new("inline", f2.clone(), done.clone()),
                SinkOptions::inline_lossy(),
            ),
        ];

        for i in 0..5u8 {
            deliver_frame(&mut sinks, &frame(i), 16, 16, i as i64).unwrap();
        }
        finish_all(&mut sinks).unwrap();

        let (f1, f2) = (f1.lock().unwrap(), f2.lock().unwrap());
        assert_eq!(f1.len(), 5);
        assert_eq!(*f1, *f2, "threaded and inline sinks saw different frames");
        assert_eq!(done.load(Ordering::SeqCst), 2, "both sinks finished");
    }

    #[test]
    fn detach_sink_failure_is_isolated() {
        let (f1, f2) = (
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Mutex::new(Vec::new())),
        );
        let done = Arc::new(AtomicU64::new(0));
        let mut failing = RecordingSink::new("flaky", f1.clone(), done.clone());
        failing.fail_after = 2;
        let mut sinks = vec![
            attach(failing, SinkOptions::inline_lossy()),
            attach(
                RecordingSink::new("stable", f2.clone(), done.clone()),
                SinkOptions::inline_lossy(),
            ),
        ];

        for i in 0..5u8 {
            deliver_frame(&mut sinks, &frame(i), 16, 16, i as i64).unwrap();
        }

        assert_eq!(sinks.len(), 1, "failing sink was detached");
        assert_eq!(f1.lock().unwrap().len(), 2, "flaky sink kept its 2 frames");
        assert_eq!(f2.lock().unwrap().len(), 5, "stable sink saw every frame");
        finish_all(&mut sinks).unwrap();
    }

    #[test]
    fn abort_sink_failure_propagates() {
        let frames = Arc::new(Mutex::new(Vec::new()));
        let done = Arc::new(AtomicU64::new(0));
        let mut failing = RecordingSink::new("critical", frames.clone(), done.clone());
        failing.fail_after = 0;
        let mut sinks = vec![attach(
            failing,
            SinkOptions {
                delivery: SinkDelivery::Inline,
                on_error: SinkErrorPolicy::Abort,
            },
        )];

        let err = deliver_frame(&mut sinks, &frame(0), 16, 16, 0);
        assert!(err.is_err(), "abort-policy sink error must propagate");
    }

    #[test]
    fn finish_reaches_every_sink_despite_failure() {
        struct FailingFinish;
        impl OutputSink for FailingFinish {
            fn name(&self) -> &str {
                "bad-finish"
            }
            fn wants(&self) -> SinkInput {
                SinkInput::CpuBytes(PixelFormat::Nv12)
            }
            fn consume(&mut self, _f: OutputFrame<'_>) -> Result<(), SinkError> {
                Ok(())
            }
            fn finish(&mut self) -> Result<(), SinkError> {
                Err(SinkError::Finish {
                    reason: "scripted".into(),
                })
            }
        }

        let frames = Arc::new(Mutex::new(Vec::new()));
        let done = Arc::new(AtomicU64::new(0));
        let mut sinks = vec![
            AttachedSink::new(
                Box::new(FailingFinish),
                SinkOptions {
                    delivery: SinkDelivery::Inline,
                    on_error: SinkErrorPolicy::Abort,
                },
                16,
                16,
            ),
            attach(
                RecordingSink::new("after", frames, done.clone()),
                SinkOptions::inline_lossy(),
            ),
        ];

        let result = finish_all(&mut sinks);
        assert!(result.is_err(), "abort-policy finish error propagates");
        assert_eq!(
            done.load(Ordering::SeqCst),
            1,
            "the sink after the failing one was still finished"
        );
    }

    #[test]
    fn rgba_sink_is_rejected() {
        struct RgbaSink;
        impl OutputSink for RgbaSink {
            fn name(&self) -> &str {
                "rgba"
            }
            fn wants(&self) -> SinkInput {
                SinkInput::CpuBytes(PixelFormat::Rgba8)
            }
            fn consume(&mut self, _f: OutputFrame<'_>) -> Result<(), SinkError> {
                Ok(())
            }
            fn finish(&mut self) -> Result<(), SinkError> {
                Ok(())
            }
        }
        assert!(validate_sink_input(RgbaSink.wants(), RgbaSink.name()).is_err());
    }
}
