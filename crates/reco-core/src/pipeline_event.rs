//! Pipeline event sink - structured observability for the stitch loop.
//!
//! `StitchSession` optionally emits a
//! [`PipelineEvent`](crate::pipeline_event::PipelineEvent) at each
//! pipeline stage. Consumers attach a
//! [`PipelineEventSink`](crate::pipeline_event::PipelineEventSink) to record
//! the stream - typically wrapped in
//! [`BackpressuredSink`](crate::pipeline_event::BackpressuredSink) so emission
//! never stalls the render loop, and written out as JSONL (see the
//! companion sink in `reco-io`).
//!
//! # Design
//!
//! - **Optional**: when no sink is attached, emission is a single `Option`
//!   check. Zero cost for consumers that don't care.
//! - **Non-blocking**:
//!   [`BackpressuredSink`](crate::pipeline_event::BackpressuredSink) hands
//!   events to a bounded channel and a background writer thread. On overflow
//!   the event is dropped (counter logged at power-of-two milestones); the
//!   render loop never waits on I/O.
//! - **Sample-rate gated**: `every_n_frames` lets callers thin the
//!   stream for long sessions without rebuilding the sink.
//!
//! # Event vocabulary
//!
//! Six variants, one per natural pipeline stage:
//!
//! 1. [`FrameStart`](crate::pipeline_event::PipelineEvent::FrameStart) - the frame loop picked up a new frame.
//! 2. [`DetectionsRaw`](crate::pipeline_event::PipelineEvent::DetectionsRaw) - detector produced mapped detections.
//! 3. [`DetectionFilter`](crate::pipeline_event::PipelineEvent::DetectionFilter) - a filter stage transformed them
//!    (before + after, so consumers can audit each filter's effect).
//!    Reserved for Step 7's `DetectionFilter` trait; not emitted yet.
//! 4. [`WorldState`](crate::pipeline_event::PipelineEvent::WorldState) - trackers produced the per-frame world.
//! 5. [`PanDecision`](crate::pipeline_event::PipelineEvent::PanDecision) - panner produced the raw viewport pose.
//! 6. [`PosePresented`](crate::pipeline_event::PipelineEvent::PosePresented) - post-clamp pose the renderer saw.

use crate::director::{MappedDetection, ViewportPosition};
use crate::tracker::TrackedEntity;

/// A single observable event from the stitch pipeline.
///
/// Events are cheap to construct (plain data, no locking). Consumers
/// that want a rich trace should use [`BackpressuredSink`] to keep
/// serialization off the render thread.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PipelineEvent {
    /// Frame loop picked up a new frame. Fires once per frame before
    /// any detection or pose work.
    FrameStart {
        /// Monotonic frame counter, starting at 0.
        frame_index: u64,
        /// Milliseconds since session start.
        timestamp_ms: f64,
    },

    /// Detector produced mapped detections for this frame, before
    /// any `DetectionFilter` stages run (Step 7).
    DetectionsRaw {
        /// Monotonic frame counter.
        frame_index: u64,
        /// Detections the detector emitted (one per class per camera,
        /// projected into panorama space).
        detections: Vec<MappedDetection>,
    },

    /// A `DetectionFilter` stage transformed the detection list.
    /// Reserved for Step 7; not emitted yet but included in the
    /// vocabulary so consumers and the JSONL schema are stable now.
    DetectionFilter {
        /// Monotonic frame counter.
        frame_index: u64,
        /// Stage name (`"FlickerFilter"`, `"FeetInRoi"`, ...).
        filter_name: &'static str,
        /// Detections the filter saw on input.
        before: Vec<MappedDetection>,
        /// Detections the filter produced on output.
        after: Vec<MappedDetection>,
    },

    /// Trackers produced the per-frame world state.
    WorldState {
        /// Monotonic frame counter.
        frame_index: u64,
        /// Milliseconds since session start.
        timestamp_ms: f64,
        /// Players the player tracker emitted this frame.
        players: Vec<TrackedEntity>,
        /// Ball the ball tracker emitted this frame, if any.
        ball: Option<TrackedEntity>,
    },

    /// Panner produced a raw viewport pose (pre-clamp).
    PanDecision {
        /// Monotonic frame counter.
        frame_index: u64,
        /// The pose the panner decided on, before any coverage clamp.
        pose: ViewportPosition,
    },

    /// Final pose the renderer received this frame (post-clamp).
    PosePresented {
        /// Monotonic frame counter.
        frame_index: u64,
        /// Pose after all clamping and FOV adjustments.
        pose: ViewportPosition,
    },
}

impl PipelineEvent {
    /// Frame index the event belongs to. Handy for sample-rate
    /// gating in `BackpressuredSink`.
    pub fn frame_index(&self) -> u64 {
        match self {
            PipelineEvent::FrameStart { frame_index, .. }
            | PipelineEvent::DetectionsRaw { frame_index, .. }
            | PipelineEvent::DetectionFilter { frame_index, .. }
            | PipelineEvent::WorldState { frame_index, .. }
            | PipelineEvent::PanDecision { frame_index, .. }
            | PipelineEvent::PosePresented { frame_index, .. } => *frame_index,
        }
    }
}

/// A sink that receives [`PipelineEvent`]s.
///
/// Implementations must be `Send` because attaching via
/// `StitchSession::set_event_sink` moves the sink across the session
/// boundary; the backpressured wrapper additionally hands it to a
/// writer thread.
///
/// `emit` is called synchronously from the render loop. Keep the
/// implementation fast, or wrap it in [`BackpressuredSink`] so the
/// heavy work runs on a background thread.
pub trait PipelineEventSink: Send {
    /// Record one event.
    fn emit(&mut self, event: PipelineEvent);
}

/// Non-blocking wrapper around an arbitrary [`PipelineEventSink`].
///
/// Events flow through a bounded mpsc channel to a background thread
/// that calls the inner sink. On overflow the event is dropped and
/// a counter is incremented; `log::warn` fires at power-of-two
/// milestones so a slow inner sink shows up without drowning the log.
///
/// The render thread never blocks on the sink: `emit` is always
/// `try_send` + a branch.
///
/// # Sample-rate gate
///
/// `every_n_frames = Some(n)` records events only when
/// `event.frame_index() % n == 0`. Useful for long sessions where a
/// full per-frame trace would be wasteful. `None` (or `Some(1)`)
/// records every event.
///
/// # Drop semantics
///
/// On drop the sender is closed first (causing the background thread's
/// `recv` to return `Err`), then the thread is joined. That drains the
/// channel so already-queued events reach the inner sink before the
/// process exits.
pub struct BackpressuredSink {
    tx: Option<std::sync::mpsc::SyncSender<PipelineEvent>>,
    thread: Option<std::thread::JoinHandle<()>>,
    dropped: u64,
    every_n_frames: Option<u32>,
}

impl BackpressuredSink {
    /// Wrap `inner` so calls to [`emit`](PipelineEventSink::emit) never
    /// block. `capacity` is the bounded channel size; `every_n_frames`
    /// is the optional sample-rate gate (see type doc).
    ///
    /// `capacity` of `256` is plenty for most sinks - a JSONL writer
    /// can keep up at hundreds of events per frame. Raise it if the
    /// inner sink does heavy I/O (network, compressed output).
    pub fn new(
        mut inner: Box<dyn PipelineEventSink>,
        capacity: usize,
        every_n_frames: Option<u32>,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<PipelineEvent>(capacity);
        let thread = std::thread::Builder::new()
            .name("pipeline-event-sink".into())
            .spawn(move || {
                while let Ok(event) = rx.recv() {
                    inner.emit(event);
                }
            })
            .expect("spawn background event-sink thread");
        Self {
            tx: Some(tx),
            thread: Some(thread),
            dropped: 0,
            every_n_frames,
        }
    }

    /// Total events dropped due to channel overflow since construction.
    /// Test-only; production code watches the `log::warn` stream.
    #[cfg(test)]
    pub(crate) fn dropped(&self) -> u64 {
        self.dropped
    }
}

impl PipelineEventSink for BackpressuredSink {
    fn emit(&mut self, event: PipelineEvent) {
        // Sample-rate gate. `Some(1)` behaves like `None` (every frame).
        if let Some(n) = self.every_n_frames
            && n > 1
            && !event.frame_index().is_multiple_of(n as u64)
        {
            return;
        }
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        match tx.try_send(event) {
            Ok(()) => {}
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                self.dropped += 1;
                if self.dropped.is_power_of_two() {
                    log::warn!(
                        "PipelineEventSink lagging: dropped {} events total",
                        self.dropped
                    );
                }
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                // Background thread exited; silently ignore.
            }
        }
    }
}

impl Drop for BackpressuredSink {
    fn drop(&mut self) {
        // Close the channel first so the background thread's recv()
        // breaks; then join so queued events get written out.
        drop(self.tx.take());
        if let Some(t) = self.thread.take() {
            // If the thread panicked we ignore the error here - the
            // process is either already unwinding or intentionally
            // tearing down.
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Captures events into a Vec for assertions in later tests.
    #[derive(Default)]
    struct VecSink(Vec<PipelineEvent>);

    impl PipelineEventSink for VecSink {
        fn emit(&mut self, event: PipelineEvent) {
            self.0.push(event);
        }
    }

    /// Thread-safe capture sink for BackpressuredSink tests.
    #[derive(Default, Clone)]
    struct SharedVecSink(Arc<Mutex<Vec<PipelineEvent>>>);

    impl PipelineEventSink for SharedVecSink {
        fn emit(&mut self, event: PipelineEvent) {
            self.0.lock().unwrap().push(event);
        }
    }

    fn mk_frame(i: u64) -> PipelineEvent {
        PipelineEvent::FrameStart {
            frame_index: i,
            timestamp_ms: i as f64,
        }
    }

    #[test]
    fn frame_index_extractor_covers_every_variant() {
        let events = [
            PipelineEvent::FrameStart {
                frame_index: 1,
                timestamp_ms: 0.0,
            },
            PipelineEvent::DetectionsRaw {
                frame_index: 2,
                detections: Vec::new(),
            },
            PipelineEvent::DetectionFilter {
                frame_index: 3,
                filter_name: "Test",
                before: Vec::new(),
                after: Vec::new(),
            },
            PipelineEvent::WorldState {
                frame_index: 4,
                timestamp_ms: 1.0,
                players: Vec::new(),
                ball: None,
            },
            PipelineEvent::PanDecision {
                frame_index: 5,
                pose: ViewportPosition::default(),
            },
            PipelineEvent::PosePresented {
                frame_index: 6,
                pose: ViewportPosition::default(),
            },
        ];
        for (i, ev) in events.iter().enumerate() {
            assert_eq!(ev.frame_index(), (i + 1) as u64);
        }
    }

    #[test]
    fn sink_trait_is_dyn_compatible() {
        // StitchSession stores this as `Option<Box<dyn PipelineEventSink>>`;
        // verify the shape now so later wiring is trivial.
        let mut sink: Box<dyn PipelineEventSink> = Box::<VecSink>::default();
        sink.emit(PipelineEvent::FrameStart {
            frame_index: 0,
            timestamp_ms: 0.0,
        });
    }

    #[test]
    fn backpressured_sink_delivers_all_events_under_capacity() {
        let captured = SharedVecSink::default();
        {
            let mut sink = BackpressuredSink::new(Box::new(captured.clone()), 64, None);
            for i in 0..50 {
                sink.emit(mk_frame(i));
            }
            // Drop joins the thread and drains.
        }
        let events = captured.0.lock().unwrap();
        assert_eq!(events.len(), 50, "all 50 events should be delivered");
        for (i, ev) in events.iter().enumerate() {
            assert_eq!(ev.frame_index(), i as u64);
        }
    }

    #[test]
    fn backpressured_sink_sample_rate_gate_drops_off_beats() {
        let captured = SharedVecSink::default();
        {
            let mut sink = BackpressuredSink::new(Box::new(captured.clone()), 64, Some(5));
            for i in 0..20 {
                sink.emit(mk_frame(i));
            }
        }
        let events = captured.0.lock().unwrap();
        // every_n=5 keeps frame_index in {0, 5, 10, 15} = 4 events.
        let indices: Vec<u64> = events.iter().map(|e| e.frame_index()).collect();
        assert_eq!(indices, vec![0, 5, 10, 15]);
    }

    #[test]
    fn backpressured_sink_drops_on_overflow_instead_of_blocking() {
        // A slow sink + tiny capacity + a burst: overflow path must
        // increment the dropped counter without wedging the caller.
        struct SlowSink;
        impl PipelineEventSink for SlowSink {
            fn emit(&mut self, _event: PipelineEvent) {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }

        let mut sink = BackpressuredSink::new(Box::new(SlowSink), 2, None);
        let start = std::time::Instant::now();
        for i in 0..100 {
            sink.emit(mk_frame(i));
        }
        let elapsed = start.elapsed();
        // 100 events with a 20ms sink and capacity 2 would take >2s
        // if emit blocked. Non-blocking emit must finish promptly.
        assert!(
            elapsed.as_millis() < 500,
            "emit burst should not block on sink; took {elapsed:?}"
        );
        assert!(
            sink.dropped() > 0,
            "overflow path must have dropped at least one event"
        );
    }

    #[test]
    fn event_serializes_to_tagged_json() {
        // JsonlSink in reco-io relies on this serialization shape.
        // Lock it here so the schema can't silently drift.
        let ev = PipelineEvent::FrameStart {
            frame_index: 7,
            timestamp_ms: 1234.5,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains(r#""kind":"frame_start""#));
        assert!(json.contains(r#""frame_index":7"#));
        assert!(json.contains(r#""timestamp_ms":1234.5"#));
    }
}
