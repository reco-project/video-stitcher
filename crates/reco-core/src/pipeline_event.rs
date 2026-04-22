//! Pipeline event sink - structured observability for the stitch loop.
//!
//! `StitchSession` optionally emits a [`PipelineEvent`] at each
//! pipeline stage. Consumers attach a [`PipelineEventSink`] to record
//! the stream - typically wrapped in [`BackpressuredSink`] so emission
//! never stalls the render loop, and written out as JSONL (see the
//! companion sink in `reco-io`).
//!
//! # Design
//!
//! - **Optional**: when no sink is attached, emission is a single `Option`
//!   check. Zero cost for consumers that don't care.
//! - **Non-blocking**: [`BackpressuredSink`] hands events to a bounded
//!   channel and a background writer thread. On overflow the event is
//!   dropped (counter logged at power-of-two milestones); the render
//!   loop never waits on I/O.
//! - **Sample-rate gated**: `every_n_frames` lets callers thin the
//!   stream for long sessions without rebuilding the sink.
//!
//! # Event vocabulary
//!
//! Six variants, one per natural pipeline stage:
//!
//! 1. [`PipelineEvent::FrameStart`] - the frame loop picked up a new frame.
//! 2. [`PipelineEvent::DetectionsRaw`] - detector produced mapped detections.
//! 3. [`PipelineEvent::DetectionFilter`] - a filter stage transformed them
//!    (before + after, so consumers can audit each filter's effect).
//!    Reserved for Step 7's `DetectionFilter` trait; not emitted yet.
//! 4. [`PipelineEvent::WorldState`] - trackers produced the per-frame world.
//! 5. [`PipelineEvent::PanDecision`] - panner produced the raw viewport pose.
//! 6. [`PipelineEvent::PosePresented`] - post-clamp pose the renderer saw.

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Captures events into a Vec for assertions in later tests.
    #[derive(Default)]
    struct VecSink(Vec<PipelineEvent>);

    impl PipelineEventSink for VecSink {
        fn emit(&mut self, event: PipelineEvent) {
            self.0.push(event);
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
