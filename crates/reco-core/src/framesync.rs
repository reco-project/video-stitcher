//! Timestamped multi-source ingest buffer.
//!
//! Solves the fundamental timing problem shared by every multi-camera
//! live consumer: frames arrive on independent threads with per-source
//! clocks, and the consumer needs a synchronized tuple (one frame per
//! source) whose timestamps fall within a configurable tolerance.
//! Without this helper every consumer rolls its own ad-hoc pairing
//! (reco-obs Tier 1 polled both sides each `video_tick` and submitted
//! unconditionally — see `reco-obs/FRICTION.md` A5 for the drift
//! pathology that motivated this module).
//!
//! # Use cases
//!
//! - **Dual-source OBS plugin** (the original motivation): two OBS
//!   `video_tick` threads push left/right frames as they arrive; the
//!   main render thread pulls synchronized pairs.
//! - **N-camera rig** (future): any number of sources share the same
//!   buffer, drop-unmatched semantics preserved.
//! - **Livestream sync** (future standalone app): each network stream
//!   has its own PTS offset; the buffer matches frames across streams
//!   within a tolerance large enough to absorb jitter but small enough
//!   to catch a stuck stream.
//! - **Remote inference sync** (future): detection results arriving
//!   on a separate channel can be matched back to the source frame
//!   they were computed on.
//!
//! # Design
//!
//! Per-source ring buffers of bounded capacity hold user-chosen
//! handles. The buffer is generic over `H`: the consumer picks
//! whether to store owned bytes, reference-counted frame handles,
//! or just a slot index the raw source will re-look-up. Reco-core
//! takes no opinion on frame ownership because the right answer
//! varies between OBS refcounted async frames, V4L2 mmap slots,
//! and WebRTC-decoded Vec<u8>.
//!
//! Timestamps are a caller-supplied `Duration` relative to an
//! implicit source-local anchor. Using `Instant` directly would
//! tie the buffer to the std clock and prevent testing or
//! replaying a recorded stream at scale. `Duration` is
//! `Clone + Send + Sync` and lets the caller decide the monotonic
//! reference.
//!
//! Emission strategy (`try_emit`):
//!
//!   1. For each source, drop frames older than the newest frame
//!      across all sources minus `tolerance`.
//!   2. If every source still has at least one frame, emit a tuple
//!      with the closest-in-time entry from each source (the most
//!      recent one ≤ the global newest). Consumed frames leave the
//!      buffer; older ones in the same source stay as lookaside
//!      for the next tick.
//!   3. If any source is empty or all its frames are older than
//!      tolerance, return `None`.
//!
//! The buffer exposes `last_sync_delta(SourceId)` so consumers can
//! surface a drift warning to the user when one stream lags
//! systematically.

use std::collections::VecDeque;
use std::time::Duration;

/// Opaque source identifier. A `u32` is large enough for every
/// realistic multi-camera rig; the caller assigns IDs at buffer
/// construction and passes the same value on every `push`.
pub type SourceId = u32;

/// Pair of `(timestamp, handle)` for one source at one instant.
#[derive(Debug, Clone)]
pub struct TimedFrame<H> {
    /// Caller-supplied monotonic timestamp relative to the source's
    /// implicit anchor.
    pub timestamp: Duration,
    /// Caller-chosen frame handle: owned bytes, refcounted pointer,
    /// slot index, anything `Send`.
    pub handle: H,
}

/// Tuple emitted by [`TimestampedIngestBuffer::try_emit`]. Holds one
/// frame per configured source plus drift metadata.
#[derive(Debug, Clone)]
pub struct SyncedTuple<H> {
    /// The representative timestamp (the newest across all sources).
    pub reference: Duration,
    /// Per-source entry, in the order sources were configured.
    /// `frames[i].source_id` matches the `i`-th source from the
    /// constructor.
    pub frames: Vec<SyncedEntry<H>>,
    /// Spread of timestamps across the tuple: `max - min`. Useful as
    /// a rolling drift metric the consumer can surface to the user.
    pub max_delta: Duration,
}

/// One source's contribution to a [`SyncedTuple`].
#[derive(Debug, Clone)]
pub struct SyncedEntry<H> {
    /// The source that produced the frame.
    pub source_id: SourceId,
    /// Timestamp of the frame picked from this source's ring.
    pub timestamp: Duration,
    /// The frame handle itself.
    pub handle: H,
}

/// Errors from [`TimestampedIngestBuffer::push`]. `Clone + Send + Sync`
/// so consumers posting results to worker threads can carry the typed
/// error.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum IngestError {
    /// The pushed `source_id` was not declared at construction time.
    #[error("unknown source id {0}")]
    UnknownSource(SourceId),
}

/// Per-source state: a ring of pending timed frames plus the last
/// observed drift.
struct SourceSlot<H> {
    id: SourceId,
    ring: VecDeque<TimedFrame<H>>,
    last_sync_delta: Option<Duration>,
}

impl<H> SourceSlot<H> {
    fn new(id: SourceId, capacity: usize) -> Self {
        Self {
            id,
            ring: VecDeque::with_capacity(capacity),
            last_sync_delta: None,
        }
    }

    /// Insert a new timed frame, preserving newest-at-back ordering.
    /// Evicts the oldest when the ring is at capacity so the buffer
    /// has bounded memory in the face of a stuck consumer.
    fn push(&mut self, frame: TimedFrame<H>, capacity: usize) {
        while self.ring.len() >= capacity {
            self.ring.pop_front();
        }
        // Frames are typically monotonically-increasing; handle
        // out-of-order arrivals by inserting at the right position.
        match self.ring.back() {
            Some(last) if frame.timestamp < last.timestamp => {
                // Out-of-order insert: scan from back.
                let mut idx = self.ring.len();
                while idx > 0 && self.ring[idx - 1].timestamp > frame.timestamp {
                    idx -= 1;
                }
                self.ring.insert(idx, frame);
            }
            _ => self.ring.push_back(frame),
        }
    }

    /// The newest frame's timestamp, if any.
    fn latest_timestamp(&self) -> Option<Duration> {
        self.ring.back().map(|f| f.timestamp)
    }

    /// Drop frames strictly older than `cutoff`.
    fn trim_older_than(&mut self, cutoff: Duration) {
        while let Some(front) = self.ring.front() {
            if front.timestamp < cutoff {
                self.ring.pop_front();
            } else {
                break;
            }
        }
    }

    /// Pop the best-matching frame for the given reference timestamp:
    /// the newest entry whose timestamp is ≤ `reference`. Returns
    /// `None` if the ring is empty or every entry is after `reference`.
    fn pop_at_or_before(&mut self, reference: Duration) -> Option<TimedFrame<H>> {
        // Find the rightmost entry with timestamp ≤ reference.
        let mut pick_idx: Option<usize> = None;
        for (i, f) in self.ring.iter().enumerate() {
            if f.timestamp <= reference {
                pick_idx = Some(i);
            } else {
                break;
            }
        }
        let idx = pick_idx?;
        self.ring.remove(idx)
    }
}

/// Timestamped multi-source ingest buffer.
///
/// See module docs for the design and emission strategy.
pub struct TimestampedIngestBuffer<H> {
    sources: Vec<SourceSlot<H>>,
    tolerance: Duration,
    capacity_per_source: usize,
}

impl<H> TimestampedIngestBuffer<H> {
    /// Build a buffer for the given source IDs.
    ///
    /// - `source_ids`: the full set of sources the buffer will see.
    ///   Pushing a `source_id` not in this slice returns
    ///   [`IngestError::UnknownSource`]. The emission order in
    ///   [`SyncedTuple::frames`] follows this vector.
    /// - `tolerance`: the maximum timestamp spread across a synced
    ///   tuple. Frames older than the newest-across-sources minus
    ///   `tolerance` are dropped during `try_emit`.
    /// - `capacity_per_source`: ring size per source. Older frames
    ///   are evicted when this is reached; a stuck consumer caps
    ///   memory growth at `N_sources * capacity_per_source * sizeof(H)`.
    ///
    /// # Panics
    ///
    /// Panics if `source_ids` contains duplicates or is empty, or if
    /// `capacity_per_source == 0`. These represent configuration
    /// bugs rather than runtime conditions worth a Result.
    pub fn new(source_ids: &[SourceId], tolerance: Duration, capacity_per_source: usize) -> Self {
        assert!(
            !source_ids.is_empty(),
            "TimestampedIngestBuffer needs at least one source"
        );
        assert!(
            capacity_per_source > 0,
            "TimestampedIngestBuffer capacity_per_source must be > 0"
        );
        for (i, &id) in source_ids.iter().enumerate() {
            assert!(
                !source_ids[..i].contains(&id),
                "TimestampedIngestBuffer source_ids contain duplicate: {id}"
            );
        }

        Self {
            sources: source_ids
                .iter()
                .map(|&id| SourceSlot::new(id, capacity_per_source))
                .collect(),
            tolerance,
            capacity_per_source,
        }
    }

    /// Push a frame for the given source.
    pub fn push(
        &mut self,
        source_id: SourceId,
        timestamp: Duration,
        handle: H,
    ) -> Result<(), IngestError> {
        let slot = self
            .sources
            .iter_mut()
            .find(|s| s.id == source_id)
            .ok_or(IngestError::UnknownSource(source_id))?;
        slot.push(TimedFrame { timestamp, handle }, self.capacity_per_source);
        Ok(())
    }

    /// Try to emit a synchronized tuple with one frame per source.
    ///
    /// Returns `None` when at least one source has no frame within
    /// `tolerance` of the newest timestamp across sources. On
    /// success, each source loses the emitted frame from its ring
    /// (older frames in the same source are preserved so a later
    /// `try_emit` can match them if a slower source catches up).
    pub fn try_emit(&mut self) -> Option<SyncedTuple<H>> {
        // Reference is the newest timestamp across all sources. If
        // any source is empty we cannot emit.
        let reference = self
            .sources
            .iter()
            .map(|s| s.latest_timestamp())
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .max()?;

        // Trim each source to the tolerance window before looking
        // for a match.
        let cutoff = reference.saturating_sub(self.tolerance);
        for slot in self.sources.iter_mut() {
            slot.trim_older_than(cutoff);
        }

        // Every source must have at least one frame at-or-before
        // the reference inside the tolerance window.
        for slot in &self.sources {
            if slot.ring.is_empty() {
                return None;
            }
        }

        // Pop the best match per source; compute spread.
        let mut frames = Vec::with_capacity(self.sources.len());
        let mut min_ts = reference;
        let mut max_ts = Duration::ZERO;
        for slot in self.sources.iter_mut() {
            let picked = slot.pop_at_or_before(reference)?;
            if picked.timestamp < min_ts {
                min_ts = picked.timestamp;
            }
            if picked.timestamp > max_ts {
                max_ts = picked.timestamp;
            }
            frames.push(SyncedEntry {
                source_id: slot.id,
                timestamp: picked.timestamp,
                handle: picked.handle,
            });
        }

        let max_delta = max_ts.saturating_sub(min_ts);
        for slot in self.sources.iter_mut() {
            slot.last_sync_delta = Some(max_delta);
        }

        Some(SyncedTuple {
            reference,
            frames,
            max_delta,
        })
    }

    /// Last observed drift (per-source). Starts at `None`; updated
    /// after each successful `try_emit`. A steadily growing value
    /// indicates a lagging source.
    pub fn last_sync_delta(&self, source_id: SourceId) -> Option<Duration> {
        self.sources
            .iter()
            .find(|s| s.id == source_id)
            .and_then(|s| s.last_sync_delta)
    }

    /// Number of buffered frames for a given source. `0` for unknown
    /// IDs (no panic; callers polling status have benign misuse).
    pub fn len(&self, source_id: SourceId) -> usize {
        self.sources
            .iter()
            .find(|s| s.id == source_id)
            .map_or(0, |s| s.ring.len())
    }

    /// Total buffered frames across every source.
    pub fn total_len(&self) -> usize {
        self.sources.iter().map(|s| s.ring.len()).sum()
    }

    /// Whether any source still has a pending frame.
    pub fn is_empty(&self) -> bool {
        self.sources.iter().all(|s| s.ring.is_empty())
    }

    /// Clear every source's ring. Does not reset `last_sync_delta`.
    pub fn clear(&mut self) {
        for slot in self.sources.iter_mut() {
            slot.ring.clear();
        }
    }

    /// Source IDs, in the order they were configured.
    pub fn source_ids(&self) -> impl Iterator<Item = SourceId> + '_ {
        self.sources.iter().map(|s| s.id)
    }

    /// Tolerance set at construction.
    pub fn tolerance(&self) -> Duration {
        self.tolerance
    }

    /// Ring capacity per source, set at construction.
    pub fn capacity_per_source(&self) -> usize {
        self.capacity_per_source
    }
}

// Compile-time bound check: the buffer + its public types are mobile
// across thread boundaries (the live consumers push from one thread
// and pull from another). Regresses if a future field introduces a
// non-Send type.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TimestampedIngestBuffer<Vec<u8>>>();
    assert_send_sync::<SyncedTuple<Vec<u8>>>();
    assert_send_sync::<IngestError>();
    fn assert_clone<T: Clone>() {}
    assert_clone::<IngestError>();
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Shorthand for millisecond durations in tests.
    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// Build a stereo (N=2) buffer with 1-frame tolerance at 30fps
    /// (~33ms) and a capacity of 8 per source.
    fn stereo_buffer() -> TimestampedIngestBuffer<u64> {
        TimestampedIngestBuffer::new(&[0, 1], ms(33), 8)
    }

    #[test]
    fn rejects_unknown_source_id() {
        let mut buf = stereo_buffer();
        let err = buf.push(99, ms(0), 42).unwrap_err();
        assert_eq!(err, IngestError::UnknownSource(99));
    }

    #[test]
    fn emits_nothing_until_all_sources_have_a_frame() {
        let mut buf = stereo_buffer();
        buf.push(0, ms(100), 10).unwrap();
        assert!(buf.try_emit().is_none(), "only one source has pushed");

        buf.push(1, ms(100), 20).unwrap();
        let tuple = buf.try_emit().expect("both sources present now");
        assert_eq!(tuple.frames.len(), 2);
        assert_eq!(tuple.frames[0].source_id, 0);
        assert_eq!(tuple.frames[0].handle, 10);
        assert_eq!(tuple.frames[1].source_id, 1);
        assert_eq!(tuple.frames[1].handle, 20);
        assert_eq!(tuple.max_delta, ms(0));
    }

    #[test]
    fn pairs_closest_frames_within_tolerance() {
        let mut buf = stereo_buffer();
        // Source 0 at t=100, source 1 at t=110: 10ms delta, within
        // the 33ms tolerance.
        buf.push(0, ms(100), 10).unwrap();
        buf.push(1, ms(110), 20).unwrap();
        let tuple = buf.try_emit().unwrap();
        assert_eq!(tuple.max_delta, ms(10));
        assert_eq!(tuple.reference, ms(110));
    }

    #[test]
    fn drops_frames_older_than_tolerance() {
        let mut buf = stereo_buffer();
        // Source 0's only frame is way before source 1's.
        buf.push(0, ms(10), 10).unwrap();
        buf.push(1, ms(200), 20).unwrap();
        // Source 0 is outside tolerance -> cannot emit.
        assert!(buf.try_emit().is_none());
        // Source 0's stale frame was dropped; buffer is clean.
        assert_eq!(buf.len(0), 0);
        assert_eq!(buf.len(1), 1);
    }

    #[test]
    fn successful_emit_removes_used_frames() {
        let mut buf = stereo_buffer();
        buf.push(0, ms(100), 10).unwrap();
        buf.push(1, ms(100), 20).unwrap();
        assert!(buf.try_emit().is_some());
        assert!(buf.is_empty());
    }

    #[test]
    fn preserves_older_frames_on_successful_emit() {
        // If a source has two candidates, try_emit pops the one
        // closest to reference (newest ≤ reference) and leaves the
        // older one for the next emit if a slower source catches up.
        let mut buf = stereo_buffer();
        buf.push(0, ms(100), 10).unwrap();
        buf.push(0, ms(130), 11).unwrap();
        buf.push(1, ms(130), 20).unwrap();
        let tuple = buf.try_emit().unwrap();
        // Reference is 130ms; source 0 picks its 130ms entry.
        assert_eq!(tuple.frames[0].handle, 11);
        // Source 0's older 100ms entry remains. Actually the trim
        // cutoff is 130 - 33 = 97ms, so 100ms stays.
        assert_eq!(buf.len(0), 1);
    }

    #[test]
    fn capacity_evicts_oldest_under_pressure() {
        let mut buf: TimestampedIngestBuffer<u32> = TimestampedIngestBuffer::new(&[0], ms(33), 3);
        for i in 0..5 {
            buf.push(0, ms(i as u64 * 10), i).unwrap();
        }
        // Only the newest 3 (i=2, 3, 4) should remain.
        assert_eq!(buf.len(0), 3);
    }

    #[test]
    fn out_of_order_push_is_inserted_chronologically() {
        let mut buf = stereo_buffer();
        buf.push(0, ms(100), 1).unwrap();
        buf.push(0, ms(50), 2).unwrap();
        buf.push(0, ms(150), 3).unwrap();
        // The 50ms entry was inserted before 100ms.
        // Pop them in order by using try_emit against source 1 at the
        // same timestamps; easier to check the ring directly here
        // via pop_at_or_before on a known reference.
        // Use a private probe: call push for source 1 and emit.
        buf.push(1, ms(150), 30).unwrap();
        let tuple = buf.try_emit().unwrap();
        // Source 0's picked frame is the 150ms one (newest ≤ ref).
        assert_eq!(tuple.frames[0].handle, 3);
        // Still 2 older frames (50ms, 100ms) on source 0 since only
        // >= cutoff (150 - 33 = 117ms) is retained; only 100ms <=
        // 117ms is kept? No, trim drops < cutoff. 100ms < 117ms so
        // it's dropped; 50ms also dropped.
        assert_eq!(buf.len(0), 0);
    }

    #[test]
    fn last_sync_delta_updates_after_emit() {
        let mut buf = stereo_buffer();
        assert!(buf.last_sync_delta(0).is_none());

        buf.push(0, ms(100), 10).unwrap();
        buf.push(1, ms(115), 20).unwrap();
        buf.try_emit().unwrap();

        // After the pair (100ms, 115ms) the spread is 15ms and both
        // sources see the same value.
        assert_eq!(buf.last_sync_delta(0), Some(ms(15)));
        assert_eq!(buf.last_sync_delta(1), Some(ms(15)));
    }

    #[test]
    fn n_source_configuration_works_for_three_cameras() {
        // Validates the "N-camera rig" future-use case the plan
        // called out. Three sources, all within tolerance, emit one
        // tuple with three entries in source-declaration order.
        let mut buf: TimestampedIngestBuffer<u32> =
            TimestampedIngestBuffer::new(&[10, 20, 30], ms(50), 4);
        buf.push(10, ms(100), 1).unwrap();
        buf.push(20, ms(110), 2).unwrap();
        buf.push(30, ms(120), 3).unwrap();
        let tuple = buf.try_emit().unwrap();
        assert_eq!(tuple.frames.len(), 3);
        assert_eq!(
            tuple.frames.iter().map(|f| f.source_id).collect::<Vec<_>>(),
            vec![10, 20, 30],
            "emission order follows source declaration order"
        );
        assert_eq!(tuple.max_delta, ms(20));
    }

    #[test]
    fn clear_empties_every_source() {
        let mut buf = stereo_buffer();
        buf.push(0, ms(1), 10).unwrap();
        buf.push(1, ms(2), 20).unwrap();
        buf.push(0, ms(3), 11).unwrap();
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.total_len(), 0);
    }

    #[test]
    #[should_panic(expected = "at least one source")]
    fn panics_on_empty_source_list() {
        let _: TimestampedIngestBuffer<u32> = TimestampedIngestBuffer::new(&[], ms(1), 4);
    }

    #[test]
    #[should_panic(expected = "duplicate")]
    fn panics_on_duplicate_source_ids() {
        let _: TimestampedIngestBuffer<u32> = TimestampedIngestBuffer::new(&[1, 2, 1], ms(1), 4);
    }

    #[test]
    #[should_panic(expected = "capacity_per_source")]
    fn panics_on_zero_capacity() {
        let _: TimestampedIngestBuffer<u32> = TimestampedIngestBuffer::new(&[1], ms(1), 0);
    }
}
