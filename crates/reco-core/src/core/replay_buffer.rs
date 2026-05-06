//! Bounded-duration ring buffer for recently-rendered panorama frames.
//!
//! Solves FRICTION A16 (OBS replay). Opt-in via
//! [`super::StitchCore::enable_replay_buffer`]; when disabled the core
//! allocates nothing for replay and `submit_frame_*` does zero extra
//! work. Ring trimming runs per-submit and is `O(frames_evicted)`.

use std::collections::VecDeque;
use std::time::Duration;

use super::types::ReplayFrame;

/// Bounded-duration ring of recently-rendered panorama frames.
pub struct ReplayBuffer {
    frames: VecDeque<ReplayFrame>,
    max_duration: Duration,
}

impl ReplayBuffer {
    pub(crate) fn new(max_duration: Duration) -> Self {
        Self {
            frames: VecDeque::new(),
            max_duration,
        }
    }

    pub(crate) fn push(&mut self, frame: ReplayFrame) {
        self.frames.push_back(frame);
        // Evict from the front until the oldest kept frame is within
        // max_duration of the newest. Using wrapping subtraction on
        // Duration is not allowed, so compare directly.
        let cutoff = self
            .frames
            .back()
            .map(|f| f.captured_at)
            .unwrap_or_default()
            .saturating_sub(self.max_duration);
        while let Some(front) = self.frames.front() {
            if front.captured_at < cutoff {
                self.frames.pop_front();
            } else {
                break;
            }
        }
    }

    /// Number of frames currently buffered.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the buffer holds zero frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Maximum age of retained frames (set at enable time).
    pub fn max_duration(&self) -> Duration {
        self.max_duration
    }

    /// Iterate buffered frames oldest-to-newest.
    pub fn iter(&self) -> impl Iterator<Item = &ReplayFrame> {
        self.frames.iter()
    }

    /// Most recently buffered frame, if any.
    pub fn latest(&self) -> Option<&ReplayFrame> {
        self.frames.back()
    }

    /// Oldest buffered frame, if any. Useful for consumers that want
    /// to know the effective buffered duration
    /// (`latest.captured_at - oldest.captured_at`).
    pub fn oldest(&self) -> Option<&ReplayFrame> {
        self.frames.front()
    }

    /// The effective buffered duration: the difference between
    /// oldest and newest frame timestamps. Returns `Duration::ZERO`
    /// for empty or single-frame buffers.
    pub fn buffered_duration(&self) -> Duration {
        match (self.frames.front(), self.frames.back()) {
            (Some(first), Some(last)) => last.captured_at.saturating_sub(first.captured_at),
            _ => Duration::ZERO,
        }
    }

    /// Drop every buffered frame without changing `max_duration`.
    /// Consumers wire this to a "Clear replay" UI button so the user
    /// can start a fresh replay window after an event.
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Clone every buffered frame into an owned vector.
    ///
    /// Used by consumers that want to ship the replay off the render
    /// thread (to disk, to a "Save replay" dialog, to a network
    /// stream). The buffer itself keeps the frames, so the consumer
    /// can keep recording while it exports a snapshot.
    ///
    /// Returns the vector in oldest-to-newest order, matching
    /// [`Self::iter`].
    pub fn snapshot(&self) -> Vec<ReplayFrame> {
        self.frames.iter().cloned().collect()
    }

    /// Drain every buffered frame into an owned vector, leaving the
    /// buffer empty. Same ordering contract as [`Self::snapshot`].
    /// Unlike `snapshot`, this transfers ownership - no clone cost
    /// for consumers that are about to discard the buffer anyway
    /// (e.g. a "Save + reset" UI flow).
    pub fn take(&mut self) -> Vec<ReplayFrame> {
        self.frames.drain(..).collect()
    }
}
