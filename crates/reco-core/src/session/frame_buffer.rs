#![allow(dead_code)] // Wired in the next step (run_buffered).
//! Lookahead frame buffer for temporal-aware processing.
//!
//! Holds N decoded frames with their detection metadata so the
//! panner can see future WorldStates when deciding the current
//! frame's viewport position.
//!
//! Phase 1: CPU-resident frames only (`StereoFrame::Yuv420p` /
//! `StereoFrame::Nv12`). GPU-resident frames require expanding the
//! decode slot pool (Phase 3).

use std::collections::VecDeque;

use crate::detect::director::MappedDetection;
use crate::detect::tracker::WorldState;
use crate::source::StereoFrame;

/// A single buffered frame: decoded pixels + detection metadata.
pub(crate) struct BufferedFrame {
    pub frame: StereoFrame,
    pub world_state: WorldState,
    pub detections: Vec<MappedDetection>,
    pub frame_index: u64,
    pub elapsed_ms: f64,
    pub decode_time: std::time::Duration,
}

/// Fixed-capacity ring buffer of decoded frames.
///
/// The producer (decode + detect) pushes frames in. The consumer
/// (direct + render) pops from the front with access to all
/// remaining entries as the lookahead window.
pub(crate) struct FrameBuffer {
    frames: VecDeque<BufferedFrame>,
    capacity: usize,
}

impl FrameBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_full(&self) -> bool {
        self.frames.len() >= self.capacity
    }

    /// Push a new frame into the buffer. Panics if full.
    pub fn push(&mut self, frame: BufferedFrame) {
        debug_assert!(
            !self.is_full(),
            "FrameBuffer::push called on full buffer (cap={})",
            self.capacity
        );
        self.frames.push_back(frame);
    }

    /// Pop the oldest frame for rendering.
    pub fn pop(&mut self) -> Option<BufferedFrame> {
        self.frames.pop_front()
    }

    /// Collect future WorldStates from all frames currently in the
    /// buffer, ordered nearest-to-farthest. Used as the lookahead
    /// window for `Panner::decide_with_lookahead`.
    pub fn future_world_states(&self) -> Vec<WorldState> {
        self.frames.iter().map(|f| f.world_state.clone()).collect()
    }

    /// Drain all remaining frames (for the drain phase at EOF).
    pub fn drain(&mut self) -> impl Iterator<Item = BufferedFrame> + '_ {
        self.frames.drain(..)
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}
