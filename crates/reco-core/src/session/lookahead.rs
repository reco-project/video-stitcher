//! Lookahead buffer for director trajectory planning.
//!
//! Stores per-frame detection results and raw director positions in a
//! ring buffer. The session uses this to decouple "when detection runs"
//! from "when rendering happens," enabling the director to see N frames
//! into the future before a frame is rendered.
//!
//! The buffer stores only detection metadata (a few bytes per frame),
//! not decoded video frames. Frames stay in their native storage (GPU
//! surfaces for zero-copy, `VecDeque<StereoFrame>` for CPU).

use std::collections::VecDeque;

use crate::director::{MappedDetection, ViewportPosition};

/// Configuration for the unified lookahead system.
///
/// Controls how many frames the director sees ahead of the renderer
/// and whether bidirectional trajectory smoothing is applied.
#[derive(Debug, Clone)]
pub struct LookaheadConfig {
    /// Number of frames to look ahead (0 = disabled).
    pub frames: usize,
    /// Enable bidirectional trajectory smoothing over the lookahead window.
    ///
    /// When true, a forward-backward One Euro filter smooths the director's
    /// raw trajectory, producing zero-phase-lag camera movements.
    /// When false, the raw director position is used (same as the old
    /// lookahead behavior where the director's EMA state is simply
    /// "warmed up" by future frames).
    pub smooth: bool,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self {
            frames: 0,
            smooth: true,
        }
    }
}

/// A single frame's detection results and raw director output.
///
/// Stored in the [`LookaheadBuffer`]; consumed by the trajectory
/// smoother and renderer.
pub(crate) struct TrajectoryEntry {
    /// Mapped detections for this frame (used by GPU zero-copy paths in Phase 2).
    #[allow(dead_code)]
    pub detections: Vec<MappedDetection>,
    /// Raw director position after `update()` for this frame.
    /// This is the unsmoothed trajectory point.
    pub raw_position: ViewportPosition,
}

/// Ring buffer of trajectory entries for lookahead.
///
/// Head = oldest (ready to render), tail = newest (just detected).
/// The buffer is pre-filled during the lookahead warmup phase, then
/// one entry is pushed and one popped per frame in steady state.
pub(crate) struct LookaheadBuffer {
    entries: VecDeque<TrajectoryEntry>,
    #[allow(dead_code)]
    capacity: usize,
}

impl LookaheadBuffer {
    /// Create a new buffer with the given lookahead capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity + 1),
            capacity,
        }
    }

    /// Push a new trajectory entry to the back of the buffer.
    pub fn push(&mut self, entry: TrajectoryEntry) {
        self.entries.push_back(entry);
    }

    /// Pop the oldest entry (the frame about to be rendered).
    pub fn pop_front(&mut self) -> Option<TrajectoryEntry> {
        self.entries.pop_front()
    }

    /// Number of entries currently in the buffer.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the buffer has reached its target lookahead depth.
    #[allow(dead_code)]
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    /// Iterator over raw positions in the buffer (oldest to newest).
    ///
    /// Used by the trajectory smoother to compute a smoothed position
    /// for the oldest entry.
    pub fn positions(&self) -> impl Iterator<Item = &ViewportPosition> {
        self.entries.iter().map(|e| &e.raw_position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::director::ViewportPosition;

    #[test]
    fn buffer_push_pop_ordering() {
        let mut buf = LookaheadBuffer::new(3);
        for i in 0..3 {
            buf.push(TrajectoryEntry {
                detections: vec![],
                raw_position: ViewportPosition {
                    yaw: i as f32 * 0.1,
                    pitch: 0.0,
                    fov_degrees: None,
                },
            });
        }
        assert!(buf.is_full());
        assert_eq!(buf.len(), 3);

        let e = buf.pop_front().unwrap();
        assert!((e.raw_position.yaw - 0.0).abs() < f32::EPSILON);
        assert_eq!(buf.len(), 2);

        let e = buf.pop_front().unwrap();
        assert!((e.raw_position.yaw - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn positions_iterator() {
        let mut buf = LookaheadBuffer::new(3);
        for i in 0..3 {
            buf.push(TrajectoryEntry {
                detections: vec![],
                raw_position: ViewportPosition {
                    yaw: i as f32 * 0.1,
                    pitch: 0.0,
                    fov_degrees: None,
                },
            });
        }
        let yaws: Vec<f32> = buf.positions().map(|p| p.yaw).collect();
        assert_eq!(yaws.len(), 3);
        assert!((yaws[0] - 0.0).abs() < f32::EPSILON);
        assert!((yaws[1] - 0.1).abs() < f32::EPSILON);
        assert!((yaws[2] - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn empty_buffer_pop_returns_none() {
        let mut buf = LookaheadBuffer::new(5);
        assert!(buf.pop_front().is_none());
        assert_eq!(buf.len(), 0);
        assert!(!buf.is_full());
    }
}
