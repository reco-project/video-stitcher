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
#[allow(dead_code)]
pub(crate) struct BufferedFrame {
    pub frame: StereoFrame,
    pub world_state: WorldState,
    pub detections: Vec<MappedDetection>,
    pub frame_index: u64,
    pub elapsed_ms: f64,
    pub decode_time: std::time::Duration,
    /// VRAM pool slot index (Some when GPU-resident, None for CPU frames).
    /// The pool slot is released after rendering.
    pub vram_slot: Option<usize>,
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

#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::tracker::{TrackState, TrackedEntity, WorldState};

    fn make_frame(index: u64, ball_yaw: f32) -> BufferedFrame {
        BufferedFrame {
            frame: StereoFrame::Nv12(crate::source::Nv12FramePair {
                left: crate::source::Nv12Data {
                    y: vec![128; 4],
                    uv: vec![128; 2],
                },
                right: crate::source::Nv12Data {
                    y: vec![128; 4],
                    uv: vec![128; 2],
                },
            }),
            world_state: WorldState {
                ball: Some(TrackedEntity {
                    id: 0,
                    class_id: 0,
                    yaw: ball_yaw,
                    pitch: 0.1,
                    confidence: 0.9,
                    state: TrackState::Tracking,
                    age_frames: 1,
                    origin: crate::detect::detector::CameraId::Left,
                }),
                players: vec![],
            },
            detections: vec![],
            frame_index: index,
            elapsed_ms: index as f64 * 33.3,
            decode_time: std::time::Duration::from_millis(3),
            vram_slot: None,
        }
    }

    #[test]
    fn push_pop_fifo_order() {
        let mut buf = FrameBuffer::new(3);
        buf.push(make_frame(0, 0.1));
        buf.push(make_frame(1, 0.2));
        buf.push(make_frame(2, 0.3));

        let f0 = buf.pop().unwrap();
        assert_eq!(f0.frame_index, 0);
        let f1 = buf.pop().unwrap();
        assert_eq!(f1.frame_index, 1);
        let f2 = buf.pop().unwrap();
        assert_eq!(f2.frame_index, 2);
        assert!(buf.pop().is_none());
    }

    #[test]
    fn capacity_and_fullness() {
        let mut buf = FrameBuffer::new(2);
        assert_eq!(buf.capacity(), 2);
        assert!(buf.is_empty());
        assert!(!buf.is_full());

        buf.push(make_frame(0, 0.0));
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_full());

        buf.push(make_frame(1, 0.0));
        assert_eq!(buf.len(), 2);
        assert!(buf.is_full());
    }

    #[test]
    fn future_world_states_returns_remaining() {
        let mut buf = FrameBuffer::new(5);
        for i in 0..4 {
            buf.push(make_frame(i, i as f32 * 0.1));
        }
        buf.pop(); // remove frame 0

        let futures = buf.future_world_states();
        assert_eq!(futures.len(), 3);
        let yaws: Vec<f32> = futures
            .iter()
            .map(|ws| ws.ball.as_ref().unwrap().yaw)
            .collect();
        assert!((yaws[0] - 0.1).abs() < 1e-6);
        assert!((yaws[1] - 0.2).abs() < 1e-6);
        assert!((yaws[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn drain_empties_buffer() {
        let mut buf = FrameBuffer::new(3);
        buf.push(make_frame(0, 0.0));
        buf.push(make_frame(1, 0.0));
        let drained: Vec<_> = buf.drain().collect();
        assert_eq!(drained.len(), 2);
        assert!(buf.is_empty());
    }

    #[test]
    #[should_panic(expected = "push called on full buffer")]
    fn push_on_full_panics() {
        let mut buf = FrameBuffer::new(1);
        buf.push(make_frame(0, 0.0));
        buf.push(make_frame(1, 0.0));
    }
}
