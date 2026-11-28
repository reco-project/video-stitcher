//! Lookahead frame buffer for temporal-aware processing.
//!
//! Holds N decoded frames with their detection metadata so the
//! panner can see future WorldStates when deciding the current
//! frame's viewport position. Shared host images and retained Metal buffers
//! stay with the frame; borrowed native images use a persistent staging slot.

use std::collections::VecDeque;

use crate::detect::mapped_detection::MappedDetection;
use crate::detect::tracker::WorldState;
use crate::source::FrameSet;

#[cfg(all(feature = "gpu", target_os = "linux"))]
pub(super) fn release_staged_native(
    frame: Option<FrameSet>,
    slot: Option<usize>,
) -> Option<FrameSet> {
    if slot.is_some()
        && matches!(
            frame,
            Some(FrameSet::NvmmResident(_) | FrameSet::GpuResident(_))
        )
    {
        None
    } else {
        frame
    }
}

/// A single buffered frame: decoded pixels + detection metadata.
pub(crate) struct BufferedFrame {
    /// None when native input has been replaced by its staged copy.
    pub frame: Option<FrameSet>,
    pub pts_us: Option<i64>,
    pub world_state: WorldState,
    pub detections: Vec<MappedDetection>,
    pub elapsed_ms: f64,
    pub timing: crate::telemetry::FrameTiming,
    /// VRAM pool slot index (Some when GPU-resident, None for CPU frames).
    /// The pool slot is released after rendering.
    #[cfg(feature = "gpu")]
    pub vram_slot: Option<usize>,
    pub retained_bytes: usize,
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
            frames: VecDeque::new(),
            capacity,
        }
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

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::tracker::{TrackState, TrackedEntity, WorldState};

    #[cfg(all(feature = "gpu", target_os = "linux"))]
    #[test]
    fn nvmm_lookahead_releases_native_owner_after_staging() {
        use crate::source::NvmmFrame;
        use std::sync::Arc;
        let mut lookahead = FrameBuffer::new(50);
        for slot in 0..50 {
            // A one-buffer capture pool could not produce another frame if
            // the previous capture owner remained in the lookahead queue.
            let owner = Arc::new(());
            let weak = Arc::downgrade(&owner);
            let frame = NvmmFrame::lifetime_test_frame(owner);
            let mut buffered = make_frame(0.0);
            buffered.frame = release_staged_native(
                Some(FrameSet::NvmmResident([frame.clone(), frame])),
                Some(slot),
            );
            buffered.vram_slot = Some(slot);
            lookahead.push(buffered);
            assert!(weak.upgrade().is_none());
        }
        assert!(lookahead.is_full());
        while let Some(frame) = lookahead.pop() {
            assert!(frame.frame.is_none());
            assert!(frame.vram_slot.is_some());
        }
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let frame = NvmmFrame::lifetime_test_frame(owner);
        let unstaged =
            release_staged_native(Some(FrameSet::NvmmResident([frame.clone(), frame])), None);
        assert!(weak.upgrade().is_some());
        drop(unstaged);
        assert!(weak.upgrade().is_none());
    }

    fn make_frame(ball_yaw: f32) -> BufferedFrame {
        BufferedFrame {
            frame: Some(FrameSet::Nv12(vec![
                crate::source::Nv12Data {
                    size: (2, 2),
                    y: vec![128; 4],
                    uv: vec![128; 2],
                }
                .into(),
                crate::source::Nv12Data {
                    size: (2, 2),
                    y: vec![128; 4],
                    uv: vec![128; 2],
                }
                .into(),
            ])),
            world_state: WorldState {
                ball: Some(TrackedEntity {
                    id: 0,
                    class_id: 0,
                    yaw: ball_yaw,
                    pitch: 0.1,
                    confidence: 0.9,
                    state: TrackState::Tracking,
                    age_frames: 1,
                    origin: 0,
                }),
                players: vec![],
            },
            detections: vec![],
            elapsed_ms: 0.0,
            pts_us: None,
            timing: crate::telemetry::FrameTiming {
                decode: Some(std::time::Duration::from_millis(3)),
                ..Default::default()
            },
            #[cfg(feature = "gpu")]
            vram_slot: None,
            retained_bytes: 0,
        }
    }

    fn ball_yaw(f: &BufferedFrame) -> f32 {
        f.world_state.ball.as_ref().unwrap().yaw
    }

    #[test]
    fn push_pop_fifo_order() {
        let mut buf = FrameBuffer::new(3);
        buf.push(make_frame(0.1));
        buf.push(make_frame(0.2));
        buf.push(make_frame(0.3));

        assert!((ball_yaw(&buf.pop().unwrap()) - 0.1).abs() < 1e-6);
        assert!((ball_yaw(&buf.pop().unwrap()) - 0.2).abs() < 1e-6);
        assert!((ball_yaw(&buf.pop().unwrap()) - 0.3).abs() < 1e-6);
        assert!(buf.pop().is_none());
    }

    #[test]
    fn fullness_tracking() {
        let mut buf = FrameBuffer::new(2);
        assert!(buf.is_empty());
        assert!(!buf.is_full());

        buf.push(make_frame(0.0));
        assert!(!buf.is_full());

        buf.push(make_frame(0.0));
        assert!(buf.is_full());
    }

    #[test]
    fn future_world_states_returns_remaining() {
        let mut buf = FrameBuffer::new(5);
        for i in 0..4 {
            buf.push(make_frame(i as f32 * 0.1));
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

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "push called on full buffer")]
    fn push_on_full_panics() {
        let mut buf = FrameBuffer::new(1);
        buf.push(make_frame(0.0));
        buf.push(make_frame(0.0));
    }
}
