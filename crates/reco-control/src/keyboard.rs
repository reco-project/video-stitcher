//! Keyboard transport.
//!
//! The trivial transport: consumers push [`ControlIntent`] values
//! onto an internal ring as they decode their native keyboard events
//! (Slint key handler, OBS hotkey API, terminal `termion`, etc.),
//! and [`KeyboardTransport::poll`] drains them into the caller's
//! buffer.
//!
//! This transport exists mostly to prove that the abstraction is
//! usable — the real payoff is uniform dispatch at the consumer
//! boundary: the same `ControlIntent` stream drives reco-obs,
//! reco-gui, and future remote clients without per-consumer event
//! translation.

use std::collections::VecDeque;

use crate::{ControlIntent, ControlTransport};

/// Keyboard-driven control transport.
///
/// Consumers [`Self::push`] intents as their key handlers fire,
/// then call [`Self::poll`] once per frame / tick to drain the
/// queue.
pub struct KeyboardTransport {
    queue: VecDeque<ControlIntent>,
}

impl Default for KeyboardTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyboardTransport {
    /// Create an empty transport.
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    /// Push an intent onto the queue. Typically called from a
    /// platform-specific key-event handler that's already mapped
    /// the key code to a [`ControlIntent`].
    pub fn push(&mut self, intent: ControlIntent) {
        self.queue.push_back(intent);
    }

    /// Number of queued intents. Useful for UI status ("3 pending
    /// hotkey events") or for tests that want to assert nothing
    /// leaked through.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }
}

impl ControlTransport for KeyboardTransport {
    fn name(&self) -> &'static str {
        "keyboard"
    }

    fn poll(&mut self, out: &mut Vec<ControlIntent>) -> usize {
        let start = out.len();
        out.extend(self.queue.drain(..));
        out.len() - start
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::pose_control::HotkeyIntent;

    #[test]
    fn name_is_stable() {
        let t = KeyboardTransport::new();
        assert_eq!(t.name(), "keyboard");
    }

    #[test]
    fn push_then_poll_drains_in_order() {
        let mut t = KeyboardTransport::new();
        t.push(ControlIntent::Hotkey(HotkeyIntent::ZoomIn));
        t.push(ControlIntent::Capture(crate::CaptureIntent::Snapshot));
        assert_eq!(t.pending(), 2);

        let mut buf = Vec::new();
        let n = t.poll(&mut buf);
        assert_eq!(n, 2);
        assert_eq!(buf.len(), 2);
        assert_eq!(t.pending(), 0, "poll drains the queue");

        matches!(buf[0], ControlIntent::Hotkey(HotkeyIntent::ZoomIn));
        matches!(
            buf[1],
            ControlIntent::Capture(crate::CaptureIntent::Snapshot)
        );
    }

    #[test]
    fn poll_on_empty_returns_zero() {
        let mut t = KeyboardTransport::new();
        let mut buf = Vec::new();
        assert_eq!(t.poll(&mut buf), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn poll_appends_does_not_clear() {
        let mut t = KeyboardTransport::new();
        t.push(ControlIntent::Hotkey(HotkeyIntent::Reset));
        let mut buf = vec![ControlIntent::Hotkey(HotkeyIntent::YawLeft)];
        t.poll(&mut buf);
        assert_eq!(buf.len(), 2, "poll appends to the caller's buffer");
    }
}
