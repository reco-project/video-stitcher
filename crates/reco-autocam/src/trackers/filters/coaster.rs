//! Fixed-budget coaster for tracker lifecycle transitions.
//!
//! When a tracker has no fresh accepted detection on a frame, it
//! typically wants to **hold the last known position** for a short
//! grace period — long enough to bridge brief occlusions and detector
//! flickers, but short enough that true track-losses are recognized
//! promptly.
//!
//! This module encapsulates that countdown into a small reusable
//! helper independent of the tracker type. It does NOT hold any
//! position data itself; the tracker owns the `LastKnown` state and
//! asks the coaster whether it's allowed to keep coasting.
//!
//! # Lifecycle
//!
//! ```text
//!     ┌─ accept_fresh() ─┐
//!     │                  │
//!     ▼                  │
//! Tracking ◀─── accept_fresh() (reset countdown)
//!     │
//!     │ no fresh detection → step_without_fresh()
//!     ▼
//! Coasting ── N frames without fresh → step_without_fresh() returns Lost
//!     │
//!     ▼
//!   Lost  ── requires accept_fresh() to re-acquire
//! ```
//!
//! The coaster does not distinguish "lost and may come back" from
//! "permanently gone" — the tracker decides based on whether a
//! subsequent [`accept_fresh`](Coaster::accept_fresh) arrives.

/// Current lifecycle state driven by the [`Coaster`].
///
/// Maps 1:1 to [`reco_core::tracker::TrackState`] — the tracker
/// translates during its update loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoastStatus {
    /// A fresh measurement was accepted this frame; countdown reset.
    Tracking,
    /// No fresh measurement this frame, but the last-known position
    /// is still within the coast budget. The tracker should emit
    /// the held position with [`TrackState::Coasting`].
    ///
    /// [`TrackState::Coasting`]: reco_core::tracker::TrackState::Coasting
    Coasting,
    /// Budget exhausted; the track is declared lost. The tracker
    /// should emit [`TrackState::Lost`] and reset its internal
    /// last-known state on the next frame.
    ///
    /// [`TrackState::Lost`]: reco_core::tracker::TrackState::Lost
    Lost,
}

/// A small frame-countdown that tracks time-since-fresh-detection.
///
/// Construct with a `max_coast_frames` budget, then call exactly one
/// of [`accept_fresh`](Self::accept_fresh) or
/// [`step_without_fresh`](Self::step_without_fresh) per frame.
#[derive(Debug)]
pub struct Coaster {
    max_coast_frames: u32,
    frames_coasting: u32,
    /// Tracks whether we've ever seen a fresh detection. Coasting
    /// before first fresh is nonsensical; always return `Lost`.
    ever_tracked: bool,
}

impl Coaster {
    /// Build a new coaster with a given coast budget in frames.
    ///
    /// A budget of `0` means "no coasting allowed" — the next
    /// non-fresh call immediately returns [`CoastStatus::Lost`].
    pub fn new(max_coast_frames: u32) -> Self {
        Self {
            max_coast_frames,
            frames_coasting: 0,
            ever_tracked: false,
        }
    }

    /// Record a fresh accepted measurement this frame. Returns
    /// [`CoastStatus::Tracking`] and resets the coast countdown.
    pub fn accept_fresh(&mut self) -> CoastStatus {
        self.frames_coasting = 0;
        self.ever_tracked = true;
        CoastStatus::Tracking
    }

    /// No fresh accepted measurement this frame. Returns:
    /// - [`CoastStatus::Coasting`] if we're still within the budget
    ///   AND have ever tracked something,
    /// - [`CoastStatus::Lost`] otherwise.
    pub fn step_without_fresh(&mut self) -> CoastStatus {
        if !self.ever_tracked {
            return CoastStatus::Lost;
        }
        if self.frames_coasting < self.max_coast_frames {
            self.frames_coasting += 1;
            CoastStatus::Coasting
        } else {
            // Stay Lost; tracker must re-acquire via `accept_fresh`.
            self.frames_coasting = self.max_coast_frames;
            self.ever_tracked = false;
            CoastStatus::Lost
        }
    }

    /// Current countdown (frames spent coasting since last fresh).
    /// Returns 0 while tracking.
    pub fn frames_coasting(&self) -> u32 {
        self.frames_coasting
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_without_fresh_is_lost() {
        let mut c = Coaster::new(5);
        assert_eq!(c.step_without_fresh(), CoastStatus::Lost);
    }

    #[test]
    fn fresh_then_coast_then_lost() {
        let mut c = Coaster::new(3);
        assert_eq!(c.accept_fresh(), CoastStatus::Tracking);
        assert_eq!(c.step_without_fresh(), CoastStatus::Coasting);
        assert_eq!(c.step_without_fresh(), CoastStatus::Coasting);
        assert_eq!(c.step_without_fresh(), CoastStatus::Coasting);
        assert_eq!(c.step_without_fresh(), CoastStatus::Lost);
    }

    #[test]
    fn zero_budget_is_never_coasting() {
        let mut c = Coaster::new(0);
        c.accept_fresh();
        assert_eq!(c.step_without_fresh(), CoastStatus::Lost);
    }

    #[test]
    fn fresh_resets_countdown() {
        let mut c = Coaster::new(2);
        c.accept_fresh();
        c.step_without_fresh();
        c.step_without_fresh();
        // Would be Lost next — but fresh arrives, resetting.
        assert_eq!(c.accept_fresh(), CoastStatus::Tracking);
        assert_eq!(c.step_without_fresh(), CoastStatus::Coasting);
    }

    #[test]
    fn frames_coasting_counter() {
        let mut c = Coaster::new(10);
        c.accept_fresh();
        assert_eq!(c.frames_coasting(), 0);
        c.step_without_fresh();
        assert_eq!(c.frames_coasting(), 1);
        c.step_without_fresh();
        assert_eq!(c.frames_coasting(), 2);
        c.accept_fresh();
        assert_eq!(c.frames_coasting(), 0);
    }

    #[test]
    fn after_lost_fresh_re_acquires() {
        let mut c = Coaster::new(1);
        c.accept_fresh();
        c.step_without_fresh();
        assert_eq!(c.step_without_fresh(), CoastStatus::Lost);
        assert_eq!(c.accept_fresh(), CoastStatus::Tracking);
    }
}
