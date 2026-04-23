//! [`TrackingMode`] enum — the one-knob config that selects which
//! tracker(s) and panner setup_autocam wires up.

/// Which automatic-camera strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrackingMode {
    /// Track a single ball (single-class model).
    /// Uses [`BallTracker`](crate::trackers::BallTracker) and
    /// [`BallPanner`](crate::panners::BallPanner).
    Ball,
    /// Track player cluster for broadcast-style coverage
    /// (multi-class model). Uses
    /// [`PlayerTracker`](crate::trackers::PlayerTracker) and
    /// [`FieldPanner`](crate::panners::FieldPanner). Ball-blend is
    /// opt-in per panner configuration.
    Field,
    /// Debug mode: slowly pan left-right across the full coverage.
    /// No AI, no tracking. Uses
    /// [`SweepPanner`](crate::panners::SweepPanner).
    Sweep,
}
