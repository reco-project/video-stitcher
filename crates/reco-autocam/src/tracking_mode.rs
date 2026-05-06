//! [`TrackingMode`] enum - the one-knob config that selects which
//! tracker(s) and panner setup_autocam wires up.

/// Which automatic-camera strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TrackingMode {
    /// Track player cluster + ball for broadcast-style coverage
    /// (multi-class model). Uses
    /// [`PlayerTracker`](crate::trackers::PlayerTracker),
    /// [`BallTracker`](crate::trackers::BallTracker), and
    /// [`FieldPanner`](crate::panners::FieldPanner).
    /// Ball influence is controlled by `FieldPannerConfig::ball_weight`.
    #[default]
    Field,
    /// Debug mode: slowly pan left-right across the full coverage.
    /// No AI, no tracking. Uses
    /// [`SweepPanner`](crate::panners::SweepPanner).
    Sweep,
}
