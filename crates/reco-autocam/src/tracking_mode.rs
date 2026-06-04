//! [`TrackingMode`] enum - the one-knob config that selects which
//! tracker(s) and panner setup_autocam wires up.

/// Which automatic-camera strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TrackingMode {
    /// Player cluster + ball for broadcast-style coverage. Uses a
    /// [`ClassProvider`](crate::trackers::ClassProvider) for players, a
    /// [`BallTracker`](crate::trackers::BallTracker), and a
    /// [`FieldPanner`](crate::panners::FieldPanner). The player provider
    /// is attached only when the model names a player class, so this mode
    /// degrades gracefully to ball-follow on a ball-only model. Ball
    /// influence is controlled by `FieldPannerConfig::ball_weight`.
    #[default]
    Field,
    /// Ball-only tracking for single-class ball detectors. No player
    /// tracker, no cluster centroid. Uses only
    /// [`BallTracker`](crate::trackers::BallTracker) with higher
    /// confidence threshold and top-1 detection per camera.
    Ball,
    /// Debug mode: slowly pan left-right across the full coverage.
    /// No AI, no tracking. Uses
    /// [`SweepPanner`](crate::panners::SweepPanner).
    Sweep,
}
