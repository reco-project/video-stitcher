//! Director implementations for automatic camera control.
//!
//! Each director implements the [`Director`](reco_core::director::Director) trait
//! and decides where the virtual camera points based on object detections.
//!
//! - [`BallDirector`] - tracks a single ball with plausibility rejection
//! - [`FieldDirector`] - tracks ball + player cluster for robust football coverage
//!
//! Use [`TrackingMode`] to select which director to create in [`setup_autocam`](crate::setup_autocam).

mod anticipation;
mod ball;
mod deadzone;
mod field;
mod sweep;
pub(crate) mod util;

pub use anticipation::AnticipatingDirector;
pub use ball::BallDirector;
pub use deadzone::DeadZoneDirector;
pub use field::FieldDirector;
pub use sweep::SweepDirector;

/// Which tracking strategy to use for automatic camera control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrackingMode {
    /// Track a single ball (single-class model). Uses [`BallDirector`].
    Ball,
    /// Track ball + players together (multi-class model). Uses [`FieldDirector`].
    /// Players anchor the camera to the action zone; the ball provides fine
    /// positioning within that zone. False ball detections far from players
    /// are automatically rejected.
    Field,
    /// Debug mode: slowly pan left-right across the full coverage.
    /// No AI, no tracking. Uses [`SweepDirector`](crate::SweepDirector).
    Sweep,
}
