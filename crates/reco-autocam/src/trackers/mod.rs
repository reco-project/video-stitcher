//! Tracker implementations for reco-autocam.
//!
//! Each tracker implements the
//! [`Tracker`](reco_core::detect::tracker::Tracker) contract, turning a
//! per-frame stream of [`MappedDetection`](reco_core::detect::director::MappedDetection)s
//! into a [`TrackedEntity`](reco_core::detect::tracker::TrackedEntity)
//! list a panner can consume.
//!
//! Layout:
//! - [`filters`] — shared filter building blocks (the coaster).
//!   Self-contained and independently testable; used by the ball
//!   tracker.
//! - [`ball`] — [`BallTracker`], the singleton ball tracker:
//!   player-anchor → nearest-to-last with cross-cam handoff → coast.
//! - [`class_provider`] — [`ClassProvider`], a stateless per-class
//!   projector (no identity, no coast: a point-cloud panner only needs
//!   this frame's points). The [`Tracker`](reco_core::detect::tracker::Tracker)
//!   trait's two real shapes are this and the stateful ball tracker.

pub mod ball;
pub mod class_provider;
pub mod filters;

pub use ball::BallTracker;
pub use class_provider::ClassProvider;
