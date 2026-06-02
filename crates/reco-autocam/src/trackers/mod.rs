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
//! - [`player`] — [`PlayerTracker`], a stateless live-players provider
//!   (no identity, no coast: the panner only needs this frame's points).

pub mod ball;
pub mod filters;
pub mod player;

pub use ball::BallTracker;
pub use player::PlayerTracker;
