//! Panner implementations for reco-autocam.
//!
//! Panners are the camera-motion half of the tracker/panner split
//! (see [`reco_core::panner::Panner`]). They consume a
//! [`WorldState`](reco_core::tracker::WorldState) each frame and
//! emit a [`ViewportPosition`](reco_core::director::ViewportPosition)
//! — without ever touching raw detections.
//!
//! Layout:
//! - [`ball`] — [`BallPanner`], follows `world.ball`, dynamic FOV.
//! - [`smoother`] — [`Smoother`], forward/backward One Euro
//!   trajectory smoothing wrapped around any inner [`Panner`](reco_core::panner::Panner).
//! - [`anticipator`] — [`Anticipator`], velocity-based lead.
//! - [`deadzone`] — [`DeadZone`], idle-hold against micro-jitter.
//!
//! Typical composition (matching the old director-side chain):
//!
//! ```text
//! BallPanner → Smoother → Anticipator → DeadZone
//! ```
//!
//! Upcoming (Phase 6 of the tracker/panner migration):
//! `field` — `FieldPanner`, blends ball with the player cluster
//! centroid.

pub mod anticipator;
pub mod ball;
pub mod deadzone;
pub mod smoother;

pub use anticipator::Anticipator;
pub use ball::BallPanner;
pub use deadzone::DeadZone;
pub use smoother::Smoother;
