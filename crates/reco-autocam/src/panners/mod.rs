//! Panner implementations for reco-autocam.
//!
//! Panners are the camera-motion half of the tracker/panner split
//! (see [`reco_core::panner::Panner`]). They consume a
//! [`WorldState`](reco_core::tracker::WorldState) each frame and
//! emit a [`ViewportPosition`](reco_core::director::ViewportPosition)
//! without ever touching raw detections.
//!
//! Layout:
//! - [`field`] - [`FieldPanner`], tracks the densest player cluster
//!   with ball blending and dynamic FOV.
//! - [`smoother`] - [`Smoother`], forward/backward One Euro
//!   trajectory smoothing wrapped around any inner panner.
//! - [`anticipator`] - [`Anticipator`], velocity-based lead.
//! - [`deadzone`] - [`DeadZone`], idle-hold against micro-jitter.
//! - [`sweep`] - [`SweepPanner`], deterministic sinusoidal debug pan
//!   that ignores the world state.
//!
//! Typical composition:
//!
//! ```text
//! FieldPanner -> Smoother -> DeadZone
//! ```

pub mod anticipator;
pub mod deadzone;
pub mod field;
pub mod smoother;
pub mod sweep;

pub use anticipator::Anticipator;
pub use deadzone::DeadZone;
pub use field::{FieldPanner, FieldPannerConfig};
pub use smoother::Smoother;
pub use sweep::SweepPanner;
