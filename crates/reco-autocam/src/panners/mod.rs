//! Panner implementations for reco-autocam.
//!
//! Panners consume a [`WorldState`](reco_core::detect::tracker::WorldState)
//! each frame and emit a
//! [`ViewportPosition`](reco_core::detect::director::ViewportPosition).
//!
//! - [`field`] - [`FieldPanner`], Huber-robust player cluster with ball blend and dynamic FOV.
//! - [`lookahead`] - [`LookaheadPanner`], future-aware panner (pre-smooth -> blend -> EMA).
//! - [`file_panner`] - [`FilePanner`], replays precomputed trajectory from CSV.
//! - [`sweep`] - [`SweepPanner`], deterministic debug pan.

pub mod field;
pub mod file_panner;
pub mod lookahead;
pub mod sweep;

pub use field::{FieldPanner, FieldPannerConfig};
pub use file_panner::FilePanner;
pub use lookahead::{LookaheadPanner, LookaheadPannerConfig};
pub use sweep::SweepPanner;
