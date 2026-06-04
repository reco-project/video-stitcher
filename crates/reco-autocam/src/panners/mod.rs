//! Panner implementations for reco-autocam.
//!
//! Panners consume a [`WorldState`](reco_core::detect::tracker::WorldState)
//! each frame and emit a
//! [`ViewportPosition`](reco_core::detect::director::ViewportPosition).
//!
//! - [`field`] - [`FieldPanner`], trimmed-robust player cluster with ball blend and dynamic FOV.
//! - [`file_panner`] - [`FilePanner`], replays precomputed trajectory from CSV.
//! - [`sweep`] - [`SweepPanner`], deterministic debug pan.
//!
//! Lookahead is not a panner type: it is a loop concern (the buffered
//! run loop centered-smooths the panner's pose stream over past + future
//! frames). FieldPanner runs the same whether the buffer is on or off.

pub mod field;
pub mod file_panner;
pub mod sweep;

pub use field::{ClusterMode, FieldPanner, FieldPannerConfig, FramingMode, PRESET_NAMES};
pub use file_panner::FilePanner;
pub use sweep::SweepPanner;
