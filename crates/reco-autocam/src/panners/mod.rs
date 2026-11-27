//! Panner implementations for reco-autocam.
//!
//! Panners consume a [`WorldState`](reco_core::detect::tracker::WorldState)
//! each frame and emit a
//! [`WorldPose`](reco_core::geometry::WorldPose).
//!
//! - [`field`] - [`FieldPanner`], trimmed-robust player cluster with ball blend and dynamic FOV.
//! - [`file_panner`] - [`FilePanner`], replays precomputed trajectory from CSV.
//! - [`sweep`] - [`SweepPanner`], deterministic debug pan.
//!
//! The session supplies future world states within the configured lookahead.

pub mod field;
pub mod file_panner;
pub mod sweep;

pub use field::{ClusterMode, FieldPanner, FieldPannerConfig, FramingMode, PRESET_NAMES};
pub use file_panner::FilePanner;
pub use sweep::SweepPanner;
