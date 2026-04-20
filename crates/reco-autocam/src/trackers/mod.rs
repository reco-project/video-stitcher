//! Tracker implementations for reco-autocam.
//!
//! Ports of the Python offline POC
//! (`/tmp/reco-ai-eval/build_tracker_video.py`) into the
//! `reco_core::tracker::Tracker` contract. Each tracker turns a
//! noisy stream of [`MappedDetection`](reco_core::director::MappedDetection)s
//! into a clean [`TrackedEntity`](reco_core::tracker::TrackedEntity)
//! stream with stable identities and a tri-valued lifecycle state.
//!
//! Layout:
//! - [`filters`] — shared filter building blocks (flicker, coaster).
//!   Each filter is self-contained and independently testable.
//! - [`ball`] — [`BallTracker`], the singleton ball tracker that
//!   composes the filters in POC order: flicker → player-anchor →
//!   nearest-to-last with cross-cam handoff → coast.
//!
//! Upcoming (Phase 5, see `~/.claude/plans/zesty-mixing-firefly.md`):
//! `player` and `ensemble` — multi-entity tracking with
//! Hungarian-matched stable IDs.

pub mod ball;
pub mod filters;

pub use ball::BallTracker;
