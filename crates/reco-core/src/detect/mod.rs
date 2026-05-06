//! Detection, tracking, and camera-control vocabulary.
//!
//! Trait definitions for the detector -> filter -> tracker -> panner -> director
//! chain. Implementations live in reco-detect (detector backends) and reco-autocam
//! (trackers, panners, filters).

pub mod detector;
pub mod director;
pub mod filter;
pub mod panner;
pub mod pipeline_event;
pub mod tracker;
