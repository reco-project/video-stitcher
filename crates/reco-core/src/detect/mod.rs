//! Detection, tracking, and camera-control vocabulary.
//!
//! Trait definitions for the detector -> tracker -> panner -> director
//! chain. Implementations live in reco-detect (detector backends) and reco-autocam
//! (trackers, panners).

pub mod detector;
pub mod director;
pub mod panner;
pub mod pipeline_event;
pub mod tracker;

/// Shared interface for types that accept detection/tracking/panning
/// configuration. Implemented by both [`StitchCore`](crate::core::StitchCore)
/// and [`StitchSession`](crate::session::StitchSession), so consumers like
/// `reco_autocam::setup_autocam` can configure either without duplication.
pub trait DetectionTarget {
    /// Attach a detector backend.
    fn set_detector(&mut self, detector: Box<dyn detector::UnifiedDetector>);
    /// Set the detection interval (run every N frames).
    fn set_detection_interval(&mut self, interval: u64);
    /// Attach a ball tracker.
    fn set_ball_tracker(&mut self, tracker: Box<dyn tracker::Tracker>);
    /// Attach a player tracker.
    fn set_player_tracker(&mut self, tracker: Box<dyn tracker::Tracker>);
    /// Attach a panner that resolves viewport pose from tracked state.
    fn set_panner(&mut self, panner: Box<dyn panner::Panner>);
    /// Shared reference to the pipeline (for source_info, calibration).
    fn pipeline(&self) -> &crate::render::pipeline::StitchPipeline;
    /// Shared reference to the GPU context.
    fn gpu(&self) -> &crate::gpu::GpuContext;
}
