//! Native stereo calibration from overlapping images.
//!
//! Pixel correspondences in the overlap initialize direct image alignment.
//! Supplied optics are preserved; missing optics are estimated for stitching, not certified as reusable lens profiles.
//! The first synchronized pair fits the model. Later pairs independently check the overlap before automatic application.
//! Consumers must explicitly accept a needs-review candidate before applying or saving it.

#![deny(unsafe_code)]

pub mod audio_sync;
mod direct;
pub mod error;
pub mod lens_database;
/// Live calibration — drive the calibration pipeline from a live
/// frame-pair source (V4L2, WebRTC, etc.). See the `live` module's
/// `calibrate_from_live` function and `LiveFramePairSource` trait. Does
/// not require the `io` feature — the source is a trait object supplied
/// by the consumer, not a file.
pub mod live;
pub mod pipeline;
pub mod sampling;
pub mod telemetry;
pub mod types;
#[cfg(feature = "io")]
pub mod video;

pub use error::CalibrateError;
pub use types::{
    CalibrationAssessment, CalibrationConfig, CalibrationProgress, CalibrationResult,
    CalibrationStep, LensProfileInfo, LensProfileSummary, OverlapCheck, ProfileSource,
    Yuv420pFrame,
};

/// Fit synchronized images using supplied optics where available and estimate missing optics.
///
/// Images must represent each lens's complete reference image. Resizing is supported; cropping requires updated intrinsics.
/// All pairs for each camera must share valid tightly packed YUV420P dimensions. Work is bounded to 16 pairs and a 60-second native deadline.
/// Cancellation is checked throughout native work. The returned assessment controls automatic application, not lens certification.
pub fn calibrate(
    frames: &[(Yuv420pFrame, Yuv420pFrame)],
    profiles: [Option<&reco_core::calibration::Lens>; 2],
    interrupted: &std::sync::atomic::AtomicBool,
) -> Result<CalibrationResult, CalibrateError> {
    direct::calibrate(frames, profiles, interrupted)
}
