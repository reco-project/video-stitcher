//! Data types for the calibration pipeline.
//!
//! Frame selection, optical provenance and independent alignment assessment.

use std::path::PathBuf;

use reco_core::calibration::Calibration;
use serde::{Deserialize, Serialize};

/// Re-export the canonical YUV frame type from reco-core.
///
/// Tightly packed YUV420P shared by calibration and source consumers.
pub use reco_core::source::Yuv420pFrame;

/// Frame selection for video calibration; fitting uses one pair and verification uses the rest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationConfig {
    /// Number of synchronized pairs to sample, from 1 to 16.
    pub num_frames: usize,
    /// Seconds excluded at the start of both clips.
    pub skip_start_secs: f64,
    /// Seconds excluded at the end of both clips.
    pub skip_end_secs: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_frames: 4,
            skip_start_secs: 0.0,
            skip_end_secs: 0.0,
        }
    }
}

impl CalibrationConfig {
    /// Reject invalid or unbounded sampling requests before I/O.
    pub fn validate(&self) -> Result<(), crate::CalibrateError> {
        if !(1..=16).contains(&self.num_frames)
            || [self.skip_start_secs, self.skip_end_secs]
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(crate::CalibrateError::InvalidConfig(
                "expected 1..=16 pairs and finite nonnegative skip durations".into(),
            ));
        }
        Ok(())
    }
}

/// Progress information for calibration operations.
#[derive(Debug, Clone)]
pub struct CalibrationProgress {
    /// Current calibration step.
    pub step: CalibrationStep,
    /// Human-readable detail about the current operation.
    pub detail: String,
}

/// Steps in the calibration pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationStep {
    /// Probing video metadata.
    Probing,
    /// Extracting audio for temporal sync.
    AudioSync,
    /// Extracting video frames.
    ExtractingFrames,
    /// Optimizing camera parameters.
    Optimizing,
}

impl std::fmt::Display for CalibrationStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Probing => write!(f, "Probing"),
            Self::AudioSync => write!(f, "AudioSync"),
            Self::ExtractingFrames => write!(f, "ExtractingFrames"),
            Self::Optimizing => write!(f, "Optimizing"),
        }
    }
}

// ── Lens Profile Types ────────────────────────────────────────────

/// How a lens profile was resolved during calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfileSource {
    /// Estimated from overlap for this calibration, not a reusable physical lens profile.
    Estimated,
    /// Explicitly supplied by the caller and held fixed during fitting.
    Supplied,
}

/// Identifies the lens profile used for one camera during calibration.
///
/// Returned on [`CalibrationResult::left_lens_profile`] and
/// [`CalibrationResult::right_lens_profile`] so the GUI can display
/// the optical provenance without parsing logs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensProfileInfo {
    /// Camera brand and model (e.g. "GoPro HERO10").
    pub camera: String,
    /// Lens mode or setting (e.g. "Linear 4K 16:9").
    pub lens: String,
    /// How the profile was resolved.
    pub source: ProfileSource,
    /// Path on disk, if loaded from a file.
    pub path: Option<PathBuf>,
}

/// Summary of a lens profile in the embedded database.
///
/// Returned by [`LensDatabase::search`](crate::lens_database::LensDatabase::search)
/// for populating a picker/dropdown in the GUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensProfileSummary {
    /// Camera brand and model (e.g. "GoPro HERO10").
    pub camera: String,
    /// Lens mode or setting (e.g. "Linear 4K 16:9").
    pub lens: String,
    /// Profile resolution width in pixels.
    pub width: u32,
    /// Profile resolution height in pixels.
    pub height: u32,
}

// ── Calibration Result ───────────────────────────────────────────

/// Whether independent overlap measurements support applying a candidate automatically.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationAssessment {
    /// Unused-frame overlap checks support the stitching alignment, not physical lens certification.
    Confirmed,
    /// The candidate requires explicit visual acceptance before applying or saving.
    NeedsReview {
        /// Why automatic confirmation was unavailable.
        reason: String,
    },
}

/// Measured alignment on an image pair not used for fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapCheck {
    /// Bidirectionally verified patch correspondences.
    pub matches: usize,
    /// Match counts in upper, middle and lower source-image bands.
    pub bands: [usize; 3],
    /// Median displacement in pixels at a 960-pixel longest image edge.
    pub median_pixels: Option<f64>,
    /// 95th-percentile displacement in the same units.
    pub p95_pixels: Option<f64>,
}

/// A fitted calibration and its independent assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// The fitted calibration. Consult `assessment` before applying or saving.
    pub calibration: Calibration,
    /// Pixel correspondences used to initialize fitting.
    pub total_matches: usize,
    /// Number of frame pairs used for fitting, excluding independent checks.
    pub frames_used: usize,
    /// Whether the independent overlap evidence permits automatic application.
    pub assessment: CalibrationAssessment,
    /// Per-pair independent measurements. These images never participate in fitting.
    pub overlap_checks: Vec<OverlapCheck>,
    /// Optical provenance for the left camera.
    pub left_lens_profile: Option<LensProfileInfo>,
    /// Optical provenance for the right camera.
    pub right_lens_profile: Option<LensProfileInfo>,
}
