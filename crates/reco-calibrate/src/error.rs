//! Error types for the calibration pipeline.

use thiserror::Error;

/// Errors produced during stitching calibration.
#[derive(Debug, Error)]
pub enum CalibrateError {
    /// No keypoints were detected in a frame.
    #[error("no keypoints detected in {camera} frame {frame_idx}")]
    NoKeypoints {
        /// Which camera (`"left"` or `"right"`).
        camera: &'static str,
        /// Frame index (0-based).
        frame_idx: usize,
    },

    /// Too few feature matches survived filtering.
    #[error("insufficient matches: got {got}, need at least {min}")]
    InsufficientMatches {
        /// Number of matches found.
        got: usize,
        /// Minimum required.
        min: usize,
    },

    /// RANSAC rejected all candidate matches.
    #[error("RANSAC rejected all matches")]
    RansacFailed,

    /// The optimizer did not converge to a solution.
    #[error("optimizer did not converge after {max_evals} evaluations")]
    OptimizerFailed {
        /// Maximum evaluations allowed.
        max_evals: usize,
    },

    /// No frame pairs produced usable matches after the full pipeline.
    #[error("no usable frame pairs (all frames failed matching)")]
    NoUsableFrames,

    /// A frame has invalid dimensions (zero or too large).
    #[error("invalid frame dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
    },
}
