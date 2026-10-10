//! Error types for the calibration pipeline.

use thiserror::Error;

/// Errors from the calibration pipeline. `Clone + Send + Sync` so
/// the calibration background thread can post results back to the
/// UI thread with typed errors (no stringification at the mpsc
/// boundary).
#[derive(Debug, Clone, Error)]
pub enum CalibrateError {
    /// Direct-image alignment or verification failed.
    #[error("{0}")]
    Alignment(String),
    /// Configuration has invalid values.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}
