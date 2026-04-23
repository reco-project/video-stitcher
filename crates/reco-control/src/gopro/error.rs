//! Error types for GoPro operations.

/// Errors from GoPro camera operations.
#[derive(Debug, thiserror::Error)]
pub enum GoProError {
    /// HTTP transport failure (connection refused, timeout, DNS).
    #[error("GoPro HTTP error: {0}")]
    Http(String),

    /// Camera returned a non-success HTTP status or an error payload.
    #[error("GoPro camera error: {0}")]
    CameraError(String),

    /// Camera is busy (encoding, processing) and cannot accept the command.
    #[error("GoPro busy: {0}")]
    Busy(String),

    /// Camera is not reachable at the expected address.
    #[error("GoPro not found at {0}")]
    NotFound(String),
}
