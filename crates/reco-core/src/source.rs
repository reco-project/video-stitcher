//! Frame source trait for pluggable input backends.
//!
//! The pipeline doesn't care where frames come from — video files,
//! live cameras, network streams, or test patterns. Each source
//! implements [`FrameSource`] and delivers stereo frame pairs.
//!
//! ## Implementations
//!
//! - `reco-ffmpeg`: file-based decode via FFmpeg (software + hardware)
//! - Future: V4L2/libcamera for direct sensor input (e.g. IMX on Jetson)
//! - Future: GStreamer pipeline for network streams
//!
//! ## Design
//!
//! Frame data is always RGBA bytes on the CPU in phase 1. In the future,
//! sources may provide GPU-resident frames (e.g. NVDEC output as a
//! wgpu texture) to avoid CPU↔GPU transfers entirely. The trait is
//! designed so that extension doesn't break existing sources.

use thiserror::Error;

/// Errors from frame sources.
#[derive(Debug, Error)]
pub enum SourceError {
    /// The source failed to open or initialize.
    #[error("source init: {0}")]
    Init(String),

    /// A frame could not be read.
    #[error("frame read: {0}")]
    Read(String),
}

/// A stereo frame pair from the source.
///
/// Contains left and right camera data as RGBA bytes (CPU-resident).
/// Both frames must have the same dimensions.
pub struct FramePair {
    /// Left camera RGBA data (width * height * 4 bytes).
    pub left: Vec<u8>,
    /// Right camera RGBA data (width * height * 4 bytes).
    pub right: Vec<u8>,
}

/// Metadata about the frame source.
pub struct SourceInfo {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frames per second (may be approximate for live sources).
    pub fps: f64,
}

/// Trait for stereo frame sources.
///
/// A frame source delivers pairs of left/right RGBA frames to the
/// pipeline. Implementations handle their own threading — the pipeline
/// calls [`Self::next_pair`] from the main thread and expects it to
/// return quickly (either with data or `None` for end-of-stream).
///
/// # Example
///
/// ```rust,ignore
/// use reco_core::source::{FrameSource, SourceInfo, FramePair, SourceError};
///
/// struct TestPatternSource { frame: u64 }
///
/// impl FrameSource for TestPatternSource {
///     fn info(&self) -> SourceInfo {
///         SourceInfo { width: 1920, height: 1080, fps: 30.0 }
///     }
///
///     fn next_pair(&mut self) -> Result<Option<FramePair>, SourceError> {
///         // Generate test pattern...
///         Ok(None)
///     }
/// }
/// ```
pub trait FrameSource: Send {
    /// Source metadata (dimensions, frame rate).
    fn info(&self) -> SourceInfo;

    /// Get the next stereo frame pair, or `None` if the source is exhausted.
    ///
    /// For live sources (cameras), this blocks until a frame is available.
    /// For file sources, returns `None` at end of file.
    fn next_pair(&mut self) -> Result<Option<FramePair>, SourceError>;
}
