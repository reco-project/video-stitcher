//! Frame source trait for pluggable input backends.
//!
//! The pipeline doesn't care where frames come from — video files,
//! live cameras, network streams, or test patterns. Each source
//! implements [`FrameSource`] and delivers stereo YUV420P frame pairs.
//!
//! ## Implementations
//!
//! - `reco-ffmpeg`: file-based decode via FFmpeg (software + hardware)
//! - Future: V4L2/libcamera for direct sensor input (e.g. IMX on Jetson)
//! - Future: GStreamer pipeline for network streams
//!
//! ## Design
//!
//! Frame data is always YUV420P on the CPU: three separate planes
//! (Y full-res, U half-res, V half-res). The GPU pipeline uploads
//! these directly and converts to RGB in the shader, avoiding any
//! CPU-side color conversion.
//!
//! In the future, sources may provide GPU-resident frames (e.g. NVDEC
//! output) to avoid CPU↔GPU transfers entirely. The trait is designed
//! so that extension doesn't break existing sources.

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

/// Owned YUV420P plane data.
///
/// Tightly packed (no stride padding):
/// - Y: `width × height` bytes
/// - U: `(width/2) × (height/2)` bytes
/// - V: `(width/2) × (height/2)` bytes
pub struct YuvData {
    /// Y (luma) plane, full resolution.
    pub y: Vec<u8>,
    /// U (Cb) plane, half resolution.
    pub u: Vec<u8>,
    /// V (Cr) plane, half resolution.
    pub v: Vec<u8>,
}

/// Owned NV12 plane data.
///
/// Tightly packed (no stride padding):
/// - Y: `width × height` bytes
/// - UV: `width × (height/2)` bytes (interleaved U,V at half resolution)
pub struct Nv12Data {
    /// Y (luma) plane, full resolution.
    pub y: Vec<u8>,
    /// Interleaved UV (CbCr) plane, half resolution in each dimension.
    pub uv: Vec<u8>,
}

/// A stereo frame pair from the source.
///
/// Contains left and right camera data as YUV420P planes (CPU-resident).
/// Both frames must have the same dimensions.
pub struct FramePair {
    /// Left camera YUV420P data.
    pub left: YuvData,
    /// Right camera YUV420P data.
    pub right: YuvData,
}

/// A stereo NV12 frame pair from the source.
///
/// Contains left and right camera data as NV12 planes (CPU-resident).
/// NV12 is the native output of NVIDIA ISP (nvarguscamerasrc) and NVDEC,
/// so this avoids an NV12 -> I420 conversion on capture.
pub struct Nv12FramePair {
    /// Left camera NV12 data.
    pub left: Nv12Data,
    /// Right camera NV12 data.
    pub right: Nv12Data,
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
/// A frame source delivers pairs of left/right YUV420P frames to the
/// pipeline. Implementations handle their own threading — the pipeline
/// calls [`Self::next_pair`] from the main thread and expects it to
/// return quickly (either with data or `None` for end-of-stream).
pub trait FrameSource: Send {
    /// Source metadata (dimensions, frame rate).
    fn info(&self) -> SourceInfo;

    /// Get the next stereo frame pair, or `None` if the source is exhausted.
    ///
    /// For live sources (cameras), this blocks until a frame is available.
    /// For file sources, returns `None` at end of file.
    fn next_pair(&mut self) -> Result<Option<FramePair>, SourceError>;
}
