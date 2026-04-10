//! Frame source trait for pluggable input backends.
//!
//! The pipeline doesn't care where frames come from - video files,
//! live cameras, network streams, or test patterns. Each source
//! implements [`FrameSource`] and delivers stereo frame pairs in
//! YUV420P or NV12 format.
//!
//! ## Implementations (in `reco-io`)
//!
//! - FFmpeg backend: file-based decode via FFmpeg (software + hardware)
//! - GStreamer backend: live camera capture (Jetson ISP, V4L2, AVFoundation, Media Foundation)
//!
//! ## Design
//!
//! Frame data is YUV420P or NV12 on the CPU. YUV420P uses three
//! separate planes (Y full-res, U half-res, V half-res). NV12 uses
//! two planes (Y full-res, interleaved UV half-res) and is the
//! native output of NVIDIA ISP and NVDEC. The GPU pipeline uploads
//! either format directly and converts to RGB in the shader, avoiding
//! any CPU-side color conversion.
//!
//! For GPU-resident frames (e.g. NVDEC output via CUDA interop),
//! sources can write directly to shared GPU textures, avoiding
//! CPU-GPU transfers entirely. See `cuda_interop` in `reco-core`.

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

/// Owned YUV420P frame data with dimensions and optional timestamp.
///
/// The canonical YUV frame type used across all reco crates.
/// Tightly packed (no stride padding):
/// - Y: `width × height` bytes
/// - U: `(width/2) × (height/2)` bytes
/// - V: `(width/2) × (height/2)` bytes
#[derive(Debug, Clone)]
pub struct YuvFrame {
    /// Y (luma) plane, full resolution.
    pub y: Vec<u8>,
    /// U (Cb) plane, half resolution.
    pub u: Vec<u8>,
    /// V (Cr) plane, half resolution.
    pub v: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in microseconds (0 if unknown).
    pub timestamp_us: i64,
}

/// Owned YUV420P plane data (without dimensions).
///
/// Used internally when dimensions are tracked separately
/// (e.g. in [`FramePair`] where both frames share dimensions).
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct Nv12FramePair {
    /// Left camera NV12 data.
    pub left: Nv12Data,
    /// Right camera NV12 data.
    pub right: Nv12Data,
}

/// A stereo frame in any supported format.
///
/// Sources produce whichever format is most efficient for their backend:
/// - File decode (CPU path): `Yuv420p`
/// - Jetson ISP / NVDEC NV12: `Nv12`
/// - NVDEC zero-copy (CUDA/Vulkan shared textures): `GpuResident`
pub enum StereoFrame {
    /// CPU-resident YUV420P planes (3 planes per camera).
    Yuv420p(FramePair),
    /// CPU-resident NV12 planes (2 planes per camera).
    Nv12(Nv12FramePair),
    /// GPU-resident: data already written to shared textures by the source.
    /// The `u8` values are double-buffer slot indices that the pipeline
    /// uses to select the correct bind group.
    GpuResident { left_slot: u8, right_slot: u8 },
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
/// A frame source delivers stereo frame pairs to the pipeline in whatever
/// format is most efficient for the backend. The pipeline handles format
/// differences internally via [`StereoFrame`].
///
/// Implementations handle their own threading (e.g. dedicated capture
/// threads with bounded channels). The pipeline calls [`Self::next_frame`]
/// and expects it to block until data is ready or return `None` for
/// end-of-stream.
pub trait FrameSource: Send {
    /// Source metadata (dimensions, frame rate).
    fn info(&self) -> SourceInfo;

    /// Get the next stereo frame, or `None` if the source is exhausted.
    ///
    /// For live sources (cameras), this blocks until a frame is available.
    /// For file sources, returns `None` at end of file.
    fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError>;

    /// Non-blocking attempt to get the next frame.
    ///
    /// Returns `Ok(None)` if no frame is available yet (not exhausted,
    /// just not ready). Used by interactive consumers (preview window)
    /// that need to poll without blocking the UI thread.
    ///
    /// Default implementation delegates to [`Self::next_frame`] (blocking).
    fn try_next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        self.next_frame()
    }
}
