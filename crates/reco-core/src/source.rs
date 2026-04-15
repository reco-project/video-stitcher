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
    #[error("source init ({path}): {reason}")]
    Init {
        /// Path or identifier of the source that failed.
        path: String,
        /// Human-readable explanation of the failure.
        reason: String,
    },

    /// A frame could not be read.
    #[error("frame read: {reason}")]
    Read {
        /// Human-readable explanation of the failure.
        reason: String,
    },
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

impl YuvFrame {
    /// Validate that plane sizes match the declared dimensions.
    ///
    /// Y plane must be `width * height` bytes, U and V planes must each
    /// be `(width / 2) * (height / 2)` bytes (YUV420P subsampling).
    ///
    /// Returns `Ok(())` if valid, or an `Err` describing the mismatch.
    pub fn validate(&self) -> Result<(), String> {
        let expected_y = self.width as usize * self.height as usize;
        if self.y.len() != expected_y {
            return Err(format!(
                "Y plane size mismatch: expected {} ({}x{}), got {}",
                expected_y,
                self.width,
                self.height,
                self.y.len()
            ));
        }
        let expected_uv = (self.width as usize / 2) * (self.height as usize / 2);
        if self.u.len() != expected_uv {
            return Err(format!(
                "U plane size mismatch: expected {} ({}x{}), got {}",
                expected_uv,
                self.width / 2,
                self.height / 2,
                self.u.len()
            ));
        }
        if self.v.len() != expected_uv {
            return Err(format!(
                "V plane size mismatch: expected {} ({}x{}), got {}",
                expected_uv,
                self.width / 2,
                self.height / 2,
                self.v.len()
            ));
        }
        Ok(())
    }
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
/// - CUDA/Vulkan zero-copy shared textures: `GpuResident`
/// - VideoToolbox/Metal zero-copy: `MetalResident`
#[non_exhaustive]
pub enum StereoFrame {
    /// CPU-resident YUV420P planes (3 planes per camera).
    Yuv420p(FramePair),
    /// CPU-resident NV12 planes (2 planes per camera).
    Nv12(Nv12FramePair),
    /// GPU-resident: data already written to shared textures by the source.
    /// The `u8` values are double-buffer slot indices that the pipeline
    /// uses to select the correct bind group.
    GpuResident {
        /// Left camera double-buffer slot index (0 or 1).
        left_slot: u8,
        /// Right camera double-buffer slot index (0 or 1).
        right_slot: u8,
    },
    /// macOS zero-copy: retained CVPixelBuffers from VideoToolbox decode.
    /// The session imports these as Metal textures each frame.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    MetalResident {
        /// Left camera retained pixel buffer.
        left: crate::metal_interop::RetainedCVPixelBuffer,
        /// Right camera retained pixel buffer.
        right: crate::metal_interop::RetainedCVPixelBuffer,
    },
}

/// Metadata about the frame source.
#[derive(Debug, Clone)]
pub struct SourceInfo {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frames per second (may be approximate for live sources).
    pub fps: f64,
    /// Exact frame rate as a rational number (numerator, denominator).
    /// For example, 29.97fps is `(30000, 1001)`. Used by encoders for
    /// precise timing. `None` if the source cannot determine exact timing.
    pub fps_rational: Option<(i32, i32)>,
    /// Total number of frames in the source (from container metadata).
    /// `None` for live sources or when the count is unknown.
    pub total_frames: Option<u64>,
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
///
/// ## GPU-resident sources
///
/// Sources that deliver GPU-resident frames (CUDA/Vulkan shared textures,
/// Metal CVPixelBuffers) should override [`is_gpu_resident`](Self::is_gpu_resident)
/// to return `true` and provide their pixel format via
/// [`gpu_pixel_format`](Self::gpu_pixel_format). The session uses this metadata
/// to auto-configure bind groups, texture formats, and rotation handling.
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

    /// Whether this source delivers GPU-resident frames.
    ///
    /// When `true`, [`next_frame`](Self::next_frame) may return
    /// [`StereoFrame::GpuResident`] or `StereoFrame::MetalResident`.
    /// The session uses this to configure GPU bind groups and select
    /// the optimal render path automatically.
    fn is_gpu_resident(&self) -> bool {
        false
    }

    /// GPU pixel format for GPU-resident sources.
    ///
    /// Only meaningful when [`is_gpu_resident`](Self::is_gpu_resident) returns `true`.
    /// Determines shared texture formats (R8Unorm for NV12, R16Unorm for P010).
    fn gpu_pixel_format(&self) -> crate::renderer::GpuPixelFormat {
        crate::renderer::GpuPixelFormat::Nv12
    }

    /// Left camera rotation from stream metadata (degrees: 0, 90, 180, 270).
    ///
    /// The session applies rotation automatically: the CPU path handles it
    /// via buffer reversal in the decoder, while the GPU zero-copy path
    /// uses a shader UV flip.
    fn left_rotation(&self) -> i32 {
        0
    }

    /// Right camera rotation from stream metadata (degrees: 0, 90, 180, 270).
    fn right_rotation(&self) -> i32 {
        0
    }

    /// Seek to a specific frame number.
    ///
    /// File-based sources should implement this for interactive scrubbing.
    /// Live sources (cameras) return `Err` from the default implementation.
    fn seek(&mut self, _frame: u64) -> Result<(), SourceError> {
        Err(SourceError::Read {
            reason: "seek not supported by this source".into(),
        })
    }

    /// Total number of frames in the source, if known.
    ///
    /// File-based sources should return the frame count from container metadata.
    /// Live sources return `None`.
    fn total_frames(&self) -> Option<u64> {
        None
    }
}
