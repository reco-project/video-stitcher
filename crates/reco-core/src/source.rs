//! Frame source trait for pluggable input backends.
//!
//! The pipeline doesn't care where frames come from - video files,
//! live cameras, network streams, or test patterns. Each source
//! implements [`FrameSource`] and delivers per-camera frame sets in
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
//! CPU-GPU transfers entirely. See `interop::cuda` in `reco-core`.

use thiserror::Error;

/// Errors from frame sources.
#[derive(Debug, Clone, Error)]
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

    /// The requested input path was rejected during pre-open validation.
    ///
    /// Emitted before any decoder is touched when the path fails basic
    /// sanity checks (missing, not a file, empty, unreadable). Consumers
    /// should prefer this over [`Init`](Self::Init) for user-facing
    /// error messages because it carries structured reasons instead of
    /// relying on stringified FFmpeg diagnostics.
    #[error("invalid input path ({path}): {reason}")]
    InvalidPath {
        /// Path the caller supplied.
        path: String,
        /// Structured reason the path was rejected.
        reason: InvalidPathReason,
    },
}

/// Structured reasons a source path can be rejected before opening.
///
/// Kept in an enum so consumers can branch on failure mode (show a
/// specific red tint for "file not found" vs "empty file") without
/// regex-matching an error string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum InvalidPathReason {
    /// Path does not exist on disk.
    #[error("file not found")]
    NotFound,
    /// Path exists but is not a regular file (directory, pipe, device).
    #[error("not a regular file")]
    NotAFile,
    /// File is zero bytes; nothing for a decoder to work with.
    #[error("file is empty")]
    Empty,
    /// Current process cannot read the file (permission denied).
    #[error("permission denied")]
    PermissionDenied,
}

/// Validate an input path against the basic prerequisites every
/// file-backed source needs: exists, is a regular file, is non-empty,
/// and is readable by the current process.
///
/// Returns `Ok(())` for valid paths, or
/// [`SourceError::InvalidPath`] describing
/// why it was rejected. Consumers should call this before attempting
/// any codec-specific open so the user sees a clear error instead of
/// a stringified FFmpeg "Invalid argument".
///
/// This is a cheap syscall (`metadata` + optional `open` probe) and is
/// safe to run on every file pick.
pub fn validate_input_path(path: &std::path::Path) -> Result<(), SourceError> {
    let make_err = |reason: InvalidPathReason| SourceError::InvalidPath {
        path: path.display().to_string(),
        reason,
    };

    let metadata = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) => {
            return Err(match e.kind() {
                std::io::ErrorKind::NotFound => make_err(InvalidPathReason::NotFound),
                std::io::ErrorKind::PermissionDenied => {
                    make_err(InvalidPathReason::PermissionDenied)
                }
                _ => SourceError::InvalidPath {
                    path: path.display().to_string(),
                    reason: InvalidPathReason::NotFound,
                },
            });
        }
    };
    if !metadata.is_file() {
        return Err(make_err(InvalidPathReason::NotAFile));
    }
    if metadata.len() == 0 {
        return Err(make_err(InvalidPathReason::Empty));
    }
    // Touch the file to surface permission errors that stat might miss
    // (e.g. readable dir, unreadable file on some filesystems).
    match std::fs::File::open(path) {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            Err(make_err(InvalidPathReason::PermissionDenied))
        }
        Err(_) => Ok(()), // surface non-permission errors via the real opener
    }
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
/// Used when dimensions are tracked separately - all cameras in a
/// [`FrameSet`] share the dimensions carried by [`SourceInfo`].
#[derive(Debug, Clone)]
pub struct YuvData {
    /// Y (luma) plane, full resolution.
    pub y: Vec<u8>,
    /// U (Cb) plane, half resolution.
    pub u: Vec<u8>,
    /// V (Cr) plane, half resolution.
    pub v: Vec<u8>,
}

impl YuvData {
    /// Borrow as pipeline-ready plane references.
    pub fn as_planes(&self) -> crate::render::planes::YuvPlanes<'_> {
        crate::render::planes::YuvPlanes {
            y: &self.y,
            u: &self.u,
            v: &self.v,
        }
    }
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

impl Nv12Data {
    /// Borrow as pipeline-ready plane references.
    pub fn as_planes(&self) -> crate::render::planes::Nv12Planes<'_> {
        crate::render::planes::Nv12Planes {
            y: &self.y,
            uv: &self.uv,
        }
    }
}

/// One camera's D3D11VA decoded frame: an array texture plus the slice
/// index within the decode pool.
///
/// Non-owning - the pointer is only valid while the source pins the
/// underlying decode slice alive (until the session has staged the
/// frame into shared textures).
#[cfg(target_os = "windows")]
#[derive(Debug, Clone, Copy)]
pub struct D3d11CameraFrame {
    /// D3D11 array texture pointer (ID3D11Texture2D*).
    pub texture: *mut std::ffi::c_void,
    /// Array slice index within the D3D11VA decode pool.
    pub slice: usize,
}

/// Per-camera metadata for an NVMM zero-copy frame (Jetson).
///
/// Carries the DMA-buf fd (Vulkan render import) and the raw
/// `NvBufSurface*` pointer (NvBufSurfTransform detection preprocessing)
/// for one camera, plus the plane geometry needed for the Vulkan import.
/// This is the reco-core mirror of reco-io's `NvmmFrameInfo` - reco-core
/// has no I/O deps, so the I/O backend constructs this from its own type.
///
/// Both the fd and the surface pointer are only valid while the source
/// holds the underlying GStreamer sample alive (until the session has
/// copied the frame into the VRAM pool and run detection on it).
#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy)]
pub struct NvmmPlaneInfo {
    /// DMA-buf file descriptor for Vulkan import (render path).
    pub dmabuf_fd: i32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Y plane byte offset within the DMA-buf.
    pub y_offset: u32,
    /// UV plane byte offset within the DMA-buf.
    pub uv_offset: u32,
    /// Total allocation size (Vulkan memory import).
    pub total_size: u32,
    /// Raw `NvBufSurface*` pointer for `NvBufSurfTransform` (detection path).
    pub surface_ptr: *mut std::ffi::c_void,
}

// SAFETY: the surface pointer is a stable kernel-managed NVMM pool address;
// the source's capture thread keeps the GstSample alive until release. The
// session that consumes this never moves it across threads.
#[cfg(target_os = "linux")]
unsafe impl Send for NvmmPlaneInfo {}

/// One time instant's frames from every camera, in projection order
/// (index `i` pairs with `calibration.lenses[i]`).
///
/// Residency - where the pixel data lives - is a set-level property:
/// every source decides its backend once at open (one decode path, one
/// shared texture pool for all cameras), so one variant covers the
/// whole set and a mixed-residency set is unrepresentable.
///
/// Sources produce whichever variant is most efficient for their backend:
/// - File decode (CPU path): `Yuv420p`
/// - Jetson ISP / NVDEC NV12: `Nv12`
/// - CUDA/Vulkan zero-copy shared textures: `GpuResident`
/// - VideoToolbox/Metal zero-copy: `MetalResident`
/// - Windows D3D11VA zero-copy: `D3d11Resident`
/// - Jetson NVMM zero-copy (DMA-buf + NvBufSurface): `NvmmResident`
///
/// CPU variants carry one entry per camera; the stitch boundary
/// rejects a length that does not match the projection's camera count
/// (`check_camera_count` in `stitch::cpu`, mirrored by the submit
/// dispatch). Platform zero-copy variants are pinned to two cameras by
/// type - their decode channels and texture pools are pair-shaped by
/// design and guarded, not generalized.
#[non_exhaustive]
pub enum FrameSet {
    /// CPU-resident YUV420P planes (3 planes per camera).
    Yuv420p(Vec<YuvData>),
    /// CPU-resident NV12 planes (2 planes per camera).
    Nv12(Vec<Nv12Data>),
    /// GPU-resident: data already written to shared textures by the
    /// source. Values are double-buffer slot indices (0 or 1), one per
    /// camera, that the pipeline uses to select the correct bind group.
    GpuResident {
        /// Per-camera double-buffer slot indices.
        slots: [u8; 2],
    },
    /// macOS zero-copy: retained CVPixelBuffers from VideoToolbox
    /// decode, one per camera. The session imports these as Metal
    /// textures each frame.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    MetalResident([crate::interop::metal::RetainedCVPixelBuffer; 2]),
    /// Windows D3D11VA zero-copy: decoded frames still on D3D11 GPU
    /// memory, one per camera. The session stages these into shared
    /// NV12 textures for wgpu rendering.
    #[cfg(target_os = "windows")]
    D3d11Resident([D3d11CameraFrame; 2]),
    /// Jetson NVMM zero-copy: NV12 frames in NvBufSurface DMA-buf
    /// memory, one per camera. The session imports the DMA-bufs as
    /// Vulkan textures for rendering (copied into the VRAM pool) and
    /// runs `NvBufSurfTransform` on the surface pointers for detection.
    /// Produced by the NVMM camera source.
    #[cfg(target_os = "linux")]
    NvmmResident([NvmmPlaneInfo; 2]),
}

impl FrameSet {
    /// Consume a two-camera YUV420P set into its per-camera payloads.
    ///
    /// `None` for any other shape - the interactive stereo consumers
    /// (CLI preview, GUI playback) that retain decoded frames are
    /// two-camera CPU paths by construction and use this instead of
    /// re-deriving the downcast.
    pub fn into_yuv_pair(self) -> Option<[YuvData; 2]> {
        match self {
            Self::Yuv420p(cams) => <[YuvData; 2]>::try_from(cams).ok(),
            _ => None,
        }
    }
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

/// Trait for frame sources.
///
/// A frame source delivers one [`FrameSet`] per time instant to the
/// pipeline in whatever format is most efficient for the backend. The
/// pipeline handles format differences internally via the set's variant.
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

    /// Get the next frame set, or `None` if the source is exhausted.
    ///
    /// For live sources (cameras), this blocks until a frame is available.
    /// For file sources, returns `None` at end of file.
    fn next_frame(&mut self) -> Result<Option<FrameSet>, SourceError>;

    /// Non-blocking attempt to get the next frame.
    ///
    /// Returns `Ok(None)` if no frame is available yet (not exhausted,
    /// just not ready). Used by interactive consumers (preview window)
    /// that need to poll without blocking the UI thread.
    ///
    /// Default implementation delegates to [`Self::next_frame`] (blocking).
    fn try_next_frame(&mut self) -> Result<Option<FrameSet>, SourceError> {
        self.next_frame()
    }

    /// Whether this source delivers GPU-resident frames.
    ///
    /// When `true`, [`next_frame`](Self::next_frame) may return
    /// [`FrameSet::GpuResident`] or `FrameSet::MetalResident`.
    /// The session uses this to configure GPU bind groups and select
    /// the optimal render path automatically.
    fn is_gpu_resident(&self) -> bool {
        false
    }

    /// GPU pixel format for GPU-resident sources.
    ///
    /// Only meaningful when [`is_gpu_resident`](Self::is_gpu_resident) returns `true`.
    /// Determines shared texture formats (R8Unorm for NV12, R16Unorm for P010).
    #[cfg(feature = "gpu")]
    fn gpu_pixel_format(&self) -> crate::render::renderer::GpuPixelFormat {
        crate::render::renderer::GpuPixelFormat::Nv12
    }

    /// Whether the source uses full-range YUV (0-255) rather than limited (16-235).
    fn is_full_range(&self) -> bool {
        false
    }

    /// Left camera rotation from stream metadata (degrees: 0, 90, 180, 270).
    ///
    /// The session applies rotation automatically: the CPU path handles it
    /// via buffer reversal in the decoder, while the GPU zero-copy path
    /// uses a shader UV flip.
    ///
    /// Pair-shaped on purpose for now: the session consumes rotation
    /// for GPU-resident sources only, which are two-camera; the
    /// accessors go per-camera when the FrameSource seam does.
    fn left_rotation(&self) -> i32 {
        0
    }

    /// Right camera rotation from stream metadata (degrees: 0, 90, 180, 270).
    fn right_rotation(&self) -> i32 {
        0
    }

    /// Skip N frames without processing them.
    ///
    /// The default implementation calls `next_frame()` in a loop, which
    /// works for CPU and D3D11VA sources. GPU zero-copy sources override
    /// this to receive and immediately release decode slots, avoiding
    /// deadlock from the bounded slot channel.
    fn skip_frames(&mut self, count: u64) -> Result<u64, SourceError> {
        for i in 0..count {
            if self.next_frame()?.is_none() {
                return Ok(i);
            }
        }
        Ok(count)
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

    /// Whether the source has reached end-of-stream.
    ///
    /// Once [`try_next_frame`](Self::try_next_frame) or
    /// [`next_frame`](Self::next_frame) has returned `Ok(None)` for reasons
    /// that are final (container EOF, camera stream ended), implementations
    /// should return `true` from this method so callers can distinguish
    /// "finished" from "frame not ready yet".
    ///
    /// Used by interactive consumers (GUI playback, OBS) that need to tell
    /// end-of-stream apart from a transiently empty decode channel without
    /// a timeout heuristic. File sources should override this to track EOF;
    /// live sources (cameras) can keep the default (`false`) since they
    /// never exhaust under normal operation.
    fn is_exhausted(&self) -> bool {
        false
    }

    /// Begin data flow (spawn decode threads, start capture, etc.).
    ///
    /// After `open()`, all metadata is available but no frames are
    /// produced yet. Call this to start the decode pipeline. If not
    /// called explicitly, `next_frame()` must auto-start.
    ///
    /// This exists so callers can initialize GPU resources (ORT/DML,
    /// Metal compute, etc.) between probe and decode without contending
    /// for the GPU device with decode threads.
    fn start_decoding(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn validate_path_accepts_real_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("video.mp4");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"not a real mp4 but non-empty").unwrap();
        assert!(validate_input_path(&path).is_ok());
    }

    #[test]
    fn validate_path_rejects_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("does-not-exist.mp4");
        match validate_input_path(&path) {
            Err(SourceError::InvalidPath {
                reason: InvalidPathReason::NotFound,
                ..
            }) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn validate_path_rejects_directory() {
        let dir = tempdir().unwrap();
        match validate_input_path(dir.path()) {
            Err(SourceError::InvalidPath {
                reason: InvalidPathReason::NotAFile,
                ..
            }) => {}
            other => panic!("expected NotAFile, got {other:?}"),
        }
    }

    #[test]
    fn validate_path_rejects_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.mp4");
        std::fs::File::create(&path).unwrap();
        match validate_input_path(&path) {
            Err(SourceError::InvalidPath {
                reason: InvalidPathReason::Empty,
                ..
            }) => {}
            other => panic!("expected Empty, got {other:?}"),
        }
    }
}
