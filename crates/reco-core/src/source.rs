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
        left: crate::interop::metal::RetainedCVPixelBuffer,
        /// Right camera retained pixel buffer.
        right: crate::interop::metal::RetainedCVPixelBuffer,
    },
    /// Windows D3D11VA zero-copy: decoded frame still on D3D11 GPU memory.
    /// The session stages these into shared NV12 textures for wgpu rendering.
    #[cfg(target_os = "windows")]
    D3d11Resident {
        /// D3D11 array texture pointer (ID3D11Texture2D*) for left camera.
        left_texture: *mut std::ffi::c_void,
        /// Array slice index within the D3D11VA decode pool for left camera.
        left_slice: usize,
        /// D3D11 array texture pointer (ID3D11Texture2D*) for right camera.
        right_texture: *mut std::ffi::c_void,
        /// Array slice index within the D3D11VA decode pool for right camera.
        right_slice: usize,
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

// ---------------------------------------------------------------------------
// M3 foundation: CameraInput trait + placeholder impls.
// ---------------------------------------------------------------------------
//
// Plan-execution §2.6: today's stack hardcodes N=2 stereo input
// everywhere (StereoFrame, FramePair, submit_frame(left, right)).
// Future 1-video mode (§8 table) and N-camera panoramic rigs need a
// trait that expresses input cardinality independently of the frame
// delivery path.
//
// CameraInput is the input-cardinality contract. It complements
// Projection (output geometry) so StitchCore can enforce that input
// and projection agree on camera_count at construction time. The
// existing FrameSource trait stays in place as the concrete
// stereo-only frame delivery interface; CameraInput is the more
// general trait that N-camera work (and MonoCameraInput) will
// implement in later tranches.

/// Input-cardinality marker trait.
///
/// Concrete impls encode how many camera streams feed the stitch
/// pipeline. Used by StitchCore (M3 refactor) to verify its
/// [`Projection`](crate::projection::Projection) matches the input
/// contract at construction time.
///
/// `Send` bound only (mobile-friendly per plan-execution §2.8). A
/// concrete impl that needs to be shared across threads adds the
/// `Sync` bound itself.
pub trait CameraInput: Send {
    /// Short human-readable name for logs + diagnostic bundles
    /// (e.g. `"stereo-2camera"`, `"mono"`, `"panoramic-6camera"`).
    fn name(&self) -> &'static str;

    /// How many camera streams this input provides. Must equal the
    /// paired [`Projection::camera_count`](crate::projection::Projection::camera_count).
    fn camera_count(&self) -> u8;
}

/// Placeholder impl for today's 2-camera stereo input. Carries no
/// state; matches the hardcoded N=2 assumption everywhere the
/// existing code paths use.
#[derive(Debug, Default, Clone, Copy)]
pub struct StereoCameraInput;

impl CameraInput for StereoCameraInput {
    fn name(&self) -> &'static str {
        "stereo-2camera"
    }
    fn camera_count(&self) -> u8 {
        2
    }
}

/// Reserved slot for future 1-video mode (single-lens undistort +
/// viewport, no stitch). No impl ships today; the marker exists so
/// MonoProjection work can land alongside its matching input type
/// when the user picks the second-projection form (§7 decision 8).
#[derive(Debug, Default, Clone, Copy)]
pub struct MonoCameraInput;

impl CameraInput for MonoCameraInput {
    fn name(&self) -> &'static str {
        "mono-1camera"
    }
    fn camera_count(&self) -> u8 {
        1
    }
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

    // ── M3 foundation: CameraInput trait tests ───────────────────────

    #[test]
    fn stereo_camera_input_is_two_camera() {
        let s = StereoCameraInput;
        assert_eq!(s.name(), "stereo-2camera");
        assert_eq!(s.camera_count(), 2);
    }

    #[test]
    fn mono_camera_input_is_one_camera() {
        let m = MonoCameraInput;
        assert_eq!(m.name(), "mono-1camera");
        assert_eq!(m.camera_count(), 1);
    }

    #[test]
    fn camera_input_is_dyn_compatible() {
        // StitchCore (M3) will hold a `Box<dyn CameraInput>` slot.
        // Verify the trait bounds support that today.
        let inputs: Vec<Box<dyn CameraInput>> =
            vec![Box::new(StereoCameraInput), Box::new(MonoCameraInput)];
        assert_eq!(inputs[0].camera_count(), 2);
        assert_eq!(inputs[1].camera_count(), 1);
    }

    #[test]
    fn camera_count_matches_projection_contract() {
        // Core invariant for StitchCore construction: the paired
        // CameraInput and Projection must agree on camera_count.
        use crate::projection::{LShapeProjection, Projection};
        let input = StereoCameraInput;
        let proj = LShapeProjection;
        assert_eq!(input.camera_count(), proj.camera_count());
    }
}
