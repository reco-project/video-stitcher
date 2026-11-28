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

use std::sync::Arc;
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
pub struct Yuv420pFrame {
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

impl Yuv420pFrame {
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

/// Owned tightly packed YUV420P image in decoded pixel coordinates.
#[derive(Debug, Clone)]
pub struct Yuv420pData {
    /// Decoded `(width, height)`, independent of calibration resolution.
    pub size: (u32, u32),
    /// Y (luma) plane, full resolution.
    pub y: Vec<u8>,
    /// U (Cb) plane, half resolution.
    pub u: Vec<u8>,
    /// V (Cr) plane, half resolution.
    pub v: Vec<u8>,
}

impl Yuv420pData {
    /// Borrow as pipeline-ready plane references.
    pub fn as_planes(&self) -> crate::render::planes::Yuv420pPlanes<'_> {
        crate::render::planes::Yuv420pPlanes {
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
    /// Decoded `(width, height)`, independent of calibration resolution.
    pub size: (u32, u32),
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
/// Retains the texture and pins the decode slice until staging completes.
#[cfg(all(feature = "gpu", target_os = "windows"))]
pub struct D3d11CameraFrame {
    pub(crate) texture: Option<windows::Win32::Graphics::Direct3D11::ID3D11Texture2D>,
    pub(crate) slice: usize,
    _owner: Box<dyn Send>,
}

#[cfg(all(feature = "gpu", target_os = "windows"))]
impl D3d11CameraFrame {
    /// Attach the owner that pins the decoded texture slice until staging completes.
    ///
    /// # Safety
    /// A non-null pointer must denote a live ID3D11Texture2D on the decode device.
    /// The owner must prevent reuse of the slice for the entire lifetime of this frame.
    pub unsafe fn from_raw(
        texture: *mut std::ffi::c_void,
        slice: usize,
        owner: impl Send + 'static,
    ) -> Self {
        use windows::Win32::Graphics::Direct3D11::ID3D11Texture2D;
        use windows::core::Interface;
        let texture = (!texture.is_null()).then(|| {
            // The caller lends its COM reference; retain a reference for this frame.
            let borrowed =
                std::mem::ManuallyDrop::new(unsafe { ID3D11Texture2D::from_raw(texture) });
            ID3D11Texture2D::clone(&borrowed)
        });
        Self {
            texture,
            slice,
            _owner: Box::new(owner),
        }
    }
}
#[cfg(target_os = "linux")]
mod nvmm;
#[cfg(target_os = "linux")]
pub use nvmm::{NvmmFrame, NvmmPlaneInfo};

/// One image from every camera, in projection order
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
/// dispatch). CUDA/Vulkan leases also carry one entry per input and are
/// validated by the session, as are D3D11 and Metal frames. NVMM imports carry stereo pairs.
#[non_exhaustive]
pub enum FrameSet {
    /// Shared CPU-resident YUV420P images (3 planes per camera).
    /// Cloning a camera handle retains the image without copying its planes.
    Yuv420p(Vec<Arc<Yuv420pData>>),
    /// Shared CPU-resident NV12 images (2 planes per camera).
    /// Cloning a camera handle retains the image without copying its planes.
    Nv12(Vec<Arc<Nv12Data>>),
    /// CUDA/Vulkan NV12 or P010 images, one exclusive lease per camera.
    /// Keep the leases until all GPU reads complete.
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    GpuResident(Vec<crate::interop::cuda_lease::CudaLease>),
    /// macOS zero-copy: retained CVPixelBuffers from VideoToolbox
    /// decode, one per camera. The session imports these as Metal
    /// textures each frame.
    #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios")))]
    MetalResident(Vec<crate::interop::metal::RetainedCVPixelBuffer>),
    /// Windows D3D11VA zero-copy: decoded frames still on D3D11 GPU
    /// memory, one per camera. The session stages these into shared
    /// NV12 textures for wgpu rendering.
    #[cfg(all(feature = "gpu", target_os = "windows"))]
    D3d11Resident(Vec<D3d11CameraFrame>),
    /// Jetson NVMM zero-copy: NV12 frames in NvBufSurface DMA-buf
    /// memory, one per camera. The session imports the DMA-bufs as
    /// Vulkan textures for rendering (staged into the VRAM pool for lookahead) and
    /// runs `NvBufSurfTransform` on the surface pointers for detection.
    /// Frames own their native buffers and mapped metadata.
    #[cfg(target_os = "linux")]
    NvmmResident([NvmmFrame; 2]),
}

impl FrameSet {
    /// Check host image shapes and plane lengths before inference or rendering.
    pub(crate) fn validate_host_images(&self, sizes: &[(u32, u32)]) -> Result<(), String> {
        let check = |camera: usize, size: (u32, u32), lengths: &[usize], expected: &[usize]| {
            if sizes.get(camera) != Some(&size) || lengths != expected {
                return Err(format!(
                    "camera {camera} image layout does not match its configured dimensions"
                ));
            }
            Ok(())
        };
        let count = match self {
            Self::Yuv420p(images) => {
                for (camera, image) in images.iter().enumerate() {
                    let (w, h) = image.size;
                    let pixels = (w as usize)
                        .checked_mul(h as usize)
                        .ok_or("image size overflow")?;
                    check(
                        camera,
                        image.size,
                        &[image.y.len(), image.u.len(), image.v.len()],
                        &[pixels, pixels / 4, pixels / 4],
                    )?;
                }
                images.len()
            }
            Self::Nv12(images) => {
                for (camera, image) in images.iter().enumerate() {
                    let (w, h) = image.size;
                    let pixels = (w as usize)
                        .checked_mul(h as usize)
                        .ok_or("image size overflow")?;
                    check(
                        camera,
                        image.size,
                        &[image.y.len(), image.uv.len()],
                        &[pixels, pixels / 2],
                    )?;
                }
                images.len()
            }
            _ => return Ok(()),
        };
        if count != sizes.len() {
            return Err(format!(
                "expected {} camera(s) but {count} frame(s) were supplied",
                sizes.len()
            ));
        }
        Ok(())
    }

    /// Conservative image payload budget retained by this descriptor, excluding
    /// separately accounted staging pools and borrowed native handles. Shared host
    /// images are charged per reference, even if another frame retains the same allocation.
    pub(crate) fn retained_image_bytes(&self) -> Result<usize, String> {
        let sum = |total: usize, bytes: usize| total.checked_add(bytes);
        let bytes = match self {
            Self::Yuv420p(frames) => frames
                .iter()
                .flat_map(|f| [f.y.capacity(), f.u.capacity(), f.v.capacity()])
                .try_fold(0, sum),
            Self::Nv12(frames) => frames
                .iter()
                .flat_map(|f| [f.y.capacity(), f.uv.capacity()])
                .try_fold(0, sum),
            #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios")))]
            Self::MetalResident(frames) => frames
                .iter()
                .try_fold(0usize, |total, frame| {
                    let bytes = frame.data_size();
                    if bytes == 0 {
                        return Err("retained Metal image byte size unavailable".to_string());
                    }
                    total
                        .checked_add(bytes)
                        .ok_or_else(|| "retained image byte size overflow".to_string())
                })
                .map(Some)?,
            #[cfg(target_os = "linux")]
            Self::NvmmResident(_) => Some(0),
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            Self::GpuResident(_) => Some(0),
            #[cfg(all(feature = "gpu", target_os = "windows"))]
            Self::D3d11Resident(_) => Some(0),
        };
        bytes.ok_or_else(|| "retained image byte size overflow".into())
    }
}

/// Source images and their presentation time, independent of pixel residency.
///
/// A timestamp is relative to the source's output timeline, not wall-clock or
/// arrival time. It must be nonnegative and strictly increasing within a run.
/// `None` requests the configured fixed-rate cadence; a run cannot mix modes.
/// Sources own clock normalization and camera alignment. A timestamp does not
/// by itself assert that the images were captured simultaneously.
pub struct SourceFrame {
    /// Shared host images or retained native handles.
    pub images: FrameSet,
    /// Presentation time in microseconds, or an explicitly untimed frame.
    pub pts_us: Option<i64>,
    /// Capture identity and source-clock assessment, or unavailable acquisition timing.
    pub capture: Option<CaptureTiming>,
}

/// Per-camera capture identity and age on one source clock, independent of output PTS.
///
/// Times are listed in projection order and must be nonnegative and nondecreasing for each camera across frames. An unchanged time identifies the same immutable image; new images require later times. For bounded reuse the assessment instant must advance with source media time even when all images repeat, not with processing elapsed time. A source run cannot change timing availability or its age policy.
#[derive(Clone, Debug)]
pub struct CaptureTiming {
    /// Original capture times, one per calibration camera.
    pub times_us: Vec<i64>,
    /// Source-clock instant at which this set's capture ages are assessed.
    pub at_us: i64,
    /// The source's maximum permitted image age. None means the source declares no age bound, not that images are fresh.
    ///
    /// Tracking may retain a successful observation for this duration plus one scheduled detection interval. Repeated display or inference never renews its original capture time.
    pub max_age_us: Option<u64>,
}

impl CaptureTiming {
    pub(crate) fn validate(&self, cameras: usize) -> Result<(), String> {
        if self.at_us < 0
            || self.times_us.len() != cameras
            || self.times_us.iter().any(|&time| {
                time < 0
                    || time > self.at_us
                    || self
                        .max_age_us
                        .is_some_and(|age| (self.at_us - time) as u64 > age)
            })
        {
            return Err("capture timing must cover all cameras with nonnegative times no later than assessment and within the declared age limit".into());
        }
        Ok(())
    }
}

impl SourceFrame {
    /// Deliver images at the consumer's configured fixed frame rate.
    pub fn untimed(images: FrameSet) -> Self {
        Self {
            images,
            pts_us: None,
            capture: None,
        }
    }
}

/// Metadata about the frame source.
#[derive(Debug, Clone)]
pub struct SourceInfo {
    /// Fixed decoded `(width, height)` for each camera, in projection order.
    pub camera_sizes: Vec<(u32, u32)>,
    /// Frames per second (may be approximate for live sources), or zero if unknown.
    pub fps: f64,
    /// Exact frame rate as a rational number (numerator, denominator).
    /// For example, 29.97fps is `(30000, 1001)`. Used by encoders for
    /// precise timing. `None` if the source cannot determine exact timing.
    pub fps_rational: Option<(i32, i32)>,
    /// Estimated total number of frame sets, after source alignment.
    /// `None` for live sources or when the count is unknown. Container metadata may be approximate; use end-of-stream, not this count, to stop processing.
    pub total_frames: Option<u64>,
}

/// Validate fixed input extents against a projection's camera count and subsampling.
pub(crate) fn validate_camera_sizes(
    sizes: &[(u32, u32)],
    cameras: usize,
    subsampled: bool,
) -> Result<(), String> {
    if sizes.len() != cameras || sizes.is_empty() {
        return Err(format!(
            "expected {cameras} input sizes, got {}",
            sizes.len()
        ));
    }
    for &(width, height) in sizes {
        if width == 0
            || height == 0
            || (subsampled && (!width.is_multiple_of(2) || !height.is_multiple_of(2)))
            || width > crate::calibration::MAX_DIM
            || height > crate::calibration::MAX_DIM
        {
            return Err(format!(
                "invalid input dimensions {width}x{height} (subsampled={subsampled})"
            ));
        }
    }
    Ok(())
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
/// `gpu_pixel_format` (with the `gpu` feature). The session uses this metadata
/// to auto-configure bind groups, texture formats, and rotation handling.
pub trait FrameSource: Send {
    /// Source metadata (dimensions, frame rate).
    fn info(&self) -> SourceInfo;

    /// Get the next images with their presentation time, or `None` if exhausted.
    ///
    /// For live sources (cameras), this blocks until a frame is available.
    /// For file sources, returns `None` at end of file.
    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError>;

    /// Non-blocking attempt to get the next frame.
    ///
    /// Returns `Ok(None)` if no frame is available yet (not exhausted,
    /// just not ready). Used by interactive consumers (preview window)
    /// that need to poll without blocking the UI thread.
    ///
    /// Sources implementing this must also report final exhaustion through [`Self::is_exhausted`]. The default returns an error without reading a frame, so an unsupported source cannot block an interactive caller.
    fn try_next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        Err(SourceError::Read {
            reason: "this source does not support non-blocking frame polling".into(),
        })
    }

    /// Whether this source delivers GPU-resident frames.
    ///
    /// When `true`, [`next_frame`](Self::next_frame) may return
    /// `FrameSet::GpuResident` or `FrameSet::MetalResident`.
    /// The session uses this to choose the appropriate rendering and lookahead path.
    fn is_gpu_resident(&self) -> bool {
        false
    }

    /// Plane layout used to configure the GPU renderer before reading images.
    ///
    /// Host NV12 sources must override the YUV420P default. Native sources use NV12 plane bindings; [`Self::gpu_pixel_format`] distinguishes their sample depth. The layout must remain fixed for the source's lifetime.
    #[cfg(feature = "gpu")]
    fn input_format(&self) -> crate::render::renderer::InputFormat {
        if self.is_gpu_resident() {
            crate::render::renderer::InputFormat::Nv12
        } else {
            crate::render::renderer::InputFormat::Yuv420p
        }
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

    /// YCbCr matrix of the delivered source planes. Untagged input defaults to BT.709.
    fn yuv_matrix(&self) -> crate::render::color::YuvMatrix {
        crate::render::color::YuvMatrix::Bt709
    }

    /// Matrix, range and sample storage of the frames supplied by this source.
    fn yuv_color(&self) -> crate::render::color::YuvColor {
        use crate::render::color::YuvColor;
        #[cfg(feature = "gpu")]
        if self.is_gpu_resident()
            && self.gpu_pixel_format() == crate::render::renderer::GpuPixelFormat::P010
        {
            return YuvColor {
                full_range: self.is_full_range(),
                samples: crate::render::color::YuvSampleFormat::P010,
                matrix: self.yuv_matrix(),
            };
        }
        YuvColor {
            full_range: self.is_full_range(),
            matrix: self.yuv_matrix(),
            ..Default::default()
        }
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
    /// The default implementation reads and drops each frame, which
    /// releases any native image leases it holds.
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

    /// Estimated total number of frame sets in the source, if known.
    ///
    /// File-based sources should return [`SourceInfo::total_frames`]. Live sources return `None`. This is not an end-of-stream boundary.
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
    fn unsupported_polling_never_calls_blocking_read() {
        struct BlockingSource;
        impl FrameSource for BlockingSource {
            fn info(&self) -> SourceInfo {
                panic!("polling must not query metadata")
            }
            fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
                panic!("polling must not invoke a blocking read")
            }
        }
        assert!(BlockingSource.try_next_frame().is_err());
    }

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
