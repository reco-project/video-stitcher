//! Object detection trait for raw camera frames.
//!
//! Detectors run on raw (pre-stitch) camera frames to find objects of interest
//! (e.g. a ball). Detections are mapped to panorama-space coordinates and fed
//! to a [`crate::director::Director`] for panning decisions.
//!
//! ## Why Raw Frames?
//!
//! The stitched panorama is an L-shaped 3D projection, not a flat image.
//! Object detection models (YOLO, etc.) work on standard 2D images, so
//! they must run on the original camera frames before stitching.
//! The slight wide-angle distortion is negligible for detection accuracy.

/// Which camera produced this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CameraId {
    /// Left camera (plane in X-Z space).
    Left,
    /// Right camera (plane in X-Y space).
    Right,
}

impl std::fmt::Display for CameraId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Left => f.write_str("L"),
            Self::Right => f.write_str("R"),
        }
    }
}

/// Raw camera frame data for detection.
///
/// Provides access to all YUV planes so detectors can use luma-only
/// (fast, sufficient for ball tracking) or full color (needed for
/// jersey classification, field segmentation, etc.).
pub struct RawFrame<'a> {
    /// Y (luma) plane, full resolution (`width x height` bytes).
    pub y: &'a [u8],
    /// Chroma plane data (format depends on the decode pipeline).
    pub chroma: ChromaFormat<'a>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

/// Chroma plane layout.
///
/// The format matches whatever the decode pipeline produces:
/// - Software decode (FFmpeg): YUV420P with separate U and V planes
/// - Hardware decode (NVDEC, V4L2): NV12 with interleaved UV
pub enum ChromaFormat<'a> {
    /// YUV420P: separate half-resolution U and V planes.
    Yuv420p {
        /// U (Cb) plane, `(width/2) x (height/2)` bytes.
        u: &'a [u8],
        /// V (Cr) plane, `(width/2) x (height/2)` bytes.
        v: &'a [u8],
    },
    /// NV12: interleaved UV plane, `width x (height/2)` bytes.
    Nv12 {
        /// Interleaved U,V data.
        uv: &'a [u8],
    },
}

/// A detected object in a raw camera frame.
///
/// Coordinates are in normalized image space `[0.0, 1.0]` relative to
/// the frame dimensions. Use [`crate::projection::camera_to_panorama`]
/// to map these to panoramic yaw/pitch.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Detection {
    /// Which camera this detection came from.
    pub camera: CameraId,

    /// Detection class index from the model (e.g. 0 = "ball", 1 = "person").
    /// Map to a human-readable label via the detector's `class_names()`.
    pub class_id: u16,

    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,

    /// Bounding box center X in normalized image coordinates `[0.0, 1.0]`.
    pub center_x: f32,

    /// Bounding box center Y in normalized image coordinates `[0.0, 1.0]`.
    pub center_y: f32,

    /// Bounding box width in normalized image coordinates.
    pub width: f32,

    /// Bounding box height in normalized image coordinates.
    pub height: f32,
}

/// Trait for object detection on raw camera frames.
///
/// A frame-level analyzer that may maintain internal state across calls
/// (e.g. tracking state, warmup flags). Implementations should be
/// async-friendly - detection may run on a separate thread or even a
/// different device (e.g. a Jetson's DLA).
///
/// # Frame Data
///
/// The [`RawFrame`] contains the full YUV data of a single camera frame
/// (before any stitching or undistortion). Most detection models only
/// need the Y (luma) plane for grayscale inference, but full color is
/// available for tasks like jersey classification.
pub trait Detector: Send {
    /// Run detection on a raw camera frame.
    ///
    /// Returns a list of detections found in the frame. May return an
    /// empty vector if no objects are detected.
    fn detect(&mut self, camera: CameraId, frame: &RawFrame<'_>) -> Vec<Detection>;
}

/// A GPU-resident NV12 frame described by CUDA device pointers.
///
/// Wraps the raw pointer/pitch/dimension parameters needed to locate the
/// Y and UV planes of an NV12 frame in GPU memory. Passed by reference
/// to [`GpuDetector::detect_gpu`] instead of many loose arguments.
#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug, Clone, Copy)]
pub struct GpuNv12Frame {
    /// CUDA device pointer to the Y (luma) plane.
    pub y_ptr: u64,
    /// CUDA device pointer to the UV (chroma) plane.
    pub uv_ptr: u64,
    /// Row pitch in bytes for the Y plane.
    pub y_pitch: usize,
    /// Row pitch in bytes for the UV plane.
    pub uv_pitch: usize,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Camera rotation from stream metadata (0, 90, 180, 270 degrees).
    ///
    /// In the GPU zero-copy path, NVDEC decodes without applying rotation
    /// metadata. The rendering shader flips UV coordinates so the display
    /// is correct, but the detector receives raw upside-down frames. When
    /// `rotation == 180`, the detector must flip the frame during
    /// preprocessing so detection models see correctly oriented images.
    pub rotation: i32,
    /// Whether this frame uses P010 (10-bit NV12) pixel format.
    ///
    /// P010 stores 10-bit luma/chroma values in the upper 10 bits of each
    /// `u16` sample. Detectors that expect 8-bit NV12 (e.g. NPP's
    /// `nppiNV12ToRGB_8u_P2C3R`) must convert P010 to 8-bit first by
    /// right-shifting each sample by 8 bits.
    pub is_10bit: bool,
}

/// Trait for object detection on GPU-resident NV12 frames.
///
/// A frame-level analyzer that may maintain internal state across calls.
/// Unlike [`Detector`] which operates on CPU-accessible [`RawFrame`] data,
/// this trait takes CUDA device pointers directly. Used in the zero-copy
/// pipeline where decoded frames never leave the GPU.
///
/// Implementations must handle GPU-side preprocessing (NV12-to-RGB
/// conversion, resize, normalization) and inference entirely on the GPU,
/// reading back only the small detection output.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub trait GpuDetector: Send {
    /// Run detection on a GPU-resident NV12 frame.
    ///
    /// The [`GpuNv12Frame`] contains CUDA device pointers to the Y and
    /// interleaved UV planes, their row pitches (may differ from width
    /// due to alignment), and frame dimensions.
    fn detect_gpu(&mut self, camera: CameraId, frame: &GpuNv12Frame) -> Vec<Detection>;
}

/// Trait for object detection on Metal-resident NV12 frames (macOS).
///
/// A frame-level analyzer that may maintain internal state across calls.
/// Unlike [`GpuDetector`] which operates on CUDA device pointers,
/// this trait takes `CVPixelBufferRef` pointers from VideoToolbox.
/// Used in the macOS zero-copy pipeline where decoded frames are
/// IOSurface-backed and can be imported as Metal textures.
///
/// Implementations must handle Metal-side preprocessing (NV12-to-RGB
/// conversion, resize, normalization via compute shaders) and inference,
/// reading back only the small detection output.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub trait MetalDetector: Send {
    /// Run detection on a Metal-resident NV12 frame.
    ///
    /// `cv_pixel_buffer` is a `CVPixelBufferRef` from VideoToolbox decode,
    /// backed by an IOSurface. The implementor imports it as Metal textures
    /// via `CVMetalTextureCache` for GPU-side preprocessing.
    ///
    /// `gpu` is needed for importing the CVPixelBuffer planes as Metal
    /// textures via `CVMetalTextureCache`.
    fn detect_metal(
        &mut self,
        camera: CameraId,
        cv_pixel_buffer: crate::metal_interop::CVPixelBufferRef,
        width: u32,
        height: u32,
        gpu: &crate::gpu::GpuContext,
    ) -> Vec<Detection>;
}
