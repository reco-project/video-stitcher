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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraId {
    /// Left camera (plane in X-Z space).
    Left,
    /// Right camera (plane in Y-Z space).
    Right,
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
#[derive(Debug, Clone)]
pub struct Detection {
    /// Which camera this detection came from.
    pub camera: CameraId,

    /// Detection class label (e.g. "ball", "player").
    pub label: String,

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
/// Implementations should be async-friendly - detection may run on a
/// separate thread or even a different device (e.g. a Jetson's DLA).
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
