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

/// A detected object in a raw camera frame.
///
/// Coordinates are in normalized image space `[0.0, 1.0]` relative to
/// the frame dimensions. The director is responsible for mapping these
/// to panorama-space yaw/pitch using the calibration data.
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
/// Implementations should be async-friendly — detection may run on a
/// separate thread or even a different device (e.g. a Jetson's DLA).
///
/// # Frame Data
///
/// The `frame_data` parameter is the raw pixel data of a single camera
/// frame (before any stitching or undistortion). The format depends on
/// the decode pipeline (typically NV12 or RGB).
pub trait Detector: Send {
    /// Run detection on a raw camera frame.
    ///
    /// Returns a list of detections found in the frame. May return an
    /// empty vector if no objects are detected.
    fn detect(
        &mut self,
        camera: CameraId,
        frame_data: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<Detection>;
}
