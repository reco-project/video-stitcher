//! Value types shared by every pose-resolution path.
//!
//! [`ViewportPosition`] is the camera's yaw / pitch / FOV triple, the
//! output of any [`Panner`](crate::panner::Panner) and the input the
//! renderer crops the panorama with. [`MappedDetection`] is a raw
//! detection enriched with panorama-space coordinates; trackers
//! consume it and emit [`TrackedEntity`](crate::tracker::TrackedEntity)
//! values, and external detection sinks can observe it directly via
//! [`StitchSession::set_detection_sink`](crate::session::StitchSession::set_detection_sink)
//! without going through a tracker.
//!
//! The module is named `director` for historical reasons — the old
//! `Director` trait lived here before the tracker/panner split. The
//! trait is gone; only these value types remain. Rename deferred to
//! avoid a repo-wide import churn.

use crate::detector::CameraId;

/// The viewport position output by a director.
///
/// Specifies the yaw, pitch, and field of view of the virtual camera.
/// The FOV allows directors to express zoom: narrow FOV = zoomed in on
/// action, wide FOV = zoomed out for context.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ViewportPosition {
    /// Horizontal pan angle in radians.
    ///
    /// `0.0` = centered on the seam between cameras.
    pub yaw: f32,

    /// Vertical tilt angle in radians.
    ///
    /// `0.0` = level. Positive = looking up.
    pub pitch: f32,

    /// Field of view in degrees, or `None` to use the pipeline's
    /// default FOV.
    ///
    /// Typical range: 30.0 (zoomed in) to 120.0 (wide). The pipeline
    /// default is 75.0.
    pub fov_degrees: Option<f32>,
}

impl Default for ViewportPosition {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: None,
        }
    }
}

/// A detection mapped to panorama coordinates.
///
/// Consumed by every [`Tracker`](crate::tracker::Tracker) each frame
/// and by external detection sinks (coaching, VAR, stats) via
/// [`StitchSession::set_detection_sink`](crate::session::StitchSession::set_detection_sink).
/// Wraps a raw camera-space detection with a panorama-space
/// [`ViewportPosition`] computed via
/// [`camera_to_panorama`](crate::projection::camera_to_panorama).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MappedDetection {
    /// Which camera this detection came from.
    pub camera: CameraId,

    /// Detection class index from the model (e.g. 0 = "ball", 1 = "person").
    /// Map to a human-readable label via the detector's `class_names()`.
    pub class_id: u16,

    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,

    /// Bounding box center in normalized camera coordinates `[0.0, 1.0]`.
    pub camera_center: (f32, f32),

    /// Bounding box size in normalized `[0, 1]` camera coordinates.
    ///
    /// Multiply by the camera's pixel dimensions to get pixel size:
    /// `pixel_w = camera_size.0 * calibration.left.width as f32`.
    pub camera_size: (f32, f32),

    /// Position in panorama space (yaw/pitch).
    /// `None` if the detection couldn't be mapped (e.g. outside camera FOV).
    pub position: Option<ViewportPosition>,
}
