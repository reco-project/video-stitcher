//! Value types shared by every pose-resolution path.
//!
//! [`Pose`] is the camera's yaw / pitch / FOV triple, the
//! output of any [`Panner`](super::panner::Panner) and the input the
//! renderer crops the panorama with. [`MappedDetection`] is a raw
//! detection enriched with panorama-space coordinates; trackers
//! consume it and emit [`TrackedEntity`](super::tracker::TrackedEntity)
//! values. External consumers can observe detections via
//! [`PipelineEventSink`](super::pipeline_event::PipelineEventSink).
//!
//! The module is named `director` for historical reasons — the old
//! `Director` trait lived here before the tracker/panner split. The
//! trait is gone; only these value types remain. Rename deferred to
//! avoid a repo-wide import churn.

use crate::geometry::CameraIndex;
use crate::geometry::Pose;

/// A detection mapped to panorama coordinates.
///
/// Consumed by every [`Tracker`](super::tracker::Tracker) each frame.
/// External consumers (coaching, VAR, stats) observe detections via
/// [`PipelineEventSink`](super::pipeline_event::PipelineEventSink).
/// Wraps a raw camera-space detection with a panorama-space
/// [`Pose`] computed via
/// [`camera_to_panorama`](crate::projection::camera_to_panorama).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MappedDetection {
    /// Which camera this detection came from.
    pub camera: CameraIndex,

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
    pub position: Option<Pose>,
}
