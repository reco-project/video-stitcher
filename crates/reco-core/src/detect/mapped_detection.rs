//! Source-image detections with optional calibrated world-reference directions.
//!
//! Trackers consume [`MappedDetection`] values and emit [`TrackedEntity`](super::tracker::TrackedEntity) values. External consumers can observe mapped detections through [`PipelineEventSink`](super::pipeline_event::PipelineEventSink).

use crate::geometry::CameraIndex;
use crate::geometry::YawPitch;

/// A source-image detection with an optional world-reference angular position.
///
/// Consumed by every [`Tracker`](super::tracker::Tracker) each frame.
/// External consumers (coaching, VAR, stats) observe detections via
/// [`PipelineEventSink`](super::pipeline_event::PipelineEventSink).
/// Wraps a raw source-image detection with a world-reference
/// [`YawPitch`] computed via [`Calibration::map_source_point`](crate::calibration::Calibration::map_source_point).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MappedDetection {
    /// Which camera this detection came from.
    pub camera: CameraIndex,

    /// Detection class index from the model (e.g. 0 = "ball", 1 = "person").
    /// Map to a human-readable label via the detector's `class_names()`.
    pub class_id: u16,

    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,

    /// Bounding box center `(x, y)` as fractions of the source image. The origin is top-left, with X right and Y down.
    pub camera_center: (f32, f32),

    /// Bounding box `(width, height)` as fractions of the source image dimensions.
    ///
    /// Multiply by the camera's pixel dimensions to get pixel size:
    /// `pixel_w = camera_size.0 * calibration.lenses[camera].width as f32`.
    pub camera_size: (f32, f32),

    /// World-reference angular position (yaw/pitch, before rig tilt/roll orientation).
    /// `None` if mapping is unsupported or the source point cannot be mapped.
    pub position: Option<YawPitch>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_position_serializes_without_an_invented_fov() {
        let detection = MappedDetection {
            camera: 0,
            class_id: 1,
            confidence: 0.9,
            camera_center: (0.5, 0.5),
            camera_size: (0.1, 0.2),
            position: Some(YawPitch {
                yaw: 0.25,
                pitch: -0.1,
            }),
        };
        let json = serde_json::to_value(detection).unwrap();
        assert_eq!(json["position"]["yaw"], 0.25);
        assert_eq!(
            json["position"]["pitch"],
            serde_json::to_value(-0.1_f32).unwrap()
        );
        assert!(json["position"].get("fov_degrees").is_none());
    }
}
