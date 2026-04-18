//! ROI-filtered detector decorator.
//!
//! Wraps any [`UnifiedDetector`] to drop detections whose camera-space
//! position falls outside a playing-field polygon. This moves
//! sports-domain filtering out of `reco-core` (which stays domain-
//! agnostic) into `reco-autocam`.
//!
//! The filter tests two points per detection: the feet position
//! (bottom center of the bounding box) and the 75th-percentile height
//! (between center and feet). Both must be inside the ROI polygon for
//! the detection to pass. This rejects people whose feet are just
//! inside the boundary but whose body is mostly outside (coaches
//! leaning in, sideline spectators).
//!
//! One decorator covers every residency (CPU / CUDA / Metal) because
//! [`UnifiedDetector`] dispatches internally on the
//! [`DetectorFrame`] variant the backend accepts. Prior to the M3
//! trait-collapse commit, there were three separate wrapper structs
//! here (one per legacy trait); all fused into this single type.

use reco_core::calibration::FieldRoi;
use reco_core::detector::{CameraId, Detection, DetectorError, DetectorFrame, UnifiedDetector};
use reco_core::projection::point_in_polygon;

/// Filter detections by field ROI polygon.
///
/// For each detection, tests the feet position and the 75th-percentile
/// height against the polygon for that camera. Detections outside the
/// polygon are discarded. Polygons with fewer than 3 vertices are
/// treated as "no filter" and pass everything (so a half-configured
/// ROI does not silently drop all detections).
fn filter_by_roi(detections: Vec<Detection>, roi: &FieldRoi) -> Vec<Detection> {
    detections
        .into_iter()
        .filter(|d| {
            let polygon = match d.camera {
                CameraId::Left => &roi.left,
                CameraId::Right => &roi.right,
            };
            if polygon.len() < 3 {
                return true;
            }
            let feet_x = d.center_x as f64;
            let feet_y = (d.center_y + d.height * 0.5) as f64;
            let p75_y = (d.center_y + d.height * 0.25) as f64;
            point_in_polygon([feet_x, feet_y], polygon)
                && point_in_polygon([feet_x, p75_y], polygon)
        })
        .collect()
}

/// An [`UnifiedDetector`] decorator that filters output detections by
/// field ROI polygon.
///
/// Wraps an inner detector and drops detections outside the playing
/// field after each `detect()` call. Works for every residency the
/// inner detector supports (CPU / CUDA / Metal / future variants)
/// because the filter runs on the post-inference `Vec<Detection>`.
///
/// `name()` forwards to the inner detector's name so telemetry
/// still identifies the underlying backend; the ROI wrap is invisible
/// to log consumers.
pub struct RoiFilteredDetector {
    inner: Box<dyn UnifiedDetector>,
    roi: FieldRoi,
}

impl RoiFilteredDetector {
    /// Create a new ROI-filtered detector wrapping `inner`.
    pub fn new(inner: Box<dyn UnifiedDetector>, roi: FieldRoi) -> Self {
        Self { inner, roi }
    }
}

impl UnifiedDetector for RoiFilteredDetector {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn detect(
        &mut self,
        camera: CameraId,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let detections = self.inner.detect(camera, frame)?;
        Ok(filter_by_roi(detections, &self.roi))
    }

    fn class_names(&self) -> Option<&[String]> {
        self.inner.class_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::calibration::FieldRoi;
    use reco_core::detector::{CameraId, Detection};

    fn make_detection(camera: CameraId, cx: f32, cy: f32, w: f32, h: f32) -> Detection {
        Detection {
            camera,
            class_id: 0,
            confidence: 0.9,
            center_x: cx,
            center_y: cy,
            width: w,
            height: h,
        }
    }

    fn full_roi() -> FieldRoi {
        FieldRoi {
            left: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            right: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        }
    }

    fn small_roi() -> FieldRoi {
        FieldRoi {
            left: vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            right: vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        }
    }

    #[test]
    fn detection_inside_roi_passes() {
        let roi = full_roi();
        let det = make_detection(CameraId::Left, 0.5, 0.4, 0.1, 0.2);
        let filtered = filter_by_roi(vec![det], &roi);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn detection_outside_roi_filtered() {
        let roi = small_roi();
        let det = make_detection(CameraId::Left, 0.05, 0.05, 0.1, 0.2);
        let filtered = filter_by_roi(vec![det], &roi);
        assert!(filtered.is_empty());
    }

    #[test]
    fn empty_polygon_passes_all() {
        let roi = FieldRoi {
            left: vec![],
            right: vec![],
        };
        let det = make_detection(CameraId::Left, 0.5, 0.5, 0.1, 0.2);
        let filtered = filter_by_roi(vec![det], &roi);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn degenerate_polygon_passes_all() {
        let roi = FieldRoi {
            left: vec![[0.0, 0.0], [1.0, 1.0]],
            right: vec![],
        };
        let det = make_detection(CameraId::Left, 0.5, 0.5, 0.1, 0.2);
        let filtered = filter_by_roi(vec![det], &roi);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn camera_id_selects_correct_polygon() {
        let roi = FieldRoi {
            left: vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            right: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        };
        let det_left = make_detection(CameraId::Left, 0.05, 0.05, 0.1, 0.2);
        let det_right = make_detection(CameraId::Right, 0.05, 0.05, 0.1, 0.2);
        let filtered_left = filter_by_roi(vec![det_left], &roi);
        let filtered_right = filter_by_roi(vec![det_right], &roi);
        assert!(filtered_left.is_empty());
        assert_eq!(filtered_right.len(), 1);
    }

    #[test]
    fn mixed_detections_filter_correctly() {
        let roi = small_roi();
        let inside = make_detection(CameraId::Left, 0.5, 0.4, 0.1, 0.2);
        let outside = make_detection(CameraId::Left, 0.05, 0.05, 0.1, 0.2);
        let filtered = filter_by_roi(vec![inside, outside], &roi);
        assert_eq!(filtered.len(), 1);
        assert!((filtered[0].center_x - 0.5).abs() < f32::EPSILON);
    }
}
