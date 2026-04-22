//! ROI-filtered detector decorator.
//!
//! Wraps any [`UnifiedDetector`] to drop detections whose camera-space
//! position falls outside a playing-field polygon. This moves
//! sports-domain filtering out of `reco-core` (which stays domain-
//! agnostic) into `reco-autocam`.
//!
//! Per-class anchor policy (Step 7c):
//!
//! - [`RoiAnchor::Center`] (default) tests the bounding-box center.
//!   Appropriate for balls and any "the detection is a point-shaped
//!   target" class.
//! - [`RoiAnchor::Bottom`] tests the feet (bottom center) AND the
//!   75th-percentile height. Both must be inside the polygon. Meant
//!   for upright classes (players, refs) so a coach leaning in from
//!   the sideline whose feet are just inside the boundary but whose
//!   body is mostly outside gets rejected.
//! - [`RoiAnchor::Top`] is the upside-down mirror of `Bottom`.
//!
//! One decorator covers every residency (CPU / CUDA / Metal) because
//! [`UnifiedDetector`] dispatches internally on the
//! [`DetectorFrame`] variant the backend accepts.

use std::collections::HashMap;

use reco_core::calibration::FieldRoi;
use reco_core::detector::{CameraId, Detection, DetectorError, DetectorFrame, UnifiedDetector};
use reco_core::projection::point_in_polygon;

/// Where on a detection's bounding box the ROI test samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoiAnchor {
    /// Sample only `(center_x, center_y)`. Default, appropriate for
    /// balls and other point-shaped targets.
    #[default]
    Center,
    /// Sample the feet `(center_x, center_y + height/2)` AND the
    /// 75th-percentile height `(center_x, center_y + height/4)`.
    /// Both must be inside the polygon. Meant for upright classes.
    Bottom,
    /// Mirror of [`Bottom`](RoiAnchor::Bottom) for upside-down
    /// captures or ceiling-mounted cameras.
    Top,
}

impl RoiAnchor {
    /// Return `true` if the detection passes the anchor test against
    /// `polygon`. A polygon with fewer than 3 vertices is treated as
    /// "no filter" by the caller; this method does not re-check that
    /// case.
    fn passes(self, d: &Detection, polygon: &[[f64; 2]]) -> bool {
        let cx = d.center_x as f64;
        let cy = d.center_y as f64;
        let half_h = (d.height as f64) * 0.5;
        let quarter_h = (d.height as f64) * 0.25;
        match self {
            RoiAnchor::Center => point_in_polygon([cx, cy], polygon),
            RoiAnchor::Bottom => {
                point_in_polygon([cx, cy + half_h], polygon)
                    && point_in_polygon([cx, cy + quarter_h], polygon)
            }
            RoiAnchor::Top => {
                point_in_polygon([cx, cy - half_h], polygon)
                    && point_in_polygon([cx, cy - quarter_h], polygon)
            }
        }
    }
}

/// Filter detections by field ROI polygon using a per-class anchor
/// policy. Classes without an explicit entry use `default_anchor`.
fn filter_by_roi(
    detections: Vec<Detection>,
    roi: &FieldRoi,
    class_anchors: &HashMap<u16, RoiAnchor>,
    default_anchor: RoiAnchor,
) -> Vec<Detection> {
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
            let anchor = class_anchors
                .get(&d.class_id)
                .copied()
                .unwrap_or(default_anchor);
            anchor.passes(d, polygon)
        })
        .collect()
}

/// An [`UnifiedDetector`] decorator that filters output detections by
/// field ROI polygon using a per-class anchor policy.
///
/// Wraps an inner detector and drops detections outside the playing
/// field after each `detect()` call. Works for every residency the
/// inner detector supports (CPU / CUDA / Metal / future variants)
/// because the filter runs on the post-inference `Vec<Detection>`.
///
/// Default anchor is [`RoiAnchor::Center`] (good for balls and
/// point-shaped targets); call
/// [`with_class_anchor`](Self::with_class_anchor) to override per
/// class. Common pattern:
///
/// ```rust,ignore
/// RoiFilteredDetector::new(inner, roi)
///     .with_class_anchor(person_class_id, RoiAnchor::Bottom)
/// ```
///
/// `name()` forwards to the inner detector's name so telemetry
/// still identifies the underlying backend; the ROI wrap is invisible
/// to log consumers.
pub struct RoiFilteredDetector {
    inner: Box<dyn UnifiedDetector>,
    roi: FieldRoi,
    class_anchors: HashMap<u16, RoiAnchor>,
    default_anchor: RoiAnchor,
}

impl RoiFilteredDetector {
    /// Create a new ROI-filtered detector wrapping `inner`. All
    /// classes default to [`RoiAnchor::Center`] until
    /// [`with_class_anchor`](Self::with_class_anchor) overrides them.
    pub fn new(inner: Box<dyn UnifiedDetector>, roi: FieldRoi) -> Self {
        Self {
            inner,
            roi,
            class_anchors: HashMap::new(),
            default_anchor: RoiAnchor::Center,
        }
    }

    /// Override the anchor for a specific class id. Chainable.
    pub fn with_class_anchor(mut self, class_id: u16, anchor: RoiAnchor) -> Self {
        self.class_anchors.insert(class_id, anchor);
        self
    }

    /// Override the default anchor for classes without an explicit
    /// [`with_class_anchor`](Self::with_class_anchor) entry. Chainable.
    pub fn with_default_anchor(mut self, anchor: RoiAnchor) -> Self {
        self.default_anchor = anchor;
        self
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
        Ok(filter_by_roi(
            detections,
            &self.roi,
            &self.class_anchors,
            self.default_anchor,
        ))
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
        make_detection_class(camera, 0, cx, cy, w, h)
    }

    fn make_detection_class(
        camera: CameraId,
        class_id: u16,
        cx: f32,
        cy: f32,
        w: f32,
        h: f32,
    ) -> Detection {
        Detection {
            camera,
            class_id,
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

    fn filter(dets: Vec<Detection>, roi: &FieldRoi) -> Vec<Detection> {
        filter_by_roi(dets, roi, &HashMap::new(), RoiAnchor::Center)
    }

    #[test]
    fn detection_inside_roi_passes() {
        let roi = full_roi();
        let det = make_detection(CameraId::Left, 0.5, 0.4, 0.1, 0.2);
        assert_eq!(filter(vec![det], &roi).len(), 1);
    }

    #[test]
    fn detection_outside_roi_filtered() {
        let roi = small_roi();
        let det = make_detection(CameraId::Left, 0.05, 0.05, 0.1, 0.2);
        assert!(filter(vec![det], &roi).is_empty());
    }

    #[test]
    fn empty_polygon_passes_all() {
        let roi = FieldRoi {
            left: vec![],
            right: vec![],
        };
        let det = make_detection(CameraId::Left, 0.5, 0.5, 0.1, 0.2);
        assert_eq!(filter(vec![det], &roi).len(), 1);
    }

    #[test]
    fn degenerate_polygon_passes_all() {
        let roi = FieldRoi {
            left: vec![[0.0, 0.0], [1.0, 1.0]],
            right: vec![],
        };
        let det = make_detection(CameraId::Left, 0.5, 0.5, 0.1, 0.2);
        assert_eq!(filter(vec![det], &roi).len(), 1);
    }

    #[test]
    fn camera_id_selects_correct_polygon() {
        let roi = FieldRoi {
            left: vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            right: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        };
        let det_left = make_detection(CameraId::Left, 0.05, 0.05, 0.1, 0.2);
        let det_right = make_detection(CameraId::Right, 0.05, 0.05, 0.1, 0.2);
        assert!(filter(vec![det_left], &roi).is_empty());
        assert_eq!(filter(vec![det_right], &roi).len(), 1);
    }

    #[test]
    fn mixed_detections_filter_correctly() {
        let roi = small_roi();
        let inside = make_detection(CameraId::Left, 0.5, 0.4, 0.1, 0.2);
        let outside = make_detection(CameraId::Left, 0.05, 0.05, 0.1, 0.2);
        let filtered = filter(vec![inside, outside], &roi);
        assert_eq!(filtered.len(), 1);
        assert!((filtered[0].center_x - 0.5).abs() < f32::EPSILON);
    }

    // ── Step 7c: per-class anchor policy ─────────────────────────

    #[test]
    fn center_anchor_passes_ball_whose_feet_fall_outside() {
        // Small ROI [0.2, 0.8]^2. Detection centered in-bounds at
        // (0.5, 0.7), height 0.3 -> feet at y=0.85 outside, center
        // at y=0.7 inside. A ball class with Center anchor passes.
        let roi = small_roi();
        let ball = make_detection_class(CameraId::Left, 0, 0.5, 0.7, 0.1, 0.3);
        assert_eq!(filter(vec![ball], &roi).len(), 1);
    }

    #[test]
    fn bottom_anchor_rejects_player_whose_feet_fall_outside() {
        // Same geometry as above but as a Player (class=1) with
        // Bottom anchor: feet at y=0.85 outside -> rejected.
        let roi = small_roi();
        let player = make_detection_class(CameraId::Left, 1, 0.5, 0.7, 0.1, 0.3);
        let mut anchors = HashMap::new();
        anchors.insert(1u16, RoiAnchor::Bottom);
        let filtered = filter_by_roi(vec![player], &roi, &anchors, RoiAnchor::Center);
        assert!(filtered.is_empty());
    }

    #[test]
    fn mixed_classes_respect_per_class_policy() {
        // Same frame: a ball and a player at identical bboxes. With
        // Center default + Bottom for players, ball survives and
        // player gets dropped.
        let roi = small_roi();
        let ball = make_detection_class(CameraId::Left, 0, 0.5, 0.7, 0.1, 0.3);
        let player = make_detection_class(CameraId::Left, 1, 0.5, 0.7, 0.1, 0.3);
        let mut anchors = HashMap::new();
        anchors.insert(1u16, RoiAnchor::Bottom);
        let filtered = filter_by_roi(vec![ball, player], &roi, &anchors, RoiAnchor::Center);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].class_id, 0, "only the ball should survive");
    }

    #[test]
    fn with_class_anchor_builder_stores_mapping() {
        // Locks the public-API shape: builder method + lookup via
        // detect() path work together.
        let roi = small_roi();
        let det = make_detection_class(CameraId::Left, 7, 0.5, 0.7, 0.1, 0.3);
        let mut anchors = HashMap::new();
        anchors.insert(7u16, RoiAnchor::Bottom);
        assert!(filter_by_roi(vec![det], &roi, &anchors, RoiAnchor::Center).is_empty());
    }
}
