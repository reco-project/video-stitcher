//! Playing-field ROI filtering for completed detections and synchronous detectors.
//!
//! [`RoiFilter`] applies the same camera-space polygon and per-class anchor policy to either input. It does not map coordinates or alter measurement identity, timing or freshness.
//!
//! Per-class anchor policy:
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
//! [`Detector`] dispatches internally on the
//! [`DetectorFrame`] variant the backend accepts.

use std::collections::HashMap;

use reco_core::calibration::FieldRoi;
use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
use reco_core::geometry::CameraIndex;
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

/// Playing-field polygons and the bounding-box anchors tested against them.
///
/// Apply this policy after inference and before source-to-world mapping. Empty or degenerate polygons, and cameras other than the pair described by [`FieldRoi`], leave their detections unchanged. This filters spatial support, not detection confidence or measurement age.
///
/// ```
/// use reco_autocam::{RoiAnchor, RoiFilter};
/// use reco_core::{calibration::FieldRoi, detect::detector::Detection};
///
/// # let roi = FieldRoi::default();
/// # let mut detections: Vec<Detection> = Vec::new();
/// let filter = RoiFilter::new(roi).with_class_anchor(0, RoiAnchor::Bottom);
/// filter.retain(&mut detections);
/// ```
pub struct RoiFilter {
    roi: FieldRoi,
    class_anchors: HashMap<u16, RoiAnchor>,
    default_anchor: RoiAnchor,
}

impl RoiFilter {
    /// Use center anchors for all classes in the supplied camera-space polygons.
    pub fn new(roi: FieldRoi) -> Self {
        Self {
            roi,
            class_anchors: HashMap::new(),
            default_anchor: RoiAnchor::Center,
        }
    }

    /// Override the anchor for a specific class ID.
    pub fn with_class_anchor(mut self, class_id: u16, anchor: RoiAnchor) -> Self {
        self.class_anchors.insert(class_id, anchor);
        self
    }

    /// Set the anchor for classes without an explicit override.
    pub fn with_default_anchor(mut self, anchor: RoiAnchor) -> Self {
        self.default_anchor = anchor;
        self
    }

    /// Retain detections inside their camera's polygon, preserving their order and allocation.
    pub fn retain(&self, detections: &mut Vec<Detection>) {
        detections.retain(|d| {
            let polygon = match d.camera {
                0 => &self.roi.left,
                1 => &self.roi.right,
                _ => return true,
            };
            if polygon.len() < 3 {
                return true;
            }
            let anchor = self
                .class_anchors
                .get(&d.class_id)
                .copied()
                .unwrap_or(self.default_anchor);
            anchor.passes(d, polygon)
        });
    }
}

/// A [`Detector`] decorator applying a [`RoiFilter`] after successful inference.
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
    inner: Box<dyn Detector>,
    filter: RoiFilter,
}

impl RoiFilteredDetector {
    /// Wrap `inner` with the supplied post-inference policy.
    pub fn new(inner: Box<dyn Detector>, filter: RoiFilter) -> Self {
        Self { inner, filter }
    }
}

impl Detector for RoiFilteredDetector {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn prefers_cuda_nv12(&self) -> bool {
        self.inner.prefers_cuda_nv12()
    }

    fn detect(
        &mut self,
        camera: CameraIndex,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let mut detections = self.inner.detect(camera, frame)?;
        self.filter.retain(&mut detections);
        Ok(detections)
    }

    fn class_names(&self) -> Option<&[String]> {
        self.inner.class_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::calibration::FieldRoi;
    use reco_core::detect::detector::Detection;
    use reco_core::geometry::CameraIndex;

    fn make_detection(camera: CameraIndex, cx: f32, cy: f32, w: f32, h: f32) -> Detection {
        make_detection_class(camera, 0, cx, cy, w, h)
    }

    fn make_detection_class(
        camera: CameraIndex,
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

    #[test]
    fn roi_preserves_native_input_selection_independently_of_name() {
        struct NativeInput(bool);
        impl Detector for NativeInput {
            fn name(&self) -> &'static str {
                "arbitrary-display-name"
            }
            fn prefers_cuda_nv12(&self) -> bool {
                self.0
            }
            fn detect(
                &mut self,
                _: CameraIndex,
                _: &DetectorFrame<'_>,
            ) -> Result<Vec<Detection>, DetectorError> {
                Ok(Vec::new())
            }
        }
        for native in [false, true] {
            let filtered =
                RoiFilteredDetector::new(Box::new(NativeInput(native)), RoiFilter::new(full_roi()));
            assert_eq!(filtered.prefers_cuda_nv12(), native);
        }
    }

    fn filter(mut dets: Vec<Detection>, roi: &FieldRoi) -> Vec<Detection> {
        RoiFilter::new(roi.clone()).retain(&mut dets);
        dets
    }

    #[test]
    fn detection_inside_roi_passes() {
        let roi = full_roi();
        let det = make_detection(0, 0.5, 0.4, 0.1, 0.2);
        assert_eq!(filter(vec![det], &roi).len(), 1);
    }

    #[test]
    fn detection_outside_roi_filtered() {
        let roi = small_roi();
        let det = make_detection(0, 0.05, 0.05, 0.1, 0.2);
        assert!(filter(vec![det], &roi).is_empty());
    }

    #[test]
    fn empty_polygon_passes_all() {
        let roi = FieldRoi {
            left: vec![],
            right: vec![],
        };
        let det = make_detection(0, 0.5, 0.5, 0.1, 0.2);
        assert_eq!(filter(vec![det], &roi).len(), 1);
    }

    #[test]
    fn degenerate_polygon_passes_all() {
        let roi = FieldRoi {
            left: vec![[0.0, 0.0], [1.0, 1.0]],
            right: vec![],
        };
        let det = make_detection(0, 0.5, 0.5, 0.1, 0.2);
        assert_eq!(filter(vec![det], &roi).len(), 1);
    }

    #[test]
    fn camera_id_selects_correct_polygon() {
        let roi = FieldRoi {
            left: vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            right: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        };
        let det_left = make_detection(0, 0.05, 0.05, 0.1, 0.2);
        let det_right = make_detection(1, 0.05, 0.05, 0.1, 0.2);
        assert!(filter(vec![det_left], &roi).is_empty());
        assert_eq!(filter(vec![det_right], &roi).len(), 1);
    }

    #[test]
    fn mixed_detections_filter_correctly() {
        let roi = small_roi();
        let inside = make_detection(0, 0.5, 0.4, 0.1, 0.2);
        let outside = make_detection(0, 0.05, 0.05, 0.1, 0.2);
        let filtered = filter(vec![inside, outside], &roi);
        assert_eq!(filtered.len(), 1);
        assert!((filtered[0].center_x - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn center_anchor_passes_ball_whose_feet_fall_outside() {
        // Small ROI [0.2, 0.8]^2. Detection centered in-bounds at
        // (0.5, 0.7), height 0.3 -> feet at y=0.85 outside, center
        // at y=0.7 inside. A ball class with Center anchor passes.
        let roi = small_roi();
        let ball = make_detection_class(0, 0, 0.5, 0.7, 0.1, 0.3);
        assert_eq!(filter(vec![ball], &roi).len(), 1);
    }

    #[test]
    fn bottom_anchor_rejects_player_whose_feet_fall_outside() {
        // Same geometry as above but as a Player (class=1) with
        // Bottom anchor: feet at y=0.85 outside -> rejected.
        let roi = small_roi();
        let player = make_detection_class(0, 1, 0.5, 0.7, 0.1, 0.3);
        let mut filtered = vec![player];
        RoiFilter::new(roi)
            .with_class_anchor(1, RoiAnchor::Bottom)
            .retain(&mut filtered);
        assert!(filtered.is_empty());
    }

    #[test]
    fn mixed_classes_respect_per_class_policy() {
        // Same frame: a ball and a player at identical bboxes. With
        // Center default + Bottom for players, ball survives and
        // player gets dropped.
        let roi = small_roi();
        let ball = make_detection_class(0, 0, 0.5, 0.7, 0.1, 0.3);
        let player = make_detection_class(0, 1, 0.5, 0.7, 0.1, 0.3);
        let mut filtered = vec![ball, player];
        RoiFilter::new(roi)
            .with_class_anchor(1, RoiAnchor::Bottom)
            .retain(&mut filtered);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].class_id, 0, "only the ball should survive");
    }

    #[test]
    fn with_class_anchor_builder_stores_mapping() {
        let roi = small_roi();
        let det = make_detection_class(0, 7, 0.5, 0.7, 0.1, 0.3);
        let mut detections = vec![det];
        RoiFilter::new(roi)
            .with_class_anchor(7, RoiAnchor::Bottom)
            .retain(&mut detections);
        assert!(detections.is_empty());
    }
}
