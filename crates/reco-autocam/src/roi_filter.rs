//! Calibration-aware image support for completed detections, independent of frame residency.

use reco_core::calibration::{Calibration, CalibrationError};
use reco_core::detect::{DetectionFilter, detector::Detection};
use reco_core::projection::{SourcePoint, field::FieldBoundary};
use std::collections::HashMap;

/// Which points of a source-image detection must lie inside the selected field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoiAnchor {
    /// Bounding-box centre, suitable for point-like targets such as balls.
    #[default]
    Center,
    /// Bottom centre and the point three quarters down the box, suitable for upright players.
    Bottom,
    /// Top centre and the point one quarter down the box, for upside-down inputs.
    Top,
}

impl RoiAnchor {
    fn passes(self, detection: &Detection, boundary: &FieldBoundary) -> bool {
        let center = f64::from(detection.center_y);
        let height = f64::from(detection.height);
        let ys = match self {
            Self::Center => [center; 2],
            Self::Bottom => [center + height * 0.5, center + height * 0.25],
            Self::Top => [center - height * 0.5, center - height * 0.25],
        };
        let point = |y| SourcePoint {
            camera: detection.camera,
            position: [f64::from(detection.center_x), y],
        };
        boundary.owns(point(ys[0])) && ys.into_iter().all(|y| boundary.contains(point(y)))
    }
}

/// Field filtering against the accepted calibration, with explicit per-class box anchors.
///
/// The derived mapping is reused until the document changes. No ROI means no filtering. Invalid field geometry returns an error, never an implicit unfiltered fallback. Retention preserves ordering, metadata and the detection vector's allocation.
#[derive(Default)]
pub struct RoiFilter {
    cached: Option<(Calibration, Option<FieldBoundary>)>,
    class_anchors: HashMap<u16, RoiAnchor>,
    default_anchor: RoiAnchor,
}

impl RoiFilter {
    /// Override the default bounding-box anchor for one class.
    pub fn with_class_anchor(mut self, class: u16, anchor: RoiAnchor) -> Self {
        self.class_anchors.insert(class, anchor);
        self
    }

    /// Select the anchor used for classes without an explicit override.
    pub fn with_default_anchor(mut self, anchor: RoiAnchor) -> Self {
        self.default_anchor = anchor;
        self
    }

    /// Filter a completed batch using the current document, rebuilding only changed geometry.
    pub fn retain(
        &mut self,
        calibration: &Calibration,
        detections: &mut Vec<Detection>,
    ) -> Result<(), CalibrationError> {
        if !self
            .cached
            .as_ref()
            .is_some_and(|(previous, _)| previous == calibration)
        {
            let boundary = calibration.field_boundary()?;
            self.cached = Some((calibration.clone(), boundary));
        }
        if let Some((_, Some(boundary))) = &self.cached {
            detections.retain(|d| {
                self.class_anchors
                    .get(&d.class_id)
                    .copied()
                    .unwrap_or(self.default_anchor)
                    .passes(d, boundary)
            });
        }
        Ok(())
    }
}

impl DetectionFilter for RoiFilter {
    fn retain(
        &mut self,
        calibration: &Calibration,
        detections: &mut Vec<Detection>,
    ) -> Result<(), CalibrationError> {
        self.retain(calibration, detections)
    }
}
