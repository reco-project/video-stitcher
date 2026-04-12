//! ROI-filtered detector decorators.
//!
//! Wraps any [`Detector`], [`GpuDetector`], or [`MetalDetector`] to drop
//! detections whose camera-space position falls outside a playing field
//! polygon. This moves sports-domain filtering logic out of `reco-core`
//! (which should be domain-agnostic) into `reco-autocam`.
//!
//! The filter tests two points per detection: the feet position (bottom
//! center of the bounding box) and the 75th percentile height (between
//! center and feet). Both must be inside the ROI polygon for the detection
//! to pass. This rejects people whose feet are just inside the boundary
//! but whose body is mostly outside (coaches leaning in, sideline spectators).

use reco_core::calibration::FieldRoi;
use reco_core::detector::{CameraId, Detection, Detector, RawFrame};
use reco_core::projection::point_in_polygon;

/// Filter detections by field ROI polygon.
///
/// For each detection, tests the feet position and 75th-percentile height
/// against the polygon for the detection's camera. Detections outside the
/// polygon are discarded.
fn filter_by_roi(detections: Vec<Detection>, roi: &FieldRoi) -> Vec<Detection> {
    detections
        .into_iter()
        .filter(|d| {
            let polygon = match d.camera {
                CameraId::Left => &roi.left,
                CameraId::Right => &roi.right,
            };
            // Only filter if the polygon has enough vertices.
            if polygon.len() < 3 {
                return true;
            }
            // Test at 75th percentile height of bbox (between center and
            // feet). Both this point AND the feet must be inside the ROI.
            let feet_x = d.center_x as f64;
            let feet_y = (d.center_y + d.height * 0.5) as f64;
            let p75_y = (d.center_y + d.height * 0.25) as f64;
            point_in_polygon([feet_x, feet_y], polygon)
                && point_in_polygon([feet_x, p75_y], polygon)
        })
        .collect()
}

/// A [`Detector`] decorator that filters detections by field ROI.
///
/// Wraps an inner detector and drops detections outside the playing field
/// polygon after each `detect()` call. Pass-through when no polygon is
/// configured for a given camera.
pub struct RoiFilteredDetector {
    inner: Box<dyn Detector>,
    roi: FieldRoi,
}

impl RoiFilteredDetector {
    /// Create a new ROI-filtered detector wrapping `inner`.
    pub fn new(inner: Box<dyn Detector>, roi: FieldRoi) -> Self {
        Self { inner, roi }
    }
}

impl Detector for RoiFilteredDetector {
    fn detect(&mut self, camera: CameraId, frame: &RawFrame<'_>) -> Vec<Detection> {
        let detections = self.inner.detect(camera, frame);
        filter_by_roi(detections, &self.roi)
    }
}

/// A [`GpuDetector`] decorator that filters detections by field ROI.
///
/// Same filtering logic as [`RoiFilteredDetector`], but for GPU-resident
/// NV12 frames in the CUDA zero-copy pipeline.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub struct RoiFilteredGpuDetector {
    inner: Box<dyn reco_core::detector::GpuDetector>,
    roi: FieldRoi,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl RoiFilteredGpuDetector {
    /// Create a new ROI-filtered GPU detector wrapping `inner`.
    pub fn new(inner: Box<dyn reco_core::detector::GpuDetector>, roi: FieldRoi) -> Self {
        Self { inner, roi }
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl reco_core::detector::GpuDetector for RoiFilteredGpuDetector {
    fn detect_gpu(
        &mut self,
        camera: CameraId,
        frame: &reco_core::detector::GpuNv12Frame,
    ) -> Vec<Detection> {
        let detections = self.inner.detect_gpu(camera, frame);
        filter_by_roi(detections, &self.roi)
    }
}

/// A [`MetalDetector`] decorator that filters detections by field ROI.
///
/// Same filtering logic as [`RoiFilteredDetector`], but for Metal-resident
/// NV12 frames in the macOS zero-copy pipeline.
#[cfg(target_os = "macos")]
pub struct RoiFilteredMetalDetector {
    inner: Box<dyn reco_core::detector::MetalDetector>,
    roi: FieldRoi,
}

#[cfg(target_os = "macos")]
impl RoiFilteredMetalDetector {
    /// Create a new ROI-filtered Metal detector wrapping `inner`.
    pub fn new(inner: Box<dyn reco_core::detector::MetalDetector>, roi: FieldRoi) -> Self {
        Self { inner, roi }
    }
}

#[cfg(target_os = "macos")]
impl reco_core::detector::MetalDetector for RoiFilteredMetalDetector {
    fn detect_metal(
        &mut self,
        camera: CameraId,
        cv_pixel_buffer: reco_core::metal_interop::CVPixelBufferRef,
        width: u32,
        height: u32,
        gpu: &reco_core::gpu::GpuContext,
    ) -> Vec<Detection> {
        let detections = self
            .inner
            .detect_metal(camera, cv_pixel_buffer, width, height, gpu);
        filter_by_roi(detections, &self.roi)
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

    /// Unit square ROI: covers [0,0] to [1,1].
    fn full_roi() -> FieldRoi {
        FieldRoi {
            left: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            right: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        }
    }

    /// Small ROI: covers [0.2, 0.2] to [0.8, 0.8].
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
        // Detection near top-left corner, feet at (0.05, 0.15) - outside small ROI.
        let det = make_detection(CameraId::Left, 0.05, 0.05, 0.1, 0.2);
        let filtered = filter_by_roi(vec![det], &roi);
        assert!(
            filtered.is_empty(),
            "detection outside ROI should be filtered"
        );
    }

    #[test]
    fn empty_polygon_passes_all() {
        let roi = FieldRoi {
            left: vec![],
            right: vec![],
        };
        let det = make_detection(CameraId::Left, 0.5, 0.5, 0.1, 0.2);
        let filtered = filter_by_roi(vec![det], &roi);
        assert_eq!(
            filtered.len(),
            1,
            "empty polygon should pass all detections"
        );
    }

    #[test]
    fn degenerate_polygon_passes_all() {
        let roi = FieldRoi {
            left: vec![[0.0, 0.0], [1.0, 1.0]], // only 2 vertices
            right: vec![],
        };
        let det = make_detection(CameraId::Left, 0.5, 0.5, 0.1, 0.2);
        let filtered = filter_by_roi(vec![det], &roi);
        assert_eq!(
            filtered.len(),
            1,
            "polygon with < 3 vertices should pass all"
        );
    }

    #[test]
    fn camera_id_selects_correct_polygon() {
        // Left has a small ROI, right has full ROI.
        let roi = FieldRoi {
            left: vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            right: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        };
        // Detection in the corner - should fail left, pass right.
        let det_left = make_detection(CameraId::Left, 0.05, 0.05, 0.1, 0.2);
        let det_right = make_detection(CameraId::Right, 0.05, 0.05, 0.1, 0.2);
        let filtered_left = filter_by_roi(vec![det_left], &roi);
        let filtered_right = filter_by_roi(vec![det_right], &roi);
        assert!(filtered_left.is_empty(), "left detection outside small ROI");
        assert_eq!(filtered_right.len(), 1, "right detection inside full ROI");
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
