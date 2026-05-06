//! `DetectionFilter` implementations that run in the session's
//! pre-tracker chain.
//!
//! Each filter here is a [`reco_core::detect::filter::DetectionFilter`]
//! impl that operates on the shared `Vec<MappedDetection>` before any
//! tracker sees it. Composability is the hard constraint: same type
//! in, same type out.
//!
//! Shipping today:
//!
//! - [`FlickerDetectionFilter`] — wraps the proven bucketed-spatial
//!   [`FlickerFilter`] and rejects recurrent static-mimic detections
//!   (advertising logos, line intersections) before they reach the
//!   tracker. Class-aware so a ball flicker doesn't penalize a
//!   stationary player at the same camera pixel.

use reco_core::detect::director::MappedDetection;
use reco_core::detect::filter::{DetectionFilter, FilterContext};

use crate::trackers::filters::FlickerFilter;

/// Pre-tracker flicker-rejection filter.
///
/// Wraps [`FlickerFilter`] and implements [`DetectionFilter`] so the
/// existing bucketed-spatial logic runs before any tracker (rather
/// than inside BallTracker, where it used to live). Class-aware via
/// the class-keyed histogram, so it is safe to attach session-wide.
pub struct FlickerDetectionFilter {
    inner: FlickerFilter,
}

impl FlickerDetectionFilter {
    /// Defaults: 2% buckets, 3.3s window, 5 hits. See
    /// [`FlickerFilter::with_defaults`].
    pub fn with_defaults() -> Self {
        Self {
            inner: FlickerFilter::with_defaults(),
        }
    }

    /// Custom parameters (typically only exercised by tests).
    pub fn new(bucket_norm: f32, window_ms: f64, min_hits: u32) -> Self {
        Self {
            inner: FlickerFilter::new(bucket_norm, window_ms, min_hits),
        }
    }
}

impl DetectionFilter for FlickerDetectionFilter {
    fn name(&self) -> &'static str {
        "FlickerFilter"
    }

    fn filter(&mut self, detections: &mut Vec<MappedDetection>, ctx: &FilterContext<'_>) {
        // Record-and-check every detection: even rejections keep the
        // spatial histogram accurate (same rule as the old BallTracker
        // inline path).
        detections.retain(|d| {
            !self
                .inner
                .record_and_check(d.camera, d.class_id, d.camera_center, ctx.timestamp_ms)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::calibration::{CameraParams, MatchCalibration, PlaneLayout};
    use reco_core::detect::detector::CameraId;
    use reco_core::detect::director::MappedDetection;

    fn test_calibration() -> MatchCalibration {
        MatchCalibration {
            left: CameraParams {
                width: 1920,
                height: 1080,
                fx: 900.0,
                fy: 900.0,
                cx: 960.0,
                cy: 540.0,
                d: [0.0; 4],
            },
            right: CameraParams {
                width: 1920,
                height: 1080,
                fx: 900.0,
                fy: 900.0,
                cx: 960.0,
                cy: 540.0,
                d: [0.0; 4],
            },
            layout: PlaneLayout {
                camera_axis_offset: 0.24,
                intersect: 0.54,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
            },
            rig_tilt: 0.0,
            rig_roll: 0.0,
            sync_offset: 0,
            field_roi: None,
        }
    }

    fn mk_det(camera: CameraId, class_id: u16, cx: f32, cy: f32) -> MappedDetection {
        MappedDetection {
            camera,
            class_id,
            confidence: 0.9,
            camera_center: (cx, cy),
            camera_size: (0.05, 0.05),
            position: None,
        }
    }

    #[test]
    fn drops_recurrent_bucket_hits_across_frames() {
        let cal = test_calibration();
        let mut filter = FlickerDetectionFilter::new(0.5, 1_000.0, 2);
        let ctx0 = FilterContext {
            frame_index: 0,
            timestamp_ms: 0.0,
            calibration: &cal,
        };
        let ctx1 = FilterContext {
            frame_index: 1,
            timestamp_ms: 100.0,
            calibration: &cal,
        };

        // bucket_norm=0.5 so the bucket boundaries sit at 0.0, 0.5,
        // 1.0. Both hits in [0.5, 1.0) x [0.5, 1.0) land on bucket (1, 1).
        let mut frame0 = vec![mk_det(CameraId::Left, 0, 0.55, 0.55)];
        filter.filter(&mut frame0, &ctx0);
        assert_eq!(frame0.len(), 1, "first hit survives");

        let mut frame1 = vec![mk_det(CameraId::Left, 0, 0.60, 0.60)];
        filter.filter(&mut frame1, &ctx1);
        assert!(frame1.is_empty(), "second hit in same bucket is dropped");
    }

    #[test]
    fn does_not_cross_classes() {
        let cal = test_calibration();
        let mut filter = FlickerDetectionFilter::new(0.5, 1_000.0, 2);
        let ctx = FilterContext {
            frame_index: 0,
            timestamp_ms: 0.0,
            calibration: &cal,
        };

        // Ball at (0.5, 0.5) + player at the same camera pixel in
        // one frame. Neither has 2 hits yet -> both survive.
        let mut dets = vec![
            mk_det(CameraId::Left, 0, 0.50, 0.50),
            mk_det(CameraId::Left, 1, 0.50, 0.50),
        ];
        filter.filter(&mut dets, &ctx);
        assert_eq!(dets.len(), 2);
    }

    #[test]
    fn filter_name_stable_for_event_sink() {
        let f = FlickerDetectionFilter::with_defaults();
        assert_eq!(f.name(), "FlickerFilter");
    }
}
