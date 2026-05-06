//! Online flicker-rejection filter.
//!
//! In real-match footage, the ball detector fires repeatedly on
//! fixed-location ball-like objects: line intersections, corner flags,
//! advertising logos, referee buttons. Those are
//! **false positives that recur at the same spatial position**
//! inside the pitch ROI, often with gaps (detector confidence
//! flickers in and out across frames as micro-lighting changes).
//!
//! The offline Python POC detected these by scanning all frames and
//! flagging any 2%-grid cell with N+ hits inside any W-sample window.
//! This module is the **online** equivalent: it maintains a rolling
//! time-windowed buffer of detection buckets per camera, and reports
//! whether the current bucket already has N+ hits in the window.
//!
//! # Units
//!
//! The filter buckets **normalized camera-frame coordinates** (from
//! [`MappedDetection::camera_center`]) — not panorama yaw/pitch. This
//! is deliberate: static mimics live at fixed *camera pixels*, not
//! fixed yaw values; the two diverge when calibration drifts or when
//! a camera moves. Flicker should reject based on where the detector
//! sees the hit, not where the projection math maps it.
//!
//! # Usage
//!
//! ```no_run
//! use reco_autocam::trackers::filters::FlickerFilter;
//! use reco_core::detect::detector::CameraId;
//! # let det_center = (0.58, 0.39);
//! # let t_ms = 120.0;
//! # let class_id = 0_u16;
//!
//! let mut f = FlickerFilter::with_defaults();
//! // For each candidate detection in this frame:
//! let is_flicker = f.record_and_check(CameraId::Left, class_id, det_center, t_ms);
//! if is_flicker {
//!     // drop — recurrent static mimic
//! }
//! ```
//!
//! [`MappedDetection::camera_center`]: reco_core::detect::director::MappedDetection::camera_center

use std::collections::VecDeque;

use reco_core::detect::detector::CameraId;

/// A bucketed position in one camera's frame, used as the spatial
/// key for flicker clustering.
///
/// Buckets are `(bx, by)` integer coords derived from normalized
/// camera-frame pixels `(cx, cy)` via
/// `bx = floor(cx / bucket_norm), by = floor(cy / bucket_norm)`.
/// A smaller `bucket_norm` means tighter bucketing (more fragments,
/// fewer flicker flags); larger means coarser (faster merges, more
/// flags).
///
/// `class_id` keeps separate spatial histograms per detector class
/// so when the filter runs before the trackers (Step 7b), a ball
/// flicker at bucket X doesn't count against a player that happens
/// to stand still at the same bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlickerKey {
    /// Which camera produced the detection.
    pub camera: CameraId,
    /// Detector class that produced the detection.
    pub class_id: u16,
    /// Integer bucket x-coordinate.
    pub bx: i32,
    /// Integer bucket y-coordinate.
    pub by: i32,
}

/// A ring-buffered "have we seen this bucket recently?" filter.
///
/// Holds the most recent `window_ms` worth of detection-bucket
/// samples across all cameras. Each sample call returns whether the
/// incoming bucket has already been seen at least `min_hits` times
/// in that window (including the current sample).
pub struct FlickerFilter {
    bucket_norm: f32,
    window_ms: f64,
    min_hits: u32,
    samples: VecDeque<Sample>,
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    key: FlickerKey,
    t_ms: f64,
}

impl FlickerFilter {
    /// Sensible default parameters derived from the Python POC's
    /// behavior on DJI 4-min (2%-grid buckets, ~3.3s window, 5 hits).
    ///
    /// - `bucket_norm = 0.02` (2% of frame width/height per bucket).
    /// - `window_ms = 3_333.0` (20 sample-frames at every=5 / 30 fps).
    /// - `min_hits = 5` (matches POC).
    pub fn with_defaults() -> Self {
        Self::new(0.02, 3_333.0, 5)
    }

    /// Build a filter with custom parameters.
    ///
    /// # Panics
    ///
    /// Panics if `bucket_norm <= 0.0`, `window_ms <= 0.0`, or
    /// `min_hits == 0`.
    pub fn new(bucket_norm: f32, window_ms: f64, min_hits: u32) -> Self {
        assert!(bucket_norm > 0.0, "bucket_norm must be positive");
        assert!(window_ms > 0.0, "window_ms must be positive");
        assert!(min_hits > 0, "min_hits must be positive");
        Self {
            bucket_norm,
            window_ms,
            min_hits,
            samples: VecDeque::with_capacity(64),
        }
    }

    /// Bucketize a normalized camera-frame position for a class.
    pub fn key(&self, camera: CameraId, class_id: u16, cx: f32, cy: f32) -> FlickerKey {
        FlickerKey {
            camera,
            class_id,
            bx: (cx / self.bucket_norm) as i32,
            by: (cy / self.bucket_norm) as i32,
        }
    }

    /// Record a detection and return `true` if its bucket is a
    /// flicker (≥ `min_hits` samples including this one inside the
    /// rolling window).
    ///
    /// - `camera`: the detection's camera.
    /// - `class_id`: detector class (separate histogram per class).
    /// - `center`: normalized camera-frame `(cx, cy)` in `[0, 1]`.
    /// - `t_ms`: monotonic timestamp; samples older than
    ///   `t_ms - window_ms` are evicted.
    #[must_use]
    pub fn record_and_check(
        &mut self,
        camera: CameraId,
        class_id: u16,
        center: (f32, f32),
        t_ms: f64,
    ) -> bool {
        self.evict(t_ms);
        let key = self.key(camera, class_id, center.0, center.1);
        self.samples.push_back(Sample { key, t_ms });
        let hits = self.samples.iter().filter(|s| s.key == key).count() as u32;
        hits >= self.min_hits
    }

    /// Drop samples outside the rolling window before `t_ms`.
    fn evict(&mut self, t_ms: f64) {
        let cutoff = t_ms - self.window_ms;
        while let Some(head) = self.samples.front() {
            if head.t_ms < cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Current number of samples held (diagnostic).
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLASS_BALL: u16 = 0;
    const CLASS_PLAYER: u16 = 1;

    #[test]
    fn single_hit_not_flicker() {
        let mut f = FlickerFilter::new(0.02, 1_000.0, 3);
        assert!(!f.record_and_check(CameraId::Left, CLASS_BALL, (0.5, 0.5), 0.0));
    }

    #[test]
    fn three_hits_same_bucket_flag() {
        let mut f = FlickerFilter::new(0.02, 1_000.0, 3);
        assert!(!f.record_and_check(CameraId::Left, CLASS_BALL, (0.50, 0.50), 0.0));
        assert!(!f.record_and_check(CameraId::Left, CLASS_BALL, (0.51, 0.51), 100.0));
        assert!(f.record_and_check(CameraId::Left, CLASS_BALL, (0.500, 0.500), 200.0));
    }

    #[test]
    fn different_buckets_do_not_accumulate() {
        let mut f = FlickerFilter::new(0.02, 1_000.0, 2);
        assert!(!f.record_and_check(CameraId::Left, CLASS_BALL, (0.10, 0.10), 0.0));
        assert!(!f.record_and_check(CameraId::Left, CLASS_BALL, (0.90, 0.90), 100.0));
        assert!(!f.record_and_check(CameraId::Left, CLASS_BALL, (0.50, 0.50), 200.0));
    }

    #[test]
    fn different_cameras_do_not_accumulate() {
        let mut f = FlickerFilter::new(0.02, 1_000.0, 2);
        assert!(!f.record_and_check(CameraId::Left, CLASS_BALL, (0.50, 0.50), 0.0));
        assert!(!f.record_and_check(CameraId::Right, CLASS_BALL, (0.50, 0.50), 100.0));
    }

    #[test]
    fn different_classes_do_not_accumulate() {
        // Step 7b: a ball flickering at (0.5, 0.5) must not count
        // against a player standing still at the same camera-pixel.
        let mut f = FlickerFilter::new(0.02, 1_000.0, 2);
        assert!(!f.record_and_check(CameraId::Left, CLASS_BALL, (0.50, 0.50), 0.0));
        assert!(!f.record_and_check(CameraId::Left, CLASS_PLAYER, (0.50, 0.50), 100.0));
    }

    #[test]
    fn window_eviction_resets_flicker() {
        let mut f = FlickerFilter::new(0.02, 500.0, 2);
        assert!(!f.record_and_check(CameraId::Left, CLASS_BALL, (0.50, 0.50), 0.0));
        // Second hit inside window triggers.
        assert!(f.record_and_check(CameraId::Left, CLASS_BALL, (0.50, 0.50), 100.0));
        // After window elapses, next hit is single again.
        assert!(!f.record_and_check(CameraId::Left, CLASS_BALL, (0.50, 0.50), 1_000.0));
    }

    #[test]
    fn bucket_key_matches_manual_compute() {
        let f = FlickerFilter::new(0.02, 1_000.0, 2);
        let k = f.key(CameraId::Left, CLASS_BALL, 0.58, 0.39);
        assert_eq!(k.bx, 29);
        assert_eq!(k.by, 19);
    }
}
