//! # reco-highlights
//!
//! Auto-highlight reel generator for panoramic sports recordings.
//!
//! This crate taps into the per-frame detection stream of a
//! [`reco_core::session::StitchSession`] and looks for moments of sustained,
//! high-confidence action. It emits a JSON "edit decision list" of highlight
//! windows that a coach, editor, or downstream tool can slice out of the
//! encoded panorama.
//!
//! ## What counts as a highlight?
//!
//! A highlight is a contiguous run of frames where:
//! - the tracked object (default: ball, `class_id == 0`) is detected with
//!   confidence above [`HighlightConfig::min_confidence`], and
//! - the run lasts at least [`HighlightConfig::min_duration_ms`].
//!
//! Short gaps up to [`HighlightConfig::max_gap_ms`] are bridged so a brief
//! occlusion (a player passing in front of the ball) doesn't chop a single
//! rally into five tiny clips.
//!
//! ## Usage
//!
//! See [`HighlightDetector`]. The detector is wired into a session via the
//! [`StitchSession::set_detection_callback`](reco_core::session::StitchSession::set_detection_callback)
//! hook, so you get the highlights reel as a side effect of the normal
//! stitch render.
//!
//! ```rust,no_run
//! use reco_highlights::{HighlightConfig, HighlightDetector};
//! use std::sync::{Arc, Mutex};
//!
//! let detector = Arc::new(Mutex::new(HighlightDetector::new(HighlightConfig::default())));
//! let shared = detector.clone();
//! // Attach `move |dets, idx, ts| shared.lock().unwrap().push(dets, idx, ts)`
//! // to StitchSession::set_detection_callback.
//! ```

use serde::{Deserialize, Serialize};

use reco_core::director::MappedDetection;

/// Configuration knobs for highlight detection.
#[derive(Debug, Clone)]
pub struct HighlightConfig {
    /// Detection class to track. Defaults to `0` (ball in the Reco YOLO models).
    pub target_class_id: u16,
    /// Minimum confidence for a detection to count as "active".
    pub min_confidence: f32,
    /// Minimum duration of a highlight window, in milliseconds. Shorter runs
    /// are discarded.
    pub min_duration_ms: f64,
    /// Maximum gap (milliseconds) between two active frames that can still
    /// be bridged into a single window.
    pub max_gap_ms: f64,
    /// Pre-roll added to the start of every emitted window, in milliseconds.
    pub pre_roll_ms: f64,
    /// Post-roll added to the end of every emitted window, in milliseconds.
    pub post_roll_ms: f64,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            target_class_id: 0,
            min_confidence: 0.45,
            min_duration_ms: 2_500.0,
            max_gap_ms: 1_000.0,
            pre_roll_ms: 1_500.0,
            post_roll_ms: 2_000.0,
        }
    }
}

/// A single highlight window in the source video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighlightWindow {
    /// 1-based sequence number of this highlight in the reel.
    pub index: u32,
    /// Inclusive start timestamp in milliseconds, already padded with `pre_roll_ms`.
    pub start_ms: f64,
    /// Exclusive end timestamp in milliseconds, already padded with `post_roll_ms`.
    pub end_ms: f64,
    /// First frame index covered by this window (pre-roll applied by timestamp).
    pub start_frame: u64,
    /// Last frame index covered by this window (post-roll applied by timestamp).
    pub end_frame: u64,
    /// Number of "active" detection frames that actually fired inside the window.
    pub active_frames: u64,
    /// Peak confidence observed inside the window.
    pub peak_confidence: f32,
    /// Mean confidence across the `active_frames`.
    pub mean_confidence: f32,
}

/// A complete highlight reel, ready to serialize to JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighlightReel {
    /// Sidecar schema version. Bump when the file format changes.
    pub version: u32,
    /// Total frames seen by the detector.
    pub total_frames: u64,
    /// Highlight windows, in chronological order.
    pub windows: Vec<HighlightWindow>,
}

/// State for a single highlight window that is still growing.
#[derive(Debug, Clone, Copy)]
struct OpenWindow {
    first_active_ms: f64,
    last_active_ms: f64,
    first_frame: u64,
    last_frame: u64,
    active_frames: u64,
    peak_confidence: f32,
    sum_confidence: f32,
}

/// Accumulator that turns a stream of [`MappedDetection`] batches into a
/// [`HighlightReel`].
#[derive(Debug)]
pub struct HighlightDetector {
    config: HighlightConfig,
    open: Option<OpenWindow>,
    closed: Vec<HighlightWindow>,
    total_frames: u64,
}

impl HighlightDetector {
    /// Create a detector with the given config.
    pub fn new(config: HighlightConfig) -> Self {
        Self {
            config,
            open: None,
            closed: Vec::new(),
            total_frames: 0,
        }
    }

    /// Feed a single frame's detections into the detector.
    ///
    /// Wire this up via [`StitchSession::set_detection_callback`](reco_core::session::StitchSession::set_detection_callback).
    pub fn push(&mut self, detections: &[MappedDetection], frame_index: u64, timestamp_ms: f64) {
        self.total_frames = self.total_frames.max(frame_index + 1);

        let strongest = detections
            .iter()
            .filter(|d| d.class_id == self.config.target_class_id)
            .filter(|d| d.confidence >= self.config.min_confidence)
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        match (strongest, self.open.as_mut()) {
            (Some(det), None) => {
                self.open = Some(OpenWindow {
                    first_active_ms: timestamp_ms,
                    last_active_ms: timestamp_ms,
                    first_frame: frame_index,
                    last_frame: frame_index,
                    active_frames: 1,
                    peak_confidence: det.confidence,
                    sum_confidence: det.confidence,
                });
            }
            (Some(det), Some(open)) => {
                open.last_active_ms = timestamp_ms;
                open.last_frame = frame_index;
                open.active_frames += 1;
                open.sum_confidence += det.confidence;
                if det.confidence > open.peak_confidence {
                    open.peak_confidence = det.confidence;
                }
            }
            (None, Some(open)) => {
                if timestamp_ms - open.last_active_ms > self.config.max_gap_ms {
                    let snapshot = *open;
                    self.close_window(snapshot);
                    self.open = None;
                }
            }
            (None, None) => {}
        }
    }

    /// Close any currently-open window and return the reel.
    ///
    /// Call this after the frame loop has finished (e.g. after `StitchJob::run`).
    pub fn finish(mut self) -> HighlightReel {
        if let Some(open) = self.open.take() {
            self.close_window(open);
        }
        HighlightReel {
            version: 1,
            total_frames: self.total_frames,
            windows: self.closed,
        }
    }

    fn close_window(&mut self, open: OpenWindow) {
        let duration_ms = open.last_active_ms - open.first_active_ms;
        if duration_ms < self.config.min_duration_ms {
            return;
        }
        let index = self.closed.len() as u32 + 1;
        let mean_confidence = if open.active_frames > 0 {
            open.sum_confidence / open.active_frames as f32
        } else {
            0.0
        };
        self.closed.push(HighlightWindow {
            index,
            start_ms: (open.first_active_ms - self.config.pre_roll_ms).max(0.0),
            end_ms: open.last_active_ms + self.config.post_roll_ms,
            start_frame: open.first_frame,
            end_frame: open.last_frame,
            active_frames: open.active_frames,
            peak_confidence: open.peak_confidence,
            mean_confidence,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detector::CameraId;
    use reco_core::director::{MappedDetection, ViewportPosition};

    fn det(conf: f32) -> MappedDetection {
        MappedDetection {
            camera: CameraId::Left,
            class_id: 0,
            confidence: conf,
            camera_center: (0.5, 0.5),
            camera_size: (0.05, 0.05),
            position: Some(ViewportPosition {
                yaw: 0.0,
                pitch: 0.0,
                fov_degrees: None,
            }),
        }
    }

    #[test]
    fn emits_window_when_sustained() {
        let cfg = HighlightConfig {
            min_duration_ms: 500.0,
            max_gap_ms: 200.0,
            pre_roll_ms: 0.0,
            post_roll_ms: 0.0,
            ..HighlightConfig::default()
        };
        let mut hl = HighlightDetector::new(cfg);
        for i in 0..31 {
            hl.push(&[det(0.9)], i, i as f64 * 33.3);
        }
        let reel = hl.finish();
        assert_eq!(reel.windows.len(), 1, "one window expected");
        assert_eq!(reel.windows[0].active_frames, 31);
    }

    #[test]
    fn ignores_too_short_bursts() {
        let cfg = HighlightConfig {
            min_duration_ms: 1_000.0,
            max_gap_ms: 200.0,
            ..HighlightConfig::default()
        };
        let mut hl = HighlightDetector::new(cfg);
        for i in 0..5 {
            hl.push(&[det(0.9)], i, i as f64 * 33.3);
        }
        for i in 5..15 {
            hl.push(&[], i, i as f64 * 33.3 + 500.0);
        }
        assert!(hl.finish().windows.is_empty());
    }

    #[test]
    fn bridges_small_gaps() {
        let cfg = HighlightConfig {
            min_duration_ms: 100.0,
            max_gap_ms: 500.0,
            pre_roll_ms: 0.0,
            post_roll_ms: 0.0,
            ..HighlightConfig::default()
        };
        let mut hl = HighlightDetector::new(cfg);
        // active burst 0..500ms
        for i in 0..15 {
            hl.push(&[det(0.8)], i, i as f64 * 33.3);
        }
        // 300ms gap (below max_gap_ms)
        for i in 15..25 {
            hl.push(&[], i, i as f64 * 33.3);
        }
        // resume 800..1500ms
        for i in 25..45 {
            hl.push(&[det(0.8)], i, i as f64 * 33.3);
        }
        let reel = hl.finish();
        assert_eq!(reel.windows.len(), 1, "gap should have been bridged");
    }

    #[test]
    fn splits_on_long_gap() {
        let cfg = HighlightConfig {
            min_duration_ms: 100.0,
            max_gap_ms: 100.0,
            pre_roll_ms: 0.0,
            post_roll_ms: 0.0,
            ..HighlightConfig::default()
        };
        let mut hl = HighlightDetector::new(cfg);
        for i in 0..10 {
            hl.push(&[det(0.8)], i, i as f64 * 33.3);
        }
        // 1.5s silence
        for i in 10..55 {
            hl.push(&[], i, i as f64 * 33.3);
        }
        for i in 55..70 {
            hl.push(&[det(0.8)], i, i as f64 * 33.3);
        }
        let reel = hl.finish();
        assert_eq!(reel.windows.len(), 2, "long gap should split");
    }
}
