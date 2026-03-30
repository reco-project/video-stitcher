//! Object tracking across frames.
//!
//! A [`Tracker`] sits between the [`Detector`](crate::detector::Detector)
//! and the [`Director`](crate::director::Director), assigning persistent
//! identity to detections across frames. This enables:
//! - Smooth camera following (director tracks a specific object, not the
//!   nearest detection each frame)
//! - Velocity estimation (same object across frames = trajectory)
//! - External analytics (possession time, distance covered, event detection)
//!
//! Without a tracker, each frame's detections are independent - the director
//! has no way to know whether two detections in consecutive frames are the
//! same object.

use crate::detector::Detection;

/// A detection with persistent identity across frames.
///
/// Produced by a [`Tracker`] from raw [`Detection`]s. The `track_id`
/// remains stable as long as the tracker considers the object the same
/// entity (even if detection is missed for a few frames via prediction).
#[derive(Debug, Clone)]
pub struct Track {
    /// Persistent identity for this tracked object.
    ///
    /// Unique within the tracker's lifetime. Once an ID is retired
    /// (object lost), it is never reused.
    pub id: u64,

    /// The underlying detection for this frame.
    pub detection: Detection,

    /// How many consecutive frames this track has been alive.
    ///
    /// A freshly created track has `age = 1`. Tracks that survive
    /// across missed detections (via prediction) still increment age.
    pub age: u64,
}

/// Trait for tracking detected objects across frames.
///
/// Implementations handle the association problem: given detections in
/// frame N and existing tracks from frame N-1, decide which detections
/// extend existing tracks and which start new ones.
///
/// Common strategies:
/// - **Nearest-neighbor**: assign each detection to the closest track (IoU or distance)
/// - **Kalman filter**: predict track positions forward, associate with Hungarian algorithm
/// - **Deep association**: use appearance features for re-identification
///
/// # Example
///
/// ```rust,ignore
/// use reco_core::tracker::{Tracker, Track};
///
/// struct SimpleTracker { next_id: u64 }
///
/// impl Tracker for SimpleTracker {
///     fn update(&mut self, _frame_index: u64, _timestamp_ms: f64,
///               detections: &[reco_core::detector::Detection]) -> Vec<Track> {
///         detections.iter().map(|d| {
///             let id = self.next_id;
///             self.next_id += 1;
///             Track { id, detection: d.clone(), age: 1 }
///         }).collect()
///     }
/// }
/// ```
pub trait Tracker: Send {
    /// Update tracks with new detections for this frame.
    ///
    /// Called once per detection frame (respects `detection_interval`).
    /// Returns the active tracks, each associated with a detection from
    /// either the current or a previous frame (predicted tracks).
    fn update(
        &mut self,
        frame_index: u64,
        timestamp_ms: f64,
        detections: &[Detection],
    ) -> Vec<Track>;
}
