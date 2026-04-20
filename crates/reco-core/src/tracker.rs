//! Tracker trait and world-state primitives for AI camera control.
//!
//! A `Tracker` turns a per-frame stream of noisy `MappedDetection`s
//! (from [`crate::director`]) into a clean list of `TrackedEntity`s
//! with stable identities, velocity estimates, and a tri-valued
//! lifecycle (`TrackState`). Trackers are single-responsibility:
//! one class (ball, player, …) per tracker. Multiple trackers run
//! in parallel each frame and the session merges their outputs into
//! a `WorldState` that a [`Panner`](crate::panner::Panner) consumes.
//!
//! The split is deliberate: detection-noise rejection, identity, and
//! trajectory smoothing live here; camera-motion decisions live in
//! the panner. Either side can be swapped without disturbing the other.
//!
//! # Pipeline
//!
//! ```text
//!  Detectors → class splitter → [BallTracker | PlayerTracker | …]
//!                                         ↓
//!                                  WorldState { ball, players, … }
//!                                         ↓
//!                                       Panner
//!                                         ↓
//!                                   ViewportPosition
//! ```
//!
//! # Non-goals of this module
//!
//! - `reco-core` stays domain-generic. All ball/player/referee logic
//!   lives in consumer crates such as `reco-autocam`. This module
//!   defines only the contract.
//! - No implementations here. `BallTracker`, `PlayerTracker`, and
//!   future tracker variants ship in `reco-autocam::trackers`.

use crate::detector::CameraId;
use crate::director::MappedDetection;

/// A tri-valued lifecycle flag attached to every [`TrackedEntity`].
///
/// The tracker owns the decision of when to transition between states;
/// consumers (panners, diagnostics, replays) only read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TrackState {
    /// A fresh detection was accepted this frame. Position, velocity,
    /// and confidence reflect the new measurement.
    Tracking,
    /// No fresh detection passed the tracker's filters this frame,
    /// but the entity is recently-enough seen that the tracker is
    /// holding its previous position as the best available estimate.
    /// Panners typically "freeze" the camera on coasting.
    Coasting,
    /// The tracker has given up on this entity. It is removed from
    /// [`WorldState`] next frame. Singletons (e.g. ball) use this as
    /// a not-present signal; panners typically recenter on `Lost`.
    Lost,
}

/// A single entity (ball, player, referee, …) tracked across frames.
///
/// Positions are in panorama space (yaw/pitch radians) so consumers
/// never need to know which raw camera observed the entity. The
/// [`origin`](Self::origin) field records the last-reporting camera
/// for diagnostics and quality-assurance overlays only — panner
/// logic should not branch on it.
///
/// Velocity is optional because:
/// - new tracks with only one measurement have no velocity estimate,
/// - some trackers may opt out of velocity estimation entirely.
///
/// `id` is persistent across frames. Singleton classes (ball) report
/// `id = 0` every frame; multi-entity classes (players) assign stable
/// per-entity IDs for the tracklet's lifetime.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct TrackedEntity {
    /// Persistent ID across frames for this tracklet. Singleton
    /// classes always report `0`; multi-entity trackers assign
    /// monotonically-increasing IDs.
    pub id: u64,
    /// Detection-model class this entity belongs to (e.g. 0 for a
    /// ball-only model, or COCO's 32 for `sports ball`).
    pub class_id: u16,
    /// Horizontal pan angle in radians, panorama frame.
    /// `0.0` = centered on the seam between cameras.
    pub yaw: f32,
    /// Vertical tilt angle in radians, panorama frame.
    /// `0.0` = level; positive = looking up.
    pub pitch: f32,
    /// Angular velocity estimate `(d_yaw, d_pitch)` per second, or
    /// `None` when the tracker has insufficient history.
    pub velocity: Option<(f32, f32)>,
    /// Last-measurement confidence in `[0.0, 1.0]`. Meaning depends
    /// on the tracker: fresh-detection confidence during `Tracking`,
    /// typically `0.0` during `Coasting`.
    pub confidence: f32,
    /// Tri-valued lifecycle flag — see [`TrackState`].
    pub state: TrackState,
    /// Number of frames this tracklet has been active. Useful for
    /// panners that weight older, more stable tracks higher.
    pub age_frames: u64,
    /// Which camera observed the most recent accepted measurement.
    /// Diagnostic field only — downstream logic should route through
    /// [`yaw`](Self::yaw) / [`pitch`](Self::pitch), not by origin.
    pub origin: CameraId,
}

/// The merged per-frame output of every registered tracker.
///
/// This is the read-only view a [`Panner`](crate::panner::Panner)
/// consumes each frame. The struct is intentionally flat — not a trait
/// — so consumer crates can extend it with new fields over time in an
/// additive, backwards-compatible way.
///
/// # Field conventions
///
/// - `ball`: at most one, per the ROI-partition design where each
///   camera's ROI covers exactly one half of the pitch; use [`None`]
///   both for "ball detector returned nothing" and for "ball tracker
///   state is [`TrackState::Lost`]".
/// - `players`: N-sized list with stable IDs across frames; empty
///   when no player tracker is registered or when the current frame
///   has no on-pitch players.
#[derive(Debug, Clone, Default)]
pub struct WorldState {
    /// Current ball position, or `None` when the ball is not being
    /// tracked this frame (lost, or no ball tracker registered).
    pub ball: Option<TrackedEntity>,
    /// All currently-tracked players. Order is not stable — consumers
    /// should key off [`TrackedEntity::id`] when identity matters.
    pub players: Vec<TrackedEntity>,
}

/// The contract implemented by every per-class tracker.
///
/// Implementations are expected to be **single-class**: `ball` has
/// its own tracker, `player` has its own tracker, etc. This keeps
/// the internal state (Kalman filter, SORT tracklet manager, ReID
/// feature memory) class-specific and avoids tangled if-branches
/// on `class_id` inside any single tracker.
///
/// # Invariants
///
/// - [`update`](Self::update) must be called exactly once per frame,
///   in frame order. Trackers are stateful; calling out of order or
///   twice per frame produces undefined behavior.
/// - The returned [`Vec`] is the complete current set of tracked
///   entities of this tracker's class; trackers must not return
///   partial lists. Singleton trackers return length 0 or 1;
///   multi-entity trackers return length 0..=N.
/// - Entities in [`TrackState::Lost`] state may still appear in the
///   returned vector for one frame (to let consumers observe the
///   transition) but must be removed on the subsequent call.
///
/// # Filtering
///
/// Trackers **must** self-filter: if [`update`](Self::update) is
/// passed detections from multiple classes, the tracker must ignore
/// detections whose `class_id` does not match its own
/// [`class_id`](Self::class_id). A class splitter in the session
/// loop may also pre-filter for zero-allocation dispatch, but the
/// self-filter is the safety net.
pub trait Tracker: Send {
    /// Incorporate the current frame's mapped detections and return
    /// the updated set of tracked entities for this tracker's class.
    ///
    /// `timestamp_ms` is elapsed-since-session-start milliseconds,
    /// monotonically increasing across calls.
    fn update(&mut self, detections: &[MappedDetection], timestamp_ms: f64) -> Vec<TrackedEntity>;

    /// The detection class this tracker cares about. Used by a
    /// class splitter in the session loop to route detections
    /// without cloning; trackers must also self-filter by class.
    fn class_id(&self) -> u16;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny in-module tracker used to smoke-test the trait object.
    struct StaticTracker {
        class_id: u16,
        emit: Vec<TrackedEntity>,
    }
    impl Tracker for StaticTracker {
        fn update(
            &mut self,
            _detections: &[MappedDetection],
            _timestamp_ms: f64,
        ) -> Vec<TrackedEntity> {
            self.emit.clone()
        }
        fn class_id(&self) -> u16 {
            self.class_id
        }
    }

    fn entity(id: u64, yaw: f32) -> TrackedEntity {
        TrackedEntity {
            id,
            class_id: 0,
            yaw,
            pitch: 0.0,
            velocity: None,
            confidence: 0.9,
            state: TrackState::Tracking,
            age_frames: 1,
            origin: CameraId::Left,
        }
    }

    #[test]
    fn trait_object_round_trip() {
        let mut t: Box<dyn Tracker> = Box::new(StaticTracker {
            class_id: 32,
            emit: vec![entity(7, 0.25)],
        });
        let out = t.update(&[], 0.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 7);
        assert_eq!(out[0].yaw, 0.25);
        assert_eq!(t.class_id(), 32);
    }

    #[test]
    fn world_state_default_is_empty() {
        let w = WorldState::default();
        assert!(w.ball.is_none());
        assert!(w.players.is_empty());
    }

    #[test]
    fn track_state_values_round_trip_json() {
        for s in [TrackState::Tracking, TrackState::Coasting, TrackState::Lost] {
            let json = serde_json::to_string(&s).unwrap();
            let back: TrackState = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }
}
