//! Live-players provider: projects the current frame's player
//! detections into the world state, with no temporal tracking.
//!
//! # Why there is no tracking here
//!
//! The panner consumes an *unordered set* of `(yaw, pitch, confidence)`
//! points to compute a cluster centroid and spread; it never keys off
//! per-entity identity. So stable IDs, age, greedy matching, velocity
//! prediction, and multi-frame coasting had no consumer on the control
//! path (verified by grep: only this module's tests referenced them).
//! Per the project's "every component needs a proven purpose" rule they
//! were removed; recover them from git history if a future consumer
//! (e.g. per-player highlight overlays) actually needs identity.
//!
//! # Why dropping the coast is safe
//!
//! Detection-interval gaps are already bridged upstream: on the frames
//! between detection cycles the session re-runs trackers on the
//! *retained* `last_detections`, so the same players are re-emitted
//! every frame without any coast state here. The old 45-frame coast did
//! one extra thing - hold a player the model dropped for a full
//! detection cycle - which at interval 15 meant inventing a stale
//! "ghost" for ~1.5s. Those ghosts inflated the cluster spread (and thus
//! the FOV), so removing the coast removes a bug, not a feature.

use reco_core::detect::director::MappedDetection;
use reco_core::detect::tracker::{TrackState, TrackedEntity, Tracker, WorldState};

/// Emits this frame's player detections as live tracked entities.
///
/// Stateless beyond the configured class id: each `update` projects
/// every in-class detection with a position into a `Tracking`
/// [`TrackedEntity`]. The identity fields it cannot meaningfully fill
/// (`id`, `age_frames`) are left at `0`; `origin` records the reporting
/// camera for diagnostics only.
pub struct PlayerTracker {
    class_id: u16,
}

impl PlayerTracker {
    /// Build a live-players provider for the given detection `class_id`.
    pub fn new(class_id: u16) -> Self {
        Self { class_id }
    }
}

impl Tracker for PlayerTracker {
    fn update(&mut self, detections: &[MappedDetection], _timestamp_ms: f64) -> Vec<TrackedEntity> {
        detections
            .iter()
            .filter(|d| d.class_id == self.class_id)
            .filter_map(|d| {
                let pos = d.position?;
                Some(TrackedEntity {
                    id: 0,
                    class_id: self.class_id,
                    yaw: pos.yaw,
                    pitch: pos.pitch,
                    confidence: d.confidence,
                    state: TrackState::Tracking,
                    age_frames: 0,
                    origin: d.camera,
                })
            })
            .collect()
    }

    fn class_id(&self) -> u16 {
        self.class_id
    }

    fn observe_world(&mut self, _world: &WorldState) {
        // First tracker the session runs each frame; no earlier
        // tracker context to observe.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detect::detector::CameraId;
    use reco_core::detect::director::ViewportPosition;

    fn det(camera: CameraId, yaw: f32, pitch: f32, conf: f32) -> MappedDetection {
        MappedDetection {
            camera,
            class_id: 0,
            confidence: conf,
            camera_center: (0.5, 0.5),
            camera_size: (0.05, 0.05),
            position: Some(ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            }),
        }
    }

    #[test]
    fn emits_one_entity_per_in_class_detection() {
        let mut t = PlayerTracker::new(0);
        let dets = vec![
            det(CameraId::Left, 0.0, 0.0, 0.9),
            det(CameraId::Left, 0.5, 0.0, 0.8),
            det(CameraId::Right, -0.5, 0.0, 0.7),
        ];
        let out = t.update(&dets, 0.0);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|e| e.state == TrackState::Tracking));
        // Positions and confidence pass straight through.
        assert!((out[0].yaw - 0.0).abs() < 1e-6 && (out[0].confidence - 0.9).abs() < 1e-6);
        assert!((out[2].yaw + 0.5).abs() < 1e-6 && out[2].origin == CameraId::Right);
    }

    #[test]
    fn same_detections_emit_same_players_across_frames() {
        // Between detection cycles the session re-feeds the retained
        // detections; the provider must be deterministic so players
        // don't flicker without any coast state.
        let mut t = PlayerTracker::new(0);
        let dets = vec![
            det(CameraId::Left, 0.1, 0.0, 0.9),
            det(CameraId::Left, 0.2, 0.0, 0.9),
        ];
        let a = t.update(&dets, 0.0);
        let b = t.update(&dets, 16.7);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.yaw - y.yaw).abs() < 1e-6 && (x.pitch - y.pitch).abs() < 1e-6);
        }
    }

    #[test]
    fn no_detections_means_no_players() {
        let mut t = PlayerTracker::new(0);
        assert!(t.update(&[], 0.0).is_empty());
    }

    #[test]
    fn only_configured_class_is_emitted() {
        let mut t = PlayerTracker::new(0);
        let mut other = det(CameraId::Left, 0.3, 0.0, 0.9);
        other.class_id = 5;
        let out = t.update(&[other, det(CameraId::Left, 0.1, 0.0, 0.9)], 0.0);
        assert_eq!(out.len(), 1, "the class-5 detection is filtered out");
        assert!((out[0].yaw - 0.1).abs() < 1e-6);
    }

    #[test]
    fn detection_without_position_is_skipped() {
        let mut t = PlayerTracker::new(0);
        let mut no_pos = det(CameraId::Left, 0.3, 0.0, 0.9);
        no_pos.position = None;
        let out = t.update(&[no_pos], 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn class_id_accessor() {
        let t = PlayerTracker::new(7);
        assert_eq!(t.class_id(), 7);
    }
}
