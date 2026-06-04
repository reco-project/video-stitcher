//! Stateless class provider: projects the current frame's detections of
//! one class into world entities, with no temporal tracking.
//!
//! # Why there is no tracking here
//!
//! A panner that aggregates a point cloud (the [`FieldPanner`] cluster)
//! gets its robustness from the aggregation itself - the trim discards
//! outliers - so per-entity identity, age, matching, velocity, and
//! coasting have no consumer on the control path (verified by grep: only
//! tests referenced them). Per the "every component needs a proven
//! purpose" rule they are gone; recover them from git history if a
//! future consumer (e.g. per-entity highlight overlays) needs identity.
//!
//! This is the counterpart to [`BallTracker`](crate::trackers::BallTracker):
//! the ball is a noisy *singleton* with no cloud to average against, so
//! it keeps a stateful tracker (jump-gate, coast, player-anchor). A
//! point-cloud class (players, and later referees/keepers) needs none of
//! that and uses this provider instead. The two are the [`Tracker`]
//! trait's two real shapes.
//!
//! # Why there is no coast
//!
//! Detection-interval gaps are already bridged upstream: on the frames
//! between detection cycles the session re-runs trackers on the
//! *retained* `last_detections`, so the same entities are re-emitted
//! every frame without any coast state here.
//!
//! [`FieldPanner`]: crate::panners::FieldPanner

use reco_core::detect::director::MappedDetection;
use reco_core::detect::tracker::{TrackState, TrackedEntity, Tracker, WorldState};

/// Emits this frame's detections of a single class as live tracked
/// entities, one per detection, with no temporal state.
///
/// The identity fields it cannot meaningfully fill (`id`, `age_frames`)
/// are left at `0`; `origin` records the reporting camera for
/// diagnostics only.
pub struct ClassProvider {
    class_id: u16,
}

impl ClassProvider {
    /// Build a provider for the given detection `class_id`.
    pub fn new(class_id: u16) -> Self {
        Self { class_id }
    }
}

impl Tracker for ClassProvider {
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
        // Stateless: nothing to observe.
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
        let mut t = ClassProvider::new(0);
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
    fn same_detections_emit_same_entities_across_frames() {
        // Between detection cycles the session re-feeds the retained
        // detections; the provider must be deterministic so entities
        // don't flicker without any coast state.
        let mut t = ClassProvider::new(0);
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
    fn no_detections_means_no_entities() {
        let mut t = ClassProvider::new(0);
        assert!(t.update(&[], 0.0).is_empty());
    }

    #[test]
    fn only_configured_class_is_emitted() {
        let mut t = ClassProvider::new(0);
        let mut other = det(CameraId::Left, 0.3, 0.0, 0.9);
        other.class_id = 5;
        let out = t.update(&[other, det(CameraId::Left, 0.1, 0.0, 0.9)], 0.0);
        assert_eq!(out.len(), 1, "the class-5 detection is filtered out");
        assert!((out[0].yaw - 0.1).abs() < 1e-6);
    }

    #[test]
    fn detection_without_position_is_skipped() {
        let mut t = ClassProvider::new(0);
        let mut no_pos = det(CameraId::Left, 0.3, 0.0, 0.9);
        no_pos.position = None;
        let out = t.update(&[no_pos], 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn class_id_accessor() {
        let t = ClassProvider::new(7);
        assert_eq!(t.class_id(), 7);
    }
}
