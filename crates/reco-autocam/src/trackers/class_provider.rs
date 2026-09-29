//! Stateless class provider: projects the current frame's detections of
//! one class into world entities, with no temporal tracking.
//!
//! [`FieldPanner`] aggregates a point cloud, so this provider needs no identity matching or trajectory state. It emits newly measured positions as tracking and retained observations as coasting, preserving confidence for cluster weighting. The core owns the per-camera detection cache; a successful empty result removes that camera's observations. See `same_detections_emit_same_entities_across_frames` and `mixed_camera_measurements_keep_held_positions_without_marking_them_fresh`.
//!
//! [`FieldPanner`]: crate::panners::FieldPanner

use reco_core::detect::mapped_detection::MappedDetection;
use reco_core::detect::tracker::{TrackState, TrackedEntity, Tracker};

/// Emits measured or retained detections of one class without temporal state.
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
    fn update(
        &mut self,
        detections: &[MappedDetection],
        fresh_cameras: &[reco_core::geometry::CameraIndex],
        _timestamp_ms: f64,
    ) -> Vec<TrackedEntity> {
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
                    state: if fresh_cameras.contains(&d.camera) {
                        TrackState::Tracking
                    } else {
                        TrackState::Coasting
                    },
                    age_frames: 0,
                    origin: d.camera,
                })
            })
            .collect()
    }

    fn class_id(&self) -> u16 {
        self.class_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::geometry::CameraIndex;
    use reco_core::geometry::YawPitch;

    fn det(camera: CameraIndex, yaw: f32, pitch: f32, conf: f32) -> MappedDetection {
        MappedDetection {
            camera,
            class_id: 0,
            confidence: conf,
            camera_center: (0.5, 0.5),
            camera_size: (0.05, 0.05),
            position: Some(YawPitch { yaw, pitch }),
        }
    }

    #[test]
    fn mixed_camera_measurements_keep_held_positions_without_marking_them_fresh() {
        let detections = [det(0, 0.1, 0.0, 0.9), det(1, -0.1, 0.0, 0.8)];
        for fresh_camera in [0, 1] {
            let mut tracker = ClassProvider::new(0);
            let entities = tracker.update(&detections, &[fresh_camera], 0.0);
            assert_eq!(entities.len(), 2);
            for entity in entities {
                assert_eq!(
                    entity.state,
                    if entity.origin == fresh_camera {
                        TrackState::Tracking
                    } else {
                        TrackState::Coasting
                    }
                );
                assert_eq!(entity.confidence, detections[entity.origin].confidence);
                assert_eq!(entity.yaw, detections[entity.origin].position.unwrap().yaw);
            }
        }
    }

    #[test]
    fn emits_one_entity_per_in_class_detection() {
        let mut t = ClassProvider::new(0);
        let dets = vec![
            det(0, 0.0, 0.0, 0.9),
            det(0, 0.5, 0.0, 0.8),
            det(1, -0.5, 0.0, 0.7),
        ];
        let out = t.update(&dets, &[0, 1], 0.0);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|e| e.state == TrackState::Tracking));
        // Positions and confidence pass straight through.
        assert!((out[0].yaw - 0.0).abs() < 1e-6 && (out[0].confidence - 0.9).abs() < 1e-6);
        assert!((out[2].yaw + 0.5).abs() < 1e-6 && out[2].origin == 1);
    }

    #[test]
    fn same_detections_emit_same_entities_across_frames() {
        // Between detection cycles the session re-feeds the retained
        // detections; the provider must be deterministic so entities
        // don't flicker without any coast state.
        let mut t = ClassProvider::new(0);
        let dets = vec![det(0, 0.1, 0.0, 0.9), det(0, 0.2, 0.0, 0.9)];
        let a = t.update(&dets, &[0, 1], 0.0);
        let b = t.update(&dets, &[], 16.7);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.yaw - y.yaw).abs() < 1e-6 && (x.pitch - y.pitch).abs() < 1e-6);
            assert_eq!(x.confidence, y.confidence);
            assert_eq!(x.state, TrackState::Tracking);
            assert_eq!(y.state, TrackState::Coasting);
        }
        assert!(t.update(&[], &[0, 1], 33.4).is_empty());
    }

    #[test]
    fn no_detections_means_no_entities() {
        let mut t = ClassProvider::new(0);
        assert!(t.update(&[], &[0, 1], 0.0).is_empty());
    }

    #[test]
    fn only_configured_class_is_emitted() {
        let mut t = ClassProvider::new(0);
        let mut other = det(0, 0.3, 0.0, 0.9);
        other.class_id = 5;
        let out = t.update(&[other, det(0, 0.1, 0.0, 0.9)], &[0, 1], 0.0);
        assert_eq!(out.len(), 1, "the class-5 detection is filtered out");
        assert!((out[0].yaw - 0.1).abs() < 1e-6);
    }

    #[test]
    fn detection_without_position_is_skipped() {
        let mut t = ClassProvider::new(0);
        let mut no_pos = det(0, 0.3, 0.0, 0.9);
        no_pos.position = None;
        let out = t.update(&[no_pos], &[0, 1], 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn class_id_accessor() {
        let t = ClassProvider::new(7);
        assert_eq!(t.class_id(), 7);
    }
}
