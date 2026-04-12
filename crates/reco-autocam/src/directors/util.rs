//! Shared utilities for director implementations.

use reco_core::detector::CameraId;
use reco_core::director::MappedDetection;

/// Default tracking field of view in degrees.
///
/// Used by directors and the smoother as a fallback when the inner director
/// does not emit an explicit FOV. Note: the pipeline's `ViewportConfig`
/// defaults to 75.0 degrees - the tracking FOV is intentionally narrower
/// to provide a tighter view of the action.
pub const DEFAULT_FOV: f32 = 55.0;

/// Minimum detection confidence for player filtering.
pub const MIN_PLAYER_CONFIDENCE: f32 = 0.3;

/// Score a detection for best-candidate selection.
///
/// Higher = better. Factors: confidence, center proximity (less fisheye
/// distortion), camera consistency (reduces oscillation in overlap).
pub fn detection_score(det: &MappedDetection, last_camera: Option<CameraId>) -> f32 {
    let mut score = det.confidence;
    let cx = det.camera_center.0;
    let cy = det.camera_center.1;
    let center_dist = ((cx - 0.5) * (cx - 0.5) + (cy - 0.5) * (cy - 0.5)).sqrt();
    score -= center_dist * 0.2;
    if let Some(last_cam) = last_camera {
        if det.camera == last_cam {
            score += 0.1;
        }
    }
    score
}

/// Deduplicate detections from different cameras at the same panorama position.
///
/// Players in the seam overlap region are detected by both cameras. This
/// merges detections from different cameras within `radius` (radians),
/// keeping the higher-confidence version.
pub fn dedup_cross_camera(detections: &[&MappedDetection], radius: f32) -> Vec<(f32, f32, f32)> {
    let mut unique: Vec<(f32, f32, f32, CameraId)> = Vec::new();
    for d in detections {
        let pos = match d.position {
            Some(p) => p,
            None => continue,
        };
        if let Some(existing) = unique.iter_mut().find(|(uy, up, _, cam)| {
            *cam != d.camera && {
                let dy = pos.yaw - *uy;
                let dp = pos.pitch - *up;
                (dy * dy + dp * dp).sqrt() < radius
            }
        }) {
            if d.confidence > existing.2 {
                *existing = (pos.yaw, pos.pitch, d.confidence, d.camera);
            }
        } else {
            unique.push((pos.yaw, pos.pitch, d.confidence, d.camera));
        }
    }
    unique.into_iter().map(|(y, p, c, _)| (y, p, c)).collect()
}
