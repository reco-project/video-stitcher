//! Multi-entity player tracker with greedy nearest-neighbor matching
//! and stable IDs across frames.
//!
//! # Design
//!
//! On each [`update`](PlayerTracker::update):
//! 1. Keep only detections matching the tracker's `class_id`.
//! 2. Greedily match detections to active tracklets by yaw/pitch
//!    distance, subject to a per-match gate (`match_gate_rad`).
//! 3. Each unmatched detection starts a new tracklet with a fresh
//!    monotonically-increasing ID.
//! 4. Each unmatched tracklet ticks its per-entity [`Coaster`]; when
//!    the coast budget runs out, the tracklet is dropped.
//! 5. Emit all active tracklets as `TrackedEntity`s with the
//!    corresponding [`TrackState`].
//!
//! # Why greedy instead of Hungarian
//!
//! The original plan called for O(n³) Hungarian assignment. In
//! practice, football has ≤22 players per frame, motion is
//! locally smooth, and detections rarely cluster tightly enough to
//! make greedy assignment wrong. Greedy is:
//!
//! - **Simpler**: ~30 lines vs ~100 for Hungarian.
//! - **Fast**: O(t · d) per frame where t = active tracklets,
//!   d = current detections. For football, ~400 comparisons/frame.
//! - **Easy to upgrade**: swap `match_greedy` below for a Hungarian
//!   call when ID-swap frequency becomes a real problem.
//!
//! If ID swaps become an issue (typical signal: players in crowds
//! where detection centers get within `match_gate_rad` of each
//! other), the right upgrade is Hungarian assignment. Opening that
//! as a follow-up issue, not a blocker.
//!
//! # Match gate
//!
//! A detection must be within `match_gate_rad` radians (panorama
//! yaw/pitch) of an existing tracklet's *predicted* position to be
//! matched to it. The prediction is just the last-known position
//! plus linear velocity × dt; a Kalman filter is future work.
//! Detections outside the gate start new tracklets.

use reco_core::detector::CameraId;
use reco_core::director::MappedDetection;
use reco_core::tracker::{TrackState, TrackedEntity, Tracker, WorldState};

use crate::trackers::filters::{CoastStatus, Coaster};

/// Default panorama distance gate for matching a detection to an
/// existing tracklet (~11° ≈ a player-width at typical broadcast
/// distance on a 1080p frame).
pub const DEFAULT_MATCH_GATE_RAD: f32 = 0.20;

/// Default coast budget for a tracklet that stops getting fresh
/// detections. Must exceed detection_interval so tracklets survive
/// between detection cycles. 45 frames at 30 fps = 1.5s, covers
/// detection_interval=30 with margin.
pub const DEFAULT_MAX_COAST_FRAMES: u32 = 45;

/// A multi-entity tracker with stable IDs.
///
/// Tracklet IDs start at 1 and increment monotonically; `0` is
/// reserved for singleton trackers per the [`TrackedEntity`]
/// convention.
pub struct PlayerTracker {
    class_id: u16,
    match_gate_rad: f32,
    max_coast_frames: u32,
    tracklets: Vec<Tracklet>,
    next_id: u64,
}

struct Tracklet {
    id: u64,
    /// Last-accepted panorama position.
    yaw: f32,
    pitch: f32,
    /// Yaw/pitch velocity in rad/s (coarse — first-order difference).
    vyaw: f32,
    vpitch: f32,
    /// Timestamp of last measurement, for dt computation.
    last_t_ms: Option<f64>,
    /// Last-accepted confidence (for diagnostics).
    confidence: f32,
    /// Camera that produced the last accepted measurement.
    origin: CameraId,
    /// Age since first creation, ticked every frame the tracklet
    /// is alive.
    age_frames: u64,
    /// Lifecycle coaster.
    coaster: Coaster,
}

impl PlayerTracker {
    /// Build a new player tracker for the given `class_id` with
    /// default parameters.
    pub fn new(class_id: u16) -> Self {
        Self {
            class_id,
            match_gate_rad: DEFAULT_MATCH_GATE_RAD,
            max_coast_frames: DEFAULT_MAX_COAST_FRAMES,
            tracklets: Vec::new(),
            next_id: 1,
        }
    }

    /// Override the match gate (panorama yaw/pitch radians).
    pub fn with_match_gate_rad(mut self, rad: f32) -> Self {
        self.match_gate_rad = rad.max(1e-3);
        self
    }

    /// Override the coast budget.
    pub fn with_max_coast_frames(mut self, n: u32) -> Self {
        self.max_coast_frames = n;
        self
    }

    /// Number of currently-active tracklets (diagnostic).
    pub fn tracklet_count(&self) -> usize {
        self.tracklets.len()
    }

    /// Predict a tracklet's current position using its last-known
    /// yaw/pitch + velocity × elapsed dt.
    fn predict(&self, t: &Tracklet, now_ms: f64) -> (f32, f32) {
        match t.last_t_ms {
            Some(prev) if now_ms > prev => {
                let dt = ((now_ms - prev) / 1000.0) as f32;
                (t.yaw + t.vyaw * dt, t.pitch + t.vpitch * dt)
            }
            _ => (t.yaw, t.pitch),
        }
    }

    /// Greedily match detections to tracklets by closest predicted
    /// distance. Returns `matches[det_idx] = Some(tracklet_idx)`
    /// for matched detections, `None` for unmatched.
    fn match_greedy(
        &self,
        detections: &[&MappedDetection],
        timestamp_ms: f64,
    ) -> Vec<Option<usize>> {
        let mut matches: Vec<Option<usize>> = vec![None; detections.len()];
        let mut claimed: Vec<bool> = vec![false; self.tracklets.len()];

        // Precompute tracklet predictions.
        let preds: Vec<(f32, f32)> = self
            .tracklets
            .iter()
            .map(|t| self.predict(t, timestamp_ms))
            .collect();

        // Candidate list: (dist, det_idx, tracklet_idx).
        let mut candidates: Vec<(f32, usize, usize)> = Vec::new();
        for (di, det) in detections.iter().enumerate() {
            let Some(pos) = det.position else { continue };
            for (ti, pred) in preds.iter().enumerate() {
                let dy = pos.yaw - pred.0;
                let dp = pos.pitch - pred.1;
                let dist = (dy * dy + dp * dp).sqrt();
                if dist <= self.match_gate_rad {
                    candidates.push((dist, di, ti));
                }
            }
        }
        // Sort by distance — smallest first — and assign greedily.
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, di, ti) in candidates {
            if matches[di].is_none() && !claimed[ti] {
                matches[di] = Some(ti);
                claimed[ti] = true;
            }
        }
        matches
    }
}

impl Tracker for PlayerTracker {
    fn update(&mut self, detections: &[MappedDetection], timestamp_ms: f64) -> Vec<TrackedEntity> {
        // Step 1: self-filter to our class.
        let in_class: Vec<&MappedDetection> = detections
            .iter()
            .filter(|d| d.class_id == self.class_id && d.position.is_some())
            .collect();

        // Step 2: greedy matching.
        let matches = self.match_greedy(&in_class, timestamp_ms);

        // Step 3: update matched tracklets.
        let mut tracklet_updated: Vec<bool> = vec![false; self.tracklets.len()];
        for (di, m) in matches.iter().enumerate() {
            if let Some(ti) = *m {
                let det = in_class[di];
                let pos = det.position.expect("self-filter guarantees Some");
                let t = &mut self.tracklets[ti];
                // Update velocity from (pos - last_pos) / dt.
                if let Some(prev) = t.last_t_ms {
                    let dt_ms = timestamp_ms - prev;
                    if dt_ms > 0.1 {
                        let dt_s = (dt_ms / 1000.0) as f32;
                        t.vyaw = (pos.yaw - t.yaw) / dt_s;
                        t.vpitch = (pos.pitch - t.pitch) / dt_s;
                    }
                }
                t.yaw = pos.yaw;
                t.pitch = pos.pitch;
                t.confidence = det.confidence;
                t.origin = det.camera;
                t.last_t_ms = Some(timestamp_ms);
                t.age_frames = t.age_frames.saturating_add(1);
                t.coaster.accept_fresh();
                tracklet_updated[ti] = true;
            }
        }

        // Step 4: coast unmatched tracklets; drop the ones whose
        // coast budget has been exceeded. Iterate in reverse so
        // swap_remove doesn't break indices.
        let mut ids_lost: Vec<(u64, f32, f32, CameraId)> = Vec::new();
        let mut i = self.tracklets.len();
        while i > 0 {
            i -= 1;
            if tracklet_updated[i] {
                continue;
            }
            let t = &mut self.tracklets[i];
            t.age_frames = t.age_frames.saturating_add(1);
            match t.coaster.step_without_fresh() {
                CoastStatus::Coasting => {
                    // Keep: next iteration sees Coasting and emits.
                }
                CoastStatus::Lost => {
                    ids_lost.push((t.id, t.yaw, t.pitch, t.origin));
                    self.tracklets.swap_remove(i);
                }
                CoastStatus::Tracking => {
                    // Unreachable: step_without_fresh never returns
                    // Tracking.
                }
            }
        }

        // Step 5: unmatched detections start new tracklets.
        for (di, m) in matches.iter().enumerate() {
            if m.is_some() {
                continue;
            }
            let det = in_class[di];
            let pos = det.position.expect("self-filter guarantees Some");
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            let mut coaster = Coaster::new(self.max_coast_frames);
            coaster.accept_fresh();
            self.tracklets.push(Tracklet {
                id,
                yaw: pos.yaw,
                pitch: pos.pitch,
                vyaw: 0.0,
                vpitch: 0.0,
                last_t_ms: Some(timestamp_ms),
                confidence: det.confidence,
                origin: det.camera,
                age_frames: 1,
                coaster,
            });
            log::debug!(
                "PlayerTracker: new tracklet id={} yaw={:.3} pitch={:.3} cam={:?} conf={:.2}",
                id,
                pos.yaw,
                pos.pitch,
                det.camera,
                det.confidence
            );
        }

        // Build the output: all live tracklets + one final Lost
        // entity per just-dropped tracklet so consumers see the
        // transition frame.
        let mut out: Vec<TrackedEntity> = Vec::with_capacity(self.tracklets.len() + ids_lost.len());
        for t in &self.tracklets {
            let state = if t.coaster.frames_coasting() == 0 {
                TrackState::Tracking
            } else {
                TrackState::Coasting
            };
            out.push(TrackedEntity {
                id: t.id,
                class_id: self.class_id,
                yaw: t.yaw,
                pitch: t.pitch,
                confidence: if matches!(state, TrackState::Tracking) {
                    t.confidence
                } else {
                    0.0
                },
                state,
                age_frames: t.age_frames,
                origin: t.origin,
            });
        }
        for (id, yaw, pitch, origin) in &ids_lost {
            log::info!(
                "PlayerTracker: tracklet id={} lost (last yaw={:.3} pitch={:.3})",
                id,
                yaw,
                pitch
            );
            out.push(TrackedEntity {
                id: *id,
                class_id: self.class_id,
                yaw: *yaw,
                pitch: *pitch,
                confidence: 0.0,
                state: TrackState::Lost,
                age_frames: 0,
                origin: *origin,
            });
        }
        out
    }

    fn class_id(&self) -> u16 {
        self.class_id
    }

    fn observe_world(&mut self, _world: &WorldState) {
        // PlayerTracker is the first tracker the session runs each
        // frame; there is no earlier-tracker context to observe.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::director::ViewportPosition;

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
    fn first_detections_create_fresh_tracklets() {
        let mut t = PlayerTracker::new(0);
        let dets = vec![
            det(CameraId::Left, 0.0, 0.0, 0.9),
            det(CameraId::Left, 0.5, 0.0, 0.9),
            det(CameraId::Right, -0.5, 0.0, 0.9),
        ];
        let out = t.update(&dets, 0.0);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|e| e.state == TrackState::Tracking));
        // IDs should be 1, 2, 3 (start at 1).
        let mut ids: Vec<_> = out.iter().map(|e| e.id).collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn ids_stable_across_frames() {
        let mut t = PlayerTracker::new(0);
        t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9)], 0.0);
        t.update(&[det(CameraId::Left, 0.01, 0.0, 0.9)], 16.7);
        let out = t.update(&[det(CameraId::Left, 0.02, 0.0, 0.9)], 33.3);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 1, "tracklet ID should persist across frames");
    }

    #[test]
    fn unmatched_tracklet_coasts_then_lost() {
        let mut t = PlayerTracker::new(0).with_max_coast_frames(2);
        t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9)], 0.0);
        // Frame 2: no detection → coasting.
        let out1 = t.update(&[], 16.7);
        assert_eq!(out1.len(), 1);
        assert_eq!(out1[0].state, TrackState::Coasting);
        // Frame 3: still coasting.
        let out2 = t.update(&[], 33.3);
        assert_eq!(out2[0].state, TrackState::Coasting);
        // Frame 4: coast budget exhausted → Lost emitted, tracklet dropped.
        let out3 = t.update(&[], 50.0);
        assert_eq!(out3.len(), 1);
        assert_eq!(out3[0].state, TrackState::Lost);
        // Next frame: no live tracklets, no output.
        let out4 = t.update(&[], 66.7);
        assert!(out4.is_empty());
    }

    #[test]
    fn detection_far_from_any_tracklet_starts_new_id() {
        let mut t = PlayerTracker::new(0).with_match_gate_rad(0.1);
        let out1 = t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9)], 0.0);
        let id1 = out1[0].id;
        // Far detection beyond match gate → new tracklet.
        let out2 = t.update(&[det(CameraId::Left, 1.0, 0.0, 0.9)], 16.7);
        // Still 2 entities: existing tracklet coasts, new one is created.
        let ids: Vec<u64> = out2.iter().map(|e| e.id).collect();
        assert!(ids.contains(&id1), "old tracklet still alive");
        assert!(ids.iter().any(|&x| x != id1), "new tracklet created");
    }

    #[test]
    fn only_configured_class_is_tracked() {
        let mut t = PlayerTracker::new(0);
        let mut d = det(CameraId::Left, 0.3, 0.0, 0.9);
        d.class_id = 5;
        let out = t.update(&[d], 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn missing_position_ignored() {
        let mut t = PlayerTracker::new(0);
        let mut d = det(CameraId::Left, 0.3, 0.0, 0.9);
        d.position = None;
        let out = t.update(&[d], 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn two_close_detections_match_two_tracklets() {
        // Matches greedy: both detections end up matched to distinct
        // existing tracklets when that produces the minimum total
        // distance — not always optimal (Hungarian handles the
        // pathological ties), but correct for reasonable inputs.
        let mut t = PlayerTracker::new(0).with_match_gate_rad(0.15);
        t.update(
            &[
                det(CameraId::Left, 0.0, 0.0, 0.9),
                det(CameraId::Left, 0.3, 0.0, 0.9),
            ],
            0.0,
        );
        let out = t.update(
            &[
                det(CameraId::Left, 0.02, 0.0, 0.9),
                det(CameraId::Left, 0.32, 0.0, 0.9),
            ],
            16.7,
        );
        assert_eq!(out.len(), 2);
        // Both should be Tracking (got their own matches).
        assert!(out.iter().all(|e| e.state == TrackState::Tracking));
    }

    #[test]
    fn class_id_accessor() {
        let t = PlayerTracker::new(7);
        assert_eq!(t.class_id(), 7);
    }
}
