//! Extended Kalman Filter tracker for persistent object identity.
//!
//! Assigns stable track IDs to detections across frames using a constant-
//! acceleration motion model with nearest-neighbor association. Tracks are
//! predicted forward during missed detections and retired after a configurable
//! number of consecutive misses.

use reco_core::detector::Detection;
use reco_core::tracker::{Track, Tracker};

/// A single tracked object's Kalman filter state.
struct TrackState {
    id: u64,
    /// State vector: [px, py, vx, vy, ax, ay] in normalized coordinates.
    state: [f64; 6],
    /// Diagonal covariance (simplified - no full matrix for perf).
    covariance: [f64; 6],
    /// Last matched detection (kept for output).
    last_detection: Detection,
    /// Frames since last successful association.
    missed_frames: u32,
    /// Total frames this track has been alive.
    age: u64,
    /// Last timestamp for dt computation.
    last_timestamp_ms: f64,
}

impl TrackState {
    fn new(id: u64, detection: Detection, timestamp_ms: f64) -> Self {
        Self {
            id,
            state: [
                detection.center_x as f64,
                detection.center_y as f64,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            covariance: [0.01, 0.01, 0.1, 0.1, 1.0, 1.0],
            last_detection: detection,
            missed_frames: 0,
            age: 1,
            last_timestamp_ms: timestamp_ms,
        }
    }

    /// Predict state forward by dt seconds using constant-acceleration model.
    fn predict(&mut self, timestamp_ms: f64) {
        let dt = ((timestamp_ms - self.last_timestamp_ms) / 1000.0).max(0.001);
        let dt2 = dt * dt / 2.0;
        self.last_timestamp_ms = timestamp_ms;

        // x += v*dt + a*dt^2/2
        self.state[0] += self.state[2] * dt + self.state[4] * dt2;
        self.state[1] += self.state[3] * dt + self.state[5] * dt2;
        // v += a*dt
        self.state[2] += self.state[4] * dt;
        self.state[3] += self.state[5] * dt;

        // Increase uncertainty (process noise).
        let q_pos = 0.001 * dt * dt;
        let q_vel = 0.01 * dt;
        let q_acc = 0.1 * dt;
        self.covariance[0] += q_pos;
        self.covariance[1] += q_pos;
        self.covariance[2] += q_vel;
        self.covariance[3] += q_vel;
        self.covariance[4] += q_acc;
        self.covariance[5] += q_acc;
    }

    /// Update state with a matched detection (Kalman update step).
    fn update(&mut self, detection: &Detection) {
        let meas_x = detection.center_x as f64;
        let meas_y = detection.center_y as f64;

        // Innovation (measurement residual).
        let innov_x = meas_x - self.state[0];
        let innov_y = meas_y - self.state[1];

        // Measurement noise.
        let r = 0.005;

        // Kalman gain (simplified diagonal).
        let k_x = self.covariance[0] / (self.covariance[0] + r);
        let k_y = self.covariance[1] / (self.covariance[1] + r);

        // State update.
        self.state[0] += k_x * innov_x;
        self.state[1] += k_y * innov_y;
        // Velocity from innovation (implicit).
        self.state[2] = self.state[2] * 0.8 + innov_x * 0.2 * 30.0; // ~30fps
        self.state[3] = self.state[3] * 0.8 + innov_y * 0.2 * 30.0;

        // Covariance update.
        self.covariance[0] *= 1.0 - k_x;
        self.covariance[1] *= 1.0 - k_y;

        self.missed_frames = 0;
        self.last_detection = detection.clone();
        // Update the detection position with the filtered state.
        self.last_detection.center_x = self.state[0] as f32;
        self.last_detection.center_y = self.state[1] as f32;
    }

    /// Squared distance from predicted position to a detection.
    fn distance_to(&self, det: &Detection) -> f64 {
        let dx = self.state[0] - det.center_x as f64;
        let dy = self.state[1] - det.center_y as f64;
        dx * dx + dy * dy
    }

    /// Build a Track output from current state.
    fn to_track(&self) -> Track {
        Track {
            id: self.id,
            detection: self.last_detection.clone(),
            age: self.age,
        }
    }
}

/// Extended Kalman Filter tracker with nearest-neighbor association.
///
/// Maintains a set of active tracks. Each frame:
/// 1. Predict all tracks forward
/// 2. Associate detections to tracks by nearest distance (greedy)
/// 3. Update matched tracks, increment miss count for unmatched
/// 4. Create new tracks for unassociated detections
/// 5. Retire tracks that have been missed too many frames
pub struct EkfTracker {
    tracks: Vec<TrackState>,
    next_id: u64,
    /// Max consecutive missed frames before retiring a track.
    max_missed: u32,
    /// Max distance (squared, normalized coords) for association.
    max_distance_sq: f64,
}

impl EkfTracker {
    /// Create a new tracker with default parameters.
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_missed: 15,
            max_distance_sq: 0.05 * 0.05, // 5% of frame dimension
        }
    }

    /// Create a tracker with custom parameters.
    pub fn with_config(max_missed_frames: u32, max_association_distance: f32) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_missed: max_missed_frames,
            max_distance_sq: (max_association_distance as f64).powi(2),
        }
    }
}

impl Default for EkfTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracker for EkfTracker {
    fn update(
        &mut self,
        _frame_index: u64,
        timestamp_ms: f64,
        detections: &[Detection],
    ) -> Vec<Track> {
        reco_core::profile_scope!("ekf_tracker_update");

        // Step 1: Predict all existing tracks forward.
        for track in &mut self.tracks {
            track.predict(timestamp_ms);
            track.age += 1;
        }

        // Step 2: Greedy nearest-neighbor association.
        let mut det_matched = vec![false; detections.len()];
        let mut track_matched = vec![false; self.tracks.len()];

        // Build distance matrix, sort by distance, assign greedily.
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
        for (ti, track) in self.tracks.iter().enumerate() {
            for (di, det) in detections.iter().enumerate() {
                // Only associate detections from the same camera.
                if det.camera != track.last_detection.camera {
                    continue;
                }
                let dist = track.distance_to(det);
                if dist < self.max_distance_sq {
                    pairs.push((ti, di, dist));
                }
            }
        }
        pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        for (ti, di, _) in &pairs {
            if track_matched[*ti] || det_matched[*di] {
                continue;
            }
            self.tracks[*ti].update(&detections[*di]);
            track_matched[*ti] = true;
            det_matched[*di] = true;
        }

        // Step 3: Mark unmatched tracks as missed.
        for (ti, matched) in track_matched.iter().enumerate() {
            if !matched {
                self.tracks[ti].missed_frames += 1;
            }
        }

        // Step 4: Create new tracks for unmatched detections.
        for (di, matched) in det_matched.iter().enumerate() {
            if !matched {
                let id = self.next_id;
                self.next_id += 1;
                self.tracks
                    .push(TrackState::new(id, detections[di].clone(), timestamp_ms));
            }
        }

        // Step 5: Retire old tracks.
        self.tracks.retain(|t| t.missed_frames <= self.max_missed);

        // Output all active tracks.
        self.tracks.iter().map(|t| t.to_track()).collect()
    }
}
