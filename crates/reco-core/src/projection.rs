//! Coordinate mapping between camera pixel space and panoramic viewport.
//!
//! These functions bridge detection coordinates (in individual camera frames)
//! and virtual camera orientation (yaw/pitch), enabling:
//! - **Detection mapping**: convert detector output to director yaw/pitch
//! - **"No-black" panning**: compute valid viewport bounds to avoid black edges
//!
//! ## Coordinate Spaces
//!
//! ```text
//! Camera pixel [0,1]  ──undistort──►  Plane UV  ──model matrix──►  World 3D
//!                                                                       │
//! Virtual camera yaw/pitch  ◄──decompose──  Direction from camera  ◄────┘
//! ```

use crate::calibration::{CameraParams, MatchCalibration};
use crate::detector::CameraId;
use crate::director::ViewportPosition;
use crate::renderer::PLANE_ASPECT;
use crate::scene::SceneGeometry;

use nalgebra::{Point3, Vector3};

/// Maximum Newton-Raphson iterations for KB4 inverse distortion.
const MAX_ITERATIONS: usize = 20;
/// Convergence threshold for Newton-Raphson.
const CONVERGENCE_EPS: f64 = 1e-10;

/// Map a detection in camera pixel space to the yaw/pitch needed to
/// center the virtual camera on it.
///
/// `norm_x` and `norm_y` are in normalized `[0.0, 1.0]` image coordinates
/// (as returned by [`Detection`](crate::detector::Detection)).
///
/// Returns `None` if the inverse distortion fails to converge (rare,
/// indicates an extreme point far outside the valid lens area).
///
/// # Example
///
/// ```rust
/// use reco_core::projection::camera_to_panorama;
/// use reco_core::detector::CameraId;
/// use reco_core::calibration::MatchCalibration;
/// use reco_core::scene::SceneGeometry;
///
/// # fn example(cal: &MatchCalibration) {
/// let scene = SceneGeometry::from_layout(&cal.layout);
/// if let Some(pos) = camera_to_panorama(CameraId::Left, 0.5, 0.5, cal, &scene) {
///     println!("Center of left camera maps to yaw={:.3}, pitch={:.3}", pos.yaw, pos.pitch);
/// }
/// # }
/// ```
pub fn camera_to_panorama(
    camera: CameraId,
    norm_x: f32,
    norm_y: f32,
    calibration: &MatchCalibration,
    scene: &SceneGeometry,
) -> Option<ViewportPosition> {
    let params = match camera {
        CameraId::Left => &calibration.left,
        CameraId::Right => &calibration.right,
    };

    // Step 1: Inverse fisheye — camera pixel [0,1] → plane UV (extended space)
    let plane_uv = inverse_fisheye(norm_x as f64, norm_y as f64, params)?;

    // Step 2: Plane UV → 3D world point
    let world_point = plane_uv_to_world(plane_uv, camera, scene);

    // Step 3: World point → yaw/pitch
    let dir = (world_point - Point3::from(Vector3::from(scene.camera_position))).normalize();
    Some(direction_to_yaw_pitch(&dir, &scene.camera_position))
}

/// Precomputed coverage boundary for the dual-camera panorama.
///
/// Built once from calibration data by densely sampling both camera planes'
/// edge loops through the projection chain (KB4 undistortion + model matrix +
/// spherical decomposition). Stored as a pitch-indexed lookup table of yaw
/// ranges, with per-plane tracking for seam gap detection.
///
/// This replaces the per-frame frontier sampling approach with a mathematically
/// exact, precomputed solution. Per-frame cost is negligible (a few table
/// lookups with linear interpolation).
#[derive(Debug, Clone)]
pub struct CoverageBoundary {
    /// Number of pitch slices in the lookup table.
    n_slices: usize,
    /// Global minimum pitch with any coverage (radians).
    pub pitch_min: f32,
    /// Global maximum pitch with any coverage (radians).
    pub pitch_max: f32,
    /// Per-slice combined coverage: `(yaw_min, yaw_max)`.
    slices: Vec<(f32, f32)>,
    /// Per-slice left-plane coverage: `(yaw_min, yaw_max)`.
    left_slices: Vec<(f32, f32)>,
    /// Per-slice right-plane coverage: `(yaw_min, yaw_max)`.
    right_slices: Vec<(f32, f32)>,
}

/// Result of clamping a viewport position to the safe panning region.
#[derive(Debug, Clone, Copy)]
pub struct ClampedPosition {
    /// Clamped yaw in radians.
    pub yaw: f32,
    /// Clamped pitch in radians.
    pub pitch: f32,
}

/// Angular offsets of the four viewport corners from center.
///
/// Each corner is `(delta_yaw, delta_pitch)` relative to the viewport center,
/// computed from the perspective projection math (not the half-FOV
/// approximation that breaks down at corners).
struct CornerOffsets {
    /// `(delta_yaw, delta_pitch)` for top-left, top-right, bottom-right, bottom-left.
    offsets: [(f32, f32); 4],
}

impl CornerOffsets {
    /// Compute the actual angular offsets of the 4 viewport corners.
    ///
    /// A perspective viewport maps a screen rectangle to a curved region
    /// in (yaw, pitch) space. The corners extend further than `half_hfov`
    /// and `half_vfov` along the diagonal. This function computes the exact
    /// angular position of each corner by projecting the screen-space corner
    /// directions through yaw/pitch decomposition.
    ///
    /// The offsets are computed at pitch=0 for simplicity. The error from
    /// spherical curvature at typical pitch values (< 0.3 rad) is negligible
    /// compared to the coverage boundary resolution.
    fn compute(fov_v_deg: f32, aspect: f32) -> Self {
        let half_v = (fov_v_deg * 0.5_f32).to_radians();
        let half_h = (half_v.tan() * aspect).atan();

        // Screen-space corner positions in the local camera frame.
        // The camera looks along -Z, with X right and Y up.
        let tan_h = half_h.tan();
        let tan_v = half_v.tan();

        let corners = [
            (-tan_h, tan_v),  // top-left
            (tan_h, tan_v),   // top-right
            (tan_h, -tan_v),  // bottom-right
            (-tan_h, -tan_v), // bottom-left
        ];

        let mut offsets = [(0.0_f32, 0.0_f32); 4];
        for (i, &(sx, sy)) in corners.iter().enumerate() {
            // Local direction: (sx, sy, -1), normalized
            let len = (sx * sx + sy * sy + 1.0).sqrt();
            let dx = sx / len;
            let dy = sy / len;
            let dz = -1.0 / len;

            // Decompose into yaw/pitch offsets.
            // yaw = atan2(dx, -dz), pitch = asin(dy)
            let delta_yaw = dx.atan2(-dz);
            let delta_pitch = dy.asin();
            offsets[i] = (delta_yaw, delta_pitch);
        }

        Self { offsets }
    }
}

impl CoverageBoundary {
    /// Build the coverage boundary from calibration data.
    ///
    /// Densely samples both planes' edge loops (and a sparse interior grid)
    /// to map coverage into (yaw, pitch) space. Groups samples into
    /// pitch slices for O(1) lookup at runtime.
    ///
    /// Typical cost: ~20k `camera_to_panorama` calls, < 1ms on any modern CPU.
    pub fn from_calibration(calibration: &MatchCalibration, scene: &SceneGeometry) -> Self {
        let n_slices: usize = 200;
        let margin = 0.02_f32;

        // Collect projected points per plane.
        let mut left_points: Vec<(f32, f32)> = Vec::new();
        let mut right_points: Vec<(f32, f32)> = Vec::new();

        for &camera in &[CameraId::Left, CameraId::Right] {
            let points = if camera == CameraId::Left {
                &mut left_points
            } else {
                &mut right_points
            };

            // Dense edge sampling: 200 points per edge (4 edges = 800 per plane).
            let edge_steps = 200_u32;
            for i in 0..=edge_steps {
                let t = margin + (1.0 - 2.0 * margin) * (i as f32 / edge_steps as f32);
                // Bottom edge
                if let Some(pos) = camera_to_panorama(camera, t, margin, calibration, scene) {
                    points.push((pos.yaw, pos.pitch));
                }
                // Top edge
                if let Some(pos) = camera_to_panorama(camera, t, 1.0 - margin, calibration, scene) {
                    points.push((pos.yaw, pos.pitch));
                }
                // Left edge
                if let Some(pos) = camera_to_panorama(camera, margin, t, calibration, scene) {
                    points.push((pos.yaw, pos.pitch));
                }
                // Right edge
                if let Some(pos) = camera_to_panorama(camera, 1.0 - margin, t, calibration, scene) {
                    points.push((pos.yaw, pos.pitch));
                }
            }

            // Sparse interior grid (20x20) for coverage at intermediate pitch levels.
            let grid_steps = 20_u32;
            for ix in 0..=grid_steps {
                let nx = margin + (1.0 - 2.0 * margin) * (ix as f32 / grid_steps as f32);
                for iy in 0..=grid_steps {
                    let ny = margin + (1.0 - 2.0 * margin) * (iy as f32 / grid_steps as f32);
                    if let Some(pos) = camera_to_panorama(camera, nx, ny, calibration, scene) {
                        points.push((pos.yaw, pos.pitch));
                    }
                }
            }
        }

        // Find global pitch range.
        let all_points = left_points.iter().chain(right_points.iter());
        let mut global_pitch_min = f32::MAX;
        let mut global_pitch_max = f32::MIN;
        for &(_, pitch) in all_points {
            global_pitch_min = global_pitch_min.min(pitch);
            global_pitch_max = global_pitch_max.max(pitch);
        }

        if global_pitch_min >= global_pitch_max {
            return Self {
                n_slices,
                pitch_min: 0.0,
                pitch_max: 0.0,
                slices: vec![(0.0, 0.0); n_slices],
                left_slices: vec![(0.0, 0.0); n_slices],
                right_slices: vec![(0.0, 0.0); n_slices],
            };
        }

        let pitch_range = global_pitch_max - global_pitch_min;
        let slice_size = pitch_range / n_slices as f32;

        // Bucket points into pitch slices.
        let mut slices = vec![(f32::MAX, f32::MIN); n_slices];
        let mut left_slices = vec![(f32::MAX, f32::MIN); n_slices];
        let mut right_slices = vec![(f32::MAX, f32::MIN); n_slices];

        let pitch_to_slice = |pitch: f32| -> usize {
            let idx = ((pitch - global_pitch_min) / slice_size) as usize;
            idx.min(n_slices - 1)
        };

        for &(yaw, pitch) in &left_points {
            let s = pitch_to_slice(pitch);
            left_slices[s].0 = left_slices[s].0.min(yaw);
            left_slices[s].1 = left_slices[s].1.max(yaw);
            slices[s].0 = slices[s].0.min(yaw);
            slices[s].1 = slices[s].1.max(yaw);
        }
        for &(yaw, pitch) in &right_points {
            let s = pitch_to_slice(pitch);
            right_slices[s].0 = right_slices[s].0.min(yaw);
            right_slices[s].1 = right_slices[s].1.max(yaw);
            slices[s].0 = slices[s].0.min(yaw);
            slices[s].1 = slices[s].1.max(yaw);
        }

        // Fill gaps: slices with no samples inherit from neighbors.
        // This handles sparse coverage at extreme pitch values.
        for i in 1..n_slices {
            if slices[i].0 > slices[i].1 {
                slices[i] = slices[i - 1];
                left_slices[i] = left_slices[i - 1];
                right_slices[i] = right_slices[i - 1];
            }
        }
        for i in (0..n_slices - 1).rev() {
            if slices[i].0 > slices[i].1 {
                slices[i] = slices[i + 1];
                left_slices[i] = left_slices[i + 1];
                right_slices[i] = right_slices[i + 1];
            }
        }

        Self {
            n_slices,
            pitch_min: global_pitch_min,
            pitch_max: global_pitch_max,
            slices,
            left_slices,
            right_slices,
        }
    }

    /// Look up the combined yaw coverage range at a given pitch.
    ///
    /// Uses linear interpolation between adjacent slices for smooth bounds.
    fn yaw_range_at(&self, pitch: f32) -> (f32, f32) {
        self.interpolate_slice(&self.slices, pitch)
    }

    /// Look up the left-plane yaw coverage range at a given pitch.
    fn left_yaw_range_at(&self, pitch: f32) -> (f32, f32) {
        self.interpolate_slice(&self.left_slices, pitch)
    }

    /// Look up the right-plane yaw coverage range at a given pitch.
    fn right_yaw_range_at(&self, pitch: f32) -> (f32, f32) {
        self.interpolate_slice(&self.right_slices, pitch)
    }

    /// Interpolate a slice table at the given pitch.
    fn interpolate_slice(&self, table: &[(f32, f32)], pitch: f32) -> (f32, f32) {
        if self.n_slices == 0 || self.pitch_max <= self.pitch_min {
            return (0.0, 0.0);
        }

        let pitch_range = self.pitch_max - self.pitch_min;
        let t = (pitch - self.pitch_min) / pitch_range;
        let idx_f = t * (self.n_slices - 1) as f32;
        let idx_lo = (idx_f.floor() as usize).min(self.n_slices - 1);
        let idx_hi = (idx_lo + 1).min(self.n_slices - 1);
        let frac = idx_f - idx_lo as f32;

        let lo = table[idx_lo];
        let hi = table[idx_hi];

        // Only interpolate if both slices have valid data.
        if lo.0 > lo.1 {
            return hi;
        }
        if hi.0 > hi.1 {
            return lo;
        }

        (lo.0 + frac * (hi.0 - lo.0), lo.1 + frac * (hi.1 - lo.1))
    }

    /// Check if coverage is contiguous (no seam gap) at a given pitch.
    ///
    /// Returns `true` if both planes contribute coverage at this pitch
    /// and there is no gap between them. A gap means the left plane's
    /// rightmost yaw is less than the right plane's leftmost yaw.
    fn is_contiguous_at(&self, pitch: f32) -> bool {
        let left = self.left_yaw_range_at(pitch);
        let right = self.right_yaw_range_at(pitch);

        // Both planes must have valid coverage.
        if left.0 > left.1 || right.0 > right.1 {
            return false;
        }

        // The planes overlap (or at least touch) at the seam.
        // Left plane tends to have more positive yaw, right plane more negative.
        // They must overlap: left_min <= right_max AND right_min <= left_max.
        left.0 <= right.1 && right.0 <= left.1
    }

    /// Clamp a viewport position to the safe panning region for a given FOV.
    ///
    /// Computes the exact angular extent of all 4 viewport corners using
    /// perspective projection math, then ensures each corner falls within
    /// the precomputed coverage boundary. This is mathematically exact -
    /// no sampling, no ad-hoc safety margins.
    ///
    /// `yaw` and `pitch` are in panorama space (radians).
    /// `fov_v_deg` is the vertical field of view in degrees.
    pub fn safe_clamp(&self, yaw: f32, pitch: f32, fov_v_deg: f32) -> ClampedPosition {
        let aspect = 16.0_f32 / 9.0;
        let corners = CornerOffsets::compute(fov_v_deg, aspect);

        // For each corner, compute the constraint it imposes on the viewport center.
        //
        // If corner c has offset (dy_c, dp_c), then the corner's absolute position
        // is (yaw + dy_c, pitch + dp_c). For this to be within coverage:
        //   yaw_min(pitch + dp_c) <= yaw + dy_c <= yaw_max(pitch + dp_c)
        // Rearranging for yaw:
        //   yaw_min(pitch + dp_c) - dy_c <= yaw <= yaw_max(pitch + dp_c) - dy_c

        // First clamp pitch. The pitch constraint from each corner is:
        //   pitch_min <= pitch + dp_c <= pitch_max
        //   => pitch_min - dp_c <= pitch <= pitch_max - dp_c
        let mut safe_pitch_min = f32::MIN;
        let mut safe_pitch_max = f32::MAX;
        for &(_, dp) in &corners.offsets {
            safe_pitch_min = safe_pitch_min.max(self.pitch_min - dp);
            safe_pitch_max = safe_pitch_max.min(self.pitch_max - dp);
        }

        // Additionally, require coverage to be contiguous at all corner pitches.
        // Scan inward from the pitch ceiling until all corners see contiguous coverage.
        let pitch_step = (self.pitch_max - self.pitch_min) / self.n_slices as f32;
        let max_corner_dp = corners.offsets.iter().map(|c| c.1).fold(f32::MIN, f32::max);
        let min_corner_dp = corners.offsets.iter().map(|c| c.1).fold(f32::MAX, f32::min);

        // Scan from top: find the highest pitch where all corners see contiguous coverage.
        {
            let mut ceiling = safe_pitch_max;
            let mut p = self.pitch_max - max_corner_dp;
            while p >= self.pitch_min - min_corner_dp {
                let all_ok = corners
                    .offsets
                    .iter()
                    .all(|&(_, dp)| self.is_contiguous_at(p + dp));
                if all_ok {
                    ceiling = p;
                    break;
                }
                p -= pitch_step;
            }
            safe_pitch_max = safe_pitch_max.min(ceiling);
        }

        // Scan from bottom: find the lowest pitch where all corners see contiguous coverage.
        {
            let mut floor = safe_pitch_min;
            let mut p = self.pitch_min - min_corner_dp;
            while p <= self.pitch_max - max_corner_dp {
                let all_ok = corners
                    .offsets
                    .iter()
                    .all(|&(_, dp)| self.is_contiguous_at(p + dp));
                if all_ok {
                    floor = p;
                    break;
                }
                p += pitch_step;
            }
            safe_pitch_min = safe_pitch_min.max(floor);
        }

        // Collapse if inverted.
        if safe_pitch_min > safe_pitch_max {
            let mid = (safe_pitch_min + safe_pitch_max) * 0.5;
            safe_pitch_min = mid;
            safe_pitch_max = mid;
        }

        let clamped_pitch = pitch.clamp(safe_pitch_min, safe_pitch_max);

        // Now compute yaw constraints using the clamped pitch.
        let mut safe_yaw_min = f32::MIN;
        let mut safe_yaw_max = f32::MAX;
        for &(dy, dp) in &corners.offsets {
            let corner_pitch = clamped_pitch + dp;
            let (cov_yaw_min, cov_yaw_max) = self.yaw_range_at(corner_pitch);
            // yaw + dy must be in [cov_yaw_min, cov_yaw_max]
            safe_yaw_min = safe_yaw_min.max(cov_yaw_min - dy);
            safe_yaw_max = safe_yaw_max.min(cov_yaw_max - dy);
        }

        // Collapse if inverted.
        if safe_yaw_min > safe_yaw_max {
            let mid = (safe_yaw_min + safe_yaw_max) * 0.5;
            safe_yaw_min = mid;
            safe_yaw_max = mid;
        }

        let clamped_yaw = yaw.clamp(safe_yaw_min, safe_yaw_max);

        ClampedPosition {
            yaw: clamped_yaw,
            pitch: clamped_pitch,
        }
    }
}

// ---- Internal functions ----

/// Inverse KB4 fisheye: distorted camera pixel [0,1] → undistorted plane UV.
///
/// Inverts the forward KB4 model used in the shader:
/// ```text
/// θ_d = θ × (1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)
/// ```
/// Uses Newton-Raphson to solve for θ given θ_d.
fn inverse_fisheye(dist_x: f64, dist_y: f64, params: &CameraParams) -> Option<(f64, f64)> {
    let w = params.width as f64;
    let h = params.height as f64;
    let fx = params.fx / w;
    let fy = params.fy / h;
    let cx = params.cx / w;
    let cy = params.cy / h;
    let k = params.d;

    // Normalized distorted coordinates
    let dx = (dist_x - cx) / fx;
    let dy = (dist_y - cy) / fy;
    let theta_d = (dx * dx + dy * dy).sqrt();

    if theta_d < 1e-12 {
        // At the optical center — no distortion
        return Some((cx, cy));
    }

    // Newton-Raphson: solve f(θ) = θ(1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸) - θ_d = 0
    let mut theta = theta_d; // initial guess
    for _ in 0..MAX_ITERATIONS {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let t8 = t4 * t4;

        let f = theta * (1.0 + k[0] * t2 + k[1] * t4 + k[2] * t6 + k[3] * t8) - theta_d;
        let f_prime = 1.0 + 3.0 * k[0] * t2 + 5.0 * k[1] * t4 + 7.0 * k[2] * t6 + 9.0 * k[3] * t8;

        if f_prime.abs() < 1e-15 {
            return None; // degenerate
        }

        let delta = f / f_prime;
        theta -= delta;

        if delta.abs() < CONVERGENCE_EPS {
            break;
        }
    }

    // Recover undistorted coordinates
    let r = theta.tan(); // theta = atan(r) → r = tan(theta)
    let scale = if theta_d > 1e-12 { theta_d / r } else { 1.0 };

    let x = dx / scale;
    let y = dy / scale;

    // Plane UV in the extended [-0.5, 1.5] space used by the shader
    let uv_x = fx * x + cx;
    let uv_y = fy * y + cy;

    Some((uv_x, uv_y))
}

/// Convert a plane UV (in extended shader space) to a 3D world point.
fn plane_uv_to_world(uv: (f64, f64), camera: CameraId, scene: &SceneGeometry) -> Point3<f32> {
    // Extended UV → texture UV [0,1]
    let tex_u = ((uv.0 + 0.5) / 2.0) as f32;
    let tex_v = ((uv.1 + 0.5) / 2.0) as f32;

    // Texture UV → local quad position (matches quad_vertices)
    let local_x = tex_u - 0.5;
    let local_y = (0.5 - tex_v) / PLANE_ASPECT;

    let local_point = nalgebra::Vector4::new(local_x, local_y, 0.0, 1.0);
    let model = match camera {
        CameraId::Left => scene.model_matrix_left(),
        CameraId::Right => scene.model_matrix_right(),
    };

    let world = model * local_point;
    Point3::new(world.x, world.y, world.z)
}

/// Decompose a direction vector into yaw/pitch relative to the virtual camera.
///
/// Matches the inverse of `view_matrix` in renderer.rs:
/// - base_forward = camera → origin
/// - yaw = horizontal rotation around Y
/// - pitch = elevation from horizontal plane
fn direction_to_yaw_pitch(dir: &Vector3<f32>, camera_position: &[f32; 3]) -> ViewportPosition {
    let eye = Vector3::new(camera_position[0], camera_position[1], camera_position[2]);
    let base_forward = (-eye).normalize();
    let world_up = Vector3::new(0.0, 1.0, 0.0);
    let base_right = base_forward.cross(&world_up).normalize();

    // Pitch: elevation angle from horizontal plane
    let pitch = dir.y.clamp(-1.0, 1.0).asin();

    // Yaw: horizontal angle relative to base_forward
    let horizontal = Vector3::new(dir.x, 0.0, dir.z);
    let h_len = horizontal.norm();

    let yaw = if h_len > 1e-6 {
        let h = horizontal / h_len;
        let cos_yaw = h.dot(&base_forward).clamp(-1.0, 1.0);
        let sin_yaw = h.dot(&base_right);
        sin_yaw.atan2(cos_yaw)
    } else {
        0.0
    };

    ViewportPosition {
        yaw,
        pitch,
        fov_degrees: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{CameraParams, MatchCalibration, PlaneLayout};

    fn test_calibration() -> MatchCalibration {
        MatchCalibration {
            left: CameraParams {
                width: 3840,
                height: 2160,
                fx: 1796.32,
                fy: 1797.22,
                cx: 1919.37,
                cy: 1063.17,
                d: [0.0342, 0.0677, -0.0741, 0.0299],
            },
            right: CameraParams {
                width: 3840,
                height: 2160,
                fx: 1796.32,
                fy: 1797.22,
                cx: 1919.37,
                cy: 1063.17,
                d: [0.0342, 0.0677, -0.0741, 0.0299],
            },
            layout: PlaneLayout {
                camera_axis_offset: 0.2398,
                intersect: 0.5446,
                x_ty: 0.00476,
                x_rz: 0.00753,
                z_rx: -0.00431,
            },
            field_roi: None,
        }
    }

    #[test]
    fn optical_center_maps_to_known_position() {
        let cal = test_calibration();
        let scene = SceneGeometry::from_layout(&cal.layout);

        // Optical center of the left camera (cx/w, cy/h)
        let cx = cal.left.cx as f32 / cal.left.width as f32;
        let cy = cal.left.cy as f32 / cal.left.height as f32;

        let pos = camera_to_panorama(CameraId::Left, cx, cy, &cal, &scene);
        assert!(pos.is_some(), "optical center should map successfully");
        let pos = pos.unwrap();
        // The optical center should produce a valid yaw/pitch (no NaN)
        assert!(pos.yaw.is_finite(), "yaw should be finite");
        assert!(pos.pitch.is_finite(), "pitch should be finite");
    }

    #[test]
    fn left_camera_left_edge_has_more_negative_yaw() {
        let cal = test_calibration();
        let scene = SceneGeometry::from_layout(&cal.layout);

        let center = camera_to_panorama(CameraId::Left, 0.5, 0.5, &cal, &scene).unwrap();
        let left_edge = camera_to_panorama(CameraId::Left, 0.1, 0.5, &cal, &scene).unwrap();

        // Left edge of left camera should have a more negative (or equal) yaw
        // than center, since the left camera faces the +X direction
        assert!(
            left_edge.yaw < center.yaw || (left_edge.yaw - center.yaw).abs() < 0.01,
            "left edge yaw ({:.4}) should be <= center yaw ({:.4})",
            left_edge.yaw,
            center.yaw
        );
    }

    #[test]
    fn right_camera_produces_different_yaw_than_left() {
        let cal = test_calibration();
        let scene = SceneGeometry::from_layout(&cal.layout);

        let left_center = camera_to_panorama(CameraId::Left, 0.5, 0.5, &cal, &scene).unwrap();
        let right_center = camera_to_panorama(CameraId::Right, 0.5, 0.5, &cal, &scene).unwrap();

        // The two cameras face different directions, so their centers
        // should map to different yaw values
        assert!(
            (left_center.yaw - right_center.yaw).abs() > 0.01,
            "left ({:.4}) and right ({:.4}) camera centers should differ in yaw",
            left_center.yaw,
            right_center.yaw
        );
    }

    #[test]
    fn inverse_fisheye_roundtrip_at_center() {
        let params = CameraParams {
            width: 3840,
            height: 2160,
            fx: 1796.32,
            fy: 1797.22,
            cx: 1919.37,
            cy: 1063.17,
            d: [0.0342, 0.0677, -0.0741, 0.0299],
        };

        // At the optical center, distortion should be zero
        let cx = params.cx / params.width as f64;
        let cy = params.cy / params.height as f64;
        let result = inverse_fisheye(cx, cy, &params).unwrap();
        assert!(
            (result.0 - cx).abs() < 1e-6 && (result.1 - cy).abs() < 1e-6,
            "optical center should be a fixed point: got ({:.6}, {:.6}), expected ({:.6}, {:.6})",
            result.0,
            result.1,
            cx,
            cy
        );
    }

    #[test]
    fn coverage_boundary_has_valid_range() {
        let cal = test_calibration();
        let scene = SceneGeometry::from_layout(&cal.layout);

        let cb = CoverageBoundary::from_calibration(&cal, &scene);
        assert!(
            cb.pitch_min < cb.pitch_max,
            "pitch range should be valid: {:.4}..{:.4}",
            cb.pitch_min,
            cb.pitch_max
        );

        // Combined yaw at center pitch should be non-trivial.
        let mid_pitch = (cb.pitch_min + cb.pitch_max) * 0.5;
        let (yaw_lo, yaw_hi) = cb.yaw_range_at(mid_pitch);
        assert!(
            yaw_hi - yaw_lo > 0.1,
            "yaw range at center pitch too small: {:.4}..{:.4}",
            yaw_lo,
            yaw_hi
        );
    }

    #[test]
    fn safe_clamp_keeps_position_within_coverage() {
        let cal = test_calibration();
        let scene = SceneGeometry::from_layout(&cal.layout);
        let cb = CoverageBoundary::from_calibration(&cal, &scene);

        // Request an extreme position that should be clamped.
        let clamped = cb.safe_clamp(10.0, 10.0, 55.0);
        assert!(clamped.yaw.is_finite());
        assert!(clamped.pitch.is_finite());

        // Clamped position should be within the coverage range
        // (with some margin for the viewport).
        assert!(
            clamped.pitch <= cb.pitch_max,
            "clamped pitch {:.4} exceeds coverage max {:.4}",
            clamped.pitch,
            cb.pitch_max
        );
        assert!(
            clamped.pitch >= cb.pitch_min,
            "clamped pitch {:.4} below coverage min {:.4}",
            clamped.pitch,
            cb.pitch_min
        );
    }

    #[test]
    fn wider_fov_produces_tighter_safe_region() {
        let cal = test_calibration();
        let scene = SceneGeometry::from_layout(&cal.layout);
        let cb = CoverageBoundary::from_calibration(&cal, &scene);

        // Clamp a moderate position with narrow vs wide FOV.
        // Wider FOV should clamp more aggressively (tighter range).
        let narrow = cb.safe_clamp(0.5, 0.1, 30.0);
        let wide = cb.safe_clamp(0.5, 0.1, 60.0);

        // With 60° FOV, the yaw should be clamped closer to center.
        assert!(
            wide.yaw <= narrow.yaw,
            "wider FOV yaw ({:.4}) should be <= narrow FOV yaw ({:.4})",
            wide.yaw,
            narrow.yaw
        );
    }

    #[test]
    fn zero_distortion_produces_identity_mapping() {
        let params = CameraParams {
            width: 1920,
            height: 1080,
            fx: 960.0,
            fy: 540.0,
            cx: 960.0,
            cy: 540.0,
            d: [0.0, 0.0, 0.0, 0.0],
        };

        // With zero distortion and fx=width/2, cx=width/2, the mapping
        // should be close to identity
        let result = inverse_fisheye(0.5, 0.5, &params).unwrap();
        assert!(
            (result.0 - 0.5).abs() < 1e-6 && (result.1 - 0.5).abs() < 1e-6,
            "zero-distortion center should map to itself"
        );
    }
}
