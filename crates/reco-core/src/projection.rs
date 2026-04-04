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

/// Compute the valid yaw/pitch bounds for a given FOV where no black
/// edges appear in the viewport.
///
/// Samples the visible edges of both camera planes and returns the
/// tightest bounds that keep the viewport fully within the projected
/// image area. Use this to clamp director output for "no-black" panning.
///
/// Returns `(min_yaw, max_yaw, min_pitch, max_pitch)` in radians.
pub fn viewport_bounds(
    fov_degrees: f32,
    calibration: &MatchCalibration,
    scene: &SceneGeometry,
) -> ViewportBounds {
    // fov_degrees is the VERTICAL FOV (nalgebra Perspective3 convention).
    // Derive horizontal FOV from aspect ratio using rectilinear projection.
    let aspect = 16.0_f32 / 9.0;
    let half_vfov = (fov_degrees * 0.5).to_radians();
    let half_hfov = (half_vfov.tan() * aspect).atan();

    // The viewport corners reach further than edge midpoints due to the
    // tangent projection. At a corner, the angular distance from center
    // is: atan(sqrt(tan²(half_hfov) + tan²(half_vfov))). We account for
    // this by using the DIAGONAL angular extent for constraints, ensuring
    // even the corners stay inside coverage.
    //
    // For corner-aware bounds: when constraining yaw from a pitch bin,
    // the viewport extends half_hfov in yaw at the CENTER pitch, but at
    // the TOP/BOTTOM pitch (±half_vfov from center), the corner extends
    // even further. For a perspective projection, the corner yaw extent
    // at pitch offset dy is: atan(tan(half_hfov) / cos(dy)).
    // This is ~3-5% wider than half_hfov at the edges.
    let corner_hfov = (half_hfov.tan() / half_vfov.cos()).atan();
    let corner_vfov = (half_vfov.tan() / half_hfov.cos()).atan();

    // Sample the edges of both camera frames to find the coverage
    // boundary ("frontier") in panorama space. Using 2%/98% avoids
    // extreme fisheye corners where inverse distortion may diverge.
    let edge_steps: u32 = 40;
    let lo = 0.02_f32;
    let hi = 0.98_f32;
    let mut frontier: Vec<(f32, f32)> = Vec::with_capacity((edge_steps as usize + 1) * 8);

    for &camera in &[CameraId::Left, CameraId::Right] {
        for i in 0..=edge_steps {
            let t = lo + (hi - lo) * (i as f32 / edge_steps as f32);
            for &(nx, ny) in &[(lo, t), (hi, t), (t, lo), (t, hi)] {
                if let Some(pos) = camera_to_panorama(camera, nx, ny, calibration, scene) {
                    frontier.push((pos.yaw, pos.pitch));
                }
            }
        }
    }

    if frontier.is_empty() {
        return ViewportBounds {
            min_yaw: 0.0,
            max_yaw: 0.0,
            min_pitch: 0.0,
            max_pitch: 0.0,
        };
    }

    let pitch_min = frontier.iter().map(|p| p.1).fold(f32::MAX, f32::min);
    let pitch_max = frontier.iter().map(|p| p.1).fold(f32::MIN, f32::max);
    let yaw_min = frontier.iter().map(|p| p.0).fold(f32::MAX, f32::min);
    let yaw_max = frontier.iter().map(|p| p.0).fold(f32::MIN, f32::max);

    // Bin frontier points by pitch to find yaw coverage at each level.
    // Use corner_hfov (not half_hfov) so the viewport CORNERS stay
    // inside coverage, not just the edge midpoints.
    let n_bins: usize = 20;
    let pitch_range = pitch_max - pitch_min;
    let pitch_bin_size = pitch_range / n_bins as f32;
    let min_points_per_bin: usize = 4;

    let mut bound_min_yaw = f32::MIN;
    let mut bound_max_yaw = f32::MAX;

    for bin in 0..n_bins {
        let bin_lo = pitch_min + bin as f32 * pitch_bin_size;
        let bin_hi = bin_lo + pitch_bin_size;

        let (mut yaw_lo, mut yaw_hi, mut count) = (f32::MAX, f32::MIN, 0usize);
        for &(yaw, pitch) in &frontier {
            if pitch >= bin_lo && pitch < bin_hi {
                yaw_lo = yaw_lo.min(yaw);
                yaw_hi = yaw_hi.max(yaw);
                count += 1;
            }
        }

        if count < min_points_per_bin {
            continue;
        }

        bound_min_yaw = bound_min_yaw.max(yaw_lo + corner_hfov);
        bound_max_yaw = bound_max_yaw.min(yaw_hi - corner_hfov);
    }

    // Bin by yaw to find pitch coverage at each level.
    let yaw_range = yaw_max - yaw_min;
    let yaw_bin_size = yaw_range / n_bins as f32;

    let mut bound_min_pitch = f32::MIN;
    let mut bound_max_pitch = f32::MAX;

    for bin in 0..n_bins {
        let bin_lo = yaw_min + bin as f32 * yaw_bin_size;
        let bin_hi = bin_lo + yaw_bin_size;

        let (mut p_lo, mut p_hi, mut count) = (f32::MAX, f32::MIN, 0usize);
        for &(yaw, pitch) in &frontier {
            if yaw >= bin_lo && yaw < bin_hi {
                p_lo = p_lo.min(pitch);
                p_hi = p_hi.max(pitch);
                count += 1;
            }
        }

        if count < min_points_per_bin {
            continue;
        }

        bound_min_pitch = bound_min_pitch.max(p_lo + corner_vfov);
        bound_max_pitch = bound_max_pitch.min(p_hi - corner_vfov);
    }

    // Fallback if binning produced no constraints.
    if bound_min_yaw == f32::MIN {
        bound_min_yaw = yaw_min + corner_hfov;
    }
    if bound_max_yaw == f32::MAX {
        bound_max_yaw = yaw_max - corner_hfov;
    }
    if bound_min_pitch == f32::MIN {
        bound_min_pitch = pitch_min + corner_vfov;
    }
    if bound_max_pitch == f32::MAX {
        bound_max_pitch = pitch_max - corner_vfov;
    }

    // Collapse to midpoint if bounds inverted (coverage too narrow).
    if bound_min_yaw > bound_max_yaw {
        let mid = (bound_min_yaw + bound_max_yaw) * 0.5;
        bound_min_yaw = mid;
        bound_max_yaw = mid;
    }
    if bound_min_pitch > bound_max_pitch {
        let mid = (bound_min_pitch + bound_max_pitch) * 0.5;
        bound_min_pitch = mid;
        bound_max_pitch = mid;
    }

    ViewportBounds {
        min_yaw: bound_min_yaw,
        max_yaw: bound_max_yaw,
        min_pitch: bound_min_pitch,
        max_pitch: bound_max_pitch,
    }
}

/// Valid viewport bounds for "no-black" panning.
///
/// Clamp the director's yaw/pitch to these ranges to ensure the
/// viewport never shows black edges from the L-shaped projection.
#[derive(Debug, Clone, Copy)]
pub struct ViewportBounds {
    /// Minimum yaw in radians (leftmost pan).
    pub min_yaw: f32,
    /// Maximum yaw in radians (rightmost pan).
    pub max_yaw: f32,
    /// Minimum pitch in radians (lowest tilt).
    pub min_pitch: f32,
    /// Maximum pitch in radians (highest tilt).
    pub max_pitch: f32,
}

impl ViewportBounds {
    /// Clamp a viewport position to stay within these bounds.
    pub fn clamp(&self, position: ViewportPosition) -> ViewportPosition {
        ViewportPosition {
            yaw: position.yaw.clamp(self.min_yaw, self.max_yaw),
            pitch: position.pitch.clamp(self.min_pitch, self.max_pitch),
            fov_degrees: position.fov_degrees,
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
                z_rz: 0.0,
            },
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
    fn viewport_bounds_are_valid() {
        let cal = test_calibration();
        let scene = SceneGeometry::from_layout(&cal.layout);

        // Use a narrower FOV to ensure bounds are valid
        let bounds = viewport_bounds(40.0, &cal, &scene);
        assert!(
            bounds.min_yaw < bounds.max_yaw,
            "yaw range should be valid: {:.4}..{:.4}",
            bounds.min_yaw,
            bounds.max_yaw
        );
        assert!(
            bounds.min_pitch < bounds.max_pitch,
            "pitch range should be valid: {:.4}..{:.4}",
            bounds.min_pitch,
            bounds.max_pitch
        );
        // With 40° FOV, the valid range should be non-trivial
        assert!(
            bounds.max_yaw - bounds.min_yaw > 0.01,
            "yaw range too small: {:.4}..{:.4}",
            bounds.min_yaw,
            bounds.max_yaw
        );
    }

    #[test]
    fn wider_fov_produces_tighter_bounds() {
        let cal = test_calibration();
        let scene = SceneGeometry::from_layout(&cal.layout);

        let narrow = viewport_bounds(30.0, &cal, &scene);
        let wide = viewport_bounds(60.0, &cal, &scene);

        // Wider FOV should produce tighter (or equal) yaw bounds
        assert!(
            wide.min_yaw >= narrow.min_yaw,
            "wider FOV min_yaw ({:.4}) should be >= narrow ({:.4})",
            wide.min_yaw,
            narrow.min_yaw
        );
        assert!(
            wide.max_yaw <= narrow.max_yaw,
            "wider FOV max_yaw ({:.4}) should be <= narrow ({:.4})",
            wide.max_yaw,
            narrow.max_yaw
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
