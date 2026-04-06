//! 3D geometric model and reprojection error objective.
//!
//! Ports the v1 Python position optimization math to Rust. Two camera
//! planes form an L-shape in 3D space with a virtual camera at the corner.
//! The optimizer adjusts 5-6 parameters to minimize the total reprojection
//! error between matched feature point pairs.
//!
//! ## Reprojection Error
//!
//! Instead of measuring the angle between rays (v1 approach), we project
//! both rays onto a virtual image plane perpendicular to the camera's
//! viewing direction. The 2D distance on this plane directly corresponds
//! to what the user sees in the stitched output. This gives proper
//! weighting based on position in the rendered panorama.
//!
//! ## Coordinate Convention
//!
//! - Left plane (x-plane): 2D `(x, y)` maps to 3D `(x, -y, 0)`
//! - Right plane (z-plane): 2D `(z, y)` maps to 3D `(0, -y, -z)`
//! - Camera sits at `[cam_d, 0, cam_d]` on the x=z bisector
//!
//! ## Left/Right Swap
//!
//! Following the v1 convention (processing.py:693), the *right camera*
//! points are placed on the x-plane (left in optimizer space) and the
//! *left camera* points on the z-plane. This swap is handled internally
//! by the public API so callers pass frames in natural left/right order.

use nalgebra::{Matrix3, Vector3};

use crate::types::MatchedPoint;

/// Plane width in the geometric model (normalized to 1.0).
pub const PLANE_WIDTH: f64 = 1.0;

/// Map a 2D point on the left plane (x-plane) to 3D.
///
/// `(x, y) -> (x, -y, 0)`
#[inline]
fn to_3d_x_plane(p: [f64; 2]) -> Vector3<f64> {
    Vector3::new(p[0], -p[1], 0.0)
}

/// Map a 2D point on the right plane (z-plane) to 3D.
///
/// `(z, y) -> (0, -y, -z)`
#[inline]
fn to_3d_z_plane(p: [f64; 2]) -> Vector3<f64> {
    Vector3::new(0.0, -p[1], -p[0])
}

/// Build a 3D rotation matrix from Euler angles (extrinsic ZYX order).
///
/// Equivalent to `Rz @ Ry @ Rx` matching the v1 Python implementation.
fn rotation_matrix(rx: f64, ry: f64, rz: f64) -> Matrix3<f64> {
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();

    #[rustfmt::skip]
    let m = Matrix3::new(
        cz * cy,    cz * sy * sx - sz * cx,    cz * sy * cx + sz * sx,
        sz * cy,    sz * sy * sx + cz * cx,    sz * sy * cx - cz * sx,
        -sy,        cy * sx,                    cy * cx,
    );
    m
}

/// Parameters for the 5-variable optimization.
///
/// These are the values COBYLA optimizes over.
#[derive(Debug, Clone, Copy)]
pub struct OptParams {
    /// Y-axis translation of the right plane (corrects vertical misalignment).
    pub x_ty: f64,
    /// Overlap ratio between the two planes `[0, 1]`.
    pub intersect: f64,
    /// Camera distance from origin along both X and Z axes.
    pub cam_d: f64,
    /// Z-axis rotation of the right plane (radians).
    pub x_rz: f64,
    /// X-axis rotation of the left plane (radians).
    pub z_rx: f64,
    /// Z-axis rotation of the left plane (radians) - the 6th parameter.
    /// `None` when running in 5-param mode.
    pub z_rz: Option<f64>,
}

impl OptParams {
    /// Unpack from a 5-element COBYLA parameter vector.
    ///
    /// Order: `[x_ty, intersect, cam_d, x_rz, z_rx]`
    pub fn from_5param(x: &[f64]) -> Self {
        Self {
            x_ty: x[0],
            intersect: x[1],
            cam_d: x[2],
            x_rz: x[3],
            z_rx: x[4],
            z_rz: None,
        }
    }

    /// Unpack from a 6-element COBYLA parameter vector.
    ///
    /// Order: `[x_ty, intersect, cam_d, x_rz, z_rx, z_rz]`
    pub fn from_6param(x: &[f64]) -> Self {
        Self {
            x_ty: x[0],
            intersect: x[1],
            cam_d: x[2],
            x_rz: x[3],
            z_rx: x[4],
            z_rz: Some(x[5]),
        }
    }

    /// Pack into a 5-element vector (ignoring `z_rz`).
    pub fn to_5param(&self) -> [f64; 5] {
        [self.x_ty, self.intersect, self.cam_d, self.x_rz, self.z_rx]
    }

    /// Pack into a 6-element vector.
    pub fn to_6param(&self) -> [f64; 6] {
        [
            self.x_ty,
            self.intersect,
            self.cam_d,
            self.x_rz,
            self.z_rx,
            self.z_rz.unwrap_or(0.0),
        ]
    }
}

/// Apply geometric transformations to matched point pairs.
///
/// Converts 2D plane coordinates to 3D, applies rotations and translations
/// based on the optimization parameters, and returns the transformed 3D
/// positions for both planes.
///
/// The left/right swap is already baked into `MatchedPoint`: `.left` is
/// on the x-plane (right camera) and `.right` is on the z-plane (left camera).
pub fn apply_transformations(
    points: &[MatchedPoint],
    params: &OptParams,
) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
    let half_offset = PLANE_WIDTH / 2.0 * (1.0 - params.intersect);

    // Left plane (x-plane): Z-rotation by x_rz, translated along X
    let r_x_plane = rotation_matrix(0.0, 0.0, params.x_rz);
    let t_x_plane = Vector3::new(half_offset, params.x_ty, 0.0);

    // Right plane (z-plane): X-rotation by z_rx, optionally Z-rotation by z_rz,
    // translated along Z
    let z_rz = params.z_rz.unwrap_or(0.0);
    let r_z_plane = rotation_matrix(params.z_rx, 0.0, z_rz);
    let t_z_plane = Vector3::new(0.0, 0.0, half_offset);

    let mut x_transformed = Vec::with_capacity(points.len());
    let mut z_transformed = Vec::with_capacity(points.len());

    for mp in points {
        let x_3d = to_3d_x_plane(mp.left);
        let z_3d = to_3d_z_plane(mp.right);

        // v1 uses `point @ R.T` (row-vector convention).
        // nalgebra uses column vectors, so `R * point` is equivalent.
        x_transformed.push(r_x_plane * x_3d + t_x_plane);
        z_transformed.push(r_z_plane * z_3d + t_z_plane);
    }

    (x_transformed, z_transformed)
}

/// Symmetric plane-to-plane reprojection error (sum of squared distances).
///
/// For each matched pair, shoots a ray from the camera through one point
/// and measures where it hits the other plane. The squared distance between
/// the intersection and the actual matched point is the error.
///
/// Both directions are computed (x-plane → z-plane and z-plane → x-plane)
/// for symmetry. This is the standard reprojection metric used in bundle
/// adjustment and has a proper global minimum in cam_d, unlike angular
/// error which is degenerate.
///
/// # Why this works
///
/// The x-plane stays at z=0 and the z-plane stays at x=0 even after
/// their respective rotations (Rz around Z-axis preserves z=0, Rx around
/// X-axis preserves x=0). So ray-plane intersection is a simple division.
pub fn reprojection_error(points: &[MatchedPoint], params: &OptParams) -> f64 {
    let camera = Vector3::new(params.cam_d, 0.0, params.cam_d);
    let (x_pts, z_pts) = apply_transformations(points, params);

    let mut total = 0.0;
    for (x_pt, z_pt) in x_pts.iter().zip(z_pts.iter()) {
        // Forward: ray from camera through x_pt, intersect z-plane (x=0)
        let dir_x = x_pt - camera;
        if dir_x.x.abs() > 1e-15 {
            let t = -camera.x / dir_x.x;
            if t > 0.0 {
                let hit = camera + t * dir_x;
                let dy = hit.y - z_pt.y;
                let dz = hit.z - z_pt.z;
                total += dy * dy + dz * dz;
            } else {
                total += 1e6;
            }
        }

        // Backward: ray from camera through z_pt, intersect x-plane (z=0)
        let dir_z = z_pt - camera;
        if dir_z.z.abs() > 1e-15 {
            let t = -camera.z / dir_z.z;
            if t > 0.0 {
                let hit = camera + t * dir_z;
                let dx = hit.x - x_pt.x;
                let dy = hit.y - x_pt.y;
                total += dx * dx + dy * dy;
            } else {
                total += 1e6;
            }
        }
    }
    total
}

/// Compute per-point symmetric reprojection errors.
///
/// Returns a vector of individual error values (sum of forward + backward
/// squared distances per pair). Used for outlier detection and trimmed
/// evaluation.
pub fn per_point_reprojection_error(points: &[MatchedPoint], params: &OptParams) -> Vec<f64> {
    let camera = Vector3::new(params.cam_d, 0.0, params.cam_d);
    let (x_pts, z_pts) = apply_transformations(points, params);

    x_pts
        .iter()
        .zip(z_pts.iter())
        .map(|(x_pt, z_pt)| {
            let mut err = 0.0;

            let dir_x = x_pt - camera;
            if dir_x.x.abs() > 1e-15 {
                let t = -camera.x / dir_x.x;
                if t > 0.0 {
                    let hit = camera + t * dir_x;
                    let dy = hit.y - z_pt.y;
                    let dz = hit.z - z_pt.z;
                    err += dy * dy + dz * dz;
                } else {
                    return 1e6;
                }
            }

            let dir_z = z_pt - camera;
            if dir_z.z.abs() > 1e-15 {
                let t = -camera.z / dir_z.z;
                if t > 0.0 {
                    let hit = camera + t * dir_z;
                    let dx = hit.x - x_pt.x;
                    let dy = hit.y - x_pt.y;
                    err += dx * dx + dy * dy;
                } else {
                    return 1e6;
                }
            }

            err
        })
        .collect()
}

/// Trimmed reprojection error for robust evaluation.
///
/// Computes per-point errors, sorts them, drops the worst `trim_fraction`
/// (e.g., 0.2 = drop worst 20%), and sums the remaining. This makes
/// the evaluation robust to outlier points that would otherwise steer
/// the optimizer toward large-rotation solutions.
pub fn trimmed_reprojection_error(
    points: &[MatchedPoint],
    params: &OptParams,
    trim_fraction: f64,
) -> f64 {
    let mut errors = per_point_reprojection_error(points, params);
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let keep = ((1.0 - trim_fraction) * errors.len() as f64).ceil() as usize;
    let keep = keep.min(errors.len()).max(1);
    errors[..keep].iter().sum()
}

/// Compute the total angular error between matched point pairs.
///
/// Legacy objective from v1. Kept for diagnostic comparison.
/// For each pair, computes the angle between the direction vectors from
/// the virtual camera to each transformed 3D point.
pub fn angular_error(points: &[MatchedPoint], params: &OptParams) -> f64 {
    let camera = Vector3::new(params.cam_d, 0.0, params.cam_d);
    let (x_pts, z_pts) = apply_transformations(points, params);

    let mut total = 0.0;
    for (x_pt, z_pt) in x_pts.iter().zip(z_pts.iter()) {
        let v_x = (x_pt - camera).normalize();
        let v_z = (z_pt - camera).normalize();
        let dot = v_x.dot(&v_z).clamp(-1.0, 1.0);
        total += dot.acos();
    }
    total
}

/// 5-parameter objective function for COBYLA (reprojection error).
///
/// Uses symmetric plane-to-plane squared reprojection error which
/// has a proper global minimum in all parameters (including cam_d).
///
/// Parameter vector order: `[x_ty, intersect, cam_d, x_rz, z_rx]`
pub fn objective_5param(x: &[f64], points: &mut &[MatchedPoint]) -> f64 {
    let params = OptParams::from_5param(x);
    reprojection_error(points, &params)
}

/// 6-parameter objective function for COBYLA (reprojection error).
///
/// Parameter vector order: `[x_ty, intersect, cam_d, x_rz, z_rx, z_rz]`
pub fn objective_6param(x: &[f64], points: &mut &[MatchedPoint]) -> f64 {
    let params = OptParams::from_6param(x);
    reprojection_error(points, &params)
}

/// Normalize a pixel coordinate to plane coordinates.
///
/// Matches v1's `_normalize_to_plane_coords`: x maps to `[-0.5, 0.5]`,
/// y maps to `[-h/(2w), h/(2w)]` preserving the image aspect ratio.
pub fn normalize_to_plane(px: f64, py: f64, img_w: u32, img_h: u32) -> [f64; 2] {
    let w = img_w as f64;
    let h = img_h as f64;
    [
        (px / w - 0.5) * PLANE_WIDTH,
        (py / h - 0.5) * PLANE_WIDTH * (h / w),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn to_3d_x_plane_maps_correctly() {
        let p = to_3d_x_plane([0.3, 0.1]);
        assert_abs_diff_eq!(p.x, 0.3, epsilon = 1e-10);
        assert_abs_diff_eq!(p.y, -0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(p.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn to_3d_z_plane_maps_correctly() {
        let p = to_3d_z_plane([0.2, 0.05]);
        assert_abs_diff_eq!(p.x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p.y, -0.05, epsilon = 1e-10);
        assert_abs_diff_eq!(p.z, -0.2, epsilon = 1e-10);
    }

    #[test]
    fn identity_rotation_is_identity() {
        let r = rotation_matrix(0.0, 0.0, 0.0);
        let identity = Matrix3::identity();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(r[(i, j)], identity[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn rotation_z_90_degrees() {
        let r = rotation_matrix(0.0, 0.0, std::f64::consts::FRAC_PI_2);
        let v = Vector3::new(1.0, 0.0, 0.0);
        let rotated = r * v;
        assert_abs_diff_eq!(rotated.x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rotated.y, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rotated.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn normalize_center_pixel_to_origin() {
        let [x, y] = normalize_to_plane(960.0, 540.0, 1920, 1080);
        assert_abs_diff_eq!(x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn normalize_top_left_pixel() {
        let [x, y] = normalize_to_plane(0.0, 0.0, 1920, 1080);
        assert_abs_diff_eq!(x, -0.5, epsilon = 1e-10);
        // y = (0/1080 - 0.5) * (1080/1920) = -0.5 * 0.5625 = -0.28125
        assert_abs_diff_eq!(y, -0.28125, epsilon = 1e-10);
    }

    #[test]
    fn reprojection_error_zero_for_perfect_alignment() {
        // With full overlap (intersect=1), both planes meet at origin.
        // Points at (0,0) on both planes map to (0,0,0) in 3D.
        // Both rays from camera hit the same point, so reprojection error = 0.
        let points = vec![MatchedPoint {
            left: [0.0, 0.0],
            right: [0.0, 0.0],
        }];
        let params = OptParams {
            x_ty: 0.0,
            intersect: 1.0,
            cam_d: 0.25,
            x_rz: 0.0,
            z_rx: 0.0,
            z_rz: None,
        };
        let err = reprojection_error(&points, &params);
        assert_abs_diff_eq!(err, 0.0, epsilon = 1e-6);

        // Angular error should also be zero
        let ang = angular_error(&points, &params);
        assert_abs_diff_eq!(ang, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn reprojection_error_increases_with_misalignment() {
        let points = vec![
            MatchedPoint {
                left: [0.1, 0.0],
                right: [0.1, 0.0],
            },
            MatchedPoint {
                left: [0.2, 0.05],
                right: [0.2, 0.05],
            },
        ];

        let good_params = OptParams {
            x_ty: 0.0,
            intersect: 0.5,
            cam_d: 0.25,
            x_rz: 0.0,
            z_rx: 0.0,
            z_rz: None,
        };

        let bad_params = OptParams {
            x_ty: 0.3,
            intersect: 0.5,
            cam_d: 0.25,
            x_rz: 0.0,
            z_rx: 0.0,
            z_rz: None,
        };

        let good_err = reprojection_error(&points, &good_params);
        let bad_err = reprojection_error(&points, &bad_params);
        assert!(
            bad_err > good_err,
            "misaligned params should have higher error: {bad_err} vs {good_err}"
        );
    }

    #[test]
    fn reprojection_error_and_angular_error_agree_on_ordering() {
        // Both metrics should agree that good params are better than bad.
        let points = vec![
            MatchedPoint {
                left: [0.1, 0.0],
                right: [0.1, 0.0],
            },
            MatchedPoint {
                left: [-0.1, 0.05],
                right: [-0.1, 0.05],
            },
        ];

        let good = OptParams {
            x_ty: 0.0,
            intersect: 0.5,
            cam_d: 0.25,
            x_rz: 0.0,
            z_rx: 0.0,
            z_rz: None,
        };

        let bad = OptParams {
            x_ty: 0.3,
            intersect: 0.5,
            cam_d: 0.25,
            x_rz: 0.2,
            z_rx: 0.0,
            z_rz: None,
        };

        let reproj_good = reprojection_error(&points, &good);
        let reproj_bad = reprojection_error(&points, &bad);
        let ang_good = angular_error(&points, &good);
        let ang_bad = angular_error(&points, &bad);

        assert!(reproj_bad > reproj_good);
        assert!(ang_bad > ang_good);
    }

    #[test]
    fn param_pack_unpack_roundtrip_5() {
        let params = OptParams {
            x_ty: 0.01,
            intersect: 0.55,
            cam_d: 0.24,
            x_rz: 0.008,
            z_rx: -0.004,
            z_rz: None,
        };
        let packed = params.to_5param();
        let unpacked = OptParams::from_5param(&packed);
        assert_abs_diff_eq!(unpacked.x_ty, params.x_ty, epsilon = 1e-15);
        assert_abs_diff_eq!(unpacked.intersect, params.intersect, epsilon = 1e-15);
        assert_abs_diff_eq!(unpacked.cam_d, params.cam_d, epsilon = 1e-15);
    }

    #[test]
    fn param_pack_unpack_roundtrip_6() {
        let params = OptParams {
            x_ty: 0.01,
            intersect: 0.55,
            cam_d: 0.24,
            x_rz: 0.008,
            z_rx: -0.004,
            z_rz: Some(0.003),
        };
        let packed = params.to_6param();
        let unpacked = OptParams::from_6param(&packed);
        assert_abs_diff_eq!(
            unpacked.z_rz.unwrap(),
            params.z_rz.unwrap(),
            epsilon = 1e-15
        );
    }
}
