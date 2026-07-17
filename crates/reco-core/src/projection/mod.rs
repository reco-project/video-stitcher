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

mod coverage;
pub(crate) mod cylinder;
mod geometry;
pub(crate) mod l_shape;

// Re-export coverage types so external code can still use
// `crate::projection::CoverageBoundary` etc.
pub use coverage::{ClampedPosition, CoverageBoundary, PanoramaExtent};
pub use cylinder::Cylinder;
pub use l_shape::{DEFAULT_BLEND_WIDTH, LShape};

// Re-export geometry utility.
pub use geometry::point_in_polygon;

use crate::calibration::{Calibration, Framing, Lens};
use crate::geometry::CameraId;
use crate::geometry::Pose;
use crate::geometry::VirtualCamera;
use crate::stitch::{BlendRule, SurfaceMap};
use l_shape::PlaneScene;

use nalgebra::{Point3, Vector3};

// ---------------------------------------------------------------------------
// The Projection trait: L1 geometry dispatch.
// ---------------------------------------------------------------------------
//
// The calibration document owns the projection: `Topology::projection()`
// borrows the topology's parameter struct as `&dyn Projection`, so
// alt-projections (the mono cylinder today, N-camera later) are drop-in
// additions without reshaping the core API and without a second object
// to keep in sync. The trait dispatches the CPU surface maps, the GPU
// program descriptor, coverage construction and the virtual-camera
// basis; the detection-side forward maps (`camera_to_panorama`) are
// the next fold-in - their `CameraId`/`Pose` currency already lives in
// the geometry leaf, so nothing blocks the move.

/// One frame's geometry question, bundled: the document, the output
/// viewport, and the pose. The pose carries fov alongside yaw/pitch -
/// [`ViewportSize`](crate::render::viewport::ViewportSize) is
/// zoom-free, so there is exactly one fov per frame and a second,
/// disagreeing copy is unrepresentable.
pub struct ProjectionContext<'a> {
    /// The calibration document (lenses, topology, framing).
    pub calibration: &'a Calibration,
    /// Output viewport geometry (dimensions).
    pub viewport_size: &'a crate::render::viewport::ViewportSize,
    /// The render pose: yaw/pitch pan plus fov zoom, one bundle.
    pub pose: Pose,
}

/// A panoramic projection geometry.
///
/// Implemented by concrete projections (today's 2-plane L-shape, the
/// mono cylinder next). Dispatched dynamically by StitchCore so swapping
/// projections at session construction time does not require
/// recompilation. The CPU geometry dispatches through
/// [`surface_maps`](Self::surface_maps); the GPU program dispatch lands
/// with the projection-shader step.
///
/// # Bounds
///
/// `Send + Sync` because StitchCore stores projections behind a
/// shared reference and the render thread reads them concurrently.
pub trait Projection: Send + Sync {
    /// Short human-readable name for logs + diagnostic bundles.
    fn name(&self) -> &'static str;

    /// Number of input cameras this projection consumes. 1 for mono,
    /// 2 for today's L-shape stereo, N>2 for future panoramic rigs.
    fn camera_count(&self) -> usize;

    /// The virtual camera basis - the eye position plus the yaw/pitch
    /// decomposition axes every view matrix and pose conversion is
    /// built from. The one piece of scene state every projection has,
    /// plane-backed or not (the mono cylinder's camera sits at the
    /// `[0, 0, 1]` convention).
    fn virtual_camera(&self, framing: &Framing) -> VirtualCamera;

    /// The ordered surface list for one frame: each surface's inverse
    /// map paired with how it blends over the surfaces before it.
    /// Surface `i` samples source camera `i`; the first surface lays
    /// the base. The CPU composite drives these directly.
    fn surface_maps(&self, ctx: &ProjectionContext) -> Vec<(Box<dyn SurfaceMap>, BlendRule)>;

    /// The GPU program this projection composites with, or `None` when
    /// the projection has no GPU pass yet (CPU-only). `None` fails
    /// [`GpuExecutor`](crate::stitch::GpuExecutor) construction with a
    /// typed error instead of binding a placeholder that renders
    /// garbage.
    ///
    /// The GPU dual of [`surface_maps`](Self::surface_maps). CPU/GPU
    /// agreement is pinned by the stitch oracle, whose fixtures are
    /// L-shape-only today: a projection growing a GPU program must
    /// bring its own agreement test with it.
    #[cfg(feature = "gpu")]
    fn gpu_program(&self) -> Option<crate::render::GpuProgram>;

    /// Build the coverage boundary for this projection's panorama.
    ///
    /// No default on purpose: representation and clamp are one coupled
    /// unit each projection must own (the L-shape samples its plane
    /// edges; the cylinder is analytic). A defaulted implementation
    /// would silently hand a new projection another one's boundary.
    fn coverage(&self, calibration: &Calibration) -> CoverageBoundary;
}

/// Maximum Newton-Raphson iterations for KB4 inverse distortion.
const MAX_ITERATIONS: usize = 20;
/// Convergence threshold for Newton-Raphson.
const CONVERGENCE_EPS: f64 = 1e-10;

/// Map a detection in camera pixel space to the yaw/pitch needed to
/// center the virtual camera on it.
///
/// `norm_x` and `norm_y` are in normalized `[0.0, 1.0]` image coordinates
/// (as returned by [`Detection`](crate::detect::detector::Detection)).
///
/// Returns `None` if the inverse distortion fails to converge (rare,
/// indicates an extreme point far outside the valid lens area).
///
/// Returns `None` for plane-less topologies: the forward map inverts
/// the L-shape's plane rasterization, and mono detection mapping is
/// its own follow-up.
///
/// # Example
///
/// ```rust
/// use reco_core::projection::camera_to_panorama;
/// use reco_core::geometry::CameraId;
/// use reco_core::calibration::Calibration;
///
/// # fn example(cal: &Calibration) {
/// if let Some(pos) = camera_to_panorama(CameraId::Left, 0.5, 0.5, cal) {
///     println!("Center of left camera maps to yaw={:.3}, pitch={:.3}", pos.yaw, pos.pitch);
/// }
/// # }
/// ```
pub fn camera_to_panorama(
    camera: CameraId,
    norm_x: f32,
    norm_y: f32,
    calibration: &Calibration,
) -> Option<Pose> {
    let topology = calibration.topology.l_shape()?;
    let scene = topology.scene(&calibration.framing, calibration.lenses[0].aspect());
    camera_to_panorama_in_scene(camera, norm_x, norm_y, calibration, &scene)
}

/// [`camera_to_panorama`] against an already-derived plane scene - the
/// coverage sampler maps thousands of edge points against one scene.
pub(crate) fn camera_to_panorama_in_scene(
    camera: CameraId,
    norm_x: f32,
    norm_y: f32,
    calibration: &Calibration,
    scene: &PlaneScene,
) -> Option<Pose> {
    let params = match camera {
        CameraId::Left => &calibration.lenses[0],
        CameraId::Right => &calibration.lenses[1],
    };

    // Step 1: Inverse fisheye - camera pixel [0,1] -> plane UV (extended space)
    let plane_uv = inverse_fisheye(norm_x as f64, norm_y as f64, params)?;

    // Step 2: Plane UV -> 3D world point
    let world_point = plane_uv_to_world(plane_uv, camera, scene);

    // Step 3: World point -> yaw/pitch
    let dir = (world_point - Point3::from(Vector3::from(scene.camera_position))).normalize();
    Some(direction_to_yaw_pitch(&dir, &scene.camera_position))
}

// ---- Internal functions ----

/// Forward KB4 fisheye: undistorted plane UV -> distorted camera pixel [0,1].
///
/// Mirror of [`inverse_fisheye`] in the same normalized-intrinsic
/// convention and the same extended-UV plane space (the shader's
/// `uv * 2.0 - 0.5` remap output). The polynomial delegates to
/// `reco_core::lens::kb4`, same canonical source as the
/// Newton-Raphson step in [`inverse_fisheye`]. Test-only: the
/// production inverse path (`inverse_fisheye`) is the live direction;
/// this exists to round-trip-verify it.
#[cfg(test)]
fn forward_fisheye(uv_x: f64, uv_y: f64, params: &Lens) -> (f64, f64) {
    let w = params.width as f64;
    let h = params.height as f64;
    let fx = params.fx / w;
    let fy = params.fy / h;
    let cx = params.cx / w;
    let cy = params.cy / h;

    let x = (uv_x - cx) / fx;
    let y = (uv_y - cy) / fy;
    let r = (x * x + y * y).sqrt();

    if r < 1e-12 {
        return (cx, cy);
    }

    let scale = crate::lens::kb4::kb4_forward_scale(r, &params.distortion);
    (fx * x * scale + cx, fy * y * scale + cy)
}

/// Inverse KB4 fisheye: distorted camera pixel [0,1] -> undistorted plane UV.
///
/// Inverts the forward KB4 model used in the shader:
/// ```text
/// theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
/// ```
/// Uses Newton-Raphson to solve for theta given theta_d.
fn inverse_fisheye(dist_x: f64, dist_y: f64, params: &Lens) -> Option<(f64, f64)> {
    let w = params.width as f64;
    let h = params.height as f64;
    let fx = params.fx / w;
    let fy = params.fy / h;
    let cx = params.cx / w;
    let cy = params.cy / h;
    let k = params.distortion;

    // Normalized distorted coordinates
    let dx = (dist_x - cx) / fx;
    let dy = (dist_y - cy) / fy;
    let theta_d = (dx * dx + dy * dy).sqrt();

    if theta_d < 1e-12 {
        // At the optical center - no distortion
        return Some((cx, cy));
    }

    // Newton-Raphson: solve f(theta) = theta_d_poly(theta) - theta_d = 0, where
    // theta_d_poly lives in `reco_core::lens::kb4` (SYNC_WITH WGSL).
    let mut theta = theta_d; // initial guess
    for _ in 0..MAX_ITERATIONS {
        let f = crate::lens::kb4::theta_d(theta, &k) - theta_d;
        let f_prime = crate::lens::kb4::theta_d_prime(theta, &k);

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
    let r = theta.tan(); // theta = atan(r) -> r = tan(theta)
    let scale = if theta.abs() < 1e-12 {
        1.0
    } else {
        theta_d / r
    };

    // Guard against Inf/NaN from degenerate theta (e.g. theta near pi/2
    // where tan diverges, or numerical edge cases).
    if !scale.is_finite() {
        return None;
    }

    let x = dx / scale;
    let y = dy / scale;

    // Plane UV in the extended [-0.5, 1.5] space used by the shader
    let uv_x = fx * x + cx;
    let uv_y = fy * y + cy;

    Some((uv_x, uv_y))
}

/// Convert a plane UV (in extended shader space) to a 3D world point.
fn plane_uv_to_world(uv: (f64, f64), camera: CameraId, scene: &PlaneScene) -> Point3<f32> {
    // Extended UV -> texture UV [0,1]
    let tex_u = ((uv.0 + 0.5) / 2.0) as f32;
    let tex_v = ((uv.1 + 0.5) / 2.0) as f32;

    // Texture UV -> local quad position (matches quad_vertices)
    let local_x = tex_u - 0.5;
    let local_y = (0.5 - tex_v) / scene.plane_aspect;

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
/// Thin wrapper over [`VirtualCamera::direction_to_yaw_pitch`] kept
/// for the existing call sites until they carry a `VirtualCamera`
/// directly. Panners, directors, and the render loop all share the
/// same basis through this path.
pub(crate) fn direction_to_yaw_pitch(dir: &Vector3<f32>, camera_position: &[f32; 3]) -> Pose {
    VirtualCamera::new(camera_position).direction_to_yaw_pitch(dir)
}

/// Exact inverse of [`direction_to_yaw_pitch`]. Test-only helper;
/// production paths call the method on [`VirtualCamera`] directly.
#[cfg(test)]
pub(crate) fn yaw_pitch_to_direction(
    yaw: f32,
    pitch: f32,
    camera_position: &[f32; 3],
) -> Vector3<f32> {
    VirtualCamera::new(camera_position).yaw_pitch_to_direction(yaw, pitch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{Calibration, Framing, Lens};
    use crate::projection::LShape;

    fn test_scene(cal: &Calibration) -> PlaneScene {
        cal.topology
            .l_shape()
            .unwrap()
            .scene(&cal.framing, cal.lenses[0].aspect())
    }

    fn test_calibration() -> Calibration {
        let cam = || {
            Lens::fisheye(
                3840,
                2160,
                1796.32,
                1797.22,
                1919.37,
                1063.17,
                [0.0342, 0.0677, -0.0741, 0.0299],
            )
        };
        Calibration::new(
            vec![cam(), cam()],
            LShape {
                intersect: 0.5446,
                x_ty: 0.00476,
                x_rz: 0.00753,
                z_rx: -0.00431,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            Framing {
                axis_offset: 0.2398,
                tilt: 0.0,
                roll: 0.0,
            },
        )
    }

    #[test]
    fn optical_center_maps_to_known_position() {
        let cal = test_calibration();

        // Optical center of the left camera (cx/w, cy/h)
        let cx = cal.lenses[0].cx as f32 / cal.lenses[0].width as f32;
        let cy = cal.lenses[0].cy as f32 / cal.lenses[0].height as f32;

        let pos = camera_to_panorama(CameraId::Left, cx, cy, &cal);
        assert!(pos.is_some(), "optical center should map successfully");
        let pos = pos.unwrap();
        // The optical center should produce a valid yaw/pitch (no NaN)
        assert!(pos.yaw.is_finite(), "yaw should be finite");
        assert!(pos.pitch.is_finite(), "pitch should be finite");
    }

    #[test]
    fn left_camera_left_edge_yaw_differs_from_center() {
        let cal = test_calibration();

        let center = camera_to_panorama(CameraId::Left, 0.5, 0.5, &cal).unwrap();
        let left_edge = camera_to_panorama(CameraId::Left, 0.1, 0.5, &cal).unwrap();

        // The left edge of the left camera image maps to a different
        // part of the panorama than the center; this test just
        // asserts the pipeline is position-sensitive and doesn't
        // collapse distinct image positions to the same yaw. Sign
        // conventions are validated end-to-end by visual review,
        // not here.
        assert!(
            (left_edge.yaw - center.yaw).abs() > 0.1,
            "left edge yaw ({:.4}) and center yaw ({:.4}) should differ by > 0.1 rad",
            left_edge.yaw,
            center.yaw
        );
    }

    #[test]
    fn right_camera_produces_different_yaw_than_left() {
        let cal = test_calibration();

        let left_center = camera_to_panorama(CameraId::Left, 0.5, 0.5, &cal).unwrap();
        let right_center = camera_to_panorama(CameraId::Right, 0.5, 0.5, &cal).unwrap();

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
    fn yaw_pitch_to_direction_roundtrips_with_direction_to_yaw_pitch() {
        // Step 1a: the two helpers must form an exact bijection on the
        // (yaw, pitch) grid used by panners and directors. All
        // shipping scenes set `camera_position = [d, 0, d]` (see
        // LShape::virtual_camera), so eye.y = 0 is the real invariant;
        // test positions honor that. Pitch stays clear of +-pi/2 where
        // yaw is undefined.
        let camera_positions: [[f32; 3]; 3] = [[0.24, 0.0, 0.24], [0.3, 0.0, 0.2], [0.1, 0.0, 0.5]];

        let yaw_steps = [-1.2_f32, -0.6, -0.2, 0.0, 0.2, 0.6, 1.2];
        let pitch_steps = [-0.9_f32, -0.4, -0.1, 0.0, 0.1, 0.4, 0.9];

        for cam in &camera_positions {
            for &yaw in &yaw_steps {
                for &pitch in &pitch_steps {
                    let dir = yaw_pitch_to_direction(yaw, pitch, cam);
                    let norm = dir.norm();
                    assert!(
                        (norm - 1.0).abs() < 1e-5,
                        "direction must be unit, got |dir| = {norm} for cam={cam:?} yaw={yaw} pitch={pitch}"
                    );

                    let pos = direction_to_yaw_pitch(&dir, cam);
                    assert!(
                        (pos.yaw - yaw).abs() < 1e-4,
                        "yaw mismatch for cam={cam:?}: sent {yaw}, got {} (dir={dir:?})",
                        pos.yaw
                    );
                    assert!(
                        (pos.pitch - pitch).abs() < 1e-4,
                        "pitch mismatch for cam={cam:?}: sent {pitch}, got {} (dir={dir:?})",
                        pos.pitch
                    );
                }
            }
        }
    }

    #[test]
    fn inverse_fisheye_roundtrips_with_forward_fisheye_on_pixel_grid() {
        // Step 1c: normalized distorted pixel -> extended plane UV ->
        // back to normalized distorted pixel must be the identity on
        // a 10x10 grid inside the valid image area. Realistic KB4
        // coefficients (the GoPro HERO10 4K test calibration) make
        // this representative of shipping workloads.
        let params = Lens::fisheye(
            3840,
            2160,
            1796.32,
            1797.22,
            1919.37,
            1063.17,
            [0.0342, 0.0677, -0.0741, 0.0299],
        );

        let steps = 10;
        // Stay inside [0.1, 0.9] to avoid extreme fisheye corners where
        // Newton-Raphson may refuse to converge (documented in
        // `inverse_fisheye`'s None return).
        let lo = 0.1_f64;
        let hi = 0.9_f64;

        for ix in 0..=steps {
            for iy in 0..=steps {
                let nx = lo + (hi - lo) * (ix as f64 / steps as f64);
                let ny = lo + (hi - lo) * (iy as f64 / steps as f64);

                let plane_uv = inverse_fisheye(nx, ny, &params)
                    .expect("inverse_fisheye should converge inside valid area");
                let (back_x, back_y) = forward_fisheye(plane_uv.0, plane_uv.1, &params);

                assert!(
                    (back_x - nx).abs() < 1e-6,
                    "x mismatch at ({nx}, {ny}): got {back_x}, plane_uv={plane_uv:?}"
                );
                assert!(
                    (back_y - ny).abs() < 1e-6,
                    "y mismatch at ({nx}, {ny}): got {back_y}, plane_uv={plane_uv:?}"
                );
            }
        }
    }

    #[test]
    fn inverse_fisheye_roundtrip_at_center() {
        let params = Lens::fisheye(
            3840,
            2160,
            1796.32,
            1797.22,
            1919.37,
            1063.17,
            [0.0342, 0.0677, -0.0741, 0.0299],
        );

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
    fn zero_distortion_produces_identity_mapping() {
        let params = Lens::fisheye(1920, 1080, 960.0, 540.0, 960.0, 540.0, [0.0, 0.0, 0.0, 0.0]);

        // With zero distortion and fx=width/2, cx=width/2, the mapping
        // should be close to identity
        let result = inverse_fisheye(0.5, 0.5, &params).unwrap();
        assert!(
            (result.0 - 0.5).abs() < 1e-6 && (result.1 - 0.5).abs() < 1e-6,
            "zero-distortion center should map to itself"
        );
    }

    // --- point_in_polygon tests ---

    /// Unit square: [0,0] -> [1,0] -> [1,1] -> [0,1].
    fn unit_square() -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }

    #[test]
    fn pip_center_of_square() {
        assert!(point_in_polygon([0.5, 0.5], &unit_square()));
    }

    #[test]
    fn pip_outside_square() {
        assert!(!point_in_polygon([1.5, 0.5], &unit_square()));
        assert!(!point_in_polygon([-0.1, 0.5], &unit_square()));
        assert!(!point_in_polygon([0.5, -0.1], &unit_square()));
        assert!(!point_in_polygon([0.5, 1.1], &unit_square()));
    }

    #[test]
    fn pip_triangle() {
        let triangle = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        // Inside
        assert!(point_in_polygon([0.5, 0.3], &triangle));
        // Outside (right of the triangle)
        assert!(!point_in_polygon([0.9, 0.8], &triangle));
    }

    #[test]
    fn pip_concave_l_shape() {
        // L-shaped polygon (concave):
        //   (0,0) -> (1,0) -> (1,0.5) -> (0.5,0.5) -> (0.5,1) -> (0,1)
        let l_shape = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [0.5, 0.5],
            [0.5, 1.0],
            [0.0, 1.0],
        ];
        // Inside the bottom-right arm
        assert!(point_in_polygon([0.75, 0.25], &l_shape));
        // Inside the top-left arm
        assert!(point_in_polygon([0.25, 0.75], &l_shape));
        // In the concave cutout (top-right) - should be outside
        assert!(!point_in_polygon([0.75, 0.75], &l_shape));
    }

    #[test]
    fn pip_degenerate_polygon() {
        // Fewer than 3 vertices: always false.
        assert!(!point_in_polygon([0.5, 0.5], &[]));
        assert!(!point_in_polygon([0.5, 0.5], &[[0.0, 0.0]]));
        assert!(!point_in_polygon([0.5, 0.5], &[[0.0, 0.0], [1.0, 1.0]]));
    }

    #[test]
    fn pip_near_edge_of_square() {
        // Just inside the edge
        assert!(point_in_polygon([0.001, 0.5], &unit_square()));
        assert!(point_in_polygon([0.999, 0.5], &unit_square()));
    }

    #[test]
    fn coverage_yaw_and_pitch_ranges_match_internal_state() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_l_shape(&cal, &scene);

        let (yaw_min, yaw_max) = coverage.yaw_range();
        assert!(yaw_min < yaw_max, "yaw range must be non-empty");
        assert!(yaw_min.is_finite() && yaw_max.is_finite());

        let (pitch_min, pitch_max) = coverage.pitch_range();
        assert_eq!(pitch_min, coverage.pitch_min);
        assert_eq!(pitch_max, coverage.pitch_max);
    }

    #[test]
    fn coverage_yaw_range_is_widest_slice_envelope() {
        // yaw_range() must be at least as wide as any yaw_range_at(pitch)
        // sample, since it's the envelope over all pitch slices.
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_l_shape(&cal, &scene);

        let (y_lo_global, y_hi_global) = coverage.yaw_range();
        let (p_lo, p_hi) = coverage.pitch_range();

        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let pitch = p_lo + t * (p_hi - p_lo);
            let (y_lo, y_hi) = coverage.yaw_range_at(pitch);
            if y_lo > y_hi {
                continue; // degenerate interpolation outside coverage
            }
            assert!(
                y_lo >= y_lo_global - 1e-4,
                "pitch {pitch} yaw lo {y_lo} below global {y_lo_global}"
            );
            assert!(
                y_hi <= y_hi_global + 1e-4,
                "pitch {pitch} yaw hi {y_hi} above global {y_hi_global}"
            );
        }
    }

    #[test]
    fn panorama_extent_normalize_is_in_range_at_corners() {
        let ext = PanoramaExtent {
            yaw_min: -0.5,
            yaw_max: 0.5,
            pitch_min: -0.3,
            pitch_max: 0.3,
        };
        assert_eq!(ext.yaw_span(), 1.0);
        assert_eq!(ext.pitch_span(), 0.6);

        let (u, v) = ext.normalize(-0.5, -0.3).unwrap();
        assert!((u - 0.0).abs() < 1e-6);
        assert!((v - 0.0).abs() < 1e-6);

        let (u, v) = ext.normalize(0.5, 0.3).unwrap();
        assert!((u - 1.0).abs() < 1e-6);
        assert!((v - 1.0).abs() < 1e-6);

        let (u, v) = ext.normalize(0.0, 0.0).unwrap();
        assert!((u - 0.5).abs() < 1e-6);
        assert!((v - 0.5).abs() < 1e-6);
    }

    #[test]
    fn panorama_extent_normalize_rejects_degenerate() {
        let ext = PanoramaExtent {
            yaw_min: 0.0,
            yaw_max: 0.0,
            pitch_min: 0.0,
            pitch_max: 0.0,
        };
        assert!(ext.normalize(0.0, 0.0).is_none());
    }

    // -- B-30 NaN-resilience regression tests --

    #[test]
    fn safe_clamp_rejects_nan_yaw() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_l_shape(&cal, &scene);
        let out = coverage.safe_clamp(f32::NAN, 0.0, 75.0, 16.0 / 9.0);
        assert!(out.yaw.is_finite(), "yaw must be finite, got {}", out.yaw);
        assert!(
            out.pitch.is_finite(),
            "pitch must be finite, got {}",
            out.pitch
        );
    }

    #[test]
    fn safe_clamp_rejects_nan_pitch() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_l_shape(&cal, &scene);
        let out = coverage.safe_clamp(0.0, f32::NAN, 75.0, 16.0 / 9.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
    }

    #[test]
    fn safe_clamp_rejects_nan_fov() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_l_shape(&cal, &scene);
        let out = coverage.safe_clamp(0.0, 0.0, f32::NAN, 16.0 / 9.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
    }

    #[test]
    fn safe_clamp_rejects_infinite_inputs() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_l_shape(&cal, &scene);
        let out = coverage.safe_clamp(f32::INFINITY, 0.0, 75.0, 16.0 / 9.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
        let out = coverage.safe_clamp(0.0, f32::NEG_INFINITY, 75.0, 16.0 / 9.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
    }

    // -- Projection trait tests --

    #[test]
    fn l_shape_projection_identifies_itself() {
        let cal = test_calibration();
        let p = cal.topology.projection();
        assert_eq!(p.name(), "l-shape-stereo-2camera");
        assert_eq!(p.camera_count(), 2);
    }

    #[test]
    fn projection_is_dyn_compatible() {
        // Core invariant: the engine reads `&dyn Projection` straight
        // out of the document (zero-copy), and injected overrides ride
        // as `Box<dyn Projection>`. Verify the trait bounds allow both
        // and that Send+Sync hold.
        let cal = test_calibration();
        let p: &dyn Projection = cal.topology.projection();
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<dyn Projection>();
        assert_eq!(p.camera_count(), 2);
    }

    #[test]
    fn l_shape_surface_maps_dispatch_two_ordered_surfaces() {
        // The L-shape emits exactly two surfaces: the opaque left base,
        // then the right fading in with the calibration's seam width.
        let cal = test_calibration();
        let config = crate::render::viewport::ViewportSize::default();
        let surfaces = cal.topology.projection().surface_maps(&ProjectionContext {
            calibration: &cal,
            viewport_size: &config,
            pose: Pose::default(),
        });
        assert_eq!(surfaces.len(), 2);
        assert_eq!(surfaces[0].1, crate::stitch::BlendRule::Opaque);
        let seam = cal.topology.l_shape().unwrap().blend_width;
        assert_eq!(
            surfaces[1].1,
            crate::stitch::BlendRule::Smoothstep(seam as f64)
        );
    }

    // ---- Cylinder projection ----------------------------------------------

    fn cylinder_cal() -> Calibration {
        Calibration::new(
            vec![Lens::flat(3840, 1080)],
            crate::projection::Cylinder::default(),
            Framing {
                axis_offset: 0.0,
                tilt: 0.0,
                roll: 0.0,
            },
        )
    }

    #[test]
    fn cylindrical_projection_reports_mono() {
        let cal = cylinder_cal();
        let p = cal.topology.projection();
        assert_eq!(p.name(), "cylindrical-mono-1camera");
        assert_eq!(
            p.camera_count(),
            1,
            "cylindrical projection consumes exactly one camera"
        );
    }

    #[test]
    fn cylindrical_surface_maps_emit_one_opaque_surface() {
        let cal = cylinder_cal();
        let config = crate::render::viewport::ViewportSize::default();
        let surfaces = cal.topology.projection().surface_maps(&ProjectionContext {
            calibration: &cal,
            viewport_size: &config,
            pose: Pose::default(),
        });
        assert_eq!(surfaces.len(), 1);
        assert_eq!(surfaces[0].1, crate::stitch::BlendRule::Opaque);
        assert!(
            surfaces[0]
                .0
                .sample_uv(config.width / 2, config.height / 2)
                .is_some(),
            "the straight-ahead pixel is covered"
        );
    }

    #[test]
    fn cylindrical_coverage_is_the_analytic_rectangle() {
        let cal = cylinder_cal();
        let coverage = cal.topology.projection().coverage(&cal);
        let (y_lo, y_hi) = coverage.yaw_range();
        assert!(
            (y_lo + std::f32::consts::FRAC_PI_2).abs() < 1e-5,
            "yaw_min {y_lo}"
        );
        assert!(
            (y_hi - std::f32::consts::FRAC_PI_2).abs() < 1e-5,
            "yaw_max {y_hi}"
        );
        // pitch half-extent = atan((h/2)/r) for h = the source's 1080
        // pixel height (the omitted-video_height default), r = 2400.
        let expected = ((1080.0f64 * 0.5) / 2400.0).atan() as f32;
        assert!((coverage.pitch_max - expected).abs() < 1e-6);
        assert!((coverage.pitch_min + expected).abs() < 1e-6);
    }

    #[test]
    fn topology_projection_resolves_the_matching_variant() {
        let l = test_calibration();
        assert_eq!(l.topology.projection().camera_count(), 2);
        let c = cylinder_cal();
        assert_eq!(c.topology.projection().camera_count(), 1);
    }

    #[test]
    fn projection_dyn_dispatch_round_trip_with_mixed_camera_counts() {
        // Compile-time: the trait is object-safe, so one collection
        // holds impls with different `camera_count()` results - what
        // lets an injected `Box<dyn Projection>` override the document.
        // The assertions pin the per-projection counts and that the
        // diagnostic names stay distinct.
        let l = test_calibration().topology.l_shape().unwrap().clone();
        let projections: Vec<Box<dyn Projection>> = vec![
            Box::new(l),
            Box::new(crate::projection::Cylinder::default()),
        ];
        assert_eq!(projections[0].camera_count(), 2);
        assert_eq!(projections[1].camera_count(), 1);
        assert_ne!(projections[0].name(), projections[1].name());
    }

    #[test]
    fn projection_params_are_send_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<LShape>();
        assert_send_sync::<crate::projection::Cylinder>();
    }
}
