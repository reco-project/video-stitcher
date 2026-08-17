//! Panoramic projection geometry and coordinate mapping.
//!
//! Projection implementations bridge camera frames and virtual-camera
//! orientation, enabling:
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

use crate::calibration::{Calibration, Framing};
use crate::geometry::CameraIndex;
use crate::geometry::Pose;
use crate::geometry::VirtualCamera;
use crate::stitch::{BlendRule, SurfaceMap};

#[cfg(test)]
use nalgebra::Vector3;

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
// basis and camera-to-panorama mapping.

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
/// Implemented by concrete projections (the 2-plane L-shape and mono
/// cylinder today). Dispatched dynamically from the calibration document,
/// so selecting another topology does not require recompilation. The CPU
/// geometry dispatches through
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

    /// Map normalized coordinates from one input camera to panorama pose.
    ///
    /// Returns `None` when this projection has no detection mapping, the
    /// camera index is invalid, or the inverse mapping cannot be solved.
    /// The returned pose carries the default FOV because a mapped point has
    /// angular position but no zoom.
    fn camera_to_panorama(
        &self,
        camera: CameraIndex,
        norm_x: f32,
        norm_y: f32,
        calibration: &Calibration,
    ) -> Option<Pose>;

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

/// Decompose a direction vector into yaw/pitch relative to the virtual camera.
///
/// Thin wrapper over [`VirtualCamera::direction_to_yaw_pitch`] kept
/// for geometry consistency tests.
#[cfg(test)]
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
    use crate::projection::l_shape::PlaneScene;

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

    fn relocation_calibration() -> Calibration {
        let mut cal = test_calibration();
        // Non-default correction and rig orientation are deliberate: these
        // goldens capture the production behavior being relocated, including
        // its current inputs, without claiming renderer/mapping equivalence.
        cal.lenses[0].correction = 0.37;
        cal.lenses[1].correction = 0.63;
        cal.framing.tilt = 0.21;
        cal.framing.roll = -0.08;
        let topology = cal.topology.l_shape_mut().unwrap();
        topology.x_rx = 0.025;
        topology.z_rz = -0.019;
        cal
    }

    fn assert_geometry_close(actual: f32, expected: f32, label: &str) {
        const EPSILON_RADIANS: f32 = 1e-6;
        assert!(
            (actual - expected).abs() <= EPSILON_RADIANS,
            "{label}: expected {expected}, got {actual}"
        );
    }

    fn map_camera(
        calibration: &Calibration,
        camera: CameraIndex,
        norm_x: f32,
        norm_y: f32,
    ) -> Option<Pose> {
        calibration
            .topology
            .projection()
            .camera_to_panorama(camera, norm_x, norm_y, calibration)
    }

    #[test]
    fn camera_mapping_matches_pre_relocation_goldens() {
        let cal = relocation_calibration();

        for (camera, x, y, expected_yaw, expected_pitch) in [
            (0, 0.23, 0.41, 1.3070116, 0.10282527),
            (0, 0.68, 0.77, 0.36100596, -0.30260217),
            (1, 0.31, 0.28, -0.33768472, 0.2593831),
            (1, 0.74, 0.62, -1.24458, -0.11580661),
        ] {
            let pose = map_camera(&cal, camera, x, y).unwrap();
            assert_geometry_close(pose.yaw, expected_yaw, "mapped yaw");
            assert_geometry_close(pose.pitch, expected_pitch, "mapped pitch");
            assert_eq!(pose.fov_degrees, 75.0);
        }

        assert!(map_camera(&cal, 2, 0.5, 0.5).is_none());
    }

    #[test]
    fn l_shape_coverage_matches_pre_relocation_goldens() {
        let cal = relocation_calibration();
        let coverage = cal.topology.projection().coverage(&cal);

        let (yaw_min, yaw_max) = coverage.yaw_range();
        assert_geometry_close(yaw_min, -1.8064561, "global yaw minimum");
        assert_geometry_close(yaw_max, 1.8043411, "global yaw maximum");

        let (pitch_min, pitch_max) = coverage.pitch_range();
        assert_geometry_close(pitch_min, -0.55399376, "global pitch minimum");
        assert_geometry_close(pitch_max, 0.5772345, "global pitch maximum");

        for (pitch, expected_min, expected_max) in [
            (-0.2, -1.7421114, 1.7409195),
            (0.0, -1.7302064, 1.7299159),
            (0.2, -1.7428147, 1.7436572),
        ] {
            let (yaw_min, yaw_max) = coverage.yaw_range_at(pitch);
            assert_geometry_close(yaw_min, expected_min, "slice yaw minimum");
            assert_geometry_close(yaw_max, expected_max, "slice yaw maximum");
        }

        let clamped = coverage.safe_clamp(2.0, 0.3, 55.0, 16.0 / 9.0);
        assert_geometry_close(clamped.yaw, 0.9272614, "clamped yaw");
        assert_geometry_close(clamped.pitch, 0.0116203725, "clamped pitch");

        // The selected 10th-percentile sample sits on a repeated plateau,
        // so this is stable against ordering among equal samples.
        assert!((coverage.max_fov_degrees() - 56.388702).abs() <= 1e-5);
    }

    #[test]
    fn optical_center_maps_to_known_position() {
        let cal = test_calibration();

        // Optical center of the left camera (cx/w, cy/h)
        let cx = cal.lenses[0].cx as f32 / cal.lenses[0].width as f32;
        let cy = cal.lenses[0].cy as f32 / cal.lenses[0].height as f32;

        let pos = map_camera(&cal, 0, cx, cy);
        assert!(pos.is_some(), "optical center should map successfully");
        let pos = pos.unwrap();
        // The optical center should produce a valid yaw/pitch (no NaN)
        assert!(pos.yaw.is_finite(), "yaw should be finite");
        assert!(pos.pitch.is_finite(), "pitch should be finite");
    }

    #[test]
    fn left_camera_left_edge_yaw_differs_from_center() {
        let cal = test_calibration();

        let center = map_camera(&cal, 0, 0.5, 0.5).unwrap();
        let left_edge = map_camera(&cal, 0, 0.1, 0.5).unwrap();

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

        let left_center = map_camera(&cal, 0, 0.5, 0.5).unwrap();
        let right_center = map_camera(&cal, 1, 0.5, 0.5).unwrap();

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
        // out of the document without building a second runtime object.
        // Verify the trait remains object-safe and Send+Sync.
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
    fn cylindrical_projection_has_no_detection_mapping() {
        let cal = cylinder_cal();
        assert!(map_camera(&cal, 0, 0.5, 0.5).is_none());
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
        // holds implementations with different camera counts. The
        // assertions pin those counts and distinct diagnostic names.
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
