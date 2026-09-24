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

pub(crate) mod coverage;
pub(crate) mod cylinder;
mod geometry;
pub(crate) mod interval;
pub(crate) mod l_shape;

pub use cylinder::Cylinder;
pub use l_shape::{DEFAULT_BLEND_WIDTH, LShape};

// Re-export geometry utility.
pub use geometry::point_in_polygon;

use crate::calibration::{Calibration, Framing, Lens};
use crate::geometry::VirtualCamera;
use crate::geometry::{CameraIndex, Pose, YawPitch};
use crate::stitch::{BlendRule, SurfaceMap};
use thiserror::Error;

/// Why a calibrated source-image point could not be mapped to an angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SourceMapError {
    /// The camera index is not present in this calibration.
    #[error("camera index is outside the calibration")]
    InvalidCamera,
    /// The point is outside the normalized image or contains a non-finite value.
    #[error("source point must be finite and within [0, 1]")]
    InvalidPoint,
    /// This projection does not implement source-point mapping yet.
    #[error("source-point mapping is unsupported by this projection")]
    Unsupported,
    /// The inverse lens/geometry calculation has no finite solution.
    #[error("source point has no finite mapped direction")]
    NoDirection,
}

// ---------------------------------------------------------------------------
// The Projection trait: L1 geometry dispatch.
// ---------------------------------------------------------------------------
//
// `Topology::projection()` borrows the document's shape parameters as `&dyn Projection`, without copying them into a separate object.

/// One frame's shape-geometry question, bundled: optical/framing inputs, the output
/// viewport, and the pose. The pose carries fov alongside yaw/pitch -
/// [`ViewportSize`](crate::render::viewport::ViewportSize) is
/// zoom-free, so there is exactly one fov per frame and a second,
/// disagreeing copy is unrepresentable.
pub(crate) struct ProjectionContext<'a> {
    /// Source optical models in camera order.
    pub lenses: &'a [Lens],
    /// Shared rig framing.
    pub framing: &'a Framing,
    /// Output viewport geometry (dimensions).
    pub viewport_size: &'a crate::render::viewport::ViewportSize,
    /// The render pose: yaw/pitch pan plus fov zoom, one bundle.
    pub pose: Pose,
}

impl Calibration {
    /// Map a normalized source-image point to world-reference yaw/pitch.
    /// A mapped direction alone does not guarantee rendered source support.
    /// The calibration must be validated before use, as for engine construction.
    pub fn map_source_point(
        &self,
        camera: CameraIndex,
        point: [f32; 2],
    ) -> Result<YawPitch, SourceMapError> {
        if camera >= self.lenses.len() || camera >= self.topology.camera_count() {
            return Err(SourceMapError::InvalidCamera);
        }
        if point
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(SourceMapError::InvalidPoint);
        }
        self.topology
            .projection()
            .map_source_point(camera, point, &self.lenses, &self.framing)
    }

    /// The virtual-camera basis derived from this complete calibration.
    pub fn virtual_camera(&self) -> VirtualCamera {
        self.topology.projection().virtual_camera(&self.framing)
    }

    /// Build ordered output-to-source maps for a render-space pose and viewport, without a GPU.
    ///
    /// Entry `i` samples source camera `i`; composite in order using [`BlendRule::alpha`] and the returned sample's edge coordinate. Maps capture this calibration, viewport and pose: rebuild them when any mapping input changes. The caller must supply a validated calibration and viewport and a finite pose with FOV in `[1, 179]`. This query does not apply coverage constraints.
    pub fn surface_maps(
        &self,
        viewport_size: &crate::render::viewport::ViewportSize,
        pose: Pose,
    ) -> Vec<(Box<dyn SurfaceMap>, BlendRule)> {
        self.topology.projection().surface_maps(&ProjectionContext {
            lenses: &self.lenses,
            framing: &self.framing,
            viewport_size,
            pose,
        })
    }
}

/// A panoramic projection geometry.
///
/// Implemented by the shape parameters stored in the calibration document. Provides surface maps, virtual-camera geometry, and source-point mapping. GPU-enabled builds also expose the rendering program descriptor.
/// Call the corresponding [`Calibration`] methods for complete geometry queries; direct trait calls require matching validated lenses and framing.
///
/// # Bounds
///
/// `Send + Sync` allows implementations to be shared across processing threads.
pub(crate) trait Projection: Send + Sync {
    /// Short human-readable name for logs + diagnostic bundles.
    fn name(&self) -> &'static str;

    /// Number of input cameras this projection consumes. 1 for mono,
    /// 2 for today's L-shape stereo, N>2 for future panoramic rigs.
    fn camera_count(&self) -> usize;

    /// The virtual camera basis - the eye position plus the yaw/pitch
    /// decomposition axes every view matrix and pose conversion is
    /// built from. The one piece of scene state every projection has,
    /// plane-backed or not. The cylinder's camera sits at the origin looking down `-Z`.
    fn virtual_camera(&self, framing: &Framing) -> VirtualCamera;

    /// Map one normalized source-image point to world-reference angles.
    /// This does not prove that the renderer samples that point at the angle.
    fn map_source_point(
        &self,
        camera: CameraIndex,
        point: [f32; 2],
        lenses: &[Lens],
        framing: &Framing,
    ) -> Result<YawPitch, SourceMapError>;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{Calibration, Framing, Lens};
    use crate::projection::LShape;
    use crate::render::viewport::ViewportSize;

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
        // Full-correction baseline with non-default rig and plane orientation.
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

    #[test]
    fn mapped_source_points_round_trip_through_render_surface() {
        let viewport_size = ViewportSize {
            width: 1,
            height: 1,
        };
        for mut cal in [test_calibration(), relocation_calibration()] {
            for correction in [0.0, 0.4, 1.0] {
                for lens in &mut cal.lenses {
                    lens.correction = correction;
                }
                let projection = cal.topology.projection();
                let virtual_camera = projection.virtual_camera(&cal.framing);
                for camera in 0..2 {
                    for ix in 0..=20 {
                        for iy in 0..=20 {
                            let x = 0.02 + 0.96 * ix as f32 / 20.0;
                            let y = 0.02 + 0.96 * iy as f32 / 20.0;
                            let mut angles = cal.map_source_point(camera, [x, y]).unwrap();
                            // Point mapping returns world angles; rendering consumes rig-relative angles.
                            (angles.yaw, angles.pitch) = crate::geometry::world_to_render_pose(
                                &virtual_camera,
                                angles.yaw,
                                angles.pitch,
                                cal.framing.tilt as f32,
                                cal.framing.roll as f32,
                            );
                            let pose = Pose {
                                yaw: angles.yaw,
                                pitch: angles.pitch,
                                ..Pose::default()
                            };
                            let surfaces = cal.surface_maps(&viewport_size, pose);
                            let uv = surfaces[camera].0.sample_uv([0.0, 0.0]).unwrap_or_else(|| {
                                panic!("missing surface: camera={camera}, correction={correction}, source=({x}, {y})")
                            });
                            let lens = &cal.lenses[camera];
                            let error = ((uv.u - x as f64) * lens.width as f64)
                                .hypot((uv.v - y as f64) * lens.height as f64);
                            assert!(
                                error < 0.01,
                                "camera={camera}, correction={correction}, source=({x}, {y}), error={error}px"
                            );
                        }
                    }
                }
            }
        }
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
            let pose = cal.map_source_point(camera, [x, y]).unwrap();
            assert_geometry_close(pose.yaw, expected_yaw, "mapped yaw");
            assert_geometry_close(pose.pitch, expected_pitch, "mapped pitch");
        }

        assert_eq!(
            cal.map_source_point(2, [0.5, 0.5]),
            Err(SourceMapError::InvalidCamera)
        );
    }

    #[test]
    fn optical_center_maps_to_known_position() {
        let cal = test_calibration();

        // Optical center of the left camera (cx/w, cy/h)
        let cx = cal.lenses[0].cx as f32 / cal.lenses[0].width as f32;
        let cy = cal.lenses[0].cy as f32 / cal.lenses[0].height as f32;

        let pos = cal.map_source_point(0, [cx, cy]);
        assert!(pos.is_ok(), "optical center should map successfully");
        let pos = pos.unwrap();
        // The optical center should produce a valid yaw/pitch (no NaN)
        assert!(pos.yaw.is_finite(), "yaw should be finite");
        assert!(pos.pitch.is_finite(), "pitch should be finite");
    }

    #[test]
    fn left_camera_left_edge_yaw_differs_from_center() {
        let cal = test_calibration();

        let center = cal.map_source_point(0, [0.5, 0.5]).unwrap();
        let left_edge = cal.map_source_point(0, [0.1, 0.5]).unwrap();

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

        let left_center = cal.map_source_point(0, [0.5, 0.5]).unwrap();
        let right_center = cal.map_source_point(1, [0.5, 0.5]).unwrap();

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
        // Reference forward is horizontal; position does not determine the angular basis. Pitch stays clear of the poles where yaw is undefined.
        let cameras = [
            ([0.24, 0.0, 0.24], [-0.24, 0.0, -0.24]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            ([0.3, 0.0, 0.2], [-0.3, 0.0, -0.2]),
            ([0.1, 0.0, 0.5], [-0.1, 0.0, -0.5]),
            ([2.0, 3.0, 4.0], [1.0, 0.0, 0.0]),
        ];

        let yaw_steps = [-1.2_f32, -0.6, -0.2, 0.0, 0.2, 0.6, 1.2];
        let pitch_steps = [-0.9_f32, -0.4, -0.1, 0.0, 0.1, 0.4, 0.9];

        for (position, forward) in &cameras {
            let camera = VirtualCamera::new(position, forward);
            for &yaw in &yaw_steps {
                for &pitch in &pitch_steps {
                    let dir = camera.yaw_pitch_to_direction(yaw, pitch);
                    let norm = dir.norm();
                    assert!(
                        (norm - 1.0).abs() < 1e-5,
                        "direction must be unit, got |dir| = {norm} for camera={camera:?} yaw={yaw} pitch={pitch}"
                    );

                    let pos = camera.direction_to_yaw_pitch(&dir);
                    assert!(
                        (pos.yaw - yaw).abs() < 1e-4,
                        "yaw mismatch for camera={camera:?}: sent {yaw}, got {} (dir={dir:?})",
                        pos.yaw
                    );
                    assert!(
                        (pos.pitch - pitch).abs() < 1e-4,
                        "pitch mismatch for camera={camera:?}: sent {pitch}, got {} (dir={dir:?})",
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
        let surfaces = cal.surface_maps(&config, Pose::default());
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
        assert_eq!(
            cal.map_source_point(0, [0.5, 0.5]),
            Err(SourceMapError::Unsupported)
        );
    }

    #[test]
    fn calibrated_mapping_rejects_invalid_source_inputs() {
        let cal = test_calibration();
        assert_eq!(
            cal.map_source_point(2, [0.5, 0.5]),
            Err(SourceMapError::InvalidCamera)
        );
        for point in [[f32::NAN, 0.5], [-0.01, 0.5], [0.5, 1.01]] {
            assert_eq!(
                cal.map_source_point(0, point),
                Err(SourceMapError::InvalidPoint)
            );
        }
    }

    #[test]
    fn cylindrical_surface_maps_emit_one_opaque_surface() {
        let cal = cylinder_cal();
        let config = crate::render::viewport::ViewportSize::default();
        let surfaces = cal.surface_maps(&config, Pose::default());
        assert_eq!(surfaces.len(), 1);
        assert_eq!(surfaces[0].1, crate::stitch::BlendRule::Opaque);
        assert!(
            surfaces[0]
                .0
                .sample_uv(config.pixel_ndc(config.width / 2, config.height / 2))
                .is_some(),
            "the straight-ahead pixel is covered"
        );
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
