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
mod geometry;

// Re-export coverage types so external code can still use
// `crate::projection::CoverageBoundary` etc.
pub use coverage::{ClampedPosition, CoverageBoundary, PanoramaExtent};

// Re-export geometry utility.
pub use geometry::point_in_polygon;

use crate::calibration::{Calibration, Lens};
use crate::geometry::CameraId;
use crate::geometry::ViewportPosition;
use crate::geometry::VirtualCamera;
use crate::geometry::view_matrix;
use crate::render::scene::SceneGeometry;
use crate::stitch::{BlendRule, SurfaceMap};

use nalgebra::{Matrix4, Perspective3, Point3, Vector3, Vector4};

// ---------------------------------------------------------------------------
// The Projection trait: L1 geometry dispatch.
// ---------------------------------------------------------------------------
//
// StitchCore holds a `Box<dyn Projection>` so alt-projections (the mono
// cylinder next, N-camera later) are drop-in additions without reshaping
// the core API. The trait dispatches the CPU surface maps and coverage
// construction today; the GPU program descriptor joins at the
// projection-shader step, and the detection-side forward maps
// (`camera_to_panorama`) fold in once their `CameraId`/`ViewportPosition`
// currency moves out of the detect layer.

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
    fn camera_count(&self) -> u8;

    /// The ordered surface list for one frame: each surface's inverse
    /// map paired with how it blends over the surfaces before it.
    /// Surface `i` samples source camera `i`; the first surface lays
    /// the base. The CPU composite drives these directly.
    fn surface_maps(
        &self,
        calibration: &Calibration,
        config: &crate::render::viewport::ViewportConfig,
        yaw: f32,
        pitch: f32,
    ) -> Vec<(Box<dyn SurfaceMap>, BlendRule)>;

    /// The GPU program this projection composites with. The render
    /// pipeline compiles and binds exactly what the descriptor says -
    /// the GPU dual of [`surface_maps`](Self::surface_maps), gated by
    /// the same CPU/GPU agreement oracle.
    #[cfg(feature = "gpu")]
    fn gpu_program(&self) -> crate::render::GpuProgram;

    /// Build the coverage boundary for this projection's panorama.
    ///
    /// Representation and clamp are one coupled unit: the default is the
    /// sampled-slice model (bounded, non-wrapping - today's L-shape). A
    /// projection that cannot reuse it overrides with its own at the
    /// topology step.
    fn coverage(&self, calibration: &Calibration, scene: &SceneGeometry) -> CoverageBoundary {
        CoverageBoundary::from_calibration(calibration, scene)
    }
}

/// Today's 2-plane L-shape stereo projection.
///
/// The plane placement is documented in
/// [`scene::SceneGeometry`](crate::render::scene::SceneGeometry); the
/// per-plane inverse maps come from the stitch geometry module. The
/// struct carries no state - the calibration document parameterizes it.
#[derive(Debug, Default, Clone, Copy)]
pub struct LShapeProjection;

impl Projection for LShapeProjection {
    fn name(&self) -> &'static str {
        "l-shape-stereo-2camera"
    }

    fn camera_count(&self) -> u8 {
        2
    }

    fn surface_maps(
        &self,
        calibration: &Calibration,
        config: &crate::render::viewport::ViewportConfig,
        yaw: f32,
        pitch: f32,
    ) -> Vec<(Box<dyn SurfaceMap>, BlendRule)> {
        let (left, right) =
            crate::stitch::geometry::l_shape_plane_maps(calibration, config, yaw, pitch);
        vec![
            (Box::new(left), BlendRule::Opaque),
            (
                Box::new(right),
                BlendRule::Smoothstep(calibration.topology.blend_width() as f64),
            ),
        ]
    }

    #[cfg(feature = "gpu")]
    fn gpu_program(&self) -> crate::render::GpuProgram {
        crate::render::GpuProgram {
            wgsl: include_str!("../shaders/fisheye.wgsl"),
            vs_entry: "vs_main",
            fs_entry: "fs_main",
            // Seam transition: the right plane's smoothstep alpha blends
            // over the opaque left base (matches BlendRule ordering).
            blend: wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            },
            vertex_layout: crate::render::renderer::Vertex::LAYOUT,
        }
    }
}

// ---------------------------------------------------------------------------
// Cylindrical single-input projection.
// ---------------------------------------------------------------------------

/// Single-input cylindrical projection.
///
/// Consumes one camera (`camera_count() == 1`) and renders it as if
/// painted on the inside of a cylinder; the virtual camera sits on the
/// cylinder axis. The parameters live in the calibration document's
/// [`CylinderTopology`](crate::calibration::CylinderTopology) - like
/// the L-shape, the struct itself carries no state. This is the
/// standard projection for pre-stitched 180-degree footage.
#[derive(Debug, Default, Clone, Copy)]
pub struct CylindricalProjection;

impl Projection for CylindricalProjection {
    fn name(&self) -> &'static str {
        "cylindrical-mono-1camera"
    }

    fn camera_count(&self) -> u8 {
        1
    }

    fn surface_maps(
        &self,
        calibration: &Calibration,
        config: &crate::render::viewport::ViewportConfig,
        yaw: f32,
        pitch: f32,
    ) -> Vec<(Box<dyn SurfaceMap>, BlendRule)> {
        let topology = calibration
            .topology
            .cylinder()
            .expect("the cylindrical projection requires the cylinder topology");
        vec![(
            Box::new(crate::stitch::cylinder::CylinderMap::new(
                topology,
                &calibration.framing,
                f64::from(calibration.lenses[0].height),
                config,
                yaw,
                pitch,
            )),
            // Single surface: nothing underneath to blend with.
            BlendRule::Opaque,
        )]
    }

    #[cfg(feature = "gpu")]
    fn gpu_program(&self) -> crate::render::GpuProgram {
        crate::render::GpuProgram {
            wgsl: include_str!("../shaders/cylindrical_mono.wgsl"),
            vs_entry: "vs_fullscreen",
            fs_entry: "fs_cylindrical_mono",
            // Mono: single surface, nothing to blend over.
            blend: wgpu::BlendState::REPLACE,
            // TODO: placeholder until the mono GPU pass is wired
            // (Step 13 PR B): its composite is a fullscreen pass with
            // its own bind layout.
            vertex_layout: crate::render::renderer::Vertex::LAYOUT,
        }
    }

    /// The cylinder's panorama is exactly rectangular in (yaw, pitch):
    /// yaw spans the angular sweep, pitch spans what the painted
    /// height subtends at the radius.
    fn coverage(&self, calibration: &Calibration, _scene: &SceneGeometry) -> CoverageBoundary {
        let t = calibration
            .topology
            .cylinder()
            .expect("the cylindrical projection requires the cylinder topology");
        let yaw_half = (t.sweep_deg.to_radians() * 0.5) as f32;
        let height = t
            .video_height
            .unwrap_or(f64::from(calibration.lenses[0].height));
        let pitch_half = (((height * 0.5) / t.focal_length).atan()) as f32;
        // The painted band is world-fixed; rig tilt/roll shape how
        // panning traverses it, not where it is - the clamp's rotated
        // viewport margining (the same mechanism a tilted L-shape
        // uses) accounts for the edge roll.
        CoverageBoundary::rectangular(
            -yaw_half,
            yaw_half,
            -pitch_half,
            pitch_half,
            calibration.framing.tilt as f32,
            calibration.framing.roll as f32,
        )
    }
}

/// The projection a calibration document calls for: the topology
/// variant picks it. Consumers with a custom projection can still
/// inject their own at executor construction.
pub fn for_topology(topology: &crate::calibration::Topology) -> Box<dyn Projection> {
    match topology {
        crate::calibration::Topology::LShape(_) => Box::new(LShapeProjection),
        crate::calibration::Topology::Cylinder(_) => Box::new(CylindricalProjection),
    }
}

/// Maximum Newton-Raphson iterations for KB4 inverse distortion.
const MAX_ITERATIONS: usize = 20;
/// Convergence threshold for Newton-Raphson.
const CONVERGENCE_EPS: f64 = 1e-10;
const CLIP_W_EPSILON: f32 = 1e-6;

/// Map a detection in camera pixel space to the yaw/pitch needed to
/// center the virtual camera on it.
///
/// `norm_x` and `norm_y` are in normalized `[0.0, 1.0]` image coordinates
/// (as returned by [`Detection`](crate::detect::detector::Detection)).
///
/// Returns `None` if the inverse distortion fails to converge (rare,
/// indicates an extreme point far outside the valid lens area).
///
/// # Example
///
/// ```rust
/// use reco_core::projection::camera_to_panorama;
/// use reco_core::geometry::CameraId;
/// use reco_core::calibration::Calibration;
/// use reco_core::render::scene::SceneGeometry;
///
/// # fn example(cal: &Calibration) {
/// let aspect = cal.lenses[0].width as f32 / cal.lenses[0].height as f32;
/// let scene = SceneGeometry::for_calibration(cal, aspect);
/// if let Some(pos) = camera_to_panorama(CameraId::Left, 0.5, 0.5, cal, &scene) {
///     println!("Center of left camera maps to yaw={:.3}, pitch={:.3}", pos.yaw, pos.pitch);
/// }
/// # }
/// ```
pub fn camera_to_panorama(
    camera: CameraId,
    norm_x: f32,
    norm_y: f32,
    calibration: &Calibration,
    scene: &SceneGeometry,
) -> Option<ViewportPosition> {
    camera_to_panorama_with_lens_correction(camera, norm_x, norm_y, calibration, scene, 1.0)
}

/// Map a distorted camera pixel into panorama space using the same lens
/// correction amount as the render shader.
///
/// `lens_correction_amount` follows the stitch renderer convention:
/// `1.0` is full KB4 correction, while `0.0` is the uncorrected/pinhole
/// plane view. Use [`camera_to_panorama`] for the normal full-correction
/// projection expected by detections and saved ROI data.
pub fn camera_to_panorama_with_lens_correction(
    camera: CameraId,
    norm_x: f32,
    norm_y: f32,
    calibration: &Calibration,
    scene: &SceneGeometry,
    lens_correction_amount: f32,
) -> Option<ViewportPosition> {
    let params = match camera {
        CameraId::Left => &calibration.lenses[0],
        CameraId::Right => &calibration.lenses[1],
    };

    // Step 1: Inverse fisheye - camera pixel [0,1] -> plane UV (extended space)
    let plane_uv = inverse_fisheye_with_correction(
        norm_x as f64,
        norm_y as f64,
        params,
        lens_correction_amount,
    )?;

    // Step 2: Plane UV -> 3D world point
    let world_point = plane_uv_to_world(plane_uv, camera, scene);

    // Step 3: World point -> yaw/pitch
    let dir = (world_point - Point3::from(Vector3::from(scene.camera_position))).normalize();
    Some(direction_to_yaw_pitch(&dir, &scene.camera_position))
}

/// Map a raw distorted camera pixel into the single-camera lens preview's
/// displayed coordinate space for a given correction amount.
///
/// The lens preview shader draws a quad with source UV `[0,1]`, remaps it to
/// extended plane UV `[-0.5,1.5]`, and then samples the raw frame through the
/// KB4 model. This helper performs the inverse mapping for overlay vertices.
pub fn camera_to_lens_preview(
    norm_x: f32,
    norm_y: f32,
    params: &Lens,
    lens_correction_amount: f32,
) -> Option<(f32, f32)> {
    if lens_correction_amount < 0.0 {
        return Some((norm_x, norm_y));
    }

    let (uv_x, uv_y) = inverse_fisheye_with_correction(
        norm_x as f64,
        norm_y as f64,
        params,
        lens_correction_amount,
    )?;
    Some((((uv_x + 0.5) * 0.5) as f32, ((uv_y + 0.5) * 0.5) as f32))
}

/// Map a panorama position (yaw/pitch) back to a camera pixel coordinate.
///
/// This is the inverse of [`camera_to_panorama`]. Given a position in the
/// panoramic view, returns the corresponding normalized pixel coordinate in
/// the specified camera's image, or `None` if the position is outside that
/// camera's field of view.
pub fn panorama_to_camera(
    yaw: f32,
    pitch: f32,
    camera: CameraId,
    calibration: &Calibration,
    scene: &SceneGeometry,
) -> Option<(f32, f32)> {
    let params = match camera {
        CameraId::Left => &calibration.lenses[0],
        CameraId::Right => &calibration.lenses[1],
    };

    let cam = VirtualCamera::new(&scene.camera_position);
    let dir = cam.yaw_pitch_to_direction(yaw, pitch);
    let cam_pos = Point3::from(cam.eye);

    let model = match camera {
        CameraId::Left => scene.model_matrix_left(),
        CameraId::Right => scene.model_matrix_right(),
    };
    let plane_origin = model.transform_point(&Point3::new(0.0, 0.0, 0.0));
    let plane_normal = model
        .transform_vector(&Vector3::new(0.0, 0.0, 1.0))
        .normalize();

    let denom = plane_normal.dot(&dir);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (plane_origin - cam_pos).dot(&plane_normal) / denom;
    if t <= 0.0 {
        return None;
    }
    let hit = cam_pos + dir * t;

    let (uv_x, uv_y) = world_to_plane_uv(hit, camera, scene)?;
    let tex_u = (uv_x + 0.5) * 0.5;
    let tex_v = (uv_y + 0.5) * 0.5;
    if !(0.0..=1.0).contains(&tex_u) || !(0.0..=1.0).contains(&tex_v) {
        return None;
    }

    let (norm_x, norm_y) = forward_fisheye(uv_x, uv_y, params);
    if (0.0..=1.0).contains(&norm_x) && (0.0..=1.0).contains(&norm_y) {
        Some((norm_x as f32, norm_y as f32))
    } else {
        None
    }
}

/// Project a panorama yaw/pitch position into the current rendered viewport.
///
/// Returns normalized screen coordinates in `[0, 1]` where `(0, 0)` is the
/// top-left of the rendered viewport. Coordinates may be outside that range
/// when the point is off-screen; `None` means the point is behind the virtual
/// camera or the projection produced a non-finite result.
#[allow(clippy::too_many_arguments)]
pub fn panorama_to_viewport(
    yaw: f32,
    pitch: f32,
    view_yaw: f32,
    view_pitch: f32,
    fov_degrees: f32,
    aspect: f32,
    rig_tilt: f32,
    rig_roll: f32,
    scene: &SceneGeometry,
) -> Option<(f32, f32)> {
    if !yaw.is_finite()
        || !pitch.is_finite()
        || !view_yaw.is_finite()
        || !view_pitch.is_finite()
        || !fov_degrees.is_finite()
        || !aspect.is_finite()
        || !rig_tilt.is_finite()
        || !rig_roll.is_finite()
        || aspect <= 0.0
        || !(1.0..179.0).contains(&fov_degrees)
    {
        return None;
    }

    let cam = VirtualCamera::new(&scene.camera_position);
    let dir = cam.yaw_pitch_to_direction(yaw, pitch);
    let world_point = Point3::from(cam.eye + dir);

    let projection =
        Perspective3::new(aspect, fov_degrees.to_radians(), 0.01, 5.0).to_homogeneous();
    let view = view_matrix(
        &scene.camera_position,
        view_yaw,
        view_pitch,
        rig_tilt,
        rig_roll,
    );

    let clip = projection * view * world_point.to_homogeneous();
    clip_to_screen(clip)
}

/// Project a panorama-space segment into the current rendered viewport.
///
/// Unlike sampling intermediate yaw/pitch points, this projects the two
/// endpoints through the same perspective camera used by the renderer and
/// clips the resulting segment in homogeneous clip space. That keeps field
/// lines visually straight and stable as the GUI viewport pans.
#[allow(clippy::too_many_arguments)]
pub fn panorama_segment_to_viewport(
    a_yaw: f32,
    a_pitch: f32,
    b_yaw: f32,
    b_pitch: f32,
    view_yaw: f32,
    view_pitch: f32,
    fov_degrees: f32,
    aspect: f32,
    rig_tilt: f32,
    rig_roll: f32,
    scene: &SceneGeometry,
) -> Option<((f32, f32), (f32, f32))> {
    if !a_yaw.is_finite()
        || !a_pitch.is_finite()
        || !b_yaw.is_finite()
        || !b_pitch.is_finite()
        || !view_yaw.is_finite()
        || !view_pitch.is_finite()
        || !fov_degrees.is_finite()
        || !aspect.is_finite()
        || !rig_tilt.is_finite()
        || !rig_roll.is_finite()
        || aspect <= 0.0
        || !(1.0..179.0).contains(&fov_degrees)
    {
        return None;
    }

    let camera = VirtualCamera::new(&scene.camera_position);
    let projection =
        Perspective3::new(aspect, fov_degrees.to_radians(), 0.01, 5.0).to_homogeneous();
    let view = view_matrix(
        &scene.camera_position,
        view_yaw,
        view_pitch,
        rig_tilt,
        rig_roll,
    );
    let projection_view = projection * view;

    let a = panorama_point_to_clip(a_yaw, a_pitch, &camera, &projection_view)?;
    let b = panorama_point_to_clip(b_yaw, b_pitch, &camera, &projection_view)?;
    clip_homogeneous_segment_to_frustum(a, b)
        .and_then(|(a, b)| Some((clip_to_screen(a)?, clip_to_screen(b)?)))
}

/// Convert a normalized rendered-viewport coordinate into stitched panorama
/// yaw/pitch coordinates.
///
/// This is the inverse of [`panorama_to_viewport`] for points on the
/// virtual camera's view ray. `screen_x` and `screen_y` are normalized
/// `[0, 1]` coordinates where `(0, 0)` is the top-left of the rendered
/// viewport.
#[allow(clippy::too_many_arguments)]
pub fn viewport_to_panorama(
    screen_x: f32,
    screen_y: f32,
    view_yaw: f32,
    view_pitch: f32,
    fov_degrees: f32,
    aspect: f32,
    rig_tilt: f32,
    rig_roll: f32,
    scene: &SceneGeometry,
) -> Option<ViewportPosition> {
    if !screen_x.is_finite()
        || !screen_y.is_finite()
        || !view_yaw.is_finite()
        || !view_pitch.is_finite()
        || !fov_degrees.is_finite()
        || !aspect.is_finite()
        || !rig_tilt.is_finite()
        || !rig_roll.is_finite()
        || aspect <= 0.0
        || !(1.0..179.0).contains(&fov_degrees)
    {
        return None;
    }

    let ndc_x = screen_x * 2.0 - 1.0;
    let ndc_y = 1.0 - screen_y * 2.0;
    let half_v = (fov_degrees * 0.5).to_radians().tan();
    let camera_dir = Vector3::new(ndc_x * aspect * half_v, ndc_y * half_v, -1.0).normalize();

    let view = view_matrix(
        &scene.camera_position,
        view_yaw,
        view_pitch,
        rig_tilt,
        rig_roll,
    );
    let inv_view = view.try_inverse()?;
    let dir = inv_view * Vector4::new(camera_dir.x, camera_dir.y, camera_dir.z, 0.0);
    let world_dir = Vector3::new(dir.x, dir.y, dir.z).try_normalize(1e-6)?;
    let pos = direction_to_yaw_pitch(&world_dir, &scene.camera_position);
    Some(ViewportPosition {
        yaw: pos.yaw,
        pitch: pos.pitch,
        fov_degrees: Some(fov_degrees),
    })
}

/// Compute valid yaw/pitch bounds for a given FOV where no black edges appear.
///
/// Samples the visible edges of both camera planes and returns the tightest
/// bounds that keep the viewport fully within the projected image area.
pub fn viewport_bounds(
    fov_degrees: f32,
    calibration: &Calibration,
    scene: &SceneGeometry,
    aspect: f32,
) -> ViewportBounds {
    let half_vfov = (fov_degrees * 0.5).to_radians();
    let half_hfov = (half_vfov.tan() * aspect).atan();
    let corner_hfov = (half_hfov.tan() / half_vfov.cos()).atan();
    let corner_vfov = (half_vfov.tan() / half_hfov.cos()).atan();

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
#[derive(Debug, Clone, Copy)]
pub struct ViewportBounds {
    /// Minimum yaw in radians.
    pub min_yaw: f32,
    /// Maximum yaw in radians.
    pub max_yaw: f32,
    /// Minimum pitch in radians.
    pub min_pitch: f32,
    /// Maximum pitch in radians.
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

/// Forward KB4 fisheye: undistorted plane UV -> distorted camera pixel [0,1].
///
/// Mirror of [`inverse_fisheye_with_correction`] in the same normalized-intrinsic
/// convention and the same extended-UV plane space (the shader's
/// `uv * 2.0 - 0.5` remap output). The polynomial delegates to
/// `reco_core::lens::kb4`, same canonical source as the
/// Newton-Raphson step in [`inverse_fisheye_with_correction`].
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
#[cfg(test)]
fn inverse_fisheye(dist_x: f64, dist_y: f64, params: &Lens) -> Option<(f64, f64)> {
    inverse_fisheye_with_correction(dist_x, dist_y, params, 1.0)
}

fn inverse_fisheye_with_correction(
    dist_x: f64,
    dist_y: f64,
    params: &Lens,
    lens_correction_amount: f32,
) -> Option<(f64, f64)> {
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
    let correction = (lens_correction_amount as f64).clamp(0.0, 1.0);

    if theta_d < 1e-12 {
        // At the optical center - no distortion
        return Some((cx, cy));
    }

    // Newton-Raphson: solve f(theta) = mix(theta, theta_d_poly(theta), correction)
    // - theta_d = 0, where theta_d_poly lives in `reco_core::lens::kb4`
    // (SYNC_WITH WGSL).
    let mut theta = theta_d; // initial guess
    for _ in 0..MAX_ITERATIONS {
        let theta_full = crate::lens::kb4::theta_d(theta, &k);
        let f = (theta * (1.0 - correction) + theta_full * correction) - theta_d;
        let f_prime = (1.0 - correction) + crate::lens::kb4::theta_d_prime(theta, &k) * correction;

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
fn plane_uv_to_world(uv: (f64, f64), camera: CameraId, scene: &SceneGeometry) -> Point3<f32> {
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

/// Exact inverse of [`plane_uv_to_world`].
///
/// Given a world-space point that lies on the named camera's plane, returns
/// its extended-UV coordinate (shader space `[-0.5, 1.5]`).
fn world_to_plane_uv(
    world: nalgebra::Point3<f32>,
    camera: CameraId,
    scene: &SceneGeometry,
) -> Option<(f64, f64)> {
    let model = match camera {
        CameraId::Left => scene.model_matrix_left(),
        CameraId::Right => scene.model_matrix_right(),
    };
    let inv_model = model.try_inverse()?;
    let local = inv_model.transform_point(&world);

    let tex_u = local.x / scene.plane_width + 0.5;
    let tex_v = 0.5 - local.y * scene.plane_aspect / scene.plane_width;

    let uv_x = (tex_u * 2.0 - 0.5) as f64;
    let uv_y = (tex_v * 2.0 - 0.5) as f64;
    Some((uv_x, uv_y))
}

fn panorama_point_to_clip(
    yaw: f32,
    pitch: f32,
    camera: &VirtualCamera,
    projection_view: &Matrix4<f32>,
) -> Option<Vector4<f32>> {
    let dir = camera.yaw_pitch_to_direction(yaw, pitch);
    let world_point = Point3::from(camera.eye + dir);
    let clip = projection_view * world_point.to_homogeneous();
    clip.iter().all(|v| v.is_finite()).then_some(clip)
}

fn clip_homogeneous_segment_to_frustum(
    mut a: Vector4<f32>,
    mut b: Vector4<f32>,
) -> Option<(Vector4<f32>, Vector4<f32>)> {
    for plane in [
        ClipPlane::Front,
        ClipPlane::Left,
        ClipPlane::Right,
        ClipPlane::Bottom,
        ClipPlane::Top,
        ClipPlane::Near,
        ClipPlane::Far,
    ] {
        (a, b) = clip_segment_to_plane(a, b, plane)?;
    }

    Some((a, b))
}

#[derive(Clone, Copy)]
enum ClipPlane {
    Front,
    Left,
    Right,
    Bottom,
    Top,
    Near,
    Far,
}

impl ClipPlane {
    fn distance(self, point: Vector4<f32>) -> f32 {
        match self {
            Self::Front => point.w - CLIP_W_EPSILON,
            Self::Left => point.x + point.w,
            Self::Right => point.w - point.x,
            Self::Bottom => point.y + point.w,
            Self::Top => point.w - point.y,
            Self::Near => point.z + point.w,
            Self::Far => point.w - point.z,
        }
    }
}

fn clip_segment_to_plane(
    mut a: Vector4<f32>,
    mut b: Vector4<f32>,
    plane: ClipPlane,
) -> Option<(Vector4<f32>, Vector4<f32>)> {
    let da = plane.distance(a);
    let db = plane.distance(b);
    let a_inside = da >= 0.0;
    let b_inside = db >= 0.0;

    match (a_inside, b_inside) {
        (true, true) => Some((a, b)),
        (false, false) => None,
        (false, true) => {
            a = interpolate_clip_at_plane(a, b, da, db)?;
            Some((a, b))
        }
        (true, false) => {
            b = interpolate_clip_at_plane(a, b, da, db)?;
            Some((a, b))
        }
    }
}

fn interpolate_clip_at_plane(
    a: Vector4<f32>,
    b: Vector4<f32>,
    da: f32,
    db: f32,
) -> Option<Vector4<f32>> {
    let denom = da - db;
    if denom.abs() < f32::EPSILON {
        return None;
    }
    let t = (da / denom).clamp(0.0, 1.0);
    Some(a + (b - a) * t)
}

fn clip_to_screen(clip: Vector4<f32>) -> Option<(f32, f32)> {
    if clip.w <= CLIP_W_EPSILON || !clip.w.is_finite() {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    (ndc_x.is_finite() && ndc_y.is_finite()).then_some(((ndc_x + 1.0) * 0.5, (1.0 - ndc_y) * 0.5))
}

/// Decompose a direction vector into yaw/pitch relative to the virtual camera.
///
/// Thin wrapper over [`VirtualCamera::direction_to_yaw_pitch`] kept
/// for the existing call sites until they carry a `VirtualCamera`
/// directly. Panners, directors, and the render loop all share the
/// same basis through this path.
pub(crate) fn direction_to_yaw_pitch(
    dir: &Vector3<f32>,
    camera_position: &[f32; 3],
) -> ViewportPosition {
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
    use crate::calibration::{Calibration, Framing, LShapeTopology, Lens};

    fn test_scene(cal: &Calibration) -> SceneGeometry {
        let aspect = cal.lenses[0].width as f32 / cal.lenses[0].height as f32;
        SceneGeometry::for_calibration(cal, aspect)
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
            LShapeTopology {
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
        let scene = test_scene(&cal);

        // Optical center of the left camera (cx/w, cy/h)
        let cx = cal.lenses[0].cx as f32 / cal.lenses[0].width as f32;
        let cy = cal.lenses[0].cy as f32 / cal.lenses[0].height as f32;

        let pos = camera_to_panorama(CameraId::Left, cx, cy, &cal, &scene);
        assert!(pos.is_some(), "optical center should map successfully");
        let pos = pos.unwrap();
        // The optical center should produce a valid yaw/pitch (no NaN)
        assert!(pos.yaw.is_finite(), "yaw should be finite");
        assert!(pos.pitch.is_finite(), "pitch should be finite");
    }

    #[test]
    fn left_camera_left_edge_yaw_differs_from_center() {
        let cal = test_calibration();
        let scene = test_scene(&cal);

        let center = camera_to_panorama(CameraId::Left, 0.5, 0.5, &cal, &scene).unwrap();
        let left_edge = camera_to_panorama(CameraId::Left, 0.1, 0.5, &cal, &scene).unwrap();

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
        let scene = test_scene(&cal);

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
    fn yaw_pitch_to_direction_roundtrips_with_direction_to_yaw_pitch() {
        // Step 1a: the two helpers must form an exact bijection on the
        // (yaw, pitch) grid used by panners and directors. All
        // shipping scenes set `camera_position = [d, 0, d]` (see
        // SceneGeometry::new), so eye.y = 0 is
        // the real invariant; test positions honor that. Pitch stays
        // clear of +-pi/2 where yaw is undefined.
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
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);

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
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);

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
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);
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
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);
        let out = coverage.safe_clamp(0.0, f32::NAN, 75.0, 16.0 / 9.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
    }

    #[test]
    fn safe_clamp_rejects_nan_fov() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);
        let out = coverage.safe_clamp(0.0, 0.0, f32::NAN, 16.0 / 9.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
    }

    #[test]
    fn safe_clamp_rejects_infinite_inputs() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);
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
        let p = LShapeProjection;
        assert_eq!(p.name(), "l-shape-stereo-2camera");
        assert_eq!(p.camera_count(), 2);
    }

    #[test]
    fn projection_is_dyn_compatible() {
        // Core invariant: StitchCore will hold `Box<dyn Projection>`.
        // Verify the trait bounds allow that today and that Send+Sync
        // both hold.
        let projections: Vec<Box<dyn Projection>> = vec![Box::new(LShapeProjection)];
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<dyn Projection>();
        assert_eq!(projections[0].camera_count(), 2);
    }

    #[test]
    fn l_shape_surface_maps_dispatch_two_ordered_surfaces() {
        // The L-shape emits exactly two surfaces: the opaque left base,
        // then the right fading in with the calibration's seam width.
        let cal = test_calibration();
        let config = crate::render::viewport::ViewportConfig::default();
        let surfaces = LShapeProjection.surface_maps(&cal, &config, 0.0, 0.0);
        assert_eq!(surfaces.len(), 2);
        assert_eq!(surfaces[0].1, crate::stitch::BlendRule::Opaque);
        assert_eq!(
            surfaces[1].1,
            crate::stitch::BlendRule::Smoothstep(cal.topology.blend_width() as f64)
        );
    }

    // ---- CylindricalProjection ---------------------------------------------

    fn cylinder_cal() -> Calibration {
        Calibration::new(
            vec![Lens::flat(3840, 1080)],
            crate::calibration::CylinderTopology::default(),
            Framing {
                axis_offset: 0.0,
                tilt: 0.0,
                roll: 0.0,
            },
        )
    }

    #[test]
    fn cylindrical_projection_reports_mono() {
        let p = CylindricalProjection;
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
        let config = crate::render::viewport::ViewportConfig::default();
        let surfaces = CylindricalProjection.surface_maps(&cal, &config, 0.0, 0.0);
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
        let scene = SceneGeometry::for_calibration(&cal, 3840.0 / 1080.0);
        let coverage = CylindricalProjection.coverage(&cal, &scene);
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
    fn for_topology_resolves_the_matching_projection() {
        let l = for_topology(&test_calibration().topology);
        assert_eq!(l.camera_count(), 2);
        let c = for_topology(&cylinder_cal().topology);
        assert_eq!(c.camera_count(), 1);
    }

    #[test]
    fn projection_dyn_dispatch_round_trip_with_mixed_camera_counts() {
        // Compile-time: the trait is object-safe, so one collection
        // holds impls with different `camera_count()` results - what
        // lets `for_topology` pick the projection at calibration-load
        // time. The assertions pin the per-projection counts and that
        // the diagnostic names stay distinct.
        let projections: Vec<Box<dyn Projection>> =
            vec![Box::new(LShapeProjection), Box::new(CylindricalProjection)];
        assert_eq!(projections[0].camera_count(), 2);
        assert_eq!(projections[1].camera_count(), 1);
        assert_ne!(projections[0].name(), projections[1].name());
    }

    #[test]
    fn cylindrical_projection_is_send_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<CylindricalProjection>();
    }
}
