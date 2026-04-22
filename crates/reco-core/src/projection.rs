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
use crate::scene::SceneGeometry;

use nalgebra::{Point3, Vector3};

// ---------------------------------------------------------------------------
// M3 foundation: Projection trait + LShapeProjection marker.
// ---------------------------------------------------------------------------
//
// Plan-execution §2.5 + §7 decision 8: the future StitchCore takes a
// `Box<dyn Projection>` instead of hardcoding the 2-plane L-shape
// geometry. This makes alt-projections (cylindrical / flat-mixing-
// shader / mono-single-plane / equirect / N-camera panoramic) drop-in
// additions later without reshaping the core session API.
//
// This commit lands the trait + a marker implementation for today's
// L-shape geometry. It does NOT move the existing `camera_to_panorama`
// etc. free functions into the trait - that migration happens when
// StitchCore is being written and the real method set emerges from
// usage. Landing the shape first lets parallel design work on a
// second projection (§7 decision 8, user-chosen form) start without
// re-plumbing reco-core.

/// A panoramic projection geometry.
///
/// Implemented by concrete projections (today's 2-plane L-shape,
/// future cylindrical / flat-mix / mono / N-camera). Dispatched
/// dynamically by StitchCore so swapping projections at session
/// construction time does not require recompilation.
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

    /// WGSL fragment shader source for the composite pass that
    /// transforms per-camera undistorted textures into the final
    /// panorama output.
    ///
    /// Returned as a string so wgpu can compile it at pipeline
    /// creation. Today's L-shape geometry returns an empty string:
    /// its shader is still embedded in `stitch_renderer.rs`. The
    /// migration happens when StitchCore takes over rendering and
    /// dispatches composite via this trait.
    fn wgsl_composite_source(&self) -> &str {
        ""
    }
}

/// Marker type for today's 2-plane L-shape stereo projection.
///
/// The geometry is documented in [`scene::SceneGeometry`](crate::scene::SceneGeometry).
/// All the real math still lives in the free functions below and in
/// `stitch_renderer.rs`; this struct carries no state today. It is
/// here to make StitchCore's `Box<dyn Projection>` slot have a
/// concrete default that matches shipping behavior.
#[derive(Debug, Default, Clone, Copy)]
pub struct LShapeProjection;

impl Projection for LShapeProjection {
    fn name(&self) -> &'static str {
        "l-shape-stereo-2camera"
    }

    fn camera_count(&self) -> u8 {
        2
    }
}

// ---------------------------------------------------------------------------
// Cylindrical single-input projection (plan step 9, second Projection impl).
// ---------------------------------------------------------------------------
//
// Models a single video as a texture painted on the inside of a
// cylinder of radius `focal_length`. The virtual camera sits on the
// cylinder axis and looks outward; pan/tilt/zoom rotate the camera and
// scale FOV. Matches the `gilbertchen/actionstitch-player` projection
// (MIT-licensed 180-degree cylindrical video player) enough that
// calibration files from that ecosystem could be consumed with small
// adapter code.
//
// Design goal of landing this here (not just as a shader file):
// proves the plan's claim that the `Projection` trait supports
// camera_count() != 2 so future mono / N-camera / alt-projection
// impls can plug in without reshaping StitchCore's API. Ships with:
//
//   - A configurable `CylindricalProjection` with defaults that
//     mirror actionstitch's (focal_length=2400, sweep=PI = 180deg,
//     screen_rotation=0, video_height sourced from the input).
//   - A WGSL shader at `shaders/cylindrical_mono.wgsl` returned
//     verbatim from `wgsl_composite_source()`.
//   - camera_count() = 1.
//
// Deliberately NOT wired into `StitchCore` / `StitchPipeline` in this
// commit - that migration is a follow-up that needs a mono submit
// path (`submit_frame_yuv_mono`) and a different bind group layout.
// The trait-side contract is the deliverable here.

/// Configuration for a [`CylindricalProjection`]. Defaults match the
/// `actionstitch-player` projection (180-degree sweep, 2400px focal
/// length, no screen rotation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CylindricalProjectionConfig {
    /// Cylinder radius in world units. Larger values = narrower
    /// cylindrical wrap per pixel, so the panorama feels flatter.
    /// `actionstitch` defaults to `2400` and exposes a slider from
    /// 1000 to 5000.
    pub focal_length: f32,
    /// Full horizontal angular sweep in radians. `std::f32::consts::PI`
    /// (180 degrees) is the canonical action-camera case; 2π would be
    /// a full 360-degree cylinder.
    pub angular_sweep_rad: f32,
    /// Screen tilt around the view axis in radians. `actionstitch`
    /// exposes this as a ±30-degree slider labelled "Screen tilt" and
    /// uses it to correct for a rig that is not level side-to-side.
    pub screen_rotation_rad: f32,
    /// Video height in world units. Defaults to `1.0` (normalized);
    /// consumers with a known camera height can pass the actual value
    /// so the cylinder has the right aspect.
    pub video_height: f32,
}

impl Default for CylindricalProjectionConfig {
    fn default() -> Self {
        Self {
            focal_length: 2400.0,
            angular_sweep_rad: std::f32::consts::PI,
            screen_rotation_rad: 0.0,
            video_height: 1.0,
        }
    }
}

/// Single-input cylindrical projection.
///
/// Consumes one camera (`camera_count() == 1`) and renders it as if
/// painted on the inside of a cylinder of radius `config.focal_length`.
/// The virtual camera sits on the cylinder axis.
///
/// Attribution: the projection geometry (focal-length, angular-sweep,
/// and screen-rotation tilt) is the one used by
/// `gilbertchen/actionstitch-player` (MIT-licensed 180-degree
/// cylindrical video player). The WGSL shader here is a from-scratch
/// reimplementation of that model for wgpu; no code is copied.
#[derive(Debug, Clone, Copy, Default)]
pub struct CylindricalProjection {
    /// Projection parameters. `Default` uses `actionstitch`-matching
    /// values (see [`CylindricalProjectionConfig::default`]).
    pub config: CylindricalProjectionConfig,
}

impl CylindricalProjection {
    /// Build a new cylindrical projection with the given config.
    pub fn new(config: CylindricalProjectionConfig) -> Self {
        Self { config }
    }

    /// Compute the cylinder's `theta_start` angle in radians:
    /// `PI/2 - angular_sweep/2`. This is where the video's left edge
    /// lands on the cylinder surface and matches actionstitch's
    /// `THREE.CylinderGeometry(..., Math.PI / 2 - s / 2, s)`.
    pub fn theta_start_rad(&self) -> f32 {
        std::f32::consts::FRAC_PI_2 - self.config.angular_sweep_rad * 0.5
    }
}

/// WGSL source for the cylindrical-mono composite pass. Embedded at
/// compile time so `wgsl_composite_source()` can return `&'static str`.
const CYLINDRICAL_MONO_WGSL: &str = include_str!("shaders/cylindrical_mono.wgsl");

impl Projection for CylindricalProjection {
    fn name(&self) -> &'static str {
        "cylindrical-mono-1camera"
    }

    fn camera_count(&self) -> u8 {
        1
    }

    fn wgsl_composite_source(&self) -> &str {
        CYLINDRICAL_MONO_WGSL
    }
}

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
/// let aspect = cal.left.width as f32 / cal.left.height as f32;
/// let scene = SceneGeometry::from_layout_with_aspect(&cal.layout, aspect);
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

/// Map a panorama position (yaw/pitch) back to a camera pixel coordinate.
///
/// This is the inverse of [`camera_to_panorama`]. Given a position in the
/// panoramic view, returns the corresponding normalized pixel coordinate
/// in the specified camera's image (or `None` if the position is outside
/// that camera's field of view).
///
/// Useful for:
/// - Projecting panorama-space detections back to camera images
/// - Computing panorama-to-pitch coordinate transforms (consumer territory)
/// - Overlay placement at specific panorama positions
pub fn panorama_to_camera(
    yaw: f32,
    pitch: f32,
    camera: CameraId,
    calibration: &MatchCalibration,
    scene: &SceneGeometry,
) -> Option<(f32, f32)> {
    use nalgebra::{Point3, Vector3};

    let params = match camera {
        CameraId::Left => &calibration.left,
        CameraId::Right => &calibration.right,
    };

    // Step 1: yaw/pitch -> world ray direction, through the same
    // VirtualCamera basis camera_to_panorama uses. Step 2 replaced
    // the previous naive "+Z forward" formula that broke the
    // Forward A vs Forward C roundtrip.
    let cam = VirtualCamera::new(&scene.camera_position);
    let dir = cam.yaw_pitch_to_direction(yaw, pitch);
    let cam_pos = Point3::from(cam.eye);

    // Step 2: ray-plane intersection.
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
        return None; // Ray parallel to plane
    }
    let t = (plane_origin - cam_pos).dot(&plane_normal) / denom;
    if t <= 0.0 {
        return None; // Behind camera
    }
    let hit = cam_pos + dir * t;

    // Step 3: world hit -> extended plane UV. Reject hits outside
    // the plane's renderable region (texture UV [0, 1], equivalently
    // extended UV [-0.5, 1.5] is the full valid range but the plane
    // only covers [0, 1] inside that).
    let (uv_x, uv_y) = world_to_plane_uv(hit, camera, scene)?;
    let tex_u = (uv_x + 0.5) * 0.5;
    let tex_v = (uv_y + 0.5) * 0.5;
    if !(0.0..=1.0).contains(&tex_u) || !(0.0..=1.0).contains(&tex_v) {
        return None;
    }

    // Step 4: extended plane UV -> distorted normalized pixel via
    // the forward KB4 model. The previous implementation passed
    // texture UV in [0, 1] to `lens::undistorted_to_distorted` which
    // expects pixel coordinates; that blew up the lens math and
    // filtered almost every in-coverage point back out as None.
    let (norm_x, norm_y) = forward_fisheye(uv_x, uv_y, params);
    if (0.0..=1.0).contains(&norm_x) && (0.0..=1.0).contains(&norm_y) {
        Some((norm_x as f32, norm_y as f32))
    } else {
        None
    }
}

/// Compute the valid yaw/pitch bounds for a given FOV where no black
/// edges appear in the viewport.
///
/// Samples the visible edges of both camera planes and returns the
/// tightest bounds that keep the viewport fully within the projected
/// image area. Use this to clamp director output for "no-black" panning.
///
/// `aspect` is the viewport width/height ratio (e.g. 16/9 for 1080p).
///
/// Returns `(min_yaw, max_yaw, min_pitch, max_pitch)` in radians.
pub fn viewport_bounds(
    fov_degrees: f32,
    calibration: &MatchCalibration,
    scene: &SceneGeometry,
    aspect: f32,
) -> ViewportBounds {
    // fov_degrees is the VERTICAL FOV (nalgebra Perspective3 convention).
    // Derive horizontal FOV from aspect ratio using rectilinear projection.
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

// ── Coverage Boundary ──────────────────────────────────────────────
//
// A precomputed, pitch-indexed lookup table of valid yaw ranges for
// "no-black" viewport constraining. Replaces the per-frame frontier
// sampling approach in `viewport_bounds` with O(1) runtime lookups.

/// Precomputed coverage boundary for the stitched panorama.
///
/// Maps each pitch angle to the valid yaw range where both camera planes
/// provide pixel data. Built once from calibration (~20k `camera_to_panorama`
/// calls, <1ms on any modern CPU). Runtime lookups are O(1) via pitch-indexed
/// linear interpolation.
///
/// Use [`safe_clamp`](Self::safe_clamp) to constrain a viewport position
/// so no black edges appear.
#[derive(Debug, Clone)]
pub struct CoverageBoundary {
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
    /// Minimum pitch range across all yaw positions.
    /// Determines the maximum safe FOV.
    min_pitch_range: f32,
}

/// Result of clamping a viewport position to the safe panning region.
#[derive(Debug, Clone, Copy)]
pub struct ClampedPosition {
    /// Clamped yaw in radians.
    pub yaw: f32,
    /// Clamped pitch in radians.
    pub pitch: f32,
}

/// Full angular extent of the stitched panorama (radians).
///
/// Returned by [`StitchSession::panorama_extent`](crate::session::StitchSession::panorama_extent).
/// Analytics consumers (heatmaps, zone stats) use this to size grids that
/// span the full coverage rather than hardcoding `±45° yaw, ±20° pitch`.
#[derive(Debug, Clone, Copy)]
pub struct PanoramaExtent {
    /// Minimum yaw with coverage from either camera (radians).
    pub yaw_min: f32,
    /// Maximum yaw with coverage from either camera (radians).
    pub yaw_max: f32,
    /// Minimum pitch with coverage from either camera (radians).
    pub pitch_min: f32,
    /// Maximum pitch with coverage from either camera (radians).
    pub pitch_max: f32,
}

impl PanoramaExtent {
    /// Width of the yaw range in radians.
    pub fn yaw_span(&self) -> f32 {
        self.yaw_max - self.yaw_min
    }

    /// Width of the pitch range in radians.
    pub fn pitch_span(&self) -> f32 {
        self.pitch_max - self.pitch_min
    }

    /// Map an angular position in radians to normalized `[0, 1]`
    /// coordinates within this extent.
    ///
    /// Returns `None` if the extent is degenerate (zero span on either
    /// axis). Values outside the extent are returned as-is (not clamped),
    /// so callers can detect out-of-bounds detections.
    pub fn normalize(&self, yaw: f32, pitch: f32) -> Option<(f32, f32)> {
        let yaw_span = self.yaw_span();
        let pitch_span = self.pitch_span();
        if yaw_span <= 0.0 || pitch_span <= 0.0 {
            return None;
        }
        Some((
            (yaw - self.yaw_min) / yaw_span,
            (pitch - self.pitch_min) / pitch_span,
        ))
    }
}

/// Angular offsets of viewport boundary points from center.
///
impl CoverageBoundary {
    /// Build the coverage boundary from calibration data.
    ///
    /// Densely samples both planes' edge loops and a sparse interior grid,
    /// projecting into (yaw, pitch) space and grouping into pitch slices.
    pub fn from_calibration(calibration: &MatchCalibration, scene: &SceneGeometry) -> Self {
        let n_slices: usize = 400;
        let margin = 0.02_f32;

        let mut left_points: Vec<(f32, f32)> = Vec::new();
        let mut right_points: Vec<(f32, f32)> = Vec::new();

        for &camera in &[CameraId::Left, CameraId::Right] {
            let points = if camera == CameraId::Left {
                &mut left_points
            } else {
                &mut right_points
            };

            // Dense edge sampling: 400 points per edge (4 edges = 1600 per plane)
            let edge_steps = 400_u32;
            for i in 0..=edge_steps {
                let t = margin + (1.0 - 2.0 * margin) * (i as f32 / edge_steps as f32);
                for &(nx, ny) in &[
                    (t, margin),
                    (t, 1.0 - margin),
                    (margin, t),
                    (1.0 - margin, t),
                ] {
                    if let Some(pos) = camera_to_panorama(camera, nx, ny, calibration, scene) {
                        points.push((pos.yaw, pos.pitch));
                    }
                }
            }

            // Sparse interior grid (20x20) for coverage at intermediate pitch levels
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

        // Find global pitch range
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
                min_pitch_range: 0.0,
            };
        }

        let pitch_range = global_pitch_max - global_pitch_min;
        let slice_size = pitch_range / n_slices as f32;

        // Bucket points into pitch slices
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

        // Compute min pitch range (determines max FOV)
        let min_pitch_range = {
            let quarter = n_slices / 4;
            let three_quarter = 3 * n_slices / 4;
            let mut yaw_lo = f32::MAX;
            let mut yaw_hi = f32::MIN;
            for s in &slices[quarter..three_quarter] {
                if s.0 <= s.1 {
                    yaw_lo = yaw_lo.min(s.0);
                    yaw_hi = yaw_hi.max(s.1);
                }
            }

            if yaw_lo >= yaw_hi {
                pitch_range
            } else {
                // Sample pitch range at multiple yaw positions and use the
                // 10th percentile instead of absolute minimum. The absolute
                // minimum is dominated by narrow seam edges which the director
                // rarely visits. The 10th percentile gives a practical FOV
                // that works across most of the useful yaw range.
                let n_samples = 50;
                let mut ranges = Vec::with_capacity(n_samples + 1);
                for j in 0..=n_samples {
                    let t = j as f32 / n_samples as f32;
                    let test_yaw = yaw_lo + t * (yaw_hi - yaw_lo);
                    let mut p_lo = f32::MAX;
                    let mut p_hi = f32::MIN;
                    for (s, (left, right)) in
                        left_slices.iter().zip(right_slices.iter()).enumerate()
                    {
                        let in_left = left.0 <= left.1 && test_yaw >= left.0 && test_yaw <= left.1;
                        let in_right =
                            right.0 <= right.1 && test_yaw >= right.0 && test_yaw <= right.1;
                        if in_left || in_right {
                            let p = global_pitch_min + (s as f32 + 0.5) * slice_size;
                            p_lo = p_lo.min(p);
                            p_hi = p_hi.max(p);
                        }
                    }
                    if p_hi > p_lo {
                        ranges.push(p_hi - p_lo);
                    }
                }
                if ranges.is_empty() {
                    pitch_range
                } else {
                    ranges.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    // 10th percentile: skip the narrowest 10%
                    let idx = (ranges.len() / 10).min(ranges.len() - 1);
                    ranges[idx]
                }
            }
        };

        // Fill gaps: slices with no samples inherit from neighbors
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

        log::info!(
            "CoverageBoundary: pitch [{:.3}, {:.3}] ({:.1} deg), min pitch range {:.1} deg",
            global_pitch_min,
            global_pitch_max,
            pitch_range.to_degrees(),
            min_pitch_range.to_degrees(),
        );

        Self {
            n_slices,
            pitch_min: global_pitch_min,
            pitch_max: global_pitch_max,
            slices,
            left_slices,
            right_slices,
            min_pitch_range,
        }
    }

    /// Global yaw coverage range across the full panorama (radians).
    ///
    /// Returns `(yaw_min, yaw_max)`, the widest-point extremes of the
    /// stitched panorama. Useful for heatmap consumers that need to
    /// bucket detections by yaw across the full coverage.
    ///
    /// This is the global extent, not the per-pitch range. For a
    /// pitch-aware range, sample with [`yaw_range_at`](Self::yaw_range_at).
    pub fn yaw_range(&self) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &(a, b) in &self.slices {
            if a <= b {
                lo = lo.min(a);
                hi = hi.max(b);
            }
        }
        if lo > hi { (0.0, 0.0) } else { (lo, hi) }
    }

    /// Global pitch coverage range across the full panorama (radians).
    ///
    /// Returns `(pitch_min, pitch_max)`. Used alongside
    /// [`yaw_range`](Self::yaw_range) by heatmap and analytics consumers
    /// that need panorama bounds without reaching into private state.
    pub fn pitch_range(&self) -> (f32, f32) {
        (self.pitch_min, self.pitch_max)
    }

    /// Look up the combined yaw coverage range at a given pitch.
    ///
    /// Returns the interpolated `(yaw_min, yaw_max)` where at least one
    /// camera plane provides coverage. Used by the director for
    /// perspective-aware clamping; analytics consumers typically want
    /// [`yaw_range`](Self::yaw_range) for the global extent instead.
    pub fn yaw_range_at(&self, pitch: f32) -> (f32, f32) {
        self.interpolate_slice(&self.slices, pitch)
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
        if lo.0 > lo.1 {
            return hi;
        }
        if hi.0 > hi.1 {
            return lo;
        }
        (lo.0 + frac * (hi.0 - lo.0), lo.1 + frac * (hi.1 - lo.1))
    }

    /// Clamp a viewport position to the safe panning region for a given FOV.
    ///
    /// `rig_tilt` (radians) accounts for the renderer's rig tilt rotation.
    /// The caller passes user-space (yaw, pitch); this method transforms
    /// to world space (+rig_tilt), clamps against coverage, then transforms
    /// back. Pass 0.0 when there is no rig tilt.
    ///
    /// `self` must be the **world-space** coverage boundary.
    pub fn safe_clamp(
        &self,
        yaw: f32,
        pitch: f32,
        fov_v_deg: f32,
        aspect: f32,
        rig_tilt: f32,
    ) -> ClampedPosition {
        // Transform to world space, clamp there, transform back.
        let world_pitch = pitch + rig_tilt;
        let clamped = self.safe_clamp_world(yaw, world_pitch, fov_v_deg, aspect);
        ClampedPosition {
            yaw: clamped.yaw,
            pitch: clamped.pitch - rig_tilt,
        }
    }

    /// Clamp viewport center to coverage with perspective-correct margins.
    fn safe_clamp_world(
        &self,
        yaw: f32,
        pitch: f32,
        fov_v_deg: f32,
        aspect: f32,
    ) -> ClampedPosition {
        // B-30 defense: non-finite inputs would propagate through the
        // clamp / comparisons and emit NaN, which then flows into the
        // MVP matrix and produces a black or garbage frame. Upstream
        // guards (B-28 detector boundary, B-29 director EMA) stop most
        // NaN at the source, but user overrides and external clients
        // can still hand us non-finite values. Fall back to the
        // coverage center.
        if !yaw.is_finite() || !pitch.is_finite() || !fov_v_deg.is_finite() || !aspect.is_finite() {
            let safe_pitch = (self.pitch_min + self.pitch_max) * 0.5;
            let (yaw_lo, yaw_hi) = self.yaw_range_at(safe_pitch);
            let safe_yaw = (yaw_lo + yaw_hi) * 0.5;
            return ClampedPosition {
                yaw: safe_yaw,
                pitch: safe_pitch,
            };
        }

        let half_vfov = (fov_v_deg * 0.5).to_radians();
        let half_hfov = (aspect * half_vfov.tan()).atan();

        // Pitch: global bounds with vertical FOV margin
        let clamped_pitch = if self.pitch_min + half_vfov <= self.pitch_max - half_vfov {
            pitch.clamp(self.pitch_min + half_vfov, self.pitch_max - half_vfov)
        } else {
            (self.pitch_min + self.pitch_max) * 0.5
        };

        // Yaw: coverage range at clamped pitch with horizontal FOV margin
        let (yaw_lo, yaw_hi) = self.yaw_range_at(clamped_pitch);
        let clamped_yaw = if yaw_lo + half_hfov <= yaw_hi - half_hfov {
            yaw.clamp(yaw_lo + half_hfov, yaw_hi - half_hfov)
        } else {
            (yaw_lo + yaw_hi) * 0.5
        };

        ClampedPosition {
            yaw: clamped_yaw,
            pitch: clamped_pitch,
        }
    }

    /// Maximum vertical FOV (degrees) that fits within the coverage.
    ///
    /// This is the widest zoom-out where at least one valid viewport
    /// position exists. Determined by the narrowest pitch range across
    /// all yaw positions (typically at the seam between planes).
    pub fn max_fov_degrees(&self) -> f32 {
        if self.min_pitch_range <= 0.0 {
            return 20.0;
        }
        self.min_pitch_range.to_degrees()
    }

    /// Create a copy with all pitch values shifted by an offset.
    ///
    /// Used to create a tilt-adjusted boundary for the director, which
    /// operates in pre-tilt space while the boundary is in world space.
    /// The director calls `safe_clamp` on the shifted boundary without
    /// needing to know about rig tilt.
    pub fn with_pitch_offset(&self, offset: f32) -> Self {
        Self {
            n_slices: self.n_slices,
            pitch_min: self.pitch_min + offset,
            pitch_max: self.pitch_max + offset,
            slices: self.slices.clone(),
            left_slices: self.left_slices.clone(),
            right_slices: self.right_slices.clone(),
            min_pitch_range: self.min_pitch_range,
        }
    }
}

// ---- Internal functions ----

/// Forward KB4 fisheye: undistorted plane UV → distorted camera pixel [0,1].
///
/// Mirror of [`inverse_fisheye`] in the same normalized-intrinsic
/// convention and the same extended-UV plane space (the shader's
/// `uv * 2.0 - 0.5` remap output). The polynomial delegates to
/// `reco_core::lens::kb4`, same canonical source as the
/// Newton-Raphson step in [`inverse_fisheye`].
fn forward_fisheye(uv_x: f64, uv_y: f64, params: &CameraParams) -> (f64, f64) {
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

    let scale = crate::lens::kb4::kb4_forward_scale(r, &params.d);
    (fx * x * scale + cx, fy * y * scale + cy)
}

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

    // Newton-Raphson: solve f(θ) = θ_d_poly(θ) - θ_d = 0, where
    // θ_d_poly lives in `reco_core::lens::kb4` (SYNC_WITH WGSL).
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
    let r = theta.tan(); // theta = atan(r) → r = tan(theta)
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

/// Exact inverse of [`plane_uv_to_world`].
///
/// Given a world-space point that lies on the named camera's plane,
/// returns its extended-UV coordinate (shader space `[-0.5, 1.5]`).
/// Off-plane points project via the model-matrix inverse: the z
/// component of the model-local position is discarded, so the result
/// is the orthographic projection onto the plane, NOT the ray-plane
/// intersection that `panorama_to_camera` does as a prior step.
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

    // local → texture UV [0,1] (inverse of plane_uv_to_world's inner
    // texture→local step, with plane_width = 1.0 baked in).
    let tex_u = local.x / scene.plane_width + 0.5;
    let tex_v = 0.5 - local.y * scene.plane_aspect / scene.plane_width;

    // Texture UV → extended shader UV (inverse of `uv * 2.0 - 0.5`).
    let uv_x = (tex_u * 2.0 - 0.5) as f64;
    let uv_y = (tex_v * 2.0 - 0.5) as f64;
    Some((uv_x, uv_y))
}

/// Convert a plane UV (in extended shader space) to a 3D world point.
fn plane_uv_to_world(uv: (f64, f64), camera: CameraId, scene: &SceneGeometry) -> Point3<f32> {
    // Extended UV → texture UV [0,1]
    let tex_u = ((uv.0 + 0.5) / 2.0) as f32;
    let tex_v = ((uv.1 + 0.5) / 2.0) as f32;

    // Texture UV → local quad position (matches quad_vertices)
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

/// The virtual camera's orthonormal basis: single source of truth for
/// `(base_forward, base_right, world_up)` shared by `view_matrix` and
/// yaw/pitch decomposition.
///
/// `base_right = base_forward × world_up` so it semantically points
/// to the viewer's right (intuitive). That makes the triple
/// left-handed, which on its own inverts yaw sign between a right-
/// hand rotation around `world_up` (what `view_matrix` does) and the
/// naive `atan2(h · base_right, h · base_forward)` decomposition.
/// The yaw API compensates by negating `h · base_right` in the
/// atan2, so `direction_to_yaw_pitch` and `view_matrix` agree on
/// yaw sign without any downstream reconciliation. `yaw_pitch_to_direction`
/// mirrors the same negation for symmetry.
///
/// Pre-Step-2 the two APIs used the literal `atan2(h · base_right,
/// h · base_forward)` and `cos(yaw)*bf + sin(yaw)*br` forms, so a
/// `yaw=+θ` on one side meant `yaw=-θ` on the other. The Step 1e
/// regression test locked that bug in; this type's yaw convention
/// un-ignores it.
///
/// Rig tilt and rig roll are NOT part of this type. `view_matrix`
/// layers them on top; Step 4 unifies them under `RigCorrection`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VirtualCamera {
    /// World-space camera eye position (copy of `camera_position`).
    pub eye: Vector3<f32>,
    /// Unit vector from eye toward the scene origin (the L-shape corner).
    pub base_forward: Vector3<f32>,
    /// Unit vector to the viewer's right (`base_forward × world_up`,
    /// left-handed triple). The yaw API compensates for the
    /// handedness so there is no sign divergence downstream.
    pub base_right: Vector3<f32>,
}

impl VirtualCamera {
    /// World up axis. Constant `+Y`; exposed as an associated method
    /// rather than a field because every VirtualCamera agrees on it.
    pub fn world_up() -> Vector3<f32> {
        Vector3::new(0.0, 1.0, 0.0)
    }

    /// Build the basis from a world-space camera position.
    pub fn new(camera_position: &[f32; 3]) -> Self {
        let eye = Vector3::new(camera_position[0], camera_position[1], camera_position[2]);
        let base_forward = (-eye).normalize();
        let base_right = base_forward.cross(&Self::world_up()).normalize();
        Self {
            eye,
            base_forward,
            base_right,
        }
    }

    /// Decompose a world-space direction into yaw/pitch relative to
    /// the base forward axis.
    pub fn direction_to_yaw_pitch(&self, dir: &Vector3<f32>) -> ViewportPosition {
        // Pitch: elevation angle from the horizontal plane.
        let pitch = dir.y.clamp(-1.0, 1.0).asin();

        // Yaw: horizontal angle relative to base_forward. The minus
        // sign on (h . base_right) compensates for the left-handed
        // basis so yaw matches view_matrix's right-hand rotation
        // around world_up (see type doc comment).
        let horizontal = Vector3::new(dir.x, 0.0, dir.z);
        let h_len = horizontal.norm();
        let yaw = if h_len > 1e-6 {
            let h = horizontal / h_len;
            let cos_yaw = h.dot(&self.base_forward).clamp(-1.0, 1.0);
            let sin_yaw = -h.dot(&self.base_right);
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

    /// Exact inverse of [`direction_to_yaw_pitch`]. `pitch` is
    /// expected in `(-π/2, π/2)`; at the poles yaw is undefined and
    /// the round-trip through `direction_to_yaw_pitch` collapses.
    pub fn yaw_pitch_to_direction(&self, yaw: f32, pitch: f32) -> Vector3<f32> {
        let cos_pitch = pitch.cos();
        // Matching sign compensation: `-sin(yaw) * base_right` pairs
        // with the `-h . base_right` in direction_to_yaw_pitch so the
        // round-trip is exact.
        let horizontal =
            self.base_forward * (cos_pitch * yaw.cos()) - self.base_right * (cos_pitch * yaw.sin());
        Vector3::new(horizontal.x, pitch.sin(), horizontal.z)
    }
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

/// Exact inverse of [`direction_to_yaw_pitch`]. Only called from
/// tests today; production panorama_to_camera uses the method on
/// [`VirtualCamera`] directly.
#[cfg(test)]
pub(crate) fn yaw_pitch_to_direction(
    yaw: f32,
    pitch: f32,
    camera_position: &[f32; 3],
) -> Vector3<f32> {
    VirtualCamera::new(camera_position).yaw_pitch_to_direction(yaw, pitch)
}

/// Test whether a point lies inside a polygon using the ray-casting algorithm.
///
/// Casts a horizontal ray from the point to the right and counts how many
/// polygon edges it crosses. An odd count means the point is inside.
///
/// Both `point` and `polygon` use `[x, y]` coordinates in any consistent
/// space (typically normalized `[0,1]` camera coordinates).
///
/// Returns `false` for degenerate polygons with fewer than 3 vertices.
pub fn point_in_polygon(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let (px, py) = (point[0], point[1]);
    let mut inside = false;

    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (polygon[i][0], polygon[i][1]);
        let (xj, yj) = (polygon[j][0], polygon[j][1]);

        // Check if the edge from j to i crosses the horizontal ray at py.
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }

        j = i;
    }

    inside
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{CameraParams, MatchCalibration, PlaneLayout};

    fn test_scene(cal: &MatchCalibration) -> SceneGeometry {
        let aspect = cal.left.width as f32 / cal.left.height as f32;
        SceneGeometry::from_layout_with_aspect(&cal.layout, aspect)
    }

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
                x_rx: 0.0,
                z_rz: 0.0,
            },
            rig_tilt: 0.0,
            rig_roll: 0.0,
            sync_offset: 0,
            field_roi: None,
        }
    }

    #[test]
    fn optical_center_maps_to_known_position() {
        let cal = test_calibration();
        let scene = test_scene(&cal);

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
        // SceneGeometry::from_layout_with_aspect), so eye.y = 0 is
        // the real invariant; test positions honor that. Pitch stays
        // clear of ±π/2 where yaw is undefined.
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
    fn world_to_plane_uv_roundtrips_with_plane_uv_to_world() {
        // Step 1b: extended UV → world → extended UV must be the
        // identity. Covers both camera planes and a grid spanning the
        // shader's extended range [-0.5, 1.5] (including points
        // outside the [0,1] texture box so we catch any implicit
        // clamp).
        let cal = test_calibration();
        let scene = test_scene(&cal);

        let uv_steps = [-0.3_f64, -0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.4];

        for &camera in &[CameraId::Left, CameraId::Right] {
            for &u in &uv_steps {
                for &v in &uv_steps {
                    let world = plane_uv_to_world((u, v), camera, &scene);
                    let back = world_to_plane_uv(world, camera, &scene)
                        .expect("model matrix should be invertible");

                    assert!(
                        (back.0 - u).abs() < 1e-5,
                        "uv.x mismatch for camera={camera:?}: sent {u}, got {} (world={world:?})",
                        back.0
                    );
                    assert!(
                        (back.1 - v).abs() < 1e-5,
                        "uv.y mismatch for camera={camera:?}: sent {v}, got {} (world={world:?})",
                        back.1
                    );
                }
            }
        }
    }

    #[test]
    fn inverse_fisheye_roundtrips_with_forward_fisheye_on_pixel_grid() {
        // Step 1c: normalized distorted pixel → extended plane UV →
        // back to normalized distorted pixel must be the identity on
        // a 10x10 grid inside the valid image area. Realistic KB4
        // coefficients (the GoPro HERO10 4K test calibration) make
        // this representative of shipping workloads.
        let params = CameraParams {
            width: 3840,
            height: 2160,
            fx: 1796.32,
            fy: 1797.22,
            cx: 1919.37,
            cy: 1063.17,
            d: [0.0342, 0.0677, -0.0741, 0.0299],
        };

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
    fn camera_to_panorama_roundtrips_with_panorama_to_camera() {
        // Step 1d (un-ignored by Step 2): the full forward chain
        // (camera_to_panorama) then backward chain (panorama_to_camera)
        // must return the same normalized pixel position for points
        // that lie unambiguously within one camera's coverage.
        //
        // Pre-Step-2 panorama_to_camera used a naive "+Z forward" ray
        // (breaking the Forward A vs Forward C agreement) AND passed
        // texture UV to a pixel-space lens helper (filtering
        // in-coverage results back out as None). Step 2 replaced
        // both with VirtualCamera + world_to_plane_uv + forward_fisheye.
        let cal = test_calibration();
        let scene = test_scene(&cal);

        let steps = 5;
        let lo = 0.3_f32;
        let hi = 0.7_f32;

        for &camera in &[CameraId::Left, CameraId::Right] {
            for ix in 0..=steps {
                for iy in 0..=steps {
                    let nx = lo + (hi - lo) * (ix as f32 / steps as f32);
                    let ny = lo + (hi - lo) * (iy as f32 / steps as f32);

                    let pos = camera_to_panorama(camera, nx, ny, &cal, &scene)
                        .expect("forward projection should succeed inside coverage");

                    let back = panorama_to_camera(pos.yaw, pos.pitch, camera, &cal, &scene)
                        .expect("backward projection should land on the same camera");

                    assert!(
                        (back.0 - nx).abs() < 1e-3,
                        "x mismatch for camera={camera:?}: sent {nx}, got {} (yaw={}, pitch={})",
                        back.0,
                        pos.yaw,
                        pos.pitch
                    );
                    assert!(
                        (back.1 - ny).abs() < 1e-3,
                        "y mismatch for camera={camera:?}: sent {ny}, got {} (yaw={}, pitch={})",
                        back.1,
                        pos.yaw,
                        pos.pitch
                    );
                }
            }
        }
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
        let scene = test_scene(&cal);

        // Use a narrower FOV to ensure bounds are valid
        let bounds = viewport_bounds(40.0, &cal, &scene, 16.0 / 9.0);
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
        let scene = test_scene(&cal);

        let narrow = viewport_bounds(30.0, &cal, &scene, 16.0 / 9.0);
        let wide = viewport_bounds(60.0, &cal, &scene, 16.0 / 9.0);

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

    // ── B-30 NaN-resilience regression tests ─────────────────────────

    #[test]
    fn safe_clamp_rejects_nan_yaw() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);
        let out = coverage.safe_clamp(f32::NAN, 0.0, 75.0, 16.0 / 9.0, 0.0);
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
        let out = coverage.safe_clamp(0.0, f32::NAN, 75.0, 16.0 / 9.0, 0.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
    }

    #[test]
    fn safe_clamp_rejects_nan_fov() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);
        let out = coverage.safe_clamp(0.0, 0.0, f32::NAN, 16.0 / 9.0, 0.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
    }

    #[test]
    fn safe_clamp_rejects_infinite_inputs() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);
        let out = coverage.safe_clamp(f32::INFINITY, 0.0, 75.0, 16.0 / 9.0, 0.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
        let out = coverage.safe_clamp(0.0, f32::NEG_INFINITY, 75.0, 16.0 / 9.0, 0.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
    }

    // ── M3 foundation: Projection trait tests ────────────────────────

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
    fn l_shape_projection_wgsl_composite_is_placeholder() {
        // Today the composite shader is embedded in stitch_renderer.
        // LShapeProjection returns "" until StitchCore migration
        // moves that shader source out through this trait.
        let p = LShapeProjection;
        assert!(p.wgsl_composite_source().is_empty());
    }

    // ---- CylindricalProjection (plan step 9) -------------------------------

    #[test]
    fn cylindrical_defaults_match_actionstitch() {
        // Reference values are the actionstitch-player defaults (focal
        // length 2400, 180-deg sweep, no screen tilt, normalized video
        // height). Regresses if someone silently changes the defaults.
        let c = CylindricalProjectionConfig::default();
        assert_eq!(c.focal_length, 2400.0);
        assert!((c.angular_sweep_rad - std::f32::consts::PI).abs() < 1e-6);
        assert_eq!(c.screen_rotation_rad, 0.0);
        assert_eq!(c.video_height, 1.0);
    }

    #[test]
    fn cylindrical_projection_reports_mono() {
        let p = CylindricalProjection::default();
        assert_eq!(p.name(), "cylindrical-mono-1camera");
        assert_eq!(
            p.camera_count(),
            1,
            "cylindrical projection consumes exactly one camera"
        );
    }

    #[test]
    fn cylindrical_theta_start_matches_actionstitch_formula() {
        // actionstitch's CylinderGeometry uses thetaStart = PI/2 - s/2
        // where s is the angular sweep. Verify for 180-deg (default)
        // and for a 90-deg cylinder (quarter sweep).
        let p180 = CylindricalProjection::default();
        assert!(
            (p180.theta_start_rad() - std::f32::consts::FRAC_PI_2 * 0.0).abs() < 1e-6,
            "180-deg sweep: theta_start = PI/2 - PI/2 = 0"
        );

        let p90 = CylindricalProjection::new(CylindricalProjectionConfig {
            angular_sweep_rad: std::f32::consts::FRAC_PI_2,
            ..Default::default()
        });
        // 90-deg: theta_start = PI/2 - PI/4 = PI/4.
        assert!((p90.theta_start_rad() - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    }

    #[test]
    fn cylindrical_wgsl_source_is_nonempty_and_has_expected_entrypoints() {
        // Sanity-check the embedded shader compiles in spirit: it must
        // declare both the vertex + fragment entry points the composite
        // pass expects. Full wgpu compilation lives in an integration
        // test behind the GPU gate.
        let p = CylindricalProjection::default();
        let src = p.wgsl_composite_source();
        assert!(!src.is_empty());
        assert!(src.contains("fn vs_fullscreen"));
        assert!(src.contains("fn fs_cylindrical_mono"));
        assert!(src.contains("CylUniforms"));
    }

    #[test]
    fn projection_dyn_dispatch_round_trip_with_mixed_camera_counts() {
        // Compile-time: `Box<dyn Projection>` can hold concrete impls
        // with different `camera_count()` results. Proves the trait
        // API's claim that consumers can swap projections without a
        // new type parameter on StitchCore.
        let projections: Vec<Box<dyn Projection>> = vec![
            Box::new(LShapeProjection),
            Box::new(CylindricalProjection::default()),
        ];
        assert_eq!(projections[0].camera_count(), 2);
        assert_eq!(projections[1].camera_count(), 1);
        assert_ne!(projections[0].name(), projections[1].name());
    }

    #[test]
    fn cylindrical_projection_is_send_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<CylindricalProjection>();
        assert_send_sync::<CylindricalProjectionConfig>();
    }
}
