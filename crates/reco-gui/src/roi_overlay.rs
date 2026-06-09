use reco_core::calibration::{CameraParams, FieldRoi, MatchCalibration};
use reco_core::detect::detector::CameraId;
use reco_core::render::scene::SceneGeometry;

const ROI_DRAG_HIT_RADIUS_PX: f64 = 12.0;

#[derive(Default)]
pub(crate) struct RoiOverlayGeometry {
    pub(crate) points_x: Vec<f32>,
    pub(crate) points_y: Vec<f32>,
    pub(crate) lines_x1: Vec<f32>,
    pub(crate) lines_y1: Vec<f32>,
    pub(crate) lines_x2: Vec<f32>,
    pub(crate) lines_y2: Vec<f32>,
}

impl RoiOverlayGeometry {
    fn push_point(&mut self, x: f32, y: f32) {
        self.points_x.push(x);
        self.points_y.push(y);
    }

    fn push_segment(&mut self, from: (f32, f32), to: (f32, f32)) {
        self.lines_x1.push(from.0);
        self.lines_y1.push(from.1);
        self.lines_x2.push(to.0);
        self.lines_y2.push(to.1);
    }
}

pub(crate) fn has_effective_roi(roi: &FieldRoi) -> bool {
    !roi.points.is_empty()
}

pub(crate) fn nearest_roi_point(
    points: &[(f32, f32)],
    point: [f32; 2],
    image_width_px: f32,
    image_height_px: f32,
) -> Option<usize> {
    let image_width_px = f64::from(image_width_px.max(1.0));
    let image_height_px = f64::from(image_height_px.max(1.0));
    let max_dist_sq = ROI_DRAG_HIT_RADIUS_PX * ROI_DRAG_HIT_RADIUS_PX;

    points
        .iter()
        .enumerate()
        .filter_map(|(idx, &(candidate_x, candidate_y))| {
            let dx = f64::from(candidate_x - point[0]) * image_width_px;
            let dy = f64::from(candidate_y - point[1]) * image_height_px;
            let dist_sq = dx * dx + dy * dy;
            (dist_sq <= max_dist_sq).then_some((idx, dist_sq))
        })
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
}

pub(crate) fn scene_for_calibration(calibration: &MatchCalibration) -> SceneGeometry {
    let scene_aspect = calibration.left.width as f32 / calibration.left.height.max(1) as f32;
    SceneGeometry::from_layout_with_aspect(&calibration.layout, scene_aspect)
}

pub(crate) fn lens_preview_roi_geometry(
    points: &[[f64; 2]],
    camera: CameraId,
    calibration: &MatchCalibration,
    scene: &SceneGeometry,
    params: &CameraParams,
    lens_correction_amount: f32,
) -> RoiOverlayGeometry {
    roi_geometry_from_projector(points, 12, |point| {
        let raw = reco_core::projection::panorama_to_camera(
            point[0] as f32,
            point[1] as f32,
            camera,
            calibration,
            scene,
        )?;
        reco_core::projection::camera_to_lens_preview(raw.0, raw.1, params, lens_correction_amount)
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn viewport_roi_geometry(
    points: &[[f64; 2]],
    scene: &SceneGeometry,
    view_yaw: f32,
    view_pitch: f32,
    fov_degrees: f32,
    aspect: f32,
    rig_tilt: f32,
    rig_roll: f32,
) -> RoiOverlayGeometry {
    let project = |point: [f64; 2]| {
        reco_core::projection::panorama_to_viewport(
            point[0] as f32,
            point[1] as f32,
            view_yaw,
            view_pitch,
            fov_degrees,
            aspect,
            rig_tilt,
            rig_roll,
            scene,
        )
    };

    let mut geometry = RoiOverlayGeometry::default();
    for &point in points {
        if let Some((x, y)) = project(point) {
            geometry.push_point(x, y);
        }
    }

    if points.len() < 2 {
        return geometry;
    }

    let edge_count = if points.len() >= 3 {
        points.len()
    } else {
        points.len() - 1
    };

    for edge_idx in 0..edge_count {
        let a = points[edge_idx];
        let b = points[(edge_idx + 1) % points.len()];

        if let Some((from, to)) = reco_core::projection::panorama_segment_to_viewport(
            a[0] as f32,
            a[1] as f32,
            b[0] as f32,
            b[1] as f32,
            view_yaw,
            view_pitch,
            fov_degrees,
            aspect,
            rig_tilt,
            rig_roll,
            scene,
        ) && let Some((from, to)) = clip_segment_to_unit_rect(from, to)
        {
            geometry.push_segment(from, to);
        }
    }

    geometry
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn roi_point_from_viewport(
    scene: &SceneGeometry,
    norm_x: f32,
    norm_y: f32,
    view_yaw: f32,
    view_pitch: f32,
    fov_degrees: f32,
    aspect: f32,
    rig_tilt: f32,
    rig_roll: f32,
) -> Option<[f64; 2]> {
    let pos = reco_core::projection::viewport_to_panorama(
        norm_x.clamp(0.0, 1.0),
        norm_y.clamp(0.0, 1.0),
        view_yaw,
        view_pitch,
        fov_degrees,
        aspect,
        rig_tilt,
        rig_roll,
        scene,
    )?;
    Some([pos.yaw as f64, pos.pitch as f64])
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn viewport_roi_screen_points(
    points: &[[f64; 2]],
    scene: &SceneGeometry,
    view_yaw: f32,
    view_pitch: f32,
    fov_degrees: f32,
    aspect: f32,
    rig_tilt: f32,
    rig_roll: f32,
) -> Vec<(f32, f32)> {
    points
        .iter()
        .filter_map(|point| {
            reco_core::projection::panorama_to_viewport(
                point[0] as f32,
                point[1] as f32,
                view_yaw,
                view_pitch,
                fov_degrees,
                aspect,
                rig_tilt,
                rig_roll,
                scene,
            )
        })
        .collect()
}

fn roi_geometry_from_projector(
    points: &[[f64; 2]],
    samples_per_edge: usize,
    mut project: impl FnMut([f64; 2]) -> Option<(f32, f32)>,
) -> RoiOverlayGeometry {
    let mut geometry = RoiOverlayGeometry::default();

    for &point in points {
        if let Some((x, y)) = project(point) {
            geometry.push_point(x, y);
        }
    }

    if points.len() < 2 {
        return geometry;
    }

    let edge_count = if points.len() >= 3 {
        points.len()
    } else {
        points.len() - 1
    };

    for edge_idx in 0..edge_count {
        let a = points[edge_idx];
        let b = points[(edge_idx + 1) % points.len()];
        let mut last_projected = None;

        for sample_idx in 0..=samples_per_edge.max(1) {
            let t = sample_idx as f64 / samples_per_edge.max(1) as f64;
            let point = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

            if let Some(projected) = project(point) {
                if let Some(previous) = last_projected {
                    geometry.push_segment(previous, projected);
                }
                last_projected = Some(projected);
            } else {
                last_projected = None;
            }
        }
    }

    geometry
}

fn clip_segment_to_unit_rect(from: (f32, f32), to: (f32, f32)) -> Option<((f32, f32), (f32, f32))> {
    let (x0, y0) = from;
    let (x1, y1) = to;
    let dx = x1 - x0;
    let dy = y1 - y0;
    let mut t0 = 0.0;
    let mut t1 = 1.0;

    for (p, q) in [(-dx, x0), (dx, 1.0 - x0), (-dy, y0), (dy, 1.0 - y0)] {
        if p.abs() < f32::EPSILON {
            if q < 0.0 {
                return None;
            }
            continue;
        }

        let r = q / p;
        if p < 0.0 {
            if r > t1 {
                return None;
            }
            if r > t0 {
                t0 = r;
            }
        } else {
            if r < t0 {
                return None;
            }
            if r < t1 {
                t1 = r;
            }
        }
    }

    Some(((x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)))
}
