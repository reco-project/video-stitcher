//! Calibration objectives and optimization must account for an out-of-plane camera pitch.

use reco_calibrate::geometry::{
    OptParams, angular_error, per_point_reprojection_error, per_point_seam_weighted_errors,
    reprojection_error, seam_weighted_reprojection_error,
};
use reco_calibrate::optimizer::optimize;
use reco_calibrate::types::{CalibrationConfig, MatchedPoint, OptimizerConfig};

fn parameters(pitch: f64) -> OptParams {
    OptParams {
        intersect: 0.5,
        cam_d: 0.25,
        x_ty: 0.0,
        x_rz: 0.0,
        z_rx: 0.0,
        z_rz: None,
        x_rx: Some(pitch),
    }
}

fn correspondences(pitch: f64) -> Vec<MatchedPoint> {
    // Independent analytic fixture: rotate (x, -y, 0) around X, translate to (0.25, 0, 0), and extend its camera ray to the other plane at world x=0.
    let mut points = Vec::new();
    for x in [-0.22, -0.18, -0.14, -0.1] {
        for y in [-0.12, -0.06, 0.0, 0.06, 0.12] {
            let wy = -y * pitch.cos();
            let wz = -y * pitch.sin();
            let t = -0.25 / x;
            let hit_y = t * wy;
            let hit_z = 0.25 + t * (wz - 0.25);
            points.push(MatchedPoint {
                left: [x, y],
                right: [0.25 - hit_z, -hit_y],
                left_pixel_nx: 0.25,
                right_pixel_nx: 0.75,
            });
        }
    }
    points
}

#[test]
fn tilted_correspondences_have_zero_cost_only_at_the_matching_pitch() {
    for pitch in [-0.2, -0.08, 0.08, 0.2] {
        let points = correspondences(pitch);
        let truth = parameters(pitch);
        let wrong = parameters(0.0);
        let raw = per_point_reprojection_error(&points, &truth);
        let weighted = per_point_seam_weighted_errors(&points, &truth, 0.08);
        assert!(raw.iter().all(|e| e.is_finite() && *e < 1e-24));
        assert!(weighted.iter().all(|e| e.is_finite() && *e < 1e-24));
        assert!(angular_error(&points, &truth) < 1e-6);
        assert_eq!(reprojection_error(&points, &truth), raw.iter().sum::<f64>());
        assert_eq!(
            seam_weighted_reprojection_error(&points, &truth, 0.08),
            weighted.iter().sum::<f64>()
        );
        assert!(reprojection_error(&points, &wrong) > 1e-4);
        assert!(seam_weighted_reprojection_error(&points, &wrong, 0.08) > 1e-5);
    }
}

#[test]
fn parallel_and_behind_camera_rays_are_penalized_consistently() {
    for position in [0.0, 0.1] {
        let points = [MatchedPoint::from_planes([position, 0.0], [position, 0.0])];
        let params = parameters(0.0);
        let per_point = per_point_reprojection_error(&points, &params);
        assert!(per_point[0].is_finite() && per_point[0] >= 1e6);
        assert_eq!(reprojection_error(&points, &params), per_point[0]);
        let weighted = per_point_seam_weighted_errors(&points, &params, 0.08);
        assert!(weighted[0].is_finite() && weighted[0] > 0.0);
        assert_eq!(
            seam_weighted_reprojection_error(&points, &params, 0.08),
            weighted[0]
        );
    }
    let degenerate = OptParams {
        cam_d: 0.0,
        intersect: 1.0,
        ..parameters(0.0)
    };
    let points = [MatchedPoint::from_planes([0.0; 2], [0.0; 2])];
    assert_eq!(reprojection_error(&points, &degenerate), 2e6);
}

#[test]
fn optimizer_recovers_pitch_without_an_imu_seed() {
    for pitch in [-0.12, 0.12] {
        let points = correspondences(pitch);
        let config = CalibrationConfig {
            optimizer: OptimizerConfig {
                enable_x_rx: true,
                lock_cam_d: true,
                lock_z_rx: true,
                trim_fraction: 0.0,
                max_iters: 1200,
                ..Default::default()
            },
            ..Default::default()
        };
        let (shape, framing, cost) = optimize(&points, &config).unwrap();
        eprintln!(
            "pitch={pitch}: recovered={}, cost={cost}, axis={}",
            shape.x_rx, framing.axis_offset
        );
        assert!((shape.x_rx - pitch).abs() < 0.002);
        assert!(cost < 1e-8);
        let disabled = CalibrationConfig {
            optimizer: OptimizerConfig {
                enable_x_rx: false,
                ..config.optimizer.clone()
            },
            ..config
        };
        let (shape, _, cost_without_pitch) = optimize(&points, &disabled).unwrap();
        assert_eq!(shape.x_rx, 0.0);
        assert!(cost_without_pitch > cost + 1e-6);
    }
}

#[test]
fn rendered_correspondences_agree_with_the_full_rotated_plane_objective() {
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::geometry::RenderPose;
    use reco_core::projection::LShape;
    use reco_core::render::viewport::ViewportSize;

    for sign in [-1.0, 1.0] {
        let params = OptParams {
            intersect: 0.7,
            x_ty: 0.01,
            x_rz: 0.08 * sign,
            z_rx: -0.06 * sign,
            z_rz: Some(0.1 * sign),
            ..parameters(0.12 * sign)
        };
        let sizes = [(128, 96), (160, 90)];
        let cal = Calibration::new(
            sizes
                .map(|(w, h)| {
                    Lens::fisheye(
                        w,
                        h,
                        w as f64 / 2.0,
                        h as f64 / 2.0,
                        w as f64 / 2.0,
                        h as f64 / 2.0,
                        [0.0; 4],
                    )
                })
                .to_vec(),
            LShape {
                intersect: params.intersect,
                x_ty: params.x_ty,
                x_rz: params.x_rz,
                z_rx: params.z_rx,
                x_rx: params.x_rx.unwrap(),
                z_rz: params.z_rz.unwrap(),
                blend_width: 0.0,
            },
            Framing {
                axis_offset: params.cam_d,
                tilt: 0.12,
                roll: -0.08,
            },
        );
        cal.validate().unwrap();
        let maps = cal.surface_maps(
            &ViewportSize {
                width: 128,
                height: 96,
            },
            RenderPose {
                yaw: 0.0,
                pitch: 0.0,
                fov_degrees: 70.0,
            },
        );
        let mut points = Vec::new();
        for y in -8..=8 {
            for x in -8..=8 {
                let p = [x as f64 / 10.0, y as f64 / 10.0];
                let (Some(left), Some(right)) = (maps[0].0.sample_uv(p), maps[1].0.sample_uv(p))
                else {
                    continue;
                };
                let local = |uv: [f64; 2], (w, h): (u32, u32)| {
                    let dx = (uv[0] - 0.5) * 2.0;
                    let dy = (uv[1] - 0.5) * 2.0;
                    let theta = dx.hypot(dy);
                    let scale = if theta < 1e-12 {
                        1.0
                    } else {
                        theta.tan() / theta
                    };
                    [dx * scale / 4.0, dy * scale / 4.0 * h as f64 / w as f64]
                };
                points.push(MatchedPoint {
                    left: local([right.u, right.v], sizes[1]),
                    right: local([left.u, left.v], sizes[0]),
                    left_pixel_nx: right.u,
                    right_pixel_nx: left.u,
                });
            }
        }
        assert!(points.len() > 20, "fixture must include overlap");
        let errors = per_point_reprojection_error(&points, &params);
        assert!(
            errors.iter().all(|e| *e < 1e-11),
            "max error: {:?}",
            errors.iter().copied().reduce(f64::max)
        );
    }
}
