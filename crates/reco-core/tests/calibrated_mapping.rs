//! External-renderer contract: use calibrated maps without duplicating projection or blend math.

use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::geometry::{RenderPose, WorldPose};
use reco_core::projection::{Cylinder, LShape};
use reco_core::render::planes::Nv12Planes;
use reco_core::render::viewport::ViewportSize;
use reco_core::stitch::{BlendRule, CpuExecutor};

#[test]
fn calibrated_pose_conversion_needs_no_engine() {
    for calibration in calibrations() {
        for yaw in [-0.7, 0.0, 0.8] {
            let world = WorldPose {
                yaw,
                pitch: 0.2,
                fov_degrees: 53.0,
            };
            let render = calibration.world_to_render_pose(world);
            let recovered = calibration.render_to_world_pose(render);
            assert!((recovered.yaw - world.yaw).abs() < 1e-5);
            assert!((recovered.pitch - world.pitch).abs() < 1e-5);
            assert_eq!(recovered.fov_degrees, world.fov_degrees);
        }
    }
}

fn calibrations() -> [Calibration; 2] {
    let lens = Lens::flat(64, 32);
    let framing = Framing {
        axis_offset: 0.25,
        tilt: 0.12,
        roll: -0.08,
    };
    [
        Calibration::new(
            vec![lens.clone(), lens.clone()],
            LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.2,
            },
            framing,
        ),
        Calibration::new(
            vec![lens],
            Cylinder {
                radius: 20.0,
                arc_deg: 180.0,
                surface_height: None,
            },
            Framing {
                axis_offset: 0.0,
                tilt: 0.12,
                roll: -0.08,
            },
        ),
    ]
}

#[test]
fn external_compositor_matches_cpu_for_both_projections() {
    let size = ViewportSize {
        width: 96,
        height: 54,
    };
    for cal in calibrations() {
        cal.validate().unwrap();
        let count = cal.projection.camera_count();
        let executor = {
            let source_sizes = vec![(64, 32); cal.projection.camera_count()];
            CpuExecutor::new(cal.clone(), size.clone(), source_sizes, false)
        }
        .unwrap();
        let mut seen = vec![false; count];
        let mut saw_blend = false;
        let mut saw_uncovered = false;
        for yaw in [-1.0, 0.0, 1.0] {
            for fov_degrees in [35.0, 110.0] {
                let pose = RenderPose {
                    yaw,
                    pitch: 0.1,
                    fov_degrees,
                };
                let maps = cal.surface_maps(&size, pose);
                assert_eq!(maps.len(), count);
                // Activate each input separately to detect incorrect source order or blending.
                for active in 0..count {
                    let luma: Vec<_> = (0..count)
                        .map(|camera| vec![if camera == active { 235 } else { 16 }; 64 * 32])
                        .collect();
                    let chroma = vec![128; 64 * 32 / 2];
                    let planes: Vec<_> =
                        luma.iter().map(|y| Nv12Planes { y, uv: &chroma }).collect();
                    let rendered = executor.stitch_nv12(&planes, pose).unwrap();
                    for y in 0..size.height {
                        for x in 0..size.width {
                            let ndc = [
                                (f64::from(x) + 0.5) / f64::from(size.width) * 2.0 - 1.0,
                                1.0 - (f64::from(y) + 0.5) / f64::from(size.height) * 2.0,
                            ];
                            let mut value = 0.0;
                            let mut supported = false;
                            for (camera, (map, blend)) in maps.iter().enumerate() {
                                if let Some(sample) = map.sample_uv(ndc) {
                                    assert!((0.0..=1.0).contains(&sample.u));
                                    assert!((0.0..=1.0).contains(&sample.v));
                                    seen[camera] = true;
                                    supported = true;
                                    let alpha = blend.alpha(sample.edge);
                                    assert!((0.0..=1.0).contains(&alpha));
                                    saw_blend |= alpha > 0.0 && alpha < 1.0;
                                    let source = if camera == active { 1.0 } else { 0.0 };
                                    value += (source - value) * alpha;
                                }
                            }
                            saw_uncovered |= !supported;
                            let expected = (value * 255.0).round() as u8;
                            let offset = ((y * size.width + x) * 4) as usize;
                            for channel in &rendered[offset..offset + 3] {
                                assert!(channel.abs_diff(expected) <= 1);
                            }
                            assert_eq!(rendered[offset + 3], 255);
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|seen| *seen));
        assert!(saw_uncovered);
        if count == 2 {
            assert!(saw_blend);
        }
    }
}

#[test]
fn public_blend_weights_cover_seam_endpoints() {
    assert_eq!(BlendRule::Opaque.alpha(-1.0), 1.0);
    assert_eq!(BlendRule::Smoothstep(0.0).alpha(-1.0), 1.0);
    let blend = BlendRule::Smoothstep(0.2);
    assert_eq!(blend.alpha(-0.1), 0.0);
    assert_eq!(blend.alpha(0.0), 0.0);
    assert_eq!(blend.alpha(0.1), 0.5);
    assert_eq!(blend.alpha(0.2), 1.0);
    assert_eq!(blend.alpha(0.3), 1.0);
}

#[test]
fn each_camera_geometry_is_independent_of_its_peer_aspect() {
    let size = ViewportSize {
        width: 96,
        height: 54,
    };
    let mut base = calibrations()[0].clone();
    base.lenses =
        vec![Lens::fisheye(128, 96, 66.0, 61.0, 62.0, 46.0, [0.03, -0.01, 0.002, 0.0]); 2];
    for camera in 0..2 {
        for correction in [0.0, 0.4, 1.0] {
            base.lenses[camera].correction = correction;
            let mut changed = base.clone();
            changed.lenses[1 - camera] = base.lenses[1 - camera].rescaled(128, 64).unwrap();
            changed.validate().unwrap();
            for point in [[0.3, 0.3], [0.5, 0.5], [0.7, 0.7]] {
                let expected = base.map_source_point(camera, point).unwrap();
                let actual = changed.map_source_point(camera, point).unwrap();
                assert!((actual.yaw - expected.yaw).abs() < 1e-6);
                assert!(
                    (actual.pitch - expected.pitch).abs() < 1e-6,
                    "camera={camera}, correction={correction}, point={point:?}: {actual:?} != {expected:?}"
                );
            }
            for yaw in [-0.7, 0.0, 0.7] {
                let pose = RenderPose {
                    yaw,
                    pitch: 0.1,
                    fov_degrees: 80.0,
                };
                let a = base.surface_maps(&size, pose);
                let b = changed.surface_maps(&size, pose);
                let (a, b) = (&a[camera].0, &b[camera].0);
                let mut visible = 0;
                for y in -10..=10 {
                    for x in -10..=10 {
                        let p = [x as f64 / 10.0, y as f64 / 10.0];
                        match (a.sample_uv(p), b.sample_uv(p)) {
                            (Some(a), Some(b)) => {
                                visible += 1;
                                assert!((a.u - b.u).abs() < 1e-9);
                                assert!((a.v - b.v).abs() < 1e-9);
                            }
                            (None, None) => {}
                            _ => panic!("peer aspect changed camera {camera} support"),
                        }
                        for rotation in [0.0, 0.002] {
                            let max = [p[0] + 0.02, p[1] + 0.02];
                            assert_eq!(
                                a.covers_ndc(p, max, rotation),
                                b.covers_ndc(p, max, rotation)
                            );
                        }
                    }
                }
                assert!(visible > 0);
            }
        }
    }
}

#[test]
fn mixed_aspect_maps_round_trip_and_certify_only_covered_regions() {
    let size = ViewportSize {
        width: 96,
        height: 54,
    };
    for heights in [[96, 64], [64, 96]] {
        let mut cal = calibrations()[0].clone();
        let sign = if heights[0] > heights[1] { 1.0 } else { -1.0 };
        let shape = cal.projection.l_shape_mut().unwrap();
        shape.x_rx = sign * 0.12;
        shape.z_rz = sign * 0.1;
        shape.x_rz = sign * 0.08;
        shape.z_rx = -sign * 0.06;
        cal.lenses = heights
            .map(|height| {
                Lens::fisheye(
                    128,
                    height,
                    66.0,
                    61.0,
                    62.0,
                    f64::from(height) * 0.48,
                    [0.03, -0.01, 0.002, 0.0],
                )
            })
            .to_vec();
        for correction in [0.0, 0.4, 1.0] {
            for lens in &mut cal.lenses {
                lens.correction = correction;
            }
            cal.validate().unwrap();
            for camera in 0..2 {
                for y in 3..=7 {
                    for x in 3..=7 {
                        let point = [x as f32 / 10.0, y as f32 / 10.0];
                        let world = cal.map_source_point(camera, point).unwrap();
                        let pose = cal.world_to_render_pose(WorldPose {
                            yaw: world.yaw,
                            pitch: world.pitch,
                            fov_degrees: 45.0,
                        });
                        let maps = cal.surface_maps(&size, pose);
                        let sampled = maps[camera].0.sample_uv([0.0; 2]).unwrap();
                        assert!((sampled.u - f64::from(point[0])).abs() < 1e-5);
                        assert!((sampled.v - f64::from(point[1])).abs() < 1e-5);
                    }
                }
            }
            let mut certified = [0; 2];
            let mut rejected = [0; 2];
            for yaw in [-0.7, 0.0, 0.7] {
                let maps = cal.surface_maps(
                    &size,
                    RenderPose {
                        yaw,
                        pitch: 0.1,
                        fov_degrees: 100.0,
                    },
                );
                for (camera, (map, _)) in maps.iter().enumerate() {
                    for y in -10..10 {
                        for x in -10..10 {
                            let min = [x as f64 / 10.0, y as f64 / 10.0];
                            let max = [min[0] + 0.1, min[1] + 0.1];
                            if !map.covers_ndc(min, max, 0.0) {
                                rejected[camera] += 1;
                                continue;
                            }
                            certified[camera] += 1;
                            for j in 0..=4 {
                                for i in 0..=4 {
                                    assert!(
                                        map.sample_uv([
                                            min[0] + i as f64 * 0.025,
                                            min[1] + j as f64 * 0.025
                                        ])
                                        .is_some()
                                    );
                                }
                            }
                        }
                    }
                }
            }
            assert!(certified.into_iter().all(|n| n > 0));
            assert!(rejected.into_iter().all(|n| n > 0));
        }
    }
}
