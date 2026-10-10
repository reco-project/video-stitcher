use super::*;
use crate::calibration::{FieldRoi, Framing, Lens};
use crate::projection::{Cylinder, LShape};

fn calibrations() -> [Calibration; 2] {
    let lens = Lens::fisheye(
        3840,
        2160,
        1796.32,
        1797.22,
        1919.37,
        1063.17,
        [0.0342, 0.0677, -0.0741, 0.0299],
    );
    [
        Calibration::new(
            vec![lens.clone(), lens],
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
                tilt: 0.13,
                roll: -0.08,
            },
        ),
        Calibration::new(
            vec![Lens::flat(3840, 2160)],
            Cylinder::default(),
            Framing {
                axis_offset: 0.0,
                tilt: 0.13,
                roll: -0.08,
            },
        ),
    ]
}

const SIZE: ViewportSize = ViewportSize {
    width: 1280,
    height: 720,
};

fn region(cal: &mut Calibration) -> RenderPose {
    let pose = cal.world_to_render_pose(WorldPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 65.0,
    });
    cal.field_roi = Some(FieldRoi {
        corners: [[0.2, 0.25], [0.8, 0.25], [0.8, 0.75], [0.2, 0.75]].map(|p| {
            cal.pick_source_point(&SIZE, pose, p)
                .expect("supported corner")
        }),
    });
    cal.validate().unwrap();
    pose
}

#[test]
fn source_selection_roundtrips_through_rotated_panned_viewports() {
    for cal in calibrations() {
        for yaw in [-0.3, 0.0, 0.3] {
            let pose = RenderPose {
                yaw,
                pitch: 0.1,
                fov_degrees: 75.0,
            };
            let mut found = 0;
            for y in 1..10 {
                for x in 1..10 {
                    let p = [x as f64 / 10.0, y as f64 / 10.0];
                    if let Some(raw) = cal.pick_source_point(&SIZE, pose, p) {
                        let back = cal.project_source_point(raw, &SIZE, pose).unwrap().unwrap();
                        assert!(
                            (back[0] - p[0]).abs() < 2e-6 && (back[1] - p[1]).abs() < 2e-6,
                            "{p:?} -> {raw:?} -> {back:?}"
                        );
                        found += 1;
                    }
                }
            }
            assert!(found > 20);
        }
    }
}

#[test]
fn calibrated_membership_and_ownership_match_the_rendered_region() {
    for mut cal in calibrations() {
        let pose = region(&mut cal);
        let boundary = cal.field_boundary().unwrap().unwrap();
        let maps = cal.surface_maps(&SIZE, pose);
        let mut overlap = 0;
        for y in 1..20 {
            for x in 1..20 {
                let p = [x as f64 / 20.0, y as f64 / 20.0];
                let ndc = [2.0 * p[0] - 1.0, 1.0 - 2.0 * p[1]];
                let expected_inside = p[0] > 0.201 && p[0] < 0.799 && p[1] > 0.251 && p[1] < 0.749;
                // Avoid the f32 renderer's subpixel boundary noise in this interior oracle.
                if [0.2, 0.8].iter().any(|v| (p[0] - v).abs() < 0.001)
                    || [0.25, 0.75].iter().any(|v| (p[1] - v).abs() < 0.001)
                {
                    continue;
                }
                let owner = dominant_source(&maps, ndc);
                let mut owners = 0;
                let mut supported = 0;
                for (camera, (map, _)) in maps.iter().enumerate() {
                    if let Some(uv) = map.sample_uv(ndc) {
                        let point = SourcePoint {
                            camera,
                            position: [uv.u, uv.v],
                        };
                        assert_eq!(boundary.contains(point), expected_inside, "{camera}: {p:?}");
                        if expected_inside {
                            assert_eq!(
                                boundary.owns(point),
                                owner.is_some_and(|o| o.camera == camera),
                                "owner {camera}: {p:?}"
                            );
                            owners += usize::from(boundary.owns(point));
                        }
                        supported += 1;
                    }
                }
                if expected_inside {
                    assert_eq!(owners, usize::from(owner.is_some()));
                }
                overlap += usize::from(supported > 1);
            }
        }
        if cal.lenses.len() == 2 {
            assert!(overlap > 0);
        }
        let outline = boundary.outline(&cal, &SIZE, pose);
        assert_eq!(outline.len(), 4);
        for (p, expected) in
            outline
                .iter()
                .zip([[0.2, 0.25], [0.8, 0.25], [0.8, 0.75], [0.2, 0.75]])
        {
            assert!((p[0] - expected[0]).abs() < 2e-6 && (p[1] - expected[1]).abs() < 2e-6);
        }
    }
}

#[test]
fn corners_are_source_attached_and_both_windings_work() {
    for mut cal in calibrations() {
        region(&mut cal);
        let original = cal.field_roi.clone();
        let point = cal.field_roi.as_ref().unwrap().corners[0];
        cal.field_roi.as_mut().unwrap().corners.reverse();
        assert!(cal.field_boundary().unwrap().unwrap().contains(point));
        cal.framing.roll += 0.25;
        cal.framing.tilt -= 0.13;
        let boundary = cal.field_boundary().unwrap().unwrap();
        assert!(boundary.contains(point));
        for yaw in [-0.4, 0.0, 0.4] {
            let pose = RenderPose {
                yaw,
                pitch: 0.1,
                fov_degrees: 45.0,
            };
            let outline = boundary.outline(&cal, &SIZE, pose);
            assert!(
                outline
                    .iter()
                    .flatten()
                    .all(|v| v.is_finite() && (-1e-8..=1.0 + 1e-8).contains(v))
            );
        }
        cal.field_roi.as_mut().unwrap().corners.reverse();
        assert_eq!(cal.field_roi, original);
        let restored: Calibration = serde_json::from_str(&cal.to_json_pretty()).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.field_roi, cal.field_roi);
    }
}

#[test]
fn bad_regions_never_turn_into_unfiltered_or_clamped_regions() {
    let mut cal = calibrations()[0].clone();
    region(&mut cal);
    let good = cal.field_roi.clone();
    for case in 0..6 {
        cal.field_roi = good.clone();
        let corners = &mut cal.field_roi.as_mut().unwrap().corners;
        match case {
            0 => corners.swap(1, 2),
            1 => corners[1] = corners[0],
            2 => corners[0].position[0] = f64::NAN,
            3 => corners[0].position[1] = -0.1,
            4 => corners[0].camera = 2,
            _ => cal.lenses.clear(),
        }
        assert!(cal.field_boundary().is_err(), "case {case}");
    }
    for raw in [r#"{"left":[],"right":[]}"#, r#"{"corners":[]}"#] {
        assert!(serde_json::from_str::<FieldRoi>(raw).is_err());
    }
}

#[test]
fn a_straight_view_edge_is_not_replaced_by_a_raw_image_chord() {
    let mut cal = calibrations()[1].clone();
    cal.field_roi = Some(FieldRoi {
        corners: [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]].map(|position| SourcePoint {
            camera: 0,
            position,
        }),
    });
    let boundary = cal.field_boundary().unwrap().unwrap();
    assert!(
        boundary.contains(SourcePoint {
            camera: 0,
            position: [0.5, 0.9]
        }),
        "the cylindrical raw-image edge bows below its endpoint chord"
    );
    assert!(!boundary.contains(SourcePoint {
        camera: 0,
        position: [0.2, 0.9]
    }));
}

#[test]
fn geometric_camera_ownership_does_not_inherit_render_depth_clipping() {
    let mut cal = calibrations()[0].clone();
    cal.lenses = vec![Lens::flat(3840, 2160); 2];
    cal.framing = Framing {
        axis_offset: 0.001,
        tilt: 0.0,
        roll: 0.0,
    };
    let shape = cal.projection.l_shape_mut().unwrap();
    shape.intersect = 1.0;
    shape.x_ty = 0.0;
    shape.x_rz = 0.0;
    shape.z_rx = 0.0;
    cal.field_roi = Some(FieldRoi {
        corners: [
            [0.499, 0.499],
            [0.501, 0.499],
            [0.501, 0.501],
            [0.499, 0.501],
        ]
        .map(|position| SourcePoint {
            camera: 0,
            position,
        }),
    });
    cal.validate().unwrap();
    let boundary = cal.field_boundary().unwrap().unwrap();
    let pose = RenderPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 90.0,
    };
    assert!(
        cal.pick_source_point(&SIZE, pose, [0.5, 0.5]).is_none(),
        "render near plane clips this fixture"
    );
    let points = [0, 1].map(|camera| SourcePoint {
        camera,
        position: [0.5, 0.5],
    });
    assert!(points.iter().all(|&p| boundary.contains(p)));
    assert_eq!(
        points.iter().filter(|&&p| boundary.owns(p)).count(),
        1,
        "ownership partitions raw evidence even outside the current render-depth range"
    );
}

#[test]
fn ownership_follows_blending_not_a_field_halfway_split() {
    let mut cal = calibrations()[0].clone();
    let pose = region(&mut cal);
    let original_corners = cal.field_roi.clone();
    cal.projection.l_shape_mut().unwrap().blend_width = 0.0;
    let hard = cal.field_boundary().unwrap().unwrap();
    let maps = cal.surface_maps(&SIZE, pose);
    cal.projection.l_shape_mut().unwrap().blend_width = 1.0;
    let soft = cal.field_boundary().unwrap().unwrap();
    let mut changed = 0;
    for y in 6..14 {
        for x in 21..79 {
            let ndc = [x as f64 / 50.0 - 1.0, 1.0 - y as f64 / 10.0];
            for (camera, (map, _)) in maps.iter().enumerate() {
                if let Some(uv) = map.sample_uv(ndc) {
                    let point = SourcePoint {
                        camera,
                        position: [uv.u, uv.v],
                    };
                    if hard.contains(point) && soft.contains(point) {
                        changed += usize::from(hard.owns(point) != soft.owns(point));
                    }
                }
            }
        }
    }
    assert!(
        changed > 0,
        "changing the stitch blend must move camera ownership"
    );
    assert_eq!(
        cal.field_roi, original_corners,
        "field geometry is a separate concern"
    );
}
