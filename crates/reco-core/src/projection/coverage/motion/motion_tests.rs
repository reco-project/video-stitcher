//! Shared-motion regressions for contact-limited zoom and periodic projections.

use super::*;

fn cylinder(sweep: f64, height: u32) -> Calibration {
    Calibration::new(
        vec![Lens::flat(3840, height)],
        Cylinder {
            focal_length: 1000.0,
            sweep_deg: sweep,
            ..Default::default()
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    )
}

fn check_step(
    cal: &Calibration,
    size: &ViewportSize,
    previous: Pose,
    velocity: [f32; 2],
    next: &Advance,
    fov: f32,
) {
    assert!((next.pose.fov_degrees - fov).abs() <= (previous.fov_degrees - fov).abs() + 1e-5);
    assert!((next.pose.fov_degrees - previous.fov_degrees).abs() <= ZOOM_RATE / 30.0 + 1e-5);
    assert!(distance(previous, next.pose) <= RATE / 30.0 + 1e-6);
    assert!(
        (next.velocity[0] - velocity[0]).hypot(next.velocity[1] - velocity[1]) * 30.0
            <= ACCELERATION + 0.001
    );
    for t in [0.0, 0.5, 1.0] {
        let maps = cal.surface_maps(size, interpolate(previous, next.pose, t));
        assert_eq!(exhaustive(&maps, size), 0);
    }
}

#[test]
fn roof_contact_keeps_tilted_cylinder_motion_bounded_and_covered() {
    let mut cal = cylinder(180.0, 1000);
    cal.framing.tilt = 0.2;
    cal.framing.roll = -0.15;
    let size = ViewportSize {
        width: 97,
        height: 55,
    };
    let roof = roof::Roof::build(
        &cal,
        &size,
        roof::Grid {
            yaw: [-1.2, 1.2],
            pitch: [-0.7, 0.7],
            columns: 64,
            rows: 40,
        },
        [10.0, 80.0],
        0.05,
    )
    .unwrap();
    let mut pose = Pose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 20.0,
    };
    let mut velocity = [0.0; 2];
    let mut route = Route::new(&cal, &roof);
    assert!(roof.contains(pose));
    for frame in 0..1200 {
        let t = frame as f32 / 30.0;
        let fov = if frame < 900 {
            20.0 + 5.0 * (t * 0.5).sin()
        } else {
            25.0
        };
        let target = direction(
            &cal,
            Pose {
                yaw: if frame < 900 {
                    1.5 * (t * 0.4).sin()
                } else {
                    1.5
                },
                pitch: if frame < 900 {
                    0.5 * (t * 0.7).cos()
                } else {
                    0.5
                },
                fov_degrees: fov,
            },
        );
        let next = solve_motion(
            &cal,
            &size,
            MotionWindow {
                previous: pose,
                velocity,
                dt: 1.0 / 30.0,
            },
            &target,
            fov,
            &roof,
            &mut route,
        )
        .unwrap_or_else(|| panic!("frame {frame}, {pose:?}"));
        check_step(&cal, &size, pose, velocity, &next, fov);
        assert!(roof.segment(pose, next.pose));
        if frame >= 1140 {
            assert!(
                distance(pose, next.pose) < 1e-6,
                "unsettled corner at {frame}"
            );
        }
        pose = next.pose;
        velocity = next.velocity;
    }
}

#[test]
fn map_motion_crosses_a_safe_corridor_without_requiring_extra_fov() {
    let cal = cylinder(180.0, 1000);
    let size = ViewportSize {
        width: 97,
        height: 55,
    };
    let roof = roof::Roof::from_cells(
        roof::Grid {
            yaw: [-0.5, 0.5],
            pitch: [-0.1, 0.1],
            columns: 8,
            rows: 1,
        },
        10.0,
        &[25.1, 25.1, 25.1, 25.1, 25.1, 25.1, 25.1, 25.4],
    );
    let mut pose = Pose {
        yaw: 0.5,
        pitch: 0.0,
        fov_degrees: 25.0,
    };
    let target_pose = Pose { yaw: -0.4, ..pose };
    let target = direction(&cal, target_pose);
    let mut velocity = [0.0; 2];
    let mut route = Route::new(&cal, &roof);
    assert!(roof.segment(pose, target_pose));
    for _ in 0..180 {
        let next = solve_motion(
            &cal,
            &size,
            MotionWindow {
                previous: pose,
                velocity,
                dt: 1.0 / 30.0,
            },
            &target,
            25.0,
            &roof,
            &mut route,
        )
        .unwrap();
        check_step(&cal, &size, pose, velocity, &next, 25.0);
        assert!(roof.segment(pose, next.pose));
        pose = next.pose;
        velocity = next.velocity;
    }
    assert!(distance(pose, target_pose) < 1e-4, "{pose:?}");
}

#[test]
fn zoom_pushes_inward_at_contact_without_anticipating() {
    for sweep in [180.0, 360.0] {
        let cal = cylinder(sweep, 2000);
        let size = ViewportSize {
            width: 97,
            height: 55,
        };
        let roof = test_roof(&cal, &size);
        let mut pose = Pose {
            yaw: 0.0,
            pitch: 0.2,
            fov_degrees: 60.0,
        };
        let target = direction(&cal, pose);
        let mut velocity = [0.0; 2];
        let mut route = Route::new(&cal, &roof);
        let fovs: Vec<_> = (0..150)
            .map(|frame| 60.0 + 20.0 * (frame as f32 / 60.0).min(1.0))
            .collect();
        for (frame, &fov) in fovs.iter().take(120).enumerate() {
            let next = solve_motion(
                &cal,
                &size,
                MotionWindow {
                    previous: pose,
                    velocity,
                    dt: 1.0 / 30.0,
                },
                &target,
                fov,
                &roof,
                &mut route,
            )
            .unwrap_or_else(|| panic!("sweep {sweep}, frame {frame}, {pose:?}, {velocity:?}"));
            if frame == 0 {
                assert!(
                    distance(next.pose, pose) < 1e-6,
                    "must not move before zoom changes"
                );
            }
            check_step(&cal, &size, pose, velocity, &next, fov);
            pose = next.pose;
            velocity = next.velocity;
        }
        assert!(
            pose.fov_degrees > 79.98,
            "zoom must resume as room becomes available: {pose:?}"
        );
        assert!(pose.pitch < 0.1, "the widening view must push inward");
    }
}

#[test]
fn oversized_zoom_caps_then_reverses_without_reset() {
    let cal = cylinder(180.0, 2000);
    let size = ViewportSize {
        width: 97,
        height: 55,
    };
    let roof = test_roof(&cal, &size);
    let mut pose = Pose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 60.0,
    };
    let target = direction(&cal, pose);
    let mut velocity = [0.0; 2];
    let mut route = Route::new(&cal, &roof);
    for frame in 0..150 {
        let requested = if frame < 90 { 160.0 } else { 45.0 };
        let next = solve_motion(
            &cal,
            &size,
            MotionWindow {
                previous: pose,
                velocity,
                dt: 1.0 / 30.0,
            },
            &target,
            requested,
            &roof,
            &mut route,
        )
        .expect("zoom may cap but must not interrupt motion");
        check_step(&cal, &size, pose, velocity, &next, requested);
        if (60..90).contains(&frame) {
            assert!(next.pose.fov_degrees > 80.0 && next.pose.fov_degrees < 90.0);
            assert!(
                distance(pose, next.pose) < 1e-5,
                "centred saturation must not oscillate"
            );
        }
        pose = next.pose;
        velocity = next.velocity;
    }
    assert_eq!(pose.fov_degrees, 45.0);
}

#[test]
fn full_cylinder_crosses_the_yaw_seam_in_both_directions_with_zoom() {
    let mut cal = cylinder(360.0, 4000);
    cal.framing.tilt = 0.25;
    cal.framing.roll = -0.2;
    for sign in [-1.0, 1.0] {
        for (width, height) in [(97, 55), (55, 97), (65, 65)] {
            let size = ViewportSize { width, height };
            let fovs: Vec<_> = (0..240)
                .map(|i| 30.0 + 5.0 * (i as f32 * 0.03).sin())
                .collect();
            let roof = test_roof(&cal, &size);
            let mut pose = Pose {
                yaw: sign * 3.0,
                pitch: 0.0,
                fov_degrees: fovs[0],
            };
            let mut velocity = [0.0; 2];
            let mut route = Route::new(&cal, &roof);
            for (frame, &fov) in fovs.iter().take(180).enumerate() {
                let target = direction(
                    &cal,
                    Pose {
                        yaw: sign * (3.0 + frame as f32 * 0.007),
                        ..pose
                    },
                );
                let next = solve_motion(
                    &cal,
                    &size,
                    MotionWindow {
                        previous: pose,
                        velocity,
                        dt: 1.0 / 30.0,
                    },
                    &target,
                    fov,
                    &roof,
                    &mut route,
                )
                .unwrap_or_else(|| panic!("seam: sign {sign}, frame {frame}, {pose:?}"));
                check_step(&cal, &size, pose, velocity, &next, fov);
                assert!(
                    sign * (next.pose.yaw - pose.yaw) >= -1e-5,
                    "must not reverse at the angular seam"
                );
                pose = next.pose;
                velocity = next.velocity;
            }
            assert!(sign * pose.yaw > 4.0);
        }
    }
}

#[test]
fn motion_path_checks_intermediate_extrema() {
    let cal = cylinder(180.0, 2000);
    let size = ViewportSize {
        width: 97,
        height: 55,
    };
    let pose = Pose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 25.0,
    };
    for interior in [
        Pose {
            fov_degrees: 160.0,
            ..pose
        },
        Pose { yaw: 3.0, ..pose },
        Pose { pitch: 1.4, ..pose },
    ] {
        assert_eq!(
            sweep::certify_path(&cal, &size, &[pose, interior, pose], 2048, 2048),
            Certificate::Uncovered
        );
    }
}

#[test]
fn malformed_motion_requests_fail_closed() {
    let cal = cylinder(180.0, 2000);
    let size = ViewportSize {
        width: 97,
        height: 55,
    };
    let roof = test_roof(&cal, &size);
    let mut route = Route::new(&cal, &roof);
    let pose = Pose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 25.0,
    };
    let target = direction(&cal, pose);
    for dt in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            solve_motion(
                &cal,
                &size,
                MotionWindow {
                    previous: pose,
                    velocity: [0.0; 2],
                    dt
                },
                &target,
                25.0,
                &roof,
                &mut route
            )
            .is_none()
        );
    }
    for fov in [f32::NAN, 180.0] {
        assert!(
            solve_motion(
                &cal,
                &size,
                MotionWindow {
                    previous: pose,
                    velocity: [0.0; 2],
                    dt: 1.0 / 30.0
                },
                &target,
                fov,
                &roof,
                &mut route
            )
            .is_none()
        );
    }
    assert!(
        braking::path(
            &MotionWindow {
                previous: pose,
                velocity: [0.5, 0.0],
                dt: 1e-30
            },
            Pose { yaw: 1.0, ..pose }
        )
        .is_none()
    );
}

fn test_roof(cal: &Calibration, size: &ViewportSize) -> roof::Roof {
    roof::Roof::build(
        cal,
        size,
        roof::Grid {
            yaw: [-4.6, 4.6],
            pitch: [-0.7, 0.7],
            columns: 368,
            rows: 56,
        },
        [10.0, 170.0],
        0.05,
    )
    .unwrap()
}

#[test]
fn initialization_uses_the_map_without_an_extra_reserve() {
    let cal = cylinder(180.0, 1000);
    let size = ViewportSize {
        width: 97,
        height: 55,
    };
    let grid = roof::Grid {
        yaw: [-0.4, 0.4],
        pitch: [-0.1, 0.1],
        columns: 4,
        rows: 2,
    };
    let roof = roof::Roof::from_cells(grid, 10.0, &[25.001; 8]);
    let route = Route::new(&cal, &roof);
    let inside = Pose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 25.0,
    };
    assert_eq!(
        initial(&cal, &size, &direction(&cal, inside), 25.0, &roof, &route),
        Some(inside)
    );
    let target = direction(&cal, Pose { yaw: 1.0, ..inside });
    let seed = initial(&cal, &size, &target, 25.0, &roof, &route).unwrap();
    assert!(roof.contains(seed));
    assert_eq!(seed.fov_degrees, 25.0);
    assert!(seed.yaw > 0.3);
    assert!(initial(&cal, &size, &target, 26.0, &roof, &route).is_none());
    let empty = roof::Roof::from_cells(grid, 10.0, &[0.0; 8]);
    assert!(
        initial(
            &cal,
            &size,
            &target,
            25.0,
            &empty,
            &Route::new(&cal, &empty)
        )
        .is_none()
    );
    assert!(initial(&cal, &size, &Vector3::zeros(), 25.0, &roof, &route).is_none());
    assert!(initial(&cal, &size, &target, f32::NAN, &roof, &route).is_none());
    let mut invalid = cal.clone();
    invalid.lenses[0].fx = 0.0;
    assert!(initial(&invalid, &size, &target, 25.0, &roof, &route).is_none());
}
