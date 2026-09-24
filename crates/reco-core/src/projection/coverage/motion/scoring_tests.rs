//! Numeric ranking and cap-level cross-camera motion regressions.

use super::*;
fn map(cal: &Calibration, size: &ViewportSize) -> (roof::Roof, f32) {
    let roof = roof::Roof::build(
        cal,
        size,
        roof::Grid {
            yaw: [-1.8, 1.8],
            pitch: [-0.9, 0.6],
            columns: 120,
            rows: 60,
        },
        [10.0, 120.0],
        0.05,
    )
    .unwrap();
    let cap = roof
        .connected_cap(Some([-0.8, 0.8]), CONTACT_TOLERANCE)
        .unwrap();
    assert!(cap > 10.0 && cap < 120.0);
    (roof, cap)
}

fn lens_target(cal: &Calibration, camera: usize, x: f32) -> Vector3<f32> {
    let projection = cal.projection.geometry();
    let pose = cal.map_source_point(camera, [x, 0.4]).unwrap();
    projection
        .virtual_camera(&cal.framing)
        .yaw_pitch_to_direction(pose.yaw, pose.pitch)
}

#[test]
fn behind_camera_ranking_resolves_sub_microradian_progress() {
    let cal = stereo();
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    let previous = RenderPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 40.0,
    };
    for sign in [-1.0, 1.0] {
        let target = direction(
            &cal,
            RenderPose {
                yaw: sign * 2.0,
                ..previous
            },
        );
        for step in [1e-7, 1e-6, 1e-5] {
            let closer = RenderPose {
                yaw: sign * step,
                ..previous
            };
            assert!(screen(&cal, &size, closer, &target).is_none());
            assert!(
                better(
                    score(&cal, &size, closer, &target),
                    score(&cal, &size, previous, &target)
                ),
                "step {step}, sign {sign}"
            );
        }
    }
}

#[test]
fn forward_class_wins_without_a_numeric_sentinel() {
    let cal = stereo();
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    let pose = RenderPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 40.0,
    };
    let target = direction(&cal, pose);
    for sign in [-1.0, 1.0] {
        let behind = RenderPose {
            yaw: sign * (std::f32::consts::FRAC_PI_2 + 1e-5),
            ..pose
        };
        let forward = RenderPose {
            yaw: sign * (std::f32::consts::FRAC_PI_2 - 2e-6),
            ..pose
        };
        assert!(screen(&cal, &size, behind, &target).is_none());
        assert!(screen(&cal, &size, forward, &target).is_some());
        assert!(better(
            score(&cal, &size, forward, &target),
            score(&cal, &size, behind, &target)
        ));
    }
}

#[test]
fn behind_camera_ranking_resolves_progress_from_the_antipode() {
    let cal = stereo();
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    let previous = RenderPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 40.0,
    };
    let target = -direction(&cal, previous);
    for sign in [-1.0, 1.0] {
        for step in [1e-7, 1e-6, 1e-5] {
            let closer = RenderPose {
                yaw: sign * step,
                ..previous
            };
            assert!(
                better(
                    score(&cal, &size, closer, &target),
                    score(&cal, &size, previous, &target)
                ),
                "step {step}, sign {sign}"
            );
        }
    }
}

#[test]
fn forward_scores_preserve_visibility_then_centering() {
    let cal = stereo();
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    let pose = RenderPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 40.0,
    };
    for yaw in [-1.2, -0.7, -0.1, 0.0, 0.1, 0.7, 1.2] {
        let target = direction(&cal, RenderPose { yaw, ..pose });
        let [x, y] = screen(&cal, &size, pose, &target).unwrap();
        let outside = (x.abs() - 0.96).max(0.0).powi(2) + (y.abs() - 0.96).max(0.0).powi(2);
        assert_eq!(
            score(&cal, &size, pose, &target),
            [
                0.0,
                if outside > 1e-10 { outside } else { 0.0 },
                x * x + y * y
            ]
        );
    }
}

fn follow(
    cal: &Calibration,
    size: &ViewportSize,
    roof: &roof::Roof,
    state: &mut (RenderPose, [f32; 2]),
    target: &Vector3<f32>,
) {
    let fov = state.0.fov_degrees;
    let mut tail_travel = 0.0;
    let mut route = Route::new(cal, roof);
    for frame in 0..900 {
        let (previous, velocity) = *state;
        let window = MotionWindow {
            previous,
            velocity,
            dt: 1.0 / 30.0,
        };
        let next = solve_motion(cal, size, window, target, fov, roof, &mut route).unwrap();
        assert_eq!(next.pose.fov_degrees, fov);
        assert!(roof.contains(next.pose));
        assert!(roof.segment(previous, next.pose));
        assert!(distance(previous, next.pose) <= RATE / 30.0 + 1e-6);
        assert!(next.velocity[0].hypot(next.velocity[1]) <= RATE + 1e-5);
        assert!(
            (next.velocity[0] - velocity[0]).hypot(next.velocity[1] - velocity[1]) * 30.0
                <= ACCELERATION + 0.001
        );
        let mut stopping = braking::path(&window, next.pose).unwrap();
        stopping.insert(0, previous);
        assert!(stopping.windows(2).all(|p| roof.segment(p[0], p[1])));
        if frame >= 750 {
            assert!(
                screen(cal, size, next.pose, target)
                    .is_some_and(|xy| xy.iter().all(|x| x.abs() <= 1.0)),
                "target lost after convergence: {next:?}",
                next = next.pose
            );
            tail_travel += distance(previous, next.pose);
        }
        *state = (next.pose, next.velocity);
    }
    assert!(
        tail_travel < 0.001,
        "unsettled static target: {tail_travel}"
    );
}

#[test]
fn cap_level_targets_converge_from_both_lenses_with_tiny_pose_changes() {
    let cal = stereo();
    let size = ViewportSize {
        width: 1920,
        height: 1080,
    };
    let (roof, cap) = map(&cal, &size);
    let perturbations = [
        (0.0, 0.0),
        (1e-7, 0.0),
        (-1e-7, 0.0),
        (0.0, 1e-7),
        (0.0, -1e-7),
        (1e-6, 0.0),
        (-1e-6, 0.0),
        (0.0, 1e-6),
        (0.0, -1e-6),
    ];
    for start_camera in 0..2 {
        let seed = initial(
            &cal,
            &size,
            &lens_target(&cal, start_camera, 0.5),
            cap,
            &roof,
            &Route::new(&cal, &roof),
        )
        .unwrap();
        for (dyaw, dpitch) in perturbations {
            let seed = RenderPose {
                yaw: seed.yaw + dyaw,
                pitch: seed.pitch + dpitch,
                ..seed
            };
            assert!(roof.contains(seed));
            for target_camera in 0..2 {
                for x in [0.3, 0.5, 0.7] {
                    let mut state = (seed, [0.0; 2]);
                    follow(
                        &cal,
                        &size,
                        &roof,
                        &mut state,
                        &lens_target(&cal, target_camera, x),
                    );
                }
            }
        }
    }
}

#[test]
fn repeated_cross_camera_steps_at_cap_never_need_zoom_or_reset() {
    let cal = stereo();
    let size = ViewportSize {
        width: 1920,
        height: 1080,
    };
    let (roof, cap) = map(&cal, &size);
    for start_camera in 0..2 {
        let seed = initial(
            &cal,
            &size,
            &lens_target(&cal, start_camera, 0.5),
            cap,
            &roof,
            &Route::new(&cal, &roof),
        )
        .unwrap();
        let mut state = (seed, [0.0; 2]);
        for camera in [
            start_camera,
            1 - start_camera,
            start_camera,
            1 - start_camera,
        ] {
            follow(
                &cal,
                &size,
                &roof,
                &mut state,
                &lens_target(&cal, camera, 0.5),
            );
        }
    }
}

#[test]
fn cylinder_targets_cross_both_directions_at_cap() {
    let cal = Calibration::new(
        vec![Lens::flat(3840, 2000)],
        Cylinder {
            radius: 1000.0,
            arc_deg: 180.0,
            ..Default::default()
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    let roof = roof::Roof::build(
        &cal,
        &size,
        roof::Grid {
            yaw: [-1.0, 1.0],
            pitch: [-0.3, 0.3],
            columns: 80,
            rows: 24,
        },
        [10.0, 120.0],
        0.05,
    )
    .unwrap();
    let cap = roof
        .connected_cap(Some([-0.8, 0.8]), CONTACT_TOLERANCE)
        .unwrap();
    let target = |yaw| {
        direction(
            &cal,
            RenderPose {
                yaw,
                pitch: 0.0,
                fov_degrees: cap,
            },
        )
    };
    for start_yaw in [-0.8, 0.8] {
        let seed = initial(
            &cal,
            &size,
            &target(start_yaw),
            cap,
            &roof,
            &Route::new(&cal, &roof),
        )
        .unwrap();
        let mut state = (seed, [0.0; 2]);
        for yaw in [-start_yaw, start_yaw, -start_yaw] {
            follow(&cal, &size, &roof, &mut state, &target(yaw));
        }
    }
}
