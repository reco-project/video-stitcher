//! Coordinate, objective, and motion-window regressions.

use super::*;
#[test]
fn target_projection_and_frustum_use_the_render_camera() {
    let mut cal = stereo();
    cal.framing.tilt = 0.4;
    cal.framing.roll = 0.3;
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    for yaw in [-1.2, 0.0, 1.2] {
        for pitch in [-0.4, 0.0, 0.4] {
            let p = Pose {
                yaw,
                pitch,
                fov_degrees: 45.0,
            };
            let d = direction(&cal, p);
            let xy = screen(&cal, &size, p, &d).unwrap();
            assert!(xy.iter().all(|v| v.abs() < 1e-5));
            assert!(distance(aim(&cal, &d, 45.0), p) < 1e-5);
            for [yaw, pitch] in footprint(&cal, &size, p) {
                let d = direction(&cal, Pose { yaw, pitch, ..p });
                let xy = screen(&cal, &size, p, &d).unwrap();
                assert!((xy[0].abs().max(xy[1].abs()) - 1.0).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn motion_window_respects_both_disks_and_predicts_braking() {
    let mut maximum_rounding_error = 0.0_f32;
    for dt in [1.0 / 24.0, 1.0 / 30.0, 1.0 / 60.0] {
        for i in 0..64 {
            let angle = i as f32 * 0.37;
            let speed = RATE * (i % 9) as f32 / 8.0;
            let previous = Pose {
                yaw: 0.3,
                pitch: -0.2,
                fov_degrees: 45.0,
            };
            let window = MotionWindow {
                previous,
                velocity: [speed * angle.cos(), speed * angle.sin()],
                dt,
            };
            for j in 0..16 {
                let target = Pose {
                    yaw: (j as f32 * 0.8).cos(),
                    pitch: (j as f32 * 0.8).sin(),
                    ..previous
                };
                let q = window.project(target);
                let v = window.velocity_at(q);
                assert!(v[0].hypot(v[1]) <= RATE + 1e-5);
                assert!(
                    (v[0] - window.velocity[0]).hypot(v[1] - window.velocity[1])
                        <= ACCELERATION * dt + 1e-5
                );
                let stop = window.stopping_pose(q);
                let mut rest = q;
                let mut velocity = v;
                for _ in 0..100 {
                    let w = MotionWindow {
                        previous: rest,
                        velocity,
                        dt,
                    };
                    let next = w.braking_step();
                    velocity = w.velocity_at(next);
                    rest = next;
                }
                maximum_rounding_error = maximum_rounding_error.max(distance(stop, rest));
            }
        }
    }
    // Reconstructing velocity from f32 positions accumulates rounding over a
    // stop. Bound this discrepancy independently of coverage clearance.
    assert!(maximum_rounding_error < 1e-4, "{maximum_rounding_error}");
}

#[test]
fn behind_camera_target_has_a_descent_direction() {
    let cal = stereo();
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    let p = Pose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 40.0,
    };
    let target = direction(&cal, Pose { yaw: 2.0, ..p });
    assert!(screen(&cal, &size, p, &target).is_none());
    let roof = roof::Roof::build(
        &cal,
        &size,
        roof::Grid {
            yaw: [-1.4, 1.4],
            pitch: [-0.7, 0.5],
            columns: 64,
            rows: 40,
        },
        [10.0, 120.0],
        0.05,
    )
    .unwrap();
    let next = solve_motion(
        &cal,
        &size,
        MotionWindow {
            previous: p,
            velocity: [0.0; 2],
            dt: 1.0 / 30.0,
        },
        &target,
        p.fov_degrees,
        &roof,
        &mut Route::new(&cal, &roof),
    )
    .unwrap()
    .pose;
    assert!(better(
        score(&cal, &size, next, &target),
        score(&cal, &size, p, &target)
    ));
}
