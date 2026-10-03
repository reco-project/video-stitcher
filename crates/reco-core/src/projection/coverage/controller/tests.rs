use super::*;

#[test]
fn mapped_camera_span_keeps_a_rolled_stereo_view_panable() {
    let mut cal = stereo();
    cal.framing.tilt = 0.308;
    cal.framing.roll = (-10.0_f64).to_radians();
    let size = ViewportSize {
        width: 1280,
        height: 720,
    };
    let mut controller = Controller::new(&cal, &size, MapConfig::default()).unwrap();
    let unrestricted = controller.roof.connected_cap(None, 1e-6).unwrap();
    assert!(controller.cap() > 40.0 && controller.cap() < 55.0);
    assert!(unrestricted > controller.cap() + 5.0);

    let left = cal.map_source_point(1, [0.5, 0.5]).unwrap();
    let right = cal.map_source_point(0, [0.5, 0.5]).unwrap();
    let request = |p: crate::geometry::YawPitch| WorldPose {
        yaw: p.yaw,
        pitch: p.pitch,
        fov_degrees: 75.0,
    };
    let mut p = controller
        .advance(&cal, &size, request(left), 0, 1.0 / 30.0)
        .unwrap();
    for i in 1..=900 {
        let target = if i <= 450 { right } else { left };
        let q = controller
            .advance(&cal, &size, request(target), i, 1.0 / 30.0)
            .unwrap();
        assert_eq!(q.fov_degrees, controller.cap());
        assert!(controller.roof.segment(p, q));
        if i % 30 == 0 {
            assert_eq!(boundary::pose(&cal, &size, q), Certificate::Covered);
        }
        if i == 450 || i == 900 {
            assert!(
                if i == 450 { q.yaw > 0.4 } else { q.yaw < -0.4 },
                "frame={i}, pose={q:?}"
            );
        }
        p = q;
    }
}

#[test]
fn steep_lens_and_roll_do_not_collapse_the_certified_cap() {
    let mut cal = stereo();
    let lens = Lens::fisheye(
        3840,
        2160,
        2026.4,
        1520.3,
        1945.1,
        1062.8,
        [0.124, -0.168, 0.13, -0.045],
    );
    cal.lenses = vec![lens.clone(), lens];
    let shape = cal.projection.l_shape_mut().unwrap();
    shape.intersect = 0.62;
    shape.x_ty = 0.0166;
    shape.x_rz = 0.0193;
    shape.z_rx = 0.0566;
    cal.framing.axis_offset = 0.2618;
    cal.framing.tilt = 0.2618;
    cal.framing.roll = (-30.0_f64).to_radians();
    let size = ViewportSize {
        width: 1280,
        height: 720,
    };
    let mut controller = Controller::new(&cal, &size, MapConfig::default()).unwrap();
    assert!(controller.cap() > 25.0, "cap={}", controller.cap());
    let request = WorldPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 75.0,
    };
    let accepted = controller
        .advance(&cal, &size, request, 0, 1.0 / 30.0)
        .unwrap();
    assert_eq!(accepted.fov_degrees, controller.cap());
    assert_eq!(boundary::pose(&cal, &size, accepted), Certificate::Covered);
}

#[test]
fn rolled_stereo_retains_useful_fov_and_bounded_cross_field_motion() {
    use crate::{
        render::planes::Nv12Planes,
        stitch::{CpuExecutor, StitchExecutor},
    };
    #[cfg(feature = "gpu")]
    let gpu = crate::stitch::test_support::gpu_or_skip();
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    for roll in [
        -40.0_f64, -35.0, -30.0, -20.0, -15.0, 0.0, 15.0, 20.0, 30.0, 35.0, 40.0,
    ] {
        let mut cal = stereo();
        cal.framing.tilt = 0.3;
        cal.framing.roll = roll.to_radians();
        let mut controller = Controller::new(&cal, &size, MapConfig::default()).unwrap();
        let render = {
            let source_sizes = vec![(64, 32); cal.projection.camera_count()];
            CpuExecutor::new(
                cal.clone(),
                ViewportSize {
                    width: 160,
                    height: 90,
                },
                source_sizes,
                false,
            )
        }
        .unwrap();
        let mut renderers: Vec<Box<dyn StitchExecutor>> = vec![Box::new(render)];
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &gpu {
            use crate::{
                render::renderer::InputFormat,
                stitch::{GpuExecutor, GpuExecutorConfig},
            };
            renderers.push(Box::new(
                GpuExecutor::new(
                    gpu.clone(),
                    GpuExecutorConfig {
                        viewport_size: ViewportSize {
                            width: 160,
                            height: 90,
                        },
                        ..{
                            let source_sizes = vec![(64, 32); cal.projection.camera_count()];
                            GpuExecutorConfig::new(cal.clone(), source_sizes, InputFormat::Nv12)
                        }
                    },
                )
                .unwrap(),
            ));
        }
        let white = [Nv12Planes {
            y: &[235; 64 * 32],
            uv: &[128; 64 * 16],
        }; 2];
        // A useful-width contract, independent of the map's own safe-FOV result.
        assert!(
            controller.cap() > 30.0,
            "roll={roll}, cap={}",
            controller.cap()
        );
        let request = |yaw| WorldPose {
            yaw,
            pitch: 0.0,
            fov_degrees: 75.0,
        };
        let dt = 1.0 / 30.0;
        let mut previous = controller
            .advance(&cal, &size, request(-0.6), 0, dt)
            .unwrap();
        let mut velocity = [0.0_f32; 2];
        for i in 1..=900 {
            let yaw = if i <= 450 { 0.6 } else { -0.6 };
            let pose = controller
                .advance(&cal, &size, request(yaw), i, dt)
                .unwrap();
            let next = [
                (pose.yaw - previous.yaw) / dt,
                (pose.pitch - previous.pitch) / dt,
            ];
            assert!(next[0].hypot(next[1]) <= 0.9001, "roll={roll}, frame={i}");
            assert!(
                (next[0] - velocity[0]).hypot(next[1] - velocity[1]) / dt <= 0.602,
                "roll={roll}, frame={i}"
            );
            assert_eq!(pose.fov_degrees, controller.cap());
            assert!(controller.roof.segment(previous, pose));
            if i % 30 == 0 {
                assert_eq!(
                    boundary::pose(&cal, &size, pose),
                    Certificate::Covered,
                    "roll={roll}, frame={i}, pose={pose:?}"
                );
            }
            if i == 450 || i == 900 {
                for render in &mut renderers {
                    let pixels = render.stitch(&white, pose).unwrap();
                    assert!(
                        pixels
                            .chunks_exact(4)
                            .all(|p| p[..3].iter().all(|&v| v >= 250)),
                        "roll={roll}, frame={i}: black pixels in {} output",
                        render.name()
                    );
                }
                let camera = cal.projection.geometry().virtual_camera(&cal.framing);
                let (expected_yaw, _) = crate::geometry::world_to_render_angles(
                    &camera,
                    yaw,
                    0.0,
                    cal.framing.tilt as f32,
                    cal.framing.roll as f32,
                );
                assert!(
                    (pose.yaw - expected_yaw).abs() < 0.1,
                    "roll={roll}, frame={i}, pose={pose:?}"
                );
            }
            previous = pose;
            velocity = next;
        }
    }
}

#[test]
fn roll_edits_do_not_accumulate_artificial_zoom() {
    use crate::{
        core::StitchCore,
        stitch::{CpuExecutor, Executor},
    };
    let mut cal = stereo();
    cal.framing.tilt = 0.3;
    let render = {
        let source_sizes = vec![(64, 32); cal.projection.camera_count()];
        CpuExecutor::new(
            cal.clone(),
            ViewportSize {
                width: 160,
                height: 90,
            },
            source_sizes,
            false,
        )
    }
    .unwrap();
    let mut core = StitchCore::new(Executor::Cpu(Box::new(render))).unwrap();
    core.prepare_coverage().unwrap();
    let rest = WorldPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 75.0,
    };
    let mut pose = core.calibration().world_to_render_pose(rest);
    pose = core.constrain_render_pose(pose, pose).unwrap();
    for roll in [0.0_f64, -5.0, -10.0, -14.0, -15.0, -20.0, -15.0, 0.0] {
        let world = core.calibration().render_to_world_pose(pose);
        cal.framing.roll = roll.to_radians();
        core.update_calibration(cal.clone()).unwrap();
        let remapped = core.calibration().world_to_render_pose(world);
        pose = core.constrain_render_pose(remapped, remapped).unwrap();
        assert!(pose.fov_degrees > 35.0, "roll={roll}, pose={pose:?}");
    }
    let reset = core.calibration().world_to_render_pose(rest);
    let reset = core.constrain_render_pose(reset, reset).unwrap();
    assert!(reset.fov_degrees > 50.0);
}

#[test]
fn source_frame_durations_preserve_motion_time_and_coverage() {
    let cal = Calibration::new(
        vec![Lens::flat(3840, 2000)],
        Cylinder {
            arc_deg: 360.0,
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
    for fps in [0.5_f32, 30.0, 120.0, 300.0, 1000.0] {
        let dt = 1.0 / fps;
        let mut c = Controller::new(
            &cal,
            &size,
            MapConfig {
                cells: [128, 64],
                ..Default::default()
            },
        )
        .unwrap();
        let initial = WorldPose {
            yaw: -0.8,
            pitch: -0.1,
            fov_degrees: 20.0,
        };
        let mut previous = c.advance(&cal, &size, initial, 0, dt).unwrap();
        let mut velocity = [0.0_f32; 2];
        for i in 1..=(8.0 * fps) as u64 {
            let target = if i as f32 * dt <= 4.0 { 0.8 } else { -0.8 };
            let p = c
                .advance(
                    &cal,
                    &size,
                    WorldPose {
                        yaw: target,
                        pitch: target * 0.125,
                        fov_degrees: if target > 0.0 { 25.0 } else { 20.0 },
                    },
                    i,
                    dt,
                )
                .unwrap_or_else(|e| panic!("fps={fps}, frame={i}, state={:?}: {e}", c.state));
            let next_velocity = [(p.yaw - previous.yaw) / dt, (p.pitch - previous.pitch) / dt];
            assert!(
                next_velocity[0].hypot(next_velocity[1]) <= 0.9001,
                "fps={fps}, frame={i}"
            );
            assert!(
                (next_velocity[0] - velocity[0]).hypot(next_velocity[1] - velocity[1]) / dt
                    <= 0.602,
                "fps={fps}, frame={i}, previous={velocity:?}, next={next_velocity:?}"
            );
            assert!(c.roof.contains(p));
            assert!((p.fov_degrees - previous.fov_degrees).abs() <= 30.0 * dt + 1e-5);
            if i == (4.0 * fps) as u64 || i == (8.0 * fps) as u64 {
                assert!(
                    (p.yaw - target).abs() < 0.02,
                    "fps={fps}, frame={i}, pose={p:?}"
                );
            }
            previous = p;
            velocity = next_velocity;
        }
    }
}

#[test]
fn manual_controller_allocates_routing_only_when_automatic_motion_starts() {
    let cal = Calibration::new(
        vec![Lens::flat(3840, 2000)],
        Cylinder {
            arc_deg: 360.0,
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
    let mut controller = Controller::new(&cal, &size, MapConfig::default()).unwrap();
    let pose = RenderPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 10.0,
    };
    assert!(controller.route.is_none());
    for yaw in [1e12, -1e12, f32::MAX, f32::MIN] {
        let accepted = controller
            .constrain(pose, RenderPose { yaw, ..pose })
            .unwrap();
        assert!(accepted.yaw.abs() <= std::f32::consts::PI);
        assert_eq!(boundary::pose(&cal, &size, accepted), Certificate::Covered);
    }
    assert!(controller.route.is_none());
    controller
        .advance(
            &cal,
            &size,
            WorldPose {
                fov_degrees: 10.0,
                ..WorldPose::default()
            },
            0,
            1.0 / 30.0,
        )
        .unwrap();
    assert!(controller.route.is_some());
    controller.reset();
    assert!(controller.route.is_some());
    assert!(controller.rebuild(&cal, &size).unwrap().route.is_none());
}

#[test]
fn default_full_cylinder_has_no_artificial_yaw_wall() {
    let cal = Calibration::new(
        vec![Lens::flat(3840, 2000)],
        Cylinder {
            arc_deg: 360.0,
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
    for sign in [-1.0, 1.0] {
        let mut c = Controller::new(&cal, &size, MapConfig::default()).unwrap();
        let mut previous = c
            .advance(
                &cal,
                &size,
                WorldPose {
                    yaw: sign * 3.0,
                    pitch: 0.0,
                    fov_degrees: 30.0,
                },
                0,
                1.0 / 30.0,
            )
            .unwrap();
        for i in 1..=900 {
            let request = WorldPose {
                yaw: sign * (3.0 + i as f32 * 0.007),
                pitch: 0.0,
                fov_degrees: 30.0,
            };
            let next = c.advance(&cal, &size, request, i, 1.0 / 30.0).unwrap();
            assert!(
                sign * (next.yaw - previous.yaw) >= -1e-5,
                "reversed at {i}: {previous:?} -> {next:?}"
            );
            assert!(
                (next.yaw - previous.yaw).hypot(next.pitch - previous.pitch) <= 0.9 / 30.0 + 1e-6
            );
            assert_eq!(boundary::pose(&cal, &size, next), Certificate::Covered);
            previous = next;
        }
        assert!(sign * previous.yaw > 9.0, "{previous:?}");
    }
}

#[test]
fn unsupported_previous_pose_keeps_periodic_yaw_near_the_request() {
    let cal = Calibration::new(
        vec![Lens::flat(3840, 2000)],
        Cylinder {
            arc_deg: 360.0,
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
    let mut controller = Controller::new(&cal, &size, MapConfig::default()).unwrap();
    for sign in [-1.0_f32, 1.0] {
        let requested = RenderPose {
            yaw: sign * (std::f32::consts::TAU + 0.7),
            pitch: 0.0,
            fov_degrees: 10.0,
        };
        let previous = RenderPose {
            pitch: std::f32::consts::FRAC_PI_2,
            ..requested
        };
        assert!(!controller.roof.contains(previous));
        let accepted = controller.constrain(previous, requested).unwrap();
        assert!((accepted.yaw - requested.yaw).abs() < 0.03, "{accepted:?}");
        assert_eq!(boundary::pose(&cal, &size, accepted), Certificate::Covered);
    }
}

#[test]
fn manual_full_cylinder_crosses_the_seam_without_release_drift() {
    let mut cal = Calibration::new(
        vec![Lens::flat(3840, 2000)],
        Cylinder {
            arc_deg: 360.0,
            ..Default::default()
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.12,
            roll: -0.09,
        },
    );
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    for sign in [-1.0, 1.0] {
        cal.framing.roll *= -1.0;
        let mut controller = Controller::new(&cal, &size, MapConfig::default()).unwrap();
        let start = RenderPose {
            yaw: sign * 3.0,
            pitch: 0.0,
            fov_degrees: 20.0,
        };
        let mut p = controller.constrain(start, start).unwrap();
        for _ in 0..900 {
            let next = controller
                .constrain(
                    p,
                    RenderPose {
                        yaw: p.yaw + sign * 0.02,
                        ..p
                    },
                )
                .unwrap();
            assert!(sign * (next.yaw - p.yaw) > 0.01, "{p:?} -> {next:?}");
            assert_eq!(boundary::pose(&cal, &size, next), Certificate::Covered);
            assert_eq!(controller.constrain(next, next).unwrap(), next);
            p = next;
        }
        assert!(sign * p.yaw > 20.0);
    }
}

#[test]
fn automatic_chart_retains_pan_travel_for_both_projections() {
    for sweep in [0.0, 180.0, 360.0] {
        for (tilt, roll) in [(0.0, 0.0), (0.12, -0.09)] {
            let mut cal = if sweep == 0.0 {
                stereo()
            } else {
                Calibration::new(
                    vec![Lens::flat(3840, 2000)],
                    Cylinder {
                        radius: 2400.0,
                        arc_deg: sweep,
                        ..Default::default()
                    },
                    Framing {
                        axis_offset: 0.0,
                        tilt: 0.0,
                        roll: 0.0,
                    },
                )
            };
            cal.framing.tilt = tilt;
            cal.framing.roll = roll;
            let size = ViewportSize {
                width: 640,
                height: 360,
            };
            let started = std::time::Instant::now();
            let mut controller = Controller::new(&cal, &size, MapConfig::default()).unwrap();
            eprintln!(
                "sweep={sweep} tilt={tilt} cap={} build={:?}",
                controller.cap(),
                started.elapsed()
            );
            assert!(controller.cap() > 20.0);
            let requested = WorldPose {
                yaw: -0.7,
                pitch: 0.0,
                fov_degrees: 120.0,
            };
            let mut p = controller
                .advance(&cal, &size, requested, 0, 1.0 / 30.0)
                .unwrap();
            for i in 1..=900 {
                let yaw = if i < 450 { 0.7 } else { -0.7 };
                let q = controller
                    .advance(&cal, &size, WorldPose { yaw, ..requested }, i, 1.0 / 30.0)
                    .unwrap();
                assert!(controller.roof.segment(p, q));
                assert!((q.yaw - p.yaw).hypot(q.pitch - p.pitch) <= 0.9 / 30.0 + 1e-6);
                assert_eq!(boundary::pose(&cal, &size, q), Certificate::Covered);
                if i == 449 {
                    assert!(q.yaw > 0.35, "sweep={sweep} tilt={tilt}: {q:?}");
                }
                p = q;
            }
            assert!(p.yaw < -0.35, "sweep={sweep} tilt={tilt}: {p:?}");
        }
    }
}

#[test]
fn manual_stereo_drag_and_zoom_are_supported_and_stationary_on_release() {
    let mut cal = stereo();
    cal.framing.tilt = 0.12;
    cal.framing.roll = -0.09;
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    let mut controller = Controller::new(
        &cal,
        &size,
        MapConfig {
            yaw: [-1.4, 1.4],
            pitch: [-0.8, 0.5],
            cells: [112, 52],
            fov: [10.0, 120.0],
            required_yaw: Some([-0.8, 0.8]),
            tolerance: 0.05,
        },
    )
    .unwrap();
    let initial = RenderPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: controller.cap(),
    };
    let mut p = controller.constrain(initial, initial).unwrap();
    for i in 0..900 {
        let angle = i as f32 * 0.023;
        let requested = RenderPose {
            yaw: p.yaw + angle.cos() * 0.04,
            pitch: p.pitch + angle.sin() * 0.02,
            fov_degrees: if i % 180 < 90 { controller.cap() } else { 25.0 },
        };
        let q = controller.constrain(p, requested).unwrap();
        assert!(controller.roof.segment(p, q), "{i}: {p:?} -> {q:?}");
        assert_eq!(boundary::pose(&cal, &size, q), Certificate::Covered);
        assert_eq!(controller.constrain(q, q).unwrap(), q);
        p = q;
    }
}
