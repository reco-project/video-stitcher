//! Consumer-facing input behavior, independent of any windowing toolkit.

use reco_control::ControlIntent;
use reco_control::pose_control::{PoseControl, PoseControlConfig};
use reco_core::geometry::RenderPose;

#[test]
fn pending_inputs_accumulate_without_changing_the_presented_pose() {
    let mut control = PoseControl::default();
    let before = control.current_pose();
    control.apply_intent(ControlIntent::DeltaYawRad(0.05));
    control.apply_intent(ControlIntent::DeltaYawRad(0.05));
    control.apply_wheel(0.5);
    assert_eq!(control.current_pose(), before);
    assert_eq!(control.target_pose().yaw, 0.1);
    assert_eq!(control.target_pose().fov_degrees, before.fov_degrees - 1.5);
}

#[test]
fn direct_input_preserves_display_scale_and_axis_choices() {
    for (invert_x, invert_y) in [(false, false), (true, false), (false, true), (true, true)] {
        let config = PoseControlConfig {
            invert_drag_x: invert_x,
            invert_drag_y: invert_y,
            ..Default::default()
        };
        let mut small = PoseControl::new(config).unwrap();
        let mut large = PoseControl::new(config).unwrap();
        small.apply_drag(30.0, 20.0, 300.0);
        large.apply_drag(60.0, 40.0, 600.0);
        assert_eq!(small.target_pose(), large.target_pose());
        assert_eq!(small.target_pose().yaw > 0.0, invert_x);
        assert_eq!(small.target_pose().pitch < 0.0, invert_y);
    }
}

#[test]
fn malformed_events_do_not_poison_either_pose() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut control = PoseControl::default();
        let before = control.target_pose();
        control.apply_wheel(bad);
        assert_eq!(control.target_pose(), before);
        control.set_target_fov(bad);
        assert_eq!(control.target_pose(), before);
        control.set_target(RenderPose { yaw: bad, ..before });
        assert_eq!(control.target_pose(), before);
        control.set_target(RenderPose {
            pitch: bad,
            ..before
        });
        control.set_pose(RenderPose {
            fov_degrees: bad,
            ..before
        });
        for intent in [
            ControlIntent::DeltaYawRad(bad),
            ControlIntent::DeltaPitchRad(bad),
            ControlIntent::DeltaFovDeg(bad),
            ControlIntent::SetFovDeg(bad),
        ] {
            control.apply_intent(intent);
            assert_eq!(control.target_pose(), before);
        }
        assert_eq!(control.current_pose(), before);
    }
}

#[test]
fn overflowing_finite_events_are_rejected_atomically() {
    let mut control = PoseControl::default();
    let initial = control.current_pose();
    for event in [
        ControlIntent::DeltaFovDeg(f32::MAX),
        ControlIntent::DeltaFovDeg(-f32::MAX),
    ] {
        control.apply_intent(event);
        assert!((1.0..=150.0).contains(&control.target_pose().fov_degrees));
    }
    control.set_target(initial);
    control.apply_wheel(f32::MAX);
    assert_eq!(control.target_pose(), initial);
    control.apply_drag(f32::MAX, 1.0, f32::MIN_POSITIVE);
    assert_eq!(control.target_pose(), initial);
    control.apply_drag(1.0, f32::MAX, f32::MIN_POSITIVE);
    assert_eq!(control.target_pose(), initial);
    let extreme = RenderPose {
        yaw: f32::MAX,
        pitch: f32::MAX,
        ..initial
    };
    control.set_target(extreme);
    for event in [
        ControlIntent::DeltaYawRad(f32::MAX),
        ControlIntent::DeltaPitchRad(f32::MAX),
    ] {
        control.apply_intent(event);
        assert_eq!(control.target_pose(), extreme);
    }
    assert_eq!(control.current_pose(), initial);
}

#[test]
fn invalid_configuration_is_rejected_without_panics() {
    let default = PoseControlConfig::default();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
        for config in [
            PoseControlConfig {
                wheel_fov_step_degrees: bad,
                ..default
            },
            PoseControlConfig {
                hotkey_fov_step_degrees: bad,
                ..default
            },
            PoseControlConfig {
                fov_min_degrees: bad,
                ..default
            },
            PoseControlConfig {
                fov_max_degrees: bad,
                ..default
            },
            PoseControlConfig {
                rest_pose: RenderPose {
                    fov_degrees: bad,
                    ..default.rest_pose
                },
                ..default
            },
        ] {
            assert!(PoseControl::new(config).is_err());
        }
    }
    for (min, max) in [(0.0, 150.0), (100.0, 99.0), (1.0, 180.0), (80.0, 150.0)] {
        assert!(
            PoseControl::new(PoseControlConfig {
                fov_min_degrees: min,
                fov_max_degrees: max,
                ..default
            })
            .is_err()
        );
    }
    for bad in [f32::NAN, f32::INFINITY] {
        for rest in [
            RenderPose {
                yaw: bad,
                ..default.rest_pose
            },
            RenderPose {
                pitch: bad,
                ..default.rest_pose
            },
        ] {
            assert!(
                PoseControl::new(PoseControlConfig {
                    rest_pose: rest,
                    ..default
                })
                .is_err()
            );
        }
    }
    for fov in [1.0, 179.0] {
        assert!(
            PoseControl::new(PoseControlConfig {
                fov_min_degrees: fov,
                fov_max_degrees: fov,
                rest_pose: RenderPose {
                    fov_degrees: fov,
                    ..default.rest_pose
                },
                wheel_fov_step_degrees: 0.0,
                hotkey_fov_step_degrees: 0.0,
                ..default
            })
            .is_ok()
        );
    }
}

#[test]
fn drag_matches_perspective_scale_and_rejects_invalid_dimensions() {
    let mut control = PoseControl::default();
    let fov = control.current_fov_deg();
    control.apply_drag(100.0, 0.0, 600.0);
    let expected = -2.0 * (fov.to_radians() * 0.5).tan() * 100.0 / 600.0;
    assert!((control.target_pose().yaw - expected).abs() < 1e-5);
    let before = control.target_pose();
    for height in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        control.apply_drag(10.0, 20.0, height);
        assert_eq!(control.target_pose(), before);
    }
    control.apply_drag(f32::NAN, 0.0, 600.0);
    control.apply_drag(0.0, f32::INFINITY, 600.0);
    assert_eq!(control.target_pose(), before);
}

#[test]
fn zoom_steps_reset_and_limits_keep_distinct_semantics() {
    let mut control = PoseControl::default();
    let rest = control.current_pose();
    control.apply_intent(ControlIntent::ZoomIn);
    assert_eq!(control.target_pose().fov_degrees, rest.fov_degrees - 5.0);
    control.apply_intent(ControlIntent::ZoomOut);
    assert_eq!(control.target_pose(), rest);
    control.apply_wheel(1.0);
    assert_eq!(control.target_pose().fov_degrees, rest.fov_degrees - 3.0);
    control.apply_intent(ControlIntent::Reset);
    assert_eq!(control.target_pose(), rest);
    for (requested, accepted) in [(-50.0, 1.0), (200.0, 150.0), (12.0, 12.0)] {
        control.apply_intent(ControlIntent::SetFovDeg(requested));
        assert_eq!(control.target_pose().fov_degrees, accepted);
    }
    control.apply_wheel(1.0);
    assert_eq!(control.target_pose().fov_degrees, 9.0);
    assert_eq!(control.current_pose(), rest);
}

#[test]
fn accepted_coverage_can_be_below_requested_zoom_minimum() {
    let mut control = PoseControl::new(PoseControlConfig {
        fov_min_degrees: 30.0,
        ..Default::default()
    })
    .unwrap();
    let fitted = RenderPose {
        fov_degrees: 12.0,
        ..RenderPose::default()
    };
    control.set_pose(fitted);
    assert_eq!(control.current_pose(), fitted);
    assert_eq!(control.target_pose(), fitted);
    for invalid in [0.0, 180.0] {
        control.set_pose(RenderPose {
            fov_degrees: invalid,
            ..fitted
        });
        assert_eq!(control.current_pose(), fitted);
        assert_eq!(control.target_pose(), fitted);
    }
}

fn cpu_core(arc_deg: f64) -> reco_core::core::StitchCore {
    use reco_core::{
        calibration::{Calibration, Framing, Lens},
        projection::Cylinder,
        render::viewport::ViewportSize,
        stitch::{CpuExecutor, Executor},
    };
    let cal = Calibration::new(
        vec![Lens::flat(64, 32)],
        Cylinder {
            radius: 20.0,
            arc_deg,
            surface_height: None,
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    reco_core::core::StitchCore::new(Executor::Cpu(Box::new(
        CpuExecutor::new(
            cal,
            ViewportSize {
                width: 64,
                height: 32,
            },
            64,
            32,
            false,
        )
        .unwrap(),
    )))
    .unwrap()
}

#[test]
fn direct_acceptance_preserves_large_reversals_without_inertia() {
    let mut core = cpu_core(180.0);
    core.set_constrained_look(false);
    let mut control = PoseControl::default();
    control.set_target_fov(1.0);
    control.constrain(&mut core).unwrap();
    assert_eq!(control.current_fov_deg(), 1.0);
    for yaw in [2.9, -0.0000001, -2.9, 0.0000001] {
        let pose = RenderPose {
            yaw,
            pitch: -yaw * 0.1,
            fov_degrees: 37.12345,
        };
        control.set_target(pose);
        control.constrain(&mut core).unwrap();
        assert_eq!(control.current_pose(), pose);
        assert_eq!(control.target_pose(), pose);
    }
}

#[test]
fn coverage_acceptance_discards_blocked_input() {
    let mut core = cpu_core(180.0);
    let mut control = PoseControl::default();
    control.set_target(RenderPose {
        yaw: 2.0,
        pitch: 0.0,
        fov_degrees: 40.0,
    });
    control.constrain(&mut core).unwrap();
    let accepted = control.current_pose();
    assert!(accepted.yaw < 2.0);
    assert_eq!(accepted, control.target_pose());
    control.constrain(&mut core).unwrap();
    assert_eq!(control.current_pose(), accepted);
    control.apply_intent(ControlIntent::DeltaYawRad(-0.05));
    control.constrain(&mut core).unwrap();
    assert!(control.current_pose().yaw < accepted.yaw);
}

#[test]
fn failed_coverage_preserves_both_poses() {
    let mut core = cpu_core(0.001);
    let mut control = PoseControl::default();
    control.apply_intent(ControlIntent::DeltaYawRad(0.1));
    let before = (control.current_pose(), control.target_pose());
    assert!(control.constrain(&mut core).is_err());
    assert_eq!((control.current_pose(), control.target_pose()), before);
}

#[test]
fn commands_have_a_flat_serialized_contract() {
    let commands = [
        ControlIntent::DeltaYawRad(0.1),
        ControlIntent::DeltaPitchRad(-0.2),
        ControlIntent::DeltaFovDeg(-3.0),
        ControlIntent::SetFovDeg(60.0),
        ControlIntent::ZoomIn,
        ControlIntent::ZoomOut,
        ControlIntent::Reset,
    ];
    for command in commands {
        let value = serde_json::to_value(command).unwrap();
        assert!(value.get("action").is_some());
        assert!(value.get("kind").is_none());
        assert_eq!(
            serde_json::from_value::<ControlIntent>(value).unwrap(),
            command
        );
    }
    assert_eq!(
        serde_json::to_value(ControlIntent::DeltaYawRad(0.5)).unwrap(),
        serde_json::json!({"action":"delta_yaw_rad", "value":0.5})
    );
    assert!(
        serde_json::from_str::<ControlIntent>(r#"{"kind":"pose","data":{"action":"reset"}}"#)
            .is_err()
    );
    fn is_reset(command: ControlIntent) -> bool {
        matches!(command, ControlIntent::Reset)
    }
    assert!(is_reset(ControlIntent::Reset));
    fn send_clone<T: Send + Clone>() {}
    send_clone::<PoseControl>();
    send_clone::<ControlIntent>();
}
