//! CPU presentation, cache ownership and invalidation regressions.

use super::*;
use crate::calibration::{Calibration, Framing, Lens};
use crate::projection::Cylinder;
use crate::render::viewport::ViewportSize;
use crate::stitch::{CpuExecutor, Executor};

fn config() -> MapConfig {
    MapConfig::default()
}

// Liveness guard for unoptimized CI, not a coverage construction performance budget.
fn wait_for_coverage(core: &mut StitchCore) -> Result<f32, StitchCoreError> {
    let started = std::time::Instant::now();
    loop {
        match core.prepare_coverage_preview()? {
            Some(cap) => return Ok(cap),
            None => {
                assert!(
                    started.elapsed() < std::time::Duration::from_secs(120),
                    "coverage worker did not complete for {:?}",
                    core.output_dims()
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }
}

#[test]
fn render_pose_round_trip_preserves_world_direction_and_fov() {
    let mut core = core();
    for (tilt, roll) in [(0.0, 0.0), (0.2, -0.1), (-0.3, 0.25)] {
        core.edit_calibration(|cal| {
            cal.framing.tilt = tilt;
            cal.framing.roll = roll;
        })
        .unwrap();
        for yaw in [-1.0, 0.0, 0.7] {
            for pitch in [-0.4, 0.0, 0.5] {
                let world = WorldPose {
                    yaw,
                    pitch,
                    fov_degrees: 37.0,
                };
                let result = core
                    .calibration()
                    .render_to_world_pose(core.calibration().world_to_render_pose(world));
                assert!((result.yaw - yaw).abs() < 1e-5);
                assert!((result.pitch - pitch).abs() < 1e-5);
                assert_eq!(result.fov_degrees, world.fov_degrees);
            }
        }
    }
}

fn core() -> StitchCore {
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
    let executor = CpuExecutor::new(
        cal,
        ViewportSize {
            width: 160,
            height: 90,
        },
        64,
        32,
        false,
    )
    .unwrap();
    StitchCore::new(Executor::Cpu(Box::new(executor))).unwrap()
}

fn assert_border_supported(core: &StitchCore, render: RenderPose) {
    let size = core.executor.viewport_size();
    let maps = core.calibration().surface_maps(size, render);
    for i in 0..=128 {
        let t = -1.0 + i as f64 / 64.0;
        for point in [[t, -1.0], [t, 1.0], [-1.0, t], [1.0, t]] {
            assert!(
                maps.iter().any(|(map, _)| map.sample_uv(point).is_some()),
                "{render:?}: {point:?}"
            );
        }
    }
}

#[test]
fn manual_input_is_immediate_and_does_not_advance_tracking() {
    let mut core = core();
    core.prepare_coverage().unwrap();
    let start = RenderPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 30.0,
    };
    let world_start = core.calibration().render_to_world_pose(start);
    let tracking = core.advance_pose(world_start, 17, 1.0 / 30.0).unwrap();
    let requested = RenderPose { yaw: 0.3, ..start };
    assert_eq!(
        core.constrain_render_pose(start, requested).unwrap(),
        requested
    );
    assert_eq!(core.current_pose(), tracking);
    assert!(core.advance_pose(world_start, 18, 1.0 / 30.0).is_ok());
    core.set_constrained_look(false);
    let outside = RenderPose { yaw: 2.0, ..start };
    assert_eq!(
        core.constrain_render_pose(requested, outside).unwrap(),
        outside
    );
    assert!(
        core.constrain_render_pose(
            outside,
            RenderPose {
                yaw: f32::NAN,
                ..start
            }
        )
        .is_err()
    );
}

#[test]
fn manual_drag_release_zoom_and_resize_keep_render_space_and_support() {
    let mut core = core();
    core.edit_calibration(|cal| {
        cal.framing.tilt = 0.12;
        cal.framing.roll = -0.09;
    })
    .unwrap();
    let cap = core.prepare_coverage().unwrap();
    let start = RenderPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: cap,
    };
    let mut p = core.constrain_render_pose(start, start).unwrap();
    for i in 0..900 {
        if i == 450 {
            assert_eq!(core.resize(90, 160), Some((90, 160)));
        }
        let angle = i as f32 * 0.019;
        let request = RenderPose {
            yaw: p.yaw + 0.025 * angle.cos(),
            pitch: p.pitch + 0.012 * angle.sin(),
            fov_degrees: if i % 60 < 30 {
                p.fov_degrees
            } else {
                p.fov_degrees + 0.3
            },
        };
        p = core.constrain_render_pose(p, request).unwrap();
        assert_border_supported(&core, p);
        let released = core.constrain_render_pose(p, p).unwrap();
        assert_eq!(released, p, "release drift at {i}");
    }
}

#[test]
fn fixed_rate_presentation_is_idempotent_and_keeps_requests_separate() {
    let mut core = core();
    let cap = core.prepare_coverage().unwrap();
    let requested = WorldPose {
        yaw: -0.7,
        pitch: 0.0,
        fov_degrees: 100.0,
    };
    core.requested_world_pose = requested;
    let mut previous = core.advance_pose(requested, 0, 1.0 / 30.0).unwrap();
    for index in 1..=900 {
        let target = WorldPose {
            yaw: if index < 450 { 0.7 } else { -0.7 },
            ..requested
        };
        let pose = core.advance_pose(target, index, 1.0 / 30.0).unwrap();
        assert_eq!(pose.fov_degrees, cap);
        assert!((pose.yaw - previous.yaw).hypot(pose.pitch - previous.pitch) <= 0.9 / 30.0 + 1e-6);
        assert_eq!(core.advance_pose(target, index, 1.0 / 30.0).unwrap(), pose);
        assert_eq!(core.current_pose(), pose);
        assert_eq!(core.requested_world_pose, requested);
        previous = pose;
    }
    assert!(previous.yaw < -0.35, "{previous:?}");
    assert_border_supported(&core, previous);
    assert!(core.advance_pose(requested, 902, 1.0 / 30.0).is_err());
    assert_eq!(core.current_pose(), previous);
    assert!(
        core.advance_pose(
            WorldPose {
                yaw: f32::NAN,
                ..requested
            },
            901,
            1.0 / 30.0
        )
        .is_err()
    );
    assert!(core.advance_pose(requested, 901, 1.0 / 30.0).is_ok());
}

#[test]
fn reconfiguration_and_geometry_edits_are_transactional() {
    let mut core = core();
    core.prepare_coverage().unwrap();
    let request = WorldPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 50.0,
    };
    let first = core.advance_pose(request, 0, 1.0 / 30.0).unwrap();
    assert!(core.advance_pose(request, 1, f32::NAN).is_err());
    assert_eq!(core.current_pose(), first);
    core.prepare_coverage().unwrap();
    assert_eq!(core.resize(0, 100), None);
    assert_eq!(core.current_pose(), first);
    assert_eq!(core.resize(90, 160), Some((90, 160)));
    let resized = core.advance_pose(request, 10, 1.0 / 30.0).unwrap();
    assert!(resized.fov_degrees <= core.coverage_max_fov_degrees().unwrap());
    core.edit_calibration(|cal| cal.framing.tilt = 0.03)
        .unwrap();
    assert!(core.advance_pose(request, 20, 1.0 / 30.0).is_ok());
    let before = core.calibration().clone();
    let pose = core.current_pose();
    assert!(
        core.edit_calibration(|cal| cal.framing.tilt = f64::NAN)
            .is_err()
    );
    assert_eq!(core.calibration(), &before);
    assert_eq!(core.current_pose(), pose);
}

#[test]
fn toggling_constraints_resets_the_clock_without_losing_configuration() {
    let mut core = core();
    let cap = core.prepare_coverage().unwrap();
    let request = WorldPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 110.0,
    };
    core.advance_pose(request, 0, 1.0 / 30.0).unwrap();
    core.set_constrained_look(false);
    assert_eq!(
        core.advance_pose(request, 50, 1.0 / 30.0).unwrap(),
        core.calibration().world_to_render_pose(request)
    );
    assert!(core.toggle_constrained_look());
    assert_eq!(
        core.advance_pose(request, 100, 1.0 / 30.0)
            .unwrap()
            .fov_degrees,
        cap
    );
}

#[test]
fn plain_presentation_prepares_coverage_without_configuration() {
    let mut core = core();
    let request = WorldPose {
        yaw: 1.0,
        pitch: 0.3,
        fov_degrees: 100.0,
    };
    let actual = core.advance_pose(request, 0, 1.0 / 30.0).unwrap();
    assert_border_supported(&core, actual);
    assert!(actual.fov_degrees <= core.prepare_coverage().unwrap());
}

#[test]
fn push_submissions_share_the_presentation_boundary() {
    use crate::source::{FrameSet, Nv12Data, Yuv420pData};
    for nv12 in [false, true] {
        let mut core = core();
        let cap = core.prepare_coverage().unwrap();
        let frames = if nv12 {
            FrameSet::Nv12(vec![Nv12Data {
                y: vec![128; 64 * 32],
                uv: vec![128; 64 * 16],
            }])
        } else {
            FrameSet::Yuv420p(vec![Yuv420pData {
                y: vec![128; 64 * 32],
                u: vec![128; 32 * 16],
                v: vec![128; 32 * 16],
            }])
        };
        for _ in 0..3 {
            core.submit_frame(&frames, 1.0 / 30.0).unwrap();
            assert!(core.current_pose().fov_degrees <= cap);
        }
        assert_eq!(core.frame_count, 3);
        assert!(core.coverage_motion.as_ref().unwrap().current().is_some());
    }
}

#[test]
fn executor_resize_cannot_reuse_a_stale_motion_map() {
    let mut core = core();
    core.prepare_coverage().unwrap();
    let request = WorldPose {
        yaw: 0.4,
        pitch: 0.0,
        fov_degrees: 100.0,
    };
    core.advance_pose(request, 0, 1.0 / 30.0).unwrap();
    core.executor.resize(90, 160).unwrap();
    let actual = core.advance_pose(request, 1, 1.0 / 30.0).unwrap();
    let mut rebuilt =
        Controller::new(core.calibration(), core.executor.viewport_size(), config()).unwrap();
    assert_eq!(
        actual,
        rebuilt
            .advance(
                core.calibration(),
                core.executor.viewport_size(),
                request,
                1,
                1.0 / 30.0,
            )
            .unwrap()
    );
    assert_eq!(core.resize(160, 90), Some((160, 90)));
    assert_eq!(core.executor.viewport_size().width, 160);
    assert_eq!(core.executor.viewport_size().height, 90);
}

#[test]
fn same_aspect_resize_preserves_the_map_and_motion() {
    let mut core = core();
    let request = WorldPose {
        yaw: -0.5,
        pitch: 0.0,
        fov_degrees: 50.0,
    };
    let initial = core.advance_pose(request, 7, 1.0 / 30.0).unwrap();
    let cap = core.coverage_max_fov_degrees();
    core.resize_preview(320, 180).unwrap();
    assert_eq!(core.current_pose(), initial);
    assert_eq!(core.coverage_max_fov_degrees(), cap);
    assert!(core.pending_coverage.is_none());
    let next = core
        .advance_pose(
            WorldPose {
                yaw: 0.5,
                ..request
            },
            8,
            1.0 / 30.0,
        )
        .unwrap();
    assert!((next.yaw - initial.yaw).hypot(next.pitch - initial.pitch) <= 0.9 / 30.0 + 1e-6);
    assert_border_supported(&core, next);
}

#[test]
fn preview_resize_rejects_superseded_dimensions_and_calibration() {
    let mut core = core();
    core.prepare_coverage().unwrap();
    core.resize_preview(90, 160).unwrap();
    assert_eq!(core.output_dims(), (90, 160));
    assert!(core.prepare_coverage_preview().unwrap().is_none());
    core.edit_calibration(|cal| cal.framing.roll = 0.08)
        .unwrap();
    core.resize_preview(200, 200).unwrap();
    wait_for_coverage(&mut core).unwrap();
    assert_eq!(core.output_dims(), (200, 200));
    let mut expected = self::core();
    expected
        .update_calibration(core.calibration().clone())
        .unwrap();
    expected.resize(200, 200).unwrap();
    assert_eq!(
        core.coverage_max_fov_degrees().unwrap(),
        expected.prepare_coverage().unwrap()
    );
    assert!(core.resize_preview(0, 100).is_err());
    assert_eq!(core.output_dims(), (200, 200));
}

#[test]
fn invalid_pose_or_timing_never_changes_presentation() {
    let mut core = core();
    let request = WorldPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 40.0,
    };
    let previous = core.advance_pose(request, 0, 1.0 / 30.0).unwrap();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for bad in [
            WorldPose {
                yaw: value,
                ..request
            },
            WorldPose {
                pitch: value,
                ..request
            },
            WorldPose {
                fov_degrees: value,
                ..request
            },
        ] {
            let bad_render = RenderPose {
                yaw: bad.yaw,
                pitch: bad.pitch,
                fov_degrees: bad.fov_degrees,
            };
            assert!(core.constrain_render_pose(previous, bad_render).is_err());
            assert!(core.advance_pose(bad, 1, 1.0 / 30.0).is_err());
            assert_eq!(core.current_pose(), previous);
        }
        assert!(core.advance_pose(request, 1, value).is_err());
        assert_eq!(core.current_pose(), previous);
    }
}

#[test]
fn invalid_requests_are_rejected_before_building_even_without_constraints() {
    for constrained in [true, false] {
        let mut core = core();
        core.set_constrained_look(constrained);
        for duration in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            assert!(
                core.advance_pose(WorldPose::default(), 0, duration)
                    .is_err()
            );
        }
        for fov_degrees in [0.0, 180.0, f32::INFINITY] {
            assert!(
                core.advance_pose(
                    WorldPose {
                        fov_degrees,
                        ..WorldPose::default()
                    },
                    0,
                    1.0 / 30.0
                )
                .is_err()
            );
        }
        assert!(core.coverage_motion.is_none());
    }
}

#[test]
fn renderable_fov_endpoints_are_accepted() {
    let mut core = core();
    core.set_constrained_look(false);
    for (index, fov_degrees) in [1.0, 179.0].into_iter().enumerate() {
        let result = core
            .advance_pose(
                WorldPose {
                    fov_degrees,
                    ..WorldPose::default()
                },
                index as u64,
                1.0 / 30.0,
            )
            .unwrap();
        assert_eq!(result.fov_degrees, fov_degrees);
    }
}

#[test]
fn first_preview_preparation_publishes_only_a_complete_map() {
    let mut core = core();
    assert!(core.coverage_motion.is_none());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let cap = loop {
        if let Some(cap) = core.prepare_coverage_preview().unwrap() {
            break cap;
        }
        assert!(core.coverage_motion.is_none());
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    assert_eq!(core.prepare_coverage_preview().unwrap(), Some(cap));
    assert_eq!(core.prepare_coverage().unwrap(), cap);
}

#[test]
fn failed_preview_resize_is_cached_and_a_valid_request_can_follow() {
    let mut core = core();
    core.prepare_coverage().unwrap();
    core.resize_preview(100_000, 1).unwrap();
    let error = wait_for_coverage(&mut core).unwrap_err().to_string();
    assert!(
        error.contains("no connected coverage"),
        "unexpected preparation failure: {error}"
    );
    assert_eq!(core.output_dims(), (100_000, 1));
    assert!(core.pending_coverage.is_some());
    assert_eq!(
        core.prepare_coverage_preview().unwrap_err().to_string(),
        error
    );
    core.resize_preview(200, 200).unwrap();
    wait_for_coverage(&mut core).unwrap();
    assert_eq!(core.output_dims(), (200, 200));
}

#[test]
fn combined_preview_edits_accept_input_until_current_coverage_is_ready() {
    let mut core = core();
    core.prepare_coverage().unwrap();
    let outside = RenderPose {
        yaw: 2.5,
        pitch: 0.7,
        fov_degrees: 120.0,
    };
    for i in 0..24 {
        let mut cal = core.calibration().clone();
        cal.framing.roll = f64::from(i) * 0.005;
        core.update_calibration_preview(cal.clone()).unwrap();
        core.resize_preview(160 + i, 100 + i % 3).unwrap();
        assert_eq!(core.calibration(), &cal);
        assert_eq!(core.output_dims(), (160 + i, 100 + i % 3));
        assert_eq!(
            core.constrain_preview_pose(outside, outside).unwrap(),
            outside
        );
        assert_eq!(core.coverage_max_fov_degrees(), None);
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let fitted = loop {
        let pose = core.constrain_preview_pose(outside, outside).unwrap();
        if core.coverage_max_fov_degrees().is_some() {
            break pose;
        }
        assert_eq!(pose, outside);
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(2));
    };
    assert_ne!(fitted, outside);
    assert_border_supported(&core, fitted);
    let mut expected = self::core();
    expected
        .update_calibration(core.calibration().clone())
        .unwrap();
    let (width, height) = core.output_dims();
    expected.resize(width, height).unwrap();
    assert_eq!(
        core.coverage_max_fov_degrees(),
        Some(expected.prepare_coverage().unwrap())
    );
    assert_eq!(core.constrain_preview_pose(fitted, fitted).unwrap(), fitted);
}

#[test]
fn strict_resolution_after_preview_edit_never_returns_uncovered_input() {
    let mut core = core();
    core.resize_preview(200, 200).unwrap();
    let outside = RenderPose {
        yaw: 2.0,
        pitch: 0.6,
        fov_degrees: 100.0,
    };
    assert_eq!(
        core.constrain_preview_pose(outside, outside).unwrap(),
        outside
    );
    let fitted = core.constrain_render_pose(outside, outside).unwrap();
    assert_border_supported(&core, fitted);
    assert!(core.pending_coverage.is_none());
    let mut cal = core.calibration().clone();
    cal.framing.roll = 0.1;
    core.update_calibration_preview(cal).unwrap();
    assert!(core.coverage_max_fov_degrees().is_none());
    let request = WorldPose {
        yaw: 2.0,
        pitch: 0.6,
        fov_degrees: 100.0,
    };
    let tracked = core.advance_pose(request, 0, 1.0 / 30.0).unwrap();
    assert_border_supported(&core, tracked);
}
