//! Headless checks for calibration editor acceptance and rollback.

use super::*;
use reco_core::calibration::{Framing, Lens};
use reco_core::projection::LShape;
use slint::Model;
use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};

#[test]
fn preview_rest_and_reset_keep_the_world_direction_on_tilted_rigs() {
    use reco_core::core::StitchCore;
    use reco_core::render::viewport::ViewportSize;
    use reco_core::stitch::{CpuExecutor, Executor};

    for (tilt, roll) in [(0.0, 0.0), (0.2, 0.0), (0.12, -0.09)] {
        let mut cal = calibration();
        cal.framing.tilt = tilt;
        cal.framing.roll = roll;
        let executor = {
            let source_sizes = vec![(64, 36); cal.projection.camera_count()];
            CpuExecutor::new(
                cal,
                ViewportSize {
                    width: 64,
                    height: 36,
                },
                source_sizes,
                false,
            )
        }
        .unwrap();
        let mut core = StitchCore::new(Executor::Cpu(Box::new(executor))).unwrap();
        core.set_constrained_look(false);
        let expected = core
            .calibration()
            .world_to_render_pose(reco_core::geometry::WorldPose::default());
        let mut pose = preview_pose_control(Some(&core));
        assert_eq!(pose.current_pose(), expected);
        pose.apply_drag(100.0, -30.0, 360.0);
        pose.constrain(&mut core).unwrap();
        assert_ne!(pose.current_pose(), expected);
        pose.apply_intent(ControlIntent::Reset);
        pose.constrain(&mut core).unwrap();
        assert_eq!(pose.current_pose(), expected);
    }
}

#[test]
fn drag_scale_uses_the_contained_image_not_the_letterbox() {
    assert_eq!(displayed_image_height(600.0, 900.0, (1920, 1080)), 337.5);
    assert_eq!(displayed_image_height(900.0, 600.0, (1080, 1920)), 600.0);
    assert_eq!(displayed_image_height(640.0, 360.0, (1920, 1080)), 360.0);
    let mut fitted = preview_pose_control(None);
    let mut letterboxed = fitted.clone();
    fitted.apply_drag(40.0, 20.0, 337.5);
    letterboxed.apply_drag(
        40.0,
        20.0,
        displayed_image_height(600.0, 900.0, (1920, 1080)),
    );
    assert_eq!(fitted.target_pose(), letterboxed.target_pose());
}

struct HeadlessPlatform;

impl slint::platform::Platform for HeadlessPlatform {
    fn create_window_adapter(
        &self,
    ) -> Result<Rc<dyn slint::platform::WindowAdapter>, slint::PlatformError> {
        Ok(MinimalSoftwareWindow::new(RepaintBufferType::ReusedBuffer))
    }
}

fn calibration() -> Calibration {
    Calibration::new(
        vec![Lens::fisheye(64, 36, 32.0, 32.0, 32.0, 18.0, [0.0; 4]); 2],
        LShape {
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        },
        Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        },
    )
}

#[test]
fn inputs_follow_calibration_without_duplicating_mono_frames() {
    let mut state = AppState::new(crate::settings::GuiSettings::default());
    let mono = Calibration::new(
        vec![Lens::flat(64, 36)],
        reco_core::projection::Cylinder::default(),
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    assert!(state.selected_inputs(&mono).is_none());
    let left = reco_io::InputPath::Chained(vec![
        PathBuf::from("first.mp4"),
        PathBuf::from("second.mp4"),
    ]);
    state.left_input = Some(left);
    let inputs = state.selected_inputs(&mono).unwrap();
    assert_eq!(inputs.len(), 1);
    assert!(matches!(&inputs[0], reco_io::InputPath::Chained(paths) if paths.len() == 2));
    assert!(state.selected_inputs(&calibration()).is_none());
    state.right_input = Some(reco_io::InputPath::Single(PathBuf::from("right.mp4")));
    assert_eq!(state.selected_inputs(&calibration()).unwrap().len(), 2);
    assert_eq!(state.selected_inputs(&mono).unwrap().len(), 1);
}

fn exercise_editor(state: &mut AppState, app: &RecoApp) {
    let accepted = serde_json::to_value(state.calibration.as_ref().unwrap()).unwrap();
    app.set_cal_camera_axis_offset(0.0);
    assert!(!state.edit_calibration(app, |cal| cal.framing.axis_offset = 0.0));
    assert_eq!(app.get_cal_camera_axis_offset(), 0.25);
    assert!(!app.get_cal_dirty());
    assert_eq!(
        serde_json::to_value(state.calibration.as_ref().unwrap()).unwrap(),
        accepted
    );
    if let Some(bridge) = &state.bridge {
        assert_eq!(
            serde_json::to_value(bridge.engine().calibration()).unwrap(),
            accepted
        );
    }

    state.lens_preview_active = true;
    assert!(state.edit_calibration(app, |cal| {
        for lens in &mut cal.lenses {
            lens.correction = 0.0;
        }
    }));
    assert_eq!(state.lens_correction_amount, 0.0);
    let untouched_right =
        serde_json::to_value(&state.calibration.as_ref().unwrap().lenses[1]).unwrap();
    let profile = Lens::fisheye(128, 72, 80.0, 82.0, 66.0, 38.0, [-0.01, 0.0, 0.0, 0.0]);
    assert!(state.apply_lens_profile(app, &profile, "left"));
    state.sync_calibration_controls(app);
    let cal = state.calibration.as_ref().unwrap();
    assert_eq!((cal.lenses[0].width, cal.lenses[0].height), (64, 36));
    assert_eq!((cal.lenses[0].fx, cal.lenses[0].fy), (40.0, 41.0));
    assert_eq!(cal.lenses[0].correction, 0.0);
    assert_eq!(
        serde_json::to_value(&cal.lenses[1]).unwrap(),
        untouched_right
    );
    assert_eq!(app.get_lens_left_fx(), 40.0);
    assert_eq!(app.get_lens_right_fx(), 32.0);
    if let Some(bridge) = &state.bridge {
        assert_eq!(
            serde_json::to_value(bridge.engine().calibration()).unwrap(),
            serde_json::to_value(cal).unwrap()
        );
    }

    let accepted = state.calibration.clone().unwrap();
    let mut invalid = profile.clone();
    invalid.height = 0;
    assert!(!state.apply_lens_profile(app, &invalid, "both"));
    assert_eq!(state.calibration.as_ref().unwrap(), &accepted);
    if let Some(bridge) = &state.bridge {
        assert_eq!(bridge.engine().calibration(), &accepted);
    }
    assert!(state.apply_lens_profile(app, &profile, "both"));
    let accepted = state.calibration.as_ref().unwrap();
    assert_eq!(accepted.lenses[0], accepted.lenses[1]);
    assert_eq!(accepted.lenses[1].correction, 0.0);
}

fn exercise_pending_framing(state: &mut AppState, app: &RecoApp) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while state
        .bridge
        .as_mut()
        .unwrap()
        .engine_mut()
        .prepare_coverage_preview()
        .unwrap()
        .is_none()
    {
        assert!(Instant::now() < deadline, "coverage preparation timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
    state.use_constrained_look = false;
    state.lens_preview_active = false;
    state.reset_view();
    state.clamp_targets();
    assert!(state.edit_calibration(app, |cal| cal.framing.tilt = 0.12));
    assert_eq!(state.calibration.as_ref().unwrap().framing.tilt, 0.12);
    assert_eq!(
        state
            .bridge
            .as_ref()
            .unwrap()
            .engine()
            .coverage_max_fov_degrees(),
        None
    );
    state.use_constrained_look = true;
    // A recording sink must not change interactive camera acceptance or editing.
    let (tx, _rx) = std::sync::mpsc::sync_channel(4);
    state.recording_tx = Some(tx);
    assert!(state.is_recording());
    assert!(state.clamp_targets());
    assert!(state.coverage_pending);
    assert!(
        state
            .bridge
            .as_ref()
            .unwrap()
            .engine()
            .coverage_max_fov_degrees()
            .is_none()
    );
    state.use_constrained_look = false;
    assert!(state.edit_calibration(app, |cal| cal.framing.roll = -0.09));
    assert_eq!(state.calibration.as_ref().unwrap().framing.tilt, 0.12);
    assert!(!state.edit_calibration(app, |cal| cal.framing.axis_offset = 0.0));
    assert_eq!(state.calibration.as_ref().unwrap().framing.roll, -0.09);
    state.apply_pan(30.0, -10.0, 640.0, 360.0);
    let world = state
        .bridge
        .as_ref()
        .unwrap()
        .engine()
        .calibration()
        .render_to_world_pose(state.pose.current_pose());
    let cal = state.calibration.as_ref().unwrap();
    assert_eq!(cal.framing.tilt, 0.12);
    assert_eq!(cal.framing.roll, -0.09);
    assert_eq!(state.bridge.as_ref().unwrap().engine().calibration(), cal);
    let accepted = state
        .bridge
        .as_ref()
        .unwrap()
        .engine()
        .calibration()
        .render_to_world_pose(state.pose.current_pose());
    assert!((world.yaw - accepted.yaw).abs() < 1e-5);
    assert!((world.pitch - accepted.pitch).abs() < 1e-5);
    assert_eq!(world.fov_degrees, accepted.fov_degrees);
    state.reset_view();
    let reset = state
        .bridge
        .as_ref()
        .unwrap()
        .engine()
        .calibration()
        .render_to_world_pose(state.pose.current_pose());
    assert!(reset.yaw.abs() < 1e-5 && reset.pitch.abs() < 1e-5);

    assert!(state.edit_calibration(app, |cal| {
        for lens in &mut cal.lenses {
            lens.cx = 1e9;
        }
    }));
    let edited = state.calibration.clone().unwrap();
    assert_eq!(edited.lenses[0].cx, 1e9);
    assert_eq!(
        state.bridge.as_ref().unwrap().engine().calibration(),
        &edited
    );
    assert_eq!(app.get_lens_left_cx(), 1e9_f32);
    assert_eq!(app.get_status_text(), "Calibration updated");
    state.use_constrained_look = true;
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        assert!(
            state.clamp_targets(),
            "coverage failure must not freeze editing"
        );
        if state.coverage_error_reported {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(state.calibration.as_ref(), Some(&edited));
    assert!(!state.coverage_pending);
    assert!(state.is_recording());
    let before = state.pose.current_pose();
    state.apply_pan(20.0, 10.0, 640.0, 360.0);
    assert_ne!(before, state.pose.current_pose());
    state.recording_tx = None;
}

#[test]
fn editor_rolls_back_invalid_controls_and_keeps_preview_and_document_together() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    let mut state = AppState::new(crate::settings::GuiSettings::default());
    state.calibration = Some(Calibration::new(
        vec![Lens::flat(64, 36)],
        reco_core::projection::Cylinder::default(),
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    ));
    state.sync_calibration_controls(&app);
    assert!(!app.get_is_l_shape());
    assert!(state.edit_calibration(&app, |cal| cal.framing.roll = 0.1));
    assert_eq!(state.calibration.as_ref().unwrap().lenses.len(), 1);
    state.calibration = Some(calibration());
    state.sync_calibration_controls(&app);
    assert!(app.get_is_l_shape());
    exercise_editor(&mut state, &app);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: reco_core::gpu::GpuContext::select_backends(),
        ..Default::default()
    });
    let adapter =
        match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        {
            Ok(adapter) => adapter,
            Err(error) => {
                assert!(
                    std::env::var("RECO_REQUIRE_GPU").is_err(),
                    "GPU required: {error}"
                );
                eprintln!("Skipping GPU editor checks: {error}");
                return;
            }
        };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap();
    state.calibration = Some(calibration());
    state.bridge = Some(
        PreviewBridge::new(
            device,
            queue,
            adapter.get_info(),
            calibration(),
            vec![(64, 36); 2],
            64,
            36,
        )
        .unwrap(),
    );
    exercise_editor(&mut state, &app);
    if state.bridge.is_some() {
        exercise_pending_framing(&mut state, &app);
    }

    // Once the segment list has changed, a failed reopen must not leave the
    // old source presented as though it matched the new order.
    state.left_input = Some(reco_io::InputPath::Single(PathBuf::new()));
    state.right_input = state.left_input.clone();
    app.set_files_loaded(true);
    app.set_playing(true);
    state.reopen_source(&app);
    assert!(state.bridge.is_none());
    assert_eq!(state.playback.state(), PlayState::Empty);
    assert!(!app.get_files_loaded());
    assert!(!app.get_playing());
    assert!(
        app.get_status_text()
            .contains("Could not reopen reordered inputs")
    );
}

fn pending_calibration(
    state: &mut AppState,
) -> (std::sync::mpsc::Sender<CalibrationResult>, Arc<AtomicBool>) {
    let (sender, receiver) = std::sync::mpsc::channel();
    let interrupted = Arc::new(AtomicBool::new(false));
    state.calibration_job = Some(calibration_job::PendingCalibration::new(
        state,
        receiver,
        Arc::clone(&interrupted),
    ));
    (sender, interrupted)
}

fn decoded_gpu_state() -> AppState {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: reco_core::gpu::GpuContext::select_backends(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let mut state = AppState::new(Default::default());
    state.shared_gpu = Some(SharedGpu {
        device,
        queue,
        adapter_info: adapter.get_info(),
    });
    let input =
        reco_io::InputPath::Single(std::env::var("RECO_PLAYBACK_TEST_VIDEO").unwrap().into());
    state.left_input = Some(input.clone());
    state.right_input = Some(input);
    state.use_constrained_look = false;
    state
}

#[test]
fn reviewable_calibration_does_not_replace_or_save_the_current_document() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    let mut state = AppState::new(Default::default());
    let accepted = calibration();
    state.calibration = Some(accepted.clone());
    state.calibration_path = Some("accepted.json".into());
    let mut candidate = accepted.clone();
    candidate.framing.roll = 0.2;
    handle_calibration_result(
        Ok(CalibrationOutput {
            calibration: candidate,
            assessment: reco_calibrate::CalibrationAssessment::NeedsReview {
                reason: "insufficient independent evidence".into(),
            },
            total_matches: 50,
            left_lens_profile: None,
            right_lens_profile: None,
        }),
        &mut state,
        &app.as_weak(),
        false,
    );
    assert_eq!(state.calibration.as_ref(), Some(&accepted));
    assert_eq!(
        state.calibration_path.as_deref(),
        Some(std::path::Path::new("accepted.json"))
    );
    assert!(state.bridge.is_none());
    assert!(app.get_calibration_needs_review());
    let (_, snapshot) = state.review_calibration.as_ref().unwrap();
    assert!(snapshot.matches(&state));
    state.calibration.as_mut().unwrap().framing.roll = 0.1;
    assert!(!state.review_calibration.as_ref().unwrap().1.matches(&state));
}

#[test]
fn clearing_sources_discards_pending_calibration_results() {
    let mut state = AppState::new(Default::default());
    let (sender, interrupted) = pending_calibration(&mut state);
    state.unload_pipeline();
    assert!(interrupted.load(Ordering::Relaxed));
    assert!(
        state.calibration_job.is_some(),
        "retain the worker until it stops"
    );
    assert!(!state.poll_calibration(&slint::Weak::default()));
    sender
        .send(Err(reco_calibrate::video::CalibrateVideosError::NoFrames))
        .unwrap();
    assert!(state.poll_calibration(&slint::Weak::default()));
    assert!(state.calibration_job.is_none());
    assert!(state.calibration.is_none());
}

#[test]
fn calibration_result_requires_the_same_inputs_and_document() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    for change in 0..4 {
        let mut state = AppState::new(Default::default());
        state.calibration = Some(calibration());
        state.calibration_path = Some("first.json".into());
        state.left_input = Some(reco_io::InputPath::Chained(vec![
            "first.mp4".into(),
            "second.mp4".into(),
        ]));
        state.right_input = Some(reco_io::InputPath::Single("right.mp4".into()));
        let (sender, _) = pending_calibration(&mut state);
        match change {
            0 => {
                state.left_input = Some(reco_io::InputPath::Chained(vec![
                    "first.mp4".into(),
                    "new.mp4".into(),
                ]))
            }
            1 => state.right_input = None,
            2 => state.calibration_path = Some("second.json".into()),
            _ => state.calibration.as_mut().unwrap().framing.roll = 0.2,
        }
        let current = state.calibration.clone();
        sender
            .send(Ok(CalibrationOutput {
                calibration: calibration(),
                assessment: reco_calibrate::CalibrationAssessment::Confirmed,
                total_matches: 50,
                left_lens_profile: None,
                right_lens_profile: None,
            }))
            .unwrap();
        app.set_calibrating(true);
        assert!(state.poll_calibration(&app.as_weak()));
        assert!(!app.get_calibrating());
        assert_eq!(state.calibration, current);
        assert!(
            state.bridge.is_none(),
            "stale results must not initialize a renderer"
        );
        assert_eq!(app.get_toasts().row_count(), 0);
    }
}

#[test]
fn recalibration_failure_and_worker_disconnect_preserve_the_current_document() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    for disconnect in [false, true] {
        let mut state = AppState::new(Default::default());
        state.calibration = Some(calibration());
        let (sender, _) = pending_calibration(&mut state);
        if !disconnect {
            sender
                .send(Err(reco_calibrate::video::CalibrateVideosError::NoFrames))
                .unwrap();
        }
        drop(sender);
        app.set_files_loaded(true);
        app.set_calibrating(true);
        assert!(state.poll_calibration(&app.as_weak()));
        assert!(state.calibration_job.is_none());
        assert!(!app.get_calibrating());
        assert!(app.get_files_loaded());
        assert_eq!(state.calibration, Some(calibration()));
        assert!(app.get_toasts().row_count() > 0);
    }
}

#[test]
fn accepted_edits_and_shutdown_cancel_calibration() {
    let mut state = AppState::new(Default::default());
    let (_, interrupted) = pending_calibration(&mut state);
    let mut invalid = calibration();
    invalid.lenses[0].fx = 0.0;
    assert!(state.apply_calibration(invalid).is_err());
    assert!(!interrupted.load(Ordering::Relaxed));
    state.apply_calibration(calibration()).unwrap();
    assert!(interrupted.load(Ordering::Relaxed));
    assert!(state.calibration_job.is_some());
    drop(state);
    let mut state = AppState::new(Default::default());
    let (_, interrupted) = pending_calibration(&mut state);
    drop(state);
    assert!(interrupted.load(Ordering::Relaxed));
}

#[test]
fn explicit_calibration_cancellation_keeps_the_current_preview() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    let mut state = AppState::new(Default::default());
    state.calibration = Some(calibration());
    let (sender, _) = pending_calibration(&mut state);
    app.set_files_loaded(true);
    app.set_calibrating(true);
    state.cancel_calibration();
    assert!(!state.poll_calibration(&app.as_weak()));
    assert!(app.get_calibrating());
    assert_eq!(app.get_calibration_step(), "Stopping…");
    sender
        .send(Err(reco_calibrate::video::CalibrateVideosError::Cancelled))
        .unwrap();
    assert!(state.poll_calibration(&app.as_weak()));
    assert!(!app.get_calibrating());
    assert!(app.get_files_loaded());
    assert_eq!(state.calibration, Some(calibration()));
    assert_eq!(app.get_status_text(), "Calibration cancelled");
    assert_eq!(app.get_toasts().row_count(), 0);
}

#[test]
#[ignore = "requires a GPU and RECO_PLAYBACK_TEST_VIDEO pointing to a readable video"]
fn failed_automatic_save_keeps_the_new_document_dirty_and_does_not_target_the_old_file() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    let mut state = decoded_gpu_state();
    let directory =
        std::env::temp_dir().join(format!("reco-auto-save-failure-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let previous_path = directory.join("previous.json");
    let destination = directory.join("left_calibration.json");
    std::fs::create_dir(&destination).unwrap();
    let baseline = calibration();
    baseline.to_file(&previous_path).unwrap();
    state.calibration_path = Some(previous_path.clone());
    state.calibration = Some(baseline.clone());
    state.cal_baseline = Some(baseline.clone());
    state.left_path = Some(directory.join("left.mp4"));
    let mut replacement = baseline.clone();
    replacement.framing.roll = 0.2;
    handle_calibration_result(
        Ok(CalibrationOutput {
            calibration: replacement.clone(),
            assessment: reco_calibrate::CalibrationAssessment::Confirmed,
            total_matches: 50,
            left_lens_profile: None,
            right_lens_profile: None,
        }),
        &mut state,
        &app.as_weak(),
        false,
    );
    assert_eq!(state.calibration.as_ref(), Some(&replacement));
    assert!(app.get_files_loaded());
    assert!(app.get_clip_duration_secs() > 0.0);
    assert_eq!(app.get_export_end_secs(), app.get_clip_duration_secs());
    assert!(
        app.get_cal_dirty(),
        "failed persistence must not claim a saved document"
    );
    assert_eq!(state.calibration_path.as_ref(), Some(&destination));
    assert_eq!(Calibration::from_file(&previous_path).unwrap(), baseline);
    assert!(
        app.get_toasts()
            .iter()
            .any(|toast| toast.title == "Calibration not saved")
    );
    std::fs::remove_dir(&destination).unwrap();
    state.save_calibration().unwrap();
    assert_eq!(Calibration::from_file(&destination).unwrap(), replacement);
    assert_eq!(Calibration::from_file(&previous_path).unwrap(), baseline);
    std::fs::remove_file(destination).unwrap();
    std::fs::remove_file(previous_path).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
#[ignore = "requires a GPU and RECO_PLAYBACK_TEST_VIDEO pointing to a readable video"]
fn explicit_candidate_preview_does_not_save_until_requested() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    let mut state = decoded_gpu_state();
    let directory =
        std::env::temp_dir().join(format!("reco-candidate-preview-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    state.left_path = Some(directory.join("left.mp4"));
    let destination = directory.join("left_calibration.json");
    let previous = calibration();
    previous.to_file(&destination).unwrap();
    let mut candidate = previous.clone();
    candidate.framing.roll = 0.2;
    handle_calibration_result(
        Ok(CalibrationOutput {
            calibration: candidate.clone(),
            assessment: reco_calibrate::CalibrationAssessment::NeedsReview {
                reason: "insufficient evidence".into(),
            },
            total_matches: 50,
            left_lens_profile: None,
            right_lens_profile: None,
        }),
        &mut state,
        &app.as_weak(),
        true,
    );
    assert_eq!(state.calibration.as_ref(), Some(&candidate));
    assert!(app.get_files_loaded());
    assert!(app.get_cal_dirty());
    assert_eq!(state.calibration_path.as_ref(), Some(&destination));
    assert_eq!(Calibration::from_file(&destination).unwrap(), previous);
    state.save_calibration().unwrap();
    assert_eq!(Calibration::from_file(&destination).unwrap(), candidate);
    std::fs::remove_file(destination).unwrap();
    std::fs::remove_dir(directory).unwrap();
}
