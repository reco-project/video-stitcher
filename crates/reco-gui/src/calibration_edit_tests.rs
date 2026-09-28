//! Headless checks for calibration editor acceptance and rollback.

use super::*;
use reco_core::calibration::{Framing, Lens};
use reco_core::projection::LShape;
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
        let executor = CpuExecutor::new(
            cal,
            ViewportSize {
                width: 64,
                height: 36,
            },
            64,
            36,
            false,
        )
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
            64,
            36,
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
