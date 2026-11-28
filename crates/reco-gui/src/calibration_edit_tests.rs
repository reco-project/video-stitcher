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
fn rejected_document_load_preserves_accepted_state_and_pending_calibration() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    let mut state = AppState::new(Default::default());
    let accepted = calibration();
    state.calibration = Some(accepted.clone());
    state.cal_baseline = Some(accepted.clone());
    state.calibration_path = Some("accepted.json".into());
    let (_sender, interrupted) = pending_calibration(&mut state);
    app.set_calibration_path("accepted.json".into());
    app.set_files_loaded(true);
    app.set_cal_dirty(true);
    app.set_lens_dirty(true);
    app.set_current_frame(42);
    let state = Rc::new(RefCell::new(state));
    let path =
        std::env::temp_dir().join(format!("reco-invalid-document-{}.json", std::process::id()));
    for contents in ["not json".to_string(), {
        let mut invalid = accepted.clone();
        invalid.lenses[0].fx = 0.0;
        serde_json::to_string(&invalid).unwrap()
    }] {
        std::fs::write(&path, contents).unwrap();
        assert!(!load_calibration_and_update(
            &state,
            &app.as_weak(),
            path.clone()
        ));
        assert_eq!(state.borrow().calibration.as_ref(), Some(&accepted));
        assert_eq!(state.borrow().cal_baseline.as_ref(), Some(&accepted));
        assert_eq!(
            state.borrow().calibration_path.as_deref(),
            Some(std::path::Path::new("accepted.json"))
        );
        assert!(!interrupted.load(Ordering::Relaxed));
        assert!(app.get_files_loaded());
        assert!(app.get_cal_dirty());
        assert!(app.get_lens_dirty());
        assert_eq!(app.get_current_frame(), 42);
        assert_eq!(app.get_calibration_path().as_str(), "accepted.json");
        assert!(
            state
                .borrow()
                .user_settings
                .recent_calibration
                .entries()
                .is_empty()
        );
    }
    std::fs::remove_file(&path).unwrap();
    assert!(!load_calibration_and_update(&state, &app.as_weak(), path));
    assert_eq!(
        app.get_toasts().row_data(0).unwrap().title,
        "Calibration unchanged"
    );
}

#[test]
fn valid_document_loads_before_its_inputs_and_cancels_superseded_work() {
    let mut state = AppState::new(Default::default());
    state.calibration = Some(calibration());
    let (_sender, interrupted) = pending_calibration(&mut state);
    let mut replacement = calibration();
    replacement.framing.roll = 0.2;
    let path =
        std::env::temp_dir().join(format!("reco-pending-document-{}.json", std::process::id()));
    replacement.to_file(&path).unwrap();
    assert!(!state.load_calibration(path.clone()).unwrap());
    assert_eq!(state.calibration.as_ref(), Some(&replacement));
    assert_eq!(state.cal_baseline.as_ref(), Some(&replacement));
    assert_eq!(state.calibration_path.as_ref(), Some(&path));
    assert!(interrupted.load(Ordering::Relaxed));
    assert!(state.bridge.is_none());
    assert!(!state.try_init().unwrap());
    std::fs::remove_file(path).unwrap();
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

#[test]
fn region_apply_is_unsaved_even_without_a_writable_path() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    let roi = reco_core::calibration::FieldRoi {
        corners: [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]].map(|position| {
            reco_core::projection::SourcePoint {
                camera: 0,
                position,
            }
        }),
    };
    for path in [None, Some(std::env::temp_dir())] {
        let mut state = AppState::new(Default::default());
        state.calibration = Some(calibration());
        state.calibration_path = path;
        app.set_cal_dirty(false);
        assert!(state.edit_calibration(&app, |cal| cal.field_roi = Some(roi.clone())));
        assert_eq!(
            state.calibration.as_ref().unwrap().field_roi,
            Some(roi.clone())
        );
        assert!(app.get_has_roi());
        assert!(
            app.get_cal_dirty(),
            "an unsaved ROI must remain visibly dirty"
        );
        assert!(state.save_calibration().is_err());
        assert!(app.get_cal_dirty());
    }
}

#[test]
fn accepted_calibration_edits_and_resets_remain_unsaved_until_persisted() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    let edits: [fn(&mut Calibration); 4] = [
        |cal| cal.framing.roll = 0.1,
        |cal| cal.lenses[0].fx *= 1.1,
        |cal| cal.sync_offset = 1,
        |cal| cal.projection.l_shape_mut().unwrap().blend_width = 0.1,
    ];
    for edit in edits {
        let mut state = AppState::new(Default::default());
        state.calibration = Some(calibration());
        app.set_cal_dirty(false);
        assert!(state.edit_calibration(&app, edit));
        assert!(
            app.get_cal_dirty(),
            "every accepted document edit must be unsaved"
        );
        app.set_cal_dirty(false);
        let baseline = calibration();
        assert!(state.edit_calibration(&app, |cal| *cal = baseline));
        assert!(
            app.get_cal_dirty(),
            "resetting values must not pretend to save them"
        );
    }
}

#[test]
fn region_changes_persist_only_when_explicitly_saved() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    let path = std::env::temp_dir().join(format!("reco-gui-roi-{}.json", uuid::Uuid::new_v4()));
    let mut state = AppState::new(Default::default());
    state.calibration = Some(calibration());
    state.calibration_path = Some(path.clone());
    app.set_lens_dirty(true);
    let roi = reco_core::calibration::FieldRoi {
        corners: [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]].map(|position| {
            reco_core::projection::SourcePoint {
                camera: 0,
                position,
            }
        }),
    };
    assert!(state.edit_calibration(&app, |cal| cal.field_roi = Some(roi.clone())));
    assert!(app.get_cal_dirty());
    assert!(app.get_lens_dirty());
    assert!(!path.exists());
    state.save_calibration().unwrap();
    assert_eq!(
        Calibration::from_file(&path).unwrap(),
        state.calibration.clone().unwrap()
    );
    assert!(state.edit_calibration(&app, |cal| cal.field_roi = None));
    assert!(app.get_cal_dirty());
    state.save_calibration().unwrap();
    assert!(!app.get_has_roi());
    assert_eq!(Calibration::from_file(&path).unwrap().field_roi, None);
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "requires RECO_PLAYBACK_TEST_VIDEO pointing to a video with at least 30 frames"]
fn relative_seek_uses_the_displayed_frame_without_fractional_rounding() {
    let mut state = AppState::new(Default::default());
    let input =
        reco_io::InputPath::Single(std::env::var("RECO_PLAYBACK_TEST_VIDEO").unwrap().into());
    state
        .playback
        .open(&[input], 0, &state.decoder_cleanup)
        .unwrap();
    state.playback.seek_frame(17).unwrap();
    state.playback.advance_blocking().unwrap();
    let fps = state.playback.fps() as f32;
    for (seconds, expected) in [(0.0, 18), (1.0 / fps, 19), (-1.0 / fps, 18), (-100.0, 1)] {
        state.seek_relative(seconds).unwrap();
        if state.playback.is_seeking() {
            state.playback.advance_blocking().unwrap();
        }
        assert_eq!(state.playback.frame_index(), expected);
    }
    state.seek_relative(100.0).unwrap();
    state.playback.advance_blocking().unwrap();
    assert_eq!(
        state.playback.frame_index(),
        state.playback.total_frames().unwrap()
    );
    state.seek_relative(-100.0).unwrap();
    state.seek_relative(1.0 / fps).unwrap();
    state.seek_relative(1.0 / fps).unwrap();
    assert_eq!(
        state.playback.requested_frame(),
        2,
        "rapid relative input accumulates on the pending target"
    );
    state.playback.advance_blocking().unwrap();
    assert_eq!(state.playback.frame_index(), 3);
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
fn export_bookmark_matches_the_document_and_every_source_segment() {
    let mut state = AppState::new(Default::default());
    state.calibration = Some(calibration());
    state.left_input = Some(reco_io::InputPath::Chained(vec![
        "left-1.mp4".into(),
        "left-2.mp4".into(),
    ]));
    state.right_input = Some(reco_io::InputPath::Single("right.mp4".into()));
    let bookmark = PreviewBookmark::capture(&state).unwrap();
    assert!(bookmark.matches(&state));
    state.calibration.as_mut().unwrap().framing.roll = 0.1;
    assert!(!bookmark.matches(&state));
    state.calibration = Some(calibration());
    state.left_input = Some(reco_io::InputPath::Chained(vec![
        "left-1.mp4".into(),
        "different.mp4".into(),
    ]));
    assert!(!bookmark.matches(&state));
    state.suspend_preview_for_export();
    assert!(state.export_preview.is_some());
    state.unload_pipeline();
    state.restore_preview_after_export().unwrap();
    assert!(state.bridge.is_none());
    assert_eq!(state.playback.state(), PlayState::Empty);
}

fn decoded_gpu_state() -> AppState {
    let mut state = AppState::new(Default::default());
    state.shared_gpu = Some(GpuContext::new_blocking().unwrap());
    let input =
        reco_io::InputPath::Single(std::env::var("RECO_PLAYBACK_TEST_VIDEO").unwrap().into());
    state.left_input = Some(input.clone());
    state.right_input = Some(input);
    state.use_constrained_look = false;
    state
}

#[test]
#[ignore = "requires a GPU and RECO_PLAYBACK_TEST_VIDEO pointing to a short video with at least 15 frames"]
fn export_preview_restores_exact_source_frame_pose_and_calibration_reset_baseline() {
    let mut state = decoded_gpu_state();
    state.init_with_calibration(calibration()).unwrap();
    let baseline = state.cal_baseline.clone();
    let mut edited = state.calibration.clone().unwrap();
    edited.framing.roll = -0.1;
    state.apply_calibration(edited.clone()).unwrap();
    let pose = reco_core::geometry::RenderPose {
        yaw: 0.3,
        pitch: -0.15,
        fov_degrees: 40.0,
    };
    state.pose.set_pose(pose);
    for _ in 0..12 {
        assert!(state.playback.advance_blocking().unwrap());
    }
    let pixels = |state: &AppState| {
        let reco_core::source::FrameSet::Yuv420p(cameras) =
            state.playback.decoded_selection().unwrap()
        else {
            panic!("expected decoded camera images");
        };
        cameras
            .iter()
            .map(|camera| camera.y.clone())
            .collect::<Vec<_>>()
    };
    let frame = state.playback.frame_index();
    let before = pixels(&state);
    let mut invalid = edited.clone();
    invalid.lenses[0].fx = 0.0;
    assert!(state.init_with_calibration(invalid).is_err());
    assert_eq!(
        state.playback.frame_index(),
        frame,
        "rejected calibration reset playback"
    );
    assert_eq!(pixels(&state), before);
    assert_eq!(
        state.cal_baseline, baseline,
        "rejected calibration replaced Reset baseline"
    );
    state.suspend_preview_for_export();
    assert!(state.bridge.is_none());
    assert_eq!(state.playback.state(), PlayState::Empty);
    state.restore_preview_after_export().unwrap();
    assert!(state.export_preview.is_none());
    assert_eq!(state.playback.state(), PlayState::Paused);
    assert_eq!(state.playback.frame_index(), frame);
    assert_eq!(pixels(&state), before);
    assert_eq!(state.pose.current_pose(), pose);
    assert_eq!(state.calibration.as_ref(), Some(&edited));
    assert_eq!(state.cal_baseline, baseline);
}

#[test]
#[ignore = "requires a GPU and RECO_PLAYBACK_TEST_VIDEO pointing to a readable video"]
fn rebuilding_sources_preserves_unsaved_calibration_and_reset_baseline() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    let mut state = decoded_gpu_state();
    let path = PathBuf::from(std::env::var("RECO_PLAYBACK_TEST_VIDEO").unwrap());
    let baseline = calibration();
    let cal_path =
        std::env::temp_dir().join(format!("reco-rebuild-document-{}.json", std::process::id()));
    baseline.to_file(&cal_path).unwrap();
    state.calibration_path = Some(cal_path.clone());
    state.init_with_calibration(baseline.clone()).unwrap();
    assert!(state.edit_calibration(&app, |cal| cal.framing.roll = -0.1));
    app.set_lens_dirty(true);
    app.set_lens_info_available(true);
    app.set_lens_left_camera("Camera fixture".into());
    let edited = state.calibration.clone().unwrap();
    state.left_input = Some(reco_io::InputPath::Chained(vec![
        path.clone(),
        path.clone(),
    ]));
    state.right_input = Some(reco_io::InputPath::Chained(vec![path.clone(), path]));
    state.unload_pipeline();
    let state = Rc::new(RefCell::new(state));
    try_init_and_update(&state, &app.as_weak());
    assert_eq!(state.borrow().calibration.as_ref(), Some(&edited));
    assert_eq!(state.borrow().cal_baseline.as_ref(), Some(&baseline));
    assert!(app.get_files_loaded());
    assert!(app.get_cal_dirty());
    assert!(app.get_lens_dirty());
    assert!(app.get_lens_info_available());
    assert_eq!(app.get_lens_left_camera().as_str(), "Camera fixture");
    assert!((app.get_rig_roll() - (-0.1_f32).to_degrees()).abs() < 1e-5);
    assert_eq!(Calibration::from_file(&cal_path).unwrap(), baseline);
    std::fs::remove_file(&cal_path).unwrap();
    state.borrow_mut().unload_pipeline();
    try_init_and_update(&state, &app.as_weak());
    assert!(
        app.get_files_loaded(),
        "an accepted document does not depend on rereading its file"
    );
    assert_eq!(state.borrow().calibration.as_ref(), Some(&edited));
    state.borrow_mut().left_input = Some(reco_io::InputPath::Single(cal_path));
    try_init_and_update(&state, &app.as_weak());
    assert!(!app.get_files_loaded());
    assert!(state.borrow().bridge.is_none());
    assert_eq!(state.borrow().playback.state(), PlayState::Empty);
    assert_eq!(state.borrow().calibration.as_ref(), Some(&edited));
    assert!(app.get_cal_dirty());
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

fn exercise_editor(state: &mut AppState, app: &RecoApp) {
    let accepted = serde_json::to_value(state.calibration.as_ref().unwrap()).unwrap();
    let was_dirty = app.get_cal_dirty();
    app.set_cal_camera_axis_offset(0.0);
    assert!(!state.edit_calibration(app, |cal| cal.framing.axis_offset = 0.0));
    assert_eq!(app.get_cal_camera_axis_offset(), 0.25);
    assert_eq!(app.get_cal_dirty(), was_dirty);
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
    assert_eq!(
        state.calibration.as_ref().unwrap().lenses[0].correction,
        0.0
    );
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

    let gpu = match GpuContext::new_blocking() {
        Ok(gpu) => gpu,
        Err(
            error @ (reco_core::gpu::GpuError::NoAdapter
            | reco_core::gpu::GpuError::AdapterRequest(_)),
        ) => {
            assert!(
                std::env::var("RECO_REQUIRE_GPU").is_err(),
                "GPU required: {error}"
            );
            eprintln!("Skipping GPU editor checks: {error}");
            return;
        }
        Err(error) => panic!("GPU init: {error}"),
    };
    state.calibration = Some(calibration());
    state.bridge = Some(PreviewBridge::new(gpu, calibration(), vec![(64, 36); 2], 64, 36).unwrap());
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
