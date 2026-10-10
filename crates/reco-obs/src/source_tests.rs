//! Thread ownership at the OBS callback boundary.

use super::*;
use reco_core::geometry::RenderPose;
use std::sync::{Arc, mpsc};
use std::time::Duration;

#[link(name = "obs")]
unsafe extern "C" {}

#[test]
fn unrelated_properties_preserve_interactive_pose_and_single_axis_edits() {
    let shared = SourceHandle::default();
    let settings = unsafe { ffi::obs_data_create() };
    assert!(!settings.is_null());
    unsafe {
        source_get_defaults(settings);
    }
    let data = &shared as *const SourceHandle as *mut c_void;
    unsafe {
        source_update(data, settings);
    }
    let requested = RenderPose {
        yaw: 0.7,
        pitch: -0.3,
        fov_degrees: 48.0,
    };
    shared
        .interaction
        .lock()
        .unwrap()
        .pose
        .set_target(requested);
    unsafe {
        ffi::obs_data_set_int(settings, PROP_OUTPUT_WIDTH.as_ptr(), 1280);
        source_update(data, settings);
    }
    assert_eq!(
        shared.interaction.lock().unwrap().pose.target_pose(),
        requested
    );
    unsafe {
        ffi::obs_data_set_double(settings, PROP_YAW.as_ptr(), 25.0);
        source_update(data, settings);
    }
    assert_eq!(
        shared.interaction.lock().unwrap().pose.target_pose(),
        RenderPose {
            yaw: 25_f32.to_radians(),
            ..requested
        }
    );
    unsafe {
        ffi::obs_data_set_double(settings, PROP_PITCH.as_ptr(), 15.0);
        source_update(data, settings);
    }
    assert_eq!(
        shared.interaction.lock().unwrap().pose.target_pose(),
        RenderPose {
            yaw: 25_f32.to_radians(),
            pitch: 15_f32.to_radians(),
            ..requested
        }
    );
    unsafe {
        ffi::obs_data_release(settings);
    }
}

#[test]
fn shared_state_is_thread_safe_and_core_can_move_between_callbacks() {
    fn send<T: Send>() {}
    fn sync<T: Sync>() {}
    send::<StitchCore>();
    send::<SourceHandle>();
    sync::<SourceHandle>();
}

#[test]
fn interaction_and_dimension_queries_do_not_wait_for_engine_or_texture() {
    let shared = Arc::new(SourceHandle::default());
    let engine = shared.engine.lock().unwrap();
    let display = shared.presentation.lock().unwrap();
    let thread_source = shared.clone();
    let (done, received) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        let data = Arc::as_ptr(&thread_source) as *mut c_void;
        let mut event = ffi::obs_mouse_event {
            modifiers: 0,
            x: 0,
            y: 0,
        };
        unsafe {
            source_mouse_click(
                data,
                &event,
                ffi::obs_mouse_button_type::MOUSE_LEFT as i32,
                false,
                1,
            );
            for value in 1..10_000 {
                event.x = value;
                source_mouse_move(data, &event, false);
                source_mouse_wheel(data, &event, 0, if value % 2 == 0 { 1 } else { -1 });
                assert_eq!(source_get_width(data), 1920);
                assert_eq!(source_get_height(data), 1080);
            }
            source_mouse_click(
                data,
                &event,
                ffi::obs_mouse_button_type::MOUSE_LEFT as i32,
                true,
                1,
            );
        }
        done.send(()).unwrap();
    });
    let completed_while_locked = received.recv_timeout(Duration::from_secs(3));
    drop(display);
    drop(engine);
    thread.join().unwrap();
    assert!(completed_while_locked.is_ok());
    assert!(shared.interaction.lock().unwrap().pose.target_pose().yaw > 0.0);
}

#[test]
fn presentation_invalidation_does_not_publish_old_pixels_with_new_dimensions() {
    let shared = SourceHandle::default();
    {
        let mut display = shared.presentation.lock().unwrap();
        display.rgba = vec![255; 64];
        display.dirty = true;
    }
    let mut engine = shared.engine.lock().unwrap();
    engine.output_width = 1024;
    engine.output_height = 512;
    shared.reset_presentation(&engine);
    let display = shared.presentation.lock().unwrap();
    assert!(display.rgba.is_empty());
    assert!(!display.dirty);
    assert_eq!((display.width, display.height), (1024, 512));
    let data = &shared as *const SourceHandle as *mut c_void;
    unsafe {
        assert_eq!(source_get_width(data), 1024);
        assert_eq!(source_get_height(data), 512);
    }
}

#[test]
fn unavailable_calibration_cannot_leave_the_previous_engine_active() {
    use reco_core::calibration::{Framing, Lens};
    use reco_core::projection::Cylinder;
    use reco_core::stitch::CpuExecutor;

    let calibration = Calibration::new(
        vec![Lens::flat(16, 16)],
        Cylinder {
            radius: 20.0,
            arc_deg: 180.0,
            surface_height: None,
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let executor = CpuExecutor::new(
        calibration,
        ViewportSize {
            width: 8,
            height: 4,
        },
        vec![(16, 16)],
        false,
    )
    .unwrap();
    let mut engine = RecoSource::new();
    engine.core = Some(StitchCore::new(Executor::Cpu(Box::new(executor))).unwrap());
    engine.try_init_pipeline();
    assert!(engine.core.is_none());
}

fn stereo_engine() -> RecoSource {
    use reco_core::calibration::{Framing, Lens};
    use reco_core::projection::LShape;
    use reco_core::stitch::CpuExecutor;
    let calibration = Calibration::new(
        vec![Lens::flat(16, 16); 2],
        LShape {
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.0,
        },
        Framing {
            axis_offset: 0.5,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let executor = CpuExecutor::new(
        calibration,
        ViewportSize {
            width: 8,
            height: 4,
        },
        vec![(16, 16); 2],
        false,
    )
    .unwrap();
    let mut engine = RecoSource::new();
    engine.core = Some(StitchCore::new(Executor::Cpu(Box::new(executor))).unwrap());
    engine
}

#[test]
fn pipeline_initialization_waits_for_observed_camera_dimensions() {
    let mut engine = stereo_engine();
    engine.calibration = Some(engine.core.as_ref().unwrap().calibration().clone());
    engine.try_init_pipeline();
    assert!(engine.core.is_none());
    assert_eq!(engine.input_sizes, None);
}

#[test]
#[ignore = "requires a GPU; run with each supported WGPU_BACKEND"]
fn pipeline_uses_each_camera_size_and_rejects_invalid_replacements() {
    for format in [InputFormat::Yuv420p, InputFormat::Rgba] {
        let mut engine = stereo_engine();
        engine.calibration = Some(engine.core.as_ref().unwrap().calibration().clone());
        engine.input_format = format;
        for sizes in [[(16, 16), (24, 12)], [(32, 18), (12, 24)]] {
            engine.input_sizes = Some(sizes);
            engine.try_init_pipeline();
            assert_eq!(
                engine.core.as_ref().expect("GPU pipeline").source_sizes(),
                &sizes
            );
        }
        engine.input_sizes = Some([(16, 16), (0, 12)]);
        engine.try_init_pipeline();
        assert!(
            engine.core.is_none(),
            "invalid inputs must not retain the previous pipeline"
        );
    }
}

#[test]
fn output_resize_keeps_raw_recording_attached_and_rejects_invalid_dimensions() {
    use reco_core::core::types::StackedReplayRecorder;
    use reco_core::render::planes::Yuv420pPlanes;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Recorder(Arc<AtomicUsize>);
    impl StackedReplayRecorder for Recorder {
        fn record_yuv(&mut self, _: &Yuv420pPlanes<'_>, _: &Yuv420pPlanes<'_>, _: u32, _: u32) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    let mut engine = stereo_engine();
    let calls = Arc::new(AtomicUsize::new(0));
    engine
        .core
        .as_mut()
        .unwrap()
        .set_stacked_recorder(Box::new(Recorder(calls.clone())))
        .unwrap();
    let y = vec![128; 16 * 16];
    let uv = vec![128; 8 * 8];
    let input = Yuv420pPlanes {
        y: &y,
        u: &uv,
        v: &uv,
    };
    for (width, height) in [(8, 4), (4, 8), (16, 8), (0, 8)] {
        engine.output_width = width;
        engine.output_height = height;
        engine.resize_output();
        let core = engine.core.as_mut().unwrap();
        assert_eq!(core.source_sizes(), &[(16, 16); 2]);
        assert_eq!(
            core.output_dims(),
            if width == 0 { (16, 8) } else { (width, height) }
        );
        assert_eq!(
            core.output_dims(),
            (engine.output_width, engine.output_height)
        );
        core.submit_frame_yuv420p_at_pose(&input, &input, RenderPose::default())
            .unwrap();
    }
    assert_eq!(calls.load(Ordering::Relaxed), 4);
}

#[test]
#[cfg(feature = "replay")]
fn replay_intent_or_path_change_finalizes_the_previous_recorder_once() {
    use reco_core::core::types::StackedReplayRecorder;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Recorder(Arc<AtomicUsize>);
    impl StackedReplayRecorder for Recorder {
        fn record_yuv(
            &mut self,
            _: &reco_core::render::planes::Yuv420pPlanes<'_>,
            _: &reco_core::render::planes::Yuv420pPlanes<'_>,
            _: u32,
            _: u32,
        ) {
        }
        fn finish(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    for change_path in [false, true] {
        let mut engine = stereo_engine();
        let finalized = Arc::new(AtomicUsize::new(0));
        engine
            .core
            .as_mut()
            .unwrap()
            .set_stacked_recorder(Box::new(Recorder(finalized.clone())))
            .unwrap();
        engine.replay_enabled = true;
        engine.replay_follow_obs = false;
        engine.replay_path = "first.mkv".into();
        engine.replay_recorder_path = Some(engine.replay_path.clone());
        engine.replay_attempted_path = engine.replay_recorder_path.clone();
        engine.update_replay_recorder(false);
        assert_eq!(finalized.load(Ordering::Relaxed), 0);
        if change_path {
            engine.replay_path = "second.mkv".into();
        } else {
            engine.replay_enabled = false;
        }
        engine.update_replay_recorder(false);
        engine.update_replay_recorder(false);
        assert_eq!(finalized.load(Ordering::Relaxed), 1);
        assert!(engine.replay_recorder_path.is_none());
        if !change_path {
            assert!(engine.replay_attempted_path.is_none());
        }
    }
}
