//! Workspace input and layout checks without a GPU or desktop session.

use std::cell::Cell;
use std::rc::Rc;

use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};
use slint::platform::{Platform, WindowAdapter, WindowEvent};
use slint::{ComponentHandle, Model, PhysicalSize};

use super::RecoApp;

struct TestPlatform(Rc<MinimalSoftwareWindow>);

impl Platform for TestPlatform {
    fn create_window_adapter(&self) -> Result<Rc<dyn WindowAdapter>, slint::PlatformError> {
        Ok(self.0.clone())
    }
}

fn workspace() -> (RecoApp, Rc<MinimalSoftwareWindow>) {
    let window = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    slint::platform::set_platform(Box::new(TestPlatform(window.clone()))).unwrap();
    let app = RecoApp::new().unwrap();
    app.show().unwrap();
    (app, window)
}

fn draw(window: &MinimalSoftwareWindow, width: u32, height: u32) -> Vec<slint::Rgb8Pixel> {
    window.set_size(PhysicalSize::new(width, height));
    window.request_redraw();
    slint::platform::update_timers_and_animations();
    let mut pixels = vec![slint::Rgb8Pixel::default(); (width * height) as usize];
    assert!(window.draw_if_needed(|renderer| {
        renderer.render(&mut pixels, width as usize);
    }));
    pixels
}

fn retain_export_screenshot(name: &str, pixels: &[slint::Rgb8Pixel]) {
    if let Some(dir) = std::env::var_os("RECO_GUI_TEST_ARTIFACT_DIR") {
        let path = std::path::Path::new(&dir).join(name).with_extension("ppm");
        assert!(!path.exists());
        let mut bytes = b"P6\n1280 820\n255\n".to_vec();
        bytes.extend(pixels.iter().flat_map(|pixel| [pixel.r, pixel.g, pixel.b]));
        std::fs::write(path, bytes).unwrap();
    }
}

fn key(app: &RecoApp, text: impl Into<slint::SharedString>) {
    let text = text.into();
    app.window()
        .dispatch_event(WindowEvent::KeyPressed { text: text.clone() });
    app.window()
        .dispatch_event(WindowEvent::KeyReleased { text });
}

fn click(app: &RecoApp, x: f32, y: f32) {
    let position = slint::LogicalPosition::new(x, y);
    let button = slint::platform::PointerEventButton::Left;
    app.window()
        .dispatch_event(WindowEvent::PointerMoved { position });
    app.window()
        .dispatch_event(WindowEvent::PointerPressed { position, button });
    app.window()
        .dispatch_event(WindowEvent::PointerReleased { position, button });
}

#[test]
fn workspace_keeps_preview_space_and_recovers_setup_after_clear() {
    let (app, window) = workspace();
    assert!(app.get_inspector_open());
    assert_eq!(app.get_inspector_tab(), 0);
    for dark in [true, false] {
        app.set_dark_mode(dark);
        for (width, height) in [(1280, 820), (1024, 700), (800, 600), (720, 600)] {
            draw(&window, width, height);
            assert!(app.get_preview_area_width() >= 300.0);
            assert!(app.get_preview_area_height() >= 400.0);
            let with_inspector = app.get_preview_area_width();
            app.set_inspector_open(false);
            draw(&window, width, height);
            assert!(app.get_preview_area_width() > with_inspector + 300.0);
            app.set_inspector_open(true);
        }
    }
    draw(&window, 800, 600);
    click(&app, 700.0, 86.0);
    assert_eq!(
        app.get_inspector_tab(),
        0,
        "Adjust is disabled without footage"
    );
    app.set_files_loaded(true);
    draw(&window, 800, 600);
    click(&app, 700.0, 86.0);
    assert_eq!(
        app.get_inspector_tab(),
        1,
        "Adjust responds to pointer input"
    );
    app.set_inspector_open(false);
    app.set_current_frame(25);
    app.set_total_frames(100);
    app.set_playing(true);
    app.set_files_loaded(false);
    draw(&window, 800, 600);
    assert!(app.get_inspector_open());
    assert_eq!(app.get_inspector_tab(), 0);
    assert_eq!(app.get_current_frame(), 0);
    assert_eq!(app.get_total_frames(), 0);
    assert!(!app.get_playing());
}

#[test]
fn dialogs_block_preview_shortcuts_and_escape_dismisses_without_playback() {
    let (app, window) = workspace();
    app.set_files_loaded(true);
    draw(&window, 800, 600);
    let calls = Rc::new(Cell::new(0));
    let count = calls.clone();
    app.on_toggle_playback(move || count.set(count.get() + 1));
    app.invoke_refocus();
    key(&app, " ");
    assert_eq!(calls.get(), 1);

    for modal in 0..6 {
        match modal {
            0 => app.set_prefs_dialog_open(true),
            1 => app.set_shortcuts_dialog_open(true),
            2 => app.set_export_dialog_open(true),
            3 => app.set_recent_dialog_open(true),
            4 => app.set_lens_picker_open(true),
            _ => app.set_bug_dialog_open(true),
        }
        draw(&window, 800, 600);
        app.invoke_refocus();
        key(&app, " ");
        assert_eq!(calls.get(), 1, "modal {modal} leaked a playback shortcut");
        key(&app, slint::platform::Key::Escape);
        assert!(!app.get_prefs_dialog_open());
        assert!(!app.get_shortcuts_dialog_open());
        assert!(!app.get_export_dialog_open());
        assert!(!app.get_recent_dialog_open());
        assert!(!app.get_lens_picker_open());
        assert!(!app.get_bug_dialog_open());
    }
    app.set_file_dialog_open(true);
    app.invoke_refocus();
    key(&app, " ");
    assert_eq!(calls.get(), 1);
    key(&app, slint::platform::Key::Escape);
    assert!(
        app.get_file_dialog_open(),
        "Escape must not pretend a native picker closed"
    );
    draw(&window, 800, 600);
    click(&app, 400.0, 300.0);
    assert!(
        app.get_file_dialog_open(),
        "workspace clicks must not release a pending picker"
    );
    app.set_file_dialog_open(false);
    app.invoke_refocus();
    key(&app, " ");
    assert_eq!(calls.get(), 2);
}

#[test]
fn export_waits_for_recording_and_calibration_to_stop() {
    let (app, window) = workspace();
    app.set_files_loaded(true);
    let calls = Rc::new(Cell::new(0));
    let count = calls.clone();
    app.on_open_export_dialog(move || count.set(count.get() + 1));
    for (recording, calibrating) in [(true, false), (false, true), (false, false)] {
        app.set_recording(recording);
        app.set_calibrating(calibrating);
        draw(&window, 1280, 820);
        click(&app, 1225.0, 30.0);
        assert_eq!(calls.get(), usize::from(!recording && !calibrating));
    }
}

#[test]
fn calibration_action_cancels_then_waits_for_the_worker() {
    let (app, window) = workspace();
    app.set_files_loaded(true);
    app.set_left_path("left.mp4".into());
    app.set_right_path("right.mp4".into());
    app.set_left_segments(slint::ModelRc::new(slint::VecModel::from(vec![
        "left.mp4".into(),
    ])));
    app.set_right_segments(slint::ModelRc::new(slint::VecModel::from(vec![
        "right.mp4".into(),
    ])));
    let starts = Rc::new(Cell::new(0));
    let cancels = Rc::new(Cell::new(0));
    let count = starts.clone();
    app.on_auto_calibrate(move || count.set(count.get() + 1));
    let count = cancels.clone();
    app.on_cancel_calibration(move || count.set(count.get() + 1));
    draw(&window, 1280, 820);
    click(&app, 1060.0, 418.0);
    assert_eq!(starts.get(), 1);
    app.set_calibrating(true);
    draw(&window, 1280, 820);
    click(&app, 1060.0, 418.0);
    assert_eq!(cancels.get(), 1);
    app.set_calibration_step("Stopping…".into());
    draw(&window, 1280, 820);
    click(&app, 1060.0, 418.0);
    assert_eq!(starts.get(), 1);
    assert_eq!(cancels.get(), 1);
}

#[test]
fn error_toasts_are_visible_in_the_workspace_and_above_dialogs() {
    let (app, window) = workspace();
    let dismissed = Rc::new(Cell::new(0));
    let last = dismissed.clone();
    app.on_toast_dismissed(move |id| last.set(id));
    for (width, height) in [(1280, 820), (720, 600)] {
        for modal in [false, true] {
            app.set_shortcuts_dialog_open(modal);
            app.set_toasts(Default::default());
            let before = draw(&window, width, height);
            app.set_toasts(slint::ModelRc::new(slint::VecModel::from(vec![
                super::Toast {
                    id: 7,
                    severity: "error".into(),
                    title: "Export failed".into(),
                    body: "The selected model could not be opened.".into(),
                },
            ])));
            let after = draw(&window, width, height);
            let changed = before
                .iter()
                .zip(after.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert!(
                changed > 2_000,
                "toast has no visible card: {width}x{height}, modal={modal}, changed={changed}"
            );
            let mut top = height;
            let mut bottom = 0;
            for (index, (a, b)) in before.iter().zip(after.iter()).enumerate() {
                if a != b {
                    let y = index as u32 / width;
                    top = top.min(y);
                    bottom = bottom.max(y);
                }
            }
            dismissed.set(0);
            let inset = if modal { 0.0 } else { 332.0 };
            click(
                &app,
                width as f32 - inset - 43.0,
                (top + bottom) as f32 / 2.0,
            );
            assert_eq!(dismissed.get(), 7, "toast close action was not reachable");
        }
    }
}

#[test]
fn notifications_do_not_replace_the_workspace_status() {
    let (app, _) = workspace();
    app.set_status_text("Coverage ready".into());
    let mut toasts = super::toast::ToastManager::default();
    toasts.push(
        super::toast::Severity::Error,
        "Export failed",
        "Model unavailable",
    );
    super::toast::sync_to_ui(&toasts, &app);
    assert_eq!(app.get_status_text(), "Coverage ready");
}

#[test]
fn roi_vertices_follow_the_contained_image_not_its_letterbox() {
    let (app, window) = workspace();
    app.set_files_loaded(true);
    app.set_field_editing(true);
    app.set_has_roi(true);
    app.set_roi_points_x(slint::ModelRc::new(slint::VecModel::from(vec![0.5])));
    app.set_roi_points_y(slint::ModelRc::new(slint::VecModel::from(vec![0.2])));
    for (image_width, image_height) in [(1920, 1080), (1080, 1920)] {
        let image = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::new(image_width, image_height);
        app.set_preview_frame(slint::Image::from_rgb8(image));
        for (width, height) in [(1280, 820), (800, 600)] {
            let pixels = draw(&window, width, height);
            let area_width = f64::from(app.get_preview_area_width());
            let area_height = f64::from(app.get_preview_area_height());
            let aspect = f64::from(image_width) / f64::from(image_height);
            let image_width = area_width.min(area_height * aspect);
            let image_height = image_width / aspect;
            let expected_x = 12.0 + area_width / 2.0;
            let expected_y = 62.0 + (area_height - image_height) / 2.0 + image_height * 0.2;
            let mut count = 0;
            let mut center = [0.0, 0.0];
            for (index, pixel) in pixels.iter().enumerate() {
                if pixel.g > 200 && pixel.r < 64 && pixel.b < 130 {
                    count += 1;
                    center[0] += (index % width as usize) as f64 + 0.5;
                    center[1] += (index / width as usize) as f64 + 0.5;
                }
            }
            assert!(count > 0, "ROI vertex is not visible");
            assert!((center[0] / f64::from(count) - expected_x).abs() < 1.0);
            assert!((center[1] / f64::from(count) - expected_y).abs() < 1.0);
        }
    }
}

#[test]
fn field_setup_is_reachable_without_scrolling_and_gated_during_output() {
    let (app, window) = workspace();
    app.set_files_loaded(true);
    let edits = Rc::new(Cell::new(0));
    let count = edits.clone();
    app.on_edit_field(move || count.set(count.get() + 1));
    draw(&window, 800, 600);
    click(&app, 625.0, 160.0);
    assert_eq!(edits.get(), 1);
    app.set_recording(true);
    draw(&window, 800, 600);
    click(&app, 625.0, 160.0);
    assert_eq!(edits.get(), 1);
    app.set_recording(false);
    app.set_export_in_progress(true);
    draw(&window, 800, 600);
    click(&app, 625.0, 160.0);
    assert_eq!(edits.get(), 1);
}

#[test]
#[ignore = "requires RECO_FIELD_TEST_CALIBRATION, RECO_FIELD_TEST_INPUTS (semicolon-separated), and RECO_FIELD_TEST_SCREENSHOT"]
fn real_footage_field_selection_drag_and_pan() {
    use reco_core::{
        calibration::Calibration,
        geometry::WorldPose,
        render::viewport::ViewportSize,
        source::{FrameSet, FrameSource},
        stitch::CpuExecutor,
    };
    use std::{cell::RefCell, path::PathBuf};
    let cal = Calibration::from_file(&PathBuf::from(
        std::env::var("RECO_FIELD_TEST_CALIBRATION").unwrap(),
    ))
    .unwrap();
    let inputs: Vec<_> = std::env::var("RECO_FIELD_TEST_INPUTS")
        .unwrap()
        .split(';')
        .map(|p| reco_io::InputPath::Single(p.into()))
        .collect();
    let mut source =
        reco_io::adapters::FfmpegFileSource::open_inputs(&inputs, cal.sync_offset, true).unwrap();
    let FrameSet::Yuv420p(cameras) = source.next_frame().unwrap().unwrap().images else {
        panic!("host fixture")
    };
    drop(source);
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    let pose = cal.world_to_render_pose(WorldPose {
        yaw: 0.0,
        pitch: -0.15,
        fov_degrees: 65.0,
    });
    let executor = CpuExecutor::new(
        cal.clone(),
        size.clone(),
        cameras.iter().map(|c| c.size).collect(),
        false,
    )
    .unwrap();
    let planes: Vec<_> = cameras.iter().map(|c| c.as_planes()).collect();
    let render = |pose| {
        let pixels = executor.stitch_yuv(&planes, pose).unwrap();
        slint::Image::from_rgba8(
            slint::SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(
                &pixels,
                size.width,
                size.height,
            ),
        )
    };
    let (app, window) = workspace();
    app.set_files_loaded(true);
    app.set_preview_frame(render(pose));
    let mut state = super::AppState::new(Default::default());
    state.calibration = Some(cal);
    state.left_input = inputs.first().cloned();
    state.right_input = inputs.get(1).cloned();
    state.pose.set_pose(pose);
    state.displayed_view = Some((size.clone(), pose));
    let state = Rc::new(RefCell::new(state));
    super::field_editor::install(&app, &state);
    let pan_count = Rc::new(Cell::new(0));
    let count = pan_count.clone();
    app.on_pan(move |_, _| count.set(count.get() + 1));
    draw(&window, 1280, 820);
    app.invoke_edit_field();
    draw(&window, 1280, 820);
    let w = app.get_preview_area_width();
    let h = app.get_preview_area_height();
    let iw = w.min(h * 16.0 / 9.0);
    let ih = iw * 9.0 / 16.0;
    let screen = |x: f32, y: f32| {
        slint::LogicalPosition::new(
            12.0 + (w - iw) * 0.5 + x * iw,
            62.0 + (h - ih) * 0.5 + y * ih,
        )
    };
    for [x, y] in [[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]] {
        let p = screen(x, y);
        click(&app, p.x, p.y);
        draw(&window, 1280, 820);
    }
    assert_eq!(app.get_field_corner_count(), 4);
    assert!(app.get_field_can_apply());
    let start = screen(0.2, 0.3);
    let end = screen(0.22, 0.32);
    let button = slint::platform::PointerEventButton::Left;
    app.window().dispatch_event(WindowEvent::PointerPressed {
        position: start,
        button,
    });
    app.window()
        .dispatch_event(WindowEvent::PointerMoved { position: end });
    app.window().dispatch_event(WindowEvent::PointerReleased {
        position: end,
        button,
    });
    assert_eq!(pan_count.get(), 0, "dragging a handle must not pan");
    assert!((app.get_roi_points_x().row_data(0).unwrap() - 0.22).abs() < 1e-4);
    assert!(app.get_field_can_apply());
    let output = PathBuf::from(std::env::var("RECO_FIELD_TEST_SCREENSHOT").unwrap());
    for (name, pose) in [
        ("selected", pose),
        (
            "panned",
            reco_core::geometry::RenderPose {
                yaw: pose.yaw + 0.15,
                fov_degrees: 55.0,
                ..pose
            },
        ),
    ] {
        app.set_preview_frame(render(pose));
        state.borrow_mut().displayed_view = Some((size.clone(), pose));
        super::field_editor::sync(&mut state.borrow_mut(), &app);
        let pixels = draw(&window, 1280, 820);
        let mut ppm = b"P6\n1280 820\n255\n".to_vec();
        for p in pixels {
            ppm.extend_from_slice(&[p.r, p.g, p.b]);
        }
        std::fs::write(output.with_extension(format!("{name}.ppm")), ppm).unwrap();
    }
    let start = screen(0.5, 0.5);
    let end = screen(0.55, 0.5);
    app.window().dispatch_event(WindowEvent::PointerPressed {
        position: start,
        button,
    });
    app.window()
        .dispatch_event(WindowEvent::PointerMoved { position: end });
    app.window().dispatch_event(WindowEvent::PointerReleased {
        position: end,
        button,
    });
    assert!(pan_count.get() > 0);
    assert_eq!(
        app.get_field_corner_count(),
        4,
        "panning must not add a corner"
    );
    app.invoke_finish_field(true);
    assert!(app.get_has_roi());
    assert!(app.get_cal_dirty());
}

#[test]
fn dialog_tab_navigation_cannot_activate_background_controls() {
    let (app, window) = workspace();
    draw(&window, 800, 600);
    app.invoke_refocus();
    app.set_shortcuts_dialog_open(true);
    draw(&window, 800, 600);
    for _ in 0..12 {
        key(&app, slint::platform::Key::Tab);
        key(&app, " ");
        assert!(
            app.get_inspector_open(),
            "dialog tab reached the workspace toggle"
        );
        // The Close action is allowed; keep exercising the next dialog instance.
        if !app.get_shortcuts_dialog_open() {
            app.set_shortcuts_dialog_open(true);
            draw(&window, 800, 600);
        }
    }
}

#[test]
fn dialog_actions_remain_reachable_without_scrolling() {
    let (app, window) = workspace();
    let saves = Rc::new(Cell::new(0));
    let count = saves.clone();
    app.on_save_prefs(move || count.set(count.get() + 1));
    let exports = Rc::new(Cell::new(0));
    let count = exports.clone();
    app.on_start_export(move || count.set(count.get() + 1));
    app.set_export_output_path("test.mp4".into());
    for (i, (width, height)) in [(1280, 820), (800, 600)].into_iter().enumerate() {
        draw(&window, width, height);
        app.set_prefs_dialog_open(true);
        draw(&window, width, height);
        let bottom = (height as f32 + (height - 60).min(680) as f32) / 2.0;
        click(&app, width as f32 / 2.0 + 192.0, bottom - 38.0);
        assert_eq!(
            saves.get(),
            i + 1,
            "Save was not reachable at {width}x{height}"
        );
        assert!(!app.get_prefs_dialog_open());

        app.set_export_dialog_open(true);
        draw(&window, width, height);
        let bottom = (height as f32 + (height - 60).min(720) as f32) / 2.0;
        click(&app, width as f32 / 2.0 + 200.0, bottom - 42.0);
        assert_eq!(
            exports.get(),
            i + 1,
            "Export was not reachable at {width}x{height}"
        );
        app.set_export_dialog_open(false);
    }
}

#[test]
fn camera_slider_keeps_pointer_while_dragging_inside_the_scroller() {
    let (app, window) = workspace();
    app.set_files_loaded(true);
    app.set_inspector_tab(1);
    app.set_fov(75.0);
    draw(&window, 1280, 820);
    let changes = Rc::new(Cell::new(0));
    let count = changes.clone();
    app.on_changed_fov(move |_| count.set(count.get() + 1));
    let button = slint::platform::PointerEventButton::Left;
    let position = slint::LogicalPosition::new(1115.0, 178.0);
    app.window()
        .dispatch_event(WindowEvent::PointerPressed { position, button });
    let mut previous = app.get_fov();
    for (step, x) in (1120..1240).step_by(10).enumerate() {
        app.window().dispatch_event(WindowEvent::PointerMoved {
            position: slint::LogicalPosition::new(x as f32, 178.0 + (step % 3) as f32),
        });
        draw(&window, 1280, 820);
        assert!(app.get_fov() > previous, "slider stopped at step {step}");
        previous = app.get_fov();
    }
    app.window().dispatch_event(WindowEvent::PointerReleased {
        position: slint::LogicalPosition::new(1230.0, 179.0),
        button,
    });
    assert!(changes.get() >= 12);
    assert!(app.get_fov() > 130.0);
}

#[test]
fn timeline_seeks_once_on_release_and_keeps_exact_frame_endpoints() {
    let (app, window) = workspace();
    app.set_files_loaded(true);
    app.set_total_frames(60);
    app.set_current_frame(18);
    let targets = Rc::new(std::cell::RefCell::new(Vec::new()));
    let recorded = targets.clone();
    app.on_seek_frame(move |frame| recorded.borrow_mut().push(frame));
    draw(&window, 1280, 820);
    app.set_current_frame(30);
    draw(&window, 1280, 820);
    assert!(
        targets.borrow().is_empty(),
        "playback must not echo as a seek"
    );
    let button = slint::platform::PointerEventButton::Left;
    app.window().dispatch_event(WindowEvent::PointerPressed {
        position: slint::LogicalPosition::new(500.0, 748.0),
        button,
    });
    for x in [510.0, 530.0, 560.0] {
        app.window().dispatch_event(WindowEvent::PointerMoved {
            position: slint::LogicalPosition::new(x, 748.0),
        });
        draw(&window, 1280, 820);
        assert!(
            targets.borrow().is_empty(),
            "drag must not seek until release"
        );
    }
    app.window().dispatch_event(WindowEvent::PointerReleased {
        position: slint::LogicalPosition::new(560.0, 748.0),
        button,
    });
    assert_eq!(targets.borrow().len(), 1);
    key(&app, slint::platform::Key::Home);
    assert_eq!(*targets.borrow().last().unwrap(), 0);
    key(&app, slint::platform::Key::RightArrow);
    assert_eq!(*targets.borrow().last().unwrap(), 1);
    key(&app, slint::platform::Key::End);
    assert_eq!(*targets.borrow().last().unwrap(), 59);
    assert_eq!(targets.borrow().len(), 4);
}

#[test]
fn native_picker_completion_cancellation_and_disconnect_release_the_guard() {
    use super::{AppState, FileDialogAction, SelectedFileDialog, poll_file_dialog};
    use std::cell::RefCell;

    let (app, window) = workspace();
    let state = Rc::new(RefCell::new(AppState::new(Default::default())));
    for outcome in 0..3 {
        let (sender, receiver) = std::sync::mpsc::channel();
        state.borrow_mut().file_dialog_open = true;
        state.borrow_mut().file_dialog_rx = Some(receiver);
        app.set_file_dialog_open(true);
        draw(&window, 800, 600);
        assert!(!poll_file_dialog(&state, &app.as_weak()));
        assert!(state.borrow().file_dialog_open);
        assert!(app.get_file_dialog_open());
        match outcome {
            0 => {
                let action = SelectedFileDialog {
                    selection: 42,
                    apply: |value, state: &Rc<RefCell<AppState>>, weak: &slint::Weak<RecoApp>| {
                        assert!(!state.borrow().file_dialog_open);
                        let app = weak.upgrade().unwrap();
                        assert!(!app.get_file_dialog_open());
                        app.set_current_frame(value);
                    },
                };
                sender
                    .send(Some(Box::new(action) as Box<dyn FileDialogAction>))
                    .unwrap();
            }
            1 => sender.send(None).unwrap(),
            _ => drop(sender),
        }
        assert!(poll_file_dialog(&state, &app.as_weak()));
        assert!(!state.borrow().file_dialog_open);
        assert!(state.borrow().file_dialog_rx.is_none());
        assert!(!app.get_file_dialog_open());
        assert!(!poll_file_dialog(&state, &app.as_weak()));
        assert_eq!(app.get_current_frame(), 42);
    }
}

#[test]
fn lens_preview_can_be_left_without_reopening_the_inspector() {
    let (app, window) = workspace();
    app.set_files_loaded(true);
    app.set_inspector_open(false);
    app.set_lens_preview_active(true);
    let pans = Rc::new(Cell::new(0));
    let count = pans.clone();
    app.on_pan(move |_, _| count.set(count.get() + 1));
    let changes = Rc::new(Cell::new(0));
    let count = changes.clone();
    app.on_changed_lens_preview(move || count.set(count.get() + 1));
    draw(&window, 800, 600);
    app.invoke_refocus();
    key(&app, slint::platform::Key::RightArrow);
    assert_eq!(pans.get(), 0);
    click(&app, 80.0, 95.0);
    assert!(!app.get_lens_preview_active());
    assert_eq!(changes.get(), 1);
    app.invoke_refocus();
    key(&app, slint::platform::Key::RightArrow);
    assert_eq!(pans.get(), 1);
    app.set_export_in_progress(true);
    key(&app, slint::platform::Key::RightArrow);
    assert_eq!(pans.get(), 1);
}

#[test]
fn preference_combobox_is_modal_and_cancel_keeps_live_recording_settings() {
    let (app, window) = workspace();
    app.set_recording_codec("h264".into());
    app.set_recording_quality("balanced".into());
    app.set_prefs_recording_codec("h264".into());
    app.set_prefs_recording_quality("balanced".into());
    draw(&window, 1280, 820);
    app.set_prefs_dialog_open(true);
    draw(&window, 1280, 820);
    click(&app, 525.0, 470.0);
    draw(&window, 1280, 820);
    key(&app, slint::platform::Key::DownArrow);
    key(&app, "\n");
    draw(&window, 1280, 820);
    assert_eq!(app.get_prefs_recording_codec(), "hevc");
    assert!(
        app.get_prefs_dialog_open(),
        "the nested menu closed its parent dialog"
    );
    assert_eq!(app.get_recording_codec(), "h264");
    click(&app, 765.0, 712.0);
    assert!(!app.get_prefs_dialog_open());
    assert_eq!(app.get_recording_codec(), "h264");
    assert_eq!(app.get_recording_quality(), "balanced");
}

#[test]
fn long_status_messages_do_not_collapse_the_preview() {
    let (app, window) = workspace();
    app.set_files_loaded(true);
    app.set_status_text(
        format!(
            "Export failed: {}",
            "a long source or output path ".repeat(30)
        )
        .into(),
    );
    draw(&window, 800, 600);
    assert!(
        app.get_preview_area_height() >= 350.0,
        "status text displaced the workspace"
    );
}

#[test]
fn processed_output_exposes_publishing_without_replacing_the_recording() {
    let (app, window) = workspace();
    app.set_files_loaded(true);
    app.set_export_dialog_open(true);
    app.set_export_output_path("recording.mp4".into());
    app.set_export_codec("hevc".into());
    app.set_export_autocam_enabled(false);
    app.set_clip_duration_secs(120.0);
    app.set_export_end_secs(120.0);
    let pixels = draw(&window, 1280, 820);
    retain_export_screenshot("export-before", &pixels);
    let started = Rc::new(Cell::new(0));
    let counter = started.clone();
    app.on_start_export(move || counter.set(counter.get() + 1));
    click(&app, 393.0, 233.0);
    assert!(app.get_export_stream_enabled());
    assert_eq!(app.get_export_codec(), "h264");
    draw(&window, 1280, 820);
    click(&app, 842.0, 728.0);
    assert_eq!(
        started.get(),
        0,
        "an empty publishing URL cannot start output"
    );
    app.set_export_stream_url("rtmps://127.0.0.1/live/private-key".into());
    let pixels = draw(&window, 1280, 820);
    assert_eq!(app.get_export_output_path(), "recording.mp4");
    retain_export_screenshot("export-publish", &pixels);
    click(&app, 842.0, 728.0);
    assert_eq!(started.get(), 1);
    click(&app, 550.0, 357.0);
    key(&app, slint::platform::Key::UpArrow);
    key(&app, "\n");
    assert_eq!(app.get_export_audio_choice(), 0);
    app.set_export_dialog_open(false);
    app.set_export_in_progress(true);
    app.set_export_status_text("Frame 120 (30 fps)".into());
    app.set_export_stream_status("Stream failed: peer disconnected".into());
    app.set_export_stream_failed(true);
    draw(&window, 800, 600);
    assert!(app.get_preview_area_height() > 350.0);
    app.set_export_in_progress(false);
    app.set_status_text("Recording saved".into());
    draw(&window, 800, 600);
    assert_eq!(
        app.get_export_stream_status(),
        "Stream failed: peer disconnected"
    );
    let mut state = crate::export::StreamState::default();
    state.finish(&app);
    assert_eq!(app.get_export_stream_status(), "Stream: not started");
    assert!(!app.get_export_stream_failed());
}

#[test]
fn network_form_edits_camera_rows_and_keeps_cancellation_visible() {
    let (app, window) = workspace();
    app.set_network_inputs(
        Rc::new(slint::VecModel::from(vec![
            super::NetworkInputRow::default();
            2
        ]))
        .into(),
    );
    app.set_network_dialog_open(true);
    let pixels = draw(&window, 1280, 820);
    retain_export_screenshot("network-inputs-empty", &pixels);
    click(&app, 550.0, 264.0);
    key(&app, "rtsp://127.0.0.1:8554/left");
    click(&app, 550.0, 303.0);
    key(&app, "0");
    click(&app, 550.0, 376.0);
    key(&app, "rtsp://127.0.0.1:8554/right");
    click(&app, 550.0, 417.0);
    key(&app, "-7000000");
    assert_eq!(
        app.get_network_inputs().row_data(0).unwrap().url,
        "rtsp://127.0.0.1:8554/left"
    );
    assert_eq!(app.get_network_inputs().row_data(0).unwrap().offset, "0");
    assert_eq!(
        app.get_network_inputs().row_data(1).unwrap().url,
        "rtsp://127.0.0.1:8554/right"
    );
    assert_eq!(
        app.get_network_inputs().row_data(1).unwrap().offset,
        "-7000000"
    );
    let pixels = draw(&window, 1280, 820);
    retain_export_screenshot("network-inputs-configured", &pixels);
    let cancelled = Rc::new(Cell::new(0));
    let count = cancelled.clone();
    app.on_cancel_network(move || count.set(count.get() + 1));
    app.set_network_opening(true);
    key(&app, slint::platform::Key::Escape);
    assert_eq!(cancelled.get(), 1);
    assert!(app.get_network_dialog_open());
    app.set_network_opening(false);
    key(&app, slint::platform::Key::Escape);
    assert!(!app.get_network_dialog_open());
}
