//! File/live setup and output controls use the same workspace without a second setup state.

use super::*;

pub(super) fn select_files(app: &RecoApp) {
    app.set_left_path("left.mp4".into());
    app.set_right_path("right.mp4".into());
    app.set_left_segments(Rc::new(slint::VecModel::from(vec!["left.mp4".into()])).into());
    app.set_right_segments(Rc::new(slint::VecModel::from(vec!["right.mp4".into()])).into());
}

#[test]
fn setup_and_output_states_fit_small_and_normal_windows() {
    let (app, window) = workspace();
    if let Some(path) = std::env::var_os("RECO_GUI_TEST_PREVIEW") {
        app.set_preview_frame(slint::Image::load_from_path(std::path::Path::new(&path)).unwrap());
    }
    for dark in [true, false] {
        app.set_dark_mode(dark);
        for (width, height) in [(1280, 820), (720, 600)] {
            for scene in ["empty", "files", "corners", "live", "live-output"] {
                app.set_export_dialog_open(false);
                app.set_field_editing(false);
                app.set_network_active(false);
                app.set_files_loaded(false);
                app.set_left_path("".into());
                app.set_right_path("".into());
                app.set_left_segments(Default::default());
                app.set_right_segments(Default::default());
                app.set_calibration_path("".into());
                if scene != "empty" {
                    select_files(&app);
                    app.set_calibration_path("pitch.json".into());
                    app.set_files_loaded(true);
                    app.set_status_text("Preview ready".into());
                } else {
                    app.set_status_text("Open video files or connect live cameras to begin".into());
                }
                if scene == "corners" {
                    app.set_field_editing(true);
                    app.set_field_corner_count(4);
                    app.set_field_can_apply(true);
                    app.set_field_message(
                        "Drag corners to adjust. Drag elsewhere to pan; scroll to zoom.".into(),
                    );
                }
                if scene.starts_with("live") {
                    app.set_network_active(true);
                    app.set_network_inputs(
                        Rc::new(slint::VecModel::from(
                            vec![super::super::NetworkInputRow::default(); 2],
                        ))
                        .into(),
                    );
                    app.set_fps(29.97);
                    app.set_left_path("".into());
                    app.set_right_path("".into());
                }
                if scene == "live-output" {
                    app.set_export_output_path("match.mp4".into());
                    app.set_export_start_secs(0.0);
                    app.set_export_end_secs(0.0);
                    app.set_export_dialog_open(true);
                }
                let pixels = draw(&window, width, height);
                assert!(app.get_preview_area_width() >= 300.0, "{scene}");
                assert!(app.get_preview_area_height() >= 400.0, "{scene}");
                let theme = if dark { "dark" } else { "light" };
                retain_screenshot(&format!("{scene}-{theme}-{width}"), &pixels, width, height);
            }
        }
    }
}

#[test]
fn live_connection_is_reachable_and_does_not_interrupt_active_work() {
    let (app, window) = workspace();
    let calls = Rc::new(Cell::new(0));
    let count = calls.clone();
    app.on_open_network(move || count.set(count.get() + 1));
    for (width, height) in [(720, 600), (1280, 820)] {
        for busy in 0..5 {
            app.set_export_in_progress(busy == 1);
            app.set_recording(busy == 2);
            app.set_calibrating(busy == 3);
            app.set_network_opening(busy == 4);
            draw(&window, width, height);
            let before = calls.get();
            click(&app, width as f32 - 80.0, 137.0);
            assert_eq!(calls.get() - before, usize::from(busy == 0));
        }
    }
}

#[test]
fn corner_actions_stay_reachable_and_adjust_cannot_hide_them() {
    let (app, window) = workspace();
    app.set_files_loaded(true);
    app.set_field_editing(true);
    app.set_field_corner_count(4);
    app.set_field_message("Drag corners to adjust. Drag elsewhere to pan; scroll to zoom.".into());
    let applied = Rc::new(Cell::new(0));
    let cancelled = Rc::new(Cell::new(0));
    let apply = applied.clone();
    let cancel = cancelled.clone();
    app.on_finish_field(move |accept| {
        if accept {
            apply.set(apply.get() + 1);
        } else {
            cancel.set(cancel.get() + 1);
        }
    });
    for (width, height) in [(720, 600), (1280, 820)] {
        app.set_field_can_apply(false);
        draw(&window, width, height);
        click(&app, width as f32 - 90.0, 86.0);
        assert_eq!(app.get_inspector_tab(), 0);
        let before = applied.get();
        click(&app, width as f32 - 250.0, 214.0);
        assert_eq!(applied.get(), before);
        app.set_field_can_apply(true);
        draw(&window, width, height);
        click(&app, width as f32 - 250.0, 214.0);
        assert_eq!(applied.get(), before + 1);
        let before = cancelled.get();
        click(&app, width as f32 - 145.0, 214.0);
        assert_eq!(cancelled.get(), before + 1);
    }
}

#[test]
fn live_output_does_not_display_a_file_trim_duration() {
    let (app, window) = workspace();
    app.set_network_active(true);
    app.set_files_loaded(true);
    app.set_export_dialog_open(true);
    app.set_export_output_path("match.mp4".into());
    app.set_export_start_secs(0.0);
    app.set_export_end_secs(0.0);
    let before = draw(&window, 1280, 820);
    app.set_export_start_secs(10.0);
    app.set_export_end_secs(120.0);
    let after = draw(&window, 1280, 820);
    assert_eq!(
        before, after,
        "file trim must not affect the live output form"
    );
}

#[test]
fn selected_files_can_be_inspected_but_not_changed_during_active_work() {
    let (app, window) = workspace();
    select_files(&app);
    app.set_files_loaded(true);
    let removals = Rc::new(Cell::new(0));
    let calls = removals.clone();
    app.on_remove_left_segment(move |_| calls.set(calls.get() + 1));
    draw(&window, 720, 600);
    click(&app, 540.0, 182.0);
    let pixels = draw(&window, 720, 600);
    retain_screenshot("files-expanded", &pixels, 720, 600);
    for busy in 0..5 {
        app.set_export_in_progress(busy == 1);
        app.set_recording(busy == 2);
        app.set_calibrating(busy == 3);
        app.set_network_opening(busy == 4);
        draw(&window, 720, 600);
        let before = removals.get();
        click(&app, 671.0, 271.0);
        assert_eq!(removals.get() - before, usize::from(busy == 0));
    }
    assert_eq!(app.get_left_segments().row_data(0).unwrap(), "left.mp4");
}

#[test]
fn live_workspace_opens_file_selection_without_discarding_the_connection() {
    let (app, window) = workspace();
    app.set_network_active(true);
    app.set_files_loaded(true);
    app.set_network_inputs(
        Rc::new(slint::VecModel::from(
            vec![super::super::NetworkInputRow::default(); 2],
        ))
        .into(),
    );
    let picks = Rc::new(Cell::new(0));
    let calls = picks.clone();
    app.on_pick_left_video(move || calls.set(calls.get() + 1));
    draw(&window, 720, 600);
    click(&app, 470.0, 249.0);
    assert_eq!(picks.get(), 1);
    assert!(app.get_network_active());
    assert!(app.get_files_loaded());
}
