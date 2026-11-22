//! Four source-attached corners edited in the displayed, framed panorama.

use super::{AppState, PreviewBookmark, RecoApp};
use reco_core::{
    calibration::FieldRoi, geometry::RenderPose, projection::SourcePoint,
    render::viewport::ViewportSize,
};
use slint::{ComponentHandle, Model};
use std::{cell::RefCell, fmt::Write, rc::Rc};

pub(super) struct FieldEditor {
    bookmark: PreviewBookmark,
    corners: Vec<SourcePoint>,
}

impl FieldEditor {
    fn region(&self) -> Result<FieldRoi, String> {
        let corners = self.corners.as_slice().try_into().map_err(|_| {
            format!(
                "Select {} more corners around the field",
                4 - self.corners.len()
            )
        })?;
        let roi = FieldRoi { corners };
        let mut cal = self.bookmark.calibration.clone();
        cal.field_roi = Some(roi.clone());
        cal.field_boundary().map_err(|e| e.to_string())?;
        Ok(roi)
    }

    fn place(&mut self, view: (ViewportSize, RenderPose), index: i32, p: [f64; 2]) -> bool {
        let Some(point) = self
            .bookmark
            .calibration
            .pick_source_point(&view.0, view.1, p)
        else {
            return false;
        };
        if index < 0 && self.corners.len() < 4 {
            self.corners.push(point);
        } else if let Some(corner) = self.corners.get_mut(index as usize) {
            *corner = point;
        } else {
            return false;
        }
        true
    }
}

pub(super) fn sync(state: &mut AppState, app: &RecoApp) {
    if state
        .field_editor
        .as_ref()
        .is_some_and(|e| !e.bookmark.matches(state))
    {
        state.field_editor = None;
    }
    let editor = state.field_editor.as_ref();
    app.set_field_editing(editor.is_some());
    app.set_has_roi(
        state
            .calibration
            .as_ref()
            .is_some_and(|c| c.field_roi.is_some()),
    );
    let region = editor.map(FieldEditor::region);
    app.set_field_can_apply(region.as_ref().is_some_and(Result::is_ok));
    app.set_field_corner_count(editor.map_or(0, |e| e.corners.len() as i32));
    app.set_field_message(
        region
            .as_ref()
            .map_or_else(String::new, |result| match result {
                Ok(_) => "Drag corners to adjust. Drag elsewhere to pan; scroll to zoom.".into(),
                Err(error) => error.clone(),
            })
            .into(),
    );
    if editor.is_none() {
        if app.get_roi_points_x().row_count() != 0 {
            app.set_roi_points_x(Default::default());
            app.set_roi_points_y(Default::default());
        }
        if !app.get_field_outline().is_empty() {
            app.set_field_outline("".into());
        }
        return;
    }
    let mut xs = vec![];
    let mut ys = vec![];
    let mut path = String::new();
    if let Some(cal) = &state.calibration
        && let Some((size, pose)) = state.displayed_view.clone()
        && !state.lens_preview_active
    {
        let corners = editor
            .map(|e| e.corners.as_slice())
            .or_else(|| cal.field_roi.as_ref().map(|r| r.corners.as_slice()))
            .unwrap_or(&[]);
        for corner in corners {
            let p = cal
                .project_source_point(*corner, &size, pose)
                .ok()
                .flatten()
                .unwrap_or([-1.0; 2]);
            xs.push(p[0] as f32);
            ys.push(p[1] as f32);
        }
        let mut draft = cal.clone();
        draft.field_roi = region.and_then(Result::ok);
        if let Ok(Some(boundary)) = draft.field_boundary() {
            let points = boundary.outline(cal, &size, pose);
            for (i, p) in points.iter().enumerate() {
                let _ = write!(
                    path,
                    "{} {} {} ",
                    if i == 0 { "M" } else { "L" },
                    p[0] * f64::from(size.width),
                    p[1] * f64::from(size.height)
                );
            }
            if !points.is_empty() {
                path.push('Z');
            }
        }
    }
    app.set_roi_points_x(slint::ModelRc::new(slint::VecModel::from(xs)));
    app.set_roi_points_y(slint::ModelRc::new(slint::VecModel::from(ys)));
    app.set_field_outline(path.into());
}

pub(super) fn install(app: &RecoApp, state: &Rc<RefCell<AppState>>) {
    let weak = app.as_weak();
    app.on_hit_field_corner(move |x, y, width, height| {
        let Some(app) = weak.upgrade() else {
            return -1;
        };
        let xs = app.get_roi_points_x();
        let ys = app.get_roi_points_y();
        (0..xs.row_count())
            .find(|&i| {
                let px = xs.row_data(i).unwrap_or(-1.0);
                let py = ys.row_data(i).unwrap_or(-1.0);
                (0.0..=1.0).contains(&px)
                    && (0.0..=1.0).contains(&py)
                    && ((px - x) * width).hypot((py - y) * height) <= 14.0
            })
            .map_or(-1, |i| i as i32)
    });
    let weak = app.as_weak();
    let shared = state.clone();
    app.on_edit_field(move || {
        let Some(app) = weak.upgrade() else {
            return;
        };
        let mut state = shared.borrow_mut();
        if state.is_recording()
            || app.get_export_in_progress()
            || app.get_calibrating()
            || state.displayed_view.is_none()
        {
            return;
        }
        let Some(bookmark) = PreviewBookmark::capture(&state) else {
            return;
        };
        let corners = bookmark
            .calibration
            .field_roi
            .as_ref()
            .map(|r| r.corners.to_vec())
            .unwrap_or_default();
        state.playback.pause();
        app.set_playing(false);
        state.lens_preview_active = false;
        app.set_lens_preview_active(false);
        state.preview_dirty = true;
        state.field_editor = Some(FieldEditor { bookmark, corners });
        sync(&mut state, &app);
    });
    let weak = app.as_weak();
    let shared = state.clone();
    app.on_place_field_corner(move |index, x, y| {
        let Some(app) = weak.upgrade() else {
            return;
        };
        let mut state = shared.borrow_mut();
        if state
            .field_editor
            .as_ref()
            .is_some_and(|e| !e.bookmark.matches(&state))
        {
            state.field_editor = None;
        }
        if let Some(view) = state.displayed_view.clone()
            && let Some(editor) = &mut state.field_editor
        {
            let placed = editor.place(view, index, [f64::from(x), f64::from(y)]);
            sync(&mut state, &app);
            if !placed {
                app.set_field_message("Choose a point on the visible source image".into());
            }
        }
    });
    let weak = app.as_weak();
    let shared = state.clone();
    app.on_finish_field(move |apply| {
        let Some(app) = weak.upgrade() else {
            return;
        };
        let mut state = shared.borrow_mut();
        if apply {
            let roi = state
                .field_editor
                .as_ref()
                .filter(|e| e.bookmark.matches(&state))
                .and_then(|e| e.region().ok());
            let Some(roi) = roi else {
                sync(&mut state, &app);
                return;
            };
            if !state.edit_calibration(&app, |c| c.field_roi = Some(roi)) {
                return;
            }
        }
        state.field_editor = None;
        sync(&mut state, &app);
    });
    let weak = app.as_weak();
    let shared = state.clone();
    app.on_reset_field_draft(move || {
        let Some(app) = weak.upgrade() else {
            return;
        };
        let mut state = shared.borrow_mut();
        if let Some(editor) = &mut state.field_editor {
            editor.corners.clear();
        }
        sync(&mut state, &app);
    });
    let weak = app.as_weak();
    let shared = state.clone();
    app.on_remove_field(move || {
        let Some(app) = weak.upgrade() else {
            return;
        };
        let mut state = shared.borrow_mut();
        state.edit_calibration(&app, |c| c.field_roi = None);
        sync(&mut state, &app);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::{
        calibration::{Calibration, Framing, Lens},
        geometry::WorldPose,
        projection::Cylinder,
    };
    use slint::platform::{
        Platform, WindowAdapter,
        software_renderer::{MinimalSoftwareWindow, RepaintBufferType},
    };

    struct Headless;
    impl Platform for Headless {
        fn create_window_adapter(&self) -> Result<Rc<dyn WindowAdapter>, slint::PlatformError> {
            Ok(MinimalSoftwareWindow::new(RepaintBufferType::ReusedBuffer))
        }
    }

    fn editor() -> (RecoApp, Rc<RefCell<AppState>>) {
        slint::platform::set_platform(Box::new(Headless)).unwrap();
        let app = RecoApp::new().unwrap();
        let mut state = AppState::new(Default::default());
        let cal = Calibration::new(
            vec![Lens::flat(3840, 2160)],
            Cylinder::default(),
            Framing {
                axis_offset: 0.0,
                tilt: 0.1,
                roll: -0.15,
            },
        );
        let pose = cal.world_to_render_pose(WorldPose {
            yaw: 0.2,
            pitch: 0.0,
            fov_degrees: 55.0,
        });
        state.pose.set_pose(pose);
        state.displayed_view = Some((
            ViewportSize {
                width: 1280,
                height: 720,
            },
            pose,
        ));
        state.calibration = Some(cal);
        state.left_input = Some(reco_io::InputPath::Single("panorama.mp4".into()));
        let shared = Rc::new(RefCell::new(state));
        install(&app, &shared);
        (app, shared)
    }

    fn place_four(app: &RecoApp) {
        for [x, y] in [[0.3, 0.35], [0.7, 0.35], [0.7, 0.65], [0.3, 0.65]] {
            app.invoke_place_field_corner(-1, x, y);
        }
    }

    #[test]
    fn draft_apply_cancel_and_explicit_save_preserve_the_framed_view() {
        let (app, state) = editor();
        let original = state.borrow().calibration.clone().unwrap();
        let pose = state.borrow().pose.current_pose();
        app.invoke_edit_field();
        assert!(app.get_field_editing());
        place_four(&app);
        assert!(app.get_field_can_apply());
        assert_eq!(app.get_field_corner_count(), 4);
        assert!(!app.get_cal_dirty());
        assert_eq!(state.borrow().calibration.as_ref(), Some(&original));
        assert_eq!(state.borrow().pose.current_pose(), pose);
        app.invoke_finish_field(false);
        assert!(!app.get_field_editing());
        assert_eq!(state.borrow().calibration.as_ref(), Some(&original));
        app.invoke_edit_field();
        place_four(&app);
        app.invoke_finish_field(true);
        assert!(app.get_has_roi());
        assert!(app.get_cal_dirty());
        assert!(!app.get_field_editing());
        assert_eq!(state.borrow().pose.current_pose(), pose);
        let path = std::env::temp_dir().join(format!("reco-corners-{}.json", uuid::Uuid::new_v4()));
        state.borrow_mut().calibration_path = Some(path.clone());
        state.borrow_mut().save_calibration().unwrap();
        assert_eq!(
            Calibration::from_file(&path).unwrap(),
            state.borrow().calibration.clone().unwrap()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn unordered_corners_apply_without_renumbering_observations() {
        let (app, state) = editor();
        app.invoke_edit_field();
        for [x, y] in [[0.3, 0.65], [0.3, 0.35], [0.7, 0.65], [0.7, 0.35]] {
            app.invoke_place_field_corner(-1, x, y);
        }
        assert!(app.get_field_can_apply());
        assert!(!app.get_field_outline().is_empty());
        let selected = state
            .borrow()
            .field_editor
            .as_ref()
            .unwrap()
            .corners
            .clone();
        app.invoke_finish_field(true);
        assert!(!app.get_field_editing());
        let state = state.borrow();
        let cal = state.calibration.as_ref().unwrap();
        assert_eq!(
            cal.field_roi.as_ref().unwrap().corners.as_slice(),
            selected.as_slice()
        );
        cal.validate().unwrap();
    }

    #[test]
    fn unsupported_and_interior_corners_never_change_the_accepted_region() {
        let (app, state) = editor();
        app.invoke_edit_field();
        app.invoke_place_field_corner(-1, -0.1, 0.5);
        assert_eq!(app.get_field_corner_count(), 0);
        app.invoke_place_field_corner(-1, 0.5, f32::NAN);
        assert_eq!(app.get_field_corner_count(), 0);
        for [x, y] in [[0.55, 0.55], [0.7, 0.65], [0.7, 0.35], [0.3, 0.65]] {
            app.invoke_place_field_corner(-1, x, y);
        }
        assert!(!app.get_field_can_apply());
        app.invoke_finish_field(true);
        assert!(app.get_field_editing());
        assert!(
            state
                .borrow()
                .calibration
                .as_ref()
                .unwrap()
                .field_roi
                .is_none()
        );
        app.invoke_reset_field_draft();
        place_four(&app);
        assert!(app.get_field_can_apply());
        state
            .borrow_mut()
            .calibration
            .as_mut()
            .unwrap()
            .framing
            .roll += 0.1;
        app.invoke_finish_field(true);
        assert!(!app.get_field_editing());
        assert!(
            state
                .borrow()
                .calibration
                .as_ref()
                .unwrap()
                .field_roi
                .is_none()
        );
    }

    #[test]
    fn handle_drag_uses_displayed_pose_and_source_anchors_survive_zoom() {
        let (app, state) = editor();
        app.invoke_edit_field();
        place_four(&app);
        assert_eq!(app.invoke_hit_field_corner(0.3, 0.35, 800.0, 450.0), 0);
        assert_eq!(app.invoke_hit_field_corner(0.5, 0.5, 800.0, 450.0), -1);
        // A newer requested pose must not be used until its image is displayed.
        state.borrow_mut().pose.set_pose(RenderPose {
            yaw: -0.3,
            pitch: 0.0,
            fov_degrees: 40.0,
        });
        app.invoke_place_field_corner(0, 0.32, 0.36);
        assert!((app.get_roi_points_x().row_data(0).unwrap() - 0.32).abs() < 1e-5);
        let before = state
            .borrow()
            .field_editor
            .as_ref()
            .unwrap()
            .corners
            .clone();
        let mut state = state.borrow_mut();
        state.displayed_view.as_mut().unwrap().1.fov_degrees = 45.0;
        sync(&mut state, &app);
        assert_eq!(state.field_editor.as_ref().unwrap().corners, before);
        assert!((app.get_roi_points_x().row_data(0).unwrap() - 0.32).abs() > 0.01);
    }
}
