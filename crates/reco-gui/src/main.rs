//! Reco GUI - Slint-based panoramic video stitcher.
//!
//! Opens a Material dark themed window with file pickers for left/right
//! video files and calibration JSON, a GPU-rendered preview panel, and
//! play/pause/seek controls.
//!
//! ## Architecture
//!
//! Slint manages the window and UI widgets with its own renderer.
//! reco-core runs a headless wgpu pipeline for GPU stitching. A CPU
//! readback bridge (`preview::PreviewBridge`) transfers rendered RGBA
//! pixels between the two. This is a temporary architecture until Slint
//! supports wgpu 29, at which point we can share the GPU device directly.

mod playback;
mod preview;

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use reco_core::calibration::MatchCalibration;
use reco_core::pipeline::YuvPlanes;

use crate::playback::{PlayState, Playback};
use crate::preview::PreviewBridge;

slint::include_modules!();

/// Default preview viewport dimensions.
const PREVIEW_WIDTH: u32 = 1920;
const PREVIEW_HEIGHT: u32 = 1080;

/// Tick interval for the playback timer (ms).
/// Slightly faster than 30fps to avoid frame drops.
const TICK_INTERVAL_MS: i64 = 8;

/// Result sent from the calibration background thread.
type CalibrationResult = Result<MatchCalibration, String>;

/// Application state shared between Slint callbacks.
struct AppState {
    left_path: Option<PathBuf>,
    right_path: Option<PathBuf>,
    calibration_path: Option<PathBuf>,
    calibration: Option<MatchCalibration>,
    playback: Playback,
    bridge: Option<PreviewBridge>,
    /// Receives calibration results from the background thread.
    cal_rx: Option<std::sync::mpsc::Receiver<CalibrationResult>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            left_path: None,
            right_path: None,
            calibration_path: None,
            calibration: None,
            playback: Playback::new(),
            bridge: None,
            cal_rx: None,
        }
    }

    /// Check if all three files are selected and try to initialize.
    fn try_init(&mut self) -> Result<bool, String> {
        let (left, right, cal_path) =
            match (&self.left_path, &self.right_path, &self.calibration_path) {
                (Some(l), Some(r), Some(c)) => (l.clone(), r.clone(), c.clone()),
                _ => return Ok(false),
            };

        // Load calibration.
        let cal = MatchCalibration::from_file(&cal_path)
            .map_err(|e| format!("Calibration load error: {e}"))?;

        // Open video source.
        let sync_offset = cal.sync_offset;
        self.playback
            .open(&left, &right, sync_offset)
            .map_err(|e| format!("Video open error: {e}"))?;

        let (input_w, input_h) = self
            .playback
            .input_dimensions()
            .ok_or("No input dimensions")?;

        // Create GPU preview bridge.
        let bridge =
            PreviewBridge::new(cal.clone(), input_w, input_h, PREVIEW_WIDTH, PREVIEW_HEIGHT)
                .map_err(|e| format!("GPU init error: {e}"))?;

        self.calibration = Some(cal);
        self.bridge = Some(bridge);
        Ok(true)
    }

    /// Initialize preview from a calibration result (no file needed).
    fn init_with_calibration(&mut self, cal: MatchCalibration) -> Result<bool, String> {
        let (left, right) = match (&self.left_path, &self.right_path) {
            (Some(l), Some(r)) => (l.clone(), r.clone()),
            _ => return Err("Both video paths required".into()),
        };

        let sync_offset = cal.sync_offset;
        self.playback
            .open(&left, &right, sync_offset)
            .map_err(|e| format!("Video open error: {e}"))?;

        let (input_w, input_h) = self
            .playback
            .input_dimensions()
            .ok_or("No input dimensions")?;

        let bridge =
            PreviewBridge::new(cal.clone(), input_w, input_h, PREVIEW_WIDTH, PREVIEW_HEIGHT)
                .map_err(|e| format!("GPU init error: {e}"))?;

        self.calibration = Some(cal);
        self.bridge = Some(bridge);
        Ok(true)
    }

    /// Render the current frame with blocking readback (for seek/step/init).
    fn render_current_sync(&mut self) -> Option<slint::Image> {
        let frame = self.playback.current_frame()?;
        let bridge = self.bridge.as_mut()?;

        let left = YuvPlanes {
            y: &frame.left.y,
            u: &frame.left.u,
            v: &frame.left.v,
        };
        let right = YuvPlanes {
            y: &frame.right.y,
            u: &frame.right.u,
            v: &frame.right.v,
        };

        match bridge.render_frame_sync(&left, &right, 0.0, 0.0) {
            Ok(img) => Some(img),
            Err(e) => {
                log::error!("Render error: {e}");
                None
            }
        }
    }

    /// Render with double-buffered readback (for playback tick).
    /// Returns the previous frame's image (one frame behind).
    fn render_current_async(&mut self) -> Option<slint::Image> {
        let frame = self.playback.current_frame()?;
        let bridge = self.bridge.as_mut()?;

        let left = YuvPlanes {
            y: &frame.left.y,
            u: &frame.left.u,
            v: &frame.left.v,
        };
        let right = YuvPlanes {
            y: &frame.right.y,
            u: &frame.right.u,
            v: &frame.right.v,
        };

        match bridge.render_frame(&left, &right, 0.0, 0.0) {
            Ok(img) => img,
            Err(e) => {
                log::error!("Render error: {e}");
                None
            }
        }
    }
}

/// Extract just the filename from a path for display.
fn display_name(path: &std::path::Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    reco_io::init();

    let app = RecoApp::new()?;
    let state = Rc::new(RefCell::new(AppState::new()));

    // ── File picker callbacks ──

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_pick_left_video(move || {
        let dialog = rfd::FileDialog::new()
            .set_title("Select left camera video")
            .add_filter("Video", &["mp4", "mov", "avi", "mkv"]);
        if let Some(path) = dialog.pick_file() {
            let mut s = state_ref.borrow_mut();
            if let Some(app) = app_weak.upgrade() {
                app.set_left_path(display_name(&path).into());
            }
            s.left_path = Some(path);
            drop(s);
            try_init_and_update(&state_ref, &app_weak);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_pick_right_video(move || {
        let dialog = rfd::FileDialog::new()
            .set_title("Select right camera video")
            .add_filter("Video", &["mp4", "mov", "avi", "mkv"]);
        if let Some(path) = dialog.pick_file() {
            let mut s = state_ref.borrow_mut();
            if let Some(app) = app_weak.upgrade() {
                app.set_right_path(display_name(&path).into());
            }
            s.right_path = Some(path);
            drop(s);
            try_init_and_update(&state_ref, &app_weak);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_pick_calibration(move || {
        let dialog = rfd::FileDialog::new()
            .set_title("Select calibration JSON")
            .add_filter("JSON", &["json"]);
        if let Some(path) = dialog.pick_file() {
            let mut s = state_ref.borrow_mut();
            if let Some(app) = app_weak.upgrade() {
                app.set_calibration_path(display_name(&path).into());
            }
            s.calibration_path = Some(path);
            drop(s);
            try_init_and_update(&state_ref, &app_weak);
        }
    });

    // ── Auto-calibration callback ──

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_auto_calibrate(move || {
        let s = state_ref.borrow();
        let (left, right) = match (&s.left_path, &s.right_path) {
            (Some(l), Some(r)) => (l.clone(), r.clone()),
            _ => return,
        };
        drop(s);

        if let Some(app) = app_weak.upgrade() {
            app.set_calibrating(true);
            app.set_calibration_step("Starting...".into());
            app.set_status_text("Auto-calibrating...".into());
        }

        let interrupted = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();

        // Store the receiver so the UI timer can poll it.
        {
            let mut s = state_ref.borrow_mut();
            s.cal_rx = Some(rx);
        }

        // Run calibration on a background thread. Only Send types
        // (PathBuf, channel, AtomicBool, Weak) cross the boundary.
        let app_weak_bg = app_weak.clone();
        std::thread::spawn(move || {
            let app_weak_progress = app_weak_bg.clone();
            let result = reco_calibrate::video::calibrate_videos(
                &left,
                &right,
                reco_calibrate::video::CalibrateVideosOptions::default(),
                &mut |progress| {
                    let step_name = format!("{:?}", progress.step);
                    let detail = progress.detail.clone();
                    let weak = app_weak_progress.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(app) = weak.upgrade() {
                            app.set_calibration_step(step_name.into());
                            app.set_status_text(format!("Calibrating: {detail}").into());
                        }
                    })
                    .ok();
                },
                &interrupted,
            );

            let cal_result: CalibrationResult = match result {
                Ok(r) => {
                    log::info!("Auto-calibration complete: {} matches", r.total_matches,);
                    Ok(r.calibration)
                }
                Err(e) => Err(format!("{e}")),
            };
            tx.send(cal_result).ok();
        });
    });

    // ── Playback callbacks ──

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_toggle_playback(move || {
        let mut s = state_ref.borrow_mut();
        let new_state = s.playback.toggle();
        if let Some(app) = app_weak.upgrade() {
            app.set_playing(new_state == PlayState::Playing);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_step_forward(move || {
        let mut s = state_ref.borrow_mut();
        if s.playback.state() == PlayState::Playing {
            s.playback.toggle();
        }
        match s.playback.step_forward() {
            Ok(true) => {
                let img = s.render_current_sync();
                if let (Some(app), Some(img)) = (app_weak.upgrade(), img) {
                    app.set_preview_frame(img);
                    app.set_current_frame(s.playback.frame_index() as i32);
                }
            }
            Ok(false) => {}
            Err(e) => log::error!("Step forward error: {e}"),
        }
        if let Some(app) = app_weak.upgrade() {
            app.set_playing(false);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_step_backward(move || {
        let mut s = state_ref.borrow_mut();
        if s.playback.state() == PlayState::Playing {
            s.playback.toggle();
        }
        // Step back = seek to current - 1.
        let target = s.playback.frame_index().saturating_sub(2);
        let total = s.playback.total_frames().unwrap_or(1).max(1);
        let fraction = target as f32 / total as f32;
        match s.playback.seek(fraction) {
            Ok(()) => {
                let img = s.render_current_sync();
                if let (Some(app), Some(img)) = (app_weak.upgrade(), img) {
                    app.set_preview_frame(img);
                    app.set_current_frame(s.playback.frame_index() as i32);
                }
            }
            Err(e) => log::error!("Step backward error: {e}"),
        }
        if let Some(app) = app_weak.upgrade() {
            app.set_playing(false);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_seek(move |fraction| {
        let mut s = state_ref.borrow_mut();
        match s.playback.seek(fraction) {
            Ok(()) => {
                let img = s.render_current_sync();
                if let (Some(app), Some(img)) = (app_weak.upgrade(), img) {
                    app.set_preview_frame(img);
                    app.set_current_frame(s.playback.frame_index() as i32);
                }
            }
            Err(e) => log::error!("Seek error: {e}"),
        }
    });

    // ── Playback timer ──

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    let timer = slint::Timer::default();
    timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(TICK_INTERVAL_MS as u64),
        move || {
            let mut s = state_ref.borrow_mut();

            // Poll for calibration results from the background thread.
            if let Some(rx) = &s.cal_rx
                && let Ok(result) = rx.try_recv()
            {
                s.cal_rx = None;
                handle_calibration_result(result, &mut s, &app_weak);
                return;
            }

            match s.playback.tick() {
                Ok(true) => {
                    let img = s.render_current_async();
                    if let (Some(app), Some(img)) = (app_weak.upgrade(), img) {
                        app.set_preview_frame(img);
                        app.set_current_frame(s.playback.frame_index() as i32);
                        if s.playback.state() == PlayState::Finished {
                            app.set_playing(false);
                            app.set_status_text("Playback finished".into());
                        }
                    }
                }
                Ok(false) => {}
                Err(e) => {
                    log::error!("Playback tick error: {e}");
                    if let Some(app) = app_weak.upgrade() {
                        app.set_status_text(format!("Error: {e}").into());
                    }
                }
            }
        },
    );

    app.run()?;
    Ok(())
}

/// Try to initialize the pipeline when all files are selected.
fn try_init_and_update(state: &Rc<RefCell<AppState>>, app_weak: &slint::Weak<RecoApp>) {
    let mut s = state.borrow_mut();
    match s.try_init() {
        Ok(true) => {
            let fps = s.playback.fps();
            let total = s.playback.total_frames().unwrap_or(0);
            let gpu_name = s
                .bridge
                .as_ref()
                .map(|b| b.renderer().gpu().gpu_name().to_string())
                .unwrap_or_default();

            // Render the first frame.
            let img = s.render_current_sync();

            if let Some(app) = app_weak.upgrade() {
                app.set_files_loaded(true);
                app.set_total_frames(total as i32);
                app.set_current_frame(s.playback.frame_index() as i32);
                app.set_fps(fps as f32);
                app.set_status_text(
                    format!("Ready - {gpu_name} - {:.1} fps - {total} frames", fps,).into(),
                );
                if let Some(img) = img {
                    app.set_preview_frame(img);
                }
            }
        }
        Ok(false) => {
            // Not all files selected yet - update status.
            let s_ref = &*s;
            let missing: Vec<&str> = [
                s_ref.left_path.is_none().then_some("left video"),
                s_ref.right_path.is_none().then_some("right video"),
                s_ref.calibration_path.is_none().then_some("calibration"),
            ]
            .into_iter()
            .flatten()
            .collect();

            if let Some(app) = app_weak.upgrade() {
                app.set_status_text(format!("Still need: {}", missing.join(", ")).into());
            }
        }
        Err(e) => {
            log::error!("Init error: {e}");
            if let Some(app) = app_weak.upgrade() {
                app.set_status_text(format!("Error: {e}").into());
                app.set_files_loaded(false);
            }
        }
    }
}

/// Handle a calibration result from the background thread.
fn handle_calibration_result(
    result: CalibrationResult,
    state: &mut AppState,
    app_weak: &slint::Weak<RecoApp>,
) {
    if let Some(app) = app_weak.upgrade() {
        app.set_calibrating(false);
    }

    match result {
        Ok(cal) => match state.init_with_calibration(cal) {
            Ok(true) => {
                let fps = state.playback.fps();
                let total = state.playback.total_frames().unwrap_or(0);
                let gpu_name = state
                    .bridge
                    .as_ref()
                    .map(|b| b.renderer().gpu().gpu_name().to_string())
                    .unwrap_or_default();
                let img = state.render_current_sync();

                if let Some(app) = app_weak.upgrade() {
                    app.set_files_loaded(true);
                    app.set_calibration_path("(auto-calibrated)".into());
                    app.set_total_frames(total as i32);
                    app.set_current_frame(state.playback.frame_index() as i32);
                    app.set_fps(fps as f32);
                    app.set_status_text(
                        format!(
                            "Auto-calibrated - {gpu_name} - {:.1} fps - {total} frames",
                            fps,
                        )
                        .into(),
                    );
                    if let Some(img) = img {
                        app.set_preview_frame(img);
                    }
                }
            }
            Ok(false) => {}
            Err(e) => {
                log::error!("Post-calibration init: {e}");
                if let Some(app) = app_weak.upgrade() {
                    app.set_status_text(format!("Error: {e}").into());
                }
            }
        },
        Err(e) => {
            log::error!("Auto-calibration failed: {e}");
            if let Some(app) = app_weak.upgrade() {
                app.set_status_text(format!("Calibration failed: {e}").into());
            }
        }
    }
}
