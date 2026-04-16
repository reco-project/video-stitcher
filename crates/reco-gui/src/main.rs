//! Reco GUI - Slint-based panoramic video stitcher.
//!
//! Opens a Material dark themed window with file pickers for left/right
//! video files and calibration JSON, a GPU-rendered preview panel, and
//! play/pause/seek controls.
//!
//! ## Architecture
//!
//! Slint and reco-core share a single wgpu 28 device. `main()` selects
//! the wgpu 28 backend via `BackendSelector::require_wgpu_28()`, and a
//! `set_rendering_notifier` callback captures Slint's device/queue on
//! `RenderingSetup`. Those handles feed `GpuContext::from_device_queue`,
//! so reco-core renders stitched frames directly into Slint-owned
//! textures with no CPU readback.

mod playback;
mod preview;

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use reco_core::calibration::MatchCalibration;
use reco_core::pipeline::YuvPlanes;
use reco_core::wgpu;

use crate::playback::{PlayState, Playback};
use crate::preview::PreviewBridge;

/// wgpu handles captured from Slint's rendering notifier. Populated once
/// on `RenderingSetup`; used to build `PreviewBridge` when files load.
#[derive(Clone)]
struct SharedGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
}

slint::include_modules!();

/// Default preview viewport dimensions.
const PREVIEW_WIDTH: u32 = 1920;
const PREVIEW_HEIGHT: u32 = 1080;

/// Tick interval for the playback timer (ms).
///
/// Needs to be much smaller than one frame (33ms at 30fps) so the
/// drift-free scheduler in `Playback::tick` can catch up promptly when
/// a scheduled frame boundary is crossed mid-tick. 2ms gives sub-1%
/// timing error at 30fps and is cheap since the tick is a no-op when
/// no frame advance is due.
const TICK_INTERVAL_MS: i64 = 2;

/// Mouse drag sensitivity: radians per pixel of cursor movement.
/// Matches the CLI preview (`MOUSE_SENSITIVITY` in crates/reco-cli/src/preview.rs).
const MOUSE_SENSITIVITY: f32 = 0.005;

/// FOV clamp range (degrees), matching CLI preview.
const FOV_MIN: f32 = 20.0;
const FOV_MAX: f32 = 150.0;
const FOV_DEFAULT: f32 = 75.0;

/// How long the seek-slider fraction must stay stable before we
/// actually execute the seek. Debouncing is required because every
/// pixel of drag emits a `changed` event, and each seek forces a
/// NVDEC codec reinit that costs ~50ms. Without debouncing, a drag
/// saturates the GPU with hundreds of pending reinits.
const SEEK_DEBOUNCE_MS: u64 = 120;

/// Exponential smoothing factor for camera moves. Each tick, current
/// moves SMOOTHING of the way toward target. 0.25 gives a time
/// constant of ~3-4 ticks at 60Hz render rate — fast enough to track
/// input, soft enough to hide per-pixel jitter.
const CAMERA_SMOOTHING: f32 = 0.25;

/// Below this threshold (radians / degrees), current is snapped to
/// target to avoid lerping forever on float rounding error.
const CAMERA_EPSILON: f32 = 0.0005;

/// Minimum interval between renders, in ms. Caps the smoothing-driven
/// render rate so the UI thread can still service input events.
const MIN_RENDER_INTERVAL_MS: u128 = 16; // ~60Hz

/// Default number of frame pairs for auto-calibration. The reco-core
/// default is 2 which is thin for high-resolution footage; 4 gives a
/// much better bundle-adjustment fit at the cost of a few extra seconds.
const CALIBRATION_FRAMES: usize = 4;

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
    /// wgpu handles captured from Slint's rendering notifier. `None`
    /// until the window has completed its first rendering setup.
    shared_gpu: Option<SharedGpu>,
    /// Current displayed camera pose (radians). Lerped each tick
    /// toward `target_yaw`/`target_pitch`, then fed to `render_frame`.
    yaw: f32,
    pitch: f32,
    /// Target camera pose — where pan/arrow inputs want the camera to be.
    /// Current eases toward this at `CAMERA_SMOOTHING` per tick so per-
    /// pixel drag inputs produce visually continuous motion.
    target_yaw: f32,
    target_pitch: f32,
    /// Target FOV (degrees). Current FOV lives on the renderer; this is
    /// where zoom events land before being lerped in.
    target_fov: f32,
    /// Pending debounced seek: (fraction, time the request was made).
    /// The timer tick executes the seek once the fraction has stopped
    /// changing for `SEEK_DEBOUNCE_MS`.
    pending_seek: Option<(f32, Instant)>,
    /// Last time we pushed a rendered frame to Slint. Used to cap the
    /// smoothing-driven render rate.
    last_render_at: Option<Instant>,
    /// Set by control changes (blend width, rig tilt) that don't go
    /// through the camera-smoothing path but still need a re-render.
    /// Cleared by the timer after it renders.
    preview_dirty: bool,
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
            shared_gpu: None,
            yaw: 0.0,
            pitch: 0.0,
            target_yaw: 0.0,
            target_pitch: 0.0,
            target_fov: FOV_DEFAULT,
            pending_seek: None,
            last_render_at: None,
            preview_dirty: false,
        }
    }

    /// Build a PreviewBridge using the captured Slint GPU handles. Fails
    /// if the rendering notifier hasn't populated `shared_gpu` yet.
    fn build_bridge(
        &self,
        cal: &MatchCalibration,
        input_w: u32,
        input_h: u32,
    ) -> Result<PreviewBridge, String> {
        let gpu = self
            .shared_gpu
            .as_ref()
            .ok_or("GPU not ready yet (Slint rendering not initialized)")?
            .clone();
        PreviewBridge::new(
            gpu.device,
            gpu.queue,
            gpu.adapter_info,
            cal.clone(),
            input_w,
            input_h,
            PREVIEW_WIDTH,
            PREVIEW_HEIGHT,
        )
        .map_err(|e| format!("GPU init error: {e}"))
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

        let bridge = self.build_bridge(&cal, input_w, input_h)?;

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

        let bridge = self.build_bridge(&cal, input_w, input_h)?;

        self.calibration = Some(cal);
        self.bridge = Some(bridge);
        Ok(true)
    }

    /// Render the current frame. With zero-copy texture sharing, the
    /// same path works for both playback ticks and seek/step — no more
    /// sync vs async distinction.
    fn render_current(&mut self) -> Option<slint::Image> {
        let frame = self.playback.current_frame()?;
        let bridge = self.bridge.as_ref()?;

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

        match bridge.render_frame(&left, &right, self.yaw, self.pitch) {
            Ok(img) => Some(img),
            Err(e) => {
                log::error!("Render error: {e}");
                None
            }
        }
    }

    /// Apply a pixel-space pan delta — updates the *target* only. The
    /// timer tick lerps the displayed yaw/pitch toward the target, so
    /// per-pixel drag inputs produce smooth visible motion.
    fn apply_pan(&mut self, dx_px: f32, dy_px: f32) {
        self.target_yaw += dx_px * MOUSE_SENSITIVITY;
        self.target_pitch += dy_px * MOUSE_SENSITIVITY;
        self.clamp_targets();
    }

    /// Apply a FOV delta (degrees). Clamps the target; lerp handles smoothing.
    fn apply_zoom(&mut self, delta_deg: f32) {
        self.target_fov = (self.target_fov + delta_deg).clamp(FOV_MIN, FOV_MAX);
        self.clamp_targets();
    }

    /// Set FOV absolute (from the slider). Updates target; lerp applies it.
    fn set_fov(&mut self, fov_deg: f32) {
        self.target_fov = fov_deg.clamp(FOV_MIN, FOV_MAX);
        self.clamp_targets();
    }

    /// Lerp current camera (yaw/pitch/fov) one step toward the targets.
    /// Returns true if any value changed — the caller uses this to
    /// decide whether to re-render. Small residuals below the epsilon
    /// are snapped to zero so we don't lerp forever on float error.
    fn smooth_camera(&mut self) -> bool {
        let mut changed = false;

        let dy = self.target_yaw - self.yaw;
        if dy.abs() > CAMERA_EPSILON {
            self.yaw += dy * CAMERA_SMOOTHING;
            changed = true;
        } else if self.yaw != self.target_yaw {
            self.yaw = self.target_yaw;
            changed = true;
        }

        let dp = self.target_pitch - self.pitch;
        if dp.abs() > CAMERA_EPSILON {
            self.pitch += dp * CAMERA_SMOOTHING;
            changed = true;
        } else if self.pitch != self.target_pitch {
            self.pitch = self.target_pitch;
            changed = true;
        }

        if let Some(bridge) = self.bridge.as_mut() {
            let current_fov = bridge.renderer().pipeline().fov();
            let df = self.target_fov - current_fov;
            if df.abs() > CAMERA_EPSILON {
                bridge
                    .renderer_mut()
                    .pipeline_mut()
                    .set_fov(current_fov + df * CAMERA_SMOOTHING);
                changed = true;
            } else if (current_fov - self.target_fov).abs() > 0.0 {
                bridge
                    .renderer_mut()
                    .pipeline_mut()
                    .set_fov(self.target_fov);
                changed = true;
            }
        }

        changed
    }

    /// Set seam blend width. Reasonable range is 0.0 to 0.3.
    fn set_blend_width(&mut self, w: f32) {
        if let Some(bridge) = self.bridge.as_mut() {
            bridge.renderer_mut().set_blend_width(w.clamp(0.0, 0.5));
            self.preview_dirty = true;
        }
    }

    /// Set rig tilt (degrees). Reasonable range ±15°.
    fn set_rig_tilt(&mut self, deg: f32) {
        if let Some(bridge) = self.bridge.as_mut() {
            bridge.renderer_mut().set_rig_tilt(deg.to_radians());
            self.preview_dirty = true;
        }
    }

    /// Reset yaw/pitch/fov targets to defaults. The lerp will ease the
    /// currently displayed view back to zero rather than snapping.
    fn reset_view(&mut self) {
        self.target_yaw = 0.0;
        self.target_pitch = 0.0;
        self.target_fov = FOV_DEFAULT;
    }

    /// Clamp the *target* pose against the coverage boundary so pan
    /// input can't set an unreachable goal. The current pose is
    /// clamped implicitly as it lerps toward the clamped target.
    fn clamp_targets(&mut self) {
        let Some(bridge) = self.bridge.as_ref() else {
            return;
        };
        let renderer = bridge.renderer();
        let (vw, vh) = bridge.viewport_size();
        let aspect = vw as f32 / vh as f32;
        let rig_tilt = renderer.pipeline().viewport().rig_tilt;
        let clamped = renderer.coverage().safe_clamp(
            self.target_yaw,
            self.target_pitch,
            self.target_fov,
            aspect,
            rig_tilt,
        );
        self.target_yaw = clamped.yaw;
        self.target_pitch = clamped.pitch;
    }

    /// Seek by a relative number of seconds (positive = forward).
    fn seek_relative(&mut self, secs: f32) -> Result<(), String> {
        let fps = self.playback.fps();
        let total = self.playback.total_frames().unwrap_or(0).max(1);
        if fps <= 0.0 || total == 0 {
            return Ok(());
        }
        let current = self.playback.frame_index() as i64;
        let delta_frames = (secs as f64 * fps) as i64;
        let target = (current + delta_frames).clamp(0, total as i64 - 1) as u64;
        let fraction = target as f32 / total as f32;
        self.playback.seek(fraction).map_err(|e| format!("{e}"))
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

    // Select wgpu 28 as Slint's rendering backend. This MUST happen
    // before creating any window — it configures the backend for the
    // entire process. `WGPUConfiguration::default()` lets Slint create
    // its own instance/adapter/device; we capture the handles later
    // via the rendering notifier.
    slint::BackendSelector::new()
        .require_wgpu_28(slint::wgpu_28::WGPUConfiguration::default())
        .select()?;

    let app = RecoApp::new()?;
    let state = Rc::new(RefCell::new(AppState::new()));

    // Capture Slint's wgpu device and queue on RenderingSetup. These
    // are reused by PreviewBridge so reco-core's stitch output lands
    // directly in Slint-owned textures with zero copies.
    let state_for_notifier = Rc::clone(&state);
    let app_weak_notifier = app.as_weak();
    app.window()
        .set_rendering_notifier(move |rendering_state, graphics_api| {
            if !matches!(rendering_state, slint::RenderingState::RenderingSetup) {
                return;
            }
            let slint::GraphicsAPI::WGPU28 {
                instance: _,
                device,
                queue,
                ..
            } = graphics_api
            else {
                log::warn!("Expected WGPU28 GraphicsAPI in rendering notifier, got something else");
                return;
            };

            // Reconstruct adapter info by enumerating the instance. The
            // notifier doesn't expose the adapter directly, but any adapter
            // matching the device's backend will have the correct GPU name
            // for logging — the device and queue are what actually matter
            // for correctness.
            let adapter_info = wgpu::AdapterInfo {
                name: "Slint-shared wgpu 28 device".into(),
                vendor: 0,
                device: 0,
                device_pci_bus_id: String::new(),
                device_type: wgpu::DeviceType::Other,
                driver: String::new(),
                driver_info: String::new(),
                backend: wgpu::Backend::Vulkan,
                subgroup_min_size: 0,
                subgroup_max_size: 0,
                transient_saves_memory: false,
            };

            state_for_notifier.borrow_mut().shared_gpu = Some(SharedGpu {
                device: device.clone(),
                queue: queue.clone(),
                adapter_info,
            });
            log::info!("Captured Slint wgpu 28 device/queue for zero-copy preview");

            // If files were picked before the renderer was ready, the init
            // path would have failed early. Retry now that we have the GPU.
            if let Some(app) = app_weak_notifier.upgrade() {
                try_init_and_update(&state_for_notifier, &app.as_weak());
            }
        })?;

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
            // Bump frame-pair count above the reco-core default of 2.
            // More frames give the bundle adjustment more constraints
            // to settle on, which especially helps at 4K where AKAZE
            // feature matches are noisier per frame.
            let config = reco_calibrate::CalibrationConfig {
                num_frames: CALIBRATION_FRAMES,
                ..Default::default()
            };
            let result = reco_calibrate::video::calibrate_videos(
                &left,
                &right,
                reco_calibrate::video::CalibrateVideosOptions {
                    config: Some(config),
                    ..Default::default()
                },
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
                let img = s.render_current();
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
                let img = s.render_current();
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

    let state_ref = Rc::clone(&state);
    app.on_seek(move |fraction| {
        let mut s = state_ref.borrow_mut();
        // Two problems the seek slider creates, both solved by debouncing:
        //
        // 1. Every Rust-driven frame advance sets `current-frame`,
        //    which updates the slider's bound `value`, which fires
        //    `changed(val)`, which calls us. A value delta of 0 is
        //    the echo — we drop it.
        //
        // 2. A user drag emits `changed` on every mouse pixel movement.
        //    Each seek reinits NVDEC (~50ms), and hundreds per second
        //    saturate the GPU. We defer the seek until the fraction
        //    has been stable for `SEEK_DEBOUNCE_MS`, which the timer
        //    tick monitors.
        let total = match s.playback.total_frames() {
            Some(t) if t > 0 => t,
            _ => return,
        };
        let target = ((fraction as f64) * total as f64) as u64;
        if target.abs_diff(s.playback.frame_index()) < 2 {
            // Echo from our own set_current_frame — ignore.
            return;
        }
        s.pending_seek = Some((fraction, Instant::now()));
    });

    // ── Camera / view control callbacks ──
    //
    // These handlers NEVER render synchronously. They only mutate
    // targets (or cheap per-renderer params like blend/rig tilt). The
    // 2ms timer tick reads targets, lerps current toward them, and
    // renders at a capped ~60Hz. This eliminates two problems at once:
    //   1. Per-pixel drag events no longer each trigger a GPU render,
    //      so the UI thread stays responsive to input
    //   2. Pan motion is visually continuous rather than the discrete
    //      jumps from raw input events

    let state_ref = Rc::clone(&state);
    app.on_pan(move |dx_px, dy_px| {
        state_ref.borrow_mut().apply_pan(dx_px, dy_px);
    });

    let state_ref = Rc::clone(&state);
    app.on_zoom(move |delta_deg| {
        state_ref.borrow_mut().apply_zoom(delta_deg);
    });

    let state_ref = Rc::clone(&state);
    app.on_reset_view(move || {
        state_ref.borrow_mut().reset_view();
    });

    let state_ref = Rc::clone(&state);
    app.on_changed_blend_width(move |w| {
        state_ref.borrow_mut().set_blend_width(w);
    });

    let state_ref = Rc::clone(&state);
    app.on_changed_rig_tilt(move |deg| {
        state_ref.borrow_mut().set_rig_tilt(deg);
    });

    let state_ref = Rc::clone(&state);
    app.on_changed_fov(move |deg| {
        state_ref.borrow_mut().set_fov(deg);
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_seek_relative(move |secs| {
        let mut s = state_ref.borrow_mut();
        if let Err(e) = s.seek_relative(secs) {
            log::error!("Seek relative error: {e}");
            return;
        }
        let img = s.render_current();
        if let (Some(app), Some(img)) = (app_weak.upgrade(), img) {
            app.set_preview_frame(img);
            app.set_current_frame(s.playback.frame_index() as i32);
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

            // Commit a debounced seek once the requested fraction has
            // stopped changing. During drag the fraction is refreshed
            // every pixel, so the elapsed check never passes. Only
            // after the user lets go does ~120ms pass without new
            // requests, triggering one seek instead of hundreds.
            if let Some((frac, requested_at)) = s.pending_seek
                && Instant::now().duration_since(requested_at)
                    >= Duration::from_millis(SEEK_DEBOUNCE_MS)
            {
                s.pending_seek = None;
                match s.playback.seek(frac) {
                    Ok(()) => {
                        let img = s.render_current();
                        if let (Some(app), Some(img)) = (app_weak.upgrade(), img) {
                            app.set_preview_frame(img);
                            app.set_current_frame(s.playback.frame_index() as i32);
                            s.last_render_at = Some(Instant::now());
                        }
                    }
                    Err(e) => log::error!("Seek error: {e}"),
                }
                return;
            }

            // Lerp camera targets toward current and advance video if
            // due. Both can happen in the same tick; we render once at
            // the end if either produced new content. Render rate is
            // capped at ~60Hz via MIN_RENDER_INTERVAL_MS.
            let camera_changed = s.smooth_camera();
            let video_advanced = match s.playback.tick() {
                Ok(advanced) => advanced,
                Err(e) => {
                    log::error!("Playback tick error: {e}");
                    if let Some(app) = app_weak.upgrade() {
                        app.set_status_text(format!("Error: {e}").into());
                    }
                    false
                }
            };

            let was_dirty = s.preview_dirty;
            if !(camera_changed || video_advanced || was_dirty) {
                return;
            }
            s.preview_dirty = false;

            let now = Instant::now();
            let ready_to_render = s
                .last_render_at
                .is_none_or(|prev| now.duration_since(prev).as_millis() >= MIN_RENDER_INTERVAL_MS);
            if !ready_to_render {
                return;
            }

            let img = s.render_current();
            if let (Some(app), Some(img)) = (app_weak.upgrade(), img) {
                app.set_preview_frame(img);
                s.last_render_at = Some(now);
                if video_advanced {
                    app.set_current_frame(s.playback.frame_index() as i32);
                    if s.playback.state() == PlayState::Finished {
                        app.set_playing(false);
                        app.set_status_text("Playback finished".into());
                    }
                }
                // Reflect camera targets to the UI properties so
                // sliders and the reset button stay in sync with what
                // the user is actually seeing.
                app.set_yaw(s.yaw);
                app.set_pitch(s.pitch);
                if let Some(bridge) = s.bridge.as_ref() {
                    app.set_fov(bridge.renderer().pipeline().fov());
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
            let img = s.render_current();

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
                let img = state.render_current();

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
