#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

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

mod export;
mod playback;
mod preview;
mod settings;
mod telemetry_client;
mod toast;

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use reco_calibrate::{LensProfileInfo, ProfileSource};
use reco_control::pose_control::{PoseControl, PoseControlConfig};
use reco_control::{ControlIntent, IntentTranslator, PoseIntent};
use reco_core::calibration::MatchCalibration;
use reco_core::detect::director::ViewportPosition;
use reco_core::wgpu;

use crate::playback::{PlayState, Playback};
use crate::preview::PreviewBridge;
use crate::toast::{Severity, ToastManager};

/// wgpu handles captured from Slint's rendering notifier. Populated once
/// on `RenderingSetup`; used to build `PreviewBridge` when files load.
#[derive(Clone)]
struct SharedGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
}

slint::include_modules!();

/// Default preview viewport dimensions (used before the first
/// adaptive resize reads the actual preview container size from Slint).
const PREVIEW_WIDTH_DEFAULT: u32 = 1920;
const PREVIEW_HEIGHT_DEFAULT: u32 = 1080;

/// Tick interval for the playback timer (ms).
///
/// Needs to be much smaller than one frame (33ms at 30fps) so the
/// drift-free scheduler in `Playback::tick` can catch up promptly when
/// a scheduled frame boundary is crossed mid-tick. 2ms gives sub-1%
/// timing error at 30fps and is cheap since the tick is a no-op when
/// no frame advance is due.
const TICK_INTERVAL_MS: i64 = 2;

/// FOV clamp range (degrees), matching CLI preview.
const FOV_MIN: f32 = 20.0;
const FOV_MAX: f32 = 150.0;
const FOV_DEFAULT: f32 = 75.0;

/// Mouse drag sensitivity passed to `PoseControlConfig`. `0.287`
/// deg/px = 0.005 rad/px - matches the pre-migration GUI constant
/// (`MOUSE_SENSITIVITY = 0.005`) and the CLI preview's
/// `drag_deg_per_pixel`.
const DRAG_DEG_PER_PIXEL: f32 = 0.287;

/// Exponential smoothing factor passed to `PoseControlConfig`. `0.25`
/// gives a time constant of ~3-4 ticks at 60Hz render rate - fast
/// enough to track input, soft enough to hide per-pixel jitter.
/// Matches the pre-migration GUI constant `CAMERA_SMOOTHING`.
const POSE_SMOOTHING: f32 = 0.25;

/// How long the seek-slider fraction must stay stable before we
/// actually execute the seek. Debouncing is required because every
/// pixel of drag emits a `changed` event, and each seek forces a
/// NVDEC codec reinit that costs ~50ms. Without debouncing, a drag
/// saturates the GPU with hundreds of pending reinits.
const SEEK_DEBOUNCE_MS: u64 = 120;

/// Calibration payload sent from the background worker: the computed
/// match calibration plus the lens profile info each side resolved to,
/// so the GUI can tell the user "we auto-detected GoPro HERO10 Linear 4K"
/// without re-running detection.
struct CalibrationOutput {
    calibration: MatchCalibration,
    confidence: f64,
    total_matches: usize,
    left_lens_profile: Option<LensProfileInfo>,
    right_lens_profile: Option<LensProfileInfo>,
}

/// Result sent from the calibration background thread. The error is
/// the typed [`reco_calibrate::video::CalibrateVideosError`] now that
/// it is `Clone + Send + Sync` (plan step 7), so the UI thread can
/// pattern-match specific failure modes (`Cancelled`, `NoFrames`,
/// `Io(...)`, etc.) instead of parsing a stringified message.
type CalibrationResult = Result<CalibrationOutput, reco_calibrate::video::CalibrateVideosError>;

/// Application state shared between Slint callbacks.
struct AppState {
    left_path: Option<PathBuf>,
    right_path: Option<PathBuf>,
    left_input: Option<reco_io::stitch_job::InputPath>,
    right_input: Option<reco_io::stitch_job::InputPath>,
    calibration_path: Option<PathBuf>,
    calibration: Option<MatchCalibration>,
    playback: Playback,
    bridge: Option<PreviewBridge>,
    recording_tx: Option<std::sync::mpsc::SyncSender<RecordingFrame>>,
    recording_thread: Option<std::thread::JoinHandle<()>>,
    recording_path: Option<PathBuf>,
    recording_frames: u64,
    /// Receives calibration results from the background thread.
    cal_rx: Option<std::sync::mpsc::Receiver<CalibrationResult>>,
    /// wgpu handles captured from Slint's rendering notifier. `None`
    /// until the window has completed its first rendering setup.
    shared_gpu: Option<SharedGpu>,
    /// Unified pose state machine (target + current + smoothing +
    /// coverage clamping). Replaces the earlier hand-rolled
    /// `yaw/pitch/target_*` fields; all input events (drag, wheel,
    /// slider, reset) feed `PoseControl` and the render loop reads
    /// `pose.current_pose()`.
    pose: PoseControl,
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
    /// Interrupt flag for a running export. Set to true when the user
    /// clicks Cancel; StitchJob checks it between frames and aborts.
    export_interrupted: Arc<AtomicBool>,
    /// Timestamp of the last time `run_export`'s progress callback
    /// fired. Used by the playback timer to detect when the encoder is
    /// in its post-last-frame finalization phase (av_write_trailer +
    /// index flush can take ~10 seconds) so we can display "Finalizing
    /// output file…" instead of a stale frame count. Shared via Arc so
    /// the worker thread can stamp it without going through
    /// `invoke_from_event_loop`.
    export_last_progress_at: Arc<Mutex<Option<Instant>>>,
    /// Join handle for the export worker. Held so the timer can see
    /// when the export finishes (via try_recv on export_rx).
    export_thread: Option<std::thread::JoinHandle<()>>,
    /// Receives export completion notifications from the worker.
    export_rx: Option<std::sync::mpsc::Receiver<ExportOutcome>>,
    /// Original PlaneLayout values — what auto-calibrate produced. Live
    /// calibration sliders edit relative to this so Reset restores.
    cal_baseline_layout: Option<reco_core::calibration::PlaneLayout>,
    /// Persisted user preferences (recent files, default export
    /// settings, AI model path). Loaded at startup from the reco-io
    /// settings namespace and saved on any change via the convenience
    /// `push_*` methods.
    user_settings: crate::settings::GuiSettings,
    /// Last window size we persisted. Used to debounce resize saves -
    /// we only write to disk when the current size actually differs
    /// from the stored value.
    last_persisted_window_size: Option<(u32, u32)>,
    /// Last time we wrote window-size settings. Combined with the
    /// debounce threshold below to avoid thrashing disk during a
    /// drag-resize (Slint reports a new size every pixel).
    last_window_size_save_at: Option<Instant>,
    /// Baseline camera intrinsics from the last successful calibration.
    /// The Lens fine-tune sliders in the Controls panel edit these; the
    /// Reset Lens button restores them. `None` until auto-calibrate or a
    /// manual match.json load populates them.
    cal_baseline_left_params: Option<reco_core::calibration::CameraParams>,
    cal_baseline_right_params: Option<reco_core::calibration::CameraParams>,
    /// When true, `clamp_targets` pins yaw/pitch to the coverage boundary
    /// via `CoverageBoundary::safe_clamp` so the viewport never shows
    /// black margins. When false, pan/zoom is unrestricted - useful for
    /// calibration debug where the user wants to see beyond the stitched
    /// region. Bound to the Slint `use-constrained-look` checkbox.
    use_constrained_look: bool,
    /// When true, the preview shows a single camera through orthographic
    /// projection instead of the stitched panorama.
    lens_preview_active: bool,
    /// Which camera to show in lens preview mode ("left" or "right").
    lens_preview_side: String,
    /// Lens correction amount for the preview (0.0 = raw, 1.0 = full).
    lens_correction_amount: f32,
    /// Set by the ROI editor thread. Timer tick reloads calibration when true.
    roi_reload_pending: Option<Arc<AtomicBool>>,
    toasts: ToastManager,
    telemetry: Option<telemetry_client::TelemetryClient>,
}

/// Runtime AI capability summary.
///
/// Calls `reco_detect::probe_execution_providers()` to discover which
/// ONNX Runtime execution providers actually load on this machine,
/// not just which were compiled in. Replaces the old compile-time
/// `cfg!()` summary that lied when a feature was baked in but the
/// runtime libraries were missing.
///
/// Returns `(status_text, any_detector_available)`.
fn ai_capability_summary() -> (String, bool) {
    #[cfg(not(feature = "autocam"))]
    return ("AI: disabled (build without autocam feature)".into(), false);

    #[cfg(feature = "autocam")]
    {
        let probe = reco_detect::probe_execution_providers();
        if !probe.is_available() {
            return (
                format!(
                    "AI: unavailable ({})",
                    probe
                        .errors
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "no execution providers loaded".into())
                ),
                false,
            );
        }
        if probe.can_run_on_gpu_frames {
            (
                format!(
                    "AI: {} (hardware decode + inference)",
                    probe.providers.join(", ")
                ),
                true,
            )
        } else {
            (
                format!(
                    "AI: {} (CPU path - works for file export, not live GPU decode)",
                    probe.best_provider()
                ),
                true,
            )
        }
    }
}

use crate::export::ExportOutcome;

fn build_bug_report(state: &AppState, app_weak: &slint::Weak<RecoApp>) -> String {
    let gpu = state
        .bridge
        .as_ref()
        .map(|b| {
            let g = b.renderer().gpu();
            format!("{} ({:?})", g.gpu_name(), g.backend_name())
        })
        .unwrap_or_else(|| "no GPU context".into());

    let version = format!(
        "v{}{}",
        env!("CARGO_PKG_VERSION"),
        option_env!("GIT_HASH")
            .filter(|h| !h.is_empty())
            .map(|h| format!(" ({h})"))
            .unwrap_or_default()
    );

    let os = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
    let (ai_status, _) = ai_capability_summary();

    let mut report = format!(
        "## Environment\n\
         - Reco {version}\n\
         - OS: {os}\n\
         - GPU: {gpu}\n\
         - {ai_status}\n"
    );

    // Telemetry snapshot if available
    if let Some(app) = app_weak.upgrade() {
        let fps_avg = app.get_telem_fps_avg();
        let fps_recent = app.get_telem_fps_recent();
        let total_ms = app.get_telem_total_ms();
        let bottleneck = app.get_telem_bottleneck().to_string();
        if fps_avg > 0.0 || total_ms > 0.0 {
            report.push_str(&format!(
                "\n## Performance\n\
                 - FPS: {fps_avg:.0} avg / {fps_recent:.0} recent\n\
                 - Frame time: {total_ms:.1} ms\n"
            ));
            if !bottleneck.is_empty() {
                report.push_str(&format!("- Bottleneck: {bottleneck}\n"));
            }
        }

        let cal_confidence = app.get_telem_cal_confidence();
        let cal_matches = app.get_telem_cal_matches();
        if cal_matches > 0 {
            let reproj = app.get_telem_cal_reproj_err();
            report.push_str(&format!(
                "\n## Calibration\n\
                 - Confidence: {:.0}%\n\
                 - Matches: {cal_matches}\n\
                 - Reprojection error: {reproj:.4}\n",
                cal_confidence * 100.0
            ));
        }
    }

    // Redact file paths
    if let Some(cal) = &state.calibration_path {
        let name = cal.file_name().unwrap_or_default().to_string_lossy();
        report.push_str(&format!("\n## Files\n- Calibration: {name}\n"));
    }

    report.push_str(
        "\n## Description\n\
         <!-- What happened? What did you expect? -->\n\n\
         ## Steps to reproduce\n\
         <!-- 1. ... 2. ... 3. ... -->\n",
    );

    if let Some(log_path) = log_file_path()
        && let Ok(contents) = std::fs::read_to_string(&log_path)
    {
        let lines: Vec<&str> = contents.lines().collect();
        let tail = if lines.len() > 200 {
            &lines[lines.len() - 200..]
        } else {
            &lines
        };
        report.push_str("\n## Log (last 200 lines)\n```\n");
        for line in tail {
            report.push_str(line);
            report.push('\n');
        }
        report.push_str("```\n");
    }

    report
}

struct RecordingFrame {
    data: Vec<u8>,
    width: u32,
    height: u32,
    pts_us: i64,
}

impl AppState {
    fn new() -> Self {
        Self {
            left_path: None,
            right_path: None,
            left_input: None,
            right_input: None,
            calibration_path: None,
            calibration: None,
            playback: Playback::new(),
            bridge: None,
            recording_tx: None,
            recording_thread: None,
            recording_path: None,
            recording_frames: 0,
            cal_rx: None,
            shared_gpu: None,
            pose: PoseControl::new(PoseControlConfig {
                drag_deg_per_pixel: DRAG_DEG_PER_PIXEL,
                smoothing: POSE_SMOOTHING,
                fov_min_degrees: FOV_MIN,
                fov_max_degrees: FOV_MAX,
                // Pre-migration GUI: drag-right -> target_yaw +=,
                // i.e. PTZ-head convention. `invert_drag_x = true`
                // keeps that exact feel.
                invert_drag_x: true,
                rest_pose: ViewportPosition {
                    yaw: 0.0,
                    pitch: 0.0,
                    fov_degrees: Some(FOV_DEFAULT),
                },
                ..PoseControlConfig::default()
            }),
            pending_seek: None,
            last_render_at: None,
            preview_dirty: false,
            export_interrupted: Arc::new(AtomicBool::new(false)),
            export_last_progress_at: Arc::new(Mutex::new(None)),
            export_thread: None,
            export_rx: None,
            cal_baseline_layout: None,
            user_settings: {
                let mut s = crate::settings::GuiSettings::load();
                if s.telemetry_client_id.is_none() {
                    s.telemetry_client_id = Some(uuid::Uuid::new_v4().to_string());
                    s.save();
                }
                s
            },
            last_persisted_window_size: None,
            last_window_size_save_at: None,
            cal_baseline_left_params: None,
            cal_baseline_right_params: None,
            use_constrained_look: true,
            lens_preview_active: false,
            lens_preview_side: "left".into(),
            lens_correction_amount: 1.0,
            roi_reload_pending: None,
            toasts: ToastManager::default(),
            telemetry: None,
        }
    }

    fn is_exporting(&self) -> bool {
        self.export_thread.is_some()
    }

    fn reset_pipeline(&mut self) {
        self.bridge = None;
        self.playback = Playback::new();
        self.pose = PoseControl::new(PoseControlConfig {
            drag_deg_per_pixel: DRAG_DEG_PER_PIXEL,
            smoothing: POSE_SMOOTHING,
            fov_min_degrees: FOV_MIN,
            fov_max_degrees: FOV_MAX,
            invert_drag_x: true,
            rest_pose: ViewportPosition {
                yaw: 0.0,
                pitch: 0.0,
                fov_degrees: Some(FOV_DEFAULT),
            },
            ..PoseControlConfig::default()
        });
        self.pending_seek = None;
        self.last_render_at = None;
        self.preview_dirty = false;
    }

    /// Build a PreviewBridge using the captured Slint GPU handles. Fails
    /// if the rendering notifier hasn't populated `shared_gpu` yet.
    fn build_bridge(
        &mut self,
        cal: &MatchCalibration,
        input_w: u32,
        input_h: u32,
    ) -> Result<PreviewBridge, String> {
        let gpu = self
            .shared_gpu
            .as_ref()
            .ok_or("GPU not ready yet (Slint rendering not initialized)")?
            .clone();
        // Save baseline layout so Reset Calibration can restore it.
        self.cal_baseline_layout = Some(cal.layout.clone());
        PreviewBridge::new(
            gpu.device,
            gpu.queue,
            gpu.adapter_info,
            cal.clone(),
            input_w,
            input_h,
            PREVIEW_WIDTH_DEFAULT,
            PREVIEW_HEIGHT_DEFAULT,
        )
        .map_err(|e| format!("GPU init error: {e}"))
    }

    /// Apply an edited PlaneLayout to the renderer. `preview_dirty`
    /// triggers a re-render on the next timer tick.
    fn apply_layout(&mut self, layout: reco_core::calibration::PlaneLayout) {
        if let Some(cal) = self.calibration.as_mut() {
            cal.layout = layout.clone();
        }
        if let Some(bridge) = self.bridge.as_mut() {
            bridge.renderer_mut().update_layout(layout);
            self.preview_dirty = true;
        }
        self.clamp_targets();
    }

    /// Write the current (edited) calibration back to disk.
    fn save_calibration(&self) -> Result<(), String> {
        let (Some(cal), Some(path)) = (&self.calibration, &self.calibration_path) else {
            return Err("No calibration or path to save".into());
        };
        let json = serde_json::to_string_pretty(cal).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("write {}: {e}", path.display()))?;
        log::info!("Saved calibration to {}", path.display());
        Ok(())
    }

    /// Restore PlaneLayout to the values loaded at init (or after auto-cal).
    fn reset_calibration(&mut self) {
        if let Some(layout) = self.cal_baseline_layout.clone() {
            self.apply_layout(layout);
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

    /// Tear down the live pipeline so the preview stops rendering the
    /// stale source after a calibration failure or an in-place file
    /// swap. Keeps the user-picked paths on `AppState` so the user can
    /// fix and retry, but drops the bridge + playback + calibration.
    fn unload_pipeline(&mut self) {
        self.stop_recording();
        self.reset_pipeline();
        self.cal_baseline_layout = None;
        self.cal_baseline_left_params = None;
        self.cal_baseline_right_params = None;
    }

    fn start_recording(&mut self, codec: &str, quality: &str) -> Result<PathBuf, String> {
        let bridge = self.bridge.as_ref().ok_or("No pipeline")?;
        let (w, h) = bridge.viewport_size();
        let rec_w = w & !3;
        let rec_h = h & !1;
        let fps = self.playback.fps();
        let fps_r = (fps.round() as i32, 1);

        let epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let folder = self
            .user_settings
            .recording_folder
            .as_ref()
            .filter(|p| p.is_dir())
            .cloned()
            .or_else(|| {
                self.left_path
                    .as_ref()
                    .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            })
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let folder = std::fs::canonicalize(&folder).unwrap_or(folder);
        let path = folder.join(format!("reco_recording_{epoch}.mp4"));

        let (encoder, enc_name) = reco_io::adapters::create_encoder(
            &path, rec_w, rec_h, fps_r, codec, quality, None, None, None,
        )
        .map_err(|e| format!("Failed to start recording: {e}"))?;

        log::info!(
            "Recording started: {} ({enc_name}, {codec}/{quality})",
            path.display()
        );

        // Spawn encoder thread with bounded channel (4 frames deep).
        // The UI thread sends NV12 data without blocking on FFmpeg.
        let (tx, rx) = std::sync::mpsc::sync_channel::<RecordingFrame>(4);
        let handle = std::thread::spawn(move || {
            let mut encoder: Box<dyn reco_core::encoder::Encoder + Send> = Box::new(encoder);
            while let Ok(frame) = rx.recv() {
                let _ = encoder.submit(reco_core::encoder::OutputFrame {
                    data: &frame.data,
                    width: frame.width,
                    height: frame.height,
                    format: reco_core::encoder::PixelFormat::Nv12,
                    pts_us: frame.pts_us,
                });
            }
            let _ = encoder.finish();
        });

        self.recording_tx = Some(tx);
        self.recording_thread = Some(handle);
        self.recording_path = Some(path.clone());
        self.recording_frames = 0;
        Ok(path)
    }

    fn stop_recording(&mut self) -> Option<(PathBuf, u64)> {
        // Drop the sender to signal the encoder thread to finish.
        self.recording_tx = None;
        if let Some(handle) = self.recording_thread.take() {
            let _ = handle.join();
        }
        let path = self.recording_path.take()?;
        let frames = self.recording_frames;
        self.recording_frames = 0;
        log::info!("Recording stopped: {frames} frames to {}", path.display());
        Some((path, frames))
    }

    fn is_recording(&self) -> bool {
        self.recording_tx.is_some()
    }

    /// Render the current frame. With zero-copy texture sharing, the
    /// same path works for both playback ticks and seek/step — no more
    /// sync vs async distinction.
    fn render_current(&mut self) -> Option<slint::Image> {
        let frame = self.playback.current_frame()?;
        let bridge = self.bridge.as_ref()?;

        let left = frame.left.as_planes();
        let right = frame.right.as_planes();

        // Lens preview mode: render single camera flat
        if self.lens_preview_active {
            let bridge = self.bridge.as_mut()?;
            let cal = bridge.renderer().pipeline().calibration();
            let (planes, params) = if self.lens_preview_side == "right" {
                (&right, cal.right.clone())
            } else {
                (&left, cal.left.clone())
            };
            return match bridge.render_lens_preview(planes, &params, self.lens_correction_amount) {
                Ok(img) => Some(img),
                Err(e) => {
                    log::error!("Lens preview error: {e}");
                    None
                }
            };
        }

        let rig_tilt = bridge.renderer().pipeline().viewport().rig_tilt;
        let pose = self.pose.render_pose(rig_tilt);

        let recording = self.is_recording();
        if recording {
            let bridge = self.bridge.as_mut().unwrap();
            let (w, h) = bridge.viewport_size();
            let fps = self.playback.fps();

            // NV12 readback for the encoder on every frame. This is the
            // only stitch render per frame - no separate display render.
            match bridge
                .renderer_mut()
                .render_and_readback_nv12(&left, &right, pose.yaw, pose.pitch)
            {
                Ok(Some(nv12)) => {
                    if let Some(tx) = self.recording_tx.as_ref() {
                        let pts = (self.recording_frames as f64 / fps * 1_000_000.0) as i64;
                        let _ = tx.try_send(RecordingFrame {
                            data: nv12.to_vec(),
                            width: w & !3,
                            height: h & !1,
                            pts_us: pts,
                        });
                    }
                    self.recording_frames += 1;
                }
                Ok(None) => {
                    self.recording_frames += 1;
                }
                Err(e) => log::error!("NV12 readback error: {e}"),
            }

            // Display preview at reduced rate (every 5th frame).
            // The display render is cheap compared to the NV12 readback
            // but we skip most frames to keep encoding smooth.
            if self.recording_frames.is_multiple_of(5) {
                let bridge = self.bridge.as_ref().unwrap();
                match bridge.render_frame(&left, &right, pose.yaw, pose.pitch) {
                    Ok(img) => return Some(img),
                    Err(e) => log::error!("Preview render error: {e}"),
                }
            }
            None
        } else {
            match bridge.render_frame(&left, &right, pose.yaw, pose.pitch) {
                Ok(img) => Some(img),
                Err(e) => {
                    log::error!("Render error: {e}");
                    None
                }
            }
        }
    }

    /// Apply a pixel-space pan delta. Feeds `PoseControl::apply_drag`
    /// then runs the coverage clamp so the resulting target stays
    /// inside the no-black region.
    fn apply_pan(&mut self, dx_px: f32, dy_px: f32) {
        self.pose.apply_drag(dx_px, dy_px);
        self.clamp_targets();
        // Flag dirty so the playback timer requests redraws until the
        // smoothing lerp settles. Without this, when paused, the lerp
        // after the mouse is released is never run (timer sees nothing
        // to do and stops nudging Slint), so pan motion snaps/stalls.
        self.preview_dirty = true;
    }

    /// Apply a FOV delta (degrees). Clamps the target; tick handles smoothing.
    fn apply_zoom(&mut self, delta_deg: f32) {
        IntentTranslator::new(&mut self.pose)
            .dispatch(ControlIntent::Pose(PoseIntent::DeltaFovDeg(delta_deg)));
        self.clamp_targets();
        self.preview_dirty = true;
    }

    /// Set FOV absolute (from the slider). Updates target; tick applies it.
    fn set_fov(&mut self, fov_deg: f32) {
        IntentTranslator::new(&mut self.pose)
            .dispatch(ControlIntent::Pose(PoseIntent::SetFovDeg(fov_deg)));
        self.clamp_targets();
        self.preview_dirty = true;
    }

    /// Advance the PoseControl one smoothing step and push the
    /// resulting FOV back to the renderer pipeline. Returns `true`
    /// when the pose changed measurably (caller uses this to decide
    /// whether to re-render).
    fn smooth_camera(&mut self) -> bool {
        let before = self.pose.current_pose();
        self.pose.tick();
        if self.use_constrained_look
            && let Some(bridge) = self.bridge.as_ref()
        {
            let renderer = bridge.renderer();
            let (vw, vh) = bridge.viewport_size();
            let aspect = vw as f32 / vh as f32;
            let rig_tilt = renderer.pipeline().viewport().rig_tilt;
            self.pose
                .clamp_via_coverage(renderer.coverage(), aspect, rig_tilt);
        }
        let after = self.pose.current_pose();

        let yaw_changed = (before.yaw - after.yaw).abs() > f32::EPSILON;
        let pitch_changed = (before.pitch - after.pitch).abs() > f32::EPSILON;
        let fov_changed = before.fov_degrees != after.fov_degrees;

        if fov_changed
            && let Some(fov) = after.fov_degrees
            && let Some(bridge) = self.bridge.as_mut()
        {
            bridge.renderer_mut().pipeline_mut().set_fov(fov);
        }

        yaw_changed || pitch_changed || fov_changed
    }

    /// Set seam blend width. Reasonable range is 0.0 to 0.3.
    fn set_blend_width(&mut self, w: f32) {
        if let Some(bridge) = self.bridge.as_mut() {
            bridge.renderer_mut().set_blend_width(w.clamp(0.0, 0.5));
            self.preview_dirty = true;
        }
    }

    fn set_rig_tilt(&mut self, deg: f32) {
        if let Some(cal) = self.calibration.as_mut() {
            cal.rig_tilt = deg as f64;
        }
        if let Some(bridge) = self.bridge.as_mut() {
            bridge.renderer_mut().set_rig_tilt(deg.to_radians());
            self.preview_dirty = true;
        }
        self.clamp_targets();
    }

    fn set_sync_offset(&mut self, frames: i32) {
        let offset = frames as i64;
        if let Some(cal) = self.calibration.as_mut() {
            cal.sync_offset = offset;
        }
        if let (Some(left), Some(right)) = (&self.left_path, &self.right_path) {
            let left = left.clone();
            let right = right.clone();
            if let Err(e) = self.playback.open(&left, &right, offset) {
                log::error!("Failed to reopen playback with sync offset {offset}: {e}");
                return;
            }
            log::info!("Sync offset changed to {offset} frames");
            self.preview_dirty = true;
        }
    }

    fn set_rig_roll(&mut self, deg: f32) {
        if let Some(cal) = self.calibration.as_mut() {
            cal.rig_roll = deg as f64;
        }
        if let Some(bridge) = self.bridge.as_mut() {
            bridge.renderer_mut().set_rig_roll(deg.to_radians());
            self.preview_dirty = true;
        }
        self.clamp_targets();
    }

    /// Reset yaw/pitch/fov targets to the rest pose. Routes through
    /// the translator so the same intent path works for both Slint
    /// callbacks and future remote transports.
    fn reset_view(&mut self) {
        IntentTranslator::new(&mut self.pose).dispatch(ControlIntent::Pose(PoseIntent::Reset));
    }

    /// Clamp the pose through the coverage boundary so pan input
    /// cannot set an unreachable goal. Delegates to
    /// `PoseControl::clamp_via_coverage`.
    fn clamp_targets(&mut self) {
        if !self.use_constrained_look {
            return;
        }
        let Some(bridge) = self.bridge.as_ref() else {
            return;
        };
        let renderer = bridge.renderer();
        let (vw, vh) = bridge.viewport_size();
        let aspect = vw as f32 / vh as f32;
        let rig_tilt = renderer.pipeline().viewport().rig_tilt;
        self.pose
            .clamp_via_coverage(renderer.coverage(), aspect, rig_tilt);
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
/// Seed the Slint lens-tune sliders and their display ranges from a
/// pair of baseline `CameraParams`. Called after auto-calibrate completes
/// and on Reset Lens. Ranges are chosen wide enough for meaningful
/// manual tuning (fx/fy: +/-15%, cx/cy: +/-10% of image dim) but tight
/// enough that the slider granularity is useful.
fn set_lens_sliders(
    app: &RecoApp,
    left: &reco_core::calibration::CameraParams,
    right: &reco_core::calibration::CameraParams,
) {
    // Ranges are computed from the left camera's baseline. In stereo
    // rigs the two lenses are typically matched models, so a single
    // range keeps the UI simpler. If the cameras ever differ materially
    // this can be revisited.
    let f_baseline = left.fx.max(left.fy);
    let fx_span = (f_baseline * 0.15).max(5.0);
    let w = left.width.max(1) as f64;
    let h = left.height.max(1) as f64;
    let cx_span = (w * 0.10).max(5.0);
    let cy_span = (h * 0.10).max(5.0);

    app.set_lens_fx_min((left.fx - fx_span) as f32);
    app.set_lens_fx_max((left.fx + fx_span) as f32);
    app.set_lens_fy_min((left.fy - fx_span) as f32);
    app.set_lens_fy_max((left.fy + fx_span) as f32);
    app.set_lens_cx_min((left.cx - cx_span) as f32);
    app.set_lens_cx_max((left.cx + cx_span) as f32);
    app.set_lens_cy_min((left.cy - cy_span) as f32);
    app.set_lens_cy_max((left.cy + cy_span) as f32);
    app.set_lens_k_range(0.3);

    app.set_lens_left_fx(left.fx as f32);
    app.set_lens_left_fy(left.fy as f32);
    app.set_lens_left_cx(left.cx as f32);
    app.set_lens_left_cy(left.cy as f32);
    app.set_lens_left_k1(left.d[0] as f32);
    app.set_lens_left_k2(left.d[1] as f32);
    app.set_lens_left_k3(left.d[2] as f32);
    app.set_lens_left_k4(left.d[3] as f32);

    app.set_lens_right_fx(right.fx as f32);
    app.set_lens_right_fy(right.fy as f32);
    app.set_lens_right_cx(right.cx as f32);
    app.set_lens_right_cy(right.cy as f32);
    app.set_lens_right_k1(right.d[0] as f32);
    app.set_lens_right_k2(right.d[1] as f32);
    app.set_lens_right_k3(right.d[2] as f32);
    app.set_lens_right_k4(right.d[3] as f32);
}

/// Human-readable description of how a lens profile was resolved.
fn profile_source_label(info: &LensProfileInfo) -> &'static str {
    match info.source {
        ProfileSource::AutoDetected => "Auto-detected",
        ProfileSource::Database => "Database match",
        ProfileSource::File(_) => "File",
        ProfileSource::Fallback => "Fallback",
    }
}

/// Populate the Slint lens-profile properties from calibration output.
///
/// Stamps the detected camera/lens/source for left and right, plus the
/// count of alternate profiles in the embedded database that match the
/// current video resolution (`in_w` x `in_h`). The candidate count lets
/// the user tell at a glance whether they could reasonably override the
/// auto-detected profile - zero means the picker has nothing new to offer.
fn set_lens_profile_props(
    app: &RecoApp,
    left: Option<LensProfileInfo>,
    right: Option<LensProfileInfo>,
    in_w: u32,
    in_h: u32,
) {
    if let Some(info) = &left {
        app.set_lens_left_camera(info.camera.clone().into());
        app.set_lens_left_lens(info.lens.clone().into());
        app.set_lens_left_source(profile_source_label(info).into());
    } else {
        app.set_lens_left_camera("Unknown".into());
        app.set_lens_left_lens("".into());
        app.set_lens_left_source("Not detected".into());
    }
    if let Some(info) = &right {
        app.set_lens_right_camera(info.camera.clone().into());
        app.set_lens_right_lens(info.lens.clone().into());
        app.set_lens_right_source(profile_source_label(info).into());
    } else {
        app.set_lens_right_camera("Unknown".into());
        app.set_lens_right_lens("".into());
        app.set_lens_right_source("Not detected".into());
    }

    // Count candidate profiles for the current resolution so the user
    // sees whether alternates exist. Loading the embedded database is
    // O(1) after the first call (static OnceCell inside reco-calibrate).
    let candidates = if in_w > 0 && in_h > 0 {
        let db = reco_calibrate::lens_database::LensDatabase::load_embedded();
        db.candidates(in_w, in_h).len() as i32
    } else {
        0
    };
    app.set_lens_candidates_count(candidates);
    app.set_lens_info_available(left.is_some() || right.is_some());
}

fn format_time(frame: u64, fps: f64) -> String {
    if fps <= 0.0 {
        return "0:00".into();
    }
    let total_secs = (frame as f64 / fps) as u64;
    let m = total_secs / 60;
    let s = total_secs % 60;
    format!("{m}:{s:02}")
}

fn sync_frame_display(app: &RecoApp, frame: u64, total: u64, fps: f64) {
    app.set_current_frame(frame as i32);
    app.set_total_frames(total as i32);
    app.set_current_time_text(format_time(frame, fps).into());
    app.set_total_time_text(format_time(total, fps).into());
}

fn display_name(path: &std::path::Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Push the current MRU lists into the Slint properties that back the
/// Recent-files dialog. Called at startup and after every file pick.
fn sync_recent_paths(settings: &settings::GuiSettings, app: &RecoApp) {
    fn to_model(paths: &[std::path::PathBuf]) -> slint::ModelRc<slint::SharedString> {
        let v: Vec<slint::SharedString> = paths
            .iter()
            .map(|p| slint::SharedString::from(p.to_string_lossy().as_ref()))
            .collect();
        slint::ModelRc::new(slint::VecModel::from(v))
    }
    app.set_recent_left_paths(to_model(settings.recent_left.entries()));
    app.set_recent_right_paths(to_model(settings.recent_right.entries()));
    app.set_recent_calibration_paths(to_model(settings.recent_calibration.entries()));
}

/// Push the per-side segment filenames into the Slint left/right-segments
/// models so the Files panel shows what was imported.
fn sync_segments(state: &AppState, app: &RecoApp) {
    fn input_to_names(input: &Option<reco_io::stitch_job::InputPath>) -> Vec<slint::SharedString> {
        match input {
            Some(reco_io::stitch_job::InputPath::Single(p)) => {
                vec![display_name(p).into()]
            }
            Some(reco_io::stitch_job::InputPath::Chained(ps)) => {
                ps.iter().map(|p| display_name(p).into()).collect()
            }
            None => vec![],
        }
    }
    app.set_left_segments(slint::ModelRc::new(slint::VecModel::from(input_to_names(
        &state.left_input,
    ))));
    app.set_right_segments(slint::ModelRc::new(slint::VecModel::from(input_to_names(
        &state.right_input,
    ))));
}

fn sync_roi_points(state: &AppState, app: &RecoApp) {
    let (xs, ys) = if let Some(cal) = &state.calibration
        && let Some(roi) = &cal.field_roi
    {
        let side = if state.lens_preview_side == "right" {
            &roi.right
        } else {
            &roi.left
        };
        let xs: Vec<f32> = side.iter().map(|p| p[0] as f32).collect();
        let ys: Vec<f32> = side.iter().map(|p| p[1] as f32).collect();
        (xs, ys)
    } else {
        (vec![], vec![])
    };
    app.set_roi_points_x(slint::ModelRc::new(slint::VecModel::from(xs)));
    app.set_roi_points_y(slint::ModelRc::new(slint::VecModel::from(ys)));
}

/// Install the standard tracing subscriber + log bridge.
///
/// Replaces the previous `env_logger::init()`. Bridges `log::*` calls
/// from reco-core / reco-io / reco-calibrate into tracing so user bug
/// reports arrive as one structured event stream instead of two
/// loggers writing to the same stderr.
///
/// M2 migration (deep-review-2026-04-18 decision 11).
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let _ = tracing_log::LogTracer::init();
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // In release builds, write logs to a file so bug reports have context.
    // Debug builds just use stderr.
    #[cfg(not(debug_assertions))]
    if let Some(log_path) = log_file_path() {
        if let Some(parent) = log_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(file) = std::fs::File::create(&log_path) {
            // Windows: file only (stderr is detached by windows_subsystem="windows").
            // Mac/Linux: file + stderr (user may launch from terminal).
            #[cfg(target_os = "windows")]
            {
                let _ = tracing_subscriber::registry()
                    .with(filter)
                    .with(
                        fmt::layer()
                            .with_target(true)
                            .with_level(true)
                            .with_ansi(false)
                            .with_writer(file),
                    )
                    .try_init();
                eprintln!("Log file: {}", log_path.display());
                return;
            }
            #[cfg(not(target_os = "windows"))]
            {
                let file = std::sync::Mutex::new(file);
                let _ = tracing_subscriber::registry()
                    .with(filter)
                    .with(
                        fmt::layer()
                            .with_target(true)
                            .with_level(true)
                            .with_ansi(false)
                            .with_writer(file),
                    )
                    .with(
                        fmt::layer()
                            .with_target(true)
                            .with_level(true)
                            .with_writer(std::io::stderr),
                    )
                    .try_init();
                eprintln!("Log file: {}", log_path.display());
                return;
            }
        }
    }

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(true).with_level(true))
        .try_init();
}

/// Platform-appropriate log file path.
///
/// - Windows: next to executable (`reco-gui.log`)
/// - macOS: `~/Library/Logs/reco-gui.log`
/// - Linux: `~/.config/reco/reco-gui.log`
fn log_file_path() -> Option<std::path::PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("reco-gui.log")))
    }
    #[cfg(target_os = "macos")]
    {
        std::env::var("HOME")
            .ok()
            .map(|h| std::path::PathBuf::from(h).join("Library/Logs/reco-gui.log"))
    }
    #[cfg(target_os = "linux")]
    {
        std::env::var("XDG_CONFIG_HOME")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| std::path::PathBuf::from(h).join(".config"))
            })
            .map(|d| d.join("reco/reco-gui.log"))
    }
}

/// Panic hook: emit panic location + payload as a `tracing::error!`
/// before the default hook runs, so a user-reported log file contains
/// the panic context alongside surrounding events.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".into());
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".into()
        };
        tracing::error!(
            target: "panic",
            location = %location,
            payload = %payload,
            "panic caught by tracing panic hook"
        );
        default_hook(info);
    }));
}

fn main() -> anyhow::Result<()> {
    init_tracing();
    install_panic_hook();
    reco_io::init();

    // Select wgpu 28 as Slint's rendering backend. This MUST happen
    // before creating any window. femtovg-wgpu renders UI through
    // wgpu (DX12/Vulkan) instead of OpenGL, so it works on GPUs
    // that lack an OpenGL driver. downlevel_defaults() ensures iGPUs
    // and older hardware can satisfy the device limits.
    slint::BackendSelector::new()
        .require_wgpu_28({
            let mut config = slint::wgpu_28::WGPUConfiguration::default();
            if let slint::wgpu_28::WGPUConfiguration::Automatic(ref mut settings) = config {
                settings.device_required_limits = reco_core::wgpu::Limits::downlevel_defaults();
            }
            config
        })
        .select()?;

    let app = RecoApp::new()?;
    let state = Rc::new(RefCell::new(AppState::new()));

    // Initialize opt-in telemetry if the user enabled it.
    {
        let mut s = state.borrow_mut();
        if s.user_settings.telemetry_enabled {
            let cid = s
                .user_settings
                .telemetry_client_id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            log::info!("Telemetry enabled (client_id={}).", &cid[..8]);
            let client = telemetry_client::TelemetryClient::new(cid);
            client.app_open();
            s.telemetry = Some(client);
        } else {
            log::info!("Telemetry disabled (opt-in via Preferences).");
        }
    }

    let (ai_status, ai_available) = ai_capability_summary();
    log::info!("{ai_status}");
    app.set_ai_status(ai_status.clone().into());
    app.set_ai_available(ai_available);

    // Send context telemetry after AI probe.
    {
        let s = state.borrow();
        if let Some(ref t) = s.telemetry {
            let os = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
            t.context("(pending GPU init)", &os, &ai_status);
        }
    }

    let version = format!(
        "v{}{}",
        env!("CARGO_PKG_VERSION"),
        option_env!("GIT_HASH")
            .filter(|h| !h.is_empty())
            .map(|h| format!(" ({h})"))
            .unwrap_or_default()
    );
    log::info!("Reco GUI {version}");
    app.set_version(version.into());

    {
        let s = state.borrow();
        app.set_dark_mode(s.user_settings.dark_mode);
    }

    // Check for updates in the background.
    // Stores result in an Arc<Mutex> that the timer tick reads once.
    let update_result: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    {
        let result = Arc::clone(&update_result);
        std::thread::spawn(move || {
            let current = env!("CARGO_PKG_VERSION");
            let resp = ureq::get(
                "https://api.github.com/repos/reco-project/video-stitcher/releases/latest",
            )
            .header("User-Agent", "reco-gui")
            .call();
            if let Ok(mut resp) = resp
                && let Ok(body) = resp.body_mut().read_to_string()
                && let Ok(json) = serde_json::from_str::<serde_json::Value>(&body)
                && let Some(tag) = json["tag_name"].as_str()
            {
                let latest = tag.trim_start_matches('v');
                let parse_ver = |s: &str| -> Option<Vec<u64>> {
                    s.split(&['.', '-'][..])
                        .take(3)
                        .map(|p| p.parse().ok())
                        .collect()
                };
                let is_newer = parse_ver(latest)
                    .zip(parse_ver(current))
                    .is_some_and(|(l, c)| l > c);
                if is_newer {
                    log::info!("Update available: {current} -> {latest}");
                    *result.lock().unwrap() = Some(tag.to_string());
                }
            }
        });
    }

    let mut codecs: Vec<slint::SharedString> = Vec::new();
    for (label, codec) in [
        ("h264", reco_io::ffmpeg::encoder::VideoCodec::H264),
        ("hevc", reco_io::ffmpeg::encoder::VideoCodec::Hevc),
        ("av1", reco_io::ffmpeg::encoder::VideoCodec::Av1),
    ] {
        if !reco_io::ffmpeg::encoder::available_encoders(codec).is_empty() {
            codecs.push(label.into());
        }
    }
    if codecs.is_empty() {
        log::warn!("No video encoders available - export will fail");
        codecs.push("h264".into());
    } else {
        log::info!(
            "Available codecs: {}",
            codecs
                .iter()
                .map(|c| c.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    app.set_available_codecs(slint::ModelRc::new(slint::VecModel::from(codecs)));

    // Seed recording and preview settings from persisted preferences.
    {
        let s = state.borrow();
        app.set_recording_codec(s.user_settings.recording_codec.clone().into());
        app.set_recording_quality(s.user_settings.recording_quality.clone().into());
        app.set_recording_folder(
            s.user_settings
                .recording_folder
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default()
                .into(),
        );
        app.set_preview_aspect(s.user_settings.preview_aspect.clone().into());
    }

    // Seed the Recent-files dialog with the persisted MRU lists. If
    // the user never loaded anything before, these are empty and the
    // Recent button in the file bar stays disabled.
    sync_recent_paths(&state.borrow().user_settings, &app);

    // Restore last window size if the user resized before. Slint's
    // `set_size` takes a `PhysicalSize`; we stored logical dimensions
    // in settings but using them as physical is close enough at 1.0
    // scale (the common case) - if the user moves to a HiDPI display
    // the next resize will correct.
    if let Some((w, h)) = state.borrow().user_settings.window_size
        && w > 0
        && h > 0
    {
        app.window()
            .set_size(slint::LogicalSize::new(w as f32, h as f32));
    }

    // Capture Slint's wgpu device and queue on RenderingSetup. These
    // are reused by PreviewBridge so reco-core's stitch output lands
    // directly in Slint-owned textures with zero copies.
    let state_for_notifier = Rc::clone(&state);
    let app_weak_notifier = app.as_weak();
    app.window()
        .set_rendering_notifier(move |rendering_state, graphics_api| {
            match rendering_state {
                slint::RenderingState::RenderingSetup => {
                    let slint::GraphicsAPI::WGPU28 {
                        instance: _,
                        device,
                        queue,
                        ..
                    } = graphics_api
                    else {
                        log::warn!(
                            "Expected WGPU28 GraphicsAPI in rendering notifier, got something else"
                        );
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
                }
                slint::RenderingState::BeforeRendering => {
                    // Vsync-locked playback tick. Previously this ran off a
                    // 2 ms free-running timer, which put set_preview_frame
                    // calls at random phases relative to Slint's 60 Hz
                    // compositor. Small (~1 ms) submission jitter around
                    // the 33 ms video interval crossed vsync boundaries
                    // unpredictably, so individual frames displayed for
                    // 1, 2, or 3 vsync slots at random, producing visible
                    // judder perceived as ~25 fps. Driving from here
                    // phase-locks everything to the compositor.
                    vsync_render_tick(&state_for_notifier, &app_weak_notifier);
                }
                _ => {}
            }
        })?;

    // ── File picker callbacks ──

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_pick_left_video(move || {
        let dialog = rfd::FileDialog::new()
            .set_title("Select left camera video(s)")
            .add_filter(
                "Video",
                &["mp4", "MP4", "mov", "MOV", "avi", "AVI", "mkv", "MKV"],
            );
        let mut paths = dialog.pick_files().unwrap_or_default();
        if paths.is_empty() {
            return;
        }
        paths.sort();
        let first = paths[0].clone();
        let input = if paths.len() == 1 {
            reco_io::stitch_job::InputPath::Single(paths.into_iter().next().unwrap())
        } else {
            log::info!(
                "Left: {} segments selected, chaining via concat demuxer",
                paths.len()
            );
            reco_io::stitch_job::InputPath::Chained(paths)
        };
        {
            let mut s = state_ref.borrow_mut();
            let changed = s.left_path.as_ref() != Some(&first);
            if changed && s.bridge.is_some() {
                s.unload_pipeline();
                if let Some(app) = app_weak.upgrade() {
                    app.set_files_loaded(false);
                    app.set_status_text("File changed - re-calibrate or load calibration".into());
                }
            }
            if let Some(app) = app_weak.upgrade() {
                let label = match &input {
                    reco_io::stitch_job::InputPath::Single(p) => display_name(p),
                    reco_io::stitch_job::InputPath::Chained(ps) => {
                        format!("{} ({} segments)", display_name(&ps[0]), ps.len())
                    }
                };
                app.set_left_path(label.into());
            }
            s.user_settings.push_left(first.clone());
            if let Some(app) = app_weak.upgrade() {
                sync_recent_paths(&s.user_settings, &app);
            }
            s.left_input = Some(input);
            s.left_path = Some(first);
            drop(s);
            try_init_and_update(&state_ref, &app_weak);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_pick_right_video(move || {
        let dialog = rfd::FileDialog::new()
            .set_title("Select right camera video(s)")
            .add_filter(
                "Video",
                &["mp4", "MP4", "mov", "MOV", "avi", "AVI", "mkv", "MKV"],
            );
        let mut paths = dialog.pick_files().unwrap_or_default();
        if paths.is_empty() {
            return;
        }
        paths.sort();
        let first = paths[0].clone();
        let input = if paths.len() == 1 {
            reco_io::stitch_job::InputPath::Single(paths.into_iter().next().unwrap())
        } else {
            log::info!(
                "Right: {} segments selected, chaining via concat demuxer",
                paths.len()
            );
            reco_io::stitch_job::InputPath::Chained(paths)
        };
        {
            let mut s = state_ref.borrow_mut();
            let changed = s.right_path.as_ref() != Some(&first);
            if changed && s.bridge.is_some() {
                s.unload_pipeline();
                if let Some(app) = app_weak.upgrade() {
                    app.set_files_loaded(false);
                    app.set_status_text("File changed - re-calibrate or load calibration".into());
                }
            }
            if let Some(app) = app_weak.upgrade() {
                let label = match &input {
                    reco_io::stitch_job::InputPath::Single(p) => display_name(p),
                    reco_io::stitch_job::InputPath::Chained(ps) => {
                        format!("{} ({} segments)", display_name(&ps[0]), ps.len())
                    }
                };
                app.set_right_path(label.into());
            }
            s.user_settings.push_right(first.clone());
            if let Some(app) = app_weak.upgrade() {
                sync_recent_paths(&s.user_settings, &app);
            }
            s.right_input = Some(input);
            s.right_path = Some(first);
            drop(s);
            try_init_and_update(&state_ref, &app_weak);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_pick_calibration(move || {
        let dialog = rfd::FileDialog::new()
            .set_title("Select calibration JSON")
            .add_filter("JSON", &["json", "JSON"]);
        if let Some(path) = dialog.pick_file() {
            let mut s = state_ref.borrow_mut();
            let changed = s.calibration_path.as_ref() != Some(&path);
            if changed && s.bridge.is_some() {
                s.unload_pipeline();
                if let Some(app) = app_weak.upgrade() {
                    app.set_files_loaded(false);
                    app.set_status_text("Calibration changed — reloading".into());
                }
            }
            if let Some(app) = app_weak.upgrade() {
                app.set_calibration_path(display_name(&path).into());
            }
            s.user_settings.push_calibration(path.clone());
            if let Some(app) = app_weak.upgrade() {
                sync_recent_paths(&s.user_settings, &app);
            }
            s.calibration_path = Some(path);
            drop(s);
            try_init_and_update(&state_ref, &app_weak);
        }
    });

    // ── Recent-files dialog callbacks ──
    //
    // Clicking an entry in the dialog is functionally equivalent to
    // picking that file via the native dialog: update the MRU (so it
    // moves to front), push to the Slint label property, and try to
    // initialize if all three slots are now filled.
    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_load_recent_left(move |entry| {
        let path = PathBuf::from(entry.as_str());
        let mut s = state_ref.borrow_mut();
        let changed = s.left_path.as_ref() != Some(&path);
        if changed && s.bridge.is_some() {
            s.unload_pipeline();
            if let Some(app) = app_weak.upgrade() {
                app.set_files_loaded(false);
                app.set_status_text("File changed — re-calibrate or load calibration".into());
            }
        }
        if let Some(app) = app_weak.upgrade() {
            app.set_left_path(display_name(&path).into());
        }
        s.user_settings.push_left(path.clone());
        if let Some(app) = app_weak.upgrade() {
            sync_recent_paths(&s.user_settings, &app);
        }
        s.left_path = Some(path);
        drop(s);
        try_init_and_update(&state_ref, &app_weak);
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_load_recent_right(move |entry| {
        let path = PathBuf::from(entry.as_str());
        let mut s = state_ref.borrow_mut();
        let changed = s.right_path.as_ref() != Some(&path);
        if changed && s.bridge.is_some() {
            s.unload_pipeline();
            if let Some(app) = app_weak.upgrade() {
                app.set_files_loaded(false);
                app.set_status_text("File changed — re-calibrate or load calibration".into());
            }
        }
        if let Some(app) = app_weak.upgrade() {
            app.set_right_path(display_name(&path).into());
        }
        s.user_settings.push_right(path.clone());
        if let Some(app) = app_weak.upgrade() {
            sync_recent_paths(&s.user_settings, &app);
        }
        s.right_path = Some(path);
        drop(s);
        try_init_and_update(&state_ref, &app_weak);
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_load_recent_calibration(move |entry| {
        let path = PathBuf::from(entry.as_str());
        let mut s = state_ref.borrow_mut();
        let changed = s.calibration_path.as_ref() != Some(&path);
        if changed && s.bridge.is_some() {
            s.unload_pipeline();
            if let Some(app) = app_weak.upgrade() {
                app.set_files_loaded(false);
                app.set_status_text("Calibration changed — reloading".into());
            }
        }
        if let Some(app) = app_weak.upgrade() {
            app.set_calibration_path(display_name(&path).into());
        }
        s.user_settings.push_calibration(path.clone());
        if let Some(app) = app_weak.upgrade() {
            sync_recent_paths(&s.user_settings, &app);
        }
        s.calibration_path = Some(path);
        drop(s);
        try_init_and_update(&state_ref, &app_weak);
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_clear_recent_files(move || {
        let mut s = state_ref.borrow_mut();
        s.user_settings.recent_left.clear();
        s.user_settings.recent_right.clear();
        s.user_settings.recent_calibration.clear();
        s.user_settings.save();
        if let Some(app) = app_weak.upgrade() {
            sync_recent_paths(&s.user_settings, &app);
        }
    });

    // ── File management callbacks (left panel) ──

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_clear_left(move || {
        let mut s = state_ref.borrow_mut();
        if s.bridge.is_some() {
            s.unload_pipeline();
        }
        s.left_path = None;
        s.left_input = None;
        if let Some(app) = app_weak.upgrade() {
            app.set_left_path("".into());
            app.set_files_loaded(false);
            app.set_status_text("Left video cleared. Calibration preserved.".into());
            sync_segments(&s, &app);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_clear_right(move || {
        let mut s = state_ref.borrow_mut();
        if s.bridge.is_some() {
            s.unload_pipeline();
        }
        s.right_path = None;
        s.right_input = None;
        if let Some(app) = app_weak.upgrade() {
            app.set_right_path("".into());
            app.set_files_loaded(false);
            app.set_status_text("Right video cleared. Calibration preserved.".into());
            sync_segments(&s, &app);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_clear_calibration(move || {
        let mut s = state_ref.borrow_mut();
        if s.bridge.is_some() {
            s.unload_pipeline();
        }
        s.calibration_path = None;
        s.calibration = None;
        if let Some(app) = app_weak.upgrade() {
            app.set_calibration_path("".into());
            app.set_files_loaded(false);
            app.set_status_text("Calibration cleared".into());
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_launch_roi_editor(move || {
        let paths = {
            let s = state_ref.borrow();
            match (
                s.left_path.clone(),
                s.right_path.clone(),
                s.calibration_path.clone(),
            ) {
                (Some(l), Some(r), Some(c)) => Some((l, r, c)),
                _ => None,
            }
        };
        let Some((left, right, cal_path)) = paths else {
            let mut s = state_ref.borrow_mut();
            if let Some(app) = app_weak.upgrade() {
                s.toasts.push(
                    crate::toast::Severity::Error,
                    "Cannot set ROI",
                    "Need left video, right video, and calibration loaded.",
                );
                crate::toast::sync_to_ui(&s.toasts, &app);
            }
            return;
        };

        // Use $HOME/.cache/reco instead of /tmp because snap-packaged
        // browsers (Firefox) can't access /tmp due to sandboxing.
        let cache_base = std::env::var("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .unwrap_or_else(|_| std::env::temp_dir());
        let tmp_dir = cache_base.join("reco").join("roi");
        if let Err(e) = std::fs::create_dir_all(&tmp_dir) {
            log::error!("Failed to create ROI temp dir {}: {e}", tmp_dir.display());
            let mut s = state_ref.borrow_mut();
            if let Some(app) = app_weak.upgrade() {
                s.toasts.push(
                    crate::toast::Severity::Error,
                    "ROI editor",
                    format!("Cannot create temp dir: {e}"),
                );
                crate::toast::sync_to_ui(&s.toasts, &app);
            }
            return;
        }
        let left_png = tmp_dir.join("left.png");
        let right_png = tmp_dir.join("right.png");

        let current_secs = {
            let s = state_ref.borrow();
            if s.playback.fps() > 0.0 {
                s.playback.frame_index() as f64 / s.playback.fps()
            } else {
                1.0
            }
        };
        let _seek_str = format!("{:.2}", current_secs);
        let frame_index = {
            let s = state_ref.borrow();
            s.playback.frame_index()
        };
        let extract_frame = |video: &std::path::Path, out: &std::path::Path, idx: u64| {
            match reco_io::ffmpeg::calibration_io::extract_frames(video, &[idx]) {
                Ok(frames) if !frames.is_empty() => {
                    let yuv = &frames[0];
                    let w = yuv.width as usize;
                    let h = yuv.height as usize;
                    let uv_w = w / 2;
                    let mut rgb = vec![0u8; w * h * 3];
                    for row in 0..h {
                        for col in 0..w {
                            let yi = row * w + col;
                            let uvi = (row / 2) * uv_w + (col / 2);
                            let y = yuv.y[yi] as f32;
                            let u = yuv.u[uvi] as f32;
                            let v = yuv.v[uvi] as f32;
                            let r = y + 1.402 * (v - 128.0);
                            let g = y - 0.344 * (u - 128.0) - 0.714 * (v - 128.0);
                            let b = y + 1.772 * (u - 128.0);
                            let pi = yi * 3;
                            rgb[pi] = r.clamp(0.0, 255.0) as u8;
                            rgb[pi + 1] = g.clamp(0.0, 255.0) as u8;
                            rgb[pi + 2] = b.clamp(0.0, 255.0) as u8;
                        }
                    }
                    if let Err(e) =
                        image::save_buffer(out, &rgb, w as u32, h as u32, image::ColorType::Rgb8)
                    {
                        log::warn!("Failed to save ROI frame {}: {e}", out.display());
                    }
                }
                Ok(_) => log::warn!("No frame decoded from {}", video.display()),
                Err(e) => log::error!("Frame extraction failed for {}: {e}", video.display()),
            }
        };
        extract_frame(&left, &left_png, frame_index);
        extract_frame(&right, &right_png, frame_index);
        log::info!(
            "ROI frame extraction: left={} right={}",
            left_png.exists(),
            right_png.exists()
        );

        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD;

        let left_data = std::fs::read(&left_png)
            .ok()
            .map(|bytes| format!("data:image/png;base64,{}", b64.encode(&bytes)))
            .unwrap_or_default();
        let right_data = std::fs::read(&right_png)
            .ok()
            .map(|bytes| format!("data:image/png;base64,{}", b64.encode(&bytes)))
            .unwrap_or_default();

        let cal_json_str = std::fs::read_to_string(&cal_path).unwrap_or_else(|_| "{}".into());

        // Template is embedded at compile time so deployed binaries work
        // without needing the source tree.
        let template = include_str!("../../../resources/roi_editor.html");
        let html = template
            .replace("'{{LEFT_IMAGE_DATA}}'", &format!("'{left_data}'"))
            .replace("'{{RIGHT_IMAGE_DATA}}'", &format!("'{right_data}'"))
            .replace("{{CAL_JSON}}", &cal_json_str)
            .replace("'{{CAL_PATH}}'", &format!("'{}'", cal_path.display()));

        let out_html = tmp_dir.join("roi_editor.html");
        if let Err(e) = std::fs::write(&out_html, &html) {
            log::error!("Failed to write ROI editor HTML: {e}");
            let mut s = state_ref.borrow_mut();
            if let Some(app) = app_weak.upgrade() {
                s.toasts.push(
                    crate::toast::Severity::Error,
                    "ROI editor",
                    format!("Cannot write temp file: {e}"),
                );
                crate::toast::sync_to_ui(&s.toasts, &app);
            }
            return;
        }
        log::info!(
            "ROI editor: wrote {} bytes to {}, opening in browser",
            html.len(),
            out_html.display()
        );

        let open_result = open::that(out_html.as_os_str());
        if let Err(e) = open_result {
            log::error!("Failed to open ROI editor: {e}");
            let mut s = state_ref.borrow_mut();
            if let Some(app) = app_weak.upgrade() {
                s.toasts.push(
                    crate::toast::Severity::Error,
                    "Cannot open browser",
                    e.to_string(),
                );
                crate::toast::sync_to_ui(&s.toasts, &app);
            }
        } else {
            let mut s = state_ref.borrow_mut();
            if let Some(app) = app_weak.upgrade() {
                s.toasts.push(
                    crate::toast::Severity::Info,
                    "ROI editor opened in browser",
                    "Draw field boundary, click Save ROI, then come back here and click Paste ROI.",
                );
                crate::toast::sync_to_ui(&s.toasts, &app);
            }
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_paste_roi(move || {
        let mut s = state_ref.borrow_mut();
        // Try manual JSON input first, then clipboard
        let manual = app_weak
            .upgrade()
            .map(|a| {
                let t = a.get_roi_manual_json().to_string();
                a.set_roi_manual_json("".into());
                t
            })
            .unwrap_or_default();
        let clipboard_text = if !manual.trim().is_empty() {
            manual
        } else {
            match arboard::Clipboard::new().and_then(|mut cb| cb.get_text()) {
                Ok(t) => t,
                Err(e) => {
                    log::warn!("Clipboard read failed: {e}");
                    if let Some(app) = app_weak.upgrade() {
                        s.toasts.push(
                            crate::toast::Severity::Error,
                            "Paste ROI",
                            "Could not read clipboard. Paste JSON in the text field instead.",
                        );
                        crate::toast::sync_to_ui(&s.toasts, &app);
                    }
                    return;
                }
            }
        };

        let roi: reco_core::calibration::FieldRoi = match serde_json::from_str(&clipboard_text) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("Clipboard is not valid ROI JSON: {e}");
                if let Some(app) = app_weak.upgrade() {
                    s.toasts.push(
                        crate::toast::Severity::Error,
                        "Paste ROI",
                        "Clipboard doesn't contain valid ROI JSON. Save ROI in the browser editor first.",
                    );
                    crate::toast::sync_to_ui(&s.toasts, &app);
                }
                return;
            }
        };

        let point_count = roi.left.len() + roi.right.len();
        if let Some(cal) = s.calibration.as_mut() {
            cal.field_roi = Some(roi);
        }
        if let Err(e) = s.save_calibration() {
            log::error!("Failed to save calibration with ROI: {e}");
        }

        if let Some(app) = app_weak.upgrade() {
            let has = point_count > 0;
            app.set_has_roi(has);
            sync_roi_points(&s, &app);
            s.toasts.push(
                crate::toast::Severity::Info,
                "ROI applied",
                format!("{point_count} points saved to calibration."),
            );
            crate::toast::sync_to_ui(&s.toasts, &app);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_remove_left_segment(move |idx| {
        let mut s = state_ref.borrow_mut();
        if let Some(reco_io::stitch_job::InputPath::Chained(ref mut paths)) = s.left_input {
            let idx = idx as usize;
            if idx < paths.len() {
                paths.remove(idx);
                if paths.is_empty() {
                    s.left_input = None;
                    s.left_path = None;
                } else if paths.len() == 1 {
                    let p = paths[0].clone();
                    s.left_path = Some(p.clone());
                    s.left_input = Some(reco_io::stitch_job::InputPath::Single(p));
                } else {
                    s.left_path = Some(paths[0].clone());
                }
                if s.bridge.is_some() {
                    s.unload_pipeline();
                }
                if let Some(app) = app_weak.upgrade() {
                    let label = s
                        .left_input
                        .as_ref()
                        .map(|i| match i {
                            reco_io::stitch_job::InputPath::Single(p) => display_name(p),
                            reco_io::stitch_job::InputPath::Chained(ps) => {
                                format!("{} ({} segments)", display_name(&ps[0]), ps.len())
                            }
                        })
                        .unwrap_or_default();
                    app.set_left_path(label.into());
                    app.set_files_loaded(false);
                    sync_segments(&s, &app);
                }
            }
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_remove_right_segment(move |idx| {
        let mut s = state_ref.borrow_mut();
        if let Some(reco_io::stitch_job::InputPath::Chained(ref mut paths)) = s.right_input {
            let idx = idx as usize;
            if idx < paths.len() {
                paths.remove(idx);
                if paths.is_empty() {
                    s.right_input = None;
                    s.right_path = None;
                } else if paths.len() == 1 {
                    let p = paths[0].clone();
                    s.right_path = Some(p.clone());
                    s.right_input = Some(reco_io::stitch_job::InputPath::Single(p));
                } else {
                    s.right_path = Some(paths[0].clone());
                }
                if s.bridge.is_some() {
                    s.unload_pipeline();
                }
                if let Some(app) = app_weak.upgrade() {
                    let label = s
                        .right_input
                        .as_ref()
                        .map(|i| match i {
                            reco_io::stitch_job::InputPath::Single(p) => display_name(p),
                            reco_io::stitch_job::InputPath::Chained(ps) => {
                                format!("{} ({} segments)", display_name(&ps[0]), ps.len())
                            }
                        })
                        .unwrap_or_default();
                    app.set_right_path(label.into());
                    app.set_files_loaded(false);
                    sync_segments(&s, &app);
                }
            }
        }
    });

    // ── Preferences dialog callbacks ──
    //
    // Open prefills the prefs-* properties from user_settings; Save
    // reads them back and persists. Cancel just closes - no state
    // change needed because the Slint properties are scratch space
    // that gets re-seeded on next open.
    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_open_prefs_dialog(move || {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let s = state_ref.borrow();
        app.set_prefs_default_codec(s.user_settings.default_codec.clone().into());
        app.set_prefs_default_quality(s.user_settings.default_quality.clone().into());
        app.set_prefs_default_blend_width(s.user_settings.default_blend_width);
        app.set_prefs_ai_model_path(
            s.user_settings
                .ai_model_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default()
                .into(),
        );
        app.set_recording_codec(s.user_settings.recording_codec.clone().into());
        app.set_recording_quality(s.user_settings.recording_quality.clone().into());
        app.set_recording_folder(
            s.user_settings
                .recording_folder
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default()
                .into(),
        );
        app.set_prefs_telemetry_enabled(s.user_settings.telemetry_enabled);
        app.set_prefs_dark_mode(s.user_settings.dark_mode);
        app.set_prefs_dialog_open(true);
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_save_prefs(move || {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let mut s = state_ref.borrow_mut();
        s.user_settings.default_codec = app.get_prefs_default_codec().to_string();
        s.user_settings.default_quality = app.get_prefs_default_quality().to_string();
        s.user_settings.default_blend_width = app.get_prefs_default_blend_width();
        let model_path = app.get_prefs_ai_model_path().to_string();
        s.user_settings.ai_model_path = if model_path.is_empty() {
            None
        } else {
            Some(PathBuf::from(model_path))
        };
        s.user_settings.recording_codec = app.get_recording_codec().to_string();
        s.user_settings.recording_quality = app.get_recording_quality().to_string();
        let rec_folder = app.get_recording_folder().to_string();
        s.user_settings.recording_folder = if rec_folder.is_empty() {
            None
        } else {
            Some(PathBuf::from(rec_folder))
        };
        let dark = app.get_prefs_dark_mode();
        s.user_settings.dark_mode = dark;
        app.set_dark_mode(dark);

        let telemetry_wanted = app.get_prefs_telemetry_enabled();
        s.user_settings.telemetry_enabled = telemetry_wanted;
        if telemetry_wanted && s.telemetry.is_none() {
            let cid = s
                .user_settings
                .telemetry_client_id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            log::info!("Telemetry enabled by user (client_id={}).", &cid[..8]);
            let client = telemetry_client::TelemetryClient::new(cid);
            client.app_open();
            s.telemetry = Some(client);
        } else if !telemetry_wanted {
            if s.telemetry.is_some() {
                log::info!("Telemetry disabled by user.");
            }
            s.telemetry = None;
        }
        s.user_settings.save();
    });

    app.on_open_website(|| {
        let _ = open::that("https://github.com/reco-project/video-stitcher");
    });

    app.on_open_forum(|| {
        let _ = open::that("https://forum.reco-project.org/");
    });

    let app_weak = app.as_weak();
    app.on_pick_prefs_model(move || {
        let dialog = rfd::FileDialog::new()
            .set_title("Select YOLO ONNX model")
            .add_filter("ONNX", &["onnx"]);
        if let Some(path) = dialog.pick_file()
            && let Some(app) = app_weak.upgrade()
        {
            app.set_prefs_ai_model_path(path.to_string_lossy().to_string().into());
        }
    });

    let app_weak = app.as_weak();
    app.on_pick_recording_folder(move || {
        let dialog = rfd::FileDialog::new().set_title("Select default recording folder");
        if let Some(folder) = dialog.pick_folder()
            && let Some(app) = app_weak.upgrade()
        {
            app.set_recording_folder(folder.to_string_lossy().to_string().into());
        }
    });

    let state_ref = Rc::clone(&state);
    app.on_changed_preview_aspect(move |aspect| {
        let mut s = state_ref.borrow_mut();
        s.user_settings.preview_aspect = aspect.to_string();
        s.user_settings.save();
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
        // Capture current playback position as calibration start time
        let current_time_secs = if s.playback.fps() > 0.0 {
            s.playback.frame_index() as f64 / s.playback.fps()
        } else {
            0.0
        };
        drop(s);

        let (use_imu_seeds, cal_frames, akaze_threshold, detect_y_min, detect_y_max, skip_end) =
            app_weak
                .upgrade()
                .map(|a| {
                    (
                        a.get_use_imu_seeds(),
                        a.get_calibration_frames().max(2) as usize,
                        a.get_cal_akaze_threshold() as f64,
                        a.get_cal_detect_y_min() as f64,
                        a.get_cal_detect_y_max() as f64,
                        a.get_cal_skip_end_secs() as f64,
                    )
                })
                .unwrap_or((false, 4, 0.0001, 0.05, 0.95, 0.0));

        if let Some(app) = app_weak.upgrade() {
            app.set_calibrating(true);
            app.set_calibration_step("Starting...".into());
            app.set_status_text("Auto-calibrating...".into());
        }

        let interrupted = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();

        // Preserve user-picked lens profiles for recalibration.
        let (existing_left_params, existing_right_params) = {
            let s = state_ref.borrow();
            if let Some(bridge) = s.bridge.as_ref() {
                let cal = bridge.renderer().pipeline().calibration();
                (Some(cal.left.clone()), Some(cal.right.clone()))
            } else {
                (
                    s.cal_baseline_left_params.clone(),
                    s.cal_baseline_right_params.clone(),
                )
            }
        };

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
            log::info!(
                "Auto-calibrate: {cal_frames} frames, skip=[{current_time_secs:.1}s, -{skip_end:.0}s], \
                 imu_seeds={use_imu_seeds}, akaze={akaze_threshold}, \
                 detect_y=[{detect_y_min:.2}, {detect_y_max:.2}]"
            );
            let mut config = reco_calibrate::CalibrationConfig {
                num_frames: cal_frames,
                skip_start_secs: current_time_secs,
                skip_end_secs: skip_end,
                use_imu_rotation_seeds: use_imu_seeds,
                ..Default::default()
            };
            config.akaze.threshold = akaze_threshold;
            config.akaze.detect_y_min = detect_y_min;
            config.akaze.detect_y_max = detect_y_max;
            if existing_left_params.is_some() {
                log::info!("Re-calibrating with user-picked lens profiles");
            }
            let result = reco_calibrate::video::calibrate_videos(
                &left,
                &right,
                reco_calibrate::video::CalibrateVideosOptions {
                    config: Some(config),
                    left_params: existing_left_params,
                    right_params: existing_right_params,
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
                    Ok(CalibrationOutput {
                        calibration: r.calibration,
                        confidence: r.confidence,
                        total_matches: r.total_matches,
                        left_lens_profile: r.left_lens_profile,
                        right_lens_profile: r.right_lens_profile,
                    })
                }
                Err(e) => Err(e),
            };
            tx.send(cal_result).ok();
        });
    });

    // ── Playback callbacks ──

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_toggle_playback(move || {
        let mut s = state_ref.borrow_mut();
        if s.is_exporting() {
            log::info!("Playback toggle ignored: export in progress");
            return;
        }
        let new_state = s.playback.toggle();
        if let Some(app) = app_weak.upgrade() {
            app.set_playing(new_state == PlayState::Playing);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_step_forward(move || {
        let mut s = state_ref.borrow_mut();
        if s.is_exporting() {
            return;
        }
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
        if s.is_exporting() {
            return;
        }
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
    let app_weak = app.as_weak();
    app.on_changed_rig_tilt(move |deg| {
        state_ref.borrow_mut().set_rig_tilt(deg);
        if let Some(app) = app_weak.upgrade() {
            app.set_cal_dirty(true);
        }
    });

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_changed_rig_roll(move |deg| {
        state_ref.borrow_mut().set_rig_roll(deg);
        if let Some(app) = app_weak.upgrade() {
            app.set_cal_dirty(true);
        }
    });

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_changed_sync_offset(move |frames| {
        state_ref.borrow_mut().set_sync_offset(frames);
        if let Some(app) = app_weak.upgrade() {
            app.set_cal_dirty(true);
        }
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

    // ── Live calibration editing callbacks ──
    //
    // Each slider writes the corresponding field on the PlaneLayout,
    // pushes the edited layout into the renderer, and flips cal-dirty
    // so the Save button becomes enabled.

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_changed_cal_intersect(move |v| {
        let mut s = state_ref.borrow_mut();
        let Some(mut layout) = s.calibration.as_ref().map(|c| c.layout.clone()) else {
            return;
        };
        layout.intersect = v as f64;
        s.apply_layout(layout);
        if let Some(app) = app_weak.upgrade() {
            app.set_cal_dirty(true);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_changed_cal_camera_axis_offset(move |v| {
        let mut s = state_ref.borrow_mut();
        let Some(mut layout) = s.calibration.as_ref().map(|c| c.layout.clone()) else {
            return;
        };
        layout.camera_axis_offset = v as f64;
        s.apply_layout(layout);
        if let Some(app) = app_weak.upgrade() {
            app.set_cal_dirty(true);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_changed_cal_x_ty(move |v| {
        let mut s = state_ref.borrow_mut();
        let Some(mut layout) = s.calibration.as_ref().map(|c| c.layout.clone()) else {
            return;
        };
        layout.x_ty = v as f64;
        s.apply_layout(layout);
        if let Some(app) = app_weak.upgrade() {
            app.set_cal_dirty(true);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_save_calibration(move || {
        let save_result = state_ref.borrow().save_calibration();
        match save_result {
            Err(e) => {
                log::error!("Save calibration: {e}");
                let mut s = state_ref.borrow_mut();
                if let Some(app) = app_weak.upgrade() {
                    app.set_status_text("Save failed".into());
                    s.toasts.push(Severity::Error, "Save failed", e);
                    crate::toast::sync_to_ui(&s.toasts, &app);
                }
            }
            Ok(()) => {
                let mut s = state_ref.borrow_mut();
                if let Some(app) = app_weak.upgrade() {
                    app.set_status_text("Calibration saved".into());
                    app.set_cal_dirty(false);
                    s.toasts.push(Severity::Info, "Calibration saved", "");
                    crate::toast::sync_to_ui(&s.toasts, &app);
                }
            }
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_reset_calibration(move || {
        let mut s = state_ref.borrow_mut();
        s.reset_calibration();
        if let (Some(app), Some(layout)) = (app_weak.upgrade(), s.cal_baseline_layout.as_ref()) {
            app.set_cal_intersect(layout.intersect as f32);
            app.set_cal_camera_axis_offset(layout.camera_axis_offset as f32);
            app.set_cal_x_ty(layout.x_ty as f32);
            app.set_cal_dirty(false);
        }
    });

    // ── Live lens tuning callbacks ──
    //
    // Each slider emits `changed-lens-param` which asks Rust to read
    // the current fx/fy/cx/cy/k1-k4 from the UI properties for the
    // selected camera, build a `CameraParams`, and push it through
    // `update_camera_params`. Cheap per reco-core Batch F.

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_changed_lens_param(move || {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let mut s = state_ref.borrow_mut();
        let selected = app.get_lens_selected_camera();
        // Width/height come from the stored calibration (resolution the
        // lens profile was modelled at) and are never user-editable.
        let (left_wh, right_wh) = s
            .bridge
            .as_ref()
            .map(|b| {
                let c = b.renderer().pipeline().calibration();
                (
                    (c.left.width, c.left.height),
                    (c.right.width, c.right.height),
                )
            })
            .unwrap_or(((0, 0), (0, 0)));

        let (left_params, right_params) = match selected.as_str() {
            "right" => {
                let p = reco_core::calibration::CameraParams {
                    fx: app.get_lens_right_fx() as f64,
                    fy: app.get_lens_right_fy() as f64,
                    cx: app.get_lens_right_cx() as f64,
                    cy: app.get_lens_right_cy() as f64,
                    d: [
                        app.get_lens_right_k1() as f64,
                        app.get_lens_right_k2() as f64,
                        app.get_lens_right_k3() as f64,
                        app.get_lens_right_k4() as f64,
                    ],
                    width: right_wh.0,
                    height: right_wh.1,
                };
                (None, Some(p))
            }
            "both" => {
                // Mirror the Left sliders to both cameras. The Both tab
                // only shows the left sliders in the UI; the user's
                // intent is "apply these values to both lenses in
                // lockstep". We also push the mirrored values back into
                // the right-* Slint properties so when the user toggles
                // to Right later the sliders show what got applied.
                app.set_lens_right_fx(app.get_lens_left_fx());
                app.set_lens_right_fy(app.get_lens_left_fy());
                app.set_lens_right_cx(app.get_lens_left_cx());
                app.set_lens_right_cy(app.get_lens_left_cy());
                app.set_lens_right_k1(app.get_lens_left_k1());
                app.set_lens_right_k2(app.get_lens_left_k2());
                app.set_lens_right_k3(app.get_lens_left_k3());
                app.set_lens_right_k4(app.get_lens_left_k4());
                let left = reco_core::calibration::CameraParams {
                    fx: app.get_lens_left_fx() as f64,
                    fy: app.get_lens_left_fy() as f64,
                    cx: app.get_lens_left_cx() as f64,
                    cy: app.get_lens_left_cy() as f64,
                    d: [
                        app.get_lens_left_k1() as f64,
                        app.get_lens_left_k2() as f64,
                        app.get_lens_left_k3() as f64,
                        app.get_lens_left_k4() as f64,
                    ],
                    width: left_wh.0,
                    height: left_wh.1,
                };
                let right = reco_core::calibration::CameraParams {
                    width: right_wh.0,
                    height: right_wh.1,
                    ..left.clone()
                };
                (Some(left), Some(right))
            }
            _ => {
                let p = reco_core::calibration::CameraParams {
                    fx: app.get_lens_left_fx() as f64,
                    fy: app.get_lens_left_fy() as f64,
                    cx: app.get_lens_left_cx() as f64,
                    cy: app.get_lens_left_cy() as f64,
                    d: [
                        app.get_lens_left_k1() as f64,
                        app.get_lens_left_k2() as f64,
                        app.get_lens_left_k3() as f64,
                        app.get_lens_left_k4() as f64,
                    ],
                    width: left_wh.0,
                    height: left_wh.1,
                };
                (Some(p), None)
            }
        };
        if let Some(bridge) = s.bridge.as_mut() {
            bridge
                .renderer_mut()
                .update_camera_params(left_params, right_params);
        }
        s.preview_dirty = true;
        app.set_lens_dirty(true);
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_reset_lens(move || {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let mut s = state_ref.borrow_mut();
        // Baseline lens params live on the calibration we snapshotted
        // when auto-calibrate completed (see cal_baseline_*_params).
        let (left_base, right_base) = (
            s.cal_baseline_left_params.clone(),
            s.cal_baseline_right_params.clone(),
        );
        if let (Some(left), Some(right)) = (left_base.as_ref(), right_base.as_ref()) {
            set_lens_sliders(&app, left, right);
            if let Some(bridge) = s.bridge.as_mut() {
                bridge
                    .renderer_mut()
                    .update_camera_params(Some(left.clone()), Some(right.clone()));
            }
            s.preview_dirty = true;
            app.set_lens_dirty(false);
        }
    });

    // ── Lens picker callbacks ──

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_lens_search_changed(move |query| {
        let s = state_ref.borrow();
        let (in_w, in_h) = s.playback.input_dimensions().unwrap_or((0, 0));
        let db = reco_calibrate::lens_database::LensDatabase::embedded();
        let results = db.search(query.as_str(), in_w, in_h);
        let model: Vec<slint::SharedString> = results
            .iter()
            .map(|r| {
                slint::SharedString::from(format!(
                    "{} - {} - {}x{}",
                    r.camera, r.lens, r.width, r.height
                ))
            })
            .collect();
        if let Some(app) = app_weak.upgrade() {
            app.set_lens_search_results(slint::ModelRc::new(slint::VecModel::from(model)));
        }
    });

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_lens_pick(move |idx, side| {
        let app_ref = app_weak.upgrade();
        let query = app_ref
            .as_ref()
            .map(|a| a.get_lens_search_query().to_string())
            .unwrap_or_default();
        let mut s = state_ref.borrow_mut();
        let (in_w, in_h) = s.playback.input_dimensions().unwrap_or((0, 0));
        let db = reco_calibrate::lens_database::LensDatabase::embedded();
        let results = db.search(&query, in_w, in_h);
        let idx = idx as usize;
        if idx < results.len() {
            let summary = &results[idx];
            if let Some(params) = db.load_by_summary(summary) {
                let side_str = side.as_str();
                log::info!(
                    "Lens picker: applying {} - {} ({}x{}) to {side_str}",
                    summary.camera,
                    summary.lens,
                    summary.width,
                    summary.height
                );
                let scale_w = in_w as f64 / params.width as f64;
                let scale_h = in_h as f64 / params.height as f64;
                let scaled = reco_core::calibration::CameraParams {
                    width: in_w,
                    height: in_h,
                    fx: params.fx * scale_w,
                    fy: params.fy * scale_h,
                    cx: params.cx * scale_w,
                    cy: params.cy * scale_h,
                    d: params.d,
                };
                let (apply_left, apply_right) = match side_str {
                    "left" => (Some(scaled.clone()), None),
                    "right" => (None, Some(scaled.clone())),
                    _ => (Some(scaled.clone()), Some(scaled.clone())),
                };
                if let Some(bridge) = s.bridge.as_mut() {
                    bridge
                        .renderer_mut()
                        .update_camera_params(apply_left, apply_right);
                }
                s.preview_dirty = true;
                if let Some(app) = app_ref.as_ref() {
                    if side_str != "right" {
                        app.set_lens_left_camera(summary.camera.clone().into());
                        app.set_lens_left_source("Picker".into());
                    }
                    if side_str != "left" {
                        app.set_lens_right_camera(summary.camera.clone().into());
                        app.set_lens_right_source("Picker".into());
                    }
                    set_lens_sliders(app, &scaled, &scaled);
                    app.set_lens_dirty(true);
                    app.set_lens_info_available(true);
                }
            }
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_lens_pick_file(move || {
        let dialog = rfd::FileDialog::new()
            .set_title("Load lens profile JSON")
            .add_filter("JSON", &["json"]);
        if let Some(path) = dialog.pick_file() {
            match reco_calibrate::lens_database::load_from_file(&path) {
                Ok(params) => {
                    let mut s = state_ref.borrow_mut();
                    let (in_w, in_h) = s.playback.input_dimensions().unwrap_or((0, 0));
                    let scale_w = if params.width > 0 {
                        in_w as f64 / params.width as f64
                    } else {
                        1.0
                    };
                    let scale_h = if params.height > 0 {
                        in_h as f64 / params.height as f64
                    } else {
                        1.0
                    };
                    let scaled = reco_core::calibration::CameraParams {
                        width: in_w,
                        height: in_h,
                        fx: params.fx * scale_w,
                        fy: params.fy * scale_h,
                        cx: params.cx * scale_w,
                        cy: params.cy * scale_h,
                        d: params.d,
                    };
                    if let Some(bridge) = s.bridge.as_mut() {
                        bridge
                            .renderer_mut()
                            .update_camera_params(Some(scaled.clone()), Some(scaled.clone()));
                    }
                    s.preview_dirty = true;
                    if let Some(app) = app_weak.upgrade() {
                        set_lens_sliders(&app, &scaled, &scaled);
                        app.set_lens_dirty(true);
                        app.set_lens_picker_open(false);
                        let name = display_name(&path);
                        app.set_lens_left_camera(name.clone().into());
                        app.set_lens_left_source("File".into());
                        app.set_lens_right_camera(name.into());
                        app.set_lens_right_source("File".into());
                        app.set_lens_info_available(true);
                        s.toasts.push(
                            crate::toast::Severity::Info,
                            "Lens profile loaded",
                            path.display().to_string(),
                        );
                        crate::toast::sync_to_ui(&s.toasts, &app);
                    }
                }
                Err(e) => {
                    log::error!("Failed to load lens profile: {e}");
                    let mut s = state_ref.borrow_mut();
                    if let Some(app) = app_weak.upgrade() {
                        s.toasts.push(
                            crate::toast::Severity::Error,
                            "Failed to load lens profile",
                            e.to_string(),
                        );
                        crate::toast::sync_to_ui(&s.toasts, &app);
                    }
                }
            }
        }
    });

    // Slint's <=> binding updates the use-constrained-look property but
    // does not call back into Rust. Without this notify, AppState's
    // use_constrained_look stays at its initial value forever and the
    // UI checkbox is cosmetic.
    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_changed_constrained_look(move || {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let new_value = app.get_use_constrained_look();
        let mut s = state_ref.borrow_mut();
        s.use_constrained_look = new_value;
        // When re-enabling, apply the clamp to the current target so
        // the camera snaps back inside coverage instead of waiting for
        // the next pan/zoom input.
        if new_value {
            s.clamp_targets();
        }
        s.preview_dirty = true;
    });

    // ── Lens preview mode callbacks ──

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_changed_lens_preview(move || {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let mut s = state_ref.borrow_mut();
        s.lens_preview_active = app.get_lens_preview_active();
        s.lens_preview_side = app.get_lens_preview_side().to_string();
        sync_roi_points(&s, &app);
        s.preview_dirty = true;
    });

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_changed_lens_correction(move |amount| {
        let mut s = state_ref.borrow_mut();
        let clamped = if amount > 0.5 { 1.0 } else { 0.0 };
        s.lens_correction_amount = clamped;
        if !s.lens_preview_active
            && let Some(bridge) = s.bridge.as_mut()
        {
            bridge
                .renderer_mut()
                .pipeline_mut()
                .set_lens_correction_amount(clamped);
        }
        s.preview_dirty = true;
        if let Some(app) = app_weak.upgrade() {
            app.set_lens_correction_amount(clamped);
        }
    });

    // ── Toast dismissal ──
    //
    // Slint's ToastStack fires `toast-dismissed(id)` when the user
    // clicks the × on a card. Rust removes the matching entry and
    // pushes the refreshed list back to Slint.
    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_toast_dismissed(move |id| {
        let mut s = state_ref.borrow_mut();
        s.toasts.dismiss(id);
        if let Some(app) = app_weak.upgrade() {
            crate::toast::sync_to_ui(&s.toasts, &app);
        }
    });

    // ── Export dialog callbacks ──
    //
    // "Open" populates default values from current state (blend width
    // from preview, output path blank so user must pick one). "Start"
    // spawns a background thread running StitchJob; progress flows
    // back via invoke_from_event_loop so Slint properties stay on the
    // UI thread. "Cancel" flips the AtomicBool the job polls.

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_open_export_dialog(move || {
        let s = state_ref.borrow();
        if let Some(app) = app_weak.upgrade() {
            // Seed from persisted user defaults first (codec, quality,
            // model path) so the dialog reflects the user's last
            // choices across sessions...
            app.set_export_codec(s.user_settings.default_codec.clone().into());
            app.set_export_quality(s.user_settings.default_quality.clone().into());
            if let Some(model_path) = s.user_settings.ai_model_path.as_ref() {
                app.set_export_model_path(model_path.to_string_lossy().to_string().into());
            }
            let clip_secs = if s.playback.fps() > 0.0 {
                s.playback.total_frames().unwrap_or(0) as f32 / s.playback.fps() as f32
            } else {
                0.0
            };
            app.set_clip_duration_secs(clip_secs);
            if app.get_export_end_secs() == 0.0 {
                app.set_export_end_secs(clip_secs);
            }
            app.set_export_dialog_open(true);
        }
    });

    let app_weak = app.as_weak();
    app.on_pick_export_output(move || {
        let dialog = rfd::FileDialog::new()
            .set_title("Export stitched video to…")
            .add_filter("MP4", &["mp4"])
            .add_filter("MOV", &["mov"])
            .add_filter("MKV", &["mkv"]);
        if let Some(mut path) = dialog.save_file() {
            // Ensure an extension — ffmpeg picks muxer by extension.
            if path.extension().is_none() {
                path.set_extension("mp4");
            }
            if let Some(app) = app_weak.upgrade() {
                app.set_export_output_path(path.to_string_lossy().to_string().into());
            }
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_pick_export_model(move || {
        let dialog = rfd::FileDialog::new()
            .set_title("Select YOLO ONNX model")
            .add_filter("ONNX", &["onnx"]);
        if let Some(path) = dialog.pick_file()
            && let Some(app) = app_weak.upgrade()
        {
            app.set_export_model_path(path.to_string_lossy().to_string().into());
            // Remember across sessions so the user doesn't re-pick
            // the same ONNX every run. Save is best-effort.
            let mut s = state_ref.borrow_mut();
            s.user_settings.ai_model_path = Some(path);
            s.user_settings.save();
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_start_export(move || {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let output_str = app.get_export_output_path().to_string();
        if output_str.is_empty() {
            return;
        }
        let mut s = state_ref.borrow_mut();
        if s.export_thread.is_some() {
            log::warn!("Export already running, ignoring start request");
            return;
        }
        let output_path = PathBuf::from(&output_str);
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                s.toasts.push(
                    Severity::Error,
                    "Output directory does not exist",
                    parent.display().to_string(),
                );
                crate::toast::sync_to_ui(&s.toasts, &app);
                return;
            }
            if parent
                .metadata()
                .map_or(true, |m| m.permissions().readonly())
            {
                s.toasts.push(
                    Severity::Error,
                    "Output directory is not writable",
                    parent.display().to_string(),
                );
                crate::toast::sync_to_ui(&s.toasts, &app);
                return;
            }
        }
        let (Some(left_path), Some(right_path), Some(cal)) = (
            s.left_path.clone(),
            s.right_path.clone(),
            s.calibration.clone(),
        ) else {
            log::error!("Cannot start export without left/right/calibration");
            return;
        };
        let left = s
            .left_input
            .clone()
            .unwrap_or(reco_io::stitch_job::InputPath::Single(left_path));
        let right = s
            .right_input
            .clone()
            .unwrap_or(reco_io::stitch_job::InputPath::Single(right_path));

        // Snapshot all export settings. Slint properties must not be
        // read from the worker thread — only the UI thread owns them.
        let output = PathBuf::from(output_str);
        let width = app.get_export_width() as u32;
        let height = app.get_export_height() as u32;
        let codec_str = app.get_export_codec().to_string();
        let quality_str = app.get_export_quality().to_string();
        let blend = app.get_blend_width();
        let start_secs = app.get_export_start_secs();
        let end_secs = app.get_export_end_secs();
        log::info!("Export range: start={start_secs:.1}s, end={end_secs:.1}s");
        let autocam_enabled = app.get_export_autocam_enabled();
        let model_path = app.get_export_model_path().to_string();
        let tracking_mode = app.get_export_tracking_mode().to_string();

        let detection_interval = app.get_export_detection_interval() as u32;
        let replay_enabled = app.get_export_replay_enabled();

        // Persist the user's codec / quality / blend choices as the
        // defaults for next session. Model path is saved in the
        // on_pick_export_model callback so it sticks even if the user
        // never actually hits Start. Save is best-effort.
        s.user_settings.default_codec = codec_str.clone();
        s.user_settings.default_quality = quality_str.clone();
        s.user_settings.default_blend_width = blend;
        s.user_settings.save();

        // Reset cancel flag, start a fresh channel for completion.
        s.export_interrupted.store(false, Ordering::Relaxed);
        let interrupted = Arc::clone(&s.export_interrupted);
        let (tx, rx) = std::sync::mpsc::channel();
        s.export_rx = Some(rx);

        // Seed the last-progress timestamp so the Finalizing detector
        // has a starting point. Cloned for the worker below.
        *s.export_last_progress_at.lock().unwrap() = Some(Instant::now());
        let last_progress_at = Arc::clone(&s.export_last_progress_at);

        // Pause preview playback to avoid GPU contention with the
        // export pipeline. Preview rendering is also gated by
        // is_exporting() in vsync_render_tick.
        s.playback.pause();
        app.set_playing(false);

        app.set_export_in_progress(true);
        app.set_export_progress(0.0);
        app.set_export_frames_done(0);
        app.set_export_frames_total(0);
        app.set_export_status_text("Initializing…".into());
        app.set_export_dialog_open(false);

        let app_weak_bg = app_weak.clone();
        let output_for_thread = output.clone();
        let handle = std::thread::spawn(move || {
            let outcome = crate::export::run_export(
                left,
                right,
                cal,
                output_for_thread,
                None, // stream URL (Phase 6 GUI wiring)
                replay_enabled,
                width,
                height,
                codec_str,
                quality_str,
                blend,
                start_secs,
                end_secs,
                autocam_enabled,
                model_path,
                tracking_mode,
                detection_interval,
                app_weak_bg,
                &interrupted,
                last_progress_at,
            );
            let _ = tx.send(outcome);
        });
        s.export_thread = Some(handle);
    });

    let state_ref = Rc::clone(&state);
    app.on_cancel_export(move || {
        let s = state_ref.borrow();
        log::info!("Cancel requested");
        s.export_interrupted.store(true, Ordering::Relaxed);
    });

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_toggle_recording(move |codec, quality| {
        let mut s = state_ref.borrow_mut();
        if s.is_recording() {
            if let Some((path, frames)) = s.stop_recording()
                && let Some(app) = app_weak.upgrade()
            {
                app.set_recording(false);
                app.set_last_output_path(path.display().to_string().into());
                s.toasts.push_with_ttl(
                    crate::toast::Severity::Info,
                    "Recording saved",
                    format!("{frames} frames\n{}", path.display()),
                    Duration::from_secs(8),
                );
                crate::toast::sync_to_ui(&s.toasts, &app);
            }
        } else {
            match s.start_recording(codec.as_str(), quality.as_str()) {
                Ok(path) => {
                    if let Some(app) = app_weak.upgrade() {
                        app.set_recording(true);
                        s.toasts.push(
                            crate::toast::Severity::Info,
                            "Recording started",
                            path.display().to_string(),
                        );
                        crate::toast::sync_to_ui(&s.toasts, &app);
                    }
                }
                Err(e) => {
                    log::error!("Recording failed: {e}");
                    if let Some(app) = app_weak.upgrade() {
                        s.toasts
                            .push(crate::toast::Severity::Error, "Recording failed", e);
                        crate::toast::sync_to_ui(&s.toasts, &app);
                    }
                }
            }
        }
    });

    app.on_show_in_folder(move |path| {
        let path = PathBuf::from(path.as_str());
        let folder = path.parent().unwrap_or(&path);
        if let Err(e) = open::that(folder) {
            log::error!("Failed to open folder: {e}");
        }
    });

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_submit_bug_report(move |user_message, contact, include_logs| {
        let mut s = state_ref.borrow_mut();
        let msg = user_message.to_string();
        let contact_str = contact.to_string();

        let report = if include_logs {
            let sys_report = build_bug_report(&s, &app_weak);
            format!(
                "## User description\n{msg}\n\n## Contact\n{}\n\n{sys_report}",
                if contact_str.trim().is_empty() {
                    "(not provided)"
                } else {
                    contact_str.trim()
                }
            )
        } else {
            format!(
                "## User description\n{msg}\n\n## Contact\n{}",
                if contact_str.trim().is_empty() {
                    "(not provided)"
                } else {
                    contact_str.trim()
                }
            )
        };

        // Always try to send via telemetry (even without opt-in -
        // bug reports are an explicit user action, not passive tracking).
        let cid = s
            .user_settings
            .telemetry_client_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let client = s.telemetry.get_or_insert_with(|| {
            log::info!("Creating one-shot telemetry client for bug report");
            telemetry_client::TelemetryClient::new(cid)
        });
        client.bug_report(&report);

        // Save contact for next time
        if !contact_str.trim().is_empty() {
            s.user_settings
                .telemetry_client_id
                .get_or_insert_with(|| uuid::Uuid::new_v4().to_string());
            s.user_settings.save();
        }

        let _ = arboard::Clipboard::new().and_then(|mut cb| cb.set_text(report));

        s.toasts.push(
            crate::toast::Severity::Info,
            "Report sent",
            "Thank you! Your report has been sent to the developer.",
        );

        if let Some(app) = app_weak.upgrade() {
            crate::toast::sync_to_ui(&s.toasts, &app);
        }
    });

    // ── Playback timer ──

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    let timer = slint::Timer::default();
    let update_check = Arc::clone(&update_result);
    timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(TICK_INTERVAL_MS as u64),
        move || {
            let mut s = state_ref.borrow_mut();

            // Check for update notification from the background thread.
            if let Ok(mut guard) = update_check.try_lock()
                && let Some(tag) = guard.take()
                && let Some(app) = app_weak.upgrade()
            {
                        let url = format!(
                            "https://github.com/reco-project/video-stitcher/releases/tag/{tag}"
                        );
                        s.toasts.push_with_ttl(
                            Severity::Info,
                            format!("Update available: {tag}"),
                            "Opening download page in your browser.",
                            Duration::from_secs(30),
                        );
                        log::info!("Toast pushed for update {tag}");
                        crate::toast::sync_to_ui(&s.toasts, &app);
                        let _ = open::that(&url);
            }

            // Poll for calibration results from the background thread.
            if let Some(rx) = &s.cal_rx
                && let Ok(result) = rx.try_recv()
            {
                s.cal_rx = None;
                handle_calibration_result(result, &mut s, &app_weak);
                return;
            }

            // Poll the export worker for completion.
            if let Some(rx) = &s.export_rx
                && let Ok(outcome) = rx.try_recv()
            {
                s.export_rx = None;
                if let Some(h) = s.export_thread.take() {
                    let _ = h.join();
                }
                if let Some(app) = app_weak.upgrade() {
                    app.set_export_in_progress(false);
                    app.set_export_progress(0.0);
                    match outcome {
                        ExportOutcome::Ok(frames, path) => {
                            app.set_export_status_text("".into());
                            app.set_status_text(
                                format!("Export complete: {frames} frames -> {}", path.display(),)
                                    .into(),
                            );
                            app.set_last_output_path(path.display().to_string().into());
                            if let Some(ref t) = s.telemetry {
                                let fps = s.playback.fps();
                                let dur = if fps > 0.0 { frames as f64 / fps } else { 0.0 };
                                let codec = app.get_export_codec().to_string();
                                t.export_complete(frames, dur, &codec);
                            }
                            s.toasts.push(
                                Severity::Info,
                                "Export complete",
                                format!("{frames} frames to {}", path.display()),
                            );
                            crate::toast::sync_to_ui(&s.toasts, &app);
                        }
                        ExportOutcome::Cancelled => {
                            app.set_export_status_text("".into());
                            app.set_status_text("Export cancelled".into());
                        }
                        ExportOutcome::Failed(err) => {
                            app.set_export_status_text("".into());
                            app.set_status_text("Export failed".into());
                            let (title, body) = match &err {
                                reco_io::stitch_job::StitchError::EmptyOutput { .. } => (
                                    "Export produced no video",
                                    "The selected codec may not be supported on this hardware. Try H.264 or HEVC.".to_string(),
                                ),
                                reco_io::stitch_job::StitchError::Gpu(e) => (
                                    "GPU error",
                                    format!("{e}"),
                                ),
                                reco_io::stitch_job::StitchError::Calibration(e) => (
                                    "Calibration error",
                                    e.clone(),
                                ),
                                other => (
                                    "Export failed",
                                    format!("{other}"),
                                ),
                            };
                            let msg = format!("{err}");
                            if let Some(ref t) = s.telemetry {
                                t.export_error(&msg);
                            }
                            s.toasts.push(Severity::Error, title, body);
                            crate::toast::sync_to_ui(&s.toasts, &app);
                        }
                    }
                }
                return;
            }

            // If an export is running and we haven't seen a progress
            // update in > 1.5 seconds, the encoder is in its tail phase
            // (av_write_trailer + index flush can take up to ~15s on
            // H.264/AV1). Switch the status text to "Finalizing..." so
            // the progress bar doesn't look hung. Time-based detection
            // is robust to whatever the probe-reported total-frames is;
            // no frame-count heuristic needed.
            if let Some(app) = app_weak.upgrade()
                && app.get_export_in_progress()
                && let Some(last) = *s.export_last_progress_at.lock().unwrap()
                && last.elapsed() > Duration::from_millis(1500)
            {
                let status = app.get_export_status_text();
                if !status.starts_with("Finalizing") {
                    app.set_export_status_text("Finalizing output file…".into());
                }
            }

            // Persist window size, debounced (Tier 3d).
            if let Some(app) = app_weak.upgrade() {
                let size = app.window().size();
                let cur = (size.width, size.height);
                let last = s.last_persisted_window_size.unwrap_or((0, 0));
                if cur != last {
                    s.last_window_size_save_at = Some(Instant::now());
                    s.last_persisted_window_size = Some(cur);
                    s.user_settings.window_size = Some(cur);
                } else if let Some(since) = s.last_window_size_save_at
                    && since.elapsed() > Duration::from_secs(2)
                {
                    s.user_settings.save();
                    s.last_window_size_save_at = None;
                }
            }

            // Check if ROI editor finished and reload calibration.
            if let Some(ref flag) = s.roi_reload_pending
                && flag.load(Ordering::Relaxed)
            {
                s.roi_reload_pending = None;
                if let Some(cal_path) = s.calibration_path.as_ref()
                    && let Ok(cal) = MatchCalibration::from_file(cal_path)
                {
                    let has_roi = cal
                        .field_roi
                        .as_ref()
                        .is_some_and(|r| !r.left.is_empty() || !r.right.is_empty());
                    s.calibration = Some(cal);
                    if let Some(app) = app_weak.upgrade() {
                        app.set_has_roi(has_roi);
                        sync_roi_points(&s, &app);
                        s.toasts.push(
                            Severity::Info,
                            "Field ROI updated",
                            if has_roi {
                                "ROI loaded from calibration"
                            } else {
                                "No ROI points saved"
                            },
                        );
                        crate::toast::sync_to_ui(&s.toasts, &app);
                    }
                }
            }

            // Expire aged toasts (Tier 3a).
            if !s.toasts.is_empty()
                && s.toasts.expire(Instant::now())
                && let Some(app) = app_weak.upgrade()
            {
                crate::toast::sync_to_ui(&s.toasts, &app);
            }

            // Commit a debounced seek once the requested fraction has
            // stopped changing. During drag the fraction is refreshed
            // every pixel, so the elapsed check never passes. Only
            // after the user lets go does ~120ms pass without new
            // requests, triggering one seek instead of hundreds.
            if !s.is_exporting()
                && let Some((frac, requested_at)) = s.pending_seek
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

            // Playback, camera-lerp, and rendering now happen in
            // `vsync_render_tick` (driven by Slint's BeforeRendering
            // notifier), so this timer only handles work that does
            // not need vsync alignment. When playback is active we
            // nudge Slint to keep redrawing so BeforeRendering fires
            // even if nothing marked the window dirty yet.
            if let Some(app) = app_weak.upgrade()
                && (app.get_playing() || s.pending_seek.is_some() || s.preview_dirty)
            {
                app.window().request_redraw();
            }
        },
    );

    // Auto-open the files panel when no files are loaded so the user
    // sees the first action they need to take.
    if !app.get_files_loaded() {
        app.set_files_panel_open(true);
    }

    app.run()?;
    Ok(())
}

/// Vsync-aligned playback tick. Called from Slint's `BeforeRendering`
/// notifier so render submissions land at deterministic phase
/// relative to the compositor's 60 Hz cycle. Returns `true` when a
/// frame was submitted.
fn vsync_render_tick(state: &Rc<RefCell<AppState>>, app_weak: &slint::Weak<RecoApp>) -> bool {
    let mut s = state.borrow_mut();
    if s.is_exporting() {
        return false;
    }

    // Adaptive preview: resize render target to match the preview
    // container. Skipped during recording because the encoder was
    // initialized at a fixed resolution and the NV12 readback must
    // match. Also caps at 1920x1080 to prevent GPU starvation on
    // high-DPI displays.
    if !s.is_recording()
        && let Some(app) = app_weak.upgrade()
        && let Some(bridge) = s.bridge.as_mut()
    {
        let area_w = (app.get_preview_area_width().max(320.0) as u32).min(1920);
        let area_h = (app.get_preview_area_height().max(240.0) as u32).min(1080);
        let (cur_w, cur_h) = bridge.viewport_size();
        if area_w.abs_diff(cur_w) > 16 || area_h.abs_diff(cur_h) > 16 {
            bridge.resize(area_w, area_h);
            s.preview_dirty = true;
        }
    }

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
        return false;
    }
    // Clear dirty only if the camera has fully converged on its target.
    // If the lerp still has work to do, keep dirty so the timer will
    // nudge Slint for another BeforeRendering on the next tick - that
    // is how paused panning stays smooth until the user lets go AND
    // the camera eases to rest.
    s.preview_dirty = camera_changed;

    let img = s.render_current();
    if let (Some(app), Some(img)) = (app_weak.upgrade(), img) {
        app.set_preview_frame(img);
        s.last_render_at = Some(Instant::now());
        if video_advanced {
            let fps = s.playback.fps();
            let total = s.playback.total_frames().unwrap_or(0);
            sync_frame_display(&app, s.playback.frame_index(), total, fps);
            if s.playback.state() == PlayState::Finished {
                app.set_playing(false);
                app.set_status_text("Playback finished".into());
            }
        }
        // Reflect camera state to the UI properties so sliders and
        // the reset button stay in sync with what the user is
        // actually seeing. FOV comes from PoseControl's current (the
        // renderer pipeline's fov is driven by `smooth_camera`).
        let current = s.pose.current_pose();
        app.set_yaw(current.yaw);
        app.set_pitch(current.pitch);
        if let Some(fov) = current.fov_degrees {
            app.set_fov(fov);
        }
        return true;
    }
    false
}

/// Try to initialize the pipeline when all files are selected.
fn try_init_and_update(state: &Rc<RefCell<AppState>>, app_weak: &slint::Weak<RecoApp>) {
    let mut s = state.borrow_mut();
    if let Some(app) = app_weak.upgrade() {
        sync_segments(&s, &app);
    }
    match s.try_init() {
        Ok(true) => {
            let fps = s.playback.fps();
            let total = s.playback.total_frames().unwrap_or(0);

            s.clamp_targets();
            let clamped_fov = s.pose.current_fov_deg();
            if let Some(bridge) = s.bridge.as_mut() {
                bridge.renderer_mut().pipeline_mut().set_fov(clamped_fov);
            }
            let img = s.render_current();
            // Seed calibration slider values from the baseline layout.
            let layout = s.cal_baseline_layout.clone();

            let (in_w, in_h) = s.playback.input_dimensions().unwrap_or((0, 0));

            // Snapshot current lens params as the fine-tune baseline so
            // Reset Lens can restore them. For manual match.json loads
            // this comes from the loaded calibration directly.
            let lens_baseline = s.bridge.as_ref().map(|b| {
                let cal = b.renderer().pipeline().calibration();
                (cal.left.clone(), cal.right.clone())
            });
            if let Some((l, r)) = lens_baseline.as_ref() {
                s.cal_baseline_left_params = Some(l.clone());
                s.cal_baseline_right_params = Some(r.clone());
            }
            // Same for viewport-level settings (rig tilt, blend width).
            let rig_tilt_rad = s
                .bridge
                .as_ref()
                .map(|b| b.renderer().pipeline().viewport().rig_tilt);
            let rig_roll_rad = s
                .bridge
                .as_ref()
                .map(|b| b.renderer().pipeline().viewport().rig_roll);
            let blend_width = s
                .bridge
                .as_ref()
                .map(|b| b.renderer().pipeline().viewport().blend_width);

            if let Some(app) = app_weak.upgrade() {
                app.set_files_loaded(true);
                app.set_has_roi(
                    s.calibration
                        .as_ref()
                        .and_then(|c| c.field_roi.as_ref())
                        .is_some_and(|r| !r.left.is_empty() || !r.right.is_empty()),
                );
                sync_frame_display(&app, s.playback.frame_index(), total, fps);
                app.set_fps(fps as f32);
                app.set_status_text(format!("Ready - {:.0} fps - {total} frames", fps).into());
                if let Some(img) = img {
                    app.set_preview_frame(img);
                }
                if let Some(layout) = layout {
                    app.set_cal_intersect(layout.intersect as f32);
                    app.set_cal_camera_axis_offset(layout.camera_axis_offset as f32);
                    app.set_cal_x_ty(layout.x_ty as f32);
                    app.set_cal_dirty(false);
                }
                if let Some(rt) = rig_tilt_rad {
                    app.set_rig_tilt(rt.to_degrees());
                }
                if let Some(rr) = rig_roll_rad {
                    app.set_rig_roll(rr.to_degrees());
                }
                if let Some(bw) = blend_width {
                    app.set_blend_width(bw);
                }
                if let Some(cal) = s.calibration.as_ref() {
                    app.set_sync_offset(cal.sync_offset as i32);
                }
                app.set_fov(clamped_fov);
                // Manual calibration JSON does not embed lens-profile info,
                // so clear the display (hide the lens card) and just show
                // the candidates count for this resolution so the user
                // still knows how many database entries could match.
                set_lens_profile_props(&app, None, None, in_w, in_h);
                // Lens fine-tune sliders are seeded from the loaded
                // calibration's camera params either way; the Lens
                // fine-tune section is gated on `files-loaded` in Slint.
                if let Some((l, r)) = lens_baseline.as_ref() {
                    set_lens_sliders(&app, l, r);
                    app.set_lens_dirty(false);
                }
                // Seed export dialog output filename suggestion to
                // sit next to the left-video file for convenience.
                let left_path = s.left_path.clone();
                if let Some(left_path) = left_path {
                    let suggested = left_path.with_file_name(format!(
                        "{}_stitched.mp4",
                        left_path
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "reco".into())
                    ));
                    app.set_export_output_path(suggested.to_string_lossy().to_string().into());
                }

                if let Some(ref t) = s.telemetry {
                    let gpu = s
                        .bridge
                        .as_ref()
                        .map(|b| {
                            let g = b.renderer().gpu();
                            format!("{} ({:?})", g.gpu_name(), g.backend_name())
                        })
                        .unwrap_or_else(|| "unknown".into());
                    let os = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
                    let (ai_status, _) = ai_capability_summary();
                    t.context(&gpu, &os, &ai_status);
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
                // Batch G emits "invalid input path (...): <reason>" for
                // validation failures. Classify so the toast carries a
                // reason-specific title; fall back to generic "Init
                // failed" otherwise.
                let (title, body) = classify_init_error(&e);
                app.set_status_text(title.clone().into());
                app.set_files_loaded(false);
                s.toasts.push(Severity::Error, title, body);
                crate::toast::sync_to_ui(&s.toasts, &app);
            }
        }
    }
}

/// Inspect an init error string and decide what to show the user.
///
/// Batch G's `SourceError::InvalidPath` display format is
/// `"invalid input path (path): reason"`. We substring-match to pick
/// a friendlier title; the full stringified error becomes the body.
fn classify_init_error(err: &str) -> (String, String) {
    if err.contains("invalid input path") {
        let title = if err.contains("file not found") {
            "File not found"
        } else if err.contains("permission denied") {
            "Permission denied"
        } else if err.contains("file is empty") {
            "Empty file"
        } else if err.contains("not a regular file") {
            "Not a video file"
        } else {
            "Invalid file"
        };
        (title.to_string(), err.to_string())
    } else {
        ("Init failed".to_string(), err.to_string())
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
        Ok(output) => {
            let confidence = output.confidence;
            let total_matches = output.total_matches;
            if let Some(ref t) = state.telemetry {
                t.calibration_complete(confidence, total_matches);
            }
            let left_profile = output.left_lens_profile.clone();
            let right_profile = output.right_lens_profile.clone();
            match state.init_with_calibration(output.calibration) {
                Ok(true) => {
                    let fps = state.playback.fps();
                    let total = state.playback.total_frames().unwrap_or(0);
                    state.reset_view();
                    state.clamp_targets();
                    let clamped_fov = state.pose.current_fov_deg();
                    if let Some(bridge) = state.bridge.as_mut() {
                        bridge.renderer_mut().pipeline_mut().set_fov(clamped_fov);
                    }
                    let img = state.render_current();
                    let (in_w, in_h) = state.playback.input_dimensions().unwrap_or((0, 0));

                    // Snapshot camera intrinsics as the Lens fine-tune
                    // baseline so Reset Lens can restore them after
                    // manual edits.
                    let lens_baseline = state.bridge.as_ref().map(|b| {
                        let cal = b.renderer().pipeline().calibration();
                        (cal.left.clone(), cal.right.clone())
                    });
                    if let Some((l, r)) = lens_baseline.as_ref() {
                        state.cal_baseline_left_params = Some(l.clone());
                        state.cal_baseline_right_params = Some(r.clone());
                    }

                    // Grab the layout baseline so the Calibration sliders
                    // (intersect, camera-axis offset, x_ty) show the
                    // auto-calibrated values instead of 0. Without this
                    // the preview looks correct while the sliders read
                    // 0; clicking any of them snaps the layout to ~0
                    // and destroys the calibration.
                    let layout_baseline = state.cal_baseline_layout.clone();
                    // Same idea for rig tilt and blend width: read the
                    // calibrated values off the viewport so the View
                    // panel sliders match what the preview actually shows.
                    let rig_tilt_rad = state
                        .bridge
                        .as_ref()
                        .map(|b| b.renderer().pipeline().viewport().rig_tilt);
                    let blend_width = state
                        .bridge
                        .as_ref()
                        .map(|b| b.renderer().pipeline().viewport().blend_width);

                    // Auto-save calibration next to the left video so
                    // it appears in Recent and can be reloaded.
                    if let Some(left) = state.left_path.as_ref() {
                        let cal_path = left.with_file_name(format!(
                            "{}_calibration.json",
                            left.file_stem()
                                .map(|s| s.to_string_lossy().into_owned())
                                .unwrap_or_else(|| "reco".into())
                        ));
                        if let Some(cal) = state.calibration.as_ref() {
                            match serde_json::to_string_pretty(cal) {
                                Ok(json) => match std::fs::write(&cal_path, json) {
                                    Ok(()) => {
                                        log::info!(
                                            "Auto-saved calibration to {}",
                                            cal_path.display()
                                        );
                                        state.calibration_path = Some(cal_path.clone());
                                        state.user_settings.push_calibration(cal_path.clone());
                                    }
                                    Err(e) => {
                                        log::warn!("Failed to auto-save calibration: {e}");
                                    }
                                },
                                Err(e) => {
                                    log::warn!("Failed to serialize calibration: {e}");
                                }
                            }
                        }
                    }

                    if let Some(app) = app_weak.upgrade() {
                        let cal_label = state
                            .calibration_path
                            .as_ref()
                            .map(|p| display_name(p))
                            .unwrap_or_else(|| "(auto-calibrated)".into());
                        app.set_files_loaded(true);
                        app.set_has_roi(
                            state
                                .calibration
                                .as_ref()
                                .and_then(|c| c.field_roi.as_ref())
                                .is_some_and(|r| !r.left.is_empty() || !r.right.is_empty()),
                        );
                        app.set_calibration_path(cal_label.into());
                        sync_recent_paths(&state.user_settings, &app);
                        sync_frame_display(&app, state.playback.frame_index(), total, fps);
                        app.set_fps(fps as f32);
                        app.set_status_text(
                            format!("Auto-calibrated - {:.0} fps - {total} frames", fps,).into(),
                        );
                        if let Some(img) = img {
                            app.set_preview_frame(img);
                        }
                        if let Some(layout) = layout_baseline.as_ref() {
                            app.set_cal_intersect(layout.intersect as f32);
                            app.set_cal_camera_axis_offset(layout.camera_axis_offset as f32);
                            app.set_cal_x_ty(layout.x_ty as f32);
                            app.set_cal_dirty(false);
                        }
                        if let Some(rt) = rig_tilt_rad {
                            app.set_rig_tilt(rt.to_degrees());
                        }
                        if let Some(bw) = blend_width {
                            app.set_blend_width(bw);
                        }
                        if let Some(cal) = state.calibration.as_ref() {
                            app.set_sync_offset(cal.sync_offset as i32);
                        }
                        app.set_fov(clamped_fov);
                        set_lens_profile_props(&app, left_profile, right_profile, in_w, in_h);
                        if let Some((l, r)) = lens_baseline.as_ref() {
                            set_lens_sliders(&app, l, r);
                            app.set_lens_dirty(false);
                        }

                        if confidence < 0.5 {
                            log::warn!(
                                "Low calibration confidence ({:.0}%, {total_matches} matches). \
                                 Stitch quality may be poor.",
                                confidence * 100.0
                            );
                            state.toasts.push(
                                Severity::Warn,
                                "Low calibration confidence",
                                format!(
                                    "{:.0}% confidence ({total_matches} matches). \
                                     Try recording with more camera overlap.",
                                    confidence * 100.0
                                ),
                            );
                            crate::toast::sync_to_ui(&state.toasts, &app);
                        }
                    }
                }
                Ok(false) => {}
                Err(e) => {
                    log::error!("Post-calibration init: {e}");
                    if let Some(app) = app_weak.upgrade() {
                        app.set_status_text("Post-calibration init failed".into());
                        state.toasts.push(Severity::Error, "Init failed", e.clone());
                        crate::toast::sync_to_ui(&state.toasts, &app);
                    }
                }
            }
        }
        Err(e) => {
            log::error!("Auto-calibration failed: {e}");
            // Critical: unload the live pipeline so the preview stops
            // rendering whatever it was showing before. Otherwise the
            // preview keeps playing the OLD right/left video while the
            // state thinks the new paths are active - and export would
            // read the new paths and produce garbage. Flipping
            // `files-loaded=false` forces the user to re-pick or
            // re-calibrate from a clean state.
            state.unload_pipeline();
            if let Some(app) = app_weak.upgrade() {
                app.set_files_loaded(false);
                app.set_status_text("Calibration failed".into());
                // Toast wants a display-ready message; stringify at
                // the UI boundary (not across the mpsc channel).
                state
                    .toasts
                    .push(Severity::Error, "Auto-calibration failed", e.to_string());
                crate::toast::sync_to_ui(&state.toasts, &app);
            }
        }
    }
}
