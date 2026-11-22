#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! Reco GUI - Slint-based panoramic video stitcher.
//!
//! Preview-first workspace for source files, calibration, camera controls and export.
//!
//! ## Architecture
//!
//! Slint and reco-core share a single wgpu 28 device. `main()` selects
//! the wgpu 28 backend via `BackendSelector::require_wgpu_28()`, and a
//! `set_rendering_notifier` callback captures Slint's device/queue on
//! `RenderingSetup`. Those handles feed `GpuContext::from_device_queue`,
//! so reco-core renders stitched frames directly into Slint-owned
//! textures with no CPU readback.

#[cfg(test)]
mod calibration_edit_tests;
mod calibration_job;
mod export;
mod field_editor;
mod network;
mod playback;
mod preview;
mod settings;
#[cfg(test)]
mod shutdown_tests;
mod telemetry_client;
mod toast;
#[cfg(test)]
mod ui_tests;

use std::cell::RefCell;
use std::future::Future;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use reco_calibrate::{LensProfileInfo, ProfileSource};
use reco_control::ControlIntent;
use reco_control::pose_control::{PoseControl, PoseControlConfig};
use reco_core::calibration::Calibration;
use reco_core::render::viewport::ViewportSize;
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
/// drift-free scheduler in `Playback::tick` can catch up when a scheduled frame boundary is crossed between callbacks. Display delivery still depends on the presentation backend.
const TICK_INTERVAL_MS: i64 = 2;

/// FOV clamp range (degrees), matching CLI preview.
const FOV_MIN: f32 = 1.0;
const FOV_MAX: f32 = 150.0;

/// Fitted calibration with the assessment and optical source for each camera.
struct CalibrationOutput {
    calibration: Calibration,
    assessment: reco_calibrate::CalibrationAssessment,
    total_matches: usize,
    left_lens_profile: Option<LensProfileInfo>,
    right_lens_profile: Option<LensProfileInfo>,
}

/// Result sent from the calibration background thread. The error is
/// the typed [`reco_calibrate::video::CalibrateVideosError`] (which
/// is `Clone + Send + Sync`), so the UI thread can pattern-match
/// specific failure modes (`Cancelled`, `NoFrames`, `Io(...)`, etc.)
/// instead of parsing a stringified message.
type CalibrationResult = Result<CalibrationOutput, reco_calibrate::video::CalibrateVideosError>;

trait FileDialogAction: Send {
    fn apply(self: Box<Self>, state: &Rc<RefCell<AppState>>, app_weak: &slint::Weak<RecoApp>);
}

struct SelectedFileDialog<T, A> {
    selection: T,
    apply: A,
}

impl<T, A> FileDialogAction for SelectedFileDialog<T, A>
where
    T: Send + 'static,
    A: FnOnce(T, &Rc<RefCell<AppState>>, &slint::Weak<RecoApp>) + Send + 'static,
{
    fn apply(self: Box<Self>, state: &Rc<RefCell<AppState>>, app_weak: &slint::Weak<RecoApp>) {
        let Self { selection, apply } = *self;
        apply(selection, state, app_weak);
    }
}

/// Headless dev/test preload hook. When `RECO_AUTOLOAD` is set the GUI loads
/// the given left/right videos and calibration on startup (and optionally
/// starts an export via `RECO_AUTOEXPORT`), so the app can be driven under
/// Xvfb for screenshots and CI smoke tests. Inert when the env var is unset.
#[cfg(feature = "automation")]
struct AutoloadSpec {
    left: Vec<PathBuf>,
    right: Vec<PathBuf>,
    cal: PathBuf,
    export: Option<AutoExportSpec>,
}

/// Optional auto-export for [`AutoloadSpec`], from `RECO_AUTOEXPORT`.
#[cfg(feature = "automation")]
struct AutoExportSpec {
    output: PathBuf,
    model: Option<PathBuf>,
    lookahead_secs: f32,
    /// Total number of exports to run back-to-back in one session, from
    /// `RECO_AUTOEXPORT_REPEAT` (default 1). Used to surface cross-export GPU
    /// resource leaks: VRAM is logged between runs, so a leak shows up as a
    /// figure that never returns to baseline. The output path gets a `_N`
    /// suffix per run so runs don't clobber each other.
    repeats: u32,
}

#[cfg(feature = "automation")]
impl AutoloadSpec {
    /// Parse `RECO_AUTOLOAD="left[;left2,...],right[;right2,...],cal.json"`
    /// (segments within a side separated by `;`). Returns `None` when unset or
    /// malformed. An empty right field selects mono input.
    fn from_env() -> Option<Self> {
        let raw = std::env::var("RECO_AUTOLOAD").ok()?;
        let parts: Vec<&str> = raw.split(',').collect();
        if parts.len() != 3 {
            log::warn!(
                "RECO_AUTOLOAD ignored: expected 'left[;left2],right[;right2],cal.json', got {raw:?}"
            );
            return None;
        }
        let to_paths =
            |s: &str| -> Vec<PathBuf> { s.split(';').map(|p| PathBuf::from(p.trim())).collect() };
        let left = to_paths(parts[0]);
        let right = if parts[1].trim().is_empty() {
            Vec::new()
        } else {
            to_paths(parts[1])
        };
        let cal = PathBuf::from(parts[2].trim());
        if left
            .iter()
            .chain(right.iter())
            .any(|p| p.as_os_str().is_empty())
        {
            log::warn!("RECO_AUTOLOAD ignored: empty path component");
            return None;
        }
        let export = std::env::var("RECO_AUTOEXPORT")
            .ok()
            .map(|out| AutoExportSpec {
                output: PathBuf::from(out),
                model: std::env::var("RECO_AUTOEXPORT_MODEL")
                    .ok()
                    .map(PathBuf::from),
                lookahead_secs: std::env::var("RECO_AUTOEXPORT_LOOKAHEAD")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.0),
                repeats: std::env::var("RECO_AUTOEXPORT_REPEAT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .filter(|&n| n > 0)
                    .unwrap_or(1),
            });
        Some(Self {
            left,
            right,
            cal,
            export,
        })
    }
}

/// Application state shared between Slint callbacks.
struct AppState {
    left_path: Option<PathBuf>,
    right_path: Option<PathBuf>,
    left_input: Option<reco_io::stitch_job::InputPath>,
    right_input: Option<reco_io::stitch_job::InputPath>,
    calibration_path: Option<PathBuf>,
    calibration: Option<Calibration>,
    playback: Playback,
    network: network::State,
    // Playback cancels before this scope waits for all retired decoders.
    decoder_cleanup: reco_io::DecoderCleanup,
    bridge: Option<PreviewBridge>,
    recording_tx: Option<std::sync::mpsc::SyncSender<RecordingFrame>>,
    recording_thread: Option<std::thread::JoinHandle<()>>,
    recording_path: Option<PathBuf>,
    recording_frames: u64,
    /// Owns a calibration worker and the document it was started for.
    calibration_job: Option<calibration_job::PendingCalibration>,
    review_calibration: Option<(CalibrationOutput, calibration_job::CalibrationSnapshot)>,
    field_editor: Option<field_editor::FieldEditor>,
    displayed_view: Option<(ViewportSize, reco_core::geometry::RenderPose)>,
    /// Receives native file dialog selections after the OS dialog
    /// completes on a background waiter.
    file_dialog_rx: Option<std::sync::mpsc::Receiver<Option<Box<dyn FileDialogAction>>>>,
    /// Prevents stacking multiple native pickers while the app window
    /// remains responsive.
    file_dialog_open: bool,
    /// wgpu handles captured from Slint's rendering notifier. `None`
    /// until the window has completed its first rendering setup.
    shared_gpu: Option<SharedGpu>,
    /// Headless preload/auto-export hook (RECO_AUTOLOAD); `None` in normal use.
    #[cfg(feature = "automation")]
    autoload: Option<AutoloadSpec>,
    /// Repeat-export driver state (RECO_AUTOEXPORT_REPEAT). When `mode` is set,
    /// the export-completion handler re-triggers another export until `done`
    /// reaches `total`, then quits the event loop. Inert otherwise.
    #[cfg(feature = "automation")]
    auto_export_mode: bool,
    /// Total exports to run back-to-back this session.
    #[cfg(feature = "automation")]
    auto_export_total: u32,
    /// Exports completed so far (the next run is `done + 1`).
    #[cfg(feature = "automation")]
    auto_export_done: u32,
    /// Base output path for repeat exports; each run appends `_N` before the
    /// extension so successive runs do not overwrite each other.
    #[cfg(feature = "automation")]
    auto_export_base: Option<PathBuf>,
    /// Pending direct input and the last pose accepted by coverage.
    pose: PoseControl,
    /// Last time we pushed a rendered frame to Slint.
    last_render_at: Option<Instant>,
    /// A changed frame or control needs rendering even while playback is paused.
    preview_dirty: bool,
    /// Keep polling the first coverage build even when playback is paused.
    coverage_pending: bool,
    coverage_error_reported: bool,
    /// Interrupt flag for a running export. Set to true when the user
    /// clicks Cancel; StitchJob checks it between frames and aborts.
    export_interrupted: Arc<AtomicBool>,
    /// Join handle for the export worker. Held so the timer can see
    /// when the export finishes (via try_recv on export_rx).
    export_thread: Option<std::thread::JoinHandle<()>>,
    /// Receives export completion notifications from the worker.
    export_rx: Option<std::sync::mpsc::Receiver<ExportOutcome>>,
    export_stream: crate::export::StreamState,
    /// Editor state retained while export releases the preview's GPU resources and decoders.
    export_preview: Option<PreviewBookmark>,
    /// Original Projection values — what auto-calibrate produced. Live
    /// calibration sliders edit relative to this so Reset restores.
    cal_baseline: Option<reco_core::calibration::Calibration>,
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
    /// Fit direct input through engine coverage. Disabling permits inspecting outside the stitched region during calibration. Bound to the Slint `use-constrained-look` checkbox.
    use_constrained_look: bool,
    /// When true, the preview shows a single camera through orthographic
    /// projection instead of the stitched panorama.
    lens_preview_active: bool,
    /// Which camera to show in lens preview mode ("left" or "right").
    lens_preview_side: String,
    toasts: ToastManager,
    telemetry: Option<telemetry_client::TelemetryClient>,
}

/// Runtime AI capability summary.
///
/// Report registered ORT providers before model-specific initialization.
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
        (
            format!("AI providers: {}", probe.providers.join(", ")),
            true,
        )
    }
}

use crate::export::ExportOutcome;

fn build_bug_report(state: &AppState, app_weak: &slint::Weak<RecoApp>) -> String {
    let gpu = state
        .bridge
        .as_ref()
        .map(|b| {
            let g = b.engine().gpu();
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
        if fps_avg > 0.0 || total_ms > 0.0 {
            report.push_str(&format!(
                "\n## Last export performance\n\
                 - FPS: {fps_avg:.0} avg / {fps_recent:.0} recent\n\
                 - Frame time: {total_ms:.1} ms\n"
            ));
        }
    }

    // Redact file paths
    if let Some(cal) = &state.calibration_path {
        let name = cal.file_name().unwrap_or_default().to_string_lossy();
        report.push_str(&format!("\n## Files\n- Calibration: {name}\n"));
    }

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
}

struct PreviewBookmark {
    inputs: Vec<Vec<PathBuf>>,
    calibration: Calibration,
    pose: PoseControl,
    frame: u64,
}

impl PreviewBookmark {
    fn capture(state: &AppState) -> Option<Self> {
        let calibration = state.calibration.clone()?;
        let inputs = state.selected_inputs(&calibration)?;
        Some(Self {
            inputs: inputs.iter().map(reco_io::InputPath::all_paths).collect(),
            calibration,
            pose: state.pose.clone(),
            frame: state.playback.frame_index().saturating_sub(1),
        })
    }

    fn matches(&self, state: &AppState) -> bool {
        state.calibration.as_ref() == Some(&self.calibration)
            && state
                .selected_inputs(&self.calibration)
                .is_some_and(|inputs| {
                    inputs
                        .iter()
                        .map(reco_io::InputPath::all_paths)
                        .eq(self.inputs.iter().cloned())
                })
    }
}

fn preview_pose_control(engine: Option<&reco_core::core::StitchCore>) -> PoseControl {
    let mut config = PoseControlConfig {
        fov_min_degrees: FOV_MIN,
        fov_max_degrees: FOV_MAX,
        invert_drag_x: true,
        ..PoseControlConfig::default()
    };
    if let Some(engine) = engine {
        config.rest_pose = engine
            .calibration()
            .world_to_render_pose(reco_core::geometry::WorldPose::default());
    }
    PoseControl::new(config).expect("valid calibrated preview controls")
}

fn displayed_image_height(width: f32, height: f32, image: (u32, u32)) -> f32 {
    height.min(width * image.1 as f32 / image.0 as f32)
}

impl Drop for AppState {
    fn drop(&mut self) {
        self.cancel_calibration();
        self.network.cancel();
        self.export_interrupted.store(true, Ordering::Relaxed);
        if let Some(worker) = self.export_thread.take() {
            let _ = worker.join();
        }
        self.stop_recording();
    }
}

impl AppState {
    fn new(user_settings: crate::settings::GuiSettings) -> Self {
        Self {
            left_path: None,
            right_path: None,
            left_input: None,
            right_input: None,
            calibration_path: None,
            calibration: None,
            playback: Playback::new(),
            network: network::State::default(),
            decoder_cleanup: reco_io::DecoderCleanup::default(),
            bridge: None,
            recording_tx: None,
            recording_thread: None,
            recording_path: None,
            recording_frames: 0,
            calibration_job: None,
            review_calibration: None,
            field_editor: None,
            displayed_view: None,
            file_dialog_rx: None,
            file_dialog_open: false,
            shared_gpu: None,
            #[cfg(feature = "automation")]
            autoload: AutoloadSpec::from_env(),
            #[cfg(feature = "automation")]
            auto_export_mode: false,
            #[cfg(feature = "automation")]
            auto_export_total: 0,
            #[cfg(feature = "automation")]
            auto_export_done: 0,
            #[cfg(feature = "automation")]
            auto_export_base: None,
            pose: preview_pose_control(None),
            last_render_at: None,
            preview_dirty: false,
            coverage_pending: false,
            coverage_error_reported: false,
            export_interrupted: Arc::new(AtomicBool::new(false)),
            export_thread: None,
            export_rx: None,
            export_stream: crate::export::StreamState::default(),
            export_preview: None,
            cal_baseline: None,
            user_settings,
            last_persisted_window_size: None,
            last_window_size_save_at: None,
            use_constrained_look: true,
            lens_preview_active: false,
            lens_preview_side: "left".into(),
            toasts: ToastManager::default(),
            telemetry: None,
        }
    }

    fn is_exporting(&self) -> bool {
        self.export_thread.is_some()
    }

    fn reset_pipeline(&mut self) {
        self.network.cancel();
        self.field_editor = None;
        self.displayed_view = None;
        self.cancel_calibration();
        self.export_preview = None;
        self.bridge = None;
        self.playback = Playback::new();
        self.pose = preview_pose_control(None);
        self.last_render_at = None;
        self.preview_dirty = false;
        self.coverage_pending = false;
        self.coverage_error_reported = false;
    }

    fn cancel_calibration(&self) {
        if let Some(job) = &self.calibration_job {
            job.cancel();
        }
    }

    fn poll_calibration(&mut self, app_weak: &slint::Weak<RecoApp>) -> bool {
        let Some(job) = &self.calibration_job else {
            return false;
        };
        let result = match job.receiver.try_recv() {
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                if job.is_cancelled()
                    && let Some(app) = app_weak.upgrade()
                    && app.get_calibration_step() != "Stopping…"
                {
                    app.set_calibration_step("Stopping…".into());
                }
                return false;
            }
            result => result,
        };
        let current = job.matches(self);
        let cancelled = job.is_cancelled();
        self.calibration_job = None;
        if let Some(app) = app_weak.upgrade() {
            app.set_calibrating(false);
        }
        if !current || cancelled {
            if current
                && cancelled
                && let Some(app) = app_weak.upgrade()
            {
                app.set_status_text("Calibration cancelled".into());
            }
            log::info!("Discarded calibration result for superseded inputs or settings");
            return true;
        }
        match result {
            Ok(result) => handle_calibration_result(result, self, app_weak, false),
            Err(_) => {
                self.toasts.push(
                    Severity::Error,
                    "Calibration stopped",
                    "The worker stopped without a result. You can retry calibration.",
                );
                if let Some(app) = app_weak.upgrade() {
                    crate::toast::sync_to_ui(&self.toasts, &app);
                }
            }
        }
        true
    }

    fn suspend_preview_for_export(&mut self) {
        let bookmark = PreviewBookmark::capture(self);
        self.reset_pipeline();
        self.export_preview = bookmark;
    }

    fn restore_preview_after_export(&mut self) -> Result<(), String> {
        if self.network.active.is_some() {
            return Ok(());
        }
        let Some(saved) = self.export_preview.take() else {
            return Ok(());
        };
        let same_context = saved.matches(self);
        let Some(cal) = self.calibration.clone() else {
            return Ok(());
        };
        self.init_preview(&cal)?;
        if same_context {
            self.pose = saved.pose;
            self.playback
                .seek_frame(saved.frame)
                .map_err(|e| e.to_string())?;
            if self.playback.is_seeking() {
                self.playback
                    .advance_blocking()
                    .map_err(|e| e.to_string())?;
            }
        }
        self.preview_dirty = true;
        Ok(())
    }

    /// Build a PreviewBridge using the captured Slint GPU handles. Fails
    /// if the rendering notifier hasn't populated `shared_gpu` yet.
    fn build_bridge(
        &self,
        cal: &Calibration,
        source_sizes: Vec<(u32, u32)>,
        full_range: bool,
    ) -> Result<PreviewBridge, String> {
        let gpu = self
            .shared_gpu
            .as_ref()
            .ok_or("GPU not ready yet (Slint rendering not initialized)")?
            .clone();
        let mut bridge = PreviewBridge::new(
            gpu.device,
            gpu.queue,
            gpu.adapter_info,
            cal.clone(),
            source_sizes,
            PREVIEW_WIDTH_DEFAULT,
            PREVIEW_HEIGHT_DEFAULT,
        )
        .map_err(|e| format!("GPU init error: {e}"))?;
        bridge.engine_mut().set_full_range(full_range);
        Ok(bridge)
    }

    /// Apply validated editor geometry without waiting for derived coverage.
    fn edit_calibration(&mut self, app: &RecoApp, edit: impl FnOnce(&mut Calibration)) -> bool {
        let Some(mut draft) = self.editor_calibration().cloned() else {
            return false;
        };
        edit(&mut draft);
        match self.apply_calibration(draft) {
            Ok(()) => self.finish_calibration_edit(app),
            Err(error) => {
                self.report_calibration_error(app, error);
                return false;
            }
        }
        true
    }

    fn editor_calibration(&self) -> Option<&Calibration> {
        self.calibration.as_ref()
    }

    fn apply_calibration(
        &mut self,
        calibration: Calibration,
    ) -> Result<(), reco_core::core::types::StitchCoreError> {
        if let Some(bridge) = self.bridge.as_mut() {
            let previous_framing = &bridge.engine().calibration().framing;
            let world = (previous_framing.tilt != calibration.framing.tilt
                || previous_framing.roll != calibration.framing.roll)
                .then(|| {
                    bridge
                        .engine()
                        .calibration()
                        .render_to_world_pose(self.pose.current_pose())
                });
            bridge
                .engine_mut()
                .update_calibration_preview(calibration.clone())?;
            if let Some(world) = world {
                self.pose
                    .set_pose(bridge.engine().calibration().world_to_render_pose(world));
            }
        } else {
            calibration.validate()?;
        }
        self.cancel_calibration();
        self.calibration = Some(calibration);
        self.displayed_view = None;
        self.preview_dirty = true;
        self.clamp_targets();
        Ok(())
    }

    fn finish_calibration_edit(&mut self, app: &RecoApp) {
        app.set_cal_dirty(true);
        self.sync_calibration_controls(app);
        field_editor::sync(self, app);
        app.set_status_text("Calibration updated".into());
    }

    fn report_calibration_error(&mut self, app: &RecoApp, error: impl std::fmt::Display) {
        self.sync_calibration_controls(app);
        self.toasts
            .push(Severity::Error, "Calibration unchanged", error.to_string());
        crate::toast::sync_to_ui(&self.toasts, app);
    }

    fn apply_lens_profile(
        &mut self,
        app: &RecoApp,
        profile: &reco_core::calibration::Lens,
        side: &str,
    ) -> bool {
        let Some(cal) = self.editor_calibration() else {
            return false;
        };
        let lenses = cal
            .lenses
            .iter()
            .enumerate()
            .map(|(index, lens)| {
                if (side == "left" && index != 0) || (side == "right" && index != 1) {
                    return Ok(lens.clone());
                }
                let mut scaled = profile.rescaled(lens.width, lens.height)?;
                scaled.correction = lens.correction;
                Ok(scaled)
            })
            .collect::<Result<Vec<_>, reco_core::calibration::CalibrationError>>();
        match lenses {
            Ok(lenses) => self.edit_calibration(app, |cal| cal.lenses = lenses),
            Err(error) => {
                self.report_calibration_error(app, error);
                false
            }
        }
    }

    fn sync_calibration_controls(&self, app: &RecoApp) {
        let Some(cal) = self.editor_calibration() else {
            return;
        };
        if let Some(layout) = cal.projection.l_shape() {
            app.set_cal_intersect(layout.intersect as f32);
            app.set_cal_x_ty(layout.x_ty as f32);
        }
        app.set_cal_camera_axis_offset(cal.framing.axis_offset as f32);
        app.set_rig_tilt((cal.framing.tilt as f32).to_degrees());
        app.set_rig_roll((cal.framing.roll as f32).to_degrees());
        app.set_blend_width(cal.projection.blend_width());
        app.set_lens_correction_amount(cal.lenses[0].correction);
        app.set_sync_offset(cal.sync_offset as i32);
        app.set_is_l_shape(cal.projection.l_shape().is_some());
        if cal.projection.camera_count() == 1 {
            app.set_lens_selected_camera("left".into());
            app.set_lens_preview_active(false);
            app.set_export_replay_enabled(false);
        }
        set_lens_sliders(app, &cal.lenses);
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

    /// Select only the inputs consumed by the calibration.
    fn selected_inputs(&self, cal: &Calibration) -> Option<Vec<reco_io::InputPath>> {
        match cal.projection.camera_count() {
            1 => Some(vec![self.left_input.clone()?]),
            2 => Some(vec![self.left_input.clone()?, self.right_input.clone()?]),
            _ => None,
        }
    }

    /// Rebuild preview from the accepted document when its inputs and GPU are ready.
    fn try_init(&mut self) -> Result<bool, String> {
        let Some(cal) = self.calibration.clone() else {
            return Ok(false);
        };
        if self.selected_inputs(&cal).is_none() || self.shared_gpu.is_none() {
            return Ok(false);
        }
        self.init_preview(&cal)?;
        Ok(true)
    }

    /// Load a document without discarding the accepted state on read or preview failure.
    fn load_calibration(&mut self, path: PathBuf) -> Result<bool, String> {
        let cal =
            Calibration::from_file(&path).map_err(|e| format!("Calibration load error: {e}"))?;
        let ready = (self.selected_inputs(&cal).is_some() || self.playback.is_live())
            && self.shared_gpu.is_some();
        if ready {
            self.init_with_calibration(cal)?;
        } else {
            self.unload_pipeline();
            self.cal_baseline = Some(cal.clone());
            self.calibration = Some(cal);
        }
        self.calibration_path = Some(path);
        Ok(ready)
    }

    /// Initialize preview from a calibration result (no file needed).
    fn init_with_calibration(&mut self, cal: Calibration) -> Result<(), String> {
        self.init_preview(&cal)?;
        self.cal_baseline = Some(cal.clone());
        self.calibration = Some(cal);
        Ok(())
    }

    /// Replace playback and rendering together without changing the document or reset baseline.
    fn init_preview(&mut self, cal: &Calibration) -> Result<(), String> {
        if self.playback.is_live() {
            let sizes = self.playback.source_sizes().unwrap().to_vec();
            if sizes.len() != cal.projection.camera_count() {
                return Err("Disconnect live inputs before changing the camera count.".into());
            }
            let bridge = self.build_bridge(cal, sizes, self.playback.is_full_range())?;
            self.bridge = Some(bridge);
            self.reset_view();
            return Ok(());
        }
        let inputs = self
            .selected_inputs(cal)
            .ok_or("Select the calibration's video inputs")?;

        let sync_offset = cal.sync_offset;
        let mut playback = Playback::new();
        playback
            .open(&inputs, sync_offset, &self.decoder_cleanup)
            .map_err(|e| format!("Video open error: {e}"))?;

        let source_sizes = playback
            .source_sizes()
            .ok_or("No input dimensions")?
            .to_vec();

        let bridge = self.build_bridge(cal, source_sizes, playback.is_full_range())?;

        self.cancel_calibration();
        self.stop_recording();
        if cal.projection.camera_count() == 1 {
            self.lens_preview_active = false;
            self.lens_preview_side = "left".into();
        }
        self.playback = playback;
        self.bridge = Some(bridge);
        self.export_preview = None;
        self.reset_view();
        Ok(())
    }

    /// Release preview resources and pending work while retaining the selected paths and calibration document.
    fn unload_pipeline(&mut self) {
        self.network.active = None;
        self.field_editor = None;
        self.displayed_view = None;
        self.stop_recording();
        self.reset_pipeline();
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

        let config = reco_io::ffmpeg::encoder::EncoderConfig {
            codec: Some(codec.parse::<reco_io::output::Codec>()?),
            quality: quality.parse()?,
            ..Default::default()
        };
        let encoder =
            reco_io::adapters::FfmpegFileEncoder::new(&path, rec_w, rec_h, fps_r, &config)
                .map_err(|e| format!("Failed to start recording: {e}"))?;
        let enc_name = encoder.encoder_name();

        log::info!(
            "Recording started: {} ({enc_name}, {codec}/{quality})",
            path.display()
        );

        // Spawn encoder thread with bounded channel (4 frames deep).
        // The UI thread sends NV12 data without blocking on FFmpeg.
        let (tx, rx) = std::sync::mpsc::sync_channel::<RecordingFrame>(4);
        let handle = std::thread::spawn(move || {
            let mut encoder: Box<dyn reco_core::sink::OutputSink> = Box::new(encoder);
            while let Ok(frame) = rx.recv() {
                let _ = encoder.consume(reco_core::sink::OutputFrame {
                    data: &frame.data,
                    width: frame.width,
                    height: frame.height,
                    format: reco_core::sink::PixelFormat::Nv12,
                    pts_us: None,
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
        if !self.lens_preview_active && !self.clamp_targets() {
            return None;
        }
        let frame = self.playback.current_frame()?;

        // Lens preview mode: render single camera flat
        if self.lens_preview_active {
            let bridge = self.bridge.as_mut()?;
            let cal = bridge.engine().calibration();
            let camera = usize::from(self.lens_preview_side == "right");
            let reco_core::source::FrameSet::Yuv420p(cameras) = frame else {
                return None;
            };
            let planes = cameras.get(camera)?.as_planes();
            let params = cal.lenses.get(camera)?.clone();
            return match bridge.render_lens_preview(
                &planes,
                cameras[camera].size,
                &params,
                self.playback.is_full_range(),
            ) {
                Ok(img) => Some(img),
                Err(e) => {
                    log::error!("Lens preview error: {e}");
                    None
                }
            };
        }

        let pose = self.pose.current_pose();

        let recording = self.is_recording();
        if recording {
            let bridge = self.bridge.as_mut().unwrap();
            let (w, h) = bridge.viewport_size();

            // Encode the accepted interactive pose, also used by the display below.
            match bridge.engine_mut().render_and_readback_nv12(frame, pose) {
                Ok(Some(nv12)) => {
                    if let Some(tx) = self.recording_tx.as_ref() {
                        let _ = tx.try_send(RecordingFrame {
                            data: nv12.to_vec(),
                            width: w & !3,
                            height: h & !1,
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
                let bridge = self.bridge.as_mut().unwrap();
                match bridge.render_frame(frame, pose) {
                    Ok(img) => return Some(img),
                    Err(e) => log::error!("Preview render error: {e}"),
                }
            }
            None
        } else {
            match self.bridge.as_mut()?.render_frame(frame, pose) {
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
    fn apply_pan(&mut self, dx_px: f32, dy_px: f32, width: f32, height: f32) {
        let Some(bridge) = &self.bridge else { return };
        let height = displayed_image_height(width, height, bridge.viewport_size());
        self.pose.apply_drag(dx_px, dy_px, height);
        self.clamp_targets();
        // Paused playback still needs a redraw after input.
        self.preview_dirty = true;
    }

    /// Apply and constrain a FOV delta in degrees.
    fn apply_zoom(&mut self, delta_deg: f32) {
        self.pose
            .apply_intent(ControlIntent::DeltaFovDeg(delta_deg));
        self.clamp_targets();
        self.preview_dirty = true;
    }

    /// Set and constrain the slider's absolute FOV.
    fn set_fov(&mut self, fov_deg: f32) {
        self.pose.apply_intent(ControlIntent::SetFovDeg(fov_deg));
        self.clamp_targets();
        self.preview_dirty = true;
    }

    /// Refit input after geometry changes, reporting whether a redraw is needed.
    fn resolve_camera(&mut self) -> bool {
        let before = self.pose.current_pose();
        self.clamp_targets();
        let after = self.pose.current_pose();

        let yaw_changed = (before.yaw - after.yaw).abs() > f32::EPSILON;
        let pitch_changed = (before.pitch - after.pitch).abs() > f32::EPSILON;
        let fov_changed = before.fov_degrees != after.fov_degrees;

        // FOV rides the pose into every render_frame call, so no cached
        // push is needed here - the render can never see a stale value.
        yaw_changed || pitch_changed || fov_changed
    }

    fn set_sync_offset(&mut self, app: &RecoApp, frames: i32) -> bool {
        if self.playback.is_live() {
            self.sync_calibration_controls(app);
            return false;
        }
        let offset = i64::from(frames);
        let mut playback = None;
        if let (Some(left), Some(right)) = (&self.left_input, &self.right_input) {
            let mut candidate = Playback::new();
            if let Err(error) = candidate.open(
                &[left.clone(), right.clone()],
                offset,
                &self.decoder_cleanup,
            ) {
                self.sync_calibration_controls(app);
                self.toasts
                    .push(Severity::Error, "Sync unchanged", error.to_string());
                crate::toast::sync_to_ui(&self.toasts, app);
                return false;
            }
            let target = self.playback.frame_index().saturating_sub(1);
            if target > 0
                && candidate.total_frames().is_some_and(|total| target < total)
                && let Err(error) = candidate.seek_frame(target)
            {
                self.sync_calibration_controls(app);
                self.toasts
                    .push(Severity::Error, "Sync unchanged", error.to_string());
                crate::toast::sync_to_ui(&self.toasts, app);
                return false;
            }
            playback = Some(candidate);
        }
        if !self.edit_calibration(app, |cal| cal.sync_offset = offset) {
            return false;
        }
        if let Some(playback) = playback {
            self.playback = playback;
        }
        true
    }

    /// Re-open the playback source from the current chained inputs without
    /// rebuilding the GPU pipeline. Used after a segment reorder: the
    /// calibration and bridge are unchanged, only the decode order differs,
    /// so the next render picks up the new order from the start.
    fn reopen_source(&mut self, app: &RecoApp) {
        self.cancel_calibration();
        // Nothing consumes the source until a preview pipeline exists (e.g.
        // before calibration), so skip the costly decoder open. try_init opens
        // it in the current order once calibration is loaded.
        if self.bridge.is_none() {
            return;
        }
        let offset = self
            .calibration
            .as_ref()
            .map(|c| c.sync_offset)
            .unwrap_or(0);
        if let Some(inputs) = self
            .calibration
            .as_ref()
            .and_then(|cal| self.selected_inputs(cal))
        {
            if let Err(e) = self.playback.open(&inputs, offset, &self.decoder_cleanup) {
                log::error!("Failed to reopen playback after reorder: {e}");
                // The list already contains the new order. Keeping the old
                // preview would misrepresent what export will read.
                self.unload_pipeline();
                app.set_files_loaded(false);
                app.set_playing(false);
                app.set_preview_frame(slint::Image::default());
                app.set_status_text("Could not reopen reordered inputs".into());
                self.toasts.push(
                    Severity::Error,
                    "Could not reopen reordered inputs",
                    e.to_string(),
                );
                crate::toast::sync_to_ui(&self.toasts, app);
                return;
            }
            if let Some(bridge) = &mut self.bridge {
                bridge
                    .engine_mut()
                    .set_full_range(self.playback.is_full_range());
            }
            self.preview_dirty = true;
        }
    }

    /// Restore the calibrated world-space rest direction as render-space input.
    fn reset_view(&mut self) {
        self.pose = preview_pose_control(self.bridge.as_ref().map(|bridge| bridge.engine()));
        self.preview_dirty = true;
    }

    /// Accept the render-space input through the engine, discarding blocked deltas.
    fn clamp_targets(&mut self) -> bool {
        let Some(bridge) = self.bridge.as_mut() else {
            self.coverage_pending = false;
            return false;
        };
        let renderer = bridge.engine_mut();
        renderer.set_constrained_look(self.use_constrained_look);
        let result = self.pose.constrain_preview(renderer);
        self.coverage_pending = result.is_ok()
            && self.use_constrained_look
            && renderer.coverage_max_fov_degrees().is_none();
        if let Err(error) = result {
            if !self.coverage_error_reported {
                log::error!("Cannot fit preview: {error}");
                self.toasts
                    .push(Severity::Error, "Coverage unavailable", error.to_string());
                self.coverage_error_reported = true;
            }
            // Valid geometry remains editable even when no covered viewport exists.
            self.pose.set_pose(self.pose.target_pose());
        } else {
            self.coverage_error_reported = false;
        }
        true
    }

    /// Seek by a relative number of seconds (positive = forward).
    fn seek_relative(&mut self, secs: f32) -> Result<(), String> {
        let fps = self.playback.fps();
        let total = self.playback.total_frames().unwrap_or(0);
        if fps <= 0.0 || total == 0 || !secs.is_finite() {
            return Ok(());
        }
        let current = self.playback.requested_frame();
        let delta = (secs as f64 * fps).round() as i64;
        let target = current.saturating_add_signed(delta).min(total - 1);
        self.playback.seek_frame(target).map_err(|e| e.to_string())
    }
}

fn begin_file_dialog<T, F, B, A>(
    state: &Rc<RefCell<AppState>>,
    app_weak: &slint::Weak<RecoApp>,
    build: B,
    apply: A,
) where
    T: Send + 'static,
    F: Future<Output = Option<T>> + Send + 'static,
    B: FnOnce() -> F,
    A: FnOnce(T, &Rc<RefCell<AppState>>, &slint::Weak<RecoApp>) + Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    {
        let mut s = state.borrow_mut();
        if s.file_dialog_open {
            if let Some(app) = app_weak.upgrade() {
                app.set_status_text("Finish the current file selection first".into());
            }
            return;
        }
        s.file_dialog_open = true;
        s.file_dialog_rx = Some(rx);
    }
    // Mirror into Slint: raises the modal-guard overlay that blocks
    // pointer input while the (unparented, non-modal) picker is up.
    if let Some(app) = app_weak.upgrade() {
        app.set_file_dialog_open(true);
    }

    let task = build();
    std::thread::spawn(move || {
        let action = pollster::block_on(task).map(|selection| {
            Box::new(SelectedFileDialog { selection, apply }) as Box<dyn FileDialogAction>
        });
        if tx.send(action).is_err() {
            log::debug!("File dialog result dropped because the GUI closed");
        }
    });
}

fn poll_file_dialog(state: &Rc<RefCell<AppState>>, app_weak: &slint::Weak<RecoApp>) -> bool {
    let action = {
        let mut s = state.borrow_mut();
        let Some(rx) = s.file_dialog_rx.as_ref() else {
            return false;
        };
        match rx.try_recv() {
            Ok(action) => {
                s.file_dialog_rx = None;
                s.file_dialog_open = false;
                action
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => return false,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                s.file_dialog_rx = None;
                s.file_dialog_open = false;
                log::warn!("File dialog worker disconnected before returning a selection");
                None
            }
        }
    };

    // Both delivery and disconnect end the dialog: drop the overlay
    // before applying so the result lands on an interactive UI.
    if let Some(app) = app_weak.upgrade() {
        app.set_file_dialog_open(false);
    }
    if let Some(action) = action {
        action.apply(state, app_weak);
    }
    true
}

fn file_handle_to_path(handle: rfd::FileHandle) -> PathBuf {
    handle.path().to_path_buf()
}

fn file_handles_to_paths(handles: Vec<rfd::FileHandle>) -> Vec<PathBuf> {
    handles.into_iter().map(file_handle_to_path).collect()
}

/// Seed the Slint lens-tune sliders and their display ranges from a
/// pair of baseline `Lens`. Called after auto-calibrate completes
/// and on Reset Lens. Ranges are chosen wide enough for meaningful
/// manual tuning (fx/fy: +/-15%, cx/cy: +/-10% of image dim) but tight
/// enough that the slider granularity is useful.
fn set_lens_sliders(app: &RecoApp, lenses: &[reco_core::calibration::Lens]) {
    let Some(left) = lenses.first() else { return };
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
    app.set_lens_left_k1(left.distortion[0] as f32);
    app.set_lens_left_k2(left.distortion[1] as f32);
    app.set_lens_left_k3(left.distortion[2] as f32);
    app.set_lens_left_k4(left.distortion[3] as f32);

    let Some(right) = lenses.get(1) else { return };
    app.set_lens_right_fx(right.fx as f32);
    app.set_lens_right_fy(right.fy as f32);
    app.set_lens_right_cx(right.cx as f32);
    app.set_lens_right_cy(right.cy as f32);
    app.set_lens_right_k1(right.distortion[0] as f32);
    app.set_lens_right_k2(right.distortion[1] as f32);
    app.set_lens_right_k3(right.distortion[2] as f32);
    app.set_lens_right_k4(right.distortion[3] as f32);
}

/// Human-readable description of how a lens profile was resolved.
fn profile_source_label(info: &LensProfileInfo) -> &'static str {
    match info.source {
        ProfileSource::Estimated => "Stitching estimate",
        ProfileSource::Supplied => "Supplied optics",
    }
}

/// Display the optical provenance reported by the fitter.
fn set_lens_profile_props(
    app: &RecoApp,
    left: Option<LensProfileInfo>,
    right: Option<LensProfileInfo>,
) {
    if let Some(info) = &left {
        app.set_lens_left_camera(info.camera.clone().into());
        app.set_lens_left_lens(info.lens.clone().into());
        app.set_lens_left_source(profile_source_label(info).into());
    } else {
        app.set_lens_left_camera("Unknown".into());
        app.set_lens_left_lens("".into());
        app.set_lens_left_source("Unknown".into());
    }
    if let Some(info) = &right {
        app.set_lens_right_camera(info.camera.clone().into());
        app.set_lens_right_lens(info.lens.clone().into());
        app.set_lens_right_source(profile_source_label(info).into());
    } else {
        app.set_lens_right_camera("Unknown".into());
        app.set_lens_right_lens("".into());
        app.set_lens_right_source("Unknown".into());
    }

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
    app.set_current_time_text(if app.get_network_active() {
        "LIVE".into()
    } else {
        format_time(frame, fps).into()
    });
    app.set_total_time_text(if app.get_network_active() {
        "".into()
    } else {
        format_time(total, fps).into()
    });
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

/// Install the standard tracing subscriber + log bridge.
///
/// Bridges `log::*` calls
/// from reco-core / reco-io / reco-calibrate into tracing so user bug
/// reports arrive as one structured event stream instead of two
/// loggers writing to the same stderr.
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let _ = tracing_log::LogTracer::init();
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort::logging=warn"));

    // In release builds, write logs to a file so bug reports have context.
    // Debug builds just use stderr.
    #[cfg(not(debug_assertions))]
    if let Some(log_path) = log_file_path() {
        if let Some(parent) = log_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // Truncate if over 2 MB to prevent unbounded growth, otherwise append
        // so crash logs survive a restart.
        if log_path
            .metadata()
            .map(|m| m.len() > 2_000_000)
            .unwrap_or(false)
        {
            let _ = std::fs::remove_file(&log_path);
        }
        let file_result = std::fs::File::options()
            .create(true)
            .append(true)
            .open(&log_path);
        if let Err(ref e) = file_result {
            eprintln!(
                "Warning: could not open log file {}: {e}",
                log_path.display()
            );
        }
        if let Ok(file) = file_result {
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
/// - Linux: `~/.local/state/reco/reco-gui.log` (XDG_STATE_HOME)
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
        std::env::var("XDG_STATE_HOME")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| std::path::PathBuf::from(h).join(".local/state"))
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
                settings.backends = reco_core::gpu::GpuContext::select_backends();
            }
            config
        })
        .select()?;

    let app = RecoApp::new()?;
    let mut user_settings = crate::settings::GuiSettings::load();
    if user_settings.telemetry_client_id.is_none() {
        user_settings.telemetry_client_id = Some(uuid::Uuid::new_v4().to_string());
        user_settings.save();
    }
    let state = Rc::new(RefCell::new(AppState::new(user_settings)));

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
        ("h264", reco_io::output::Codec::H264),
        ("hevc", reco_io::output::Codec::HEVC),
        ("av1", reco_io::output::Codec::AV1),
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
    // Window size restore disabled - Slint's preferred size (1280x820)
    // is a safe default across all displays. Users can maximize manually.

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
                    // Produce the preview immediately before Slint renders its scene.
                    vsync_render_tick(&state_for_notifier, &app_weak_notifier);
                }
                _ => {}
            }
        })?;

    network::bind(&app, &state);

    // ── File picker callbacks ──

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_pick_left_video(move || {
        begin_file_dialog(
            &state_ref,
            &app_weak,
            || {
                let task = rfd::AsyncFileDialog::new()
                    .set_title("Select left camera video(s)")
                    .add_filter(
                        "Video",
                        &["mp4", "MP4", "mov", "MOV", "avi", "AVI", "mkv", "MKV"],
                    )
                    .pick_files();
                async move { task.await.map(file_handles_to_paths) }
            },
            |mut paths, state_ref, app_weak| {
                if paths.is_empty() {
                    return;
                }
                paths.sort();
                let input = {
                    let s = state_ref.borrow();
                    match &s.left_input {
                        Some(existing) => {
                            let mut all = match existing {
                                reco_io::stitch_job::InputPath::Single(p) => vec![p.clone()],
                                reco_io::stitch_job::InputPath::Chained(ps) => ps.clone(),
                            };
                            all.extend(paths);
                            log::info!("Left: appended to {} total segments", all.len());
                            reco_io::stitch_job::InputPath::Chained(all)
                        }
                        None => {
                            if paths.len() > 1 {
                                log::info!(
                                    "Left: {} segments selected, chaining via concat demuxer",
                                    paths.len()
                                );
                            }
                            reco_io::stitch_job::InputPath::from_segments(paths)
                        }
                    }
                };
                let first = match &input {
                    reco_io::stitch_job::InputPath::Single(p) => p.clone(),
                    reco_io::stitch_job::InputPath::Chained(ps) => ps[0].clone(),
                };
                {
                    let mut s = state_ref.borrow_mut();
                    let changed = s.left_path.as_ref() != Some(&first);
                    if changed {
                        s.unload_pipeline();
                        if let Some(app) = app_weak.upgrade() {
                            app.set_files_loaded(false);
                            app.set_status_text(
                                "File changed - re-calibrate or load calibration".into(),
                            );
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
                    try_init_and_update(state_ref, app_weak);
                }
            },
        );
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_pick_right_video(move || {
        begin_file_dialog(
            &state_ref,
            &app_weak,
            || {
                let task = rfd::AsyncFileDialog::new()
                    .set_title("Select right camera video(s)")
                    .add_filter(
                        "Video",
                        &["mp4", "MP4", "mov", "MOV", "avi", "AVI", "mkv", "MKV"],
                    )
                    .pick_files();
                async move { task.await.map(file_handles_to_paths) }
            },
            |mut paths, state_ref, app_weak| {
                if paths.is_empty() {
                    return;
                }
                paths.sort();
                let input = {
                    let s = state_ref.borrow();
                    match &s.right_input {
                        Some(existing) => {
                            let mut all = match existing {
                                reco_io::stitch_job::InputPath::Single(p) => vec![p.clone()],
                                reco_io::stitch_job::InputPath::Chained(ps) => ps.clone(),
                            };
                            all.extend(paths);
                            log::info!("Right: appended to {} total segments", all.len());
                            reco_io::stitch_job::InputPath::Chained(all)
                        }
                        None => {
                            if paths.len() > 1 {
                                log::info!(
                                    "Right: {} segments selected, chaining via concat demuxer",
                                    paths.len()
                                );
                            }
                            reco_io::stitch_job::InputPath::from_segments(paths)
                        }
                    }
                };
                let first = match &input {
                    reco_io::stitch_job::InputPath::Single(p) => p.clone(),
                    reco_io::stitch_job::InputPath::Chained(ps) => ps[0].clone(),
                };
                {
                    let mut s = state_ref.borrow_mut();
                    let changed = s.right_path.as_ref() != Some(&first);
                    if changed {
                        s.unload_pipeline();
                        if let Some(app) = app_weak.upgrade() {
                            app.set_files_loaded(false);
                            app.set_status_text(
                                "File changed - re-calibrate or load calibration".into(),
                            );
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
                    try_init_and_update(state_ref, app_weak);
                }
            },
        );
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_pick_calibration(move || {
        begin_file_dialog(
            &state_ref,
            &app_weak,
            || {
                let task = rfd::AsyncFileDialog::new()
                    .set_title("Select calibration JSON")
                    .add_filter("JSON", &["json", "JSON"])
                    .pick_file();
                async move { task.await.map(file_handle_to_path) }
            },
            |path, state_ref, app_weak| {
                load_calibration_and_update(state_ref, app_weak, path);
            },
        );
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
        if changed {
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
        // try_init and the segment list read left_input, not left_path -
        // without it a recent pick never loads and the UI keeps saying
        // "No video selected" (#328).
        s.left_input = Some(reco_io::stitch_job::InputPath::Single(path.clone()));
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
        if changed {
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
        s.right_input = Some(reco_io::stitch_job::InputPath::Single(path.clone()));
        s.right_path = Some(path);
        drop(s);
        try_init_and_update(&state_ref, &app_weak);
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_load_recent_calibration(move |entry| {
        load_calibration_and_update(&state_ref, &app_weak, PathBuf::from(entry.as_str()));
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
        s.unload_pipeline();
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
        s.unload_pipeline();
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
        s.unload_pipeline();
        s.calibration_path = None;
        s.calibration = None;
        s.cal_baseline = None;
        if let Some(app) = app_weak.upgrade() {
            app.set_cal_dirty(false);
            app.set_lens_dirty(false);
            app.set_calibration_path("".into());
            app.set_files_loaded(false);
            app.set_status_text("Calibration cleared".into());
        }
    });

    field_editor::install(&app, &state);

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_remove_left_segment(move |idx| {
        let mut s = state_ref.borrow_mut();
        let removed = match s.left_input {
            Some(reco_io::stitch_job::InputPath::Single(_)) if idx == 0 => {
                s.left_input = None;
                s.left_path = None;
                true
            }
            Some(reco_io::stitch_job::InputPath::Chained(ref mut paths)) => {
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
                    true
                } else {
                    false
                }
            }
            _ => false,
        };
        if removed {
            s.unload_pipeline();
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
            drop(s);
            try_init_and_update(&state_ref, &app_weak);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_remove_right_segment(move |idx| {
        let mut s = state_ref.borrow_mut();
        let removed = match s.right_input {
            Some(reco_io::stitch_job::InputPath::Single(_)) if idx == 0 => {
                s.right_input = None;
                s.right_path = None;
                true
            }
            Some(reco_io::stitch_job::InputPath::Chained(ref mut paths)) => {
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
                    true
                } else {
                    false
                }
            }
            _ => false,
        };
        if removed {
            s.unload_pipeline();
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
            drop(s);
            try_init_and_update(&state_ref, &app_weak);
        }
    });

    // Drag-to-reorder within a side. The segments are the same cameras in a
    // new temporal order, so the calibration still applies: reorder the
    // chained paths and rebuild the preview. The frame total is unchanged,
    // so the export trim is deliberately left as-is.
    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_reorder_left_segment(move |from, to| {
        let mut s = state_ref.borrow_mut();
        let (from, to) = (from as usize, to as usize);
        let new_first = match s.left_input {
            Some(reco_io::stitch_job::InputPath::Chained(ref mut paths)) => {
                if from < paths.len() && to < paths.len() && from != to {
                    let p = paths.remove(from);
                    paths.insert(to, p);
                    Some(paths[0].clone())
                } else {
                    None
                }
            }
            _ => None,
        };
        let Some(first) = new_first else {
            return;
        };
        log::info!("Left: reordered segment {from} -> {to}");
        s.left_path = Some(first);
        if let Some(app) = app_weak.upgrade() {
            if let Some(reco_io::stitch_job::InputPath::Chained(ps)) = s.left_input.as_ref() {
                app.set_left_path(
                    format!("{} ({} segments)", display_name(&ps[0]), ps.len()).into(),
                );
            }
            sync_segments(&s, &app);
        }
        drop(s);
        // Re-open the preview source off the drop handler: the list reorders
        // and repaints immediately, then the (slow) 4K concat re-probe runs on
        // the next tick instead of freezing the UI on every drop.
        let reopen = Rc::clone(&state_ref);
        let app_weak = app_weak.clone();
        slint::Timer::single_shot(std::time::Duration::from_millis(40), move || {
            if let Some(app) = app_weak.upgrade() {
                reopen.borrow_mut().reopen_source(&app);
            }
        });
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_reorder_right_segment(move |from, to| {
        let mut s = state_ref.borrow_mut();
        let (from, to) = (from as usize, to as usize);
        let new_first = match s.right_input {
            Some(reco_io::stitch_job::InputPath::Chained(ref mut paths)) => {
                if from < paths.len() && to < paths.len() && from != to {
                    let p = paths.remove(from);
                    paths.insert(to, p);
                    Some(paths[0].clone())
                } else {
                    None
                }
            }
            _ => None,
        };
        let Some(first) = new_first else {
            return;
        };
        log::info!("Right: reordered segment {from} -> {to}");
        s.right_path = Some(first);
        if let Some(app) = app_weak.upgrade() {
            if let Some(reco_io::stitch_job::InputPath::Chained(ps)) = s.right_input.as_ref() {
                app.set_right_path(
                    format!("{} ({} segments)", display_name(&ps[0]), ps.len()).into(),
                );
            }
            sync_segments(&s, &app);
        }
        drop(s);
        // Re-open the preview source off the drop handler (see left handler).
        let reopen = Rc::clone(&state_ref);
        let app_weak = app_weak.clone();
        slint::Timer::single_shot(std::time::Duration::from_millis(40), move || {
            if let Some(app) = app_weak.upgrade() {
                reopen.borrow_mut().reopen_source(&app);
            }
        });
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
        app.set_prefs_recording_codec(s.user_settings.recording_codec.clone().into());
        app.set_prefs_recording_quality(s.user_settings.recording_quality.clone().into());
        app.set_prefs_recording_folder(
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
        s.user_settings.recording_codec = app.get_prefs_recording_codec().to_string();
        s.user_settings.recording_quality = app.get_prefs_recording_quality().to_string();
        app.set_recording_codec(app.get_prefs_recording_codec());
        app.set_recording_quality(app.get_prefs_recording_quality());
        let rec_folder = app.get_prefs_recording_folder().to_string();
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
    let state_ref = Rc::clone(&state);
    app.on_pick_prefs_model(move || {
        begin_file_dialog(
            &state_ref,
            &app_weak,
            || {
                let task = rfd::AsyncFileDialog::new()
                    .set_title("Select YOLO ONNX model")
                    .add_filter("ONNX", &["onnx"])
                    .pick_file();
                async move { task.await.map(file_handle_to_path) }
            },
            |path, _, app_weak| {
                if let Some(app) = app_weak.upgrade() {
                    app.set_prefs_ai_model_path(path.to_string_lossy().to_string().into());
                }
            },
        );
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_pick_recording_folder(move || {
        begin_file_dialog(
            &state_ref,
            &app_weak,
            || {
                let task = rfd::AsyncFileDialog::new()
                    .set_title("Select default recording folder")
                    .pick_folder();
                async move { task.await.map(file_handle_to_path) }
            },
            |path, _, app_weak| {
                if let Some(app) = app_weak.upgrade() {
                    app.set_prefs_recording_folder(path.to_string_lossy().to_string().into());
                }
            },
        );
    });

    let state_ref = Rc::clone(&state);
    app.on_changed_preview_aspect(move |aspect| {
        let mut s = state_ref.borrow_mut();
        s.user_settings.preview_aspect = aspect.to_string();
        s.user_settings.save();
    });

    // ── Auto-calibration callback ──

    let state_ref = Rc::clone(&state);
    app.on_cancel_calibration(move || state_ref.borrow().cancel_calibration());

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_review_calibration(move || {
        let mut s = state_ref.borrow_mut();
        if let Some(app) = app_weak.upgrade() {
            app.set_calibration_needs_review(false);
        }
        if let Some((output, snapshot)) = s.review_calibration.take() {
            if snapshot.matches(&s)
                && !s.is_exporting()
                && !s.is_recording()
                && s.calibration_job.is_none()
            {
                handle_calibration_result(Ok(output), &mut s, &app_weak, true);
            } else {
                s.toasts.push(
                    Severity::Warn,
                    "Calibration discarded",
                    "Inputs changed. Run calibration again.",
                );
            }
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_auto_calibrate(move || {
        let s = state_ref.borrow();
        if s.calibration_job.is_some() || s.is_exporting() || s.is_recording() {
            return;
        }
        let (left, right) = match (&s.left_path, &s.right_path) {
            (Some(l), Some(r)) => (l.clone(), r.clone()),
            _ => return,
        };
        if [&s.left_input, &s.right_input]
            .into_iter()
            .flatten()
            .any(|input| matches!(input, reco_io::InputPath::Chained(_)))
        {
            if let Some(app) = app_weak.upgrade() {
                app.set_status_text(
                    "Select one file per camera to calibrate a segmented recording.".into(),
                );
            }
            return;
        }
        // Capture current playback position as calibration start time
        let current_time_secs = if s.playback.fps() > 0.0 {
            s.playback.frame_index() as f64 / s.playback.fps()
        } else {
            0.0
        };
        drop(s);

        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let cal_frames = app.get_calibration_frames() as usize;
        let skip_end = app.get_cal_skip_end_secs() as f64;
        let input = app.get_calibration_offset_input();
        let sync_offset = if input.trim().is_empty() {
            None
        } else if let Ok(offset) = input.trim().parse::<i64>() {
            Some(offset)
        } else {
            app.set_status_text("Enter a whole number for the synchronization offset.".into());
            return;
        };
        app.set_calibrating(true);
        app.set_calibration_needs_review(false);
        app.set_calibration_step("Starting...".into());
        app.set_status_text("Auto-calibrating...".into());

        let interrupted = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();

        let (existing_left_params, existing_right_params) = {
            let s = state_ref.borrow();
            let lenses = s
                .calibration
                .as_ref()
                .map(|c| c.lenses.as_slice())
                .unwrap_or_default();
            (
                lenses
                    .first()
                    .filter(|_| app.get_fix_left_optics())
                    .cloned(),
                lenses
                    .get(1)
                    .filter(|_| app.get_fix_right_optics())
                    .cloned(),
            )
        };

        {
            let mut s = state_ref.borrow_mut();
            s.calibration_job = Some(calibration_job::PendingCalibration::new(
                &s,
                rx,
                Arc::clone(&interrupted),
            ));
            s.review_calibration = None;
        }

        // Run calibration on a background thread. Only Send types
        // (PathBuf, channel, AtomicBool, Weak) cross the boundary.
        let app_weak_bg = app_weak.clone();
        std::thread::spawn(move || {
            let app_weak_progress = app_weak_bg.clone();
            let config = reco_calibrate::CalibrationConfig {
                num_frames: cal_frames,
                skip_start_secs: current_time_secs,
                skip_end_secs: skip_end,
            };
            let result = reco_calibrate::video::calibrate_videos(
                &left,
                &right,
                reco_calibrate::video::CalibrateVideosOptions {
                    config: Some(config),
                    left_params: existing_left_params,
                    right_params: existing_right_params,
                    sync_offset,
                },
                &mut |progress| {
                    let step_name = format!("{:?}", progress.step);
                    let detail = progress.detail.clone();
                    let weak = app_weak_progress.clone();
                    let active = Arc::clone(&interrupted);
                    slint::invoke_from_event_loop(move || {
                        if !active.load(Ordering::Relaxed)
                            && let Some(app) = weak.upgrade()
                        {
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
                        assessment: r.assessment,
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
        if s.is_exporting() || s.playback.is_live() {
            return;
        }
        if s.playback.state() == PlayState::Playing {
            s.playback.toggle();
        }
        let target = s.playback.requested_frame().saturating_add(1);
        if let Err(e) = s.playback.seek_frame(target) {
            log::error!("Step forward error: {e}");
        }
        if let Some(app) = app_weak.upgrade() {
            app.set_playing(false);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_step_backward(move || {
        let mut s = state_ref.borrow_mut();
        if s.is_exporting() || s.playback.is_live() {
            return;
        }
        if s.playback.state() == PlayState::Playing {
            s.playback.toggle();
        }
        let target = s.playback.requested_frame().saturating_sub(1);
        if let Err(e) = s.playback.seek_frame(target) {
            log::error!("Step backward error: {e}");
        }
        if let Some(app) = app_weak.upgrade() {
            app.set_playing(false);
        }
    });

    let state_ref = Rc::clone(&state);
    app.on_seek_frame(move |frame| {
        let mut s = state_ref.borrow_mut();
        if s.is_exporting() || frame < 0 {
            return;
        }
        // Slint emits this only on pointer/key release, never from playback updates.
        if let Err(e) = s.playback.seek_frame(frame as u64) {
            log::error!("Seek error: {e}");
        }
    });

    // ── Camera / view control callbacks ──
    //
    // Fit input immediately, then let the rendering notifier coalesce redraws.

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_pan(move |dx_px, dy_px| {
        if let Some(app) = app_weak.upgrade() {
            state_ref.borrow_mut().apply_pan(
                dx_px,
                dy_px,
                app.get_preview_area_width(),
                app.get_preview_area_height(),
            );
        }
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
    let app_weak = app.as_weak();
    app.on_changed_blend_width(move |v| {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        state_ref.borrow_mut().edit_calibration(&app, |cal| {
            if let Some(layout) = cal.projection.l_shape_mut() {
                layout.blend_width = v;
            }
        });
    });

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_changed_rig_tilt(move |v| {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        state_ref.borrow_mut().edit_calibration(&app, |cal| {
            cal.framing.tilt = (v as f64).to_radians();
        });
    });

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_changed_rig_roll(move |v| {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        state_ref.borrow_mut().edit_calibration(&app, |cal| {
            cal.framing.roll = (v as f64).to_radians();
        });
    });

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_changed_sync_offset(move |frames| {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        state_ref.borrow_mut().set_sync_offset(&app, frames);
    });

    let state_ref = Rc::clone(&state);
    app.on_changed_fov(move |deg| {
        state_ref.borrow_mut().set_fov(deg);
    });

    let state_ref = Rc::clone(&state);
    app.on_seek_relative(move |secs| {
        let mut s = state_ref.borrow_mut();
        if let Err(e) = s.seek_relative(secs) {
            log::error!("Seek relative error: {e}");
        }
    });

    // ── Live calibration editing callbacks ──
    //
    // Accepted document edits share validation, renderer updates and unsaved-state tracking.

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_changed_cal_intersect(move |v| {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        state_ref.borrow_mut().edit_calibration(&app, |cal| {
            if let Some(layout) = cal.projection.l_shape_mut() {
                layout.intersect = v as f64;
            }
        });
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_changed_cal_camera_axis_offset(move |v| {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        state_ref.borrow_mut().edit_calibration(&app, |cal| {
            cal.framing.axis_offset = v as f64;
        });
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_changed_cal_x_ty(move |v| {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        state_ref.borrow_mut().edit_calibration(&app, |cal| {
            if let Some(layout) = cal.projection.l_shape_mut() {
                layout.x_ty = v as f64;
            }
        });
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
                    // Clear BOTH dirty flags: the Save button is shown on
                    // `cal-dirty || lens-dirty`, so leaving lens-dirty set
                    // kept the button visible after a successful save.
                    app.set_cal_dirty(false);
                    app.set_lens_dirty(false);
                    s.toasts.push(Severity::Info, "Calibration saved", "");
                    crate::toast::sync_to_ui(&s.toasts, &app);
                }
            }
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_reset_calibration(move || {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let mut s = state_ref.borrow_mut();
        let Some(base) = s.cal_baseline.clone() else {
            return;
        };
        s.edit_calibration(&app, |cal| {
            cal.projection = base.projection;
            cal.framing = base.framing;
        });
    });

    // ── Live lens tuning callbacks ──
    //
    // Lens sliders edit intrinsics while preserving dimensions and correction.

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_changed_lens_param(move || {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let mut s = state_ref.borrow_mut();
        let selected = app.get_lens_selected_camera();
        // Slider edits replace only the intrinsics; everything else on the
        // lens (dims, correction strength) is preserved from the current
        // document lens - rebuilding via a constructor here used to reset
        // the user's lens-correction toggle to full.
        let (fx, fy, cx, cy, distortion) = if selected == "right" {
            (
                app.get_lens_right_fx(),
                app.get_lens_right_fy(),
                app.get_lens_right_cx(),
                app.get_lens_right_cy(),
                [
                    app.get_lens_right_k1(),
                    app.get_lens_right_k2(),
                    app.get_lens_right_k3(),
                    app.get_lens_right_k4(),
                ],
            )
        } else {
            (
                app.get_lens_left_fx(),
                app.get_lens_left_fy(),
                app.get_lens_left_cx(),
                app.get_lens_left_cy(),
                [
                    app.get_lens_left_k1(),
                    app.get_lens_left_k2(),
                    app.get_lens_left_k3(),
                    app.get_lens_left_k4(),
                ],
            )
        };
        if s.edit_calibration(&app, |cal| {
            for (camera, lens) in cal.lenses.iter_mut().enumerate() {
                if selected == "both" || camera == usize::from(selected == "right") {
                    lens.fx = f64::from(fx);
                    lens.fy = f64::from(fy);
                    lens.cx = f64::from(cx);
                    lens.cy = f64::from(cy);
                    lens.distortion = distortion.map(f64::from);
                }
            }
        }) {
            app.set_lens_dirty(true);
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_reset_lens(move || {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let mut s = state_ref.borrow_mut();
        if let Some(lenses) = s.cal_baseline.as_ref().map(|cal| cal.lenses.clone())
            && s.edit_calibration(&app, |cal| cal.lenses = lenses)
        {
            s.sync_calibration_controls(&app);
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
                let Some(app) = app_ref.as_ref() else {
                    return;
                };
                if !s.apply_lens_profile(app, &params, side_str) {
                    return;
                }
                if let Some(app) = app_ref.as_ref() {
                    if side_str != "right" {
                        app.set_fix_left_optics(true);
                        app.set_lens_left_camera(summary.camera.clone().into());
                        app.set_lens_left_source("Picker".into());
                    }
                    if side_str != "left" {
                        app.set_fix_right_optics(true);
                        app.set_lens_right_camera(summary.camera.clone().into());
                        app.set_lens_right_source("Picker".into());
                    }
                    s.sync_calibration_controls(app);
                    app.set_lens_dirty(true);
                    app.set_lens_info_available(true);
                }
            }
        }
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_lens_pick_file(move || {
        begin_file_dialog(
            &state_ref,
            &app_weak,
            || {
                let task = rfd::AsyncFileDialog::new()
                    .set_title("Load lens profile JSON")
                    .add_filter("JSON", &["json"])
                    .pick_file();
                async move { task.await.map(file_handle_to_path) }
            },
            |path, state_ref, app_weak| match reco_calibrate::lens_database::load_from_file(&path) {
                Ok(params) => {
                    let mut s = state_ref.borrow_mut();
                    let Some(app) = app_weak.upgrade() else {
                        return;
                    };
                    if !s.apply_lens_profile(&app, &params, "both") {
                        return;
                    }
                    app.set_fix_left_optics(true);
                    app.set_fix_right_optics(true);
                    if let Some(app) = app_weak.upgrade() {
                        s.sync_calibration_controls(&app);
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
            },
        );
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
        field_editor::sync(&mut s, &app);
        s.preview_dirty = true;
    });

    let state_ref = Rc::clone(&state);
    let app_weak = app.as_weak();
    app.on_changed_lens_correction(move |amount| {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let mut s = state_ref.borrow_mut();
        let correction = if amount > 0.5 { 1.0 } else { 0.0 };
        s.edit_calibration(&app, |cal| {
            for lens in &mut cal.lenses {
                lens.correction = correction;
            }
        });
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
            app.set_export_codec(if app.get_export_stream_enabled() {
                "h264".into()
            } else {
                s.user_settings.default_codec.clone().into()
            });
            app.set_export_quality(s.user_settings.default_quality.clone().into());
            let count = s.calibration.as_ref().map_or(0, |cal| cal.lenses.len());
            let choices: Vec<slint::SharedString> = std::iter::once("No source audio".into())
                .chain((1..=count).map(|index| format!("Camera {index}").into()))
                .collect();
            app.set_export_audio_options(std::rc::Rc::new(slint::VecModel::from(choices)).into());
            if app.get_export_audio_choice() as usize > count {
                app.set_export_audio_choice(0);
            }
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
    let state_ref = Rc::clone(&state);
    app.on_pick_export_output(move || {
        begin_file_dialog(
            &state_ref,
            &app_weak,
            || {
                let task = rfd::AsyncFileDialog::new()
                    .set_title("Export stitched video to...")
                    .add_filter("MP4", &["mp4"])
                    .add_filter("MOV", &["mov"])
                    .add_filter("MKV", &["mkv"])
                    .save_file();
                async move { task.await.map(file_handle_to_path) }
            },
            |mut path, _, app_weak| {
                // Ensure an extension - ffmpeg picks muxer by extension.
                if path.extension().is_none() {
                    path.set_extension("mp4");
                }
                if let Some(app) = app_weak.upgrade() {
                    app.set_export_output_path(path.to_string_lossy().to_string().into());
                }
            },
        );
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_pick_export_model(move || {
        begin_file_dialog(
            &state_ref,
            &app_weak,
            || {
                let task = rfd::AsyncFileDialog::new()
                    .set_title("Select YOLO ONNX model")
                    .add_filter("ONNX", &["onnx"])
                    .pick_file();
                async move { task.await.map(file_handle_to_path) }
            },
            |path, state_ref, app_weak| {
                if let Some(app) = app_weak.upgrade() {
                    app.set_export_model_path(path.to_string_lossy().to_string().into());
                    // Remember across sessions so the user doesn't re-pick
                    // the same ONNX every run. Save is best-effort.
                    let mut s = state_ref.borrow_mut();
                    s.user_settings.ai_model_path = Some(path);
                    s.user_settings.save();
                }
            },
        );
    });

    let app_weak = app.as_weak();
    app.on_apply_panner_preset(move |name| {
        #[cfg(feature = "autocam")]
        if let Some(app) = app_weak.upgrade() {
            use reco_autocam::panners::{ClusterMode, FieldPannerConfig, FramingMode};
            let cfg = FieldPannerConfig::from_preset_name(name.as_str()).unwrap_or_default();
            let framing = if cfg.framing == FramingMode::FrameAll {
                "frame_all"
            } else {
                "action"
            };
            let cluster = if cfg.cluster_mode == ClusterMode::TrimmedMean {
                "trimmed_mean"
            } else {
                "density"
            };
            app.set_export_framing(framing.into());
            app.set_export_cluster_mode(cluster.into());
            app.set_export_lock_pitch(cfg.lock_pitch);
            app.set_export_cluster_bandwidth(cfg.cluster_bandwidth_rad);
            app.set_export_dead_zone(cfg.dead_zone_rad);
            app.set_export_ball_weight(cfg.ball_weight);
            app.set_export_fov_tight(cfg.fov_tight);
            app.set_export_fov_wide(cfg.fov_wide);
            app.set_export_fov_default(cfg.fov_default);
            app.set_export_frame_all_margin(cfg.frame_all_margin_deg);
        }
        #[cfg(not(feature = "autocam"))]
        let _ = (&app_weak, &name);
    });

    let app_weak = app.as_weak();
    let state_ref = Rc::clone(&state);
    app.on_start_export(move || {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let output_str = app.get_export_output_path().to_string();
        if output_str.is_empty() {
            log::warn!("Export: no output path set, ignoring");
            return;
        }
        log::info!("Export requested: output={output_str}");
        let mut s = state_ref.borrow_mut();
        if s.export_thread.is_some() {
            log::warn!("Export already running, ignoring start request");
            return;
        }
        if s.is_recording() {
            app.set_export_error_text("Stop quick recording before starting export.".into());
            return;
        }
        if s.calibration_job.is_some() {
            app.set_export_error_text("Wait for calibration to finish before exporting.".into());
            return;
        }
        if s.network.is_opening() {
            app.set_export_error_text("Wait for the RTSP connection to finish or cancel it.".into());
            return;
        }
        let output_path = PathBuf::from(&output_str);
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                log::warn!("Export: output directory does not exist: {}", parent.display());
                app.set_export_error_text(
                    format!("Output directory does not exist: {}", parent.display()).into(),
                );
                return;
            }
            if parent
                .metadata()
                .is_ok_and(|m| m.permissions().readonly())
            {
                log::warn!(
                    "Export: output directory has read-only attribute: {} (may be normal on Windows)",
                    parent.display()
                );
            }
        }
        let Some(cal) = s.calibration.clone() else {
            log::error!("Cannot start export without calibration");
            return;
        };
        let inputs = if let Some(config) = &s.network.active {
            crate::export::Inputs::Network(config.clone())
        } else if let Some(inputs) = s.selected_inputs(&cal) {
            crate::export::Inputs::Files(inputs)
        } else {
            log::error!("Cannot start export without the calibrated inputs");
            return;
        };

        // Snapshot all export settings. Slint properties must not be
        // read from the worker thread — only the UI thread owns them.
        let output = PathBuf::from(output_str);
        let width = app.get_export_width() as u32;
        let height = app.get_export_height() as u32;
        let codec_str = app.get_export_codec().to_string();
        let quality_str = app.get_export_quality().to_string();
        let stream_url = app.get_export_stream_enabled().then(|| app.get_export_stream_url().trim().to_string());
        if let Err(error) = crate::export::validate_stream(stream_url.as_deref(), &codec_str) {
            app.set_export_error_text(error.into());
            return;
        }
        let audio = if app.get_export_audio_choice() == 0 { reco_io::output::AudioMode::Disabled }
            else { reco_io::output::AudioMode::CopyFrom(app.get_export_audio_choice().saturating_sub(1) as usize) };
        let blend = app.get_blend_width();
        let start_secs = if s.network.active.is_some() { 0.0 } else { app.get_export_start_secs() };
        let end_secs = if s.network.active.is_some() { 0.0 } else { app.get_export_end_secs() };
        log::info!("Export range: start={start_secs:.1}s, end={end_secs:.1}s");
        let autocam = crate::export::AutocamUiConfig {
            enabled: app.get_export_autocam_enabled(),
            model_path: app.get_export_model_path().to_string(),
            tracking_mode: app.get_export_tracking_mode().to_string(),
            detection_interval: app.get_export_detection_interval() as u32,
            lookahead_secs: app.get_export_lookahead_secs() as f64,
            preset: app.get_export_panner_preset().to_string(),
            framing: app.get_export_framing().to_string(),
            lock_pitch: app.get_export_lock_pitch(),
            cluster_mode: app.get_export_cluster_mode().to_string(),
            cluster_bandwidth_rad: app.get_export_cluster_bandwidth(),
            dead_zone_rad: app.get_export_dead_zone(),
            ball_weight: app.get_export_ball_weight(),
            fov_tight: app.get_export_fov_tight(),
            fov_wide: app.get_export_fov_wide(),
            fov_default: app.get_export_fov_default(),
            frame_all_margin_deg: app.get_export_frame_all_margin(),
        };
        let replay_enabled = app.get_export_replay_enabled();
        let events_enabled = app.get_export_events_enabled();
        let events_path = if events_enabled {
            Some(output_path.with_extension("events.jsonl"))
        } else {
            None
        };

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
        s.export_stream = crate::export::StreamState::default();
        let stream_monitor = s.export_stream.monitor();
        app.set_export_stream_status(if stream_url.is_some() { "Stream: waiting for output…".into() } else { "".into() });
        app.set_export_stream_failed(false);

        // Pause preview playback to avoid GPU contention with the
        // export pipeline. Preview rendering is also gated by
        // is_exporting() in vsync_render_tick.
        s.playback.pause();
        app.set_playing(false);

        // Release the preview pipeline so its VRAM is free for the export;
        // rebuilt on completion. run_export uses its own source.
        log::info!("Releasing preview GPU pipeline to free VRAM for export");
        s.suspend_preview_for_export();
        let preview_cleanup = std::mem::take(&mut s.decoder_cleanup);

        app.set_export_error_text("".into());
        app.set_export_in_progress(true);
        app.set_export_progress(0.0);
        app.set_telem_fps_avg(0.0);
        app.set_telem_fps_recent(0.0);
        app.set_export_frames_done(0);
        app.set_export_frames_total(0);
        app.set_export_status_text("Initializing…".into());
        app.set_export_dialog_open(false);

        let app_weak_bg = app_weak.clone();
        let output_for_thread = output.clone();
        let handle = std::thread::spawn(move || {
            // Close the preview connections before opening output inputs, off the UI thread.
            drop(preview_cleanup);
            let outcome = crate::export::run_export(
                crate::export::ExportRequest {
                    inputs, calibration: cal, output: output_for_thread, stream_url, audio,
                    replay_enabled, events_path, width, height, codec: codec_str, quality: quality_str,
                    start_secs, end_secs, autocam,
                },
                stream_monitor,
                app_weak_bg,
                &interrupted,
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
        if s.network.active.is_some() {
            return;
        }
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
        if let Some(client) = &s.telemetry {
            client.bug_report(&report);
        } else {
            telemetry_client::TelemetryClient::new(cid).bug_report(&report);
        }

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
            "Report queued",
            "The report has been queued for delivery.",
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
            // Dev/test preload (RECO_AUTOLOAD): fire once, after the GPU
            // device has been captured by the rendering notifier.
            #[cfg(feature = "automation")]
            {
                let ready = {
                    let s = state_ref.borrow();
                    s.autoload.is_some() && s.shared_gpu.is_some()
                };
                if ready {
                    let spec = state_ref.borrow_mut().autoload.take();
                    if let Some(spec) = spec {
                        run_autoload(&state_ref, &app_weak, spec);
                    }
                }
            }

            // Poll native file dialogs. The OS dialog waits off the
            // Slint event loop, but path application stays on this UI
            // thread because AppState is Rc<RefCell<_>>.
            if poll_file_dialog(&state_ref, &app_weak) {
                return;
            }

            let mut s = state_ref.borrow_mut();

            network::poll(&mut s, &app_weak);
            if let Some(app) = app_weak.upgrade() {
                app.set_network_active(s.network.active.is_some());
                app.set_network_opening(s.network.is_opening());
            }

            // Seek readiness is independent of display cadence; paused seeks
            // must progress without submitting unchanged preview frames.
            if s.playback.is_seeking() {
                match s.playback.poll_seek() {
                    Ok(ready) => s.preview_dirty |= ready,
                    Err(error) => {
                        log::error!("Seek failed: {error}");
                        if let Some(app) = app_weak.upgrade() {
                            app.set_status_text(format!("Seek failed: {error}").into());
                        }
                    }
                }
            }

            // Check for update notification from the background thread.
            if let Ok(mut guard) = update_check.try_lock()
                && let Some(tag) = guard.take()
                && let Some(app) = app_weak.upgrade()
            {
                s.toasts.push_with_ttl(
                    Severity::Info,
                    format!("Update available: {tag}"),
                    "Download it from the project website under Help.",
                    Duration::from_secs(30),
                );
                log::info!("Toast pushed for update {tag}");
                crate::toast::sync_to_ui(&s.toasts, &app);
            }

            // Poll for calibration results from the background thread.
            if s.poll_calibration(&app_weak) {
                return;
            }

            // Poll the export worker for completion.
            if let Some(app) = app_weak.upgrade() { s.export_stream.update(&app); }
            if let Some(rx) = &s.export_rx
                && let Some(outcome) = crate::export::poll_export(rx)
            {
                s.export_rx = None;
                if let Some(h) = s.export_thread.take() {
                    let _ = h.join();
                }
                // Rebuild the preview from the in-memory calibration, on any
                // outcome.
                if let Err(e) = s.restore_preview_after_export() {
                    log::warn!("Failed to rebuild preview after export: {e}");
                    s.unload_pipeline();
                    s.toasts.push(Severity::Error, "Could not restore preview", e);
                    if let Some(app) = app_weak.upgrade() {
                        app.set_files_loaded(false);
                        app.set_preview_frame(slint::Image::default());
                        crate::toast::sync_to_ui(&s.toasts, &app);
                    }
                }
                if let Some(app) = app_weak.upgrade() {
                    s.export_stream.finish(&app);
                    if s.network.active.is_some() {
                        app.set_files_loaded(false);
                        app.set_preview_frame(slint::Image::default());
                        app.set_playing(false);
                    }
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
                                if app.get_export_stream_failed() { Severity::Warn } else { Severity::Info },
                                if app.get_export_stream_failed() { "Recording saved; streaming failed" } else { "Export complete" },
                                format!("{frames} frames to {}", path.display()),
                            );
                            crate::toast::sync_to_ui(&s.toasts, &app);

                            // Repeat-export driver: measure VRAM now that the
                            // preview has been rebuilt (the steady state between
                            // runs), then either trigger the next export or quit.
                            // A leak shows up as `used` that never returns to the
                            // baseline logged at startup. Deferred via single_shot
                            // so the next start_export runs after this borrow of
                            // `s` is released (start_export borrows it too).
                            #[cfg(feature = "automation")]
                            if s.auto_export_mode {
                                s.auto_export_done += 1;
                                let done = s.auto_export_done;
                                let total = s.auto_export_total;
                                let vram = autoexport_vram();
                                if done < total {
                                    log::info!(
                                        "RECO_AUTOEXPORT: run {done}/{total} complete, {vram}; starting next"
                                    );
                                    let base = s.auto_export_base.clone();
                                    let app_w = app_weak.clone();
                                    slint::Timer::single_shot(
                                        std::time::Duration::from_millis(800),
                                        move || {
                                            if let Some(app) = app_w.upgrade() {
                                                if let Some(base) = base {
                                                    let out = autoexport_run_path(&base, done + 1);
                                                    app.set_export_output_path(
                                                        out.display().to_string().into(),
                                                    );
                                                }
                                                app.invoke_start_export();
                                            }
                                        },
                                    );
                                } else {
                                    log::info!(
                                        "RECO_AUTOEXPORT: all {total} runs complete, final {vram}; quitting"
                                    );
                                    slint::Timer::single_shot(
                                        std::time::Duration::from_millis(1500),
                                        || {
                                            let _ = slint::quit_event_loop();
                                        },
                                    );
                                }
                            }
                        }
                        ExportOutcome::Cancelled => {
                            app.set_export_status_text("".into());
                            app.set_status_text("Export cancelled".into());
                            #[cfg(feature = "automation")]
                            if s.auto_export_mode {
                                log::warn!(
                                    "RECO_AUTOEXPORT: run cancelled, {}; quitting",
                                    autoexport_vram()
                                );
                                slint::Timer::single_shot(
                                    std::time::Duration::from_millis(1000),
                                    || {
                                        let _ = slint::quit_event_loop();
                                    },
                                );
                            }
                        }
                        ExportOutcome::Failed(err) => {
                            app.set_export_status_text("".into());
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
                                reco_io::stitch_job::StitchError::Session(
                                    reco_core::session::types::SessionError::Config(detail),
                                )
                                | reco_io::stitch_job::StitchError::Session(
                                    reco_core::session::types::SessionError::Core(
                                        reco_core::core::types::StitchCoreError::Config(detail),
                                    ),
                                ) => ("Export failed", detail.clone()),
                                other => (
                                    "Export failed",
                                    format!("{other}"),
                                ),
                            };
                            // Retain the detailed status before moving the body into the toast.
                            let status = format!("Export failed - {body}");
                            let msg = format!("{err}");
                            if let Some(ref t) = s.telemetry {
                                let codec = app.get_export_codec().to_string();
                                t.export_error(&msg, &codec);
                            }
                            s.toasts.push(Severity::Error, title, body);
                            crate::toast::sync_to_ui(&s.toasts, &app);
                            app.set_status_text(status.into());
                            #[cfg(feature = "automation")]
                            if s.auto_export_mode {
                                log::warn!(
                                    "RECO_AUTOEXPORT: run failed ({msg}), {}; quitting",
                                    autoexport_vram()
                                );
                                slint::Timer::single_shot(
                                    std::time::Duration::from_millis(1000),
                                    || {
                                        let _ = slint::quit_event_loop();
                                    },
                                );
                            }
                        }
                    }
                }
                return;
            }

            // Finalizing is now signaled explicitly by the export thread
            // via StitchJob::on_finalizing(), no timer heuristic needed.

            // Persist window size, debounced (Tier 3d).
            if let Some(app) = app_weak.upgrade() {
                let size = app.window().size();
                let cur = (size.width, size.height);
                let maximized = app.window().is_maximized();
                s.user_settings.window_maximized = maximized;
                let last = s.last_persisted_window_size.unwrap_or((0, 0));
                if cur != last {
                    s.last_window_size_save_at = Some(Instant::now());
                    s.last_persisted_window_size = Some(cur);
                    if !maximized {
                        s.user_settings.window_size = Some(cur);
                    }
                } else if let Some(since) = s.last_window_size_save_at
                    && since.elapsed() > Duration::from_secs(2)
                {
                    s.user_settings.save();
                    s.last_window_size_save_at = None;
                }
            }

            // Expire aged toasts (Tier 3a).
            if !s.toasts.is_empty()
                && s.toasts.expire(Instant::now())
                && let Some(app) = app_weak.upgrade()
            {
                crate::toast::sync_to_ui(&s.toasts, &app);
            }

            // Playback, pose fitting, and rendering happen in
            // `vsync_render_tick` (driven by Slint's BeforeRendering
            // notifier), so this timer only handles work that does
            // not need vsync alignment. When playback is active we
            // nudge Slint to keep redrawing so BeforeRendering fires
            // even if nothing marked the window dirty yet.
            if let Some(app) = app_weak.upgrade()
                && (app.get_playing() || s.preview_dirty || s.coverage_pending)
            {
                app.window().request_redraw();
            }
        },
    );

    app.run()?;
    Ok(())
}

/// Advance playback and render from Slint's `BeforeRendering` notifier.
/// Returns `true` when a frame was submitted; display cadence depends on the backend.
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
            match bridge.resize(area_w, area_h) {
                Ok(()) => s.preview_dirty = true,
                Err(error) => app.set_status_text(format!("Cannot resize preview: {error}").into()),
            }
        }
    }

    let camera_changed = s.resolve_camera();
    if let Some(app) = app_weak.upgrade() {
        const PREPARING: &str = "Updating coverage… preview may show black edges";
        if s.coverage_pending {
            if app.get_status_text() != PREPARING {
                app.set_status_text(PREPARING.into());
            }
        } else if app.get_status_text() == PREPARING {
            app.set_status_text(if s.coverage_error_reported {
                "Coverage unavailable; preview remains unconstrained".into()
            } else {
                "Coverage ready".into()
            });
        }
    }
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

    if let Some(app) = app_weak.upgrade()
        && app.get_playing() != (s.playback.state() == PlayState::Playing)
    {
        app.set_playing(s.playback.state() == PlayState::Playing);
        if s.playback.state() == PlayState::Finished {
            app.set_status_text("Playback finished".into());
        }
    }

    let was_dirty = s.preview_dirty;
    // Poll readiness while paused without submitting unchanged video frames.
    s.preview_dirty = false;
    if !(camera_changed || video_advanced || was_dirty) {
        return false;
    }

    let img = s.render_current();
    if let (Some(app), Some(img)) = (app_weak.upgrade(), img) {
        s.displayed_view = (!s.lens_preview_active).then_some((
            ViewportSize {
                width: img.size().width,
                height: img.size().height,
            },
            s.pose.current_pose(),
        ));
        app.set_preview_frame(img);
        field_editor::sync(&mut s, &app);
        s.last_render_at = Some(Instant::now());
        let fps = s.playback.fps();
        let total = s.playback.total_frames().unwrap_or(0);
        sync_frame_display(&app, s.playback.frame_index(), total, fps);
        // Show the accepted pose, including any geometric FOV cap.
        let current = s.pose.current_pose();
        app.set_yaw(current.yaw);
        app.set_pitch(current.pitch);
        app.set_fov(current.fov_degrees);
        return true;
    }
    false
}

/// GPU free/used VRAM (MB) via `nvidia-smi`, for repeat-export leak logging.
/// Returns a short descriptive string; never fails the run.
#[cfg(feature = "automation")]
fn autoexport_vram() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| format!("free/used MB = {}", s.trim()))
        .unwrap_or_else(|| "nvidia-smi unavailable".into())
}

/// Output path for repeat-export run `n`: `base_n.ext` (1-based), so successive
/// runs in one session do not overwrite each other.
#[cfg(feature = "automation")]
fn autoexport_run_path(base: &std::path::Path, n: u32) -> PathBuf {
    let stem = base
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("export");
    let ext = base.extension().and_then(|e| e.to_str()).unwrap_or("mp4");
    let name = format!("{stem}_{n}.{ext}");
    base.with_file_name(name)
}

/// Execute an [`AutoloadSpec`]: set inputs + calibration, build the preview,
/// and optionally start an export. Dev/test hook (RECO_AUTOLOAD) only.
#[cfg(feature = "automation")]
fn run_autoload(
    state: &Rc<RefCell<AppState>>,
    app_weak: &slint::Weak<RecoApp>,
    spec: AutoloadSpec,
) {
    log::info!(
        "RECO_AUTOLOAD: preloading {} left + {} right segment(s), cal {}",
        spec.left.len(),
        spec.right.len(),
        spec.cal.display()
    );
    let make_input =
        |paths: &[PathBuf]| reco_io::stitch_job::InputPath::from_segments(paths.to_vec());
    {
        let mut s = state.borrow_mut();
        s.left_input = Some(make_input(&spec.left));
        s.left_path = Some(spec.left[0].clone());
        s.right_input = (!spec.right.is_empty()).then(|| make_input(&spec.right));
        s.right_path = spec.right.first().cloned();
        if let Some(app) = app_weak.upgrade() {
            app.set_left_path(display_name(&spec.left[0]).into());
            app.set_right_path(
                spec.right
                    .first()
                    .map(|p| display_name(p))
                    .unwrap_or_default()
                    .into(),
            );
        }
    }
    if !load_calibration_and_update(state, app_weak, spec.cal) {
        return;
    }
    if let Some(exp) = spec.export
        && let Some(app) = app_weak.upgrade()
    {
        // Repeat-export driver: run `exp.repeats` exports back-to-back in this
        // one session so a cross-export GPU leak shows up as VRAM that never
        // returns to baseline. The completion handler re-triggers the next run
        // and quits when done. `remaining` counts runs AFTER this first one.
        {
            let mut s = state.borrow_mut();
            s.auto_export_mode = true;
            s.auto_export_total = exp.repeats;
            s.auto_export_done = 0;
            s.auto_export_base = Some(exp.output.clone());
        }
        let first_out = autoexport_run_path(&exp.output, 1);
        app.set_export_output_path(first_out.display().to_string().into());
        if let Some(model) = &exp.model {
            app.set_export_autocam_enabled(true);
            app.set_export_model_path(model.display().to_string().into());
            app.set_export_tracking_mode("field".into());
            app.set_export_lookahead_secs(exp.lookahead_secs);
        }
        log::info!(
            "RECO_AUTOEXPORT: baseline {}; starting export 1/{} -> {} (lookahead {}s)",
            autoexport_vram(),
            exp.repeats,
            first_out.display(),
            exp.lookahead_secs
        );
        app.invoke_start_export();
    }
}

fn try_init_and_update(state: &Rc<RefCell<AppState>>, app_weak: &slint::Weak<RecoApp>) {
    let mut s = state.borrow_mut();
    if let Some(app) = app_weak.upgrade() {
        sync_segments(&s, &app);
    }
    // Capture the pre-init clip length so we can distinguish an input
    // change (new load / appended segments) from a calibration-only
    // reload further down.
    let prev_total = s.playback.total_frames();
    let result = s.try_init();
    if result.is_err() {
        // Selected inputs now differ from the old playback. Do not keep rendering that old source.
        s.unload_pipeline();
    }
    update_preview_ui(&mut s, app_weak, result, prev_total);
}

fn load_calibration_and_update(
    state: &Rc<RefCell<AppState>>,
    app_weak: &slint::Weak<RecoApp>,
    path: PathBuf,
) -> bool {
    let mut s = state.borrow_mut();
    let prev_total = s.playback.total_frames();
    match s.load_calibration(path.clone()) {
        Ok(ready) => {
            s.user_settings.push_calibration(path.clone());
            if let Some(app) = app_weak.upgrade() {
                app.set_calibration_path(display_name(&path).into());
                app.set_cal_dirty(false);
                app.set_lens_dirty(false);
                app.set_files_loaded(ready);
                s.sync_calibration_controls(&app);
                app.set_fix_left_optics(false);
                app.set_fix_right_optics(false);
                set_lens_profile_props(&app, None, None);
                sync_recent_paths(&s.user_settings, &app);
                sync_segments(&s, &app);
            }
            update_preview_ui(&mut s, app_weak, Ok(ready), prev_total);
            ready
        }
        Err(error) => {
            log::error!("Calibration load failed: {error}");
            if let Some(app) = app_weak.upgrade() {
                s.toasts
                    .push(Severity::Error, "Calibration unchanged", error);
                crate::toast::sync_to_ui(&s.toasts, &app);
            }
            false
        }
    }
}

fn update_preview_ui(
    s: &mut AppState,
    app_weak: &slint::Weak<RecoApp>,
    result: Result<bool, String>,
    prev_total: Option<u64>,
) {
    if let Some(app) = app_weak.upgrade() {
        app.set_recording(s.is_recording());
    }
    match result {
        Ok(true) => {
            let fps = s.playback.fps();
            let total = s.playback.total_frames().unwrap_or(0);

            s.clamp_targets();
            let clamped_fov = s.pose.current_fov_deg();
            let img = s.render_current();
            if let Some(app) = app_weak.upgrade() {
                app.set_files_loaded(true);
                app.set_has_roi(
                    s.calibration
                        .as_ref()
                        .and_then(|c| c.field_roi.as_ref())
                        .is_some(),
                );
                sync_frame_display(&app, s.playback.frame_index(), total, fps);
                app.set_fps(fps as f32);
                app.set_status_text(format!("Ready - {:.0} fps - {total} frames", fps).into());
                // The export trim defaults to the whole clip. When the input
                // length changes (new file, appended segments, or a shorter
                // clip) a previously seeded trim end would stick to the old
                // duration and silently truncate the export, so refresh it
                // here. A manual trim survives a calibration-only reload,
                // where the frame total is unchanged.
                if prev_total != Some(total) && fps > 0.0 {
                    let clip_secs = total as f32 / fps as f32;
                    app.set_clip_duration_secs(clip_secs);
                    app.set_export_start_secs(0.0);
                    app.set_export_end_secs(clip_secs);
                    log::info!(
                        "Export trim reset to full clip ({clip_secs:.1}s, {total} frames) after input change"
                    );
                }
                if let Some(img) = img {
                    s.displayed_view = (!s.lens_preview_active).then_some((
                        ViewportSize {
                            width: img.size().width,
                            height: img.size().height,
                        },
                        s.pose.current_pose(),
                    ));
                    app.set_preview_frame(img);
                    field_editor::sync(s, &app);
                }
                app.set_fov(clamped_fov);
                // Lens fine-tune sliders are seeded from the loaded
                // calibration's camera params either way; the Lens
                // fine-tune section is gated on `files-loaded` in Slint.
                s.sync_calibration_controls(&app);
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
                            let g = b.engine().gpu();
                            format!("{} ({:?})", g.gpu_name(), g.backend_name())
                        })
                        .unwrap_or_else(|| "unknown".into());
                    let os = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
                    let (ai_status, _) = ai_capability_summary();
                    t.context(&gpu, &os, &ai_status);
                    let (w, h) = s.playback.input_dimensions().unwrap_or((0, 0));
                    let fps = s.playback.fps();
                    let decoder = s
                        .bridge
                        .as_ref()
                        .map(|_| "D3D11VA/NVDEC/VT")
                        .unwrap_or("unknown");
                    let sync = s.calibration.as_ref().map(|c| c.sync_offset).unwrap_or(0);
                    t.source_info(w, h, fps, decoder, sync);
                }
            }
        }
        Ok(false) => {
            if let Some(app) = app_weak.upgrade() {
                app.set_status_text("Select a calibration and its video inputs".into());
            }
        }
        Err(e) => {
            log::error!("Init error: {e}");
            if let Some(app) = app_weak.upgrade() {
                // Source validation emits "invalid input path (...): <reason>" for
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
/// `SourceError::InvalidPath` display format is
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
    explicit_preview: bool,
) {
    if let Some(app) = app_weak.upgrade() {
        app.set_calibrating(false);
    }

    match result {
        Ok(output) => {
            let confirmed = output.assessment == reco_calibrate::CalibrationAssessment::Confirmed;
            if let reco_calibrate::CalibrationAssessment::NeedsReview { reason } =
                &output.assessment
                && !explicit_preview
            {
                state
                    .toasts
                    .push(Severity::Warn, "Calibration needs review", reason.clone());
                if let Some(app) = app_weak.upgrade() {
                    app.set_calibration_needs_review(true);
                    app.set_status_text("Candidate ready. Preview it before saving.".into());
                    crate::toast::sync_to_ui(&state.toasts, &app);
                }
                state.review_calibration =
                    Some((output, calibration_job::CalibrationSnapshot::new(state)));
                return;
            }
            let total_matches = output.total_matches;
            if let Some(ref t) = state.telemetry {
                t.calibration_complete(confirmed, total_matches);
            }
            let left_profile = output.left_lens_profile.clone();
            let right_profile = output.right_lens_profile.clone();
            let prev_total = state.playback.total_frames();
            match state.init_with_calibration(output.calibration) {
                Ok(()) => {
                    let fps = state.playback.fps();
                    let total = state.playback.total_frames().unwrap_or(0);

                    // Keep the intended destination on failure so Save cannot overwrite an older document.
                    state.calibration_path = state.left_path.as_ref().map(|left| {
                        left.with_file_name(format!(
                            "{}_calibration.json",
                            left.file_stem()
                                .map(|s| s.to_string_lossy().into_owned())
                                .unwrap_or_else(|| "reco".into())
                        ))
                    });
                    let saved = if !confirmed {
                        false
                    } else {
                        match state.save_calibration() {
                            Ok(()) => {
                                if let Some(path) = &state.calibration_path {
                                    state.user_settings.push_calibration(path.clone());
                                }
                                true
                            }
                            Err(error) => {
                                log::warn!("Failed to auto-save calibration: {error}");
                                state
                                    .toasts
                                    .push(Severity::Error, "Calibration not saved", error);
                                false
                            }
                        }
                    };
                    update_preview_ui(state, app_weak, Ok(true), prev_total);

                    if let Some(app) = app_weak.upgrade() {
                        let cal_label = state
                            .calibration_path
                            .as_ref()
                            .map(|p| display_name(p))
                            .unwrap_or_else(|| "(auto-calibrated)".into());
                        app.set_calibration_path(cal_label.into());
                        sync_recent_paths(&state.user_settings, &app);
                        app.set_status_text(
                            format!("Auto-calibrated - {:.0} fps - {total} frames", fps,).into(),
                        );
                        set_lens_profile_props(&app, left_profile, right_profile);
                        app.set_cal_dirty(!saved);
                        app.set_lens_dirty(false);
                        crate::toast::sync_to_ui(&state.toasts, &app);

                        if !confirmed {
                            state.toasts.push(
                                Severity::Warn,
                                "Unsaved calibration preview",
                                "Inspect the seam and use Save to accept this candidate.",
                            );
                            crate::toast::sync_to_ui(&state.toasts, &app);
                        }
                    }
                }
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
        Err(reco_calibrate::video::CalibrateVideosError::TimingRequired) => {
            if let Some(app) = app_weak.upgrade() {
                app.set_status_text("Match an event in both recordings. Enter the right frame minus the left frame, then calibrate again.".into());
            }
        }
        Err(e) => {
            log::error!("Auto-calibration failed: {e}");
            if let Some(ref t) = state.telemetry {
                t.calibration_error(&e.to_string());
            }
            if let Some(app) = app_weak.upgrade() {
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
