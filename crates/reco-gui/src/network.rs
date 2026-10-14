//! RTSP selection and cancellable opening outside the interactive event loop.

#[cfg(test)]
#[path = "../../reco-io/src/rtsp/tests/peer.rs"]
mod peer;
#[cfg(test)]
mod tests;

use super::{
    AppState, Calibration, NetworkInputRow, Playback, RecoApp, sync_segments, update_preview_ui,
};
use reco_core::source::{FrameSource, SourceFrame};
use reco_io::rtsp::{RtspConfig, RtspInput, RtspSource};
use slint::ComponentHandle;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;

/// Session-only input selection; URLs are neither logged nor persisted by this controller.
#[derive(Clone)]
pub struct Config {
    /// Camera-ordered addresses and explicit clock mappings.
    pub inputs: Vec<RtspInput>,
    /// Capture limits shared by preview and processed output.
    pub options: RtspConfig,
}

struct Ready {
    source: RtspSource,
    first: SourceFrame,
}

struct Opening {
    config: Config,
    calibration: Calibration,
    interrupted: Arc<AtomicBool>,
    receiver: Receiver<Result<Ready, String>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Opening {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            self.interrupted.store(true, Ordering::Relaxed);
            let _ = worker.join();
        }
    }
}

/// Retains the accepted selection while a replacement is opened independently.
#[derive(Default)]
pub struct State {
    /// Selected live inputs, including while preview is suspended for output.
    pub active: Option<Config>,
    opening: Option<Opening>,
}

impl State {
    /// Whether a connection attempt still owns a worker.
    pub fn is_opening(&self) -> bool {
        self.opening.is_some()
    }

    /// Request cancellation without waiting on network I/O.
    pub fn cancel(&self) {
        if let Some(opening) = &self.opening {
            opening.interrupted.store(true, Ordering::Relaxed);
        }
    }

    fn begin(&mut self, config: Config, calibration: Calibration) {
        let interrupted = Arc::new(AtomicBool::new(false));
        let token = interrupted.clone();
        let request = config.clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = (|| {
                let mut source = RtspSource::open(request.inputs, request.options, token)
                    .map_err(|error| error.to_string())?;
                let first = source
                    .next_frame()
                    .map_err(|error| error.to_string())?
                    .ok_or("RTSP opening cancelled")?;
                Ok(Ready { source, first })
            })();
            let _ = sender.send(result);
        });
        self.opening = Some(Opening {
            config,
            calibration,
            interrupted,
            receiver,
            worker: Some(worker),
        });
    }
}

fn parse(app: &RecoApp, cameras: usize) -> Result<Config, String> {
    use slint::Model;
    let rows = app.get_network_inputs();
    if cameras == 0 || rows.row_count() != cameras {
        return Err("Load the calibration for these cameras first.".into());
    }
    let inputs = rows.iter().enumerate().map(|(camera, row)| {
        let url = row.url.trim();
        if !url.starts_with("rtsp://") || url.len() <= 7
            || url.bytes().any(|c| c.is_ascii_whitespace() || c.is_ascii_control())
        {
            return Err(format!("Camera {} needs an rtsp:// URL without whitespace.", camera + 1));
        }
        let timestamp_offset_us = row.offset.trim().parse::<i64>()
            .map_err(|_| format!("Enter camera {}'s measured timestamp offset in microseconds (0 if already aligned).", camera + 1))?;
        Ok(RtspInput { url: url.into(), timestamp_offset_us })
    }).collect::<Result<Vec<_>, String>>()?;
    let fps_text = app.get_network_fps();
    let (num, den) = fps_text
        .trim()
        .split_once('/')
        .unwrap_or((fps_text.trim(), "1"));
    let fps = (
        num.parse::<u32>()
            .map_err(|_| "Enter FPS as 30 or 30000/1001.")?,
        den.parse::<u32>()
            .map_err(|_| "Enter FPS as 30 or 30000/1001.")?,
    );
    if fps.0 == 0
        || fps.1 == 0
        || fps.0 > i32::MAX as u32
        || fps.1 > i32::MAX as u32
        || u64::from(fps.0) > 1_000_000 * u64::from(fps.1)
    {
        return Err("Frame rate must be positive and representable.".into());
    }
    Ok(Config {
        inputs,
        options: RtspConfig {
            fps,
            synchronization: reco_io::frame_sync::Config {
                wait_us: 150_000,
                skew_us: 30_000,
                max_age_us: 350_000,
                capacity: 4,
            },
            open_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(3),
            software_decode: false,
        },
    })
}

/// Connect the source dialog to application state without moving Slint objects to workers.
pub fn bind(app: &RecoApp, state: &Rc<RefCell<AppState>>) {
    let weak = app.as_weak();
    let owner = state.clone();
    app.on_open_network(move || {
        use slint::Model;
        let Some(app) = weak.upgrade() else { return };
        let state = owner.borrow();
        if state.is_exporting() || state.is_recording() || state.calibration_job.is_some() {
            return;
        }
        let count = state
            .calibration
            .as_ref()
            .map_or(0, |cal| cal.projection.camera_count());
        if app.get_network_inputs().row_count() != count {
            app.set_network_inputs(
                Rc::new(slint::VecModel::from(vec![
                    NetworkInputRow::default();
                    count
                ]))
                .into(),
            );
        }
        app.set_network_message(if count == 0 {
            "Load a calibration in the workspace first.".into()
        } else {
            "".into()
        });
        app.set_network_dialog_open(true);
    });
    let weak = app.as_weak();
    let owner = state.clone();
    app.on_connect_network(move || {
        let Some(app) = weak.upgrade() else { return };
        let mut state = owner.borrow_mut();
        if state.network.is_opening()
            || state.is_exporting()
            || state.is_recording()
            || state.calibration_job.is_some()
        {
            return;
        }
        let Some(cal) = state.calibration.clone() else {
            app.set_network_message("Load a calibration first.".into());
            return;
        };
        if state.shared_gpu.is_none() {
            app.set_network_message("Preview renderer is not ready.".into());
            return;
        }
        match parse(&app, cal.projection.camera_count()) {
            Ok(config) => {
                state.network.begin(config, cal);
                app.set_network_opening(true);
                app.set_network_message("Connecting and matching camera timestamps…".into());
            }
            Err(error) => app.set_network_message(error.into()),
        }
    });
    let weak = app.as_weak();
    let owner = state.clone();
    app.on_cancel_network(move || {
        owner.borrow().network.cancel();
        if let Some(app) = weak.upgrade() {
            app.set_network_message("Stopping…".into());
        }
    });
    let weak = app.as_weak();
    let owner = state.clone();
    app.on_disconnect_network(move || {
        let mut state = owner.borrow_mut();
        if state.is_exporting() {
            return;
        }
        state.unload_pipeline();
        if let Some(app) = weak.upgrade() {
            app.set_network_active(false);
            app.set_files_loaded(false);
            app.set_playing(false);
            app.set_preview_frame(slint::Image::default());
            app.set_status_text("Live inputs disconnected. Calibration preserved.".into());
        }
    });
}

/// Poll opening without waiting on either camera reads or a discarded source's cleanup.
pub fn poll(state: &mut AppState, weak: &slint::Weak<RecoApp>) {
    let Some(opening) = &state.network.opening else {
        return;
    };
    let result = match opening.receiver.try_recv() {
        Err(TryRecvError::Empty) => return,
        Err(TryRecvError::Disconnected) => {
            Err("RTSP opening worker stopped without a result.".into())
        }
        Ok(result) => result,
    };
    let mut opening = state.network.opening.take().unwrap();
    if let Some(worker) = opening.worker.take() {
        let _ = worker.join();
    }
    let current = state.calibration.as_ref() == Some(&opening.calibration);
    let cancelled = opening.interrupted.load(Ordering::Relaxed);
    let result = result.and_then(|mut ready| {
        ready.source.defer_cleanup(&state.decoder_cleanup);
        if cancelled || !current {
            return Err("Connection cancelled or calibration changed.".into());
        }
        let playback = Playback::from_live(ready.source, ready.first, &state.decoder_cleanup);
        let bridge = state.build_bridge(
            &opening.calibration,
            playback.source_sizes().unwrap().to_vec(),
            playback.is_full_range(),
        )?;
        state.unload_pipeline();
        state.left_input = None;
        state.right_input = None;
        state.left_path = None;
        state.right_path = None;
        state.playback = playback;
        state.bridge = Some(bridge);
        state.network.active = Some(opening.config.clone());
        state.reset_view();
        Ok(())
    });
    if let Some(app) = weak.upgrade() {
        app.set_network_opening(false);
        match result {
            Ok(()) => {
                app.set_network_dialog_open(false);
                app.set_network_active(true);
                app.set_left_path("".into());
                app.set_right_path("".into());
                sync_segments(state, &app);
                update_preview_ui(state, weak, Ok(true), None);
                app.set_playing(true);
                app.set_status_text("Live RTSP preview - host-backed decoding".into());
            }
            Err(error) => {
                app.set_network_message(error.clone().into());
                app.set_status_text(error.into());
            }
        }
    }
}
