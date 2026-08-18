//! Background Headless Chromium runtime with cached transparent RGBA output.

use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError};
use std::time::{Duration, Instant};

use headless_chrome::browser::{LaunchOptionsBuilder, default_executable};
use headless_chrome::protocol::cdp::{Emulation, Page};
use headless_chrome::{Browser, Tab};
use reco_core::render::overlay::{OverlayFrame, OverlayFrameSource};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::local_server::LocalPackageServer;
use crate::manifest::ScoreboardPackage;

enum RuntimeCommand {
    Update(String),
    Reset,
    Shutdown,
}

/// Handle to an independently rendered scoreboard page.
pub struct ScoreboardRuntime {
    command_tx: SyncSender<RuntimeCommand>,
    frame_rx: Receiver<Result<OverlayFrame, String>>,
}

impl ScoreboardRuntime {
    /// Start a sandboxed browser worker for one validated package.
    ///
    /// Browser startup and rendering happen off the caller thread. The local
    /// Chrome/Chromium executable is resolved synchronously so an unavailable
    /// engine can be shown to the user immediately.
    pub fn start(package: ScoreboardPackage, max_fps: u32) -> Result<Self, RuntimeError> {
        let browser_path = default_executable().map_err(RuntimeError::BrowserNotFound)?;
        let max_fps = max_fps.clamp(1, 30);
        let (command_tx, command_rx) = std::sync::mpsc::sync_channel(32);
        let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel(2);
        std::thread::Builder::new()
            .name(format!("scoreboard-{}", package.manifest.id))
            .spawn(move || {
                if let Err(error) =
                    run_worker(package, browser_path, max_fps, &command_rx, &frame_tx)
                {
                    let _ = frame_tx.try_send(Err(error.to_string()));
                }
            })
            .map_err(RuntimeError::Spawn)?;
        Ok(Self {
            command_tx,
            frame_rx,
        })
    }

    /// Queue a generic JSON state update without waiting for JavaScript.
    pub fn update(&self, state: &Value) -> Result<(), RuntimeError> {
        let json = serde_json::to_string(state).map_err(RuntimeError::SerializeState)?;
        try_send_command(&self.command_tx, RuntimeCommand::Update(json))
    }

    /// Queue the package's optional `reset()` hook.
    pub fn reset(&self) -> Result<(), RuntimeError> {
        try_send_command(&self.command_tx, RuntimeCommand::Reset)
    }
}

impl OverlayFrameSource for ScoreboardRuntime {
    fn try_frame(&mut self) -> Result<Option<OverlayFrame>, String> {
        let mut newest = None;
        loop {
            match self.frame_rx.try_recv() {
                Ok(Ok(frame)) => newest = Some(frame),
                Ok(Err(error)) => return Err(error),
                Err(TryRecvError::Empty) => return Ok(newest),
                Err(TryRecvError::Disconnected) => {
                    return if newest.is_some() {
                        Ok(newest)
                    } else {
                        Err("HTML renderer stopped unexpectedly".into())
                    };
                }
            }
        }
    }
}

impl Drop for ScoreboardRuntime {
    fn drop(&mut self) {
        let _ = self.command_tx.try_send(RuntimeCommand::Shutdown);
    }
}

fn try_send_command(
    sender: &SyncSender<RuntimeCommand>,
    command: RuntimeCommand,
) -> Result<(), RuntimeError> {
    match sender.try_send(command) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(_)) => Err(RuntimeError::CommandQueueFull),
        Err(TrySendError::Disconnected(_)) => Err(RuntimeError::RendererStopped),
    }
}

fn run_worker(
    package: ScoreboardPackage,
    browser_path: std::path::PathBuf,
    max_fps: u32,
    command_rx: &Receiver<RuntimeCommand>,
    frame_tx: &SyncSender<Result<OverlayFrame, String>>,
) -> Result<(), RuntimeError> {
    let viewport = (
        package.manifest.viewport.width,
        package.manifest.viewport.height,
    );
    let options = LaunchOptionsBuilder::default()
        .path(Some(browser_path))
        .headless(true)
        .sandbox(true)
        .enable_gpu(true)
        .ignore_certificate_errors(false)
        .window_size(Some(viewport))
        .build()
        .map_err(|error| RuntimeError::BrowserLaunch(error.to_string()))?;
    let browser = Browser::new(options).map_err(RuntimeError::Browser)?;
    let tab = browser.new_tab().map_err(RuntimeError::Browser)?;
    tab.set_default_timeout(Duration::from_secs(10));
    set_viewport(&tab, viewport)?;
    install_error_hook(&tab)?;
    tab.set_transparent_background_color()
        .map_err(RuntimeError::Browser)?;
    let asset_server = LocalPackageServer::start(package.directory.clone())
        .map_err(|error| RuntimeError::AssetServer(error.to_string()))?;
    let url = asset_server.url_for(&package.manifest.entry);
    tab.navigate_to(&url)
        .and_then(|tab| tab.wait_until_navigated())
        .map_err(RuntimeError::Browser)?;

    install_host_bridge(&tab, &package)?;
    let initial_status = render_status(&tab)?;
    if let Some(error) = initial_status.error {
        return Err(RuntimeError::JavaScript(error));
    }
    let _ = capture(&tab, viewport, frame_tx)?;
    let mut last_version = initial_status.version;
    let interval = Duration::from_secs_f64(1.0 / f64::from(max_fps));
    let mut next_capture_check = Instant::now() + interval;

    loop {
        loop {
            match command_rx.try_recv() {
                Ok(RuntimeCommand::Update(json)) => evaluate_update(&tab, &json)?,
                Ok(RuntimeCommand::Reset) => evaluate_optional(&tab, "reset")?,
                Ok(RuntimeCommand::Shutdown) | Err(TryRecvError::Disconnected) => {
                    let _ = evaluate_optional(&tab, "destroy");
                    return Ok(());
                }
                Err(TryRecvError::Empty) => break,
            }
        }

        let now = Instant::now();
        if now >= next_capture_check {
            let status = render_status(&tab)?;
            if let Some(error) = status.error {
                return Err(RuntimeError::JavaScript(error));
            }
            if (status.version != last_version || status.animated)
                && capture(&tab, viewport, frame_tx)?
            {
                last_version = status.version;
            }
            next_capture_check = now + interval;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn set_viewport(tab: &Tab, viewport: (u32, u32)) -> Result<(), RuntimeError> {
    tab.call_method(Emulation::SetDeviceMetricsOverride {
        width: viewport.0,
        height: viewport.1,
        device_scale_factor: 1.0,
        mobile: false,
        scale: None,
        screen_width: Some(viewport.0),
        screen_height: Some(viewport.1),
        position_x: None,
        position_y: None,
        dont_set_visible_size: None,
        screen_orientation: None,
        viewport: None,
        display_feature: None,
        device_posture: None,
    })
    .map_err(RuntimeError::Browser)?;
    Ok(())
}

fn install_error_hook(tab: &Tab) -> Result<(), RuntimeError> {
    tab.call_method(Page::AddScriptToEvaluateOnNewDocument {
        source: r#"
            window.__recoLastError = null;
            window.addEventListener("error", event => {
                window.__recoLastError = event.message || "JavaScript error";
            });
            window.addEventListener("unhandledrejection", event => {
                window.__recoLastError = String(event.reason || "Unhandled promise rejection");
            });
        "#
        .into(),
        world_name: None,
        include_command_line_api: None,
        run_immediately: None,
    })
    .map_err(RuntimeError::Browser)?;
    Ok(())
}

fn install_host_bridge(tab: &Tab, package: &ScoreboardPackage) -> Result<(), RuntimeError> {
    let context = serde_json::json!({
        "apiVersion": 1,
        "package": {
            "id": package.manifest.id,
            "sport": package.manifest.sport,
            "version": package.manifest.version,
        },
        "viewport": {
            "width": package.manifest.viewport.width,
            "height": package.manifest.viewport.height,
        }
    });
    let script = format!(
        r#"(() => {{
            let version = 1;
            window.__recoMarkDirty = () => {{ version += 1; }};
            Object.defineProperty(window, "__recoRenderVersion", {{ get: () => version }});
            window.__recoLastError ??= null;
            new MutationObserver(window.__recoMarkDirty).observe(document.documentElement, {{
                attributes: true, childList: true, characterData: true, subtree: true
            }});
            if (!window.RecoScoreboard || window.RecoScoreboard.apiVersion !== 1 ||
                typeof window.RecoScoreboard.update !== "function") {{
                throw new Error("window.RecoScoreboard apiVersion 1 with update(state) is required");
            }}
            if (typeof window.RecoScoreboard.init === "function") {{
                window.RecoScoreboard.init({context});
            }}
            return true;
        }})()"#,
        context = context
    );
    tab.evaluate(&script, false)
        .map_err(|error| RuntimeError::JavaScript(error.to_string()))?;
    Ok(())
}

fn evaluate_update(tab: &Tab, json: &str) -> Result<(), RuntimeError> {
    let script = format!(
        r#"(() => {{
            if (!window.RecoScoreboard || typeof window.RecoScoreboard.update !== "function") {{
                throw new Error("RecoScoreboard.update is unavailable");
            }}
            window.RecoScoreboard.update({json});
            window.__recoMarkDirty();
        }})()"#
    );
    tab.evaluate(&script, false)
        .map_err(|error| RuntimeError::JavaScript(error.to_string()))?;
    Ok(())
}

fn evaluate_optional(tab: &Tab, method: &str) -> Result<(), RuntimeError> {
    let script = format!(
        r#"(() => {{
            if (window.RecoScoreboard && typeof window.RecoScoreboard.{method} === "function") {{
                window.RecoScoreboard.{method}();
                window.__recoMarkDirty?.();
            }}
        }})()"#
    );
    tab.evaluate(&script, false)
        .map_err(|error| RuntimeError::JavaScript(error.to_string()))?;
    Ok(())
}

#[derive(Deserialize)]
struct RenderStatus {
    version: u64,
    animated: bool,
    error: Option<String>,
}

fn render_status(tab: &Tab) -> Result<RenderStatus, RuntimeError> {
    let remote = tab
        .evaluate(
            r#"JSON.stringify({
                version: window.__recoRenderVersion || 0,
                animated: document.getAnimations().some(animation => animation.playState === "running"),
                error: window.__recoLastError || null
            })"#,
            false,
        )
        .map_err(RuntimeError::Browser)?;
    let json = remote
        .value
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| RuntimeError::JavaScript("renderer status returned no value".into()))?;
    serde_json::from_str(&json).map_err(RuntimeError::Status)
}

fn capture(
    tab: &Tab,
    expected_size: (u32, u32),
    frame_tx: &SyncSender<Result<OverlayFrame, String>>,
) -> Result<bool, RuntimeError> {
    let png = tab
        .capture_screenshot(Page::CaptureScreenshotFormatOption::Png, None, None, true)
        .map_err(RuntimeError::Browser)?;
    let rgba = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
        .map_err(RuntimeError::DecodePng)?
        .into_rgba8();
    if rgba.dimensions() != expected_size {
        return Err(RuntimeError::UnexpectedSize {
            expected: expected_size,
            actual: rgba.dimensions(),
        });
    }
    let frame = OverlayFrame {
        width: expected_size.0,
        height: expected_size.1,
        rgba: rgba.into_raw(),
    };
    match frame_tx.try_send(Ok(frame)) {
        Ok(()) => Ok(true),
        Err(TrySendError::Full(_)) => Ok(false),
        Err(TrySendError::Disconnected(_)) => Err(RuntimeError::RendererStopped),
    }
}

/// Browser/runtime startup and communication failures.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// No supported browser executable is installed.
    #[error("Chrome or Chromium was not found: {0}")]
    BrowserNotFound(String),
    /// Worker thread could not be created.
    #[error("cannot start HTML renderer thread: {0}")]
    Spawn(std::io::Error),
    /// Browser options were invalid.
    #[error("cannot configure Chrome: {0}")]
    BrowserLaunch(String),
    /// Chrome DevTools operation failed.
    #[error("HTML renderer failed: {0}")]
    Browser(anyhow::Error),
    /// Loopback-only package asset server could not start.
    #[error("cannot serve scoreboard package: {0}")]
    AssetServer(String),
    /// JavaScript API or page code failed.
    #[error("scoreboard JavaScript failed: {0}")]
    JavaScript(String),
    /// State could not be encoded as JSON.
    #[error("cannot serialize scoreboard state: {0}")]
    SerializeState(serde_json::Error),
    /// Browser status response was malformed.
    #[error("invalid HTML renderer status: {0}")]
    Status(serde_json::Error),
    /// Transparent screenshot could not be decoded.
    #[error("cannot decode transparent HTML frame: {0}")]
    DecodePng(image::ImageError),
    /// Browser ignored the declared viewport.
    #[error("HTML renderer returned {actual:?}, expected {expected:?}")]
    UnexpectedSize {
        expected: (u32, u32),
        actual: (u32, u32),
    },
    /// Caller is updating faster than the browser can consume commands.
    #[error("HTML renderer command queue is full")]
    CommandQueueFull,
    /// Browser worker exited.
    #[error("HTML renderer stopped")]
    RendererStopped,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basketball_api_updates_dom_and_keeps_transparent_pixels() {
        let Ok(browser_path) = default_executable() else {
            eprintln!("Chrome/Chromium unavailable; skipping browser integration test");
            return;
        };
        let directory =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scoreboards/basketball");
        let package = ScoreboardPackage::load(&directory).unwrap();
        let viewport = (
            package.manifest.viewport.width,
            package.manifest.viewport.height,
        );
        let options = LaunchOptionsBuilder::default()
            .path(Some(browser_path))
            .headless(true)
            .sandbox(true)
            .enable_gpu(true)
            .ignore_certificate_errors(false)
            .window_size(Some(viewport))
            .build()
            .unwrap();
        let browser = Browser::new(options).unwrap();
        let tab = browser.new_tab().unwrap();
        set_viewport(&tab, viewport).unwrap();
        install_error_hook(&tab).unwrap();
        tab.set_transparent_background_color().unwrap();
        let asset_server = LocalPackageServer::start(package.directory.clone()).unwrap();
        let url = asset_server.url_for(&package.manifest.entry);
        tab.navigate_to(&url)
            .and_then(|tab| tab.wait_until_navigated())
            .unwrap();
        install_host_bridge(&tab, &package).unwrap();
        evaluate_update(
            &tab,
            r#"{
                "version":1,
                "game":{"clock":"01:23","period":4,"running":true,"status":"live"},
                "home":{"shortName":"HOME","score":42},
                "away":{"shortName":"AWAY","score":39},
                "sport":{"teamFoulsHome":2,"teamFoulsAway":3}
            }"#,
        )
        .unwrap();
        let remote = tab
            .evaluate(
                r##"JSON.stringify([
                    document.querySelector("#home-score").textContent,
                    document.querySelector("#away-score").textContent,
                    document.querySelector("#clock").textContent,
                    document.querySelector("#period").textContent
                ])"##,
                false,
            )
            .unwrap();
        let json = remote.value.unwrap();
        let values: Vec<String> = serde_json::from_str(json.as_str().unwrap()).unwrap();
        assert_eq!(values, ["42", "39", "01:23", "Q4"]);

        let png = tab
            .capture_screenshot(Page::CaptureScreenshotFormatOption::Png, None, None, true)
            .unwrap();
        let rgba = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
            .unwrap()
            .into_rgba8();
        assert_eq!(rgba.dimensions(), viewport);
        assert_eq!(rgba.get_pixel(0, 0).0[3], 0);
    }
}
