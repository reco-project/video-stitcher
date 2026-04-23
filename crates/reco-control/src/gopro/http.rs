//! HTTP client for the OpenGoPro REST API.
//!
//! Wraps `reqwest` behind a synchronous interface. The camera serves
//! HTTP at `10.5.5.9:8080` (WiFi AP mode) or `172.2X.1YZ.51:8080`
//! (USB, where XYZ are the last 3 digits of the serial number).
//!
//! All endpoints use `GET` with query parameters. Responses are JSON.

use std::time::Duration;

use super::error::GoProError;

/// HTTP timeout for camera commands. GoPro cameras can take 2-3
/// seconds to process settings changes when encoding is active.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Thin wrapper around a `reqwest` blocking client pinned to a
/// camera's base URL.
pub(crate) struct GoProHttpClient {
    client: reqwest::blocking::Client,
    base_url: String,
}

impl GoProHttpClient {
    /// Connect to a camera at the given base URL (e.g.
    /// `http://172.20.151.51:8080` for a USB-connected camera with
    /// serial ending in `051`).
    pub fn new(base_url: impl Into<String>) -> Result<Self, GoProError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .map_err(|e| GoProError::Http(e.to_string()))?;
        let base_url = base_url.into();
        log::info!("GoPro HTTP client: connecting to {base_url}");
        Ok(Self { client, base_url })
    }

    /// `GET /gopro/camera/state` - full camera state (settings + statuses).
    pub fn state(&self) -> Result<serde_json::Value, GoProError> {
        self.get("/gopro/camera/state")
    }

    /// `GET /gopro/camera/info` - hardware and firmware info.
    pub fn info(&self) -> Result<serde_json::Value, GoProError> {
        self.get("/gopro/camera/info")
    }

    /// `GET /gopro/camera/shutter/start` - start recording / take photo.
    pub fn shutter_start(&self) -> Result<(), GoProError> {
        self.get_ok("/gopro/camera/shutter/start")
    }

    /// `GET /gopro/camera/shutter/stop` - stop recording.
    pub fn shutter_stop(&self) -> Result<(), GoProError> {
        self.get_ok("/gopro/camera/shutter/stop")
    }

    /// `GET /gopro/camera/keep_alive` - prevent auto-sleep.
    pub fn keep_alive(&self) -> Result<(), GoProError> {
        self.get_ok("/gopro/camera/keep_alive")
    }

    /// `GET /gopro/camera/setting?setting={id}&option={value}` - set a camera setting.
    pub fn set_setting(&self, setting_id: u16, option: u32) -> Result<(), GoProError> {
        self.get_ok(&format!(
            "/gopro/camera/setting?setting={setting_id}&option={option}"
        ))
    }

    /// `GET /gopro/camera/presets/set_group?id={group_id}` - switch preset group.
    pub fn set_preset_group(&self, group_id: u16) -> Result<(), GoProError> {
        self.get_ok(&format!("/gopro/camera/presets/set_group?id={group_id}"))
    }

    /// `GET /gopro/camera/digital_zoom?percent={n}` - digital zoom (0-100).
    pub fn digital_zoom(&self, percent: u8) -> Result<(), GoProError> {
        let p = percent.min(100);
        self.get_ok(&format!("/gopro/camera/digital_zoom?percent={p}"))
    }

    /// `GET /gopro/webcam/start` - start webcam streaming.
    pub fn webcam_start(
        &self,
        resolution: super::constants::WebcamResolution,
        fov: super::constants::WebcamFov,
        port: u16,
        protocol: super::constants::WebcamProtocol,
    ) -> Result<(), GoProError> {
        self.get_ok(&format!(
            "/gopro/webcam/start?res={}&fov={}&port={port}&protocol={}",
            resolution as u8,
            fov as u8,
            protocol.as_str(),
        ))
    }

    /// `GET /gopro/webcam/stop` - stop webcam streaming.
    pub fn webcam_stop(&self) -> Result<(), GoProError> {
        self.get_ok("/gopro/webcam/stop")
    }

    /// `GET /gopro/webcam/status` - webcam state.
    pub fn webcam_status(&self) -> Result<serde_json::Value, GoProError> {
        self.get("/gopro/webcam/status")
    }

    /// `GET /gopro/webcam/exit` - exit webcam mode entirely.
    pub fn webcam_exit(&self) -> Result<(), GoProError> {
        self.get_ok("/gopro/webcam/exit")
    }

    /// `GET /gopro/media/list` - list files on SD card.
    pub fn media_list(&self) -> Result<serde_json::Value, GoProError> {
        self.get("/gopro/media/list")
    }

    /// `GET /gopro/media/last_captured` - most recent file.
    pub fn media_last_captured(&self) -> Result<serde_json::Value, GoProError> {
        self.get("/gopro/media/last_captured")
    }

    /// `GET /gopro/media/telemetry?path={path}` - GPMF telemetry data.
    pub fn media_telemetry(&self, path: &str) -> Result<Vec<u8>, GoProError> {
        let url = format!("{}/gopro/media/telemetry?path={path}", self.base_url);
        log::debug!("GoPro GET (bytes): {url}");
        let resp = self
            .client
            .get(&url)
            .send()
            .map_err(|e| GoProError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(GoProError::CameraError(format!(
                "HTTP {} from {url}",
                resp.status()
            )));
        }
        resp.bytes()
            .map(|b| b.to_vec())
            .map_err(|e| GoProError::Http(e.to_string()))
    }

    /// `GET /gopro/camera/control/wired_usb?p=1` - enable USB control.
    pub fn enable_wired_usb(&self) -> Result<(), GoProError> {
        self.get_ok("/gopro/camera/control/wired_usb?p=1")
    }

    /// `GET /gopro/media/turbo_transfer?p=1` - enable turbo transfer.
    pub fn enable_turbo_transfer(&self) -> Result<(), GoProError> {
        self.get_ok("/gopro/media/turbo_transfer?p=1")
    }

    /// `GET /gopro/media/turbo_transfer?p=0` - disable turbo transfer.
    pub fn disable_turbo_transfer(&self) -> Result<(), GoProError> {
        self.get_ok("/gopro/media/turbo_transfer?p=0")
    }

    /// Base URL of this client.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    // ---------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------

    fn get(&self, path: &str) -> Result<serde_json::Value, GoProError> {
        let url = format!("{}{path}", self.base_url);
        log::debug!("GoPro GET: {url}");
        let resp = self
            .client
            .get(&url)
            .send()
            .map_err(|e| GoProError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(GoProError::CameraError(format!(
                "HTTP {} from {url}",
                resp.status()
            )));
        }
        resp.json::<serde_json::Value>()
            .map_err(|e| GoProError::Http(e.to_string()))
    }

    fn get_ok(&self, path: &str) -> Result<(), GoProError> {
        let url = format!("{}{path}", self.base_url);
        log::debug!("GoPro GET: {url}");
        let resp = self
            .client
            .get(&url)
            .send()
            .map_err(|e| GoProError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(GoProError::CameraError(format!(
                "HTTP {} from {url}",
                resp.status()
            )));
        }
        Ok(())
    }
}
