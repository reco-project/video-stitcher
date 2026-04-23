//! GoPro camera integration via the OpenGoPro protocol.
//!
//! This module is a *device helper* (Layer 4 of the reco-control
//! architecture), not a [`ControlTransport`](crate::ControlTransport).
//! A GoPro is a command target (start/stop recording, sync settings,
//! query status), not a source of operator intents.
//!
//! # Communication channels
//!
//! OpenGoPro supports BLE, WiFi HTTP, and USB HTTP. This module
//! starts with USB HTTP (Phase 1): reliable, wired, each camera at a
//! unique IP (`172.2X.1YZ.51:8080` where XYZ are the serial suffix).
//! BLE discovery/wake and streaming land in later phases.
//!
//! # Multi-camera for stereo sports
//!
//! Two GoPros on separate USB connections each get unique IPs. Send
//! identical settings, then rapid sequential shutter start. Frame-
//! accurate sync is NOT possible via the protocol; post-capture
//! alignment via GPMF gyro timestamps (400 Hz) is the reliable
//! approach, which reco-calibrate already handles.
//!
//! # Reference
//!
//! - OpenGoPro repo: `../external/OpenGoPro`
//! - Official docs: <https://gopro.github.io/OpenGoPro/>
//! - Vault design note: `architecture/gopro-transport-2026-04-23.md`

pub mod constants;
pub mod error;
mod http;
pub mod status;

pub use constants::*;
pub use error::GoProError;
pub use status::{GoProInfo, GoProStatus};

use http::GoProHttpClient;

/// High-level handle to a connected GoPro camera.
///
/// Wraps the HTTP client and provides typed methods for the
/// operations reco consumers need: recording control, settings
/// sync, status queries, and webcam streaming.
///
/// # Construction
///
/// Use [`GoProCamera::connect_usb`] for USB-connected cameras or
/// [`GoProCamera::connect_wifi`] for WiFi AP mode.
pub struct GoProCamera {
    client: GoProHttpClient,
    info: Option<GoProInfo>,
}

impl GoProCamera {
    /// Connect to a USB-attached GoPro. The camera's IP is derived
    /// from the last 3 digits of its serial number:
    /// `http://172.2{d1}.1{d2}{d3}.51:8080`.
    ///
    /// Pass the full serial or just the last 3 digits.
    pub fn connect_usb(serial_suffix: &str) -> Result<Self, GoProError> {
        let suffix = if serial_suffix.len() >= 3 {
            &serial_suffix[serial_suffix.len() - 3..]
        } else {
            serial_suffix
        };
        if suffix.len() != 3 || !suffix.chars().all(|c| c.is_ascii_digit()) {
            return Err(GoProError::NotFound(format!(
                "serial suffix must be 3 digits, got '{suffix}'"
            )));
        }
        let d1 = &suffix[0..1];
        let d2 = &suffix[1..2];
        let d3 = &suffix[2..3];
        let base_url = format!("http://172.2{d1}.1{d2}{d3}.51:8080");
        log::info!("GoPro USB: serial suffix={suffix}, base_url={base_url}");
        let client = GoProHttpClient::new(&base_url)?;
        let mut cam = Self { client, info: None };
        cam.probe()?;
        Ok(cam)
    }

    /// Connect to a GoPro in WiFi AP mode at the standard address.
    pub fn connect_wifi() -> Result<Self, GoProError> {
        let client = GoProHttpClient::new("http://10.5.5.9:8080")?;
        let mut cam = Self { client, info: None };
        cam.probe()?;
        Ok(cam)
    }

    /// Connect to a GoPro at an arbitrary HTTP base URL.
    pub fn connect_url(base_url: &str) -> Result<Self, GoProError> {
        let client = GoProHttpClient::new(base_url)?;
        let mut cam = Self { client, info: None };
        cam.probe()?;
        Ok(cam)
    }

    /// Probe the camera to verify connectivity and cache info.
    fn probe(&mut self) -> Result<(), GoProError> {
        let json = self.client.info()?;
        let info = GoProInfo::from_info_json(&json);
        log::info!(
            "GoPro connected: model={}, fw={}, serial={}",
            info.model_name.as_deref().unwrap_or("unknown"),
            info.firmware_version.as_deref().unwrap_or("unknown"),
            info.serial_number.as_deref().unwrap_or("unknown"),
        );
        self.info = Some(info);
        Ok(())
    }

    // ---------------------------------------------------------------
    // Camera info
    // ---------------------------------------------------------------

    /// Cached camera info from the initial probe.
    pub fn info(&self) -> Option<&GoProInfo> {
        self.info.as_ref()
    }

    /// Re-fetch camera info from the device.
    pub fn refresh_info(&mut self) -> Result<&GoProInfo, GoProError> {
        self.probe()?;
        Ok(self.info.as_ref().unwrap())
    }

    // ---------------------------------------------------------------
    // Recording control
    // ---------------------------------------------------------------

    /// Start recording (or take a photo, depending on current mode).
    pub fn start_recording(&self) -> Result<(), GoProError> {
        log::info!("GoPro: starting recording");
        self.client.shutter_start()
    }

    /// Stop recording.
    pub fn stop_recording(&self) -> Result<(), GoProError> {
        log::info!("GoPro: stopping recording");
        self.client.shutter_stop()
    }

    /// Send keep-alive to prevent auto-sleep.
    pub fn keep_alive(&self) -> Result<(), GoProError> {
        self.client.keep_alive()
    }

    // ---------------------------------------------------------------
    // Status
    // ---------------------------------------------------------------

    /// Query current camera status (battery, encoding, SD card, etc.).
    pub fn status(&self) -> Result<GoProStatus, GoProError> {
        let json = self.client.state()?;
        Ok(GoProStatus::from_state_json(&json))
    }

    /// Check if the camera is currently encoding (recording).
    pub fn is_encoding(&self) -> Result<bool, GoProError> {
        Ok(self.status()?.encoding.unwrap_or(false))
    }

    // ---------------------------------------------------------------
    // Settings
    // ---------------------------------------------------------------

    /// Set a camera setting by ID and option value.
    pub fn set_setting(&self, setting: SettingId, option: u32) -> Result<(), GoProError> {
        log::info!("GoPro: setting {setting:?} = {option}");
        self.client.set_setting(setting as u16, option)
    }

    /// Switch to a preset group (Video, Photo, Timelapse).
    pub fn set_preset_group(&self, group: PresetGroup) -> Result<(), GoProError> {
        log::info!("GoPro: switching to preset group {group:?}");
        self.client.set_preset_group(group as u16)
    }

    /// Apply the recommended settings for stereo sports stitching.
    ///
    /// Disables HyperSmooth and horizon leveling (both break stereo
    /// geometry), sets video mode, and configures the resolution +
    /// FPS + lens mode passed as arguments.
    pub fn apply_sports_preset(
        &self,
        resolution: VideoResolution,
        fps: Fps,
        lens: VideoLens,
    ) -> Result<(), GoProError> {
        log::info!("GoPro: applying sports preset: {resolution:?} @ {fps:?}, lens={lens:?}");
        self.set_preset_group(PresetGroup::Video)?;
        self.set_setting(SettingId::VideoResolution, resolution as u32)?;
        self.set_setting(SettingId::FramesPerSecond, fps as u32)?;
        self.set_setting(SettingId::VideoLens, lens as u32)?;
        self.set_setting(SettingId::HyperSmooth, HyperSmooth::Off as u32)?;
        self.set_setting(SettingId::HorizonLeveling, 0)?;
        self.set_setting(SettingId::AutoPowerDown, 0)?;
        Ok(())
    }

    // ---------------------------------------------------------------
    // Webcam / streaming
    // ---------------------------------------------------------------

    /// Start webcam mode (USB only). Returns the UDP stream address.
    pub fn webcam_start(
        &self,
        resolution: WebcamResolution,
        fov: WebcamFov,
        port: u16,
        protocol: WebcamProtocol,
    ) -> Result<String, GoProError> {
        log::info!("GoPro: starting webcam: {resolution:?}, {fov:?}, port={port}, {protocol:?}");
        self.client.webcam_start(resolution, fov, port, protocol)?;
        let addr = match protocol {
            WebcamProtocol::Ts => format!("udp://0.0.0.0:{port}"),
            WebcamProtocol::Rtsp => format!("rtsp://{}:{port}/live", self.client.base_url()),
        };
        log::info!("GoPro webcam stream at: {addr}");
        Ok(addr)
    }

    /// Stop webcam streaming.
    pub fn webcam_stop(&self) -> Result<(), GoProError> {
        log::info!("GoPro: stopping webcam");
        self.client.webcam_stop()
    }

    /// Exit webcam mode entirely.
    pub fn webcam_exit(&self) -> Result<(), GoProError> {
        log::info!("GoPro: exiting webcam mode");
        self.client.webcam_exit()
    }

    /// Query the current webcam streaming state.
    pub fn webcam_status(&self) -> Result<serde_json::Value, GoProError> {
        self.client.webcam_status()
    }

    /// Set digital zoom level (0-100 percent).
    pub fn digital_zoom(&self, percent: u8) -> Result<(), GoProError> {
        log::info!("GoPro: digital zoom = {percent}%");
        self.client.digital_zoom(percent)
    }

    // ---------------------------------------------------------------
    // Media
    // ---------------------------------------------------------------

    /// List files on the SD card.
    pub fn media_list(&self) -> Result<serde_json::Value, GoProError> {
        self.client.media_list()
    }

    /// Get the path of the most recently captured file.
    pub fn last_captured(&self) -> Result<serde_json::Value, GoProError> {
        self.client.media_last_captured()
    }

    /// Download GPMF telemetry for a media file. The returned bytes
    /// contain gyroscope (400 Hz), accelerometer (200 Hz), and GPS
    /// (18 Hz) data in GPMF KLV format.
    pub fn media_telemetry(&self, path: &str) -> Result<Vec<u8>, GoProError> {
        self.client.media_telemetry(path)
    }

    /// Enable turbo transfer mode for faster media downloads.
    pub fn enable_turbo_transfer(&self) -> Result<(), GoProError> {
        log::info!("GoPro: enabling turbo transfer");
        self.client.enable_turbo_transfer()
    }

    /// Disable turbo transfer mode.
    pub fn disable_turbo_transfer(&self) -> Result<(), GoProError> {
        self.client.disable_turbo_transfer()
    }

    /// Enable USB wired control.
    pub fn enable_wired_usb(&self) -> Result<(), GoProError> {
        log::info!("GoPro: enabling wired USB control");
        self.client.enable_wired_usb()
    }

    // ---------------------------------------------------------------
    // Connection info
    // ---------------------------------------------------------------

    /// The HTTP base URL this camera is connected at.
    pub fn base_url(&self) -> &str {
        self.client.base_url()
    }
}

impl std::fmt::Debug for GoProCamera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoProCamera")
            .field("base_url", &self.client.base_url())
            .field("info", &self.info)
            .finish()
    }
}
