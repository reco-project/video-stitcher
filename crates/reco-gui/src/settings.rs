//! GUI-specific persisted settings.
//!
//! Wraps `reco_io::settings` with a concrete `GuiSettings` struct that
//! captures the subset of user-facing state worth remembering across
//! sessions: recent file pairs, default export configuration, AI model
//! path, last window size.
//!
//! The split is deliberate: reco-io owns the generic load/save/MRU
//! machinery and doesn't know what a "codec" or "blend width" is,
//! while this module owns the GUI's specific schema. If reco-cli ever
//! wants its own persisted defaults it would define a separate
//! `CliSettings` struct in its own crate and use the `"cli"` namespace.

use std::path::PathBuf;

use reco_io::settings::RecentFiles;
use serde::{Deserialize, Serialize};

/// Reco-gui's on-disk settings. Stored at `<config>/reco/gui.json`.
///
/// All fields carry `#[serde(default)]` so adding new fields in future
/// releases does not invalidate existing settings files - missing
/// fields just fall back to the `Default` impl's value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuiSettings {
    /// Most recently opened left camera videos.
    #[serde(default)]
    pub recent_left: RecentFiles,
    /// Most recently opened right camera videos.
    #[serde(default)]
    pub recent_right: RecentFiles,
    /// Most recently loaded calibration JSON files.
    #[serde(default)]
    pub recent_calibration: RecentFiles,

    /// Default export codec (`"h264"`, `"hevc"`, `"av1"`).
    #[serde(default = "default_codec")]
    pub default_codec: String,
    /// Default export quality (`"fast"`, `"balanced"`, `"high"`).
    #[serde(default = "default_quality")]
    pub default_quality: String,
    /// Default seam blend width used when the export dialog opens.
    #[serde(default = "default_blend_width")]
    pub default_blend_width: f32,

    /// Last chosen AI model path (YOLOv26n, RF-DETR, etc.) so the
    /// export dialog doesn't force the user to pick it every time.
    #[serde(default)]
    pub ai_model_path: Option<PathBuf>,

    /// Last window size, remembered across restarts. `None` means
    /// "use Slint's preferred-width / preferred-height defaults".
    #[serde(default)]
    pub window_size: Option<(u32, u32)>,

    /// Recording codec preference (h264, hevc, av1).
    #[serde(default = "default_codec")]
    pub recording_codec: String,

    /// Recording quality preference (fast, balanced, high).
    #[serde(default = "default_quality")]
    pub recording_quality: String,

    /// Default folder for preview recordings. `None` means "same
    /// directory as the source video".
    #[serde(default)]
    pub recording_folder: Option<PathBuf>,

    /// Preview aspect ratio mode (fill, 16:9, 4:3, 21:9).
    #[serde(default = "default_preview_aspect")]
    pub preview_aspect: String,

    /// Opt-in anonymous telemetry. Default false - no data sent until
    /// the user explicitly enables it in preferences.
    #[serde(default)]
    pub telemetry_enabled: bool,

    /// Persistent anonymous client ID for telemetry. Generated once on
    /// first enable, never reset. UUID v4, no PII.
    #[serde(default)]
    pub telemetry_client_id: Option<String>,

    /// Dark mode preference. Default true.
    #[serde(default = "default_dark_mode")]
    pub dark_mode: bool,
}

fn default_dark_mode() -> bool {
    true
}

fn default_codec() -> String {
    "h264".into()
}
fn default_quality() -> String {
    "balanced".into()
}
fn default_blend_width() -> f32 {
    0.05
}
fn default_preview_aspect() -> String {
    "auto".into()
}

impl Default for GuiSettings {
    fn default() -> Self {
        Self {
            recent_left: RecentFiles::default(),
            recent_right: RecentFiles::default(),
            recent_calibration: RecentFiles::default(),
            default_codec: default_codec(),
            default_quality: default_quality(),
            default_blend_width: default_blend_width(),
            ai_model_path: None,
            window_size: None,
            recording_codec: default_codec(),
            recording_quality: default_quality(),
            recording_folder: None,
            preview_aspect: default_preview_aspect(),
            telemetry_enabled: false,
            telemetry_client_id: None,
            dark_mode: true,
        }
    }
}

/// Namespace used under the reco config directory. All reco-gui
/// settings live at `<config>/reco/gui.json`.
pub const NAMESPACE: &str = "gui";

impl GuiSettings {
    /// Load settings from disk. Missing or malformed files fall back
    /// to defaults per `reco_io::settings::load_or_default` (the
    /// fallback is logged but never fatal - we never refuse to start
    /// because a settings file went bad).
    pub fn load() -> Self {
        reco_io::settings::load_or_default::<GuiSettings>(NAMESPACE)
    }

    /// Persist settings atomically. Errors are logged and swallowed -
    /// a failure to save preferences should never block user work
    /// (worst case: the user has to re-pick defaults next session).
    pub fn save(&self) {
        if let Err(e) = reco_io::settings::save(NAMESPACE, self) {
            log::warn!("failed to save GUI settings: {e}");
        }
    }

    /// Convenience: push a newly-picked left video into MRU and save.
    pub fn push_left(&mut self, path: PathBuf) {
        self.recent_left.push(path);
        self.save();
    }

    /// Convenience: push a newly-picked right video into MRU and save.
    pub fn push_right(&mut self, path: PathBuf) {
        self.recent_right.push(path);
        self.save();
    }

    /// Convenience: push a newly-loaded calibration file into MRU and save.
    pub fn push_calibration(&mut self, path: PathBuf) {
        self.recent_calibration.push(path);
        self.save();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_are_sensible() {
        let s = GuiSettings::default();
        assert_eq!(s.default_codec, "h264");
        assert_eq!(s.default_quality, "balanced");
        assert!((s.default_blend_width - 0.05).abs() < 1e-6);
        assert!(s.recent_left.is_empty());
    }

    #[test]
    fn missing_fields_roundtrip_via_defaults() {
        // Simulate loading an older-version settings JSON where new
        // fields don't exist yet; the serde defaults should fill in.
        let json = r#"{ "default_codec": "hevc" }"#;
        let s: GuiSettings = serde_json::from_str(json).unwrap();
        assert_eq!(s.default_codec, "hevc");
        assert_eq!(s.default_quality, "balanced");
        assert!(s.recent_left.is_empty());
    }
}
