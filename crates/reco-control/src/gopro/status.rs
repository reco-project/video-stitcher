//! Camera status types.
//!
//! Parsed from the JSON response of `GET /gopro/camera/state`.

/// Snapshot of GoPro camera state. Fields are `Option` because older
/// firmware may not report every status.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct GoProStatus {
    pub battery_present: Option<bool>,
    pub battery_bars: Option<u8>,
    pub battery_percent: Option<u8>,
    pub encoding: Option<bool>,
    pub encoding_duration_secs: Option<u32>,
    pub busy: Option<bool>,
    pub overheating: Option<bool>,
    pub cold: Option<bool>,
    pub sd_remaining_kb: Option<u64>,
    pub gps_lock: Option<bool>,
    pub orientation: Option<u8>,
}

/// Camera hardware and firmware info.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct GoProInfo {
    pub model_number: Option<String>,
    pub model_name: Option<String>,
    pub firmware_version: Option<String>,
    pub serial_number: Option<String>,
    pub ap_ssid: Option<String>,
    pub ap_mac_address: Option<String>,
}

impl GoProStatus {
    /// Parse status from the raw JSON `state` response.
    ///
    /// The camera returns `{ "status": { "1": val, "2": val, ... }, "settings": { ... } }`.
    /// We extract the `status` map and look up known status IDs.
    pub(crate) fn from_state_json(json: &serde_json::Value) -> Self {
        let status = json.get("status").and_then(|v| v.as_object());
        let get_u64 = |id: u8| -> Option<u64> {
            status?.get(&id.to_string())?.as_u64()
        };
        let get_bool = |id: u8| -> Option<bool> {
            Some(get_u64(id)? != 0)
        };

        use super::constants::StatusId;
        Self {
            battery_present: get_bool(StatusId::BatteryPresent as u8),
            battery_bars: get_u64(StatusId::BatteryBars as u8).map(|v| v as u8),
            battery_percent: get_u64(StatusId::BatteryPercentage as u8).map(|v| v as u8),
            encoding: get_bool(StatusId::Encoding as u8),
            encoding_duration_secs: get_u64(StatusId::EncodingDuration as u8).map(|v| v as u32),
            busy: get_bool(StatusId::Busy as u8),
            overheating: get_bool(StatusId::Overheating as u8),
            cold: get_bool(StatusId::ColdTemperature as u8),
            sd_remaining_kb: get_u64(StatusId::SdCardRemaining as u8),
            gps_lock: get_bool(StatusId::GpsLock as u8),
            orientation: get_u64(StatusId::Orientation as u8).map(|v| v as u8),
        }
    }
}

impl GoProInfo {
    /// Parse info from the JSON response of `GET /gopro/camera/info`.
    pub(crate) fn from_info_json(json: &serde_json::Value) -> Self {
        let get_str = |key: &str| -> Option<String> {
            json.get(key)?.as_str().map(|s| s.to_string())
        };
        Self {
            model_number: json
                .get("info")
                .and_then(|i| i.get("model_number"))
                .and_then(|v| v.as_u64())
                .map(|v| v.to_string())
                .or_else(|| get_str("model_number")),
            model_name: json
                .get("info")
                .and_then(|i| i.get("model_name"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| get_str("model_name")),
            firmware_version: json
                .get("info")
                .and_then(|i| i.get("firmware_version"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| get_str("firmware_version")),
            serial_number: json
                .get("info")
                .and_then(|i| i.get("serial_number"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| get_str("serial_number")),
            ap_ssid: json
                .get("info")
                .and_then(|i| i.get("ap_ssid"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| get_str("ap_ssid")),
            ap_mac_address: json
                .get("info")
                .and_then(|i| i.get("ap_mac_addr"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| get_str("ap_mac_addr")),
        }
    }
}
