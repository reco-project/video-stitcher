//! OpenGoPro protocol constants.
//!
//! Setting and status IDs from the OpenGoPro HTTP/BLE spec.
//! Only the IDs relevant to sports stereo stitching are included;
//! the full set is in the OpenGoPro repo (`../external/OpenGoPro`).

/// Camera setting IDs (HTTP `setting` parameter, BLE setting characteristic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum SettingId {
    VideoResolution = 2,
    FramesPerSecond = 3,
    AutoPowerDown = 59,
    Gps = 83,
    VideoAspectRatio = 108,
    VideoLens = 121,
    AntiFlicker = 134,
    HyperSmooth = 135,
    HorizonLeveling = 150,
    WirelessBand = 178,
    VideoBitRate = 182,
    BitDepth = 183,
    ColorProfile = 184,
}

/// Camera status IDs (HTTP `state` response keys, BLE query characteristic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StatusId {
    BatteryPresent = 1,
    BatteryBars = 2,
    Overheating = 6,
    Busy = 8,
    Encoding = 10,
    EncodingDuration = 13,
    SdCardRemaining = 54,
    BatteryPercentage = 70,
    GpsLock = 68,
    ColdTemperature = 85,
    Orientation = 86,
    CameraControlOwner = 114,
    SdCardCapacity = 117,
}

/// Video resolution options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VideoResolution {
    Res4K = 1,
    Res2_7K = 4,
    Res1080p = 9,
    Res5_3K = 100,
}

/// Frames per second options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Fps {
    Fps24 = 10,
    Fps30 = 8,
    Fps60 = 5,
    Fps120 = 0,
    Fps240 = 2,
}

/// Video lens / FOV mode options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VideoLens {
    Wide = 0,
    Linear = 4,
    SuperView = 3,
    Narrow = 2,
}

/// HyperSmooth stabilization. Must be OFF for stereo stitching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HyperSmooth {
    Off = 0,
    On = 1,
    High = 2,
    Boost = 3,
    AutoBoost = 4,
}

/// Preset group IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum PresetGroup {
    Video = 1000,
    Photo = 1001,
    Timelapse = 1002,
}

/// Webcam resolution options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WebcamResolution {
    Res480p = 4,
    Res720p = 7,
    Res1080p = 12,
}

/// Webcam FOV options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WebcamFov {
    Wide = 0,
    Narrow = 2,
    SuperView = 3,
    Linear = 4,
}

/// Webcam streaming protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebcamProtocol {
    Ts,
    Rtsp,
}

impl WebcamProtocol {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Ts => "TS",
            Self::Rtsp => "RTSP",
        }
    }
}
