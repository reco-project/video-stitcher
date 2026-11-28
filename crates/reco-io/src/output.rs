//! Output configuration for the encoding pipeline.
//!
//! These types describe codec, quality, format, and audio choices for
//! encoded video output. Encoder backends in this crate map these to
//! their native parameters (NVENC CQ values, x264 CRF, etc.).

use std::fmt;
use std::str::FromStr;

#[cfg(feature = "ffmpeg")]
pub(crate) fn is_network_destination(path: &std::path::Path) -> bool {
    path.to_str()
        .is_some_and(|value| value.contains("://") && !value.starts_with("file://"))
}

/// Video codec for the output stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Codec {
    /// H.264 / AVC. Widest compatibility.
    #[default]
    H264,
    /// H.265 / HEVC. Better compression, less compatible.
    HEVC,
    /// AV1. Best compression, newest.
    AV1,
}

impl FromStr for Codec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "h264" | "avc" | "h.264" | "x264" => Ok(Self::H264),
            "hevc" | "h265" | "h.265" | "x265" => Ok(Self::HEVC),
            "av1" | "svt-av1" | "libaom-av1" => Ok(Self::AV1),
            _ => Err(format!("unknown codec: {s:?}")),
        }
    }
}

impl fmt::Display for Codec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::H264 => f.write_str("h264"),
            Self::HEVC => f.write_str("hevc"),
            Self::AV1 => f.write_str("av1"),
        }
    }
}

/// Sample precision for encoded video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BitDepth {
    /// 8-bit 4:2:0. Accepts NV12, YUV420P, or RGBA8 input.
    #[default]
    Eight,
    /// 10-bit 4:2:0. Requires packed P010LE input and encodes HEVC Main 10.
    Ten,
}

/// Mutually exclusive encoder quality selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// Normalized quality, from 0 to 100 (higher is better).
    Value(u8),
    /// Best-effort encoder tuning selected from a shared tier. Backends without portable quality controls use their driver defaults.
    Preset(QualityPreset),
}

impl Default for Quality {
    fn default() -> Self {
        Self::Preset(QualityPreset::default())
    }
}

impl FromStr for Quality {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse().map(Self::Preset)
    }
}

/// Encoder-agnostic quality tier.
///
/// Each encoder backend maps these to its own CRF/CQ/preset values,
/// abstracting away encoder-specific knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualityPreset {
    /// Prioritize encode speed over quality/compression.
    Fast,
    /// Balance between speed and quality.
    #[default]
    Balanced,
    /// Prioritize quality and compression over speed.
    High,
}

impl FromStr for QualityPreset {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "fast" | "low" => Ok(Self::Fast),
            "balanced" | "medium" => Ok(Self::Balanced),
            "high" | "slow" => Ok(Self::High),
            _ => Err(format!("unknown quality: {s:?}")),
        }
    }
}

impl fmt::Display for QualityPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fast => f.write_str("fast"),
            Self::Balanced => f.write_str("balanced"),
            Self::High => f.write_str("high"),
        }
    }
}

/// Output container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Container {
    /// MPEG-4 Part 14. Widest compatibility. `moov` atom finalized
    /// at close - partial files are unreadable.
    #[default]
    Mp4,
    /// Fragmented MP4 (empty_moov + frag_keyframe).
    /// Readable mid-write, self-contained fragments on keyframes.
    Mp4Fragmented,
    /// Matroska (`.mkv`). Naturally streamable, crash-safe.
    Mkv,
    /// QuickTime. Preferred on macOS.
    Mov,
    /// FLV container for RTMP streaming (YouTube, Twitch).
    Flv,
}

impl Container {
    /// Detect the appropriate container from an output path or URL.
    /// RTMP URLs get FLV, SRT URLs get Matroska; file paths infer from extension.
    pub fn for_output(path: &str) -> Self {
        if path.starts_with("rtmp://") || path.starts_with("rtmps://") {
            Self::Flv
        } else if path.starts_with("srt://") {
            Self::Mkv
        } else {
            match std::path::Path::new(path)
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
                .as_deref()
            {
                Some("mkv") => Self::Mkv,
                Some("mov") => Self::Mov,
                Some("flv") => Self::Flv,
                _ => Self::Mp4,
            }
        }
    }

    /// Whether this format targets a network URL (not a local file).
    pub fn is_streaming(&self) -> bool {
        matches!(self, Self::Flv)
    }
}

impl FromStr for Container {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mp4" => Ok(Self::Mp4),
            "fmp4" | "mp4-fragmented" | "mp4_fragmented" => Ok(Self::Mp4Fragmented),
            "mkv" | "matroska" => Ok(Self::Mkv),
            "mov" | "quicktime" => Ok(Self::Mov),
            "flv" => Ok(Self::Flv),
            _ => Err(format!("unknown format: {s:?}")),
        }
    }
}

/// Audio handling for the output.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AudioMode {
    /// Copy audio from an input stream without re-encoding.
    /// The index refers to the input position (0 = first/left, 1 = second/right).
    CopyFrom(usize),
    /// No audio track in the output.
    Disabled,
}

impl Default for AudioMode {
    fn default() -> Self {
        Self::CopyFrom(0)
    }
}
