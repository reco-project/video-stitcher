//! Output configuration for the encoding pipeline.
//!
//! These types describe codec, quality, format, and audio choices for
//! encoded video output. Encoder backends in this crate map these to
//! their native parameters (NVENC CQ values, x264 CRF, etc.).

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

/// Bitrate control strategy for the encoder.
///
/// Different encoders map these to their native rate control modes:
/// - NVENC: CRF maps to constqp, VBR to vbr, CBR to cbr
/// - libx264/libx265: CRF maps to `-crf`, VBR to `-b:v`
/// - SVT-AV1: CRF maps to `-crf`
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Bitrate {
    /// Constant rate factor (quality-based, variable bitrate).
    /// Lower values = higher quality. Typical range: 18-28 for H.264.
    Crf(u8),
    /// Variable bitrate with target and optional maximum (kbps).
    Vbr {
        /// Target bitrate in kbps.
        target_kbps: u32,
        /// Maximum bitrate in kbps (optional cap).
        max_kbps: Option<u32>,
    },
    /// Constant bitrate (kbps). Predictable file size.
    Cbr(u32),
    /// Encoder-agnostic quality preset. Each encoder backend maps this
    /// to appropriate CRF/CQ values internally.
    Quality(Quality),
}

impl Default for Bitrate {
    fn default() -> Self {
        Self::Quality(Quality::default())
    }
}

/// Encoder-agnostic quality tier.
///
/// Each encoder backend maps these to its own CRF/CQ/preset values,
/// abstracting away encoder-specific knobs like x264's `ultrafast..veryslow`
/// or NVENC's `p1..p7`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quality {
    /// Prioritize encode speed over quality/compression.
    Fast,
    /// Balance between speed and quality.
    #[default]
    Balanced,
    /// Prioritize quality and compression over speed.
    High,
}

/// Output container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Format {
    /// MPEG-4 Part 14. Widest compatibility. `moov` atom finalized
    /// at close — partial files are unreadable and external tools
    /// can't stream the output while it's still being written.
    #[default]
    Mp4,
    /// Fragmented MP4 (`.mp4` with empty_moov + frag_keyframe).
    /// Readable mid-write, self-contained fragments on keyframes.
    /// Use this or [`Self::Mkv`] if you need to tee the output
    /// via `ffmpeg -c copy -f flv rtmp://...` while the stitch is
    /// still running.
    Mp4Fragmented,
    /// Matroska (`.mkv`). Naturally streamable, crash-safe.
    /// OBS's default for live recording.
    Mkv,
    /// QuickTime. Preferred on macOS.
    Mov,
    /// FLV container for RTMP streaming (YouTube, Twitch).
    Flv,
}

impl Format {
    /// Detect the appropriate container from an output path or URL.
    /// RTMP/RTSP URLs get FLV; file paths infer from extension.
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

/// Audio handling for the output.
///
/// Extensible for future audio processing (noise cancellation, stereo
/// mixing, wind filtering) via additional variants.
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

/// Complete output configuration for encoding.
///
/// Passed to [`StitchJob`](crate::StitchJob) or the encoder factory.
/// The encoder backend maps these to encoder-specific parameters.
#[derive(Debug, Clone, Default)]
pub struct OutputConfig {
    /// Video codec.
    pub codec: Codec,
    /// Bitrate / quality control.
    pub bitrate: Bitrate,
    /// Container format.
    pub format: Format,
    /// Audio handling.
    pub audio: AudioMode,
    /// Output resolution. `None` means match input dimensions.
    pub resolution: Option<(u32, u32)>,
    /// Force a specific encoder by name (e.g. `"h264_nvenc"`, `"libx264"`).
    /// When `None`, the backend auto-selects the best available encoder.
    pub encoder_name: Option<String>,
    /// Override the CRF/quality value (passed through to the encoder).
    pub crf: Option<u8>,
    /// Override the encoder preset string (passed through to the encoder).
    pub preset: Option<String>,
}
