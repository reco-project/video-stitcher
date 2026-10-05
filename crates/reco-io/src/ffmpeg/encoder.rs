//! Encode RGBA, NV12 or planar YUV420P frames with FFmpeg.
//!
//! Wraps FFmpeg's muxer, encoder, and swscale with explicit codec and container selection. Supports hardware-accelerated encoding
//! (NVENC, QSV, VideoToolbox, VAAPI) with automatic fallback to software.

extern crate ffmpeg_next as ffmpeg;

use ffmpeg::format::Pixel;
use ffmpeg::software::scaling::{context::Context as ScalingContext, flag::Flags as ScalingFlags};
use ffmpeg::util::frame::video::Video as VideoFrame;
use ffmpeg::{Rational, Rescale, codec, encoder, format};
use std::path::{Path, PathBuf};
use thiserror::Error;

use super::hevc_color::HevcColor;
use super::hw_upload::{HardwareUpload, staging_pixel_format};
use super::output_io::OutputIo;
use crate::output::{Codec, Container, Quality, QualityPreset};

/// Errors from the video encoder.
#[derive(Debug, Error)]
pub enum EncodeError {
    /// The caller interrupted network output.
    #[error("output cancelled")]
    Cancelled,
    /// Invalid or unsupported encoder configuration.
    #[error("invalid encoder configuration: {0}")]
    InvalidConfig(String),

    /// FFmpeg error.
    #[error("FFmpeg: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// No encoder available for the requested codec.
    #[error("no encoder found for codec '{0}' — is FFmpeg built with the right encoder?")]
    CodecNotFound(String),

    /// Frame data has wrong size.
    #[error("frame data size mismatch: expected {expected} bytes, got {actual}")]
    FrameSizeMismatch { expected: usize, actual: usize },

    /// The encoded picture does not match the requested output dimensions.
    #[error(
        "encoded picture is {actual:?}, expected {expected:?}; the encoder or driver produced incompatible dimensions"
    )]
    EncodedDimensions {
        /// Requested visible width and height.
        expected: (u32, u32),
        /// Visible width and height advertised in the bitstream.
        actual: (i32, i32),
    },
}

impl Codec {
    fn id(self) -> codec::Id {
        match self {
            Self::H264 => codec::Id::H264,
            Self::HEVC => codec::Id::HEVC,
            Self::AV1 => codec::Id::AV1,
        }
    }
    fn candidates(self) -> &'static [EncoderCandidate] {
        match self {
            Self::H264 => H264_ENCODERS,
            Self::HEVC => HEVC_ENCODERS,
            Self::AV1 => AV1_ENCODERS,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::HEVC => "HEVC",
            Self::AV1 => "AV1",
        }
    }
}

struct EncoderCandidate {
    name: &'static str,
    is_hardware: bool,
    /// Pixel format the encoder is opened with. For most encoders this is the
    /// CPU swscale target (NV12 or YUV420P). `Pixel::VAAPI` is a sentinel
    /// meaning "GPU surface, sw_format = NV12": the CPU stages frames in NV12
    /// (see `staging_pixel_format`) and uploads them to a VAAPI surface before
    /// encoding (see `hw_upload`).
    pixel_format: Pixel,
}

/// Known H.264 encoder candidates, in preference order.
///
/// Covers: NVIDIA (NVENC), Intel (QSV), Apple (VideoToolbox), AMD (AMF/VAAPI),
/// embedded Linux (V4L2 M2M), Android (MediaCodec), Windows fallback (MF),
/// and software (libx264).
const H264_ENCODERS: &[EncoderCandidate] = &[
    EncoderCandidate {
        name: "h264_nvenc",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "h264_qsv",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "h264_videotoolbox",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "h264_amf",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "h264_vaapi",
        is_hardware: true,
        pixel_format: Pixel::VAAPI,
    },
    EncoderCandidate {
        name: "h264_v4l2m2m",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "h264_mediacodec",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "h264_mf",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "libx264",
        is_hardware: false,
        pixel_format: Pixel::YUV420P,
    },
];

/// Known HEVC encoder candidates, in preference order.
const HEVC_ENCODERS: &[EncoderCandidate] = &[
    EncoderCandidate {
        name: "hevc_nvenc",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "hevc_qsv",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "hevc_videotoolbox",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "hevc_amf",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "hevc_vaapi",
        is_hardware: true,
        pixel_format: Pixel::VAAPI,
    },
    EncoderCandidate {
        name: "hevc_v4l2m2m",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "hevc_mediacodec",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "hevc_mf",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "libx265",
        is_hardware: false,
        pixel_format: Pixel::YUV420P,
    },
];

/// Known AV1 encoder candidates, in preference order.
const AV1_ENCODERS: &[EncoderCandidate] = &[
    EncoderCandidate {
        name: "av1_nvenc",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "av1_qsv",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "av1_amf",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "av1_vaapi",
        is_hardware: true,
        pixel_format: Pixel::VAAPI,
    },
    EncoderCandidate {
        name: "libsvtav1",
        is_hardware: false,
        pixel_format: Pixel::YUV420P,
    },
    EncoderCandidate {
        name: "libaom-av1",
        is_hardware: false,
        pixel_format: Pixel::YUV420P,
    },
];

/// Information about an available encoder.
#[derive(Debug, Clone)]
pub struct EncoderInfo {
    /// FFmpeg codec name (e.g., `"h264_nvenc"`, `"libx264"`).
    pub name: String,
    /// Human-readable description from FFmpeg.
    pub description: String,
    /// Whether this is a hardware-accelerated encoder.
    pub is_hardware: bool,
}

/// List encoders registered in FFmpeg for a given codec.
///
/// Returns encoders in preference order (hardware first, then software). Registration does not guarantee that the required hardware or driver is usable.
pub fn available_encoders(codec: Codec) -> Vec<EncoderInfo> {
    crate::init();
    codec
        .candidates()
        .iter()
        .filter_map(|c| {
            encoder::find_by_name(c.name).map(|codec| EncoderInfo {
                name: c.name.to_string(),
                description: codec.description().to_string(),
                is_hardware: c.is_hardware,
            })
        })
        .collect()
}

/// Detect which H.264 encoders are available (convenience wrapper).
pub fn available_h264_encoders() -> Vec<EncoderInfo> {
    available_encoders(Codec::H264)
}

fn auto_candidate_allowed(name: &str) -> bool {
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    if name.ends_with("_nvenc") {
        return false;
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    if name.ends_with("_qsv") {
        return false;
    }

    #[cfg(target_os = "linux")]
    if name.ends_with("_qsv")
        && let Some(vaapi_name) = vaapi_peer_encoder(name)
        && encoder::find_by_name(vaapi_name).is_some()
    {
        return false;
    }

    #[cfg(not(target_os = "macos"))]
    if name.ends_with("_videotoolbox") {
        return false;
    }

    #[cfg(not(target_os = "windows"))]
    if name.ends_with("_amf") || name.ends_with("_mf") {
        return false;
    }

    #[cfg(not(target_os = "linux"))]
    if name.ends_with("_vaapi") || name.ends_with("_v4l2m2m") {
        return false;
    }

    #[cfg(not(target_os = "android"))]
    if name.ends_with("_mediacodec") {
        return false;
    }

    true
}

#[cfg(target_os = "linux")]
fn vaapi_peer_encoder(name: &str) -> Option<&'static str> {
    match name {
        "h264_qsv" => Some("h264_vaapi"),
        "hevc_qsv" => Some("hevc_vaapi"),
        "av1_qsv" => Some("av1_vaapi"),
        _ => None,
    }
}

impl Container {
    fn muxer_name(self) -> &'static str {
        match self {
            Self::Mp4 | Self::Mp4Fragmented => "mp4",
            Self::Mov => "mov",
            Self::Mkv => "matroska",
            Self::Flv => "flv",
        }
    }
}

/// Matrix and range of YUV values, independent of primaries and transfer function.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum YuvColor {
    /// No known source interpretation. Do not assign rendered-output metadata to raw camera planes.
    #[default]
    Unspecified,
    /// Known camera range, without assigning the renderer's matrix to raw planes.
    SourceRange(bool),
    /// BT.709 matrix with limited-range YUV, as produced by the renderer.
    /// This specifies neither colour primaries nor a transfer function.
    Bt709Limited,
}

/// Configuration for the video encoder.
#[derive(Debug, Clone, Default)]
pub struct EncoderConfig {
    /// Force a specific encoder by name, or `None` for auto-detection.
    pub encoder_name: Option<String>,
    /// Output codec. When absent, infer from the selected encoder, or use H.264 for automatic selection.
    pub codec: Option<Codec>,
    /// Restrict automatic selection to software encoders.
    pub software_only: bool,
    /// Explicit VAAPI render device, or NVENC GPU index. Requires a matching named encoder.
    pub device: Option<String>,
    /// One quality selection, mapped to the selected encoder's native controls.
    pub quality: Quality,
    /// Override the encoder preset string (passed through to the encoder).
    pub preset: Option<String>,
    /// Source files to copy audio from, in order (stream copy, no re-encoding).
    /// For a chained recording, list every segment so audio spans the whole
    /// output; passing a single file copies just that one. The first audio
    /// stream of each file is used.
    pub audio_source: Option<Vec<std::path::PathBuf>>,
    /// Start copying audio at this source timestamp, in seconds.
    ///
    /// Used when the video export starts from a later processing window
    /// (`--start-time`, GUI export start). Copied audio packets are
    /// rebased so the selected timestamp lands at output time zero.
    pub audio_start_time: f64,
    /// Output container format. Defaults to plain MP4 to match the
    /// existing stitch-output behavior; opt in to fragmented MP4 or
    /// Matroska for streamable / write-while-read workflows (e.g.,
    /// the stacked-video replay backend).
    pub container: Container,
    /// Override the encoder's group-of-pictures size (frames
    /// between keyframes). `None` leaves ffmpeg / libx264 defaults
    /// (typically 250 frames). Set a small value (e.g. 30) when
    /// the output needs frequent keyframes for seekable replay or
    /// fragmented-MP4 fragment flush cadence.
    pub gop_size: Option<u32>,
    /// Optional RTMP stream URL for simultaneous file + stream output.
    ///
    /// Shares the primary encoded packets through a bounded 60-packet FIFO.
    /// Overflow skips to a keyframe; failures disable streaming without reconnecting or dropping recording frames.
    pub stream_url: Option<String>,
    /// Existing application cancellation signal. Interrupts network output, not local recording finalization.
    pub interrupted: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Network open, synchronous I/O and final-drain deadline. `None` selects ten seconds.
    /// Cooperative FFmpeg interruption does not bound native DNS resolution. Optional FIFO writes are isolated, not timed by producer activity.
    pub network_timeout: Option<std::time::Duration>,
}

type OpenedVideoEncoder = (
    ffmpeg::encoder::video::Encoder,
    ScalingContext,
    usize,
    Rational,
    Rational,
    Option<HardwareUpload>,
);

/// Video encoder that writes RGBA frames to an MP4 file.
///
/// # Example
///
/// ```rust,no_run
/// use reco_io::ffmpeg::encoder::{VideoEncoder, EncoderConfig};
/// use std::path::Path;
/// use ffmpeg_next::Rational;
///
/// let config = EncoderConfig::default();
/// let mut enc = VideoEncoder::new(Path::new("out.mp4"), 1920, 1080, Rational(30, 1), &config).unwrap();
/// // enc.write_frame(&rgba_data, None).unwrap();
/// enc.finish().unwrap();
/// ```
pub struct VideoEncoder {
    octx: OutputIo,
    encoder: ffmpeg::encoder::video::Encoder,
    scaler: ScalingContext,
    stream_index: usize,
    encoder_time_base: Rational,
    output_time_base: Rational,
    timeline: super::encoder_timing::Timeline,
    width: u32,
    height: u32,
    finished: bool,
    encoder_name: String,
    header_check: Option<HevcHeaderCheck>,
    hevc_color: Option<HevcColor>,
    /// Reusable frame buffers to avoid per-frame allocation.
    rgba_frame: VideoFrame,
    yuv_frame: VideoFrame,
    hardware_upload: Option<HardwareUpload>,
    /// Audio passthrough state (if enabled).
    audio: Option<AudioPassthrough>,
    /// Secondary output context for RTMP streaming (same encoded packets).
    stream: Option<StreamOutput>,
}

/// State for a secondary RTMP stream output that receives cloned packets
/// from the primary encode pass.
struct StreamOutput {
    octx: OutputIo,
    video_index: usize,
    video_time_base: Rational,
    audio: Option<SilentAudio>,
}

/// Generates a silent AAC track on the streamed video timeline.
struct SilentAudio {
    encoder: ffmpeg::encoder::audio::Encoder,
    stream_index: usize,
    output_time_base: Rational,
    frame: ffmpeg::frame::Audio,
    next_pts: Option<i64>,
    samples_per_frame: usize,
    sample_rate: u32,
}

/// State for copying audio from one or more input files to the output.
///
/// For a chained recording the segments are copied back-to-back: when the
/// current segment's audio ends, the next opens and its timestamps are
/// shifted by `segment_offset` so the output audio is gapless.
struct AudioPassthrough {
    ictx: format::context::Input,
    /// Audio stream index in the current input file.
    input_stream_index: usize,
    /// Audio stream index in the output file.
    output_stream_index: usize,
    /// Current input audio stream time base (for rescaling).
    input_time_base: Rational,
    /// Output audio stream time base.
    output_time_base: Rational,
    /// Source timestamp where audio passthrough starts, in seconds. Applies to
    /// the first segment only (the start-time trim is within file 1).
    start_time_secs: f64,
    /// Whether all segments have been read.
    exhausted: bool,
    /// Remaining chained segments to copy after the current one ends.
    remaining_segments: std::collections::VecDeque<std::path::PathBuf>,
    /// Output-timebase shift added to the current segment's timestamps so
    /// audio is continuous across chained segments. Recomputed per segment
    /// from its first packet (segments may not start at timestamp 0 - AAC
    /// priming can even make the first DTS negative).
    segment_offset: i64,
    /// Highest packet end (pts + duration) written, in output ticks; the next
    /// segment is rebased so its first packet lands here.
    next_offset: i64,
    /// When set, the next packet read is the first of a freshly opened segment;
    /// its timestamp is rebased so the segment starts at this output position.
    pending_start: Option<i64>,
    /// True only while copying the first segment (start-time trim applies there).
    first_segment: bool,
}

// SAFETY: VideoEncoder is only used from a single thread. The raw pointers
// inside FFmpeg's SwsContext/Encoder are not shared across threads.
unsafe impl Send for VideoEncoder {}
// SAFETY: Same single-thread usage as VideoEncoder.
unsafe impl Send for StreamOutput {}
unsafe impl Send for SilentAudio {}

impl Drop for VideoEncoder {
    fn drop(&mut self) {
        if !self.finished {
            log::warn!(
                "VideoEncoder dropped without calling finish() — output file may be corrupt"
            );
            let _ = self.flush_and_finalize();
        }
    }
}

impl VideoEncoder {
    /// Create an encoder for the selected codec and container.
    ///
    /// Auto-detects the best available encoder (hardware first, then software)
    /// unless `config.encoder_name` is set.
    pub fn new(
        path: &Path,
        width: u32,
        height: u32,
        fps: Rational,
        config: &EncoderConfig,
    ) -> Result<Self, EncodeError> {
        Self::new_with_color(path, width, height, fps, config, YuvColor::Unspecified)
    }

    pub(crate) fn new_rendered(
        path: &Path,
        width: u32,
        height: u32,
        fps: Rational,
        config: &EncoderConfig,
    ) -> Result<Self, EncodeError> {
        Self::new_with_color(path, width, height, fps, config, YuvColor::Bt709Limited)
    }

    #[cfg(feature = "stacked-output")]
    pub(crate) fn new_source(
        path: &Path,
        width: u32,
        height: u32,
        fps: Rational,
        config: &EncoderConfig,
        full_range: bool,
    ) -> Result<Self, EncodeError> {
        Self::new_with_color(
            path,
            width,
            height,
            fps,
            config,
            YuvColor::SourceRange(full_range),
        )
    }

    fn new_with_color(
        path: &Path,
        width: u32,
        height: u32,
        fps: Rational,
        config: &EncoderConfig,
        color: YuvColor,
    ) -> Result<Self, EncodeError> {
        if width == 0
            || height == 0
            || !width.is_multiple_of(2)
            || !height.is_multiple_of(2)
            || width > i32::MAX as u32
            || height > i32::MAX as u32
        {
            return Err(EncodeError::InvalidConfig(
                "4:2:0 dimensions must be positive even values representable as signed 32-bit integers".into(),
            ));
        }
        if fps.0 <= 0 || fps.1 <= 0 {
            return Err(EncodeError::InvalidConfig(
                "frame-rate numerator and denominator must be positive".into(),
            ));
        }
        if matches!(config.quality, Quality::Value(101..)) {
            return Err(EncodeError::InvalidConfig(
                "quality must be between 0 and 100".into(),
            ));
        }
        crate::init();

        let all_candidates = config.codec.unwrap_or_default().candidates();

        let forced_encoder = config.encoder_name.is_some();
        if config.device.is_some() && !forced_encoder {
            return Err(EncodeError::InvalidConfig(
                "device selection requires a named encoder".into(),
            ));
        }

        // Build candidate list
        let candidates: Vec<(&str, bool, Pixel)> = if let Some(ref name) = config.encoder_name {
            let selected = encoder::find_by_name(name)
                .ok_or_else(|| EncodeError::CodecNotFound(name.clone()))?;
            if config
                .codec
                .is_some_and(|codec| selected.id() != codec.id())
            {
                return Err(EncodeError::InvalidConfig(format!(
                    "encoder {name} does not encode the requested codec"
                )));
            }
            // When a specific encoder is forced, look it up in all codec tables
            let candidate = all_candidates
                .iter()
                .find(|c| c.name == name.as_str())
                .or_else(|| {
                    // Also check other codec tables for cross-codec --encoder usage
                    [H264_ENCODERS, HEVC_ENCODERS, AV1_ENCODERS]
                        .iter()
                        .flat_map(|t| t.iter())
                        .find(|c| c.name == name.as_str())
                });
            let pixel_fmt = candidate.map_or(Pixel::YUV420P, |c| c.pixel_format);
            let is_hw = candidate.is_some_and(|c| c.is_hardware);
            vec![(name.as_str(), is_hw, pixel_fmt)]
        } else {
            all_candidates
                .iter()
                .filter(|c| encoder::find_by_name(c.name).is_some())
                .filter(|c| !config.software_only || !c.is_hardware)
                .filter(|c| auto_candidate_allowed(c.name))
                .filter(|c| validate_selection(c.name, config).is_ok())
                .map(|c| (c.name, c.is_hardware, c.pixel_format))
                .collect()
        };

        if candidates.is_empty() {
            return Err(EncodeError::CodecNotFound(
                config.codec.unwrap_or_default().label().to_string(),
            ));
        }
        if forced_encoder {
            validate_selection(candidates[0].0, config)?;
        }

        // Try each candidate — create a fresh output context per attempt
        // because add_stream + write_header can leave the context in a
        // bad state if the encoder fails to open.
        let mut last_err = None;

        for (name, is_hw, pixel_fmt) in &candidates {
            let codec = match encoder::find_by_name(name) {
                Some(c) => c,
                None => continue,
            };

            if color != YuvColor::Unspecified
                && codec.id() == codec::Id::HEVC
                && !HevcColor::available()
            {
                last_err = Some(ffmpeg::Error::BsfNotFound.into());
                continue;
            }

            // The selected container is authoritative, even without a matching extension.
            let mut octx = OutputIo::open(path, config.container.muxer_name(), config, false)?;

            match Self::try_open(
                &mut octx, codec, *pixel_fmt, *is_hw, width, height, fps, config, name, color,
            ) {
                Ok((enc_opened, scaler, stream_index, encoder_time_base, _, hardware_upload)) => {
                    let hevc_color =
                        if color != YuvColor::Unspecified && codec.id() == codec::Id::HEVC {
                            let filter = match HevcColor::new(
                                &enc_opened,
                                (color == YuvColor::Bt709Limited)
                                    .then_some(ffmpeg::color::Space::BT709),
                                color == YuvColor::SourceRange(true),
                            ) {
                                Ok(filter) => filter,
                                Err(error) => {
                                    last_err = Some(error.into());
                                    continue;
                                }
                            };
                            octx.stream_mut(stream_index)
                                .expect("video stream we just added")
                                .set_parameters(filter.parameters()?);
                            Some(filter)
                        } else {
                            None
                        };
                    let hw_tag = if *is_hw { " (hardware)" } else { " (software)" };
                    log::info!(
                        "Encoder: {}x{} {}{} @ {}/{} fps",
                        width,
                        height,
                        name,
                        hw_tag,
                        fps.0,
                        fps.1,
                    );
                    // Story-tell which frame-delivery path this encoder uses, so
                    // the upload cost is visible in the field.
                    if hardware_upload.is_some() {
                        log::info!(
                            "Encoder frame path: explicit VAAPI surface upload (NV12 -> GPU surface)"
                        );
                    } else if *is_hw {
                        log::info!(
                            "Encoder frame path: implicit driver upload (software NV12 -> GPU, handled by FFmpeg)"
                        );
                    } else {
                        log::info!("Encoder frame path: software encode (no GPU upload)");
                    }

                    // Set up audio passthrough before writing the header.
                    let audio = if let Some(ref audio_paths) = config.audio_source {
                        Self::setup_audio_stream(&mut octx, audio_paths, config.audio_start_time)?
                    } else {
                        None
                    };

                    // Fragmented MP4 needs `movflags` so the muxer
                    // writes an `empty_moov` up front and flushes
                    // self-contained fragments on every keyframe.
                    // A concurrent reader can then parse the file
                    // mid-write; plain MP4 would park the `moov`
                    // until `write_trailer` and break replay.
                    let mut movflags = Vec::new();
                    if config.container == Container::Mp4Fragmented {
                        movflags.extend(["empty_moov", "frag_keyframe"]);
                    }
                    // Preserve known matrix/range even when primaries and transfer are unspecified.
                    // Some hardware encoders omit these labels from their bitstream headers.
                    if color != YuvColor::Unspecified
                        && matches!(
                            config.container,
                            Container::Mp4 | Container::Mp4Fragmented | Container::Mov
                        )
                    {
                        movflags.push("write_colr");
                    }
                    if !movflags.is_empty() {
                        let mut opts = ffmpeg::Dictionary::new();
                        opts.set("movflags", &movflags.join("+"));
                        octx.write_header(opts)?;
                    } else {
                        octx.write_header(ffmpeg::Dictionary::new())?;
                    }

                    let output_time_base = octx
                        .stream(stream_index)
                        .expect("video stream we just added")
                        .time_base();

                    // Update audio output time base after write_header
                    // (the muxer may adjust it).
                    let audio = audio.map(|mut a| {
                        if let Some(s) = octx.stream(a.output_stream_index) {
                            a.output_time_base = s.time_base();
                        }
                        a
                    });

                    let stream = if let Some(ref url) = config.stream_url {
                        match Self::open_stream_output(url, &octx, stream_index, config) {
                            Ok(s) => {
                                log::info!("Optional stream output queued");
                                Some(s)
                            }
                            Err(e) => {
                                log::warn!(
                                    "Failed to open stream output ({e}), recording without stream"
                                );
                                None
                            }
                        }
                    } else {
                        None
                    };

                    let header_check = if codec.id() == codec::Id::HEVC {
                        Some(HevcHeaderCheck::new(&enc_opened)?)
                    } else {
                        None
                    };
                    return Ok(Self {
                        octx,
                        encoder: enc_opened,
                        scaler,
                        stream_index,
                        encoder_time_base,
                        output_time_base,
                        timeline: super::encoder_timing::Timeline::new(fps),
                        width,
                        height,
                        finished: false,
                        encoder_name: name.to_string(),
                        header_check,
                        hevc_color,
                        rgba_frame: VideoFrame::new(Pixel::RGBA, width, height),
                        yuv_frame: VideoFrame::new(staging_pixel_format(*pixel_fmt), width, height),
                        hardware_upload,
                        audio,
                        stream,
                    });
                }
                Err(e) => {
                    if forced_encoder {
                        log::warn!("Encoder {name} failed to open: {e}");
                    } else {
                        log::debug!("Encoder {name} failed to open: {e}, trying next...");
                    }
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or(EncodeError::CodecNotFound(
            config.codec.unwrap_or_default().label().to_string(),
        )))
    }

    /// Attempt to open a specific encoder. Returns the opened encoder + scaler
    /// + stream metadata, or an error.
    #[allow(clippy::too_many_arguments)]
    fn try_open(
        octx: &mut format::context::Output,
        codec: ffmpeg::Codec,
        pixel_fmt: Pixel,
        is_hw: bool,
        width: u32,
        height: u32,
        fps: Rational,
        config: &EncoderConfig,
        name: &str,
        color: YuvColor,
    ) -> Result<OpenedVideoEncoder, EncodeError> {
        let needs_global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

        let mut ost = octx.add_stream(codec)?;
        let stream_index = ost.index();

        let mut enc = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()?;

        enc.set_width(width);
        enc.set_height(height);
        enc.set_format(pixel_fmt);
        if color == YuvColor::Bt709Limited {
            enc.set_colorspace(ffmpeg::color::Space::BT709);
            enc.set_color_range(ffmpeg::color::Range::MPEG);
        }
        if let YuvColor::SourceRange(full_range) = color {
            enc.set_color_range(if full_range {
                ffmpeg::color::Range::JPEG
            } else {
                ffmpeg::color::Range::MPEG
            });
        }
        enc.set_frame_rate(Some(fps));
        let encoder_time_base = super::encoder_timing::TIME_BASE;
        enc.set_time_base(encoder_time_base);

        // Optional GOP override for callers that need short keyframe
        // intervals (replay-recording fMP4, live streaming). Applied
        // before `open_with` so libx264 / libx265 / etc. pick it up.
        if let Some(gop) = config.gop_size {
            enc.set_gop(gop);
        }

        if needs_global_header {
            enc.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        if !is_hw {
            enc.set_threading(ffmpeg::threading::Config::count(0));
        }

        let hardware_upload = if pixel_fmt == Pixel::VAAPI {
            let upload = HardwareUpload::new_vaapi(width, height, config.device.as_deref())?;
            // SAFETY: `enc` is a live, not-yet-opened encoder context.
            let codec_ctx = unsafe { enc.as_mut_ptr() };
            upload.attach_to_encoder(codec_ctx)?;
            Some(upload)
        } else {
            None
        };

        // Seed the output stream with the encoder's unopened
        // parameters BEFORE `open_with` so the muxer has valid
        // codec parameters when it allocates its internal fragment
        // state (matters for the fMP4 muxer with `empty_moov`,
        // which writes the moov before any packet lands). The
        // canonical ffmpeg-next transcode example follows the same
        // pattern: set params, open, re-set params.
        ost.set_parameters(&enc);
        let mut opts = build_encoder_opts(
            name,
            config.quality,
            config.preset.as_deref(),
            width,
            height,
        );
        if name.ends_with("_nvenc")
            && let Some(device) = &config.device
        {
            opts.set("gpu", device);
        }
        // B-frames cause negative initial PTS offsets and frame reordering
        // in the encoded output. For real-time stitching they add latency
        // with no visual benefit, so default to 0. RECO_X264_BFRAMES
        // overrides for offline encodes that can tolerate the reordering.
        let bf = std::env::var("RECO_X264_BFRAMES").unwrap_or_else(|_| "0".to_string());
        opts.set("bf", &bf);
        validate_encoder_options(&enc, &opts)?;
        let encoder = enc.open_with(opts)?;
        ost.set_parameters(&encoder);
        // Stamp both stream rates so our own outputs re-probe cleanly on
        // every consumer path (Matroska and fragmented MP4 don't derive
        // them from sample timing the way plain mov does).
        ost.set_rate(fps);
        ost.set_avg_frame_rate(fps);

        // Note: write_header is called by the caller (new()) after optional
        // audio stream setup. output_time_base is read after write_header.

        let output_time_base = Rational(0, 0); // placeholder, updated after write_header

        let staging_pixel_fmt = staging_pixel_format(pixel_fmt);
        let mut scaler = ScalingContext::get(
            Pixel::RGBA,
            width,
            height,
            staging_pixel_fmt,
            width,
            height,
            ScalingFlags::BILINEAR,
        )?;
        if color == YuvColor::Bt709Limited {
            // SAFETY: the scaler is live and uniquely borrowed; FFmpeg owns the static coefficient table.
            let result = unsafe {
                let coefficients = ffmpeg::sys::sws_getCoefficients(ffmpeg::sys::SWS_CS_ITU709);
                ffmpeg::sys::sws_setColorspaceDetails(
                    scaler.as_mut_ptr(),
                    coefficients,
                    1,
                    coefficients,
                    0,
                    0,
                    1 << 16,
                    1 << 16,
                )
            };
            if result < 0 {
                return Err(ffmpeg::Error::from(result).into());
            }
        }

        Ok((
            encoder,
            scaler,
            stream_index,
            encoder_time_base,
            output_time_base,
            hardware_upload,
        ))
    }

    /// The name of the active encoder (e.g., `"h264_nvenc"`, `"libx264"`).
    pub fn encoder_name(&self) -> &str {
        &self.encoder_name
    }

    /// Pick the best encoder for a secondary stream (replay) that
    /// runs concurrently with a primary encode.
    ///
    /// NVENC has a dedicated ASIC and benefits from hardware encode
    /// even under concurrent stitch load. All other hardware encoders
    /// (VideoToolbox, AMF, QSV, VAAPI) share GPU resources with the
    /// stitch shader and benchmark slower than libx264 ultrafast
    /// when both run simultaneously.
    pub fn replay_encoder_name(primary_encoder: &str) -> Option<&'static str> {
        if primary_encoder.contains("nvenc") {
            None
        } else {
            Some("libx264")
        }
    }

    /// Set up audio passthrough by adding an audio stream to the output context.
    ///
    /// Called before `write_header`. Returns `None` if no audio stream found.
    fn setup_audio_stream(
        octx: &mut format::context::Output,
        sources: &[PathBuf],
        start_time_secs: f64,
    ) -> Result<Option<AudioPassthrough>, EncodeError> {
        let Some((source_path, rest)) = sources.split_first() else {
            return Ok(None);
        };
        let mut ictx = format::input(source_path)?;
        let start_time_secs = sanitize_audio_start_time(start_time_secs);

        let audio_stream = ictx.streams().best(ffmpeg::media::Type::Audio);

        let Some(audio_stream) = audio_stream else {
            log::info!("No audio stream found in {}", source_path.display());
            return Ok(None);
        };

        let input_stream_index = audio_stream.index();
        let input_time_base = audio_stream.time_base();
        let codec_params = audio_stream.parameters();

        // Add an audio stream to the output with copied codec parameters.
        let mut ost = octx.add_stream(ffmpeg::encoder::find(codec::Id::None))?;
        ost.set_parameters(codec_params);
        // SAFETY: set codec_tag to 0 so the muxer picks the right tag
        // for the container format. codec_par is valid for the stream's lifetime.
        unsafe {
            (*ost.parameters().as_mut_ptr()).codec_tag = 0;
        }

        let output_stream_index = ost.index();
        let output_time_base = ost.time_base();

        if start_time_secs > 0.0 {
            let seek_ts = (start_time_secs * f64::from(ffmpeg::ffi::AV_TIME_BASE)).round() as i64;
            match ictx.seek(seek_ts, ..seek_ts) {
                Ok(()) => {
                    log::info!(
                        "Audio passthrough: seeking {} to {start_time_secs:.3}s",
                        source_path.display()
                    );
                }
                Err(e) => {
                    log::warn!(
                        "Audio passthrough seek to {start_time_secs:.3}s failed for {} ({e}); \
                         falling back to packet filtering from the beginning",
                        source_path.display()
                    );
                }
            }
        }

        log::info!(
            "Audio passthrough from {} (stream {}, codec: {:?})",
            source_path.display(),
            input_stream_index,
            ictx.stream(input_stream_index)
                .map(|s| s.parameters().id())
                .unwrap_or(codec::Id::None),
        );

        Ok(Some(AudioPassthrough {
            ictx,
            input_stream_index,
            output_stream_index,
            input_time_base,
            output_time_base,
            start_time_secs,
            exhausted: false,
            remaining_segments: rest.iter().cloned().collect(),
            segment_offset: 0,
            next_offset: 0,
            pending_start: None,
            first_segment: true,
        }))
    }

    /// Open the next chained segment for audio copy, continuing the timeline.
    ///
    /// Skips segments with no audio stream. Returns `false` when none remain.
    fn advance_segment(audio: &mut AudioPassthrough) -> Result<bool, EncodeError> {
        loop {
            let Some(next) = audio.remaining_segments.pop_front() else {
                return Ok(false);
            };
            let ictx = format::input(&next)?;
            let Some(stream) = ictx.streams().best(ffmpeg::media::Type::Audio) else {
                log::warn!(
                    "Chained audio: {} has no audio stream, skipping",
                    next.display()
                );
                continue;
            };
            audio.input_stream_index = stream.index();
            audio.input_time_base = stream.time_base();
            audio.ictx = ictx;
            // Rebase from this segment's first packet (computed in the loop).
            audio.pending_start = Some(audio.next_offset);
            audio.first_segment = false;
            log::info!(
                "Chained audio: continuing from {} at +{:.2}s",
                next.display(),
                audio.next_offset as f64 * f64::from(audio.output_time_base),
            );
            return Ok(true);
        }
    }

    /// Open a secondary FLV output context for RTMP streaming.
    ///
    /// Copies codec parameters from the primary video stream and adds a
    /// silent AAC audio track.
    fn open_stream_output(
        url: &str,
        primary_octx: &format::context::Output,
        primary_video_index: usize,
        config: &EncoderConfig,
    ) -> Result<StreamOutput, EncodeError> {
        if !url.starts_with("rtmp://") && !url.starts_with("rtmps://") {
            return Err(EncodeError::InvalidConfig(
                "secondary output requires RTMP or RTMPS".into(),
            ));
        }
        let mut octx = OutputIo::open(Path::new(url), "flv", config, true)?;

        let primary_video = primary_octx
            .stream(primary_video_index)
            .ok_or_else(|| EncodeError::CodecNotFound("primary video stream missing".into()))?;

        let params = primary_video.parameters();
        if params.id() != codec::Id::H264 {
            return Err(EncodeError::InvalidConfig(
                "secondary FLV streaming requires H.264".into(),
            ));
        }
        let extradata_len = unsafe { (*params.as_ptr()).extradata_size };
        log::info!("Stream output: copying H.264 params (extradata={extradata_len} bytes)");
        if extradata_len == 0 {
            log::warn!(
                "Stream output: H.264 extradata is empty - FLV header will \
                 lack SPS/PPS, YouTube will likely reject the stream"
            );
        }

        let mut ost = octx.add_stream(ffmpeg::encoder::find(codec::Id::None))?;
        ost.set_parameters(params);
        unsafe {
            (*ost.parameters().as_mut_ptr()).codec_tag = 0;
        }
        let video_index = ost.index();

        let mut silent_audio = SilentAudio::new(&mut octx)?;

        octx.write_header(ffmpeg::Dictionary::new())?;

        let video_time_base = octx
            .stream(video_index)
            .expect("stream we just added")
            .time_base();

        silent_audio.output_time_base = octx
            .stream(silent_audio.stream_index)
            .expect("stream we just added")
            .time_base();

        log::info!("Optional stream configured (video tb={video_time_base})");

        Ok(StreamOutput {
            octx,
            video_index,
            video_time_base,
            audio: Some(silent_audio),
        })
    }

    /// Forward audio packets up to the given video duration limit.
    ///
    /// Reads audio packets from the input and writes those whose timestamp
    /// falls within `max_pts` (in the output time base). Stops when audio
    /// is exhausted or exceeds the limit.
    fn forward_audio_packets_until(&mut self, max_pts: i64) -> Result<(), EncodeError> {
        let Some(ref mut audio) = self.audio else {
            return Ok(());
        };
        if audio.exhausted {
            return Ok(());
        }
        // Start-time trim is expressed in the (fixed) output time base.
        let start_pts = seconds_to_pts(audio.start_time_secs, audio.output_time_base);

        loop {
            let mut packet = ffmpeg::Packet::empty();
            match packet.read(&mut audio.ictx) {
                Ok(()) => {
                    if packet.stream() != audio.input_stream_index {
                        continue; // skip video/subtitle packets
                    }
                    packet.set_stream(audio.output_stream_index);
                    packet.rescale_ts(audio.input_time_base, audio.output_time_base);

                    // The start-time trim applies to the first segment only.
                    if audio.first_segment && start_pts > 0 {
                        let packet_start = match (packet.pts(), packet.dts()) {
                            (Some(pts), Some(dts)) => Some(pts.min(dts)),
                            (Some(pts), None) => Some(pts),
                            (None, Some(dts)) => Some(dts),
                            (None, None) => None,
                        };
                        if packet_start.is_some_and(|ts| ts < start_pts) {
                            continue;
                        }
                        if let Some(pts) = packet.pts() {
                            packet.set_pts(Some(pts - start_pts));
                        }
                        if let Some(dts) = packet.dts() {
                            packet.set_dts(Some(dts - start_pts));
                        }
                    }

                    // First packet of a continuation segment: rebase it so the
                    // segment starts exactly where the previous one ended,
                    // regardless of the segment's own (possibly negative) start.
                    if let Some(target) = audio.pending_start {
                        let anchor = packet.dts().or_else(|| packet.pts()).unwrap_or(0);
                        audio.segment_offset = target - anchor;
                        audio.pending_start = None;
                    }

                    // Shift later segments so audio is gapless across the chain.
                    if audio.segment_offset != 0 {
                        if let Some(pts) = packet.pts() {
                            packet.set_pts(Some(pts + audio.segment_offset));
                        }
                        if let Some(dts) = packet.dts() {
                            packet.set_dts(Some(dts + audio.segment_offset));
                        }
                    }

                    // Track the running end so the next segment continues from here.
                    if let Some(pts) = packet.pts() {
                        let end = pts + packet.duration().max(0);
                        if end > audio.next_offset {
                            audio.next_offset = end;
                        }
                    }

                    // Stop if this audio packet is beyond the video duration.
                    if packet.pts().is_some_and(|pts| pts > max_pts) {
                        audio.exhausted = true;
                        break;
                    }

                    self.octx.write_packet(&mut packet)?;
                }
                Err(ffmpeg::Error::Eof) => {
                    // Current segment done; continue with the next chained one.
                    if !Self::advance_segment(audio)? {
                        audio.exhausted = true;
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        Ok(())
    }

    fn send_current_yuv_frame(&mut self, pts_us: Option<i64>) -> Result<(), EncodeError> {
        if self.finished {
            return Err(EncodeError::InvalidConfig(
                "output is already finalized".into(),
            ));
        }
        let pts = self.timeline.next(pts_us)?;
        self.yuv_frame.set_pts(Some(pts));
        if let Some(ref mut upload) = self.hardware_upload {
            let frame = upload.upload(&self.yuv_frame)?;
            self.encoder.send_frame(frame)?;
        } else {
            self.encoder.send_frame(&self.yuv_frame)?;
        }
        self.timeline.accept(pts, pts_us.is_some());
        self.receive_and_write_packets()?;
        Ok(())
    }

    /// Write an RGBA frame to the output file.
    ///
    /// `pts_us` follows [`reco_core::sink::OutputFrame::pts_us`]. The output uses a microsecond encoder clock; the container may quantize it further. No frame pacing or duplication is performed.
    ///
    /// `rgba_data` must be exactly `width * height * 4` bytes (tightly packed).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "encode_frame")
    )]
    pub fn write_frame(
        &mut self,
        rgba_data: &[u8],
        pts_us: Option<i64>,
    ) -> Result<(), EncodeError> {
        let expected = self.width as usize * self.height as usize * 4;
        if rgba_data.len() != expected {
            return Err(EncodeError::FrameSizeMismatch {
                expected,
                actual: rgba_data.len(),
            });
        }

        let stride = self.rgba_frame.stride(0);
        let row_bytes = self.width as usize * 4;
        if stride == row_bytes {
            self.rgba_frame.data_mut(0)[..rgba_data.len()].copy_from_slice(rgba_data);
        } else {
            for y in 0..self.height as usize {
                let src_start = y * row_bytes;
                let dst_start = y * stride;
                self.rgba_frame.data_mut(0)[dst_start..dst_start + row_bytes]
                    .copy_from_slice(&rgba_data[src_start..src_start + row_bytes]);
            }
        }

        self.scaler.run(&self.rgba_frame, &mut self.yuv_frame)?;
        self.send_current_yuv_frame(pts_us)
    }

    /// Write a pre-converted NV12 frame to the output file.
    ///
    /// `pts_us` follows [`Self::write_frame`].
    ///
    /// `nv12_data` must be exactly `width * height * 3 / 2` bytes:
    /// Y plane (`width * height`) followed by interleaved UV plane (`width * height / 2`).
    ///
    /// This bypasses the CPU swscale RGBA→NV12 conversion, which is the main
    /// performance benefit of GPU-side NV12 conversion.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "encode_nv12_frame")
    )]
    pub fn write_nv12_frame(
        &mut self,
        nv12_data: &[u8],
        pts_us: Option<i64>,
    ) -> Result<(), EncodeError> {
        let expected = self.width as usize * self.height as usize * 3 / 2;
        if nv12_data.len() != expected {
            return Err(EncodeError::FrameSizeMismatch {
                expected,
                actual: nv12_data.len(),
            });
        }

        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;

        // Copy Y plane
        let y_stride = self.yuv_frame.stride(0);
        if y_stride == w {
            self.yuv_frame.data_mut(0)[..y_size].copy_from_slice(&nv12_data[..y_size]);
        } else {
            for row in 0..h {
                let src_start = row * w;
                let dst_start = row * y_stride;
                self.yuv_frame.data_mut(0)[dst_start..dst_start + w]
                    .copy_from_slice(&nv12_data[src_start..src_start + w]);
            }
        }

        // Copy UV/chroma data depending on encoder pixel format
        let uv_data = &nv12_data[y_size..];
        let chroma_h = h / 2;
        let chroma_w = w / 2;

        if self.yuv_frame.format() == Pixel::NV12 {
            // NV12: interleaved UV plane (same layout as input)
            let uv_stride = self.yuv_frame.stride(1);
            if uv_stride == w {
                self.yuv_frame.data_mut(1)[..uv_data.len()].copy_from_slice(uv_data);
            } else {
                for row in 0..chroma_h {
                    let src_start = row * w;
                    let dst_start = row * uv_stride;
                    self.yuv_frame.data_mut(1)[dst_start..dst_start + w]
                        .copy_from_slice(&uv_data[src_start..src_start + w]);
                }
            }
        } else {
            // YUV420P: de-interleave UV into separate U and V planes
            let u_stride = self.yuv_frame.stride(1);
            let v_stride = self.yuv_frame.stride(2);
            for row in 0..chroma_h {
                for col in 0..chroma_w {
                    let uv_idx = row * w + col * 2;
                    self.yuv_frame.data_mut(1)[row * u_stride + col] = uv_data[uv_idx];
                    self.yuv_frame.data_mut(2)[row * v_stride + col] = uv_data[uv_idx + 1];
                }
            }
        }

        self.send_current_yuv_frame(pts_us)
    }

    /// Write a pre-converted YUV420P planar frame.
    ///
    /// `pts_us` follows [`Self::write_frame`].
    ///
    /// Plane slices must be tightly packed (no padding between
    /// rows): Y is `width * height`, U and V are each
    /// `(width / 2) * (height / 2)`. Odd dimensions are rejected
    /// because 4:2:0 chroma subsampling requires even width/height.
    ///
    /// The stacked-video replay encoder (`stacked-output` feature) produces YUV420P directly. Feeding RGBA through [`Self::write_frame`] would add a YUV→RGBA→YUV roundtrip; this path avoids that conversion and its rounding loss.
    ///
    /// Adapts to the encoder's staging pixel format: writes separate
    /// U and V planes for YUV420P (software encoders), or interleaves
    /// the chroma into a single UV plane for NV12 (hardware encoders,
    /// including the VAAPI path, whose NV12 staging frame is then
    /// uploaded to a GPU surface before encoding).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "encode_yuv420p_frame")
    )]
    pub fn write_yuv420p_planes(
        &mut self,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        pts_us: Option<i64>,
    ) -> Result<(), EncodeError> {
        if !self.width.is_multiple_of(2) || !self.height.is_multiple_of(2) {
            return Err(EncodeError::FrameSizeMismatch {
                expected: 0,
                actual: 0,
            });
        }

        let w = self.width as usize;
        let h = self.height as usize;
        let chroma_w = w / 2;
        let chroma_h = h / 2;
        let y_expected = w * h;
        let uv_expected = chroma_w * chroma_h;

        if y.len() != y_expected {
            return Err(EncodeError::FrameSizeMismatch {
                expected: y_expected,
                actual: y.len(),
            });
        }
        if u.len() != uv_expected || v.len() != uv_expected {
            return Err(EncodeError::FrameSizeMismatch {
                expected: uv_expected,
                actual: u.len().max(v.len()),
            });
        }

        // Y plane: identical for both YUV420P and NV12.
        let y_stride = self.yuv_frame.stride(0);
        if y_stride == w {
            self.yuv_frame.data_mut(0)[..y_expected].copy_from_slice(y);
        } else {
            for row in 0..h {
                let src_start = row * w;
                let dst_start = row * y_stride;
                self.yuv_frame.data_mut(0)[dst_start..dst_start + w]
                    .copy_from_slice(&y[src_start..src_start + w]);
            }
        }

        if self.yuv_frame.format() == Pixel::NV12 {
            // Hardware encoders (NVENC, AMF, VT): interleave U+V
            // into a single UV plane. ~0.3ms for 1080p.
            let uv_stride = self.yuv_frame.stride(1);
            let dst = self.yuv_frame.data_mut(1);
            for row in 0..chroma_h {
                let u_start = row * chroma_w;
                let v_start = row * chroma_w;
                let dst_start = row * uv_stride;
                for col in 0..chroma_w {
                    dst[dst_start + col * 2] = u[u_start + col];
                    dst[dst_start + col * 2 + 1] = v[v_start + col];
                }
            }
        } else {
            // Software encoders (libx264): separate U and V planes.
            for (plane_idx, src) in [(1usize, u), (2, v)] {
                let stride = self.yuv_frame.stride(plane_idx);
                if stride == chroma_w {
                    self.yuv_frame.data_mut(plane_idx)[..uv_expected].copy_from_slice(src);
                } else {
                    for row in 0..chroma_h {
                        let src_start = row * chroma_w;
                        let dst_start = row * stride;
                        self.yuv_frame.data_mut(plane_idx)[dst_start..dst_start + chroma_w]
                            .copy_from_slice(&src[src_start..src_start + chroma_w]);
                    }
                }
            }
        }

        self.send_current_yuv_frame(pts_us)
    }

    /// Publish buffered muxer packets and bytes without finalizing the container.
    ///
    /// Frames still buffered by the encoder are not drained. This does not synchronize storage; [`Self::finish`] is still required to write the final trailer.
    pub fn flush_to_disk(&mut self) -> Result<(), EncodeError> {
        if self.finished {
            return Ok(());
        }
        self.octx.flush()?;
        if let Some(ref mut stream) = self.stream
            && let Err(error) = stream.octx.flush()
        {
            log::warn!("RTMP stream flush failed ({error}), disabling stream");
            self.stream = None;
        }
        Ok(())
    }

    /// Flush the encoder and finalize the output file.
    ///
    /// Must be called after all frames have been written. Safe to call
    /// multiple times — subsequent calls are no-ops.
    pub fn finish(&mut self) -> Result<(), EncodeError> {
        if self.finished {
            return Ok(());
        }
        self.flush_and_finalize()?;
        self.finished = true;
        log::info!("Encoder finished: {} frames written", self.timeline.frames);
        Ok(())
    }

    fn flush_and_finalize(&mut self) -> Result<(), EncodeError> {
        self.encoder.send_eof()?;
        self.receive_and_write_packets()?;
        if let Some(filter) = &mut self.hevc_color {
            filter.send(None)?;
            self.write_filtered_packets()?;
        }
        if self.header_check.is_some() && self.timeline.frames > 0 {
            return Err(EncodeError::InvalidConfig(
                "HEVC output has no readable picture header".into(),
            ));
        }

        // Forward audio packets trimmed to the video duration.
        if self.audio.is_some() {
            let audio_tb = self.audio.as_ref().unwrap().output_time_base;
            // SAFETY: pure rational rescaling into the copied audio stream's clock.
            let video_duration_pts = unsafe {
                ffmpeg::sys::av_rescale_q(
                    self.timeline.end_us(),
                    self.encoder_time_base.into(),
                    audio_tb.into(),
                )
            };
            self.forward_audio_packets_until(video_duration_pts)?;
        }

        // Publish the recording trailer before waiting for optional transport drain.
        let primary_result = self.octx.finish();
        // Finalize stream output (non-fatal).
        if let Some(ref mut stream) = self.stream {
            if let Some(ref mut audio) = stream.audio
                && let Err(error) = audio.finish(&mut stream.octx, self.timeline.end_us())
            {
                log::warn!("Silent audio finalization failed: {error}");
            }
            if let Err(error) = stream.octx.finish() {
                log::warn!("RTMP stream finalization failed: {error}");
            }
        }
        self.stream = None;

        primary_result
    }

    fn receive_and_write_packets(&mut self) -> Result<(), EncodeError> {
        let mut packet = ffmpeg::Packet::empty();
        while receive_encoded_packet(&mut self.encoder, &mut packet)? {
            if let Some(check) = &mut self.header_check
                && check.verify(&packet, (self.width, self.height))?
            {
                self.header_check = None;
            }
            if let Some(filter) = &mut self.hevc_color {
                filter.send(Some(&mut packet))?;
                self.write_filtered_packets()?;
            } else {
                self.write_video_packet(&mut packet)?;
            }
        }
        Ok(())
    }

    fn write_filtered_packets(&mut self) -> Result<(), EncodeError> {
        let mut packet = ffmpeg::Packet::empty();
        while self
            .hevc_color
            .as_mut()
            .expect("active HEVC filter")
            .receive(&mut packet)?
        {
            self.write_video_packet(&mut packet)?;
        }
        Ok(())
    }

    fn write_video_packet(&mut self, packet: &mut ffmpeg::Packet) -> Result<(), EncodeError> {
        packet.set_stream(self.stream_index);
        packet.rescale_ts(self.encoder_time_base, self.output_time_base);
        // Give the final sample a defined end when the encoder leaves duration unset.
        if packet.duration() <= 0 {
            // SAFETY: `av_rescale_q` is a pure arithmetic helper
            // (a / b * c rounded). No FFI state, no pointers,
            // no lifetime concerns.
            let one_frame = unsafe {
                ffmpeg::sys::av_rescale_q(
                    self.timeline.duration_us(),
                    self.encoder_time_base.into(),
                    self.output_time_base.into(),
                )
            };
            packet.set_duration(one_frame.max(1));
        }

        // Clone for stream BEFORE primary write (write_interleaved unrefs the packet).
        if let Some(ref mut stream) = self.stream {
            let mut clone = packet.clone();
            clone.set_stream(stream.video_index);
            clone.rescale_ts(self.output_time_base, stream.video_time_base);
            // Save PTS before write_interleaved blanks the packet
            // (av_interleaved_write_frame resets all fields to defaults).
            let video_pts = clone.pts().unwrap_or(0);
            let video_duration = clone.duration();
            if let Err(e) = stream.octx.write_packet(&mut clone) {
                log::warn!("RTMP stream write failed ({e}), disabling stream");
                self.stream = None;
            } else if let Some(ref mut sa) = stream.audio
                && let Err(e) = sa.write_for_video(
                    &mut stream.octx,
                    video_pts,
                    video_duration,
                    stream.video_time_base,
                )
            {
                log::warn!("Silent audio write failed ({e}), continuing without audio");
                stream.audio = None;
            }
        }

        self.octx.write_packet(packet)?;
        Ok(())
    }
}

fn sanitize_audio_start_time(secs: f64) -> f64 {
    if secs.is_finite() && secs > 0.0 {
        secs
    } else {
        0.0
    }
}

struct HevcHeaderCheck {
    parser: std::ptr::NonNull<ffmpeg::sys::AVCodecParserContext>,
    context: codec::context::Context,
}

impl HevcHeaderCheck {
    fn new(encoder: &ffmpeg::encoder::video::Encoder) -> Result<Self, EncodeError> {
        let context = codec::context::Context::from_parameters(codec::Parameters::from(encoder))?;
        // SAFETY: allocate an independent parser, owned until Drop.
        let parser =
            unsafe { ffmpeg::sys::av_parser_init(ffmpeg::sys::AVCodecID::AV_CODEC_ID_HEVC as _) };
        let parser = std::ptr::NonNull::new(parser).ok_or_else(|| {
            EncodeError::InvalidConfig("HEVC header parser is unavailable".into())
        })?;
        // SAFETY: parser is live and uniquely owned; encoder packets are complete access units.
        unsafe {
            (*parser.as_ptr()).flags |= ffmpeg::sys::PARSER_FLAG_COMPLETE_FRAMES;
        }
        Ok(Self { parser, context })
    }

    fn verify(
        &mut self,
        packet: &ffmpeg::Packet,
        expected: (u32, u32),
    ) -> Result<bool, EncodeError> {
        use ffmpeg::packet::Ref;
        // SAFETY: FFmpeg packets include parser padding; all pointers remain live for this call.
        let (result, actual) = unsafe {
            let packet = &*packet.as_ptr();
            let mut parsed = std::ptr::null_mut();
            let mut size = 0;
            let result = ffmpeg::sys::av_parser_parse2(
                self.parser.as_ptr(),
                self.context.as_mut_ptr(),
                &mut parsed,
                &mut size,
                packet.data,
                packet.size,
                packet.pts,
                packet.dts,
                packet.pos,
            );
            let parser = self.parser.as_ref();
            let actual = (parser.width, parser.height);
            (result, actual)
        };
        if result < 0 {
            return Err(ffmpeg::Error::from(result).into());
        }
        // Some backends publish parameter sets separately, before a picture supplies dimensions.
        if actual == (0, 0) {
            return Ok(false);
        }
        if actual != (expected.0 as i32, expected.1 as i32) {
            return Err(EncodeError::EncodedDimensions { expected, actual });
        }
        Ok(true)
    }
}

impl Drop for HevcHeaderCheck {
    fn drop(&mut self) {
        // SAFETY: this object exclusively owns this parser and closes it exactly once.
        unsafe {
            ffmpeg::sys::av_parser_close(self.parser.as_ptr());
        }
    }
}

fn validate_selection(name: &str, config: &EncoderConfig) -> Result<(), EncodeError> {
    if let Some(device) = &config.device {
        if device.is_empty() || device.contains('\0') {
            return Err(EncodeError::InvalidConfig(
                "device must be nonempty and contain no NUL bytes".into(),
            ));
        }
        if name.ends_with("_nvenc") {
            if device.parse::<u32>().is_err() {
                return Err(EncodeError::InvalidConfig(
                    "NVENC device must be a nonnegative GPU index".into(),
                ));
            }
        } else if !name.ends_with("_vaapi") {
            return Err(EncodeError::InvalidConfig(format!(
                "explicit device selection is not supported for {name}"
            )));
        }
    }
    if matches!(config.quality, Quality::Value(_))
        && !matches!(name, "libx264" | "libx265" | "libsvtav1" | "libaom-av1")
        && !["_nvenc", "_qsv", "_vaapi", "_amf", "_videotoolbox"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
    {
        return Err(EncodeError::InvalidConfig(format!(
            "quality selection is not supported by {name}"
        )));
    }
    Ok(())
}

fn validate_encoder_options(
    encoder: &ffmpeg::encoder::video::Video,
    options: &ffmpeg::Dictionary<'_>,
) -> Result<(), EncodeError> {
    for (key, _) in options.iter() {
        let key_c = std::ffi::CString::new(key)
            .map_err(|_| EncodeError::InvalidConfig("encoder option contains a NUL byte".into()))?;
        // SAFETY: the encoder and its private AVOptions context are alive; this only looks up an option.
        let option = unsafe {
            ffmpeg::sys::av_opt_find(
                encoder.as_ptr().cast_mut().cast(),
                key_c.as_ptr(),
                std::ptr::null(),
                0,
                ffmpeg::sys::AV_OPT_SEARCH_CHILDREN,
            )
        };
        if option.is_null() {
            return Err(EncodeError::InvalidConfig(format!(
                "encoder does not support option {key}"
            )));
        }
    }
    Ok(())
}

fn seconds_to_pts(secs: f64, time_base: Rational) -> i64 {
    let secs = sanitize_audio_start_time(secs);
    if secs == 0.0 {
        return 0;
    }

    let ts = (secs * f64::from(ffmpeg::ffi::AV_TIME_BASE)).round() as i64;
    unsafe {
        ffmpeg::sys::av_rescale_q(
            ts,
            Rational(1, ffmpeg::ffi::AV_TIME_BASE).into(),
            time_base.into(),
        )
    }
}

impl SilentAudio {
    /// Add a silent AAC audio stream to the output context and prepare the encoder.
    fn new(octx: &mut format::context::Output) -> Result<Self, EncodeError> {
        let aac = ffmpeg::encoder::find(codec::Id::AAC)
            .ok_or_else(|| EncodeError::CodecNotFound("AAC".into()))?;

        let needs_global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

        let mut ost = octx.add_stream(aac)?;
        let stream_index = ost.index();

        let mut enc = codec::context::Context::new_with_codec(aac)
            .encoder()
            .audio()?;

        let sample_rate = 44100u32;
        enc.set_rate(sample_rate as i32);
        enc.set_format(ffmpeg::format::Sample::F32(
            ffmpeg::format::sample::Type::Planar,
        ));
        enc.set_channel_layout(ffmpeg::ChannelLayout::STEREO);
        enc.set_bit_rate(128_000);
        enc.set_time_base(Rational(1, sample_rate as i32));

        if needs_global_header {
            enc.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        ost.set_parameters(&enc);
        let encoder = enc.open()?;
        ost.set_parameters(&encoder);

        let samples_per_frame = unsafe { (*encoder.as_ptr()).frame_size as usize };
        let samples_per_frame = if samples_per_frame == 0 {
            1024
        } else {
            samples_per_frame
        };

        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar),
            samples_per_frame,
            ffmpeg::ChannelLayout::STEREO,
        );
        frame.set_rate(sample_rate);

        // Native frame allocation does not initialize samples.
        for plane in 0..frame.planes() {
            frame.data_mut(plane).fill(0);
        }
        let output_time_base = Rational(0, 0); // updated after write_header

        Ok(Self {
            encoder,
            stream_index,
            output_time_base,
            frame,
            next_pts: None,
            samples_per_frame,
            sample_rate,
        })
    }

    fn write_for_video(
        &mut self,
        octx: &mut OutputIo,
        start: i64,
        duration: i64,
        time_base: Rational,
    ) -> Result<(), EncodeError> {
        let end = start.checked_add(duration).ok_or_else(|| {
            EncodeError::InvalidConfig("stream video packet time overflow".into())
        })?;
        if self.next_pts.is_none() {
            self.next_pts = Some(start.rescale(time_base, (1, self.sample_rate as i32)));
        }
        self.write_up_to(octx, end, time_base)
    }

    /// Generate silence through the supplied presentation time.
    fn write_up_to(
        &mut self,
        octx: &mut OutputIo,
        video_pts: i64,
        video_time_base: Rational,
    ) -> Result<(), EncodeError> {
        // Convert video PTS to audio sample count.
        let video_samples = video_pts.rescale(video_time_base, (1, self.sample_rate as i32));

        let Some(mut next_pts) = self.next_pts else {
            return Ok(());
        };
        while next_pts < video_samples {
            self.frame.set_pts(Some(next_pts));
            self.encoder.send_frame(&self.frame)?;
            next_pts += self.samples_per_frame as i64;
            self.next_pts = Some(next_pts);

            self.write_packets(octx)?;
        }
        Ok(())
    }

    fn finish(&mut self, octx: &mut OutputIo, end_us: i64) -> Result<(), EncodeError> {
        self.write_up_to(octx, end_us, Rational(1, 1_000_000))?;
        self.encoder.send_eof()?;
        self.write_packets(octx)
    }

    fn write_packets(&mut self, octx: &mut OutputIo) -> Result<(), EncodeError> {
        let mut packet = ffmpeg::Packet::empty();
        while receive_encoded_packet(&mut self.encoder, &mut packet)? {
            packet.set_stream(self.stream_index);
            packet.rescale_ts(Rational(1, self.sample_rate as i32), self.output_time_base);
            octx.write_packet(&mut packet)?;
        }
        Ok(())
    }
}

fn receive_encoded_packet(
    encoder: &mut ffmpeg::encoder::Encoder,
    packet: &mut ffmpeg::Packet,
) -> Result<bool, ffmpeg::Error> {
    match encoder.receive_packet(packet) {
        Ok(()) => Ok(true),
        Err(ffmpeg::Error::Eof) => Ok(false),
        Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => Ok(false),
        Err(error) => Err(error),
    }
}

/// Build encoder-specific FFmpeg options.
/// Scale a 1080p-tuned bitrate ceiling (Mbps) by output pixel count, so
/// higher resolutions get a proportionally higher cap. Clamped to
/// [0.5x, 8x] of the 1080p baseline; always >= 1 Mbps.
fn scale_bitrate_mbps(base_mbps: u32, width: u32, height: u32) -> u32 {
    let scale = (f64::from(width) * f64::from(height) / (1920.0 * 1080.0)).clamp(0.5, 8.0);
    ((f64::from(base_mbps) * scale).round() as u32).max(1)
}

fn build_encoder_opts(
    name: &str,
    quality: Quality,
    preset_override: Option<&str>,
    width: u32,
    height: u32,
) -> ffmpeg::Dictionary<'static> {
    let mut opts = ffmpeg::Dictionary::new();

    // Bitrate ceilings below are tuned for 1080p. Scale them with pixel
    // count so higher resolutions (2K/4K) aren't capped far below the
    // quality the CQ setting actually wants. Returns an "NM" string.
    let mbps = |base: u32| format!("{}M", scale_bitrate_mbps(base, width, height));

    // When a quality override is set, derive the effective preset from
    // the quality value so NVENC preset/maxrate scale with the CQ.
    let quality_override = match quality {
        Quality::Value(value) => Some(value),
        _ => None,
    };
    let effective_preset = match quality {
        Quality::Value(75..) => QualityPreset::High,
        Quality::Value(40..) => QualityPreset::Balanced,
        Quality::Value(_) => QualityPreset::Fast,
        Quality::Preset(preset) => preset,
    };

    match name {
        "h264_nvenc" => {
            // NVENC VBR with per-quality bitrate ceiling. Prior defaults
            // (10M / 15M across all quality presets) artificially capped
            // High output at ~15 Mbps even with cq=19. Scale the ceilings
            // with quality so "high" actually delivers the visual bump.
            let (preset, cq, bv, maxrate) = match effective_preset {
                QualityPreset::Fast => ("p3", "28", 8, 12),
                QualityPreset::Balanced => ("p4", "23", 12, 18),
                QualityPreset::High => ("p5", "19", 20, 30),
            };
            opts.set("preset", preset);
            opts.set("tune", "hq");
            opts.set("rc", "vbr");
            opts.set("cq", cq);
            opts.set("b", &mbps(bv));
            opts.set("maxrate", &mbps(maxrate));
            opts.set("profile", "high");
            opts.set("spatial-aq", "1");
            opts.set("temporal-aq", "1");
        }
        "hevc_nvenc" => {
            // HEVC ~30% more efficient than H264, so ceilings scale down.
            let (preset, cq, bv, maxrate) = match effective_preset {
                QualityPreset::Fast => ("p3", "28", 6, 10),
                QualityPreset::Balanced => ("p4", "23", 9, 14),
                QualityPreset::High => ("p5", "19", 15, 22),
            };
            opts.set("preset", preset);
            opts.set("tune", "hq");
            opts.set("rc", "vbr");
            opts.set("cq", cq);
            opts.set("b", &mbps(bv));
            opts.set("maxrate", &mbps(maxrate));
            opts.set("profile", "main");
            opts.set("spatial-aq", "1");
            opts.set("temporal-aq", "1");
        }
        "av1_nvenc" => {
            // AV1 another ~20% tighter than HEVC.
            let (preset, cq, bv, maxrate) = match effective_preset {
                QualityPreset::Fast => ("p3", "32", 5, 8),
                QualityPreset::Balanced => ("p4", "27", 7, 11),
                QualityPreset::High => ("p5", "22", 12, 18),
            };
            opts.set("preset", preset);
            opts.set("tune", "hq");
            opts.set("rc", "vbr");
            opts.set("cq", cq);
            opts.set("b", &mbps(bv));
            opts.set("maxrate", &mbps(maxrate));
        }
        "h264_qsv" | "hevc_qsv" | "av1_qsv" => {
            let gq = match effective_preset {
                QualityPreset::Fast => "28",
                QualityPreset::Balanced => "23",
                QualityPreset::High => "19",
            };
            opts.set("preset", "medium");
            opts.set("global_quality", gq);
            if name == "h264_qsv" {
                opts.set("profile", "high");
            }
        }
        "h264_videotoolbox" => {
            // VideoToolbox quality maps to kVTCompressionPropertyKey_Quality
            // (0.0-1.0 float, FFmpeg divides global_quality by 100). The
            // relationship between VT quality and bitrate is exponential:
            // 0.80 -> ~104 Mbps, 1.0 -> ~297 Mbps (near-lossless). To keep
            // bitrates comparable with NVENC / software CRF at the same
            // --quality-value, we cap values at 75 (Apple's "high" tier)
            // and enforce a maxrate ceiling via kVTCompressionPropertyKey_DataRateLimits.
            let (q, maxrate) = match effective_preset {
                QualityPreset::Fast => ("50", "12000000"),
                QualityPreset::Balanced => ("60", "18000000"),
                QualityPreset::High => ("70", "30000000"),
            };
            opts.set("global_quality", q);
            opts.set("maxrate", maxrate);
            opts.set("profile", "high");
        }
        "hevc_videotoolbox" => {
            // Same VT quality model as H.264; HEVC is ~30% more efficient
            // so we use slightly lower ceilings.
            let (q, maxrate) = match effective_preset {
                QualityPreset::Fast => ("50", "10000000"),
                QualityPreset::Balanced => ("60", "14000000"),
                QualityPreset::High => ("70", "22000000"),
            };
            opts.set("global_quality", q);
            opts.set("maxrate", maxrate);
            opts.set("profile", "main");
        }
        "h264_vaapi" | "hevc_vaapi" | "av1_vaapi" => {
            let qp = match effective_preset {
                QualityPreset::Fast => "28",
                QualityPreset::Balanced => "23",
                QualityPreset::High => "19",
            };
            opts.set("qp", qp);
            if name == "h264_vaapi" {
                opts.set("profile", "high");
            }
        }
        "h264_v4l2m2m" | "hevc_v4l2m2m" => {
            // The wrapper exposes no portable constant-quality control.
            log::info!("Encoder '{name}' uses driver quality defaults");
        }
        "h264_amf" | "hevc_amf" | "av1_amf" => {
            let q = match effective_preset {
                QualityPreset::Fast => "22",
                QualityPreset::Balanced => "18",
                QualityPreset::High => "14",
            };
            opts.set("quality", "balanced");
            opts.set("qp_i", q);
            opts.set("qp_p", q);
            opts.set("rc", "cqp");
        }
        "libx265" => {
            let (preset, crf_val) = match effective_preset {
                QualityPreset::Fast => ("veryfast", "28"),
                QualityPreset::Balanced => ("fast", "25"),
                QualityPreset::High => ("medium", "21"),
            };
            opts.set("preset", preset);
            opts.set("crf", crf_val);
            opts.set("profile", "main");
        }
        "libsvtav1" => {
            let (preset, crf_val) = match effective_preset {
                QualityPreset::Fast => ("10", "35"),
                QualityPreset::Balanced => ("7", "30"),
                QualityPreset::High => ("4", "25"),
            };
            opts.set("preset", preset);
            opts.set("crf", crf_val);
        }
        "libaom-av1" => {
            let (crf_val, cpu_used) = match effective_preset {
                QualityPreset::Fast => ("35", "8"),
                QualityPreset::Balanced => ("30", "6"),
                QualityPreset::High => ("25", "4"),
            };
            opts.set("crf", crf_val);
            opts.set("cpu-used", cpu_used);
            opts.set("row-mt", "1");
        }
        "libx264" => {
            let (preset, crf_val) = match effective_preset {
                QualityPreset::Fast => ("ultrafast", "32"),
                QualityPreset::Balanced => ("veryfast", "28"),
                QualityPreset::High => ("fast", "23"),
            };
            opts.set("preset", preset);
            opts.set("crf", crf_val);
            opts.set("profile", "high");
            // On Tegra (Jetson), limit threads to avoid starving other
            // pipeline stages (capture + pairing threads need CPU too).
            // x264 defaults to num_cpus which over-subscribes.
            if std::path::Path::new("/etc/nv_tegra_release").exists()
                || std::fs::read_to_string("/proc/device-tree/compatible")
                    .unwrap_or_default()
                    .contains("nvidia,tegra")
            {
                let threads = std::env::var("RECO_X264_THREADS")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(4);
                log::info!("Tegra platform detected, limiting libx264 to {threads} threads");
                opts.set("threads", &threads.to_string());
            }
        }
        _ => {
            log::info!("Encoder '{name}' has no quality presets configured, using FFmpeg defaults");
        }
    }

    // Apply normalized quality override (0-100, higher = better).
    // Converts from the consumer-facing scale to encoder-specific parameters.
    if let Some(q) = quality_override {
        let is_videotoolbox = name.contains("videotoolbox");
        if is_videotoolbox {
            // VideoToolbox quality is exponential: VT 0.80 -> ~104 Mbps,
            // VT 1.0 -> ~297 Mbps (near-lossless). To produce bitrates
            // comparable with NVENC at the same --quality-value, compress
            // the consumer 0-100 range into VT 40-75 and enforce a maxrate
            // ceiling via DataRateLimits. VT 0.75 is Apple's "high" tier;
            // anything above produces diminishing-returns bitrate inflation.
            let vt_q = 40.0 + (q as f64 / 100.0) * 35.0;
            let val = format!("{:.0}", vt_q);
            opts.set("global_quality", &val);
            // Tiered maxrate matches NVENC ceilings (in bits/sec).
            let maxrate = if q >= 75 {
                "30000000"
            } else if q >= 40 {
                "18000000"
            } else {
                "12000000"
            };
            opts.set("maxrate", maxrate);
            log::info!(
                "Encoder quality: {q} -> {name} global_quality={val} (VT {:.2}), maxrate={maxrate}",
                vt_q / 100.0
            );
        } else {
            // CRF-style encoders: crf = 40.0 - (quality / 100.0) * 28.0
            let crf = 40.0 - (q as f64 / 100.0) * 28.0;
            let crf_str = format!("{crf:.2}");
            let integer = format!("{crf:.0}");
            match name {
                "libx264" | "libx265" => opts.set("crf", &crf_str),
                "libsvtav1" | "libaom-av1" => opts.set("crf", &integer),
                name if name.ends_with("_nvenc") => opts.set("cq", &crf_str),
                name if name.ends_with("_qsv") => opts.set("global_quality", &integer),
                name if name.ends_with("_vaapi") => opts.set("qp", &integer),
                name if name.ends_with("_amf") => {
                    opts.set("qp_i", &integer);
                    opts.set("qp_p", &integer);
                }
                _ => {}
            }
        }
    }

    // Apply preset override.
    if let Some(preset) = preset_override {
        opts.set("preset", preset);
    }

    opts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_audio_samples_are_initialized_to_silence() {
        crate::init();
        let dir = tempfile::tempdir().unwrap();
        let mut output = format::output_as(&dir.path().join("audio.flv"), "flv").unwrap();
        let silent = SilentAudio::new(&mut output).unwrap();
        assert_eq!(silent.frame.planes(), 2);
        for plane in 0..silent.frame.planes() {
            assert!(silent.frame.data(plane).iter().all(|sample| *sample == 0));
        }
    }

    #[test]
    fn streaming_audio_matches_the_video_timeline() {
        crate::init();
        for fps in [Rational(30, 1), Rational(30000, 1001)] {
            for (frames, origin, gap) in [
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (30, 0, 0),
                (30, 1_000_000, 0),
                (30, 60_000_000, 0),
                (30, 1_000_000, 150_000),
            ] {
                let dir = tempfile::tempdir().unwrap();
                let config = EncoderConfig {
                    software_only: true,
                    encoder_name: Some("libx264".into()),
                    preset: Some("ultrafast".into()),
                    ..Default::default()
                };
                let mut encoder =
                    VideoEncoder::new(&dir.path().join("primary.mp4"), 64, 64, fps, &config)
                        .unwrap();
                let received = dir.path().join("stream.flv");
                let mut output = OutputIo::open(&received, "flv", &config, false).unwrap();
                let mut stream = output
                    .add_stream(ffmpeg::encoder::find(codec::Id::None))
                    .unwrap();
                stream.set_parameters(
                    encoder
                        .octx
                        .stream(encoder.stream_index)
                        .unwrap()
                        .parameters(),
                );
                // SAFETY: the new stream owns writable codec parameters.
                unsafe {
                    (*stream.parameters().as_mut_ptr()).codec_tag = 0;
                }
                let video_index = stream.index();
                let mut audio = SilentAudio::new(&mut output).unwrap();
                let audio_frame_us = (audio.samples_per_frame as i64)
                    .rescale((1, audio.sample_rate as i32), (1, 1_000_000));
                output.write_header(ffmpeg::Dictionary::new()).unwrap();
                audio.output_time_base = output.stream(audio.stream_index).unwrap().time_base();
                encoder.stream = Some(StreamOutput {
                    video_index,
                    video_time_base: output.stream(video_index).unwrap().time_base(),
                    octx: output,
                    audio: Some(audio),
                });
                for frame in 0..frames {
                    let pts = origin
                        + frame * 1_000_000 * i64::from(fps.1) / i64::from(fps.0)
                        + if frame >= frames / 2 { gap } else { 0 };
                    encoder
                        .write_nv12_frame(&vec![128; 64 * 64 * 3 / 2], Some(pts))
                        .unwrap();
                }
                encoder.finish().unwrap();
                encoder.finish().unwrap();
                drop(encoder);

                if frames == 0 {
                    continue;
                }

                let mut received = format::input(&received).unwrap();
                let mut audio_end = None;
                let mut last_video = None;
                let mut audio_start = None;
                let mut video_start = None;
                for (stream, packet) in received.packets() {
                    let pts = packet.pts().unwrap();
                    match stream.parameters().medium() {
                        ffmpeg::media::Type::Audio => {
                            audio_start
                                .get_or_insert(pts.rescale(stream.time_base(), (1, 1_000_000)));
                            let end = (pts + packet.duration())
                                .rescale(stream.time_base(), (1, 1_000_000));
                            audio_end = Some(audio_end.map_or(end, |last: i64| last.max(end)));
                        }
                        ffmpeg::media::Type::Video => {
                            video_start
                                .get_or_insert(pts.rescale(stream.time_base(), (1, 1_000_000)));
                            last_video = Some(pts.rescale(stream.time_base(), (1, 1_000_000)));
                        }
                        _ => {}
                    }
                }
                let video_end =
                    last_video.unwrap() + 1_000_000 * i64::from(fps.1) / i64::from(fps.0);
                let audio_end = audio_end.expect("stream contains no encoded audio");
                assert!(
                    video_start.unwrap().abs_diff(audio_start.unwrap())
                        <= (audio_frame_us + 2_000) as u64,
                    "generated audio does not share the video origin: video={video_start:?}, audio={audio_start:?}"
                );
                assert!(
                    audio_end >= video_end - 2_000,
                    "audio ends at {audio_end}, video at {video_end}; fps={fps}, frames={frames}"
                );
                assert!(
                    audio_end <= video_end + audio_frame_us + 2_000,
                    "audio exceeds one AAC frame of tail padding"
                );
            }
        }
    }

    #[test]
    fn hevc_header_check_accepts_separate_parameter_sets_and_rejects_wrong_dimensions() {
        crate::init();
        let codec = encoder::find_by_name("libx265").expect("software HEVC encoder");
        let mut video = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .unwrap();
        video.set_width(64);
        video.set_height(64);
        video.set_format(Pixel::YUV420P);
        video.set_time_base((1, 30));
        video.set_flags(codec::Flags::GLOBAL_HEADER);
        let mut options = ffmpeg::Dictionary::new();
        options.set("preset", "ultrafast");
        options.set("x265-params", "log-level=error:pools=none");
        let mut encoder = video.open_with(options).unwrap();
        let mut check = HevcHeaderCheck::new(&encoder).unwrap();
        // Exercise in-band headers without relying on a duplicate in codec extradata.
        check.context = codec::context::Context::new_with_codec(codec);
        // SAFETY: the opened encoder owns the extradata; Packet::copy takes an owned padded copy.
        let headers = unsafe {
            let context = &*encoder.as_ptr();
            assert!(context.extradata_size > 0);
            ffmpeg::Packet::copy(std::slice::from_raw_parts(
                context.extradata,
                context.extradata_size as usize,
            ))
        };
        assert!(!check.verify(&headers, (64, 64)).unwrap());
        let mut frame = VideoFrame::new(Pixel::YUV420P, 64, 64);
        for plane in 0..3 {
            frame.data_mut(plane).fill(128);
        }
        frame.set_pts(Some(0));
        encoder.send_frame(&frame).unwrap();
        encoder.send_eof().unwrap();
        let mut packet = ffmpeg::Packet::empty();
        encoder.receive_packet(&mut packet).unwrap();
        assert!(check.verify(&packet, (64, 64)).unwrap());
        assert!(matches!(
            check.verify(&packet, (66, 64)),
            Err(EncodeError::EncodedDimensions {
                expected: (66, 64),
                actual: (64, 64)
            })
        ));
    }

    #[test]
    fn platform_defaults_remain_available_without_numeric_quality() {
        for name in [
            "h264_v4l2m2m",
            "hevc_v4l2m2m",
            "h264_mf",
            "h264_mediacodec",
            "libopenh264",
        ] {
            validate_selection(name, &EncoderConfig::default()).unwrap();
            let explicit = EncoderConfig {
                quality: Quality::Value(80),
                ..Default::default()
            };
            assert!(validate_selection(name, &explicit).is_err());
            assert!(
                build_encoder_opts(name, Quality::default(), None, 1920, 1080)
                    .get("qp")
                    .is_none()
            );
        }
    }

    #[test]
    fn explicit_selection_errors_preserve_destination() {
        for config in [
            EncoderConfig {
                codec: Some(Codec::HEVC),
                encoder_name: Some("libx264".into()),
                ..Default::default()
            },
            EncoderConfig {
                encoder_name: Some("libx264".into()),
                quality: Quality::Value(101),
                ..Default::default()
            },
            EncoderConfig {
                encoder_name: Some("libx264".into()),
                device: Some("0".into()),
                ..Default::default()
            },
            EncoderConfig {
                device: Some("/dev/dri/renderD129".into()),
                ..Default::default()
            },
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("existing.mp4");
            std::fs::write(&path, b"keep").unwrap();
            assert!(VideoEncoder::new(&path, 64, 64, (30, 1).into(), &config).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), b"keep");
        }
    }

    #[test]
    fn unknown_encoder_options_are_not_silently_dropped() {
        crate::init();
        let codec = encoder::find_by_name("libx264").unwrap();
        let encoder = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .unwrap();
        let mut options = ffmpeg::Dictionary::new();
        options.set("b:v", "1000000");
        assert!(validate_encoder_options(&encoder, &options).is_err());
        let options = build_encoder_opts("libx264", Quality::default(), None, 1920, 1080);
        validate_encoder_options(&encoder, &options).unwrap();
    }

    #[test]
    fn invalid_quality_does_not_open_the_destination() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("existing.mp4");
        std::fs::write(&path, b"existing output").unwrap();
        let config = EncoderConfig {
            encoder_name: Some("libx264".into()),
            quality: Quality::Value(255),
            ..Default::default()
        };
        assert!(VideoEncoder::new(&path, 64, 64, (30, 1).into(), &config).is_err());
        assert_eq!(std::fs::read(path).unwrap(), b"existing output");
    }

    #[test]
    fn invalid_video_layout_does_not_open_the_destination() {
        let config = EncoderConfig {
            encoder_name: Some("libx264".into()),
            ..Default::default()
        };
        for (width, height, fps) in [
            (0, 64, (30, 1)),
            (64, 0, (30, 1)),
            (63, 64, (30, 1)),
            (64, 63, (30, 1)),
            (u32::MAX - 1, 64, (30, 1)),
            (64, u32::MAX - 1, (30, 1)),
            (64, 64, (0, 1)),
            (64, 64, (30, 0)),
            (64, 64, (-30, 1)),
            (64, 64, (30, -1)),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("existing.mp4");
            std::fs::write(&path, b"existing output").unwrap();
            assert!(matches!(
                VideoEncoder::new(&path, width, height, fps.into(), &config),
                Err(EncodeError::InvalidConfig(_))
            ));
            assert_eq!(std::fs::read(path).unwrap(), b"existing output");
        }
    }

    #[test]
    fn nvenc_bitrate_uses_the_library_option_name() {
        for name in ["h264_nvenc", "hevc_nvenc", "av1_nvenc"] {
            let options = build_encoder_opts(name, Quality::default(), None, 1920, 1080);
            assert!(options.get("b").is_some(), "{name}: missing bitrate");
            assert!(
                options.get("b:v").is_none(),
                "CLI stream specifier in library options"
            );
        }
    }

    #[test]
    fn packet_receive_errors_are_not_exhaustion() {
        crate::init();
        let codec = encoder::find_by_name("libx264").expect("test encoder");
        let mut encoder = codec::context::Context::new_with_codec(codec).encoder();
        let mut packet = ffmpeg::Packet::empty();
        let expected = encoder.receive_packet(&mut packet).unwrap_err();
        assert_eq!(
            receive_encoded_packet(&mut encoder, &mut packet),
            Err(expected),
        );
    }

    #[test]
    fn packet_receive_accepts_pending_and_drained_encoders() {
        crate::init();
        let codec = encoder::find_by_name("libx264").expect("test encoder");
        let mut encoder = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .unwrap();
        encoder.set_width(64);
        encoder.set_height(64);
        encoder.set_format(Pixel::YUV420P);
        encoder.set_time_base((1, 30));
        encoder.set_threading(ffmpeg::threading::Config::count(1));
        let mut encoder = encoder.open().unwrap();
        let mut packet = ffmpeg::Packet::empty();
        assert_eq!(receive_encoded_packet(&mut encoder, &mut packet), Ok(false));
        encoder.send_eof().unwrap();
        assert_eq!(receive_encoded_packet(&mut encoder, &mut packet), Ok(false));
        assert_eq!(receive_encoded_packet(&mut encoder, &mut packet), Ok(false));
    }

    /// Only a positive, finite start time should shift audio; anything else
    /// (default exports, garbage values) must leave the soundtrack untouched.
    #[test]
    fn sanitize_audio_start_time_keeps_only_valid_offsets() {
        assert_eq!(sanitize_audio_start_time(12.5), 12.5);
        assert_eq!(sanitize_audio_start_time(0.0), 0.0);
        assert_eq!(sanitize_audio_start_time(-4.0), 0.0);
        assert_eq!(sanitize_audio_start_time(f64::NAN), 0.0);
        assert_eq!(sanitize_audio_start_time(f64::INFINITY), 0.0);
    }

    /// The audio rebase offset must equal the requested start time expressed
    /// in the output stream's time base, so trimmed audio lands at time zero
    /// in sync with the video. This is the core of the `--start-time` fix.
    #[test]
    fn seconds_to_pts_converts_into_the_stream_time_base() {
        // One second in a 44.1 kHz audio time base is exactly 44_100 ticks.
        assert_eq!(seconds_to_pts(1.0, Rational(1, 44_100)), 44_100);
        // Half a second in a millisecond time base is 500 ticks.
        assert_eq!(seconds_to_pts(0.5, Rational(1, 1_000)), 500);
    }

    /// A zero or invalid start time produces no shift, so exports without
    /// `--start-time` mux audio exactly as before this fix.
    #[test]
    fn seconds_to_pts_is_zero_for_no_offset() {
        assert_eq!(seconds_to_pts(0.0, Rational(1, 44_100)), 0);
        assert_eq!(seconds_to_pts(-2.0, Rational(1, 44_100)), 0);
        assert_eq!(seconds_to_pts(f64::NAN, Rational(1, 44_100)), 0);
    }

    /// Bitrate ceilings scale with output resolution vs the 1080p baseline.
    #[test]
    fn bitrate_scales_with_resolution() {
        // 1080p: unchanged baseline.
        assert_eq!(scale_bitrate_mbps(30, 1920, 1080), 30);
        // 2560x1440 (~1.78x pixels).
        assert_eq!(scale_bitrate_mbps(30, 2560, 1440), 53);
        // 4K (4x pixels).
        assert_eq!(scale_bitrate_mbps(30, 3840, 2160), 120);
        // Clamped: very high res caps at 8x.
        assert_eq!(scale_bitrate_mbps(30, 7680, 4320), 240);
        // Clamped: low res floors at 0.5x, never below 1 Mbps.
        assert_eq!(scale_bitrate_mbps(30, 640, 360), 15);
        assert_eq!(scale_bitrate_mbps(1, 320, 240), 1);
    }
}
