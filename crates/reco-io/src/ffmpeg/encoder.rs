//! Encode RGBA, NV12 or planar YUV420P frames with FFmpeg.
//!
//! Wraps FFmpeg's muxer, encoder, and swscale with explicit codec and container selection. Supports hardware-accelerated encoding
//! (NVENC, QSV, VideoToolbox, VAAPI) with automatic fallback to software.

extern crate ffmpeg_next as ffmpeg;

mod audio;
mod catalog;
mod container;
mod options;
mod p010;
mod silent_audio;

use audio::AudioPassthrough;
use catalog::{
    EncoderFamily, EncoderSpec, StagingMode, find_encoder_spec, specs_for_codec, validate_selection,
};
use container::{
    ContainerPolicy, HevcHeaderCheck, OutputColor, apply_video_codec_tag,
    muxer_requires_global_header,
};
use options::validate_encoder_options;
use silent_audio::SilentAudio;
// Re-imported for `audio.rs` (`use super::*`) and unit tests.
use silent_audio::{sanitize_audio_start_time, seconds_to_pts};

use super::hevc_color::HevcColor;
use super::hw_upload::HardwareUpload;
use super::live_audio::LiveAudio;
use super::output_io::OutputIo;
pub use super::output_io::{StreamMonitor, StreamStatus};
use crate::output::{BitDepth, Codec, Container, Quality};
pub use catalog::{available_encoders, available_h264_encoders};

use ffmpeg::format::Pixel;
use ffmpeg::software::scaling::{context::Context as ScalingContext, flag::Flags as ScalingFlags};
use ffmpeg::util::frame::video::Video as VideoFrame;
use ffmpeg::{Rational, Rescale, codec, encoder, format};
use std::path::{Path, PathBuf};
use thiserror::Error;

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

/// Configuration for the video encoder.
#[derive(Debug, Clone, Default)]
pub struct EncoderConfig {
    /// Force a specific encoder by name, or `None` for auto-detection.
    pub encoder_name: Option<String>,
    /// Output codec. When absent, infer from the named encoder, or select H.264 for 8-bit and HEVC for 10-bit.
    pub codec: Option<Codec>,
    /// Output sample precision. 10-bit requires P010LE input and HEVC Main 10. FLV and RTMP are incompatible.
    ///
    /// Main 10 VAAPI uses the `hev1` sample entry in MP4 and MOV to retain in-band parameter sets.
    pub bit_depth: BitDepth,
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
    pub audio_source: Option<Vec<PathBuf>>,
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
    /// Requires FFmpeg 8+ for video-keyframe recovery with audio. Observe rejection or transport failure with [`VideoEncoder::stream_monitor`].
    pub stream_url: Option<String>,
    /// Existing application cancellation signal. Interrupts network output, not local recording finalization.
    pub interrupted: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Network open, synchronous I/O and final-drain deadline. `None` selects ten seconds.
    /// Cooperative FFmpeg interruption does not bound native DNS resolution. Optional FIFO writes are isolated, not timed by producer activity.
    pub network_timeout: Option<std::time::Duration>,
    /// Source colourimetry metadata (primaries, transfer characteristics, matrix) to tag on the output stream.
    ///
    /// When empty (default), rendered output uses BT.709 limited-range matrix with unspecified
    /// primaries and transfer, and generic encoding leaves VUI colour fields unspecified.
    /// When populated (for example from an HLG or PQ BT.2020 source), the encoder propagates the
    /// metadata to the codec context, container headers, and HEVC VUI headers.
    pub color_metadata: super::ColorMetadata,
}

impl EncoderConfig {
    fn auto_codec(&self) -> Codec {
        self.codec.unwrap_or(match self.bit_depth {
            BitDepth::Eight => Codec::H264,
            BitDepth::Ten => Codec::HEVC,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct ResolvedCandidate<'a> {
    name: &'a str,
    family: EncoderFamily,
    is_hardware: bool,
    staging: StagingMode,
}

struct OpenedVideoEncoder {
    encoder: ffmpeg::encoder::video::Encoder,
    scaler: Option<ScalingContext>,
    stream_index: usize,
    encoder_time_base: Rational,
    hardware_upload: Option<HardwareUpload>,
}

/// FFmpeg video encoder for 8-bit frames or packed 10-bit P010LE frames.
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
    rgba_input: Option<(ScalingContext, VideoFrame)>,
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
    yuv_frame: VideoFrame,
    hardware_upload: Option<HardwareUpload>,
    /// Audio passthrough state (if enabled).
    audio: Option<AudioPassthrough>,
    /// Secondary output context for RTMP streaming (same encoded packets).
    stream: Option<StreamOutput>,
    stream_monitor: Option<StreamMonitor>,
    broadcast_clock: Option<super::encoder_timing::BroadcastClock>,
}

/// State for a secondary RTMP stream output that receives cloned packets
/// from the primary encode pass.
struct StreamOutput {
    octx: OutputIo,
    video_index: usize,
    video_time_base: Rational,
    audio: StreamAudio,
}

enum StreamAudio {
    Copy { index: usize, time_base: Rational },
    Silence(SilentAudio),
}

// SAFETY: VideoEncoder is only used from a single thread. The raw pointers
// inside FFmpeg's SwsContext/Encoder are not shared across threads.
unsafe impl Send for VideoEncoder {}
// SAFETY: Same single-thread usage as VideoEncoder.
unsafe impl Send for StreamOutput {}

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
        Self::new_with_color(
            path,
            width,
            height,
            fps,
            config,
            OutputColor {
                metadata: config.color_metadata.normalized(),
                full_range: None,
                rgba_matrix: None,
            },
            None,
        )
    }

    pub(crate) fn new_rendered(
        path: &Path,
        width: u32,
        height: u32,
        fps: Rational,
        config: &EncoderConfig,
        live_audio: Option<LiveAudio>,
        output_color: reco_core::render::color::YuvColor,
    ) -> Result<Self, EncodeError> {
        let metadata = config.color_metadata.with_matrix(output_color.matrix);
        Self::new_with_color(
            path,
            width,
            height,
            fps,
            config,
            OutputColor {
                metadata,
                full_range: Some(output_color.full_range),
                rgba_matrix: Some(output_color.matrix),
            },
            live_audio,
        )
    }

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
            OutputColor {
                metadata: config.color_metadata.normalized(),
                full_range: Some(full_range),
                rgba_matrix: None,
            },
            None,
        )
    }

    fn new_with_color(
        path: &Path,
        width: u32,
        height: u32,
        fps: Rational,
        config: &EncoderConfig,
        color: OutputColor,
        live_audio: Option<LiveAudio>,
    ) -> Result<Self, EncodeError> {
        Self::validate_layout_and_config(width, height, fps, config)?;
        if config.bit_depth == BitDepth::Ten
            && (config.container == Container::Flv
                || config.stream_url.is_some()
                || path.to_str().is_some_and(|path| {
                    path.starts_with("rtmp://") || path.starts_with("rtmps://")
                }))
        {
            return Err(EncodeError::InvalidConfig(
                "10-bit HEVC output is incompatible with FLV or RTMP".into(),
            ));
        }
        crate::init();

        let forced_encoder = config.encoder_name.is_some();
        let candidates = Self::resolve_candidates(config)?;
        let policy = ContainerPolicy::for_container(config.container);

        // Try each candidate — create a fresh output context per attempt
        // because add_stream + write_header can leave the context in a
        // bad state if the encoder fails to open.
        let mut last_err = None;

        for candidate in candidates {
            let Some(codec) = encoder::find_by_name(candidate.name) else {
                continue;
            };

            if color.has_tags() && codec.id() == codec::Id::HEVC && !HevcColor::available() {
                last_err = Some(ffmpeg::Error::BsfNotFound.into());
                continue;
            }

            // The selected container is authoritative, even without a matching extension.
            let mut octx = OutputIo::open(path, policy.muxer_name(), config, false)?;

            match Self::try_open(
                &mut octx, codec, candidate, width, height, fps, config, color,
            ) {
                Ok(opened) => {
                    let hevc_color = if color.has_tags() && codec.id() == codec::Id::HEVC {
                        let filter =
                            match HevcColor::new(&opened.encoder, color.metadata, color.full_range)
                            {
                                Ok(filter) => filter,
                                Err(error) => {
                                    last_err = Some(error.into());
                                    continue;
                                }
                            };
                        octx.stream_mut(opened.stream_index)
                            .expect("video stream we just added")
                            .set_parameters(filter.parameters()?);
                        Some(filter)
                    } else {
                        None
                    };

                    return Self::finish_muxer_setup(
                        octx,
                        opened,
                        hevc_color,
                        codec.id(),
                        candidate,
                        policy,
                        path,
                        width,
                        height,
                        fps,
                        config,
                        color,
                        live_audio,
                    );
                }
                Err(e) => {
                    if forced_encoder {
                        log::warn!("Encoder {} failed to open: {e}", candidate.name);
                    } else {
                        log::debug!(
                            "Encoder {} failed to open: {e}, trying next...",
                            candidate.name
                        );
                    }
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or(EncodeError::CodecNotFound(
            config.auto_codec().label().to_string(),
        )))
    }

    fn validate_layout_and_config(
        width: u32,
        height: u32,
        fps: Rational,
        config: &EncoderConfig,
    ) -> Result<(), EncodeError> {
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
        if config.device.is_some() && config.encoder_name.is_none() {
            return Err(EncodeError::InvalidConfig(
                "device selection requires a named encoder".into(),
            ));
        }
        Ok(())
    }

    fn resolve_candidates(
        config: &EncoderConfig,
    ) -> Result<Vec<ResolvedCandidate<'_>>, EncodeError> {
        let bit_depth = config.bit_depth;
        if bit_depth == BitDepth::Ten && config.auto_codec() != Codec::HEVC {
            return Err(EncodeError::InvalidConfig(
                "10-bit output requires HEVC Main 10".into(),
            ));
        }

        let candidates: Vec<ResolvedCandidate<'_>> = if let Some(ref name) = config.encoder_name {
            let spec = find_encoder_spec(name);
            let staging = match (bit_depth, spec.and_then(|spec| spec.staging_for(bit_depth))) {
                (_, Some(staging)) => staging,
                (BitDepth::Eight, None) => StagingMode::Yuv420p,
                (BitDepth::Ten, None) => {
                    return Err(EncodeError::InvalidConfig(format!(
                        "encoder {name} is unsupported for 10-bit HEVC Main 10 output"
                    )));
                }
            };
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
            validate_selection(name, config)?;
            let family = spec
                .map(|s| s.family)
                .or_else(|| EncoderFamily::from_suffix(name))
                .unwrap_or(EncoderFamily::Software);
            let is_hardware = spec.is_some_and(EncoderSpec::is_hardware);
            vec![ResolvedCandidate {
                name: name.as_str(),
                family,
                is_hardware,
                staging,
            }]
        } else {
            specs_for_codec(config.auto_codec())
                .filter(|spec| encoder::find_by_name(spec.name).is_some())
                .filter(|spec| !config.software_only || !spec.is_hardware())
                .filter(|spec| spec.auto_candidate_allowed(bit_depth))
                .filter(|spec| validate_selection(spec.name, config).is_ok())
                .filter_map(|spec| {
                    Some(ResolvedCandidate {
                        name: spec.name,
                        family: spec.family,
                        is_hardware: spec.is_hardware(),
                        staging: spec.staging_for(bit_depth)?,
                    })
                })
                .collect()
        };

        if candidates.is_empty() {
            return Err(EncodeError::CodecNotFound(
                config.auto_codec().label().to_string(),
            ));
        }
        Ok(candidates)
    }

    /// Attempt to open a specific encoder. Returns the opened encoder + scaler
    /// + stream metadata, or an error.
    #[allow(clippy::too_many_arguments)]
    fn try_open(
        octx: &mut format::context::Output,
        codec: ffmpeg::Codec,
        candidate: ResolvedCandidate<'_>,
        width: u32,
        height: u32,
        fps: Rational,
        config: &EncoderConfig,
        color: OutputColor,
    ) -> Result<OpenedVideoEncoder, EncodeError> {
        let needs_global_header = muxer_requires_global_header(octx);

        let mut ost = octx.add_stream(codec)?;
        let stream_index = ost.index();

        let mut enc = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()?;

        enc.set_width(width);
        enc.set_height(height);
        enc.set_format(candidate.staging.encoder_pixel_format());
        if let Some(primaries) = color.metadata.primaries {
            enc.set_color_primaries(primaries);
        }
        if let Some(transfer) = color.metadata.transfer {
            enc.set_color_transfer_characteristic(transfer);
        }
        if let Some(space) = color.metadata.space {
            enc.set_colorspace(space);
        }
        if let Some(full_range) = color.full_range {
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

        if !candidate.is_hardware {
            enc.set_threading(ffmpeg::threading::Config::count(0));
        }

        let hardware_upload = if candidate.staging.uses_vaapi_upload() {
            let upload = HardwareUpload::new_vaapi(
                width,
                height,
                config.device.as_deref(),
                candidate.staging.staging_pixel_format(),
            )?;
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
        let mut opts = options::build_encoder_opts(
            candidate.name,
            config.quality,
            config.preset.as_deref(),
            width,
            height,
            config.bit_depth,
        );
        if candidate.family == EncoderFamily::Nvenc
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

        let scaler = (config.bit_depth == BitDepth::Eight)
            .then(|| Self::build_rgba_scaler(width, height, candidate.staging, color))
            .transpose()?;

        Ok(OpenedVideoEncoder {
            encoder,
            scaler,
            stream_index,
            encoder_time_base,
            hardware_upload,
        })
    }

    fn build_rgba_scaler(
        width: u32,
        height: u32,
        staging: StagingMode,
        color: OutputColor,
    ) -> Result<ScalingContext, EncodeError> {
        let mut scaler = ScalingContext::get(
            Pixel::RGBA,
            width,
            height,
            staging.staging_pixel_format(),
            width,
            height,
            ScalingFlags::BILINEAR,
        )?;
        if let Some(matrix) = color.rgba_matrix {
            use ffmpeg::sys as ffi;
            let colorspace = super::color::swscale_matrix(matrix);
            // SAFETY: the scaler is live and uniquely borrowed; FFmpeg owns the static coefficient table.
            let result = unsafe {
                let coefficients = ffi::sws_getCoefficients(colorspace);
                ffmpeg::sys::sws_setColorspaceDetails(
                    scaler.as_mut_ptr(),
                    coefficients,
                    1,
                    coefficients,
                    i32::from(color.full_range.unwrap_or_default()),
                    0,
                    1 << 16,
                    1 << 16,
                )
            };
            if result < 0 {
                return Err(ffmpeg::Error::from(result).into());
            }
        }
        Ok(scaler)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_muxer_setup(
        mut octx: OutputIo,
        opened: OpenedVideoEncoder,
        hevc_color: Option<HevcColor>,
        codec_id: codec::Id,
        candidate: ResolvedCandidate<'_>,
        policy: ContainerPolicy,
        path: &Path,
        width: u32,
        height: u32,
        fps: Rational,
        config: &EncoderConfig,
        color: OutputColor,
        live_audio: Option<LiveAudio>,
    ) -> Result<Self, EncodeError> {
        // VAAPI drivers can emit Main 10 parameter sets that differ from
        // FFmpeg's extradata. Keep them in-band instead of stripping them with hvc1.
        if !matches!(candidate.staging, StagingMode::VaapiSurface(BitDepth::Ten)) {
            apply_video_codec_tag(&mut octx, opened.stream_index, codec_id);
        }
        Self::log_opened_encoder(
            candidate,
            width,
            height,
            fps,
            opened.hardware_upload.is_some(),
        );

        // Set up audio passthrough before writing the header.
        let audio = AudioPassthrough::open(&mut octx, config, live_audio)?;

        octx.write_header(policy.muxer_options(color))?;

        let output_time_base = octx
            .stream(opened.stream_index)
            .expect("video stream we just added")
            .time_base();

        // Update audio output time base after write_header (the muxer may adjust it).
        let audio = audio.map(|mut a| {
            a.refresh_time_base(&octx);
            a
        });

        let (stream, stream_monitor) = if let Some(ref url) = config.stream_url {
            match Self::open_stream_output(url, &octx, opened.stream_index, config) {
                Ok(s) => {
                    log::info!("Optional stream output queued");
                    let monitor = s.octx.monitor();
                    (Some(s), Some(monitor))
                }
                Err(e) => {
                    log::warn!("Failed to open stream output ({e}), recording without stream");
                    (None, Some(StreamMonitor::failed(&e)))
                }
            }
        } else {
            (None, None)
        };

        let header_check = if codec_id == codec::Id::HEVC {
            Some(HevcHeaderCheck::new(&opened.encoder)?)
        } else {
            None
        };
        let broadcast_clock = super::encoder_timing::BroadcastClock::new(
            crate::output::is_network_destination(path),
            stream_monitor.clone(),
            config.interrupted.clone(),
        );

        Ok(Self {
            octx,
            encoder: opened.encoder,
            rgba_input: opened
                .scaler
                .map(|scaler| (scaler, VideoFrame::new(Pixel::RGBA, width, height))),
            stream_index: opened.stream_index,
            encoder_time_base: opened.encoder_time_base,
            output_time_base,
            timeline: super::encoder_timing::Timeline::new(fps),
            width,
            height,
            finished: false,
            encoder_name: candidate.name.to_string(),
            header_check,
            hevc_color,
            yuv_frame: VideoFrame::new(candidate.staging.staging_pixel_format(), width, height),
            hardware_upload: opened.hardware_upload,
            audio,
            stream,
            stream_monitor,
            broadcast_clock,
        })
    }

    fn log_opened_encoder(
        candidate: ResolvedCandidate<'_>,
        width: u32,
        height: u32,
        fps: Rational,
        uses_vaapi_upload: bool,
    ) {
        let hw_tag = if candidate.is_hardware {
            " (hardware)"
        } else {
            " (software)"
        };
        log::info!(
            "Encoder: {}x{} {}{} @ {}/{} fps",
            width,
            height,
            candidate.name,
            hw_tag,
            fps.0,
            fps.1,
        );
        // Story-tell which frame-delivery path this encoder uses, so
        // the upload cost is visible in the field.
        if uses_vaapi_upload {
            log::info!("Encoder frame path: explicit VAAPI surface upload (NV12 -> GPU surface)");
        } else if candidate.is_hardware {
            log::info!(
                "Encoder frame path: implicit driver upload (software NV12 -> GPU, handled by FFmpeg)"
            );
        } else {
            log::info!("Encoder frame path: software encode (no GPU upload)");
        }
    }

    /// The name of the active encoder (e.g., `"h264_nvenc"`, `"libx264"`).
    pub fn encoder_name(&self) -> &str {
        &self.encoder_name
    }

    /// Observe the optional stream independently of recording, including asynchronous and finalization failures.
    /// Returns `None` only when streaming was not requested.
    pub fn stream_monitor(&self) -> Option<StreamMonitor> {
        self.stream_monitor.clone()
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

    /// Open a secondary FLV output context for RTMP streaming.
    ///
    /// Shares video and selected audio packets. Without source audio, adds a silent AAC track.
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
        // Older FIFO versions resume on audio keyframes, corrupting predicted video after overflow.
        if format::version() >> 16 < 62 {
            return Err(EncodeError::InvalidConfig(
                "optional RTMP streaming requires FFmpeg 8+ for audio/video queue recovery; use the bundled runtime".into(),
            ));
        }
        let octx = OutputIo::open(Path::new(url), "flv", config, true)?;
        Self::configure_stream_output(octx, primary_octx, primary_video_index)
    }

    fn configure_stream_output(
        mut octx: OutputIo,
        primary_octx: &format::context::Output,
        primary_video_index: usize,
    ) -> Result<StreamOutput, EncodeError> {
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

        let mut audio = if let Some(source) = primary_octx
            .streams()
            .find(|s| s.parameters().medium() == ffmpeg::media::Type::Audio)
        {
            if !matches!(source.parameters().id(), codec::Id::AAC | codec::Id::MP3) {
                return Err(EncodeError::InvalidConfig(
                    "RTMP audio copy requires AAC or MP3 source audio".into(),
                ));
            }
            let mut target = octx.add_stream(ffmpeg::encoder::find(codec::Id::None))?;
            target.set_parameters(source.parameters());
            target.set_time_base(source.time_base());
            // SAFETY: the new stream owns writable codec parameters.
            unsafe {
                (*target.parameters().as_mut_ptr()).codec_tag = 0;
            }
            StreamAudio::Copy {
                index: target.index(),
                time_base: target.time_base(),
            }
        } else {
            StreamAudio::Silence(SilentAudio::new(&mut octx)?)
        };

        octx.write_header(ffmpeg::Dictionary::new())?;

        let video_time_base = octx
            .stream(video_index)
            .expect("stream we just added")
            .time_base();

        match &mut audio {
            StreamAudio::Copy { index, time_base } => {
                *time_base = octx
                    .stream(*index)
                    .expect("audio stream we just added")
                    .time_base()
            }
            StreamAudio::Silence(silent) => {
                silent.output_time_base = octx
                    .stream(silent.stream_index)
                    .expect("audio stream we just added")
                    .time_base()
            }
        }

        log::info!("Optional stream configured (video tb={video_time_base})");

        Ok(StreamOutput {
            octx,
            video_index,
            video_time_base,
            audio,
        })
    }

    /// Forward available selected audio through the encoded video time.
    fn forward_audio_packets_until(&mut self, max_pts: i64) -> Result<(), EncodeError> {
        let Some(audio) = &mut self.audio else {
            return Ok(());
        };
        while let Some(mut packet) = audio.next_until(max_pts)? {
            if let Some(ref mut stream) = self.stream
                && let StreamAudio::Copy { index, time_base } = stream.audio
            {
                let mut copy = packet.clone();
                copy.set_stream(index);
                copy.rescale_ts(audio.time_base(), time_base);
                if let Err(error) = stream.octx.write_packet(&mut copy) {
                    log::warn!("RTMP audio write failed ({error}), disabling stream");
                    self.stream = None;
                }
            }
            self.octx.write_packet(&mut packet)?;
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
        {
            reco_core::profile_scope!("encoder_submit");
            if let Some(ref mut upload) = self.hardware_upload {
                let frame = upload.upload(&self.yuv_frame)?;
                self.encoder.send_frame(frame)?;
            } else {
                self.encoder.send_frame(&self.yuv_frame)?;
            }
        }
        self.timeline.accept(pts, pts_us.is_some());
        self.receive_and_write_packets()?;
        Ok(())
    }

    pub(crate) fn input_pixel_format(&self) -> reco_core::sink::PixelFormat {
        match self.yuv_frame.format() {
            Pixel::P010LE | Pixel::YUV420P10LE => reco_core::sink::PixelFormat::P010,
            _ => reco_core::sink::PixelFormat::Nv12,
        }
    }

    fn require_8bit_input(&self) -> Result<(), EncodeError> {
        if self.rgba_input.is_none() {
            return Err(EncodeError::InvalidConfig(
                "10-bit encoders require P010LE frames".into(),
            ));
        }
        Ok(())
    }

    /// Write an RGBA frame to an 8-bit encoder.
    ///
    /// `pts_us` follows [`reco_core::sink::OutputFrame::pts_us`]. The output uses a microsecond encoder clock; the container may quantize it further. Network destinations pace encoded packets on one wall clock; ordinary file output is unrestricted. Frames are not duplicated.
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
        self.require_8bit_input()?;
        let (scaler, rgba_frame) = self
            .rgba_input
            .as_mut()
            .expect("8-bit encoder has an RGBA scaler");
        let expected = self.width as usize * self.height as usize * 4;
        if rgba_data.len() != expected {
            return Err(EncodeError::FrameSizeMismatch {
                expected,
                actual: rgba_data.len(),
            });
        }

        let stride = rgba_frame.stride(0);
        let row_bytes = self.width as usize * 4;
        if stride == row_bytes {
            rgba_frame.data_mut(0)[..rgba_data.len()].copy_from_slice(rgba_data);
        } else {
            for y in 0..self.height as usize {
                let src_start = y * row_bytes;
                let dst_start = y * stride;
                rgba_frame.data_mut(0)[dst_start..dst_start + row_bytes]
                    .copy_from_slice(&rgba_data[src_start..src_start + row_bytes]);
            }
        }

        scaler.run(rgba_frame, &mut self.yuv_frame)?;
        self.send_current_yuv_frame(pts_us)
    }

    /// Write a pre-converted NV12 frame to an 8-bit encoder.
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
        self.require_8bit_input()?;
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

    /// Write a tightly packed P010LE frame to a 10-bit encoder.
    ///
    /// The layout matches [`reco_core::sink::DeliveryFormat::P010`]. The buffer must
    /// contain exactly `width * height * 3` bytes, with `width * height * 2`
    /// bytes of Y followed by `width * height` bytes of interleaved UV. Each
    /// little-endian 16-bit word stores its 10-bit sample in bits 15 through 6.
    /// `pts_us` follows [`Self::write_frame`].
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "encode_p010_frame")
    )]
    pub fn write_p010_frame(
        &mut self,
        p010_data: &[u8],
        pts_us: Option<i64>,
    ) -> Result<(), EncodeError> {
        if self.rgba_input.is_some() {
            return Err(EncodeError::InvalidConfig(
                "P010LE frames require a 10-bit encoder".into(),
            ));
        }
        let expected = self.width as usize * self.height as usize * 3;
        if p010_data.len() != expected {
            return Err(EncodeError::FrameSizeMismatch {
                expected,
                actual: p010_data.len(),
            });
        }

        // SAFETY: the staging AVFrame is live. The encoder may still hold a
        // reference after a previous submission, so ensure the planes are writable.
        let status = unsafe { ffmpeg::sys::av_frame_make_writable(self.yuv_frame.as_mut_ptr()) };
        if status < 0 {
            return Err(ffmpeg::Error::from(status).into());
        }
        p010::stage(p010_data, &mut self.yuv_frame);
        self.send_current_yuv_frame(pts_us)
    }

    /// Write a pre-converted YUV420P planar frame to an 8-bit encoder.
    ///
    /// `pts_us` follows [`Self::write_frame`].
    ///
    /// Plane slices must be tightly packed (no padding between
    /// rows): Y is `width * height`, U and V are each
    /// `(width / 2) * (height / 2)`. Odd dimensions are rejected
    /// because 4:2:0 chroma subsampling requires even width/height.
    ///
    /// The stacked-video replay encoder produces YUV420P directly. Feeding RGBA through [`Self::write_frame`] would add a YUV→RGBA→YUV roundtrip; this path avoids that conversion and its rounding loss.
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
        self.require_8bit_input()?;
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
            let audio_tb = self.audio.as_ref().unwrap().time_base();
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
            if let StreamAudio::Silence(ref mut audio) = stream.audio
                && let Err(error) = audio.finish(&mut stream.octx, self.timeline.end_us())
            {
                log::warn!("Silent audio finalization failed: {error}");
                stream.octx.record_failure(&error);
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
        if let Some(clock) = &mut self.broadcast_clock
            && let Some(time) = packet.dts().or_else(|| packet.pts())
            && !clock.wait(time.rescale(self.encoder_time_base, super::encoder_timing::TIME_BASE))
        {
            self.broadcast_clock = None;
        }
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

        // Interleave source audio as encoded video advances, not only at finalization.
        // DTS is ordered even when the video encoder reorders presentation times.
        if let Some(audio) = &self.audio
            && let Some(time) = packet.dts().or_else(|| packet.pts())
        {
            let limit = time.rescale(self.output_time_base, audio.time_base());
            self.forward_audio_packets_until(limit)?;
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
            } else if let StreamAudio::Silence(ref mut sa) = stream.audio
                && let Err(e) = sa.write_for_video(
                    &mut stream.octx,
                    video_pts,
                    video_duration,
                    stream.video_time_base,
                )
            {
                log::warn!("Stream audio write failed ({e}), disabling stream");
                stream.octx.record_failure(&e);
                self.stream = None;
            }
        }

        self.octx.write_packet(packet)?;
        Ok(())
    }
}

fn receive_encoded_packet(
    encoder: &mut ffmpeg::encoder::Encoder,
    packet: &mut ffmpeg::Packet,
) -> Result<bool, ffmpeg::Error> {
    reco_core::profile_scope!("encoder_receive");
    match encoder.receive_packet(packet) {
        Ok(()) => Ok(true),
        Err(ffmpeg::Error::Eof) => Ok(false),
        Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::container::{HVC1_CODEC_TAG, video_codec_tag};
    use super::options::scale_bitrate_mbps;
    use super::*;
    use crate::output::QualityPreset;

    mod audio_copy;
    mod spec_matrix;

    fn build_encoder_opts(
        name: &str,
        quality: Quality,
        preset: Option<&str>,
        width: u32,
        height: u32,
    ) -> ffmpeg::Dictionary<'static> {
        options::build_encoder_opts(name, quality, preset, width, height, BitDepth::Eight)
    }

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
                    audio: StreamAudio::Silence(audio),
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

    fn parse_mbps(value: &str) -> u32 {
        value
            .strip_suffix('M')
            .expect("bitrate ceiling should use 'M' suffix")
            .parse()
            .expect("bitrate ceiling should be an integer Mbps")
    }

    #[test]
    fn videotoolbox_presets_scale_maxrate_with_resolution() {
        let cases = [
            ("h264_videotoolbox", "high", QualityPreset::Fast, "50", 12),
            (
                "h264_videotoolbox",
                "high",
                QualityPreset::Balanced,
                "60",
                18,
            ),
            ("h264_videotoolbox", "high", QualityPreset::High, "70", 30),
            ("hevc_videotoolbox", "main", QualityPreset::Fast, "50", 10),
            (
                "hevc_videotoolbox",
                "main",
                QualityPreset::Balanced,
                "60",
                14,
            ),
            ("hevc_videotoolbox", "main", QualityPreset::High, "70", 22),
        ];

        for (encoder, profile, preset, expected_gq, base_mbps) in cases {
            let opts_1080p = build_encoder_opts(encoder, Quality::Preset(preset), None, 1920, 1080);
            assert_eq!(opts_1080p.get("global_quality"), Some(expected_gq));
            assert_eq!(
                opts_1080p.get("maxrate").map(parse_mbps),
                Some(base_mbps),
                "{encoder} {preset:?} 1080p maxrate"
            );
            assert_eq!(opts_1080p.get("profile"), Some(profile));

            let opts_4k = build_encoder_opts(encoder, Quality::Preset(preset), None, 3840, 2160);
            assert_eq!(opts_4k.get("global_quality"), Some(expected_gq));
            assert_eq!(
                opts_4k.get("maxrate").map(parse_mbps),
                Some(base_mbps * 4),
                "{encoder} {preset:?} 4K maxrate should scale 4x from 1080p"
            );
            assert_eq!(opts_4k.get("profile"), Some(profile));

            let opts_720p = build_encoder_opts(encoder, Quality::Preset(preset), None, 1280, 720);
            assert_eq!(
                opts_720p.get("maxrate").map(parse_mbps),
                Some(scale_bitrate_mbps(base_mbps, 1280, 720)),
                "{encoder} {preset:?} 720p maxrate"
            );
        }
    }

    #[test]
    fn videotoolbox_numeric_quality_scales_with_resolution_and_progresses_above_75() {
        for (encoder, expected_profile) in
            [("h264_videotoolbox", "high"), ("hevc_videotoolbox", "main")]
        {
            let mut prev_gq = 0u32;
            let mut prev_1080p = 0u32;
            let mut prev_4k = 0u32;

            for q in 0..=100u8 {
                let opts_1080p = build_encoder_opts(encoder, Quality::Value(q), None, 1920, 1080);
                let opts_4k = build_encoder_opts(encoder, Quality::Value(q), None, 3840, 2160);

                let gq_1080p: u32 = opts_1080p
                    .get("global_quality")
                    .expect("global_quality")
                    .parse()
                    .unwrap();
                let gq_4k: u32 = opts_4k
                    .get("global_quality")
                    .expect("global_quality")
                    .parse()
                    .unwrap();
                let maxrate_1080p = parse_mbps(opts_1080p.get("maxrate").expect("maxrate"));
                let maxrate_4k = parse_mbps(opts_4k.get("maxrate").expect("maxrate"));

                assert_eq!(gq_1080p, gq_4k, "{encoder} q={q}: quality is res-invariant");
                assert!(
                    (1..=90).contains(&gq_1080p),
                    "{encoder} q={q}: global_quality {gq_1080p} out of expected VT range"
                );
                assert_eq!(
                    maxrate_4k,
                    maxrate_1080p * 4,
                    "{encoder} q={q}: 4K maxrate should be 4x 1080p"
                );
                assert_eq!(opts_1080p.get("profile"), Some(expected_profile));
                assert_eq!(opts_4k.get("profile"), Some(expected_profile));

                assert!(
                    gq_1080p >= prev_gq,
                    "{encoder} q={q}: non-monotonic quality"
                );
                assert!(
                    maxrate_1080p >= prev_1080p,
                    "{encoder} q={q}: non-monotonic 1080p maxrate"
                );
                assert!(
                    maxrate_4k >= prev_4k,
                    "{encoder} q={q}: non-monotonic 4K maxrate"
                );

                prev_gq = gq_1080p;
                prev_1080p = maxrate_1080p;
                prev_4k = maxrate_4k;
            }

            // Representative checkpoints across the full range, specifically verifying
            // meaningful upward progression above 75 rather than collapsing at High.
            let samples: Vec<(u32, u32, u32)> = [0u8, 25, 50, 75, 85, 95, 100]
                .into_iter()
                .map(|q| {
                    let o1080 = build_encoder_opts(encoder, Quality::Value(q), None, 1920, 1080);
                    let o4k = build_encoder_opts(encoder, Quality::Value(q), None, 3840, 2160);
                    (
                        o1080.get("global_quality").unwrap().parse().unwrap(),
                        parse_mbps(o1080.get("maxrate").unwrap()),
                        parse_mbps(o4k.get("maxrate").unwrap()),
                    )
                })
                .collect();

            for pair in samples.windows(2) {
                assert!(
                    pair[1].0 > pair[0].0,
                    "{encoder}: expected strictly higher global_quality: {pair:?}"
                );
                assert!(
                    pair[1].1 > pair[0].1,
                    "{encoder}: expected strictly higher 1080p maxrate: {pair:?}"
                );
                assert!(
                    pair[1].2 > pair[0].2,
                    "{encoder}: expected strictly higher 4K maxrate: {pair:?}"
                );
            }
        }

        // HEVC keeps lower bitrate ceilings than H.264 across the numeric quality scale.
        for q in [0u8, 25, 50, 75, 85, 100] {
            let h264 = build_encoder_opts("h264_videotoolbox", Quality::Value(q), None, 3840, 2160);
            let hevc = build_encoder_opts("hevc_videotoolbox", Quality::Value(q), None, 3840, 2160);
            assert!(
                parse_mbps(hevc.get("maxrate").unwrap()) < parse_mbps(h264.get("maxrate").unwrap()),
                "q={q}: HEVC 4K maxrate should stay below H.264"
            );
        }

        // Mid-range and High anchors match the preset tiers cleanly.
        let hevc_balanced =
            build_encoder_opts("hevc_videotoolbox", Quality::Value(50), None, 3840, 2160);
        assert_eq!(hevc_balanced.get("global_quality"), Some("60"));
        assert_eq!(hevc_balanced.get("maxrate"), Some("56M"));

        let hevc_high =
            build_encoder_opts("hevc_videotoolbox", Quality::Value(75), None, 3840, 2160);
        assert_eq!(hevc_high.get("global_quality"), Some("75"));
        assert_eq!(hevc_high.get("maxrate"), Some("88M"));

        let hevc_85 = build_encoder_opts("hevc_videotoolbox", Quality::Value(85), None, 3840, 2160);
        assert_eq!(hevc_85.get("global_quality"), Some("81"));
        assert_eq!(hevc_85.get("maxrate"), Some("124M"));
    }

    #[test]
    fn hevc_mp4_mov_selects_hvc1_codec_tag() {
        assert_eq!(HVC1_CODEC_TAG, u32::from_le_bytes(*b"hvc1"));
        assert_eq!(
            video_codec_tag(codec::Id::HEVC, "mp4", true),
            Some(HVC1_CODEC_TAG)
        );
        assert_eq!(
            video_codec_tag(codec::Id::HEVC, "mov", true),
            Some(HVC1_CODEC_TAG)
        );
        // hvc1 requires global headers (VPS/SPS/PPS in the hvcC sample entry).
        assert_eq!(video_codec_tag(codec::Id::HEVC, "mp4", false), None);
        assert_eq!(video_codec_tag(codec::Id::HEVC, "mov", false), None);
        // Non-ISOBMFF containers must keep their muxer default.
        assert_eq!(video_codec_tag(codec::Id::HEVC, "matroska", true), None);
        assert_eq!(video_codec_tag(codec::Id::HEVC, "flv", true), None);
        assert_eq!(video_codec_tag(codec::Id::HEVC, "mpegts", false), None);
        // Other codecs keep their muxer default.
        assert_eq!(video_codec_tag(codec::Id::H264, "mp4", true), None);
        assert_eq!(video_codec_tag(codec::Id::H264, "mov", true), None);
        assert_eq!(video_codec_tag(codec::Id::AV1, "mp4", true), None);
    }

    #[test]
    fn hevc_muxed_mp4_and_mov_write_hvc1_codec_tag() {
        crate::init();
        if encoder::find_by_name("libx265").is_none() {
            return;
        }
        let cases = [
            (Container::Mp4, "out.mp4", Some(HVC1_CODEC_TAG)),
            (Container::Mp4, "out", Some(HVC1_CODEC_TAG)),
            (Container::Mp4Fragmented, "frag.mp4", Some(HVC1_CODEC_TAG)),
            (Container::Mov, "out.mov", Some(HVC1_CODEC_TAG)),
            // Container muxer is authoritative even when the filename ends in .mp4.
            (Container::Mkv, "out.mkv", None),
            (Container::Mkv, "misleading.mp4", None),
        ];
        for rendered in [false, true] {
            for (container, filename, expected_tag) in cases {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join(filename);
                let config = EncoderConfig {
                    encoder_name: Some("libx265".into()),
                    codec: Some(Codec::HEVC),
                    preset: Some("ultrafast".into()),
                    container,
                    ..Default::default()
                };
                let mut enc = if rendered {
                    VideoEncoder::new_rendered(
                        &path,
                        64,
                        64,
                        Rational(30, 1),
                        &config,
                        None,
                        Default::default(),
                    )
                    .unwrap()
                } else {
                    VideoEncoder::new(&path, 64, 64, Rational(30, 1), &config).unwrap()
                };
                for _ in 0..3 {
                    enc.write_nv12_frame(&[128; 64 * 64 * 3 / 2], None).unwrap();
                }
                enc.finish().unwrap();
                drop(enc);

                let input = format::input(&path).unwrap();
                let stream = input
                    .streams()
                    .best(ffmpeg::media::Type::Video)
                    .expect("muxed video stream");
                let params = stream.parameters();
                assert_eq!(params.id(), codec::Id::HEVC);
                // SAFETY: `params` is live for the duration of this inspection.
                let (codec_tag, extradata_size) = unsafe {
                    let raw = &*params.as_ptr();
                    (raw.codec_tag, raw.extradata_size)
                };
                match expected_tag {
                    Some(tag) => {
                        assert_eq!(
                            codec_tag,
                            tag,
                            "expected hvc1 for {container:?} ({filename}, rendered={rendered}), got {:?}",
                            std::str::from_utf8(&codec_tag.to_le_bytes())
                        );
                        assert!(
                            extradata_size > 0,
                            "expected global header extradata (hvcC) for {container:?} ({filename}, rendered={rendered})"
                        );
                    }
                    None => {
                        assert_ne!(
                            codec_tag, HVC1_CODEC_TAG,
                            "must not write hvc1 for {container:?} ({filename}, rendered={rendered})"
                        );
                    }
                }
            }
        }
    }
}
