//! Video encoder: RGBA frames → MP4 file (H.264, HEVC, or AV1).
//!
//! Wraps FFmpeg's muxer, encoder, and swscale to write RGBA pixel data
//! to an encoded MP4 file. Supports hardware-accelerated encoding
//! (NVENC, QSV, VideoToolbox, VAAPI) with automatic fallback to software.

extern crate ffmpeg_next as ffmpeg;

use ffmpeg::format::Pixel;
use ffmpeg::software::scaling::{context::Context as ScalingContext, flag::Flags as ScalingFlags};
use ffmpeg::util::frame::video::Video as VideoFrame;
use ffmpeg::{Rational, codec, encoder, format};
use std::path::Path;
use thiserror::Error;

/// Errors from the video encoder.
#[derive(Debug, Error)]
pub enum EncodeError {
    /// FFmpeg error.
    #[error("FFmpeg: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// No encoder available for the requested codec.
    #[error("no encoder found for codec '{0}' — is FFmpeg built with the right encoder?")]
    CodecNotFound(String),

    /// Frame data has wrong size.
    #[error("frame data size mismatch: expected {expected} bytes, got {actual}")]
    FrameSizeMismatch { expected: usize, actual: usize },
}

/// Output video codec.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264 / AVC — widest compatibility.
    #[default]
    H264,
    /// H.265 / HEVC — better quality per bit, good hardware support.
    Hevc,
    /// AV1 — best compression, requires modern hardware for encoding.
    Av1,
}

impl VideoCodec {
    /// Parse from a string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "h264" | "avc" | "h.264" => Some(Self::H264),
            "hevc" | "h265" | "h.265" => Some(Self::Hevc),
            "av1" => Some(Self::Av1),
            _ => None,
        }
    }

    fn candidates(self) -> &'static [EncoderCandidate] {
        match self {
            Self::H264 => H264_ENCODERS,
            Self::Hevc => HEVC_ENCODERS,
            Self::Av1 => AV1_ENCODERS,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Av1 => "AV1",
        }
    }
}

struct EncoderCandidate {
    name: &'static str,
    is_hardware: bool,
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
        pixel_format: Pixel::NV12,
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
        pixel_format: Pixel::NV12,
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
        pixel_format: Pixel::NV12,
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

/// Detect which encoders are available for a given codec.
///
/// Returns encoders in preference order (hardware first, then software).
pub fn available_encoders(codec: VideoCodec) -> Vec<EncoderInfo> {
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
    available_encoders(VideoCodec::H264)
}

/// Encoder quality preset.
#[derive(Debug, Clone, Copy, Default)]
pub enum Quality {
    /// Prioritize speed over quality (for previewing / testing).
    Fast,
    /// Balanced speed and quality.
    #[default]
    Balanced,
    /// Prioritize quality (for final output).
    High,
}

/// Output container format.
///
/// The default [`Container::Mp4Fragmented`] is the right choice for
/// write-while-read workflows (replay backends, live uploads) because
/// fragmented MP4 writes a minimal `moov` atom up front and flushes
/// self-contained fragments on keyframes, so a concurrent reader can
/// parse the file mid-write. Plain [`Container::Mp4`] writes the
/// `moov` at close, so partial files are unreadable and concurrent
/// readers see only the header. [`Container::Matroska`] is the
/// crash-safe alternative with similar streaming properties.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Container {
    /// Plain MP4 (`.mp4`). Final-file-only; index written at close.
    /// Default for the stitch output path so the existing behavior
    /// is preserved; opt in to a streamable container explicitly
    /// when needed.
    #[default]
    Mp4,
    /// Fragmented MP4 (`.mp4`) with `empty_moov` + `frag_keyframe`
    /// movflags. Readable while still being written. Default for
    /// the stacked-video replay encoder.
    Mp4Fragmented,
    /// Matroska (`.mkv`). Naturally streamable; recommended by OBS
    /// for crash-safe recording.
    Matroska,
}

impl Container {
    /// FFmpeg muxer name for this container.
    fn muxer_name(self) -> &'static str {
        match self {
            Self::Mp4 | Self::Mp4Fragmented => "mp4",
            Self::Matroska => "matroska",
        }
    }

    /// Parse from a string (case-insensitive). Accepts `"mp4"`,
    /// `"fmp4"`/`"mp4-fragmented"`, and `"mkv"`/`"matroska"`.
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "mp4" => Some(Self::Mp4),
            "fmp4" | "mp4-fragmented" | "mp4_fragmented" => Some(Self::Mp4Fragmented),
            "mkv" | "matroska" => Some(Self::Matroska),
            _ => None,
        }
    }
}

/// Configuration for the video encoder.
#[derive(Debug, Clone, Default)]
pub struct EncoderConfig {
    /// Force a specific encoder by name, or `None` for auto-detection.
    pub encoder_name: Option<String>,
    /// Output video codec (H.264, HEVC, AV1). Default: H.264.
    pub codec: VideoCodec,
    /// Quality preset.
    pub quality: Quality,
    /// Override the CRF/quality value (passed through to the encoder).
    pub crf: Option<u8>,
    /// Override the encoder preset string (passed through to the encoder).
    pub preset: Option<String>,
    /// Path to a source file to copy audio from (stream copy, no re-encoding).
    /// The first audio stream found will be muxed into the output.
    pub audio_source: Option<std::path::PathBuf>,
    /// Output container format. Defaults to plain MP4 to match the
    /// existing stitch-output behavior; opt in to fragmented MP4 or
    /// Matroska for streamable / write-while-read workflows (e.g.,
    /// the M6.5 stacked-video replay backend).
    pub container: Container,
    /// Override the encoder's group-of-pictures size (frames
    /// between keyframes). `None` leaves ffmpeg / libx264 defaults
    /// (typically 250 frames). Set a small value (e.g. 30) when
    /// the output needs frequent keyframes for seekable replay or
    /// fragmented-MP4 fragment flush cadence.
    pub gop_size: Option<u32>,
}

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
/// // enc.write_frame(&rgba_data).unwrap();
/// enc.finish().unwrap();
/// ```
pub struct VideoEncoder {
    octx: format::context::Output,
    encoder: ffmpeg::encoder::video::Encoder,
    scaler: ScalingContext,
    stream_index: usize,
    encoder_time_base: Rational,
    output_time_base: Rational,
    frame_count: i64,
    width: u32,
    height: u32,
    finished: bool,
    encoder_name: String,
    /// Reusable frame buffers to avoid per-frame allocation.
    rgba_frame: VideoFrame,
    yuv_frame: VideoFrame,
    /// Audio passthrough state (if enabled).
    audio: Option<AudioPassthrough>,
}

/// State for copying an audio stream from an input file to the output.
struct AudioPassthrough {
    ictx: format::context::Input,
    /// Audio stream index in the input file.
    input_stream_index: usize,
    /// Audio stream index in the output file.
    output_stream_index: usize,
    /// Input audio stream time base (for rescaling).
    input_time_base: Rational,
    /// Output audio stream time base.
    output_time_base: Rational,
    /// Whether all audio packets have been read.
    exhausted: bool,
}

// SAFETY: VideoEncoder is only used from a single thread. The raw pointers
// inside FFmpeg's SwsContext/Encoder are not shared across threads.
unsafe impl Send for VideoEncoder {}

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
    /// Create a new MP4 encoder.
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
        crate::init();

        let all_candidates = config.codec.candidates();

        // Build candidate list
        let candidates: Vec<(&str, bool, Pixel)> = if let Some(ref name) = config.encoder_name {
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
                .map(|c| (c.name, c.is_hardware, c.pixel_format))
                .collect()
        };

        if candidates.is_empty() {
            return Err(EncodeError::CodecNotFound(config.codec.label().to_string()));
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

            // Fresh output context for each attempt. Use
            // `output_as` for non-default containers (fragmented
            // MP4 via movflags still goes through the mp4 muxer, so
            // the default path works via extension lookup; Matroska
            // needs the explicit muxer name when the extension
            // doesn't match).
            // `format::output(path)` infers muxer from extension.
            // `output_as(path, name)` forces by name. We use the
            // latter for Matroska so callers can point at `.mp4`
            // or any extension and still get MKV - consumers'
            // opt-in container choice wins over filename.
            let mut octx = match config.container {
                Container::Mp4 | Container::Mp4Fragmented => format::output(path)?,
                Container::Matroska => format::output_as(path, config.container.muxer_name())?,
            };

            match Self::try_open(
                &mut octx, codec, *pixel_fmt, *is_hw, width, height, fps, config, name,
            ) {
                Ok((enc_opened, scaler, stream_index, encoder_time_base, _)) => {
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

                    // Set up audio passthrough before writing the header.
                    let audio = if let Some(ref audio_path) = config.audio_source {
                        Self::setup_audio_stream(&mut octx, audio_path)?
                    } else {
                        None
                    };

                    // Fragmented MP4 needs `movflags` so the muxer
                    // writes an `empty_moov` up front and flushes
                    // self-contained fragments on every keyframe.
                    // A concurrent reader can then parse the file
                    // mid-write; plain MP4 would park the `moov`
                    // until `write_trailer` and break replay.
                    if config.container == Container::Mp4Fragmented {
                        let mut opts = ffmpeg::Dictionary::new();
                        opts.set("movflags", "empty_moov+frag_keyframe");
                        let _ = octx.write_header_with(opts)?;
                    } else {
                        octx.write_header()?;
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

                    return Ok(Self {
                        octx,
                        encoder: enc_opened,
                        scaler,
                        stream_index,
                        encoder_time_base,
                        output_time_base,
                        frame_count: 0,
                        width,
                        height,
                        finished: false,
                        encoder_name: name.to_string(),
                        rgba_frame: VideoFrame::new(Pixel::RGBA, width, height),
                        yuv_frame: VideoFrame::new(*pixel_fmt, width, height),
                        audio,
                    });
                }
                Err(e) => {
                    log::warn!("Encoder {name} failed to open: {e}, trying next...");
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or(EncodeError::CodecNotFound(config.codec.label().to_string())))
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
    ) -> Result<
        (
            ffmpeg::encoder::video::Encoder,
            ScalingContext,
            usize,
            Rational,
            Rational,
        ),
        EncodeError,
    > {
        let needs_global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

        let mut ost = octx.add_stream(codec)?;
        let stream_index = ost.index();

        let mut enc = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()?;

        enc.set_width(width);
        enc.set_height(height);
        enc.set_format(pixel_fmt);
        enc.set_frame_rate(Some(fps));
        let encoder_time_base = Rational(fps.1, fps.0);
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

        // Seed the output stream with the encoder's unopened
        // parameters BEFORE `open_with` so the muxer has valid
        // codec parameters when it allocates its internal fragment
        // state (matters for the fMP4 muxer with `empty_moov`,
        // which writes the moov before any packet lands). The
        // canonical ffmpeg-next transcode example follows the same
        // pattern: set params, open, re-set params.
        ost.set_parameters(&enc);
        let opts = build_encoder_opts(name, config.quality, config.crf, config.preset.as_deref());
        let encoder = enc.open_with(opts)?;
        ost.set_parameters(&encoder);

        // Note: write_header is called by the caller (new()) after optional
        // audio stream setup. output_time_base is read after write_header.

        let output_time_base = Rational(0, 0); // placeholder, updated after write_header

        let scaler = ScalingContext::get(
            Pixel::RGBA,
            width,
            height,
            pixel_fmt,
            width,
            height,
            ScalingFlags::BILINEAR,
        )?;

        Ok((
            encoder,
            scaler,
            stream_index,
            encoder_time_base,
            output_time_base,
        ))
    }

    /// The name of the active encoder (e.g., `"h264_nvenc"`, `"libx264"`).
    pub fn encoder_name(&self) -> &str {
        &self.encoder_name
    }

    /// Set up audio passthrough by adding an audio stream to the output context.
    ///
    /// Called before `write_header`. Returns `None` if no audio stream found.
    fn setup_audio_stream(
        octx: &mut format::context::Output,
        source_path: &Path,
    ) -> Result<Option<AudioPassthrough>, EncodeError> {
        let ictx = format::input(source_path)?;

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
            exhausted: false,
        }))
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

        loop {
            let mut packet = ffmpeg::Packet::empty();
            match packet.read(&mut audio.ictx) {
                Ok(()) => {
                    if packet.stream() != audio.input_stream_index {
                        continue; // skip video/subtitle packets
                    }
                    packet.set_stream(audio.output_stream_index);
                    packet.rescale_ts(audio.input_time_base, audio.output_time_base);

                    // Stop if this audio packet is beyond the video duration.
                    if packet.pts().is_some_and(|pts| pts > max_pts) {
                        audio.exhausted = true;
                        break;
                    }

                    packet.write_interleaved(&mut self.octx)?;
                }
                Err(ffmpeg::Error::Eof) => {
                    audio.exhausted = true;
                    break;
                }
                Err(_) => break,
            }
        }
        Ok(())
    }

    /// Write an RGBA frame to the output file.
    ///
    /// `rgba_data` must be exactly `width * height * 4` bytes (tightly packed).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "encode_frame")
    )]
    pub fn write_frame(&mut self, rgba_data: &[u8]) -> Result<(), EncodeError> {
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
        self.yuv_frame.set_pts(Some(self.frame_count));

        self.encoder.send_frame(&self.yuv_frame)?;
        self.receive_and_write_packets()?;

        self.frame_count += 1;
        Ok(())
    }

    /// Write a pre-converted NV12 frame to the output file.
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
    pub fn write_nv12_frame(&mut self, nv12_data: &[u8]) -> Result<(), EncodeError> {
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

        self.yuv_frame.set_pts(Some(self.frame_count));
        self.encoder.send_frame(&self.yuv_frame)?;
        self.receive_and_write_packets()?;

        self.frame_count += 1;
        Ok(())
    }

    /// Write a pre-converted YUV420P planar frame.
    ///
    /// Plane slices must be tightly packed (no padding between
    /// rows): Y is `width * height`, U and V are each
    /// `(width / 2) * (height / 2)`. Odd dimensions are rejected
    /// because 4:2:0 chroma subsampling requires even width/height.
    ///
    /// This path exists for the stacked-video replay encoder
    /// ([`crate::stacked_video::encoder::StackedEncoder`]), which
    /// produces YUV420P natively from its pack primitive. Feeding
    /// RGBA through [`Self::write_frame`] would trigger a
    /// YUV→RGBA→YUV roundtrip for every replay frame; skipping the
    /// scaler saves ~1-2ms per frame at 1080p and avoids colorspace
    /// drift from repeated range conversion.
    ///
    /// Requires the encoder's internal pixel format to be
    /// YUV420P. Currently all software encoders
    /// (`libx264`/`libx265`/`libsvtav1`/`libaom-av1`) use YUV420P;
    /// hardware encoders use NV12 and are not supported on this
    /// path (they should go through [`Self::write_nv12_frame`]
    /// instead).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "encode_yuv420p_frame")
    )]
    pub fn write_yuv420p_planes(
        &mut self,
        y: &[u8],
        u: &[u8],
        v: &[u8],
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

        if self.yuv_frame.format() != Pixel::YUV420P {
            // Hardware encoders take NV12; plane-packed YUV420P is
            // for software encoders only. Callers hitting this path
            // have either misconfigured the encoder or need
            // `write_nv12_frame`.
            return Err(EncodeError::CodecNotFound(format!(
                "write_yuv420p_planes requires YUV420P encoder, got {:?}",
                self.yuv_frame.format(),
            )));
        }

        // Row-by-row copy honoring ffmpeg's plane strides (which may
        // exceed the logical row width due to SIMD alignment).
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

        self.yuv_frame.set_pts(Some(self.frame_count));
        self.encoder.send_frame(&self.yuv_frame)?;
        self.receive_and_write_packets()?;

        self.frame_count += 1;
        Ok(())
    }

    /// Flush muxer + AVIO buffers to disk without finalizing the
    /// container.
    ///
    /// Forces any fragments or packets currently buffered in ffmpeg
    /// (either the muxer's internal queue or the AVIO output layer)
    /// out to the file descriptor. A subsequent [`Self::finish`] is
    /// still required to write the final trailer.
    ///
    /// Needed for write-while-read workflows on fragmented MP4 /
    /// Matroska where a concurrent reader only sees bytes once
    /// they've actually hit disk. Call periodically (e.g. every
    /// keyframe) from the stacked-video replay path.
    ///
    /// `av_write_frame(ctx, NULL)` prompts the muxer to emit any
    /// queued packets; `avio_flush` then forces the AVIO layer to
    /// write its buffer to the OS. Both are safe to call multiple
    /// times and at any point after `write_header`.
    pub fn flush_to_disk(&mut self) -> Result<(), EncodeError> {
        // SAFETY: `octx` is a live output context (created in
        // `new`, never dropped until `Drop` runs). `avio_flush` is
        // safe on any live AVIO and doesn't alter muxer state —
        // just forces the output-layer buffer to the file
        // descriptor. We intentionally avoid
        // `av_write_frame(ctx, NULL)` because fMP4's
        // `frag_keyframe` mode treats that as "close current
        // fragment" which clashes with the subsequent
        // `write_trailer` on finish (observed as AVERROR -105).
        unsafe {
            let pb = (*self.octx.as_mut_ptr()).pb;
            if !pb.is_null() {
                ffmpeg::sys::avio_flush(pb);
            }
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
        log::info!("Encoder finished: {} frames written", self.frame_count);
        Ok(())
    }

    fn flush_and_finalize(&mut self) -> Result<(), EncodeError> {
        self.encoder.send_eof()?;
        self.receive_and_write_packets()?;

        // Forward audio packets trimmed to the video duration.
        if self.audio.is_some() {
            // Compute video duration in the audio output time base.
            // frame_count / (fps_num / fps_den) = duration in seconds.
            // Convert to audio PTS: duration_secs * audio_time_base_den / audio_time_base_num.
            let audio_tb = self.audio.as_ref().unwrap().output_time_base;
            let video_duration_pts = self.frame_count
                * i64::from(self.encoder_time_base.numerator())
                * i64::from(audio_tb.denominator())
                / (i64::from(self.encoder_time_base.denominator())
                    * i64::from(audio_tb.numerator()).max(1));
            self.forward_audio_packets_until(video_duration_pts)?;
        }

        self.octx.write_trailer()?;
        Ok(())
    }

    fn receive_and_write_packets(&mut self) -> Result<(), EncodeError> {
        let mut packet = ffmpeg::Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(self.stream_index);
            packet.rescale_ts(self.encoder_time_base, self.output_time_base);
            // The fMP4 muxer refuses to finalize a fragment whose
            // last packet has no duration (raises a warning then
            // fails `write_trailer` with AVERROR(EINVAL)). libx264
            // doesn't always populate `duration` on output packets,
            // so fill in the one-frame default (1 unit in encoder
            // time base, rescaled to output time base) when it's
            // missing. Safe for the non-fragmented muxers too,
            // which happily accept explicit durations.
            if packet.duration() <= 0 {
                // SAFETY: `av_rescale_q` is a pure arithmetic helper
                // (a / b * c rounded). No FFI state, no pointers,
                // no lifetime concerns.
                let one_frame = unsafe {
                    ffmpeg::sys::av_rescale_q(
                        1,
                        self.encoder_time_base.into(),
                        self.output_time_base.into(),
                    )
                };
                packet.set_duration(one_frame.max(1));
            }
            packet.write_interleaved(&mut self.octx)?;
        }
        // Audio packets are forwarded during flush (after all video is written)
        // to avoid complex per-frame PTS synchronization. write_interleaved
        // handles the interleaving.
        Ok(())
    }
}

/// Build encoder-specific FFmpeg options.
fn build_encoder_opts(
    name: &str,
    quality: Quality,
    crf_override: Option<u8>,
    preset_override: Option<&str>,
) -> ffmpeg::Dictionary<'static> {
    let mut opts = ffmpeg::Dictionary::new();

    match name {
        "h264_nvenc" => {
            // NVENC VBR with per-quality bitrate ceiling. Prior defaults
            // (10M / 15M across all quality presets) artificially capped
            // High output at ~15 Mbps even with cq=19. Scale the ceilings
            // with quality so "high" actually delivers the visual bump.
            let (preset, cq, bv, maxrate) = match quality {
                Quality::Fast => ("p3", "28", "8M", "12M"),
                Quality::Balanced => ("p4", "23", "12M", "18M"),
                Quality::High => ("p5", "19", "20M", "30M"),
            };
            opts.set("preset", preset);
            opts.set("tune", "hq");
            opts.set("rc", "vbr");
            opts.set("cq", cq);
            opts.set("b:v", bv);
            opts.set("maxrate", maxrate);
            opts.set("profile", "high");
            opts.set("spatial-aq", "1");
            opts.set("temporal-aq", "1");
        }
        "hevc_nvenc" => {
            // HEVC ~30% more efficient than H264, so ceilings scale down.
            let (preset, cq, bv, maxrate) = match quality {
                Quality::Fast => ("p3", "28", "6M", "10M"),
                Quality::Balanced => ("p4", "23", "9M", "14M"),
                Quality::High => ("p5", "19", "15M", "22M"),
            };
            opts.set("preset", preset);
            opts.set("tune", "hq");
            opts.set("rc", "vbr");
            opts.set("cq", cq);
            opts.set("b:v", bv);
            opts.set("maxrate", maxrate);
            opts.set("profile", "main");
            opts.set("spatial-aq", "1");
            opts.set("temporal-aq", "1");
        }
        "av1_nvenc" => {
            // AV1 another ~20% tighter than HEVC.
            let (preset, cq, bv, maxrate) = match quality {
                Quality::Fast => ("p3", "32", "5M", "8M"),
                Quality::Balanced => ("p4", "27", "7M", "11M"),
                Quality::High => ("p5", "22", "12M", "18M"),
            };
            opts.set("preset", preset);
            opts.set("tune", "hq");
            opts.set("rc", "vbr");
            opts.set("cq", cq);
            opts.set("b:v", bv);
            opts.set("maxrate", maxrate);
        }
        "h264_qsv" => {
            let gq = match quality {
                Quality::Fast => "28",
                Quality::Balanced => "23",
                Quality::High => "19",
            };
            opts.set("preset", "medium");
            opts.set("global_quality", gq);
            opts.set("profile", "high");
        }
        "h264_videotoolbox" => {
            // global_quality maps to VTCompressionPropertyKey_Quality (0-100).
            // "q:v" doesn't work through the C API dictionary (colon is CLI syntax).
            let q = match quality {
                Quality::Fast => "55",
                Quality::Balanced => "65",
                Quality::High => "80",
            };
            opts.set("global_quality", q);
            opts.set("profile", "high");
        }
        "hevc_videotoolbox" => {
            let q = match quality {
                Quality::Fast => "55",
                Quality::Balanced => "65",
                Quality::High => "80",
            };
            opts.set("global_quality", q);
            opts.set("profile", "main");
        }
        "h264_vaapi" | "hevc_vaapi" | "av1_vaapi" => {
            let qp = match quality {
                Quality::Fast => "28",
                Quality::Balanced => "23",
                Quality::High => "19",
            };
            opts.set("qp", qp);
            if name == "h264_vaapi" {
                opts.set("profile", "high");
            }
        }
        "h264_v4l2m2m" | "hevc_v4l2m2m" => {
            let qp = match quality {
                Quality::Fast => "28",
                Quality::Balanced => "23",
                Quality::High => "19",
            };
            opts.set("qp", qp);
            // Don't set profile - v4l2m2m drivers pick appropriate defaults.
        }
        "h264_amf" | "hevc_amf" | "av1_amf" => {
            let q = match quality {
                Quality::Fast => "22",
                Quality::Balanced => "18",
                Quality::High => "14",
            };
            opts.set("quality", "balanced");
            opts.set("qp_i", q);
            opts.set("qp_p", q);
            opts.set("rc", "cqp");
        }
        "libx265" => {
            let (preset, crf) = match quality {
                Quality::Fast => ("veryfast", "28"),
                Quality::Balanced => ("fast", "25"),
                Quality::High => ("medium", "21"),
            };
            opts.set("preset", preset);
            opts.set("crf", crf);
            opts.set("profile", "main");
        }
        "libsvtav1" => {
            let (preset, crf) = match quality {
                Quality::Fast => ("10", "35"),
                Quality::Balanced => ("7", "30"),
                Quality::High => ("4", "25"),
            };
            opts.set("preset", preset);
            opts.set("crf", crf);
        }
        "libaom-av1" => {
            let (crf, cpu_used) = match quality {
                Quality::Fast => ("35", "8"),
                Quality::Balanced => ("30", "6"),
                Quality::High => ("25", "4"),
            };
            opts.set("crf", crf);
            opts.set("cpu-used", cpu_used);
            opts.set("row-mt", "1");
        }
        "libx264" => {
            let (preset, crf) = match quality {
                Quality::Fast => ("ultrafast", "32"),
                Quality::Balanced => ("veryfast", "28"),
                Quality::High => ("fast", "23"),
            };
            opts.set("preset", preset);
            opts.set("tune", "zerolatency");
            opts.set("crf", crf);
            opts.set("profile", "high");
            // On Tegra (Jetson), limit threads to avoid starving other
            // pipeline stages (capture + pairing threads need CPU too).
            // x264 defaults to num_cpus which over-subscribes.
            if std::path::Path::new("/etc/nv_tegra_release").exists()
                || std::fs::read_to_string("/proc/device-tree/compatible")
                    .unwrap_or_default()
                    .contains("nvidia,tegra")
            {
                log::info!("Tegra platform detected, limiting libx264 to 4 threads");
                opts.set("threads", "4");
            }
        }
        _ => {
            log::info!("Encoder '{name}' has no quality presets configured, using FFmpeg defaults");
        }
    }

    // Apply CRF override: detect which quality key the encoder uses and replace it.
    if let Some(crf_val) = crf_override {
        let val = crf_val.to_string();
        // Check known quality keys in priority order.
        let quality_keys: &[&str] = &["crf", "cq", "global_quality", "qp", "qp_i", "q:v"];
        let mut found = false;
        for &key in quality_keys {
            if opts.get(key).is_some() {
                opts.set(key, &val);
                // Also override qp_p when qp_i is present (AMF uses paired keys).
                if key == "qp_i" && opts.get("qp_p").is_some() {
                    opts.set("qp_p", &val);
                }
                found = true;
                break;
            }
        }
        if !found {
            // Fallback: set "crf" for unknown encoders.
            opts.set("crf", &val);
        }
    }

    // Apply preset override.
    if let Some(preset) = preset_override {
        opts.set("preset", preset);
    }

    opts
}
