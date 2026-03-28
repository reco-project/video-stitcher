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

/// Configuration for the video encoder.
#[derive(Debug, Clone, Default)]
pub struct EncoderConfig {
    /// Force a specific encoder by name, or `None` for auto-detection.
    pub encoder_name: Option<String>,
    /// Output video codec (H.264, HEVC, AV1). Default: H.264.
    pub codec: VideoCodec,
    /// Quality preset.
    pub quality: Quality,
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

            // Fresh output context for each attempt
            let mut octx = format::output(path)?;

            match Self::try_open(
                &mut octx, codec, *pixel_fmt, *is_hw, width, height, fps, config, name,
            ) {
                Ok((enc_opened, scaler, stream_index, encoder_time_base, output_time_base)) => {
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

        if needs_global_header {
            enc.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        if !is_hw {
            enc.set_threading(ffmpeg::threading::Config::count(0));
        }

        let opts = build_encoder_opts(name, config.quality);
        let encoder = enc.open_with(opts)?;
        ost.set_parameters(&encoder);

        octx.write_header()?;

        let output_time_base = octx.stream(stream_index).unwrap().time_base();

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

    /// Write an RGBA frame to the output file.
    ///
    /// `rgba_data` must be exactly `width * height * 4` bytes (tightly packed).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "encode_frame")
    )]
    pub fn write_frame(&mut self, rgba_data: &[u8]) -> Result<(), EncodeError> {
        let expected = (self.width * self.height * 4) as usize;
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
        let expected = (self.width * self.height * 3 / 2) as usize;
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
        self.octx.write_trailer()?;
        Ok(())
    }

    fn receive_and_write_packets(&mut self) -> Result<(), EncodeError> {
        let mut packet = ffmpeg::Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(self.stream_index);
            packet.rescale_ts(self.encoder_time_base, self.output_time_base);
            packet.write_interleaved(&mut self.octx)?;
        }
        Ok(())
    }
}

/// Build encoder-specific FFmpeg options.
fn build_encoder_opts(name: &str, quality: Quality) -> ffmpeg::Dictionary<'static> {
    let mut opts = ffmpeg::Dictionary::new();

    match name {
        "h264_nvenc" => {
            let (preset, cq) = match quality {
                Quality::Fast => ("p3", "28"),
                Quality::Balanced => ("p4", "23"),
                Quality::High => ("p5", "19"),
            };
            opts.set("preset", preset);
            opts.set("tune", "hq");
            opts.set("rc", "vbr");
            opts.set("cq", cq);
            opts.set("b:v", "10M");
            opts.set("maxrate", "15M");
            opts.set("profile", "high");
            opts.set("spatial-aq", "1");
            opts.set("temporal-aq", "1");
        }
        "hevc_nvenc" => {
            let (preset, cq) = match quality {
                Quality::Fast => ("p3", "28"),
                Quality::Balanced => ("p4", "23"),
                Quality::High => ("p5", "19"),
            };
            opts.set("preset", preset);
            opts.set("tune", "hq");
            opts.set("rc", "vbr");
            opts.set("cq", cq);
            opts.set("b:v", "10M");
            opts.set("maxrate", "15M");
            opts.set("profile", "main");
            opts.set("spatial-aq", "1");
            opts.set("temporal-aq", "1");
        }
        "av1_nvenc" => {
            let (preset, cq) = match quality {
                Quality::Fast => ("p3", "32"),
                Quality::Balanced => ("p4", "27"),
                Quality::High => ("p5", "22"),
            };
            opts.set("preset", preset);
            opts.set("tune", "hq");
            opts.set("rc", "vbr");
            opts.set("cq", cq);
            opts.set("b:v", "8M");
            opts.set("maxrate", "12M");
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
            let q = match quality {
                Quality::Fast => "55",
                Quality::Balanced => "65",
                Quality::High => "80",
            };
            opts.set("q:v", q);
            opts.set("profile", "high");
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
        _ => {
            // libx264 and unknown encoders
            let (preset, crf) = match quality {
                Quality::Fast => ("ultrafast", "28"),
                Quality::Balanced => ("veryfast", "25"),
                Quality::High => ("fast", "21"),
            };
            opts.set("preset", preset);
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
                opts.set("threads", "3");
            }
        }
    }

    opts
}
