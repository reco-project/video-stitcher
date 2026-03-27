//! Video encoder: RGBA frames → H.264 MP4 file.
//!
//! Wraps FFmpeg's muxer, encoder, and swscale to write RGBA pixel data
//! to an H.264-encoded MP4 file. Supports hardware-accelerated encoding
//! (NVENC, QSV, VideoToolbox, VAAPI) with automatic fallback to libx264.

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

    /// No H.264 encoder available.
    #[error("no H.264 encoder found — is FFmpeg built with libx264 or a hardware encoder?")]
    CodecNotFound,

    /// Frame data has wrong size.
    #[error("frame data size mismatch: expected {expected} bytes, got {actual}")]
    FrameSizeMismatch { expected: usize, actual: usize },
}

/// Known H.264 encoder candidates, in preference order.
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
        name: "h264_vaapi",
        is_hardware: true,
        pixel_format: Pixel::NV12,
    },
    EncoderCandidate {
        name: "libx264",
        is_hardware: false,
        pixel_format: Pixel::YUV420P,
    },
];

struct EncoderCandidate {
    name: &'static str,
    is_hardware: bool,
    pixel_format: Pixel,
}

/// Information about an available H.264 encoder.
#[derive(Debug, Clone)]
pub struct EncoderInfo {
    /// FFmpeg codec name (e.g., `"h264_nvenc"`, `"libx264"`).
    pub name: String,
    /// Human-readable description from FFmpeg.
    pub description: String,
    /// Whether this is a hardware-accelerated encoder.
    pub is_hardware: bool,
}

/// Detect which H.264 encoders are available in the linked FFmpeg build.
///
/// Returns encoders in preference order (hardware first, then software).
/// An encoder appearing here means it is compiled in, but it may still
/// fail to open if the hardware is absent at runtime.
pub fn available_h264_encoders() -> Vec<EncoderInfo> {
    crate::init();
    H264_ENCODERS
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
    /// Quality preset.
    pub quality: Quality,
}

/// Video encoder that writes RGBA frames to an H.264 MP4 file.
///
/// # Example
///
/// ```rust,no_run
/// use reco_ffmpeg::encoder::{VideoEncoder, EncoderConfig};
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
    /// Create a new H.264 MP4 encoder.
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

        // Build candidate list
        let candidates: Vec<(&str, bool, Pixel)> = if let Some(ref name) = config.encoder_name {
            let candidate = H264_ENCODERS.iter().find(|c| c.name == name.as_str());
            let pixel_fmt = candidate.map_or(Pixel::YUV420P, |c| c.pixel_format);
            let is_hw = candidate.is_some_and(|c| c.is_hardware);
            vec![(name.as_str(), is_hw, pixel_fmt)]
        } else {
            H264_ENCODERS
                .iter()
                .filter(|c| encoder::find_by_name(c.name).is_some())
                .map(|c| (c.name, c.is_hardware, c.pixel_format))
                .collect()
        };

        if candidates.is_empty() {
            return Err(EncodeError::CodecNotFound);
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
                        yuv_frame: VideoFrame::empty(),
                    });
                }
                Err(e) => {
                    log::warn!("Encoder {name} failed to open: {e}, trying next...");
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or(EncodeError::CodecNotFound))
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
        "h264_vaapi" => {
            let qp = match quality {
                Quality::Fast => "28",
                Quality::Balanced => "23",
                Quality::High => "19",
            };
            opts.set("qp", qp);
            opts.set("profile", "high");
        }
        _ => {
            let (preset, crf) = match quality {
                Quality::Fast => ("veryfast", "25"),
                Quality::Balanced => ("fast", "23"),
                Quality::High => ("medium", "19"),
            };
            opts.set("preset", preset);
            opts.set("crf", crf);
            opts.set("profile", "high");
        }
    }

    opts
}
