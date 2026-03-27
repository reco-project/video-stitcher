//! Video decoder: file → YUV420P frames.
//!
//! Wraps FFmpeg's demuxer and decoder to produce YUV420P plane data
//! frame-by-frame from any video file FFmpeg can read. YUV planes are
//! uploaded directly to the GPU, eliminating CPU-side color conversion.

extern crate ffmpeg_next as ffmpeg;

use ffmpeg::format::{Pixel, input};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context as ScalingContext, flag::Flags as ScalingFlags};
use ffmpeg::util::frame::video::Video as VideoFrame;

/// Create a tracing span guard (no-op when `profiling` feature is disabled).
#[cfg(feature = "profiling")]
macro_rules! profile_scope {
    ($name:expr) => {
        let _span = tracing::info_span!($name).entered();
    };
}

#[cfg(not(feature = "profiling"))]
macro_rules! profile_scope {
    ($name:expr) => {};
}
use std::path::Path;
use thiserror::Error;

/// Errors from the video decoder.
#[derive(Debug, Error)]
pub enum DecodeError {
    /// FFmpeg error.
    #[error("FFmpeg: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// No video stream found in the file.
    #[error("no video stream found")]
    NoVideoStream,
}

/// A decoded YUV420P frame with timestamp.
///
/// Contains tightly-packed plane data (no stride padding):
/// - Y: `width × height` bytes (luma, full resolution)
/// - U: `(width/2) × (height/2)` bytes (chroma blue, half resolution)
/// - V: `(width/2) × (height/2)` bytes (chroma red, half resolution)
pub struct YuvFrame {
    /// Y (luma) plane data, tightly packed.
    pub y: Vec<u8>,
    /// U (Cb) plane data, tightly packed.
    pub u: Vec<u8>,
    /// V (Cr) plane data, tightly packed.
    pub v: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in microseconds.
    pub timestamp_us: i64,
}

/// Video decoder that produces YUV420P frames from a video file.
///
/// Uses FFmpeg's software decoder. If the input pixel format is already
/// YUV420P (the common case for H.264), planes are extracted directly
/// without any color conversion. For other formats, swscale converts
/// to YUV420P first.
///
/// # Example
///
/// ```rust,no_run
/// use reco_ffmpeg::decoder::VideoDecoder;
/// use std::path::Path;
///
/// let mut decoder = VideoDecoder::open(Path::new("video.mp4")).unwrap();
/// while let Some(frame) = decoder.next_frame().unwrap() {
///     println!("Frame: {}x{} @ {}us", frame.width, frame.height, frame.timestamp_us);
/// }
/// ```
pub struct VideoDecoder {
    input: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    /// Only created when input format is not YUV420P.
    scaler: Option<ScalingContext>,
    video_stream_index: usize,
    time_base_num: i64,
    time_base_den: i64,
    width: u32,
    height: u32,
    eof_sent: bool,
    /// Reusable decode buffer.
    decoded_frame: VideoFrame,
    /// Reusable conversion buffer (only used when scaler is active).
    converted_frame: VideoFrame,
}

impl VideoDecoder {
    /// Open a video file for decoding.
    pub fn open(path: &Path) -> Result<Self, DecodeError> {
        crate::init();

        let ictx = input(path)?;

        let stream = ictx
            .streams()
            .best(Type::Video)
            .ok_or(DecodeError::NoVideoStream)?;
        let video_stream_index = stream.index();
        let time_base = stream.time_base();

        let mut context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
        // Enable multithreaded decode (frame-level threading).
        // count(0) = auto-detect optimal thread count.
        context.set_threading(ffmpeg::threading::Config::count(0));
        let decoder = context.decoder().video()?;

        let width = decoder.width();
        let height = decoder.height();
        let format = decoder.format();

        log::info!(
            "Decoder: {}x{} {:?}, time_base={}/{}",
            width,
            height,
            format,
            time_base.0,
            time_base.1
        );

        // Only create a scaler if the input isn't already YUV420P.
        // For YUV420P (the common H.264 case), we extract planes directly.
        let scaler = if format != Pixel::YUV420P {
            log::info!(
                "Input format {:?} is not YUV420P — swscale will convert",
                format
            );
            Some(ScalingContext::get(
                format,
                width,
                height,
                Pixel::YUV420P,
                width,
                height,
                ScalingFlags::POINT,
            )?)
        } else {
            log::info!("Input is YUV420P — direct plane extraction (no swscale)");
            None
        };

        Ok(Self {
            input: ictx,
            decoder,
            scaler,
            video_stream_index,
            time_base_num: time_base.0 as i64,
            time_base_den: time_base.1 as i64,
            width,
            height,
            eof_sent: false,
            decoded_frame: VideoFrame::empty(),
            converted_frame: VideoFrame::empty(),
        })
    }

    /// Frame width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Frame rate as an FFmpeg rational (numerator/denominator).
    ///
    /// Falls back to 30fps if the decoder cannot determine the frame rate.
    pub fn frame_rate(&self) -> ffmpeg::Rational {
        self.decoder.frame_rate().unwrap_or_else(|| {
            log::warn!("Could not determine frame rate, defaulting to 30fps");
            ffmpeg::Rational(30, 1)
        })
    }

    /// Frame rate as frames per second.
    pub fn fps(&self) -> f64 {
        let r = self.frame_rate();
        r.0 as f64 / r.1 as f64
    }

    /// Decode the next YUV420P frame, or `None` if the video is finished.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "decode_frame")
    )]
    pub fn next_frame(&mut self) -> Result<Option<YuvFrame>, DecodeError> {
        if self.eof_sent {
            return Ok(None);
        }

        // Try to receive a frame from packets already sent to the decoder,
        // or read new packets until we get one.
        loop {
            {
                profile_scope!("h264_decode");
                if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                    return Ok(Some(self.extract_yuv()));
                }
            }

            // Need more data — read next video packet
            let mut found_packet = false;
            for (stream, packet) in self.input.packets() {
                if stream.index() == self.video_stream_index {
                    profile_scope!("send_packet");
                    self.decoder.send_packet(&packet)?;
                    found_packet = true;
                    break;
                }
            }

            if !found_packet {
                self.eof_sent = true;
                self.decoder.send_eof()?;
                if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                    return Ok(Some(self.extract_yuv()));
                }
                return Ok(None);
            }
        }
    }

    /// Extract YUV420P planes from the current decoded frame.
    ///
    /// If input is YUV420P, reads planes directly (zero conversion).
    /// Otherwise, uses swscale to convert to YUV420P first.
    fn extract_yuv(&mut self) -> YuvFrame {
        let source = if let Some(scaler) = &mut self.scaler {
            profile_scope!("swscale");
            scaler
                .run(&self.decoded_frame, &mut self.converted_frame)
                .expect("swscale conversion failed");
            &self.converted_frame
        } else {
            &self.decoded_frame
        };

        let pts = self.decoded_frame.pts().unwrap_or(0);
        let timestamp_us = if self.time_base_den != 0 {
            pts * self.time_base_num * 1_000_000 / self.time_base_den
        } else {
            0
        };

        let w = self.width as usize;
        let h = self.height as usize;
        let uv_w = w / 2;
        let uv_h = h / 2;

        let y = extract_plane(source.data(0), source.stride(0), w, h);
        let u = extract_plane(source.data(1), source.stride(1), uv_w, uv_h);
        let v = extract_plane(source.data(2), source.stride(2), uv_w, uv_h);

        YuvFrame {
            y,
            u,
            v,
            width: self.width,
            height: self.height,
            timestamp_us,
        }
    }
}

/// Copy one plane from an FFmpeg frame, removing stride padding.
///
/// If stride == width (common for 1920-wide frames), this is a single memcpy.
fn extract_plane(data: &[u8], stride: usize, width: usize, height: usize) -> Vec<u8> {
    if stride == width {
        data[..width * height].to_vec()
    } else {
        let mut out = Vec::with_capacity(width * height);
        for row in 0..height {
            let start = row * stride;
            out.extend_from_slice(&data[start..start + width]);
        }
        out
    }
}
