//! Video decoder: file → RGBA frames.
//!
//! Wraps FFmpeg's demuxer, decoder, and swscale to produce RGBA pixel data
//! frame-by-frame from any video file FFmpeg can read.

extern crate ffmpeg_next as ffmpeg;

use ffmpeg::format::{Pixel, input};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context as ScalingContext, flag::Flags as ScalingFlags};
use ffmpeg::util::frame::video::Video as VideoFrame;
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

/// A decoded RGBA frame with timestamp.
pub struct RgbaFrame {
    /// Raw RGBA pixel data (width * height * 4 bytes, tightly packed).
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in microseconds.
    pub timestamp_us: i64,
}

/// Video decoder that produces RGBA frames from a video file.
///
/// Uses FFmpeg's software decoder and swscale for pixel format conversion.
/// Call [`Self::next_frame`] repeatedly to decode frames.
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
    scaler: ScalingContext,
    video_stream_index: usize,
    time_base_num: i64,
    time_base_den: i64,
    width: u32,
    height: u32,
    eof_sent: bool,
    /// Reusable frame buffers to avoid per-frame allocation.
    decoded_frame: VideoFrame,
    rgba_frame: VideoFrame,
    /// Pre-allocated output buffer, reused across frames.
    output_buf: Vec<u8>,
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

        log::info!(
            "Decoder: {}x{} {:?}, time_base={}/{}",
            width,
            height,
            decoder.format(),
            time_base.0,
            time_base.1
        );

        let scaler = ScalingContext::get(
            decoder.format(),
            width,
            height,
            Pixel::RGBA,
            width,
            height,
            ScalingFlags::BILINEAR,
        )?;

        let frame_bytes = width as usize * height as usize * 4;

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
            rgba_frame: VideoFrame::empty(),
            output_buf: vec![0u8; frame_bytes],
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

    /// Decode the next RGBA frame, or `None` if the video is finished.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "decode_frame")
    )]
    pub fn next_frame(&mut self) -> Result<Option<RgbaFrame>, DecodeError> {
        if self.eof_sent {
            return Ok(None);
        }

        // Try to receive a frame from packets already sent to the decoder,
        // or read new packets until we get one.
        loop {
            if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                return Ok(Some(self.convert_frame()));
            }

            // Need more data — read next video packet
            let mut found_packet = false;
            // We need to use packets() iterator but break after processing one video packet
            for (stream, packet) in self.input.packets() {
                if stream.index() == self.video_stream_index {
                    self.decoder.send_packet(&packet)?;
                    found_packet = true;
                    break;
                }
            }

            if !found_packet {
                // EOF — flush the decoder
                self.eof_sent = true;
                self.decoder.send_eof()?;
                if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                    return Ok(Some(self.convert_frame()));
                }
                return Ok(None);
            }
        }
    }

    /// Convert the current decoded frame to RGBA and return it.
    /// Reuses internal buffers to avoid allocation.
    fn convert_frame(&mut self) -> RgbaFrame {
        self.scaler
            .run(&self.decoded_frame, &mut self.rgba_frame)
            .expect("swscale conversion failed");

        let pts = self.decoded_frame.pts().unwrap_or(0);
        let timestamp_us = if self.time_base_den != 0 {
            pts * self.time_base_num * 1_000_000 / self.time_base_den
        } else {
            0
        };

        // Copy RGBA data into reusable buffer, handling potential stride padding
        let stride = self.rgba_frame.stride(0);
        let row_bytes = self.width as usize * 4;
        let src = self.rgba_frame.data(0);

        if stride == row_bytes {
            self.output_buf[..row_bytes * self.height as usize]
                .copy_from_slice(&src[..row_bytes * self.height as usize]);
        } else {
            for y in 0..self.height as usize {
                let src_start = y * stride;
                let dst_start = y * row_bytes;
                self.output_buf[dst_start..dst_start + row_bytes]
                    .copy_from_slice(&src[src_start..src_start + row_bytes]);
            }
        }

        // We still need to clone here because the caller owns the Vec.
        // But output_buf stays allocated for next frame.
        RgbaFrame {
            data: self.output_buf.clone(),
            width: self.width,
            height: self.height,
            timestamp_us,
        }
    }
}
