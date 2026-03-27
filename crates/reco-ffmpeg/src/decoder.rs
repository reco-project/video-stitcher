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
    pending_frames: Vec<RgbaFrame>,
    eof_sent: bool,
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

        let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
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

        Ok(Self {
            input: ictx,
            decoder,
            scaler,
            video_stream_index,
            time_base_num: time_base.0 as i64,
            time_base_den: time_base.1 as i64,
            width,
            height,
            pending_frames: Vec::new(),
            eof_sent: false,
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
    pub fn next_frame(&mut self) -> Result<Option<RgbaFrame>, DecodeError> {
        // Return buffered frames first
        if let Some(frame) = self.pending_frames.pop() {
            return Ok(Some(frame));
        }

        if self.eof_sent {
            return Ok(None);
        }

        // Read packets until we get decoded frames or hit EOF.
        // Using disjoint field borrows: self.input is borrowed by packets(),
        // while self.decoder/scaler/pending_frames are accessed separately.
        for (stream, packet) in self.input.packets() {
            if stream.index() == self.video_stream_index {
                self.decoder.send_packet(&packet)?;
                drain_decoder(
                    &mut self.decoder,
                    &mut self.scaler,
                    &mut self.pending_frames,
                    self.time_base_num,
                    self.time_base_den,
                    self.width,
                    self.height,
                )?;
                if !self.pending_frames.is_empty() {
                    return Ok(self.pending_frames.pop());
                }
            }
        }

        // EOF reached — flush remaining frames from the decoder
        self.eof_sent = true;
        self.decoder.send_eof()?;
        drain_decoder(
            &mut self.decoder,
            &mut self.scaler,
            &mut self.pending_frames,
            self.time_base_num,
            self.time_base_den,
            self.width,
            self.height,
        )?;
        Ok(self.pending_frames.pop())
    }
}

/// Drain all available decoded frames from the decoder, convert to RGBA.
fn drain_decoder(
    decoder: &mut ffmpeg::decoder::Video,
    scaler: &mut ScalingContext,
    pending: &mut Vec<RgbaFrame>,
    tb_num: i64,
    tb_den: i64,
    width: u32,
    height: u32,
) -> Result<(), DecodeError> {
    let mut decoded = VideoFrame::empty();
    while decoder.receive_frame(&mut decoded).is_ok() {
        let mut rgba = VideoFrame::empty();
        scaler.run(&decoded, &mut rgba)?;

        let pts = decoded.pts().unwrap_or(0);
        let timestamp_us = if tb_den != 0 {
            pts * tb_num * 1_000_000 / tb_den
        } else {
            0
        };

        // Copy RGBA data, handling potential stride padding
        let stride = rgba.stride(0);
        let row_bytes = width as usize * 4;
        let data = if stride == row_bytes {
            rgba.data(0)[..row_bytes * height as usize].to_vec()
        } else {
            let mut buf = Vec::with_capacity(row_bytes * height as usize);
            for y in 0..height as usize {
                let start = y * stride;
                buf.extend_from_slice(&rgba.data(0)[start..start + row_bytes]);
            }
            buf
        };

        pending.push(RgbaFrame {
            data,
            width,
            height,
            timestamp_us,
        });
    }
    Ok(())
}
