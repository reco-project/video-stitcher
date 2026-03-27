//! Video encoder: RGBA frames → H.264 MP4 file.
//!
//! Wraps FFmpeg's muxer, encoder, and swscale to write RGBA pixel data
//! to an H.264-encoded MP4 file.

extern crate ffmpeg_next as ffmpeg;

use ffmpeg::format::Pixel;
use ffmpeg::software::scaling::{context::Context as ScalingContext, flag::Flags as ScalingFlags};
use ffmpeg::util::frame::video::Video as VideoFrame;
use ffmpeg::{Rational, codec, format};
use std::path::Path;
use thiserror::Error;

/// Errors from the video encoder.
#[derive(Debug, Error)]
pub enum EncodeError {
    /// FFmpeg error.
    #[error("FFmpeg: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// H.264 encoder not found (libx264 not available).
    #[error("H.264 encoder not found — is FFmpeg built with libx264?")]
    CodecNotFound,
}

/// Video encoder that writes RGBA frames to an H.264 MP4 file.
///
/// # Example
///
/// ```rust,no_run
/// use reco_ffmpeg::encoder::VideoEncoder;
/// use std::path::Path;
/// use ffmpeg_next::Rational;
///
/// let mut enc = VideoEncoder::new(Path::new("out.mp4"), 1920, 1080, Rational(30, 1)).unwrap();
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
}

impl VideoEncoder {
    /// Create a new H.264 MP4 encoder.
    ///
    /// # Arguments
    ///
    /// - `path`: Output file path (must end in `.mp4`)
    /// - `width`: Frame width in pixels
    /// - `height`: Frame height in pixels
    /// - `fps`: Frame rate as a rational (e.g., `Rational(30, 1)` for 30fps)
    pub fn new(path: &Path, width: u32, height: u32, fps: Rational) -> Result<Self, EncodeError> {
        ffmpeg::init()?;

        let mut octx = format::output(path)?;

        let codec = ffmpeg::encoder::find(codec::Id::H264).ok_or(EncodeError::CodecNotFound)?;

        // Check global header flag before adding stream (avoids borrow conflict)
        let needs_global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

        let mut ost = octx.add_stream(codec)?;
        let stream_index = ost.index();

        let mut encoder = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()?;

        encoder.set_width(width);
        encoder.set_height(height);
        encoder.set_format(Pixel::YUV420P);
        encoder.set_frame_rate(Some(fps));
        // Time base = inverse of fps for frame-index PTS
        let encoder_time_base = Rational(fps.1, fps.0);
        encoder.set_time_base(encoder_time_base);

        if needs_global_header {
            encoder.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        let mut opts = ffmpeg::Dictionary::new();
        opts.set("preset", "medium");
        opts.set("crf", "23");

        let encoder = encoder.open_with(opts)?;
        ost.set_parameters(&encoder);

        octx.write_header()?;

        // Read back the muxer's output time base (set after write_header)
        let output_time_base = octx.stream(stream_index).unwrap().time_base();

        log::info!(
            "Encoder: {}x{} H.264 @ {}/{} fps, output time_base={}/{}",
            width,
            height,
            fps.0,
            fps.1,
            output_time_base.0,
            output_time_base.1,
        );

        // Scaler: RGBA → YUV420P
        let scaler = ScalingContext::get(
            Pixel::RGBA,
            width,
            height,
            Pixel::YUV420P,
            width,
            height,
            ScalingFlags::BILINEAR,
        )?;

        Ok(Self {
            octx,
            encoder,
            scaler,
            stream_index,
            encoder_time_base,
            output_time_base,
            frame_count: 0,
            width,
            height,
        })
    }

    /// Write an RGBA frame to the output file.
    ///
    /// `rgba_data` must be exactly `width * height * 4` bytes (tightly packed).
    pub fn write_frame(&mut self, rgba_data: &[u8]) -> Result<(), EncodeError> {
        // Create RGBA source frame
        let mut rgba_frame = VideoFrame::new(Pixel::RGBA, self.width, self.height);

        // Copy data, handling stride padding
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

        // Convert RGBA → YUV420P
        let mut yuv_frame = VideoFrame::empty();
        self.scaler.run(&rgba_frame, &mut yuv_frame)?;
        yuv_frame.set_pts(Some(self.frame_count));

        self.encoder.send_frame(&yuv_frame)?;
        self.receive_and_write_packets()?;

        self.frame_count += 1;
        Ok(())
    }

    /// Flush the encoder and finalize the output file.
    ///
    /// Must be called after all frames have been written.
    pub fn finish(&mut self) -> Result<(), EncodeError> {
        self.encoder.send_eof()?;
        self.receive_and_write_packets()?;
        self.octx.write_trailer()?;
        log::info!("Encoder finished: {} frames written", self.frame_count);
        Ok(())
    }

    /// Drain encoded packets from the encoder and write them to the muxer.
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
