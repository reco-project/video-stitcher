//! Silent AAC stream generation and audio start-time timestamp helpers.

extern crate ffmpeg_next as ffmpeg;

use ffmpeg::{Rational, Rescale, codec, format};

use super::super::output_io::OutputIo;
use super::{EncodeError, receive_encoded_packet};

pub(super) fn sanitize_audio_start_time(secs: f64) -> f64 {
    if secs.is_finite() && secs > 0.0 {
        secs
    } else {
        0.0
    }
}

pub(super) fn seconds_to_pts(secs: f64, time_base: Rational) -> i64 {
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

/// Generates a silent AAC track on the streamed video timeline.
pub(super) struct SilentAudio {
    encoder: ffmpeg::encoder::audio::Encoder,
    pub(super) stream_index: usize,
    pub(super) output_time_base: Rational,
    pub(super) frame: ffmpeg::frame::Audio,
    next_pts: Option<i64>,
    pub(super) samples_per_frame: usize,
    pub(super) sample_rate: u32,
}

// SAFETY: SilentAudio is only used from a single thread (owned by StreamOutput / VideoEncoder).
unsafe impl Send for SilentAudio {}

impl SilentAudio {
    /// Add a silent AAC audio stream to the output context and prepare the encoder.
    pub(super) fn new(octx: &mut format::context::Output) -> Result<Self, EncodeError> {
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

    pub(super) fn write_for_video(
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

    pub(super) fn finish(&mut self, octx: &mut OutputIo, end_us: i64) -> Result<(), EncodeError> {
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
