//! Encoded audio selection and timestamp mapping for shared output packets.
use super::super::live_audio::{LiveAudio, TIME_BASE};
use super::*;

pub(super) enum AudioPassthrough {
    File(Box<FileAudio>),
    Live {
        source: LiveAudio,
        index: usize,
        time_base: Rational,
    },
}

impl AudioPassthrough {
    pub(super) fn open(
        octx: &mut format::context::Output,
        config: &EncoderConfig,
        live: Option<LiveAudio>,
    ) -> Result<Option<Self>, EncodeError> {
        if let Some(source) = live {
            if config.audio_source.is_some() {
                return Err(EncodeError::InvalidConfig(
                    "select either live or file audio".into(),
                ));
            }
            let mut stream = octx.add_stream(ffmpeg::encoder::find(codec::Id::None))?;
            stream.set_parameters(source.parameters.clone());
            stream.set_time_base(TIME_BASE);
            // SAFETY: this newly added stream owns writable parameters.
            unsafe {
                (*stream.parameters().as_mut_ptr()).codec_tag = 0;
            }
            return Ok(Some(Self::Live {
                source,
                index: stream.index(),
                time_base: TIME_BASE,
            }));
        }
        match &config.audio_source {
            Some(paths) => Ok(FileAudio::open(octx, paths, config.audio_start_time)?
                .map(|file| Self::File(Box::new(file)))),
            None => Ok(None),
        }
    }

    pub(super) fn refresh_time_base(&mut self, octx: &format::context::Output) {
        let (index, time_base) = match self {
            Self::File(file) => (file.output_stream_index, &mut file.output_time_base),
            Self::Live {
                index, time_base, ..
            } => (*index, time_base),
        };
        *time_base = octx
            .stream(index)
            .expect("audio stream we just added")
            .time_base();
    }

    pub(super) fn time_base(&self) -> Rational {
        match self {
            Self::File(file) => file.output_time_base,
            Self::Live { time_base, .. } => *time_base,
        }
    }

    pub(super) fn next_until(
        &mut self,
        max_pts: i64,
    ) -> Result<Option<ffmpeg::Packet>, EncodeError> {
        match self {
            Self::File(file) => file.next_until(max_pts),
            Self::Live {
                source,
                index,
                time_base,
            } => {
                let Some(mut packet) = source
                    .next_until(max_pts.rescale(*time_base, TIME_BASE))
                    .map_err(EncodeError::InvalidConfig)?
                else {
                    return Ok(None);
                };
                packet.set_stream(*index);
                packet.rescale_ts(TIME_BASE, *time_base);
                Ok(Some(packet))
            }
        }
    }
}

/// State for copying audio from one or more input files to the output.
///
/// For a chained recording the segments are copied back-to-back: when the
/// current segment's audio ends, the next opens and its timestamps are
/// shifted by `segment_offset` so the output audio is gapless.
pub(super) struct FileAudio {
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
    /// One already-mapped packet beyond the current video decode time.
    pending: Option<ffmpeg::Packet>,
}

impl FileAudio {
    /// Set up audio passthrough by adding an audio stream to the output context.
    ///
    /// Called before `write_header`. Returns `None` if no audio stream found.
    fn open(
        octx: &mut format::context::Output,
        sources: &[PathBuf],
        start_time_secs: f64,
    ) -> Result<Option<FileAudio>, EncodeError> {
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

        Ok(Some(FileAudio {
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
            pending: None,
        }))
    }

    /// Open the next chained segment for audio copy, continuing the timeline.
    ///
    /// Skips segments with no audio stream. Returns `false` when none remain.
    fn advance_segment(audio: &mut FileAudio) -> Result<bool, EncodeError> {
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

    fn next_until(&mut self, max_pts: i64) -> Result<Option<ffmpeg::Packet>, EncodeError> {
        let audio = self;
        if audio.exhausted {
            return Ok(None);
        }
        // Start-time trim is expressed in the (fixed) output time base.
        let start_pts = seconds_to_pts(audio.start_time_secs, audio.output_time_base);

        loop {
            let packet = if let Some(packet) = audio.pending.take() {
                packet
            } else {
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
                    }
                    Err(ffmpeg::Error::Eof) => {
                        // Current segment done; continue with the next chained one.
                        if !Self::advance_segment(audio)? {
                            audio.exhausted = true;
                            return Ok(None);
                        }
                        continue;
                    }
                    Err(error) => return Err(error.into()),
                }
                packet
            };
            if packet
                .pts()
                .or_else(|| packet.dts())
                .is_some_and(|pts| pts > max_pts)
            {
                audio.pending = Some(packet);
                return Ok(None);
            }
            return Ok(Some(packet));
        }
    }
}
