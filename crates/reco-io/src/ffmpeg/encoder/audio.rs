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
/// Each chapter keeps its position on the video timeline, including chapters
/// without audio. Packet times are translated only after the muxer fixes its time base.
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
    start_time_secs: f64,
    exhausted: bool,
    remaining_segments: std::collections::VecDeque<PathBuf>,
    /// Offset of the current file's presentation origin on the combined timeline.
    shift_secs: f64,
    chapter_start_secs: f64,
    /// Next chapter's video start, in seconds from the start of the whole chain.
    next_start_secs: f64,
    chained: bool,
    /// One already-mapped packet beyond the current video decode time.
    pending: Option<ffmpeg::Packet>,
}

/// A chapter's video defines its place in the recording even when audio is absent.
fn chapter_timing(ictx: &format::context::Input) -> Result<(f64, f64), EncodeError> {
    let stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .or_else(|| ictx.streams().best(ffmpeg::media::Type::Audio))
        .ok_or_else(|| {
            EncodeError::InvalidConfig("chapter has no audio or video timeline".into())
        })?;
    let tick = f64::from(stream.time_base());
    let start = if stream.start_time() == ffmpeg::ffi::AV_NOPTS_VALUE {
        0.0
    } else {
        stream.start_time() as f64 * tick
    };
    let duration = if stream.duration() > 0 {
        stream.duration() as f64 * tick
    } else if stream.frames() > 0 && stream.avg_frame_rate().0 > 0 {
        stream.frames() as f64 / f64::from(stream.avg_frame_rate())
    } else {
        ictx.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
    };
    if !duration.is_finite() || duration <= 0.0 {
        return Err(EncodeError::InvalidConfig(
            "cannot determine chained video duration for audio copy".into(),
        ));
    }
    Ok((start, duration))
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
        let chained = sources.len() > 1;
        let mut remaining_segments: std::collections::VecDeque<_> =
            sources.iter().cloned().collect();
        let mut next_start_secs = 0.0;
        let start_time_secs = sanitize_audio_start_time(start_time_secs);
        let (source_path, mut ictx, shift_secs, chapter_start_secs) = loop {
            let Some(path) = remaining_segments.pop_front() else {
                return Ok(None);
            };
            let ictx = format::input(&path)?;
            let chapter_start_secs = next_start_secs;
            let mut shift = 0.0;
            if chained {
                let (origin, duration) = chapter_timing(&ictx)?;
                shift = next_start_secs - origin;
                next_start_secs += duration;
                if next_start_secs <= start_time_secs {
                    continue;
                }
            }
            if ictx.streams().best(ffmpeg::media::Type::Audio).is_some() {
                break (path, ictx, shift, chapter_start_secs);
            }
        };
        let audio_stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .expect("selected an audible chapter");
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

        if start_time_secs > shift_secs {
            let seek_ts = ((start_time_secs - shift_secs) * f64::from(ffmpeg::ffi::AV_TIME_BASE))
                .round() as i64;
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
            remaining_segments,
            shift_secs,
            chapter_start_secs,
            next_start_secs,
            chained,
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
            let (origin, duration) = chapter_timing(&ictx)?;
            audio.chapter_start_secs = audio.next_start_secs;
            audio.shift_secs = audio.next_start_secs - origin;
            audio.next_start_secs += duration;
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

                        let shift =
                            (audio.shift_secs / f64::from(audio.output_time_base)).round() as i64;
                        let delta = shift - start_pts;
                        packet.set_pts(packet.pts().map(|ts| ts + delta));
                        packet.set_dts(packet.dts().map(|ts| ts + delta));
                        // Skip pre-roll packets that would overlap the preceding chapter.
                        if start_pts > 0 || audio.chapter_start_secs > 0.0 {
                            let lower =
                                seconds_to_pts(audio.chapter_start_secs, audio.output_time_base)
                                    .saturating_sub(start_pts)
                                    .max(0);
                            if packet
                                .dts()
                                .or_else(|| packet.pts())
                                .is_some_and(|ts| ts < lower)
                            {
                                continue;
                            }
                        }
                        // Audio can extend past its own chapter's video end.
                        if audio.chained
                            && packet.pts().is_some_and(|pts| {
                                pts >= seconds_to_pts(audio.next_start_secs, audio.output_time_base)
                                    - start_pts
                            })
                        {
                            continue;
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
