//! Container and stream metadata inspection and concat-seek helpers.

use super::{DecodeError, ffmpeg, is_full_range, raw_i32_to_pix_fmt};
use crate::ffmpeg::ColorMetadata;
use ffmpeg::ffi;
use ffmpeg::format::Pixel;
use ffmpeg::format::input;
use ffmpeg::media::Type;
use std::path::{Path, PathBuf};

/// Range and colour metadata reported by a probed video stream.
pub(super) fn stream_color(stream: &ffmpeg::Stream<'_>) -> (bool, ColorMetadata) {
    let parameters = stream.parameters();
    // SAFETY: parameters owns a live AVCodecParameters for this video stream.
    let raw = unsafe { &*parameters.as_ptr() };
    let format = Pixel::from(raw_i32_to_pix_fmt(raw.format));
    (
        is_full_range(format, raw.color_range.into()),
        ColorMetadata::from_raw(raw.color_primaries, raw.color_trc, raw.color_space),
    )
}

/// Container metadata for chain compatibility checks, without decoding frames.
pub(crate) struct RecordingMetadata {
    pub frames: Option<u64>,
    pub size: (u32, u32),
    pub rate: Option<ffmpeg::Rational>,
    pub rotation: i32,
}

pub(super) fn stream_rotation(stream: &ffmpeg::Stream<'_>) -> i32 {
    let Some(side_data) = stream
        .side_data()
        .find(|data| data.kind() == ffmpeg::codec::packet::side_data::Type::DisplayMatrix)
    else {
        return 0;
    };
    let Some(data) = side_data.data().get(..36) else {
        return 0;
    };
    let mut matrix = [0i32; 9];
    for (entry, bytes) in matrix.iter_mut().zip(data.chunks_exact(4)) {
        *entry = i32::from_ne_bytes(bytes.try_into().expect("four-byte matrix entry"));
    }
    // SAFETY: matrix contains the nine aligned native-endian entries required by FFmpeg.
    let angle = unsafe { ffi::av_display_rotation_get(matrix.as_ptr()) };
    ((-angle).round() as i32).rem_euclid(360)
}

/// Read each recording independently, rather than trusting concat's first stream.
pub(crate) fn probe_recording(path: &Path) -> Result<RecordingMetadata, DecodeError> {
    ffmpeg::init()?;
    let context = input(path)?;
    let stream = context
        .streams()
        .best(Type::Video)
        .ok_or(DecodeError::NoVideoStream)?;
    let parameters = stream.parameters();
    // SAFETY: parameters owns a live AVCodecParameters for this video stream.
    let size = unsafe {
        let parameters = &*parameters.as_ptr();
        (
            parameters.width.max(0) as u32,
            parameters.height.max(0) as u32,
        )
    };
    Ok(RecordingMetadata {
        frames: video_frame_count(&stream, context.duration()),
        size,
        rate: pick_frame_rate(Some(stream.rate()), Some(stream.avg_frame_rate()), None),
        rotation: stream_rotation(&stream),
    })
}

fn video_frame_count(stream: &ffmpeg::Stream<'_>, container_duration: i64) -> Option<u64> {
    if stream.frames() > 0 {
        return Some(stream.frames() as u64);
    }
    let rate = pick_frame_rate(Some(stream.rate()), Some(stream.avg_frame_rate()), None)?;
    let time_base = stream.time_base();
    let seconds = if stream.duration() > 0 && time_base.0 > 0 && time_base.1 > 0 {
        stream.duration() as f64 * time_base.0 as f64 / time_base.1 as f64
    } else if container_duration > 0 {
        container_duration as f64 / f64::from(ffi::AV_TIME_BASE)
    } else {
        return None;
    };
    let frames = (seconds * rate.0 as f64 / rate.1 as f64).round();
    (frames.is_finite() && frames > 0.0 && frames < u64::MAX as f64).then_some(frames as u64)
}

pub(super) struct ConcatProbe {
    pub durations: Option<Vec<f64>>,
    pub full_range: bool,
    pub color: ColorMetadata,
}

pub(super) fn probe_concat(
    first: &Path,
    first_context: &ffmpeg::format::context::Input,
    rest: &[PathBuf],
) -> Result<ConcatProbe, DecodeError> {
    let mut durations = Vec::with_capacity(1 + rest.len());
    let probe_segment = |path: &Path, context: &ffmpeg::format::context::Input| {
        let stream = context
            .streams()
            .best(Type::Video)
            .ok_or(DecodeError::NoVideoStream)?;
        let duration = context.duration();
        let duration = (duration > 0).then(|| duration as f64 / f64::from(ffi::AV_TIME_BASE));
        if duration.is_none() {
            log::warn!(
                "Concat demuxer: duration unknown for {}; cross-segment seek may be unavailable",
                path.display()
            );
        }
        Ok::<_, DecodeError>((stream_color(&stream), duration))
    };
    let ((full_range, mut color), duration) =
        probe_segment(first, first_context).map_err(|error| {
            DecodeError::Ffmpeg(format!("concat segment {}: {error}", first.display()))
        })?;
    durations.push(duration);
    for path in rest {
        let ((segment_range, segment_color), duration) = input(path)
            .map_err(DecodeError::from)
            .and_then(|context| probe_segment(path, &context))
            .map_err(|error| {
                DecodeError::Ffmpeg(format!("concat segment {}: {error}", path.display()))
            })?;
        durations.push(duration);
        if segment_range != full_range {
            return Err(DecodeError::ConversionError(format!(
                "segment color ranges differ ({} and {})",
                first.display(),
                path.display()
            )));
        }
        let compatible = color
            .matches_description(segment_color)
            .map_err(DecodeError::ConversionError)?;
        if !compatible {
            return Err(DecodeError::ConversionError(format!(
                "segment color metadata differ ({}: {color}; {}: {segment_color})",
                first.display(),
                path.display()
            )));
        }
        color = color.or(segment_color);
    }
    Ok(ConcatProbe {
        durations: durations.into_iter().collect(),
        full_range,
        color,
    })
}

/// Upper bound for a believable source frame rate. High-speed phone modes
/// reach 960fps; a 90000/1 rational is a demuxer surfacing its 90kHz time
/// base as `r_frame_rate`, not a camera.
const MAX_SANE_FPS: f64 = 1000.0;

/// FFmpeg reports "unset" as 0/0 or 0/1; anything non-positive or above
/// [`MAX_SANE_FPS`] is a placeholder or time-base artifact, not a rate.
pub(super) fn is_plausible_rate(r: &ffmpeg::Rational) -> bool {
    r.0 > 0 && r.1 > 0 && (r.0 as f64 / r.1 as f64) <= MAX_SANE_FPS
}

/// Pick the frame rate, nominal first, mirroring FFmpeg's own
/// `av_guess_frame_rate`: container `r_frame_rate` (for MP4 the demuxer
/// derives it from packet durations and snaps to the standard-rate table,
/// so it reads 30000/1001 where a raw measured average would read
/// 13455000/448949), then `avg_frame_rate`, then the codec context (needs
/// bitstream VUI timing). The `avg < 70 && r > 210` guard is ported from
/// av_guess_frame_rate: an r that high with a sane avg is a field or
/// telecine rate, not a picture rate.
pub(super) fn pick_frame_rate(
    stream_r: Option<ffmpeg::Rational>,
    stream_avg: Option<ffmpeg::Rational>,
    codec: Option<ffmpeg::Rational>,
) -> Option<ffmpeg::Rational> {
    let as_fps = |r: &ffmpeg::Rational| r.0 as f64 / r.1 as f64;
    let r = stream_r.filter(is_plausible_rate);
    let avg = stream_avg.filter(is_plausible_rate);
    match (r, avg) {
        (Some(r), Some(avg)) if as_fps(&avg) < 70.0 && as_fps(&r) > 210.0 => Some(avg),
        (Some(r), _) => Some(r),
        (None, Some(avg)) => Some(avg),
        (None, None) => codec.filter(is_plausible_rate),
    }
}

#[cfg(test)]
mod tests {
    use super::{ffi, input, pick_frame_rate, video_frame_count};
    use ffmpeg_next as ffmpeg;

    const R60: ffmpeg::Rational = ffmpeg::Rational(60, 1);
    const R30: ffmpeg::Rational = ffmpeg::Rational(30, 1);
    const NTSC: ffmpeg::Rational = ffmpeg::Rational(30000, 1001);
    const UNSET: ffmpeg::Rational = ffmpeg::Rational(0, 0);

    #[test]
    fn prefers_nominal_r_frame_rate() {
        // r is the demuxer's measured-then-snapped rate (30000/1001), avg
        // the raw average; nominal wins per av_guess_frame_rate.
        assert_eq!(
            pick_frame_rate(Some(NTSC), Some(R60), Some(R30)),
            Some(NTSC)
        );
    }

    #[test]
    fn falls_back_to_avg_frame_rate_when_r_unset() {
        assert_eq!(pick_frame_rate(Some(UNSET), Some(R60), None), Some(R60));
    }

    #[test]
    fn telecine_guard_prefers_sane_avg_over_field_rate() {
        // Ported av_guess_frame_rate guard: r=240 with avg=29.97 is a
        // field/telecine rate, not the picture rate.
        let field = ffmpeg::Rational(240, 1);
        assert_eq!(pick_frame_rate(Some(field), Some(NTSC), None), Some(NTSC));
        // r=120 is under the 210 threshold: a real high-speed rate wins.
        let r120 = ffmpeg::Rational(120, 1);
        assert_eq!(pick_frame_rate(Some(r120), Some(R60), None), Some(r120));
    }

    #[test]
    fn falls_back_to_codec_context_last() {
        assert_eq!(
            pick_frame_rate(Some(UNSET), Some(ffmpeg::Rational(0, 1)), Some(NTSC)),
            Some(NTSC)
        );
    }

    #[test]
    fn none_when_everything_is_unset() {
        assert_eq!(pick_frame_rate(Some(UNSET), Some(UNSET), None), None);
        assert_eq!(pick_frame_rate(None, None, None), None);
    }

    #[test]
    fn rejects_negative_rationals() {
        assert_eq!(
            pick_frame_rate(Some(ffmpeg::Rational(-1, 1)), Some(R30), None),
            Some(R30)
        );
    }

    /// A timing-less stream surfaces its 90kHz time base as r_frame_rate;
    /// trusting it would size the encoder at 90000fps, far worse than the
    /// old wrong-but-sane 30fps default.
    #[test]
    fn rejects_time_base_artifact_rates() {
        let tb = ffmpeg::Rational(90000, 1);
        assert_eq!(pick_frame_rate(Some(tb), Some(tb), Some(NTSC)), Some(NTSC));
        assert_eq!(pick_frame_rate(Some(tb), Some(tb), None), None);
    }

    #[test]
    fn frame_count_prefers_video_metadata_and_preserves_unknowns() {
        ffmpeg::init().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mjpeg60.mov");
        super::super::tests::write_mjpeg_60fps(&path);
        let mut context = input(&path).unwrap();
        let mut stream = context.stream_mut(0).unwrap();
        stream.set_rate(NTSC);
        stream.set_avg_frame_rate(NTSC);
        stream.set_time_base((1, 30000));
        // SAFETY: this test exclusively owns the demuxer and its stream.
        unsafe {
            (*stream.as_mut_ptr()).nb_frames = 48;
            (*stream.as_mut_ptr()).duration = 300300;
        }
        let longer_audio_duration = 100 * i64::from(ffi::AV_TIME_BASE);
        assert_eq!(video_frame_count(&stream, longer_audio_duration), Some(48));
        unsafe {
            (*stream.as_mut_ptr()).nb_frames = 0;
        }
        assert_eq!(video_frame_count(&stream, longer_audio_duration), Some(300));
        unsafe {
            (*stream.as_mut_ptr()).duration = ffi::AV_NOPTS_VALUE;
        }
        assert_eq!(
            video_frame_count(&stream, 10 * i64::from(ffi::AV_TIME_BASE)),
            Some(300)
        );
        assert_eq!(video_frame_count(&stream, ffi::AV_NOPTS_VALUE), None);
        stream.set_rate(UNSET);
        stream.set_avg_frame_rate(UNSET);
        assert_eq!(video_frame_count(&stream, longer_audio_duration), None);
    }
}
