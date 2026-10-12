//! Selected encoded audio must be shared with the optional stream, not replaced by silence.

use super::*;
use ffmpeg::Rescale;

fn tone(path: &Path) {
    let result = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=997:sample_rate=48000",
            "-t",
            "2",
            "-c:a",
            "aac",
        ])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn packets(path: &Path, medium: ffmpeg::media::Type) -> Vec<(i64, Vec<u8>)> {
    let mut input = format::input(path).unwrap();
    input
        .packets()
        .filter_map(|(stream, packet)| {
            (stream.parameters().medium() == medium).then(|| {
                (
                    packet
                        .pts()
                        .unwrap()
                        .rescale(stream.time_base(), (1, 1_000_000)),
                    packet.data().unwrap().to_vec(),
                )
            })
        })
        .collect()
}

#[test]
fn secondary_audio_is_the_selected_source_not_silence() {
    for (preset, start, chained, gaps) in [
        ("ultrafast", 0.0, false, false),
        ("ultrafast", 0.25, false, true),
        ("medium", 0.25, true, false),
    ] {
        compare_outputs(preset, start, chained, gaps);
    }
}

fn compare_outputs(preset: &str, start: f64, chained: bool, gaps: bool) {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("tone.m4a");
    tone(&input);
    let primary = dir.path().join("primary.mp4");
    let secondary = dir.path().join("secondary.flv");
    let config = EncoderConfig {
        encoder_name: Some("libx264".into()),
        preset: Some(preset.into()),
        software_only: true,
        audio_source: Some(if chained {
            vec![input.clone(), input]
        } else {
            vec![input]
        }),
        audio_start_time: start,
        ..Default::default()
    };
    let mut encoder = VideoEncoder::new(&primary, 64, 64, Rational(30, 1), &config).unwrap();
    let output = OutputIo::open(&secondary, "flv", &config, false).unwrap();
    let monitor = output.monitor();
    encoder.stream = Some(
        VideoEncoder::configure_stream_output(output, &encoder.octx, encoder.stream_index).unwrap(),
    );
    assert_eq!(monitor.status(), StreamStatus::Connected);
    let count = if chained { 90 } else { 30 };
    for index in 0..count {
        let gap = if gaps && index > 15 { 250_000 } else { 0 };
        encoder
            .write_nv12_frame(&[128; 64 * 64 * 3 / 2], Some(index * 1_000_000 / 30 + gap))
            .unwrap();
    }
    encoder.stream.as_mut().unwrap().octx.flush().unwrap();
    assert!(
        packets(&secondary, ffmpeg::media::Type::Audio).len() > 10,
        "source audio must reach the muxer before finalization"
    );
    encoder.finish().unwrap();
    drop(encoder);
    assert_eq!(monitor.status(), StreamStatus::Finished);
    let recording = packets(&primary, ffmpeg::media::Type::Audio);
    let streaming = packets(&secondary, ffmpeg::media::Type::Audio);
    assert!(recording.len() > 30);
    assert_eq!(
        streaming.len(),
        recording.len(),
        "stream must carry selected audio packets"
    );
    let shift = streaming[0].0 - recording[0].0;
    for ((time, data), (expected_time, expected_data)) in streaming.iter().zip(&recording) {
        assert_eq!(
            data, expected_data,
            "stream replaced or re-encoded selected source audio"
        );
        assert!(
            (time - expected_time).abs_diff(shift) <= 1000,
            "relative audio timing changed"
        );
    }
    let recorded_video = packets(&primary, ffmpeg::media::Type::Video);
    let streamed_video = packets(&secondary, ffmpeg::media::Type::Video);
    assert_eq!(recorded_video.len(), count as usize);
    assert_eq!(streamed_video.len(), recorded_video.len());
    for ((time, data), (expected_time, expected_data)) in streamed_video.iter().zip(&recorded_video)
    {
        assert_eq!(data, expected_data, "video packets must also be shared");
        assert!(
            (time - expected_time).abs_diff(shift) <= 1000,
            "relative audio/video timing changed"
        );
    }
    let expected_end = if gaps {
        1_250_000
    } else {
        count * 1_000_000 / 30
    };
    assert!(recording.last().unwrap().0.abs_diff(expected_end) < 25_000);
}
