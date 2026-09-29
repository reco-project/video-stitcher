//! Actual encoded presentation timestamps and image order, decoded in software.

#![cfg(feature = "ffmpeg")]

use reco_core::sink::{OutputFrame, OutputSink, PixelFormat};
use reco_io::adapters::FfmpegFileEncoder;
use reco_io::ffmpeg::decoder::VideoDecoder;
use reco_io::ffmpeg::encoder::EncoderConfig;

fn roundtrip(name: &str, times: &[Option<i64>], fps: (i32, i32)) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("timing.mp4");
    let config = EncoderConfig {
        encoder_name: Some(name.into()),
        device: std::env::var("RECO_TEST_ENCODER_DEVICE").ok(),
        ..Default::default()
    };
    let mut encoder = FfmpegFileEncoder::new(&path, 256, 128, fps, &config).unwrap();
    let mut data = vec![128; 256 * 128 * 3 / 2];
    for (i, &pts_us) in times.iter().enumerate() {
        data[..256 * 128].fill(48 + i as u8 * 20);
        encoder
            .consume(OutputFrame {
                data: &data,
                width: 256,
                height: 128,
                format: PixelFormat::Nv12,
                pts_us,
            })
            .unwrap();
    }
    encoder.finish().unwrap();
    drop(encoder);
    let mut decoder = VideoDecoder::open_software(&path).unwrap();
    for (i, &supplied) in times.iter().enumerate() {
        let image = decoder.next_frame().unwrap().expect("missing image");
        let expected = supplied
            .unwrap_or_else(|| (i as f64 * 1e6 * fps.1 as f64 / fps.0 as f64).round() as i64);
        assert!(
            (image.timestamp_us - expected).abs() <= 1,
            "{name} frame{i}: {} != {expected}",
            image.timestamp_us
        );
        assert!(
            image.y[64 * 256 + 128].abs_diff(48 + i as u8 * 20) <= 3,
            "wrong image at frame{i}"
        );
    }
    assert!(decoder.next_frame().unwrap().is_none());
    let input = ffmpeg_next::format::input(&path).unwrap();
    let stream = input
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .unwrap();
    let tb = stream.time_base();
    let end = stream.duration() as f64 * tb.0 as f64 / tb.1 as f64;
    let expected_end = match times.last().unwrap() {
        Some(last) => *last as f64 / 1e6 + fps.1 as f64 / fps.0 as f64,
        None => times.len() as f64 * fps.1 as f64 / fps.0 as f64,
    };
    let start = times[0].unwrap_or(0) as f64 / 1e6;
    assert!(
        (end - (expected_end - start)).abs() < 0.00001,
        "{name}: end {end}, expected {expected_end}"
    );
}

#[test]
fn software_preserves_gaps_subframe_times_and_rational_cadence() {
    roundtrip("libx264", &[Some(1_000_000), Some(1_100_000)], (30, 1));
    roundtrip(
        "libx264",
        &[Some(0), Some(33_333), Some(133_333), Some(141_667)],
        (30, 1),
    );
    roundtrip("libx264", &[None; 8], (30000, 1001));
}

#[test]
fn sink_rejects_nonmonotonic_and_mixed_timing() {
    let dir = tempfile::tempdir().unwrap();
    let config = EncoderConfig {
        encoder_name: Some("libx264".into()),
        ..Default::default()
    };
    let mut encoder =
        FfmpegFileEncoder::new(&dir.path().join("invalid.mp4"), 64, 64, (30, 1), &config).unwrap();
    let data = [128; 64 * 64 * 3 / 2];
    let frame = |pts_us| OutputFrame {
        data: &data,
        width: 64,
        height: 64,
        format: PixelFormat::Nv12,
        pts_us,
    };
    assert!(encoder.consume(frame(Some(-1))).is_err());
    encoder.consume(frame(Some(0))).unwrap();
    assert!(encoder.consume(frame(Some(0))).is_err());
    assert!(encoder.consume(frame(None)).is_err());
    encoder.consume(frame(Some(33_333))).unwrap();
    encoder.finish().unwrap();
    assert!(encoder.consume(frame(Some(66_667))).is_err());
}

#[test]
#[ignore = "requires RECO_TEST_ENCODERS naming installed software/native backends"]
fn selected_encoders_preserve_output_timing() {
    for name in std::env::var("RECO_TEST_ENCODERS")
        .expect("explicit encoder list")
        .split(',')
    {
        roundtrip(
            name,
            &[Some(0), Some(33_333), Some(133_333), Some(141_667)],
            (30, 1),
        );
        roundtrip(name, &[None; 8], (30000, 1001));
    }
}

#[test]
fn copied_audio_uses_presentation_end_not_submitted_frame_count() {
    let dir = tempfile::tempdir().unwrap();
    let audio = dir.path().join("audio.m4a");
    let generated = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "1",
            "-c:a",
            "aac",
        ])
        .arg(&audio)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    for (filename, times, expected_end) in [
        ("fixed.mp4", [None; 3], 0.1),
        ("timed.mp4", [Some(0), Some(33_333), Some(466_667)], 0.5),
    ] {
        let path = dir.path().join(filename);
        let config = EncoderConfig {
            encoder_name: Some("libx264".into()),
            audio_source: Some(vec![audio.clone()]),
            ..Default::default()
        };
        let mut encoder = FfmpegFileEncoder::new(&path, 64, 64, (30, 1), &config).unwrap();
        for pts_us in times {
            encoder
                .consume(OutputFrame {
                    data: &[128; 64 * 64 * 3 / 2],
                    width: 64,
                    height: 64,
                    format: PixelFormat::Nv12,
                    pts_us,
                })
                .unwrap();
        }
        encoder.finish().unwrap();
        drop(encoder);
        let mut input = ffmpeg_next::format::input(&path).unwrap();
        let stream = input
            .streams()
            .best(ffmpeg_next::media::Type::Audio)
            .unwrap();
        let index = stream.index();
        let tb = stream.time_base();
        let mut end: f64 = 0.0;
        for (stream, packet) in input.packets() {
            if stream.index() == index {
                end = end.max(
                    (packet.pts().unwrap() + packet.duration()) as f64 * tb.0 as f64 / tb.1 as f64,
                );
            }
        }
        assert!(
            (end - expected_end).abs() < 0.025,
            "{filename}: audio ends at {end}, expected {expected_end}"
        );
    }
}
