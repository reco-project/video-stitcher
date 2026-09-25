//! Encoded content, order and timing verified through software decoding.
//!
//! Native encoder runs are opt-in: set `RECO_TEST_ENCODERS` to comma-separated FFmpeg encoder names and run the ignored test. `RECO_TEST_ENCODE_FRAMES` optionally increases the sequence length. Every requested encoder must work; unavailable hardware is a failure, not a silently skipped test.

#![cfg(feature = "ffmpeg")]

use reco_io::ffmpeg::decoder::{DecodeBackend, VideoDecoder};
use reco_io::ffmpeg::encoder::{EncoderConfig, VideoEncoder};
use reco_io::output::{Codec, Container, Quality};

#[test]
fn explicit_container_overrides_the_destination_extension() {
    for container in [
        Container::Mp4,
        Container::Mp4Fragmented,
        Container::Mov,
        Container::Mkv,
        Container::Flv,
    ] {
        for filename in ["output.mp4", "output.mov", "output.mkv", "output"] {
            eprintln!("container={container:?} destination={filename}");
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(filename);
            let config = EncoderConfig {
                container,
                encoder_name: Some("libx264".into()),
                ..Default::default()
            };
            let mut encoder = VideoEncoder::new(&path, 64, 64, (30, 1).into(), &config).unwrap();
            for _ in 0..3 {
                encoder.write_nv12_frame(&[128; 64 * 64 * 3 / 2]).unwrap();
            }
            encoder.finish().unwrap();
            drop(encoder);
            let input = ffmpeg_next::format::input(&path).unwrap();
            match container {
                Container::Mp4 | Container::Mp4Fragmented => {
                    assert_eq!(
                        input.metadata().get("major_brand"),
                        Some("isom"),
                        "{container:?}: {filename}"
                    );
                }
                Container::Mkv => assert_eq!(input.format().name(), "matroska,webm"),
                Container::Flv => assert_eq!(input.format().name(), "flv"),
                Container::Mov => assert_eq!(input.metadata().get("major_brand"), Some("qt  ")),
                _ => panic!("untested container"),
            }
            drop(input);
            let mut decoder = VideoDecoder::open_software(&path).unwrap();
            for _ in 0..3 {
                let frame = decoder.next_frame().unwrap().expect("missing output frame");
                assert_eq!((frame.width, frame.height), (64, 64));
                assert_near(frame.y[32 * 64 + 32], 128, 0, "Y");
            }
            assert!(decoder.next_frame().unwrap().is_none());
        }
    }
}

const WIDTH: usize = 260;
const HEIGHT: usize = 130;
const PATCHES: [(u8, u8, u8); 4] = [
    (16, 128, 128),
    (235, 128, 128),
    (63, 102, 240),
    (173, 42, 26),
];

#[test]
fn software_h264_selection_is_not_hardware() {
    software_selection(&[(Codec::H264, ffmpeg_next::codec::Id::H264)]);
}

#[test]
#[ignore = "requires software H.264, HEVC and AV1 encoders"]
fn software_selection_preserves_requested_codec() {
    software_selection(&[
        (Codec::H264, ffmpeg_next::codec::Id::H264),
        (Codec::HEVC, ffmpeg_next::codec::Id::HEVC),
        (Codec::AV1, ffmpeg_next::codec::Id::AV1),
    ]);
}

fn software_selection(cases: &[(Codec, ffmpeg_next::codec::Id)]) {
    for &(codec, expected) in cases {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("output.mp4");
        let config = EncoderConfig {
            codec: Some(codec),
            software_only: true,
            quality: Quality::Preset(reco_io::output::QualityPreset::Fast),
            ..Default::default()
        };
        let mut encoder = VideoEncoder::new(&path, 256, 128, (30, 1).into(), &config).unwrap();
        assert!(encoder.encoder_name().starts_with("lib"));
        encoder.write_nv12_frame(&[128; 256 * 128 * 3 / 2]).unwrap();
        encoder.finish().unwrap();
        drop(encoder);
        let input = ffmpeg_next::format::input(&path).unwrap();
        let stream = input
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .unwrap();
        assert_eq!(stream.parameters().id(), expected);
    }
}

#[test]
#[ignore = "requires RECO_TEST_ENCODERS with a HEVC backend and optional RECO_TEST_ENCODER_DEVICE"]
fn hevc_dimensions_are_correct_or_explicitly_rejected() {
    use reco_io::ffmpeg::encoder::EncodeError;
    let name = std::env::var("RECO_TEST_ENCODERS").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("output.mp4");
    let config = EncoderConfig {
        encoder_name: Some(name),
        codec: Some(Codec::HEVC),
        device: std::env::var("RECO_TEST_ENCODER_DEVICE").ok(),
        ..Default::default()
    };
    let mut encoder = VideoEncoder::new(&path, 260, 130, (30, 1).into(), &config).unwrap();
    let result = (|| {
        for _ in 0..48 {
            encoder.write_nv12_frame(&[128; 260 * 130 * 3 / 2])?;
        }
        encoder.finish()
    })();
    match result {
        Err(EncodeError::EncodedDimensions { expected, actual }) => {
            assert_eq!(expected, (260, 130));
            assert!(
                actual.0 > 0 && actual.1 > 0,
                "header dimensions were not parsed: {actual:?}"
            );
            assert_ne!(actual, (260, 130));
        }
        Err(error) => panic!("unexpected encoding failure: {error}"),
        Ok(()) => {
            drop(encoder);
            let mut decoder = VideoDecoder::open_software(&path).unwrap();
            for _ in 0..48 {
                let frame = decoder.next_frame().unwrap().unwrap();
                assert_eq!((frame.width, frame.height), (260, 130));
            }
            assert!(decoder.next_frame().unwrap().is_none());
        }
    }
}

fn planes(index: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![16; WIDTH * HEIGHT];
    let mut u = vec![128; WIDTH * HEIGHT / 4];
    let mut v = u.clone();
    // A deliberately black frame is valid content, not an encoder failure.
    if index == 0 {
        return (y, u, v);
    }
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            let patch = if row < 64 && col < 256 {
                PATCHES[col / 64]
            } else if col < 128 {
                (48 + (index % 32) as u8 * 5, 128, 128)
            } else {
                (48 + ((index / 32) % 32) as u8 * 5, 128, 128)
            };
            y[row * WIDTH + col] = patch.0;
            u[(row / 2) * (WIDTH / 2) + col / 2] = patch.1;
            v[(row / 2) * (WIDTH / 2) + col / 2] = patch.2;
        }
    }
    (y, u, v)
}

fn assert_near(actual: u8, expected: u8, index: usize, plane: &str) {
    assert!(
        actual.abs_diff(expected) <= 4,
        "frame {index} {plane}: expected {expected}, got {actual}",
    );
}

fn roundtrip(name: &str, count: usize, planar: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("output.mp4");
    let codec = if name.contains("265") || name.starts_with("hevc") {
        Codec::HEVC
    } else if name.contains("av1") {
        Codec::AV1
    } else {
        Codec::H264
    };
    let config = EncoderConfig {
        encoder_name: Some(name.into()),
        device: std::env::var("RECO_TEST_ENCODER_DEVICE").ok(),
        codec: Some(codec),
        quality: Quality::Value(90),
        gop_size: Some(30),
        ..Default::default()
    };
    let mut encoder = VideoEncoder::new(
        &path,
        WIDTH as u32,
        HEIGHT as u32,
        (30000, 1001).into(),
        &config,
    )
    .unwrap();
    assert_eq!(encoder.encoder_name(), name, "forced encoder changed");
    for index in 0..count {
        let (y, u, v) = planes(index);
        if planar {
            encoder.write_yuv420p_planes(&y, &u, &v).unwrap();
        } else {
            let mut nv12 = y;
            for (u, v) in u.into_iter().zip(v) {
                nv12.extend_from_slice(&[u, v]);
            }
            encoder.write_nv12_frame(&nv12).unwrap();
        }
    }
    encoder.finish().unwrap();
    encoder.finish().unwrap();
    drop(encoder);

    let mut decoder = VideoDecoder::open_software(&path).unwrap();
    assert_eq!(decoder.backend(), DecodeBackend::Software);
    for index in 0..count {
        let frame = decoder.next_frame().unwrap().expect("missing output frame");
        assert_eq!((frame.width, frame.height), (WIDTH as u32, HEIGHT as u32));
        let expected_pts = (index as i64 * 1_001_000 + 15) / 30;
        assert!(
            frame.timestamp_us.abs_diff(expected_pts) <= 1,
            "frame {index}: PTS {}, expected {expected_pts}",
            frame.timestamp_us,
        );
        let (y, u, v) = planes(index);
        for (col, row) in [
            (32, 16),
            (96, 16),
            (160, 16),
            (224, 16),
            (64, 96),
            (192, 96),
        ] {
            let luma = row * WIDTH + col;
            let chroma = (row / 2) * (WIDTH / 2) + col / 2;
            assert_near(frame.y[luma], y[luma], index, "Y");
            assert_near(frame.u[chroma], u[chroma], index, "U");
            assert_near(frame.v[chroma], v[chroma], index, "V");
        }
    }
    assert!(
        decoder.next_frame().unwrap().is_none(),
        "extra output frame"
    );
    assert!(decoder.next_frame().unwrap().is_none(), "EOF not retained");
}

#[test]
fn software_output_preserves_pixels_order_and_fractional_timing() {
    for planar in [false, true] {
        roundtrip("libx264", 48, planar);
    }
}

#[test]
#[ignore = "requires RECO_TEST_ENCODERS and the corresponding encoder hardware/libraries"]
fn selected_encoders_preserve_pixels_order_and_fractional_timing() {
    let names = std::env::var("RECO_TEST_ENCODERS").expect("set RECO_TEST_ENCODERS");
    let count: usize = std::env::var("RECO_TEST_ENCODE_FRAMES")
        .map(|value| value.parse().expect("positive frame count"))
        .unwrap_or(48);
    assert!(count > 0);
    for name in names.split(',') {
        assert!(!name.is_empty(), "empty encoder name");
        for planar in [false, true] {
            eprintln!("encoder={name} frames={count} planar={planar}");
            roundtrip(name, count, planar);
        }
    }
}
