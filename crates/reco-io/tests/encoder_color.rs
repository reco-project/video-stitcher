//! Rendered-output conversion agrees across RGBA and NV12 delivery without relabelling raw planes.

#![cfg(feature = "ffmpeg")]

use reco_core::render::nv12_cpu::rgba_to_nv12_into;
use reco_core::sink::{OutputFrame, OutputSink, PixelFormat};
use reco_io::adapters::FfmpegFileEncoder;
use reco_io::ffmpeg::decoder::VideoDecoder;
use reco_io::ffmpeg::encoder::{EncoderConfig, VideoEncoder};
use reco_io::output::Container;

#[test]
fn rendered_rgba_and_nv12_have_matching_colors() {
    check_rendered_colors("libx264", None, 3, Container::Mp4);
}

#[test]
fn software_hevc_preserves_rendered_color_across_containers() {
    for container in [
        Container::Mp4,
        Container::Mp4Fragmented,
        Container::Mov,
        Container::Mkv,
    ] {
        check_rendered_colors("libx265", None, 3, container);
    }
}

#[test]
#[ignore = "requires explicitly selected native encoders"]
fn selected_encoders_preserve_rendered_colors() {
    for name in std::env::var("RECO_TEST_ENCODERS")
        .expect("set RECO_TEST_ENCODERS")
        .split(',')
    {
        let count = std::env::var("RECO_TEST_ENCODE_FRAMES")
            .map(|value| value.parse().unwrap())
            .unwrap_or(3);
        check_rendered_colors(
            name,
            std::env::var("RECO_TEST_ENCODER_DEVICE").ok(),
            count,
            Container::Mp4,
        );
    }
}

fn check_rendered_colors(name: &str, device: Option<String>, count: usize, container: Container) {
    assert!(count > 0);
    let dir = tempfile::tempdir().unwrap();
    let patches = [[208, 48, 48], [48, 208, 48], [48, 48, 208], [208, 208, 48]];
    let mut rgba = vec![255; 256 * 128 * 4];
    for row in 0..128 {
        for col in 0..256 {
            rgba[(row * 256 + col) * 4..(row * 256 + col) * 4 + 3]
                .copy_from_slice(&patches[col / 64]);
        }
    }
    let mut nv12 = Vec::new();
    rgba_to_nv12_into(&rgba, 256, 256, 128, &mut nv12).unwrap();
    let mut decoded = Vec::new();
    for is_rgba in [true, false] {
        let path = dir.path().join(format!("{is_rgba}.mp4"));
        let config = EncoderConfig {
            encoder_name: Some(name.into()),
            device: device.clone(),
            quality: reco_io::output::Quality::Value(100),
            gop_size: Some(3),
            container,
            ..Default::default()
        };
        let mut encoder = FfmpegFileEncoder::new(&path, 256, 128, (30, 1), &config).unwrap();
        assert_eq!(encoder.encoder_name(), name);
        for index in 0..count {
            encoder
                .consume(OutputFrame {
                    data: if is_rgba { &rgba } else { &nv12 },
                    width: 256,
                    height: 128,
                    pts_us: Some(index as i64 * 1_000_000 / 30),
                    format: if is_rgba {
                        PixelFormat::Rgba8
                    } else {
                        PixelFormat::Nv12
                    },
                })
                .unwrap();
        }
        encoder.finish().unwrap();
        drop(encoder);
        assert_matrix(&path, true);
        assert_decoded_matrix(&path);
        let mut decoder = VideoDecoder::open_software(&path).unwrap();
        for index in 0..count {
            let frame = decoder.next_frame().unwrap().expect("missing output frame");
            // Matroska defaults to millisecond timestamps; MP4/MOV preserve finer ticks.
            let tolerance_us = if container == Container::Mkv { 1000 } else { 1 };
            assert!(
                (frame.timestamp_us - index as i64 * 1_000_000 / 30).abs() <= tolerance_us,
                "{container:?} frame {index}: {} us",
                frame.timestamp_us
            );
            for col in [32, 96, 160, 224] {
                let y = 32 * 256 + col;
                let uv = 16 * 128 + col / 2;
                for (actual, expected) in [
                    (frame.y[y], nv12[y]),
                    (frame.u[uv], nv12[256 * 128 + 2 * uv]),
                    (frame.v[uv], nv12[256 * 128 + 2 * uv + 1]),
                ] {
                    assert!(
                        actual.abs_diff(expected) <= 3,
                        "{name} frame {index}: {actual} vs {expected}"
                    );
                }
            }
            if index == 0 {
                decoded.push(frame);
            }
        }
        assert!(decoder.next_frame().unwrap().is_none());
    }
    for col in [32, 96, 160, 224] {
        let a = &decoded[0];
        let b = &decoded[1];
        let y = 32 * 256 + col;
        let uv = 16 * 128 + col / 2;
        let expected = [
            nv12[y],
            nv12[256 * 128 + 2 * uv],
            nv12[256 * 128 + 2 * uv + 1],
        ];
        for ((left, right), expected) in [(a.y[y], b.y[y]), (a.u[uv], b.u[uv]), (a.v[uv], b.v[uv])]
            .into_iter()
            .zip(expected)
        {
            assert!(
                left.abs_diff(right) <= 3,
                "{name}: RGBA/NV12 differ: {left} vs {right}"
            );
            assert!(
                left.abs_diff(expected) <= 3 && right.abs_diff(expected) <= 3,
                "{name}: decoded values {left}, {right} differ from rendered BT.709 value {expected}"
            );
        }
    }
}

fn assert_decoded_matrix(path: &std::path::Path) {
    let mut input = ffmpeg_next::format::input(path).unwrap();
    let stream = input
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .unwrap();
    let index = stream.index();
    let mut decoder = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
        .unwrap()
        .decoder()
        .video()
        .unwrap();
    let mut frame = ffmpeg_next::frame::Video::empty();
    for (stream, packet) in input.packets() {
        if stream.index() != index {
            continue;
        }
        decoder.send_packet(&packet).unwrap();
        if decoder.receive_frame(&mut frame).is_ok() {
            break;
        }
    }
    if frame.width() == 0 {
        decoder.send_eof().unwrap();
        decoder.receive_frame(&mut frame).unwrap();
    }
    assert_eq!(
        frame.color_space(),
        ffmpeg_next::color::Space::BT709,
        "decoded frame must not override the container's matrix"
    );
    assert_eq!(frame.color_range(), ffmpeg_next::color::Range::MPEG);
    assert_eq!(
        frame.color_primaries(),
        ffmpeg_next::color::Primaries::Unspecified
    );
    assert_eq!(
        frame.color_transfer_characteristic(),
        ffmpeg_next::color::TransferCharacteristic::Unspecified
    );
}

fn assert_matrix(path: &std::path::Path, rendered: bool) {
    let input = ffmpeg_next::format::input(path).unwrap();
    let stream = input
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .unwrap();
    let parameters = stream.parameters();
    // SAFETY: parameters remains alive while its fields are inspected.
    let parameters = unsafe { &*parameters.as_ptr() };
    use ffmpeg_next::ffi::{
        AVColorPrimaries, AVColorRange, AVColorSpace, AVColorTransferCharacteristic,
    };
    assert_eq!(
        parameters.color_space,
        if rendered {
            AVColorSpace::AVCOL_SPC_BT709
        } else {
            AVColorSpace::AVCOL_SPC_UNSPECIFIED
        }
    );
    if rendered {
        assert_eq!(parameters.color_range, AVColorRange::AVCOL_RANGE_MPEG);
    }
    assert_eq!(
        parameters.color_primaries,
        AVColorPrimaries::AVCOL_PRI_UNSPECIFIED
    );
    assert_eq!(
        parameters.color_trc,
        AVColorTransferCharacteristic::AVCOL_TRC_UNSPECIFIED
    );
}

#[test]
fn session_sink_labels_rendered_pixels_but_generic_encoder_does_not() {
    use reco_core::sink::{OutputFrame, OutputSink, PixelFormat};
    let dir = tempfile::tempdir().unwrap();
    let config = EncoderConfig {
        encoder_name: Some("libx264".into()),
        ..Default::default()
    };
    let rendered_path = dir.path().join("rendered.mp4");
    let mut sink = FfmpegFileEncoder::new(&rendered_path, 64, 64, (30, 1), &config).unwrap();
    sink.consume(OutputFrame {
        data: &[128; 64 * 64 * 3 / 2],
        width: 64,
        height: 64,
        format: PixelFormat::Nv12,
        pts_us: Some(0),
    })
    .unwrap();
    sink.finish().unwrap();
    drop(sink);
    assert_matrix(&rendered_path, true);
    let bytes = std::fs::read(&rendered_path).unwrap();
    let colr = bytes
        .windows(8)
        .position(|tag| tag == b"colrnclx")
        .expect("container must carry colour information independently of encoder headers");
    assert_eq!(
        &bytes[colr + 8..colr + 15],
        &[0, 2, 0, 2, 0, 1, 0],
        "unspecified primaries/transfer, BT.709 matrix, limited range"
    );
    let raw_path = dir.path().join("raw.mp4");
    let mut encoder = VideoEncoder::new(&raw_path, 64, 64, (30, 1).into(), &config).unwrap();
    encoder
        .write_nv12_frame(&[128; 64 * 64 * 3 / 2], None)
        .unwrap();
    encoder.finish().unwrap();
    drop(encoder);
    assert_matrix(&raw_path, false);
}
