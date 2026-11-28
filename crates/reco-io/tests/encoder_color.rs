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

fn assert_color_metadata(
    path: &std::path::Path,
    expected: reco_io::ffmpeg::ColorMetadata,
    expected_full_range: bool,
) {
    use ffmpeg_next::color::{Primaries, Range, Space, TransferCharacteristic};

    let decoder = VideoDecoder::open_software(path).unwrap();
    assert_eq!(decoder.color_metadata(), expected, "{}", path.display());
    assert_eq!(
        decoder.is_full_range(),
        expected_full_range,
        "{}",
        path.display()
    );
    drop(decoder);

    let mut input = ffmpeg_next::format::input(path).unwrap();
    let stream = input
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .unwrap();
    let index = stream.index();
    let parameters = stream.parameters();
    // SAFETY: parameters remains alive while its fields are inspected.
    let raw = unsafe { &*parameters.as_ptr() };
    assert_eq!(
        Primaries::from(raw.color_primaries),
        expected.primaries.unwrap_or(Primaries::Unspecified),
        "stream primaries for {}",
        path.display()
    );
    assert_eq!(
        TransferCharacteristic::from(raw.color_trc),
        expected
            .transfer
            .unwrap_or(TransferCharacteristic::Unspecified),
        "stream transfer for {}",
        path.display()
    );
    assert_eq!(
        Space::from(raw.color_space),
        expected.space.unwrap_or(Space::Unspecified),
        "stream space for {}",
        path.display()
    );
    assert_eq!(
        Range::from(raw.color_range),
        if expected_full_range {
            Range::JPEG
        } else {
            Range::MPEG
        },
        "stream range for {}",
        path.display()
    );

    let mut codec_decoder =
        ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .unwrap()
            .decoder()
            .video()
            .unwrap();
    let mut frame = ffmpeg_next::frame::Video::empty();
    for (stream, packet) in input.packets() {
        if stream.index() != index {
            continue;
        }
        codec_decoder.send_packet(&packet).unwrap();
        if codec_decoder.receive_frame(&mut frame).is_ok() {
            break;
        }
    }
    if frame.width() == 0 {
        codec_decoder.send_eof().unwrap();
        codec_decoder.receive_frame(&mut frame).unwrap();
    }
    assert_eq!(
        frame.color_primaries(),
        expected.primaries.unwrap_or(Primaries::Unspecified),
        "decoded frame primaries for {}",
        path.display()
    );
    assert_eq!(
        frame.color_transfer_characteristic(),
        expected
            .transfer
            .unwrap_or(TransferCharacteristic::Unspecified),
        "decoded frame transfer for {}",
        path.display()
    );
    assert_eq!(
        frame.color_space(),
        expected.space.unwrap_or(Space::Unspecified),
        "decoded frame space for {}",
        path.display()
    );
    assert_eq!(
        frame.color_range(),
        if expected_full_range {
            Range::JPEG
        } else {
            Range::MPEG
        },
        "decoded frame range for {}",
        path.display()
    );
}

#[test]
fn hdr_hlg_and_pq_bt2020_metadata_roundtrips_for_h264_and_hevc() {
    use ffmpeg_next::color::{Primaries, Space, TransferCharacteristic};
    use reco_io::ffmpeg::ColorMetadata;

    let cases = [
        (
            "hlg",
            ColorMetadata {
                primaries: Some(Primaries::BT2020),
                transfer: Some(TransferCharacteristic::ARIB_STD_B67),
                space: Some(Space::BT2020NCL),
            },
            [0u8, 9, 0, 18, 0, 9, 0],
        ),
        (
            "pq",
            ColorMetadata {
                primaries: Some(Primaries::BT2020),
                transfer: Some(TransferCharacteristic::SMPTE2084),
                space: Some(Space::BT2020NCL),
            },
            [0u8, 9, 0, 16, 0, 9, 0],
        ),
    ];

    let targets = [
        ("libx265", Container::Mp4),
        ("libx265", Container::Mp4Fragmented),
        ("libx265", Container::Mov),
        ("libx265", Container::Mkv),
        ("libx264", Container::Mp4),
    ];

    let dir = tempfile::tempdir().unwrap();
    for (label, metadata, expected_nclx) in cases {
        for (encoder_name, container) in targets {
            let path = dir
                .path()
                .join(format!("{label}-{encoder_name}-{container:?}.bin"));
            let config = EncoderConfig {
                encoder_name: Some(encoder_name.into()),
                container,
                gop_size: Some(2),
                color_metadata: metadata,
                ..Default::default()
            };
            let mut sink = FfmpegFileEncoder::new(&path, 64, 64, (30, 1), &config).unwrap();
            for index in 0..3 {
                sink.consume(OutputFrame {
                    data: &[128; 64 * 64 * 3 / 2],
                    width: 64,
                    height: 64,
                    format: PixelFormat::Nv12,
                    pts_us: Some(index * 1_000_000 / 30),
                })
                .unwrap();
            }
            sink.finish().unwrap();
            drop(sink);

            assert_color_metadata(&path, metadata, false);

            match container {
                Container::Mp4 | Container::Mp4Fragmented => {
                    let bytes = std::fs::read(&path).unwrap();
                    let colr = bytes
                        .windows(8)
                        .position(|tag| tag == b"colrnclx")
                        .expect("MP4 container must carry colr nclx box");
                    assert_eq!(
                        &bytes[colr + 8..colr + 15],
                        &expected_nclx,
                        "{label} {encoder_name} {container:?}: unexpected colr nclx bytes"
                    );
                }
                Container::Mov => {
                    let bytes = std::fs::read(&path).unwrap();
                    let colr = bytes
                        .windows(8)
                        .position(|tag| tag == b"colrnclc")
                        .expect("MOV container must carry colr nclc box");
                    assert_eq!(
                        &bytes[colr + 8..colr + 14],
                        &expected_nclx[..6],
                        "{label} {encoder_name} {container:?}: unexpected colr nclc bytes"
                    );
                }
                _ => {}
            }
        }
    }
}

#[test]
fn stitch_job_exports_bt2020_pixels_and_rejects_mixed_hdr_sdr() {
    use ffmpeg_next::color::{Primaries, Space, TransferCharacteristic};
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::projection::LShape;
    use reco_io::StitchJob;
    use reco_io::ffmpeg::ColorMetadata;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    let hlg_bt2020 = ColorMetadata {
        primaries: Some(Primaries::BT2020),
        transfer: Some(TransferCharacteristic::ARIB_STD_B67),
        space: Some(Space::BT2020NCL),
    };
    let write_fixture = |path: &std::path::Path, encoder_name: &str, color_metadata| {
        let config = EncoderConfig {
            encoder_name: Some(encoder_name.into()),
            color_metadata,
            ..Default::default()
        };
        let mut sink = FfmpegFileEncoder::new(path, 64, 64, (30, 1), &config).unwrap();
        // This YCbCr triplet maps outside RGB gamut. Clamping makes an incorrect
        // input matrix observable even when the output repeats the same mistake.
        let mut samples = vec![168; 64 * 64];
        samples.extend([224, 16].repeat(32 * 32));
        for index in 0..5 {
            sink.consume(OutputFrame {
                data: &samples,
                width: 64,
                height: 64,
                format: PixelFormat::Nv12,
                pts_us: Some(index * 1_000_000 / 30),
            })
            .unwrap();
        }
        sink.finish().unwrap();
    };

    let calibration = || {
        let lens = Lens::fisheye(64, 64, 32.0, 32.0, 32.0, 32.0, [0.0; 4]);
        Calibration::new(
            vec![lens.clone(), lens],
            LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            Framing {
                axis_offset: 0.25,
                tilt: 0.0,
                roll: 0.0,
            },
        )
    };

    let dir = tempfile::tempdir().unwrap();
    let hdr_in = dir.path().join("hdr_hlg_in.mp4");
    let sdr_in = dir.path().join("sdr_in.mp4");
    write_fixture(&hdr_in, "libx264", hlg_bt2020);
    write_fixture(&sdr_in, "libx264", ColorMetadata::default());

    #[cfg(feature = "gpu")]
    if reco_core::gpu::GpuContext::new_blocking().is_ok() {
        let output = dir.path().join("stitched-hdr.mp4");
        StitchJob::with_calibration([hdr_in.clone(), hdr_in.clone()], calibration(), &output)
            .force_cpu_decode()
            .resolution(64, 64)
            .encoder_name("libx265")
            .quality(reco_io::output::Quality::Value(100))
            .max_frames(4)
            .on_session(|session, _| {
                session.core_mut().set_constrained_look(false);
                Ok(())
            })
            .run(&Arc::new(AtomicBool::new(false)))
            .unwrap();
        assert_color_metadata(&output, hlg_bt2020, false);
        let mut decoder = VideoDecoder::open_software(&output).unwrap();
        let frame = decoder.next_frame().unwrap().unwrap();
        // BT.2020 NCL maps [168, 224, 16] to clamped 8-bit RGB [0, 232, 255].
        // Re-encoding produces [164, 167, 25]. BT.709 on both legs yields Cr=32.
        let actual = [
            frame.y[32 * 64 + 32],
            frame.u[16 * 32 + 16],
            frame.v[16 * 32 + 16],
        ];
        let expected = [164, 167, 25];
        assert!(
            actual
                .into_iter()
                .zip(expected)
                .all(|(a, b)| a.abs_diff(b) <= 2),
            "{actual:?} vs {expected:?}"
        );
    }

    let mismatched_out = dir.path().join("stitched-mismatch.mp4");
    let error = StitchJob::with_calibration(
        [hdr_in.clone(), sdr_in.clone()],
        calibration(),
        &mismatched_out,
    )
    .cpu()
    .run(&Arc::new(AtomicBool::new(false)))
    .unwrap_err();
    assert!(
        error.to_string().contains("camera color metadata differ"),
        "{error}"
    );
    assert!(!mismatched_out.exists());
}

fn write_color_fixture(path: &std::path::Path, color_args: &[&str]) {
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=s=64x64:r=30",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-bf",
            "0",
        ])
        .args(color_args)
        .arg(path)
        .output()
        .expect("ffmpeg fixture encoder");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

const HLG_TAGS: &[&str] = &[
    "-colorspace",
    "bt2020nc",
    "-color_primaries",
    "bt2020",
    "-color_trc",
    "arib-std-b67",
    "-color_range",
    "tv",
];

const BT709_TAGS: &[&str] = &[
    "-colorspace",
    "bt709",
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
    "-color_range",
    "tv",
];

const UNSPECIFIED_H264_COLOR: &str = "h264_metadata=video_format=0:colour_primaries=2:transfer_characteristics=2:matrix_coefficients=2";

#[test]
fn concat_rejects_incompatible_segment_color_before_decoding() {
    use reco_io::{InputPath, SmartFileSource};

    let dir = tempfile::tempdir().unwrap();
    let hdr = dir.path().join("hdr.mp4");
    let untagged = dir.path().join("untagged.mp4");
    let full_range = dir.path().join("full-range.mp4");
    write_color_fixture(&hdr, HLG_TAGS);
    write_color_fixture(&untagged, &[]);
    write_color_fixture(&full_range, &["-color_range", "pc"]);

    let untagged_probe = VideoDecoder::open_software(&untagged).unwrap();
    assert!(untagged_probe.color_metadata().is_empty());
    assert!(!untagged_probe.is_full_range());
    assert!(
        VideoDecoder::open_software(&full_range)
            .unwrap()
            .is_full_range()
    );
    #[cfg(feature = "gpu")]
    let gpu = reco_core::gpu::GpuContext::new_blocking().ok();
    for (paths, expected) in [
        ([hdr, untagged.clone()], "segment color metadata differ"),
        ([untagged, full_range], "segment color ranges differ"),
    ] {
        let input = InputPath::Chained(paths.to_vec());
        let decoder_error = VideoDecoder::open_input_software(&input)
            .err()
            .expect("concat open");
        assert!(
            decoder_error.to_string().contains(expected),
            "{decoder_error}"
        );
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &gpu {
            let source_error = SmartFileSource::open(std::slice::from_ref(&input), gpu, 0)
                .err()
                .expect("smart source open");
            assert!(
                source_error.to_string().contains(expected),
                "{source_error}"
            );
        }
        let source_error = SmartFileSource::open_cpu_only(&[input], 0, true)
            .err()
            .expect("source open");
        assert!(
            source_error.to_string().contains(expected),
            "{source_error}"
        );
    }
}

#[test]
fn ordinary_and_stacked_files_reject_midstream_color_changes() {
    assert_midstream_color_change(BT709_TAGS);
}

#[test]
fn ordinary_and_stacked_files_reject_midstream_unspecified_color() {
    // An SPS without colour description can leave the previous tags in FFmpeg.
    // Write explicit H.264 "unspecified" values to clear all three tags.
    assert_midstream_color_change(&["-bsf:v", UNSPECIFIED_H264_COLOR]);
}

fn assert_midstream_color_change(second_segment_args: &[&str]) {
    use reco_io::stacked_video::{GridLayout, source::StackedSource};

    let dir = tempfile::tempdir().unwrap();
    let hdr = dir.path().join("hdr.ts");
    let second = dir.path().join("second.ts");
    let combined = dir.path().join("transition.ts");
    write_color_fixture(&hdr, HLG_TAGS);
    let mut args = second_segment_args.to_vec();
    args.extend(["-output_ts_offset", "2"]);
    write_color_fixture(&second, &args);
    join_color_segments(&hdr, &second, &combined);

    let mut decoder = VideoDecoder::open_software(&combined).unwrap();
    assert!(decoder.color_metadata().is_hdr());
    let layout = GridLayout::vstack(64, 32, 2).unwrap();
    let mut stacked = StackedSource::open(layout, &combined).unwrap();
    for _ in 0..60 {
        assert!(decoder.next_frame().unwrap().is_some());
        assert!(stacked.next_tuple().unwrap().is_some());
    }

    let decoder_error = decoder.next_frame().err();
    let stacked_error = stacked.next_tuple().err();
    assert!(
        decoder_error
            .as_ref()
            .is_some_and(|error| error.to_string().contains("decoded color metadata changed"))
            && stacked_error
                .as_ref()
                .is_some_and(|error| error.to_string().contains("decoded color metadata changed")),
        "expected both readers to reject frame 61: decoder={decoder_error:?}, stacked={stacked_error:?}",
    );
}

fn join_color_segments(
    first: &std::path::Path,
    second: &std::path::Path,
    combined: &std::path::Path,
) {
    let mut stream = std::fs::read(first).unwrap();
    stream.extend(std::fs::read(second).unwrap());
    std::fs::write(combined, stream).unwrap();
}

#[test]
fn seek_clears_frame_color_history() {
    let dir = tempfile::tempdir().unwrap();
    let untagged = dir.path().join("untagged.ts");
    let tagged = dir.path().join("bt709.ts");
    let combined = dir.path().join("transition.ts");
    write_color_fixture(
        &untagged,
        &["-bsf:v", UNSPECIFIED_H264_COLOR, "-muxdelay", "0"],
    );
    let mut args = BT709_TAGS.to_vec();
    args.extend(["-muxdelay", "0", "-output_ts_offset", "2"]);
    write_color_fixture(&tagged, &args);
    join_color_segments(&untagged, &tagged, &combined);

    let mut decoder = VideoDecoder::open_software(&combined).unwrap();
    assert!(decoder.color_metadata().is_empty());
    for _ in 0..120 {
        assert!(decoder.next_frame().unwrap().is_some());
    }
    assert!(decoder.next_frame().unwrap().is_none());

    decoder.seek_to_secs(0.0).unwrap();
    let frame = decoder
        .next_frame()
        .unwrap()
        .expect("first frame after seek");
    assert_eq!(frame.timestamp_us, 0);
}

#[test]
fn container_only_color_tags_keep_decoding() {
    use ffmpeg_next::color::{Primaries, Space, TransferCharacteristic};
    use reco_io::ffmpeg::ColorMetadata;
    use reco_io::stacked_video::{GridLayout, source::StackedSource};

    let dir = tempfile::tempdir().unwrap();
    let untagged = dir.path().join("untagged.mp4");
    write_color_fixture(&untagged, &[]);
    assert!(
        VideoDecoder::open_software(&untagged)
            .unwrap()
            .color_metadata()
            .is_empty()
    );
    let expected = ColorMetadata {
        primaries: Some(Primaries::BT2020),
        transfer: Some(TransferCharacteristic::ARIB_STD_B67),
        space: Some(Space::BT2020NCL),
    };

    for extension in ["mp4", "mkv"] {
        let path = dir.path().join(format!("container-only.{extension}"));
        // Stream copy keeps the SPS untagged while the muxer writes colour tags.
        let output = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(&untagged)
            .args(["-c:v", "copy"])
            .args(HLG_TAGS)
            .arg(&path)
            .output()
            .expect("ffmpeg fixture remuxer");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let mut decoder = VideoDecoder::open_software(&path).unwrap();
        assert_eq!(decoder.color_metadata(), expected, "{extension}");
        let layout = GridLayout::vstack(64, 32, 2).unwrap();
        let mut stacked = StackedSource::open(layout, &path).unwrap();
        for _ in 0..60 {
            assert!(decoder.next_frame().unwrap().is_some(), "{extension}");
            assert!(stacked.next_tuple().unwrap().is_some(), "{extension}");
        }
        assert!(decoder.next_frame().unwrap().is_none(), "{extension}");
        assert!(stacked.next_tuple().unwrap().is_none(), "{extension}");
    }
}

#[test]
fn rendered_partial_hdr_uses_bt2020_matrix_when_primaries_are_bt2020() {
    use ffmpeg_next::color::{Primaries, Space, TransferCharacteristic};
    use reco_io::ffmpeg::ColorMetadata;

    let partial_hdr = ColorMetadata {
        primaries: Some(Primaries::BT2020),
        transfer: Some(TransferCharacteristic::ARIB_STD_B67),
        space: None,
    };

    let dir = tempfile::tempdir().unwrap();
    for encoder_name in ["libx265", "libx264"] {
        let path = dir.path().join(format!("partial-hdr-{encoder_name}.mp4"));
        let config = EncoderConfig {
            encoder_name: Some(encoder_name.into()),
            gop_size: Some(2),
            color_metadata: partial_hdr,
            ..Default::default()
        };
        let mut sink = FfmpegFileEncoder::new(&path, 64, 64, (30, 1), &config).unwrap();
        for index in 0..3 {
            sink.consume(OutputFrame {
                data: &[128; 64 * 64 * 3 / 2],
                width: 64,
                height: 64,
                format: PixelFormat::Nv12,
                pts_us: Some(index * 1_000_000 / 30),
            })
            .unwrap();
        }
        sink.finish().unwrap();
        drop(sink);

        assert_color_metadata(
            &path,
            ColorMetadata {
                space: Some(Space::BT2020NCL),
                ..partial_hdr
            },
            false,
        );
    }
}

#[test]
fn ffmpeg_rgba_scaler_uses_the_tagged_output_matrix() {
    use ffmpeg_next::color::Space;
    use reco_io::ffmpeg::ColorMetadata;
    let dir = tempfile::tempdir().unwrap();
    for (space, expected) in [
        (Space::BT709, [63, 102, 240]),
        (Space::BT2020NCL, [74, 97, 240]),
        (Space::SMPTE170M, [81, 90, 240]),
        (Space::BT470BG, [81, 90, 240]),
    ] {
        let path = dir.path().join(format!("{space:?}.mp4"));
        let config = EncoderConfig {
            encoder_name: Some("libx264".into()),
            quality: reco_io::output::Quality::Value(100),
            color_metadata: ColorMetadata {
                space: Some(space),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut encoder = FfmpegFileEncoder::new(&path, 64, 64, (30, 1), &config).unwrap();
        let rgba = [255, 0, 0, 255].repeat(64 * 64);
        encoder
            .consume(OutputFrame {
                data: &rgba,
                width: 64,
                height: 64,
                format: PixelFormat::Rgba8,
                pts_us: Some(0),
            })
            .unwrap();
        encoder.finish().unwrap();
        drop(encoder);
        let mut decoder = VideoDecoder::open_software(&path).unwrap();
        assert_eq!(decoder.color_metadata().space, Some(space));
        assert!(!decoder.is_full_range());
        let frame = decoder.next_frame().unwrap().unwrap();
        let actual = [frame.y[0], frame.u[0], frame.v[0]];
        assert!(
            actual
                .into_iter()
                .zip(expected)
                .all(|(a, b)| a.abs_diff(b) <= 2),
            "{space:?}: {actual:?} vs {expected:?}"
        );
    }
}
