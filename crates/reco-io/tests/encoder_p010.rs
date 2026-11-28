//! Main 10 P010 delivery through the FFmpeg sink and a GPU stitch session.

#![cfg(feature = "ffmpeg")]

use std::path::Path;

use ffmpeg::color::{Primaries, Range, Space, TransferCharacteristic as Transfer};
use ffmpeg::format::Pixel;
use ffmpeg_next as ffmpeg;
use reco_core::sink::{OutputFrame, OutputSink, PixelFormat, SinkInput};
use reco_io::adapters::FfmpegFileEncoder;
use reco_io::ffmpeg::ColorMetadata;
use reco_io::ffmpeg::encoder::{EncodeError, EncoderConfig, VideoEncoder};
use reco_io::output::{BitDepth, Quality};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 128;
const PIXELS: usize = (WIDTH * HEIGHT) as usize;

fn config(transfer: Transfer) -> EncoderConfig {
    EncoderConfig {
        bit_depth: BitDepth::Ten,
        quality: Quality::Value(100),
        color_metadata: ColorMetadata {
            primaries: Some(Primaries::BT2020),
            transfer: Some(transfer),
            space: Some(Space::BT2020NCL),
        },
        ..Default::default()
    }
}

fn software_main10_available() -> bool {
    reco_io::init();
    let available = ffmpeg::encoder::find_by_name("libx265")
        .and_then(|codec| codec.video().ok())
        .and_then(|encoder| encoder.formats())
        .is_some_and(|mut formats| formats.any(|format| format == Pixel::YUV420P10LE));
    if !available {
        eprintln!("skipping software encode: linked libx265 lacks yuv420p10le");
    }
    available
}

#[test]
fn software_main10_sink_round_trips_hlg_and_pq() {
    if software_main10_available() {
        for transfer in [Transfer::ARIB_STD_B67, Transfer::SMPTE2084] {
            // Leave the codec and encoder name unset to exercise automatic HEVC selection.
            let config = EncoderConfig {
                software_only: true,
                ..config(transfer)
            };
            encode_p010(&config, "libx265", b"hvc1");
        }
    }
}

#[test]
#[ignore = "requires HEVC Main 10 capable NVENC hardware"]
fn automatic_main10_prefers_nvenc() {
    encode_p010(&config(Transfer::SMPTE2084), "hevc_nvenc", b"hvc1");
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires a Main 10 capable VAAPI render device in RECO_TEST_ENCODER_DEVICE"]
fn explicit_main10_vaapi_upload() {
    let device = std::env::var("RECO_TEST_ENCODER_DEVICE").expect("set RECO_TEST_ENCODER_DEVICE");
    let config = EncoderConfig {
        encoder_name: Some("hevc_vaapi".into()),
        device: Some(device),
        ..config(Transfer::SMPTE2084)
    };
    encode_p010(&config, "hevc_vaapi", b"hev1");
}

fn encode_p010(config: &EncoderConfig, encoder_name: &str, tag: &[u8; 4]) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("p010.mp4");
    let mut sink = FfmpegFileEncoder::new(&path, WIDTH, HEIGHT, (30, 1), config).unwrap();
    assert_eq!(sink.encoder_name(), encoder_name);
    assert!(matches!(
        sink.wants(),
        SinkInput::CpuBytes(PixelFormat::P010)
    ));
    let consume = |sink: &mut FfmpegFileEncoder, data: &[u8], format, pts_us| {
        sink.consume(OutputFrame {
            data,
            width: WIDTH,
            height: HEIGHT,
            pts_us,
            format,
        })
    };
    for format in [PixelFormat::Nv12, PixelFormat::Rgba8] {
        assert!(
            consume(&mut sink, &[], format, Some(0))
                .unwrap_err()
                .to_string()
                .contains("require P010LE")
        );
    }
    let uv = [p010_word(378), p010_word(675)].concat().repeat(PIXELS / 4);
    for (i, luma) in [512, 513].into_iter().enumerate() {
        let mut data = p010_word(luma).repeat(PIXELS);
        data.extend_from_slice(&uv);
        consume(&mut sink, &data, PixelFormat::P010, Some(i as i64 * 33_333)).unwrap();
    }
    sink.finish().unwrap();
    let decoded = decode_main10(&path, config.color_metadata.transfer.unwrap(), tag);
    assert_eq!(decoded.len(), 2);
    assert_eq!(
        decoded
            .iter()
            .map(|frame| center(frame, 0))
            .collect::<Vec<_>>(),
        [512, 513]
    );
    for frame in &decoded {
        assert!(center(frame, 1).abs_diff(378) <= 1);
        assert!(center(frame, 2).abs_diff(675) <= 1);
    }
}

#[test]
fn video_encoder_rejects_wrong_frame_depth_and_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("output.mp4");
    let sdr = EncoderConfig {
        software_only: true,
        ..Default::default()
    };
    let mut encoder = VideoEncoder::new(&path, WIDTH, HEIGHT, (30, 1).into(), &sdr).unwrap();
    assert!(matches!(
        encoder.write_p010_frame(&[], None),
        Err(EncodeError::InvalidConfig(_))
    ));
    drop(encoder);
    let sink = FfmpegFileEncoder::new(&path, WIDTH, HEIGHT, (30, 1), &sdr).unwrap();
    assert!(matches!(
        sink.wants(),
        SinkInput::CpuBytes(PixelFormat::Nv12)
    ));
    drop(sink);
    if software_main10_available() {
        let hdr = EncoderConfig {
            software_only: true,
            ..config(Transfer::SMPTE2084)
        };
        let mut encoder = VideoEncoder::new(&path, WIDTH, HEIGHT, (30, 1).into(), &hdr).unwrap();
        assert!(matches!(
            encoder.write_frame(&[], None),
            Err(EncodeError::InvalidConfig(_))
        ));
        assert!(matches!(
            encoder.write_nv12_frame(&[], None),
            Err(EncodeError::InvalidConfig(_))
        ));
        assert!(matches!(
            encoder.write_yuv420p_planes(&[], &[], &[], None),
            Err(EncodeError::InvalidConfig(_))
        ));
        assert!(
            matches!(encoder.write_p010_frame(&[], None), Err(EncodeError::FrameSizeMismatch { expected, actual: 0 }) if expected == PIXELS * 3)
        );
    }
}

fn p010_word(sample: u16) -> [u8; 2] {
    (sample << 6).to_le_bytes()
}

fn center(frame: &ffmpeg::frame::Video, plane: usize) -> u16 {
    let offset = frame.plane_height(plane) as usize / 2 * frame.stride(plane)
        + frame.plane_width(plane) as usize;
    u16::from_le_bytes(frame.data(plane)[offset..offset + 2].try_into().unwrap())
}

fn decode_main10(path: &Path, transfer: Transfer, tag: &[u8; 4]) -> Vec<ffmpeg::frame::Video> {
    let mut input = ffmpeg::format::input(path).unwrap();
    let stream = input.streams().best(ffmpeg::media::Type::Video).unwrap();
    let index = stream.index();
    let mut parameters = stream.parameters().clone();
    assert_eq!(parameters.id(), ffmpeg::codec::Id::HEVC);
    // SAFETY: parameters owns an independent copy of the stream's AVCodecParameters.
    unsafe {
        let raw = &mut *parameters.as_mut_ptr();
        assert_eq!(raw.codec_tag, u32::from_le_bytes(*tag));
        assert_eq!(Primaries::from(raw.color_primaries), Primaries::BT2020);
        assert_eq!(Transfer::from(raw.color_trc), transfer);
        assert_eq!(Space::from(raw.color_space), Space::BT2020NCL);
        assert_eq!(Range::from(raw.color_range), Range::MPEG);
        // Require the decoded tags to come from the HEVC VUI.
        raw.color_primaries = Primaries::Unspecified.into();
        raw.color_trc = Transfer::Unspecified.into();
        raw.color_space = Space::Unspecified.into();
        raw.color_range = Range::Unspecified.into();
    }
    let mut decoder = ffmpeg::codec::context::Context::from_parameters(parameters)
        .unwrap()
        .decoder()
        .video()
        .unwrap();
    let mut frames = Vec::new();
    let mut drain = |decoder: &mut ffmpeg::decoder::Video| loop {
        let mut frame = ffmpeg::frame::Video::empty();
        match decoder.receive_frame(&mut frame) {
            Ok(()) => {
                assert_eq!(frame.format(), Pixel::YUV420P10LE);
                assert_eq!(frame.color_primaries(), Primaries::BT2020);
                assert_eq!(frame.color_transfer_characteristic(), transfer);
                assert_eq!(frame.color_space(), Space::BT2020NCL);
                assert_eq!(frame.color_range(), Range::MPEG);
                frames.push(frame);
            }
            Err(
                ffmpeg::Error::Eof
                | ffmpeg::Error::Other {
                    errno: ffmpeg::error::EAGAIN,
                },
            ) => break,
            Err(error) => panic!("decode failed: {error}"),
        }
    };
    for (stream, packet) in input.packets() {
        if stream.index() == index {
            decoder.send_packet(&packet).unwrap();
            drain(&mut decoder);
        }
    }
    decoder.send_eof().unwrap();
    drain(&mut decoder);
    assert_eq!(
        decoder.profile(),
        ffmpeg::codec::Profile::HEVC(ffmpeg::codec::profile::HEVC::Main10)
    );
    frames
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_stitch_delivers_main10_without_losing_adjacent_luma_codes() {
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::geometry::RenderPose;
    use reco_core::gpu::{GpuContext, OutputFormat};
    use reco_core::projection::Cylinder;
    use reco_core::render::color::{YuvColor, YuvMatrix};
    use reco_core::render::renderer::{GpuPixelFormat, InputFormat};
    use reco_core::render::viewport::ViewportSize;
    use reco_core::session::{SinkOptions, StitchSession};
    use reco_core::wgpu;

    if !software_main10_available() {
        return;
    }
    let Ok(gpu) = GpuContext::new_blocking() else {
        eprintln!("skipping stitch encode: no GPU context");
        return;
    };
    if !gpu
        .device()
        .features()
        .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
    {
        eprintln!("skipping stitch encode: no P010 input texture support");
        return;
    }
    let calibration = Calibration::new(
        vec![Lens::flat(WIDTH, HEIGHT)],
        Cylinder {
            radius: 20.0,
            arc_deg: 180.0,
            surface_height: None,
        },
        Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stitched.mp4");
    let transfer = Transfer::SMPTE2084;
    let config = EncoderConfig {
        software_only: true,
        ..config(transfer)
    };
    let sink = FfmpegFileEncoder::new(&path, WIDTH, HEIGHT, (30, 1), &config).unwrap();
    let mut session = StitchSession::builder()
        .gpu(gpu.clone())
        .calibration(calibration)
        .source_sizes(vec![(WIDTH, HEIGHT)])
        .viewport_size(ViewportSize {
            width: WIDTH,
            height: HEIGHT,
        })
        .output_format(OutputFormat::Rgb10a2Unorm)
        .input_format(InputFormat::Nv12)
        .sink(Box::new(sink), SinkOptions::inline_lossy())
        .build()
        .unwrap();
    session
        .core_mut()
        .set_source_color(YuvColor::delivery(YuvMatrix::Bt2020Ncl));

    let texture = |width, height, format, data: &[u8]| {
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("P010 test plane"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue().write_texture(
            texture.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH * 2),
                rows_per_image: None,
            },
            texture.size(),
        );
        texture.create_view(&Default::default())
    };
    let format = GpuPixelFormat::P010;
    let uv = texture(
        WIDTH / 2,
        HEIGHT / 2,
        format.uv_format(),
        &p010_word(512).repeat(PIXELS / 2),
    );
    for (i, code) in [512, 513].into_iter().enumerate() {
        let y = texture(
            WIDTH,
            HEIGHT,
            format.y_format(),
            &p010_word(code).repeat(PIXELS),
        );
        session
            .render_frame_imported_nv12(
                &[(&y, &uv)],
                RenderPose::default(),
                Some(i as i64 * 33_333),
            )
            .unwrap();
    }
    session.finish().unwrap();
    let decoded = decode_main10(&path, transfer, b"hvc1");
    assert_eq!(
        decoded
            .iter()
            .map(|frame| center(frame, 0))
            .collect::<Vec<_>>(),
        [512, 513]
    );
}
