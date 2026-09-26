//! Finite-input integrity across extraction and worker boundaries.

#![cfg(feature = "ffmpeg")]

use ffmpeg_next as ffmpeg;
use reco_core::source::{FrameSet, FrameSource};
use reco_io::InputPath;
use reco_io::adapters::FfmpegFileSource;
use reco_io::ffmpeg::decoder::{DecodeBackend, DecodeError, VideoDecoder};

const FRAMES: usize = 48;
const WIDTH: u32 = 128;
const HEIGHT: u32 = 96;

fn fixture(path: &std::path::Path) {
    fixture_encoding(path, "libx264");
}

#[test]
fn job_setup_and_completion_are_explicit() {
    use reco_core::session::types::RunCompletion;
    use reco_io::stitch_job::StitchError;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.mp4");
    fixture(&input);
    let output = dir.path().join("output.mp4");
    let job = || {
        let lens =
            reco_core::calibration::Lens::fisheye(WIDTH, HEIGHT, 64.0, 64.0, 64.0, 48.0, [0.0; 4]);
        let cal = reco_core::calibration::Calibration::new(
            vec![lens.clone(), lens],
            reco_core::projection::LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            reco_core::calibration::Framing {
                axis_offset: 0.25,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        reco_io::StitchJob::with_calibration([input.clone(), input.clone()], cal, &output)
            .cpu()
            .resolution(64, 64)
            .audio(reco_io::output::AudioMode::Disabled)
            .encoder_name("libx264")
            .on_session(|session, _| {
                session.core_mut().set_constrained_look(false);
                Ok(())
            })
    };
    let error = job()
        .on_session(|_, _| Err(StitchError::Other("setup failure".into())))
        .on_session(|_, _| panic!("must not run after failure"))
        .run(&AtomicBool::new(false))
        .unwrap_err();
    assert_eq!(error.to_string(), "setup failure");
    assert!(!output.exists(), "failed setup must not create an output");

    for cancel_during_frames in [true, false] {
        let flag = Arc::new(AtomicBool::new(false));
        let progress_flag = flag.clone();
        let finish_flag = flag.clone();
        let result = job()
            .max_frames(6)
            .on_progress(move |p| {
                if cancel_during_frames && p.frames_completed == 3 {
                    progress_flag.store(true, Ordering::Relaxed);
                }
            })
            .on_finalizing(move || finish_flag.store(true, Ordering::Relaxed))
            .run(&flag)
            .unwrap();
        let expected = if cancel_during_frames { 3 } else { 6 };
        assert_eq!(result.frames_processed, expected);
        assert_eq!(
            result.completion,
            if cancel_during_frames {
                RunCompletion::Cancelled
            } else {
                RunCompletion::Completed
            }
        );
        let mut decoded = VideoDecoder::open_software(&output).unwrap();
        let mut count = 0;
        while decoded.next_frame().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(
            count, expected,
            "partial output must be finalized and decodable"
        );
    }

    struct FailingFinish;
    impl reco_core::sink::OutputSink for FailingFinish {
        fn name(&self) -> &str {
            "failing-finish"
        }
        fn wants(&self) -> reco_core::sink::SinkInput {
            reco_core::sink::SinkInput::CpuBytes(reco_core::sink::PixelFormat::Nv12)
        }
        fn consume(
            &mut self,
            _: reco_core::sink::OutputFrame<'_>,
        ) -> Result<(), reco_core::sink::SinkError> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), reco_core::sink::SinkError> {
            Err(reco_core::sink::SinkError::Finish {
                reason: "finalization failure".into(),
            })
        }
    }
    let flag = Arc::new(AtomicBool::new(false));
    let signal = flag.clone();
    let result = job()
        .on_session(|session, _| {
            session.add_sink(
                Box::new(FailingFinish),
                reco_core::session::SinkOptions::threaded(1),
            )?;
            Ok(())
        })
        .on_progress(move |_| signal.store(true, Ordering::Relaxed))
        .run(&flag);
    assert!(flag.load(Ordering::Relaxed));
    assert!(matches!(result, Err(StitchError::Session(_))), "{result:?}");
}

fn fixture_encoding(path: &std::path::Path, codec_name: &str) {
    fixture_size(path, codec_name, WIDTH, HEIGHT);
}

fn fixture_size(path: &std::path::Path, codec_name: &str, width: u32, height: u32) {
    ffmpeg::init().unwrap();
    let codec = ffmpeg::encoder::find_by_name(codec_name).expect("test encoder");
    let mut output = ffmpeg::format::output(path).unwrap();
    let mut stream = output.add_stream(codec).unwrap();
    let mut encoder = ffmpeg::codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()
        .unwrap();
    encoder.set_width(width);
    encoder.set_height(height);
    encoder.set_format(ffmpeg::format::Pixel::YUV420P);
    encoder.set_time_base((1, 30));
    encoder.set_frame_rate(Some((30, 1)));
    encoder.set_gop(48);
    encoder.set_max_b_frames(if codec_name == "libx264" { 3 } else { 0 });
    encoder.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
    stream.set_parameters(&encoder);
    let mut options = ffmpeg::Dictionary::new();
    if codec_name == "libx264" {
        options.set("crf", "10");
        options.set("x264-params", "b-adapt=0:scenecut=0");
    }
    let mut encoder = encoder.open_with(options).unwrap();
    stream.set_parameters(&encoder);
    output.write_header().unwrap();
    let time_base = output.stream(0).unwrap().time_base();

    let drain = |encoder: &mut ffmpeg::encoder::video::Encoder,
                 output: &mut ffmpeg::format::context::Output| {
        loop {
            let mut packet = ffmpeg::Packet::empty();
            match encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_stream(0);
                    packet.set_duration(1);
                    packet.rescale_ts((1, 30), time_base);
                    packet.write_interleaved(output).unwrap();
                }
                Err(ffmpeg::Error::Eof) => break,
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => break,
                Err(error) => panic!("encode: {error}"),
            }
        }
    };
    for i in 0..FRAMES {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, width, height);
        frame.data_mut(0).fill(32 + i as u8 * 4);
        frame.data_mut(1).fill(96);
        frame.data_mut(2).fill(160);
        frame.set_pts(Some(i as i64));
        encoder.send_frame(&frame).unwrap();
        drain(&mut encoder, &mut output);
    }
    encoder.send_eof().unwrap();
    drain(&mut encoder, &mut output);
    output.write_trailer().unwrap();
}

fn assert_pixels(y: &[u8], u: &[u8], v: &[u8], index: usize) {
    for (plane, expected) in [(y, 32 + (index % FRAMES) as u8 * 4), (u, 96), (v, 160)] {
        assert!(
            plane.iter().all(|value| value.abs_diff(expected) <= 2),
            "frame {index}: expected plane value {expected}"
        );
    }
}

fn drain_cpu(decoder: &mut VideoDecoder, count: usize) {
    for index in 0..count {
        let frame = decoder.next_frame().unwrap().expect("missing frame");
        assert_eq!((frame.width, frame.height), (WIDTH, HEIGHT));
        assert_eq!(frame.timestamp_us, index as i64 * 1_000_000 / 30);
        assert_pixels(&frame.y, &frame.u, &frame.v, index);
    }
    for _ in 0..3 {
        assert!(decoder.next_frame().unwrap().is_none());
    }
}

#[test]
fn reordered_frames_drain_and_seek_without_loss() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    for input in [
        InputPath::Single(path.clone()),
        InputPath::Chained(vec![path.clone(), path.clone()]),
    ] {
        let count = if matches!(input, InputPath::Chained(_)) {
            FRAMES * 2
        } else {
            FRAMES
        };
        let mut decoder = VideoDecoder::open_input_software(&input).unwrap();
        drain_cpu(&mut decoder, count);
        decoder.seek_to_secs(0.0).unwrap();
        drain_cpu(&mut decoder, count);
    }
}

#[test]
fn wrong_backend_is_error_without_consuming_a_frame() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    let mut decoder = VideoDecoder::open_software(&path).unwrap();
    assert!(matches!(
        decoder.next_frame_gpu(),
        Err(DecodeError::BackendMismatch { .. })
    ));
    drain_cpu(&mut decoder, FRAMES);
}

#[test]
fn missing_later_segment_is_a_read_error_not_successful_eof() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    let input = InputPath::Chained(vec![path, dir.path().join("missing.mp4")]);
    let mut decoder = VideoDecoder::open_input_software(&input).unwrap();
    let mut received = 0;
    loop {
        match decoder.next_frame() {
            Ok(Some(_)) => received += 1,
            Ok(None) => panic!("missing segment reported as EOF"),
            Err(DecodeError::Ffmpeg(_)) => break,
            Err(error) => panic!("unexpected error: {error}"),
        }
    }
    assert!(received > 0, "must exercise a mid-stream failure");
    let mut source = FfmpegFileSource::open_inputs(std::slice::from_ref(&input), 0, true).unwrap();
    let mut delivered = 0;
    loop {
        match source.next_frame() {
            Ok(Some(_)) => delivered += 1,
            Ok(None) => panic!("worker failure reported as EOF"),
            Err(_) => break,
        }
    }
    assert_eq!(delivered, received);
    assert!(!source.is_exhausted());

    // The absent segment prevents seeking into it. The respawned worker
    // must expose that seek failure, not disconnect as successful EOF.
    assert!(decoder.seek_to_secs(1000.0 / 30.0).is_err());
    source.seek(1000).unwrap();
    assert!(source.next_frame().is_err());
    assert!(!source.is_exhausted());
    source.seek(0).unwrap();
    assert!(source.next_frame().unwrap().is_some());

    let lens =
        reco_core::calibration::Lens::fisheye(WIDTH, HEIGHT, 64.0, 64.0, 64.0, 48.0, [0.0; 4]);
    let calibration = reco_core::calibration::Calibration::new(
        vec![lens.clone(), lens],
        reco_core::projection::LShape {
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        },
        reco_core::calibration::Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let result = reco_io::StitchJob::with_calibration(
        [input.clone(), input],
        calibration,
        dir.path().join("out.mp4"),
    )
    .cpu()
    .resolution(64, 64)
    .audio(reco_io::output::AudioMode::Disabled)
    .encoder_name("libx264")
    .run(&std::sync::atomic::AtomicBool::new(false));
    assert!(
        matches!(
            result,
            Err(reco_io::stitch_job::StitchError::Session(
                reco_core::session::types::SessionError::Source(_)
            ))
        ),
        "{result:?}"
    );
}

#[test]
fn file_source_delivers_every_frame_and_persists_eof() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    let input = InputPath::Single(path);
    for inputs in [vec![input.clone()], vec![input.clone(); 2], vec![input; 3]] {
        let mut source = FfmpegFileSource::open_inputs(&inputs, 0, true).unwrap();
        assert_eq!(source.decode_backend(), DecodeBackend::Software);
        for index in 0..FRAMES {
            let FrameSet::Yuv420p(cameras) = source.next_frame().unwrap().unwrap() else {
                panic!("expected CPU frames");
            };
            assert_eq!(cameras.len(), inputs.len());
            for frame in cameras {
                assert_pixels(&frame.y, &frame.u, &frame.v, index);
            }
        }
        assert!(source.next_frame().unwrap().is_none());
        assert!(source.is_exhausted());
        assert!(source.next_frame().unwrap().is_none());
        assert!(source.try_next_frame().unwrap().is_none());
        source.seek(0).unwrap();
        assert!(source.next_frame().unwrap().is_some());
        source.seek(FRAMES as u64 + 1).unwrap();
        assert!(source.next_frame().unwrap().is_none());
        assert!(source.is_exhausted());
    }
}

/// Opt-in hardware gate: fails rather than silently skipping when CUDA is absent.
#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC"]
fn cuda_drains_reordered_frames_and_seeks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    for input in [
        InputPath::Single(path.clone()),
        InputPath::Chained(vec![path.clone(), path.clone()]),
    ] {
        let count = if matches!(input, InputPath::Chained(_)) {
            FRAMES * 2
        } else {
            FRAMES
        };
        let mut decoder = VideoDecoder::open_input(&input).unwrap();
        assert_eq!(decoder.backend(), DecodeBackend::Cuda);
        for _ in 0..2 {
            for index in 0..count {
                let frame = decoder
                    .next_frame_gpu()
                    .unwrap()
                    .expect("missing GPU frame");
                assert_eq!((frame.width, frame.height), (WIDTH, HEIGHT));
                assert_eq!(frame.timestamp_us, index as i64 * 1_000_000 / 30);
            }
            for _ in 0..3 {
                assert!(decoder.next_frame_gpu().unwrap().is_none());
            }
            assert!(
                decoder.next_frame().unwrap().is_none(),
                "stranded frames after GPU EOF"
            );
            decoder.seek_to_secs(0.0).unwrap();
        }
        // Validate this hardware decoder's decoded pixel identities as well.
        drain_cpu(&mut decoder, count);
    }
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC and Vulkan external-memory interop"]
fn cuda_workers_preserve_completion_failure_and_cancellation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    let software_path = dir.path().join("software.mkv");
    fixture_encoding(&software_path, "ffv1");
    let gpu = reco_core::gpu::GpuContext::new_blocking().unwrap();
    let input = InputPath::Single(path.clone());

    for offset in [-3_i64, 0, 3] {
        let mut source = reco_io::SmartFileSource::open(&input, &input, &gpu, offset).unwrap();
        assert!(source.is_gpu_resident(), "requires the CUDA zero-copy path");
        assert_eq!(
            source.skip_frames((FRAMES + 1) as u64).unwrap(),
            FRAMES as u64 - offset.unsigned_abs()
        );
        assert!(source.is_exhausted());
        assert!(source.next_frame().unwrap().is_none());
        assert_eq!(source.skip_frames(1).unwrap(), 0);
    }

    // A shorter input terminates the pair, and dropping the source releases
    // the longer producer even when it is waiting for a reusable slot.
    let chain = InputPath::Chained(vec![path.clone(), path]);
    for (left, right) in [(&chain, &input), (&input, &chain)] {
        let mut source = reco_io::SmartFileSource::open(left, right, &gpu, 0).unwrap();
        assert!(source.is_gpu_resident());
        assert_eq!(
            source.skip_frames((FRAMES * 2) as u64).unwrap(),
            FRAMES as u64
        );
    }

    let software_input = InputPath::Single(software_path);
    for (left, right) in [(&input, &software_input), (&software_input, &input)] {
        let mut source = reco_io::SmartFileSource::open(left, right, &gpu, 0).unwrap();
        assert!(source.is_gpu_resident());
        let error = source
            .next_frame()
            .err()
            .expect("software frame must not become EOF");
        assert!(
            error
                .to_string()
                .contains("unexpected decoded frame format"),
            "{error}"
        );
        assert!(!source.is_exhausted());
        assert_eq!(
            source.next_frame().err().unwrap().to_string(),
            error.to_string()
        );
        assert_eq!(
            source.skip_frames(1).unwrap_err().to_string(),
            error.to_string()
        );
    }

    let smaller = dir.path().join("smaller.mp4");
    fixture_size(&smaller, "libx264", 64, 64);
    let resized = InputPath::Chained(vec![input.first_path().to_path_buf(), smaller]);
    let mut source = reco_io::SmartFileSource::open(&resized, &resized, &gpu, 0).unwrap();
    assert!(source.is_gpu_resident());
    assert!(source.skip_frames((FRAMES * 2) as u64).is_err());
    assert!(!source.is_exhausted());
    drop(source);

    let mut source = reco_io::SmartFileSource::open(&input, &input, &gpu, 0).unwrap();
    assert!(source.is_gpu_resident());
    assert_eq!(source.skip_frames(1).unwrap(), 1);
    drop(source);
}

#[test]
fn dimensions_cannot_change_under_a_fixed_frame_layout() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.mp4");
    let second = dir.path().join("smaller.mp4");
    fixture(&first);
    fixture_size(&second, "libx264", 64, 64);
    let input = InputPath::Chained(vec![first, second]);
    let mut decoder = VideoDecoder::open_input_software(&input).unwrap();
    let mut count = 0;
    loop {
        match decoder.next_frame() {
            Ok(Some(_)) => count += 1,
            Ok(None) => panic!("changed dimensions must be rejected"),
            Err(error) => {
                assert!(
                    error.to_string().contains("decoded dimensions changed"),
                    "{error}"
                );
                break;
            }
        }
    }
    assert_eq!(count, FRAMES);
}
