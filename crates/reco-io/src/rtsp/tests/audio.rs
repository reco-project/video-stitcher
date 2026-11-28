//! Actual same-connection audio/video clock mapping before output rebasing.

use super::*;
use crate::adapters::FfmpegFileEncoder;
use crate::ffmpeg::encoder::EncoderConfig;
use crate::output::AudioMode;
use ffmpeg_next::{self as ffmpeg, Rescale};
use reco_core::render::viewport::ViewportSize;
use reco_core::session::{SinkOptions, StitchSession};
use reco_core::stitch::{CpuExecutor, Executor};

#[test]
fn opening_discovers_late_video_without_rebasing_early_audio() {
    let peer = Peer::delayed_video_with_audio(50, 6);
    let mut source = RtspSource::open_with_audio(
        vec![input(&peer)],
        config(),
        Arc::default(),
        Some(0),
        #[cfg(feature = "gpu")]
        None,
    )
    .unwrap();
    assert_eq!(source.info().camera_sizes, [(64, 32)]);
    let first = source.next_frame().unwrap().unwrap();
    assert_eq!(first.pts_us, Some(0));
    let capture = first.capture.unwrap().times_us[0];
    assert!(capture + source.media_origin_us >= 600_000);
    let audio = source.audio().unwrap();
    let packet = audio
        .next_until(100_000)
        .unwrap()
        .expect("audio retained at video origin");
    assert!((0..=21_334).contains(&packet.pts().unwrap()));
}

#[test]
fn rtsp_demux_preserves_audio_phase_and_common_clock_origin() {
    for (origin, phase) in [(0, 250_000), (7_000_000, 125_000)] {
        let peer = Peer::with_audio(25, origin, phase);
        let mut options = ffmpeg::Dictionary::new();
        options.set("rtsp_transport", "tcp");
        options.set("allowed_media_types", "video+audio");
        options.set("analyzeduration", "100000");
        options.set("probesize", "1048576");
        options.set("fpsprobesize", "0");
        let deadline = Instant::now() + Duration::from_secs(4);
        let mut input = ffmpeg::format::input_with_interrupt_and_dictionary(
            std::path::Path::new(&peer.url),
            move || Instant::now() >= deadline,
            options,
        )
        .unwrap();
        assert_eq!(input.streams().len(), 2);
        let mut audio = Vec::new();
        let mut video = Vec::new();
        while audio.len() < 25 || video.len() < 8 {
            let mut packet = ffmpeg::Packet::empty();
            packet.read(&mut input).unwrap();
            let stream = input.stream(packet.stream()).unwrap();
            let Some(time) = packet.pts() else {
                continue;
            };
            let time = time.rescale(stream.time_base(), (1, 1_000_000));
            match stream.parameters().medium() {
                ffmpeg::media::Type::Video => video.push(time),
                ffmpeg::media::Type::Audio => audio.push(time),
                _ => panic!("unexpected stream"),
            }
        }
        assert_eq!(
            audio[0],
            origin + phase,
            "audio must not be independently rebased"
        );
        assert!(video[0].abs_diff(origin) <= 100_000, "{video:?}");
        for (index, time) in audio.into_iter().enumerate() {
            let expected =
                origin + phase + (index as i64 * 1024).rescale((1, 48000), (1, 1_000_000));
            assert!(
                time.abs_diff(expected) <= 1,
                "audio time {time} != {expected}"
            );
        }
        for pair in video.windows(2) {
            assert_eq!(pair[1] - pair[0], 100_000);
        }
    }
}

#[test]
fn selected_live_audio_survives_startup_trim_offsets_and_lookahead() {
    for (camera, delay_ms, depth) in [(0, 0, 0), (1, 700, 3)] {
        let phase = 125_000;
        let peers = [
            Peer::with_audio(40, 0, phase),
            Peer::with_audio(40, 7_000_000, phase),
        ];
        let mut inputs: Vec<_> = peers.iter().map(input).collect();
        inputs[1].timestamp_offset_us = -7_000_000;
        let stop = Arc::new(AtomicBool::new(false));
        let mut source = RtspSource::open_with_audio(
            inputs,
            config(),
            stop.clone(),
            Some(camera),
            #[cfg(feature = "gpu")]
            None,
        )
        .unwrap();
        let audio = source.audio().expect("selected AAC track");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live-audio.mp4");
        let encoder = FfmpegFileEncoder::with_live_audio(
            &path,
            64,
            32,
            (10, 1),
            &EncoderConfig {
                encoder_name: Some("libx264".into()),
                preset: Some("ultrafast".into()),
                ..Default::default()
            },
            Some(audio),
            Default::default(),
        )
        .unwrap();
        let executor = CpuExecutor::new(
            pipeline::calibration(),
            ViewportSize {
                width: 64,
                height: 32,
            },
            vec![(64, 32); 2],
            false,
        )
        .unwrap();
        let mut session = StitchSession::with_executor(Executor::Cpu(Box::new(executor))).unwrap();
        session.core_mut().set_constrained_look(false);
        session.set_lookahead(depth, 1024 * 1024);
        session
            .add_sink(Box::new(encoder), SinkOptions::threaded(2))
            .unwrap();
        std::thread::sleep(Duration::from_millis(delay_ms));
        session.run(&mut source, 12, &stop, None).unwrap();
        session.finish().unwrap();
        let origin = source.media_origin_us + source.output_origin_us.unwrap();
        if delay_ms > 0 {
            assert!(origin >= 700_000, "startup backlog must not become output");
        }
        drop(source);
        let mut output = ffmpeg::format::input(&path).unwrap();
        let audio = output.streams().best(ffmpeg::media::Type::Audio).unwrap();
        assert_eq!(audio.parameters().id(), ffmpeg::codec::Id::AAC);
        let index = audio.index();
        let time_base = audio.time_base();
        let times: Vec<_> = output
            .packets()
            .filter(|(_, p)| p.stream() == index)
            .map(|(_, p)| p.pts().unwrap().rescale(time_base, (1, 1_000_000)))
            .collect();
        assert!(times.len() >= 40, "{times:?}");
        let first_index = (u128::try_from((origin - phase).max(0)).unwrap() * 48000)
            .div_ceil(1_024_000_000) as i64;
        for (i, &time) in times.iter().enumerate() {
            let expected = phase + (first_index + i as i64) * 1_024_000_000 / 48000 - origin;
            assert!(
                time.abs_diff(expected) <= 50,
                "audio {time} expected {expected}, origin {origin}"
            );
        }
        assert!(
            times.last().unwrap() <= &1_201_000,
            "audio must stop with the video"
        );
    }
}

#[test]
fn live_job_copies_available_audio_and_honors_disable() {
    for mode in [AudioMode::CopyFrom(1), AudioMode::Disabled] {
        let peers = [
            Peer::video(30, None),
            Peer::with_audio(30, 7_000_000, 125_000),
        ];
        let mut inputs: Vec<_> = peers.iter().map(input).collect();
        inputs[1].timestamp_offset_us = -7_000_000;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("job.mp4");
        let has_audio = matches!(mode, AudioMode::CopyFrom(_));
        let result = crate::StitchJob::from_rtsp(inputs, config(), pipeline::calibration(), &path)
            .audio(mode)
            .cpu()
            .resolution(64, 32)
            .encoder_name("libx264")
            .lookahead(0.3)
            .lookahead_memory_limit(1024 * 1024)
            .max_frames(12)
            .on_session(|session, _| {
                session.core_mut().set_constrained_look(false);
                Ok(())
            })
            .run(&Arc::new(AtomicBool::new(false)))
            .unwrap();
        assert_eq!(result.frames_processed, 12);
        let mut output = ffmpeg::format::input(&path).unwrap();
        let audio = output
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .map(|s| s.index());
        assert_eq!(audio.is_some(), has_audio);
        if let Some(index) = audio {
            assert!(
                output
                    .packets()
                    .filter(|(_, p)| p.stream() == index)
                    .count()
                    >= 40
            );
        }
    }
}
