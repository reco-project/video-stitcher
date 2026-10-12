//! RTSP source through actual session detection, lookahead, rendering and encoded delivery.

use super::*;
use crate::adapters::FfmpegFileEncoder;
use crate::ffmpeg::{decoder::VideoDecoder, encoder::EncoderConfig};
use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
use reco_core::geometry::CameraIndex;
use reco_core::projection::LShape;
use reco_core::render::viewport::ViewportSize;
use reco_core::session::{SinkOptions, StitchSession};
use reco_core::sink::{OutputFrame, OutputSink, SinkError, SinkInput};
use reco_core::stitch::{CpuExecutor, Executor};

struct Observed {
    source: RtspSource,
    captures: Vec<Vec<i64>>,
}

impl FrameSource for Observed {
    fn info(&self) -> SourceInfo {
        self.source.info()
    }
    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        let frame = self.source.next_frame()?;
        if let Some(frame) = &frame {
            self.captures
                .push(frame.capture.as_ref().unwrap().times_us.clone());
        }
        Ok(frame)
    }
}

struct Counter(Arc<Mutex<[usize; 2]>>);
impl Detector for Counter {
    fn name(&self) -> &'static str {
        "capture-count"
    }
    fn detect(
        &mut self,
        camera: CameraIndex,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let DetectorFrame::Yuv(image) = frame else {
            panic!("host image expected")
        };
        assert!(image.y.iter().any(|&value| value > 32));
        self.0.lock().unwrap()[camera] += 1;
        Ok(Vec::new())
    }
}

type Frames = Arc<Mutex<Vec<(i64, Vec<u8>)>>>;
struct Recording {
    encoder: FfmpegFileEncoder,
    frames: Frames,
}
impl OutputSink for Recording {
    fn name(&self) -> &str {
        "verified recording"
    }
    fn wants(&self) -> SinkInput {
        self.encoder.wants()
    }
    fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
        self.frames.lock().unwrap().push((
            frame.pts_us.expect("live presentation time"),
            frame.data.to_vec(),
        ));
        self.encoder.consume(frame)
    }
    fn finish(&mut self) -> Result<(), SinkError> {
        self.encoder.finish()
    }
}

pub(super) fn calibration() -> Calibration {
    let lens = Lens::fisheye(64, 32, 32.0, 32.0, 32.0, 16.0, [0.0; 4]);
    Calibration::new(
        vec![lens.clone(), lens],
        LShape {
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.0,
        },
        Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        },
    )
}

#[test]
fn mapped_rtsp_images_survive_detection_lookahead_and_encoding() {
    for depth in [0, 3] {
        let executor = CpuExecutor::new(
            calibration(),
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
        session.set_detection_interval(1);
        let calls = Arc::new(Mutex::new([0; 2]));
        session.set_detector(Box::new(Counter(calls.clone())));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rtsp.mp4");
        let encoder = FfmpegFileEncoder::new(
            &path,
            64,
            32,
            (10, 1),
            &EncoderConfig {
                encoder_name: Some("libx264".into()),
                quality: crate::output::Quality::Value(100),
                ..Default::default()
            },
        )
        .unwrap();
        let frames = Frames::default();
        session
            .add_sink(
                Box::new(Recording {
                    encoder,
                    frames: frames.clone(),
                }),
                SinkOptions::threaded(1),
            )
            .unwrap();
        let peers = [
            Peer::mapped(30, 0, None),
            Peer::mapped(30, 7_000_000, Some(6)),
        ];
        let cancelled = Arc::new(AtomicBool::new(false));
        let source = RtspSource::open(
            vec![
                input(&peers[0]),
                RtspInput {
                    timestamp_offset_us: -7_000_000,
                    ..input(&peers[1])
                },
            ],
            config(),
            cancelled.clone(),
        )
        .unwrap();
        let mut source = Observed {
            source,
            captures: Vec::new(),
        };
        let result = session.run(&mut source, 12, &cancelled, None);
        session.finish().unwrap();
        assert_eq!(result.unwrap().frames_processed, 12);
        for camera in 0..2 {
            let distinct = source
                .captures
                .iter()
                .map(|times| times[camera])
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            assert_eq!(
                calls.lock().unwrap()[camera],
                distinct,
                "reused images must not become new observations"
            );
        }
        drop(source);
        drop(session);
        let mut decoder = VideoDecoder::open_software(&path).unwrap();
        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 12);
        for (time, expected) in frames.iter() {
            let decoded = decoder
                .next_frame()
                .unwrap()
                .expect("missing encoded frame");
            assert!(decoded.timestamp_us.abs_diff(*time) <= 1);
            assert!(decoded.y.iter().any(|&value| value > 32));
            let error: u64 = decoded
                .y
                .iter()
                .zip(expected)
                .map(|(&a, &b)| u64::from(a.abs_diff(b)))
                .sum();
            assert!(error <= decoded.y.len() as u64 * 2, "encoded image changed");
        }
        assert!(decoder.next_frame().unwrap().is_none());
    }
}

#[test]
fn job_uses_shared_lookahead_and_finalizes_live_output() {
    run_job(false);
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "requires a working GPU backend"]
fn gpu_job_uses_shared_lookahead_and_finalizes_live_output() {
    run_job(true);
}

fn run_job(gpu: bool) {
    for lookahead in [0.0, 0.3] {
        let peers = [Peer::video(30, None), Peer::video(30, None)];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("job.mp4");
        let calls = Arc::new(Mutex::new([0; 2]));
        let observed = calls.clone();
        let job = crate::StitchJob::from_rtsp(
            peers.iter().map(input).collect(),
            config(),
            calibration(),
            &path,
        )
        .resolution(64, 32)
        .encoder_name("libx264")
        .lookahead(lookahead)
        .lookahead_memory_limit(1024 * 1024)
        .max_frames(12)
        .on_session(move |session, source| {
            assert_eq!(source.info().camera_sizes, [(64, 32); 2]);
            assert!(!source.is_gpu_resident());
            session.core_mut().set_constrained_look(false);
            session.set_detector(Box::new(Counter(observed)));
            session.set_detection_interval(1);
            Ok(())
        });
        let job = if gpu { job } else { job.cpu() };
        let result = job.run(&Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(result.frames_processed, 12);
        assert_eq!(result.decode_mode, "RTSP host YUV420P");
        assert!(calls.lock().unwrap().iter().all(|&calls| calls >= 10));
        let mut decoder = VideoDecoder::open_software(&path).unwrap();
        let mut previous = None;
        for _ in 0..12 {
            let frame = decoder.next_frame().unwrap().expect("finalized frame");
            assert!(previous.is_none_or(|last| frame.timestamp_us > last));
            assert!(frame.y.iter().any(|&value| value > 32));
            previous = Some(frame.timestamp_us);
        }
        assert!(decoder.next_frame().unwrap().is_none());
    }
}

#[test]
fn job_rejects_file_only_controls_before_open_and_preserves_destination() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("keep.mp4");
    std::fs::write(&path, b"existing output").unwrap();
    let make = || {
        crate::StitchJob::from_rtsp(
            vec![
                RtspInput {
                    url: "rtsp://127.0.0.1:1/unopened".into(),
                    timestamp_offset_us: 0
                };
                2
            ],
            config(),
            calibration(),
            &path,
        )
        .cpu()
    };
    for job in [
        make().start_time(0.0),
        make().end_time(1.0),
        make().sync_offset(0),
        make().audio(crate::output::AudioMode::CopyFrom(2)),
    ] {
        let error = job.run(&Arc::new(AtomicBool::new(false))).unwrap_err();
        assert!(
            matches!(error, crate::stitch_job::StitchError::Other(_)),
            "{error}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"existing output");
    }
}

#[test]
fn job_finalizes_recording_when_live_input_stalls() {
    let peers = [Peer::video(8, None), Peer::video(8, None)];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stalled.mp4");
    let result = crate::StitchJob::from_rtsp(
        peers.iter().map(input).collect(),
        config(),
        calibration(),
        &path,
    )
    .cpu()
    .resolution(64, 32)
    .encoder_name("libx264")
    .on_session(|session, _| {
        session.core_mut().set_constrained_look(false);
        Ok(())
    })
    .run(&Arc::new(AtomicBool::new(false)));
    assert!(result.is_err(), "stalled input must not appear successful");
    let mut decoder = VideoDecoder::open_software(&path).unwrap();
    let mut frames = 0;
    while decoder.next_frame().unwrap().is_some() {
        frames += 1;
    }
    assert!(frames > 0, "already rendered frames must be finalized");
}

#[test]
#[ignore = "requires RECO_TEST_CLI pointing to a current reco binary"]
fn cli_stitches_rtsp_through_the_job() {
    use std::process::{Child, Command, Stdio};
    struct Process(Child);
    impl Drop for Process {
        fn drop(&mut self) {
            if self.0.try_wait().ok().flatten().is_none() {
                let _ = self.0.kill();
            }
            let _ = self.0.wait();
        }
    }
    let peers = [
        Peer::video(400, None),
        Peer::with_audio(400, 7_000_000, 125_000),
    ];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cli.mp4");
    let cal = dir.path().join("calibration.json");
    calibration().to_file(&cal).unwrap();
    let log = dir.path().join("cli.log");
    let stdout = std::fs::File::create(&log).unwrap();
    let stderr = stdout.try_clone().unwrap();
    let mut child = Process(
        Command::new(std::env::var_os("RECO_TEST_CLI").expect("RECO_TEST_CLI"))
            .args([
                "stitch",
                &peers[0].url,
                &peers[1].url,
                "--rtsp-fps",
                "10",
                "--rtsp-offsets-us=0,-7000000",
                "--audio-source",
                "1",
                "--cpu",
                "--encoder",
                "libx264",
                "--width",
                "64",
                "--height",
                "32",
                "--max-frames",
                "20",
                "-c",
            ])
            .arg(cal)
            .arg("-o")
            .arg(&path)
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(stderr)
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "CLI deadline exceeded: {}",
            std::fs::read_to_string(&log).unwrap()
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    assert!(
        status.success(),
        "{}",
        std::fs::read_to_string(&log).unwrap()
    );
    let mut decoder = VideoDecoder::open_software(&path).unwrap();
    let mut previous = None;
    for _ in 0..20 {
        let frame = decoder.next_frame().unwrap().expect("CLI output frame");
        assert!(previous.is_none_or(|last| frame.timestamp_us > last));
        assert!(frame.y.iter().any(|&value| value > 32));
        previous = Some(frame.timestamp_us);
    }
    assert!(decoder.next_frame().unwrap().is_none());
    if let Some(destination) = std::env::var_os("RECO_TEST_RTSP_OUTPUT") {
        assert!(
            !std::path::Path::new(&destination).exists(),
            "refusing to replace a retained output"
        );
        std::fs::copy(&path, destination).unwrap();
    }
    let mut media = ffmpeg_next::format::input(&path).unwrap();
    let audio_index = media
        .streams()
        .best(ffmpeg_next::media::Type::Audio)
        .unwrap()
        .index();
    assert!(
        media
            .packets()
            .filter(|(_, packet)| packet.stream() == audio_index)
            .count()
            > 80
    );
}
