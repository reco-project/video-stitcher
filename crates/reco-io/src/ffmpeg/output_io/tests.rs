//! Loopback fault injection for owned output cancellation and optional transport isolation.

use super::*;
use crate::ffmpeg::{decoder::VideoDecoder, encoder::VideoEncoder};
use std::net::TcpListener;
use std::thread::{self, JoinHandle};

mod overflow;

struct StalledPeer {
    address: String,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl StalledPeer {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut peer = None;
            while !flag.load(Ordering::Relaxed) && Instant::now() < deadline {
                if peer.is_none() {
                    match listener.accept() {
                        Ok((socket, _)) => peer = Some(socket),
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => (),
                        Err(e) => panic!("accept: {e}"),
                    }
                }
                thread::sleep(Duration::from_millis(2));
            }
        });
        Self {
            address,
            stop,
            worker: Some(worker),
        }
    }
    fn url(&self, protocol: &str) -> String {
        format!("{protocol}://{}/test/diagnostic", self.address)
    }
}

impl Drop for StalledPeer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap();
    }
}

fn config() -> EncoderConfig {
    EncoderConfig {
        software_only: true,
        encoder_name: Some("libx264".into()),
        preset: Some("ultrafast".into()),
        network_timeout: Some(Duration::from_millis(250)),
        ..Default::default()
    }
}

#[test]
fn destination_classification_keeps_local_containers_and_windows_paths_local() {
    for path in [
        "rtmp://127.0.0.1/live/key",
        "rtmps://host/live/key",
        "srt://host:9000",
        "tcp://host:9000",
    ] {
        assert!(crate::output::is_network_destination(Path::new(path)));
    }
    for path in [
        "recording.flv",
        "file:///tmp/recording.mp4",
        r"C:\video\output.mp4",
        "/tmp/output.mp4",
    ] {
        assert!(!crate::output::is_network_destination(Path::new(path)));
    }
}

#[test]
fn later_cancellation_does_not_replace_a_transport_error() {
    let dir = tempfile::tempdir().unwrap();
    let flag = Arc::new(AtomicBool::new(false));
    let config = EncoderConfig {
        interrupted: Some(flag.clone()),
        ..config()
    };
    let mut output = OutputIo::open(&dir.path().join("unused.mp4"), "mp4", &config, false).unwrap();
    Arc::get_mut(&mut output.control).unwrap().cancelled = Some(flag.clone());
    output.control.record(ffmpeg::Error::InvalidData.into());
    flag.store(true, Ordering::Relaxed);
    assert_eq!(
        interrupted(Arc::as_ptr(&output.control).cast_mut().cast()),
        1
    );
    assert!(matches!(
        output.control.status(),
        Err(EncodeError::Ffmpeg(ffmpeg::Error::InvalidData))
    ));
}

fn video(path: &Path, config: &EncoderConfig, count: usize) -> Option<StreamMonitor> {
    let mut encoder = VideoEncoder::new(path, 64, 64, ffmpeg::Rational(30, 1), config).unwrap();
    let monitor = encoder.stream_monitor();
    let mut pixels = vec![128; 64 * 64 * 3 / 2];
    for i in 0..count {
        pixels[..4096].fill(32 + (i % 160) as u8);
        encoder.write_nv12_frame(&pixels, None).unwrap();
    }
    encoder.finish().unwrap();
    monitor
}

#[test]
fn network_open_deadline_and_cancellation_release_owned_state() {
    crate::ffmpeg::init();
    for cancel in [false, true] {
        let peer = StalledPeer::new();
        let flag = Arc::new(AtomicBool::new(false));
        let mut config = config();
        config.interrupted = Some(flag.clone());
        let worker = cancel.then(|| {
            config.network_timeout = Some(Duration::from_secs(5));
            let flag = flag.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(100));
                flag.store(true, Ordering::Relaxed);
            })
        });
        let start = Instant::now();
        let error = OutputIo::open(Path::new(&peer.url("rtmp")), "flv", &config, false)
            .err()
            .unwrap();
        assert_eq!(matches!(error, EncodeError::Cancelled), cancel);
        assert!(start.elapsed() < Duration::from_secs(2));
        if let Some(worker) = worker {
            worker.join().unwrap();
        }
        drop(config);
        assert_eq!(Arc::strong_count(&flag), 1);
    }
}

#[test]
fn optional_stalled_stream_preserves_every_recording_frame() {
    let dir = tempfile::tempdir().unwrap();
    let peer = StalledPeer::new();
    let primary = dir.path().join("primary.mp4");
    let reference = dir.path().join("reference.mp4");
    assert!(video(&reference, &config(), 900).is_none());
    let config = EncoderConfig {
        stream_url: Some(peer.url("rtmp")),
        ..config()
    };
    let start = Instant::now();
    let monitor = video(&primary, &config, 900).unwrap();
    assert!(matches!(monitor.status(), StreamStatus::Failed(_)));
    assert!(start.elapsed() < Duration::from_secs(3));
    let mut expected = VideoDecoder::open_software(&reference).unwrap();
    let mut actual = VideoDecoder::open_software(&primary).unwrap();
    for _ in 0..900 {
        let expected = expected.next_frame().unwrap().unwrap();
        let actual = actual.next_frame().unwrap().unwrap();
        assert_eq!(
            (actual.timestamp_us, actual.y, actual.u, actual.v),
            (expected.timestamp_us, expected.y, expected.u, expected.v)
        );
    }
    assert!(actual.next_frame().unwrap().is_none());
}

#[test]
fn optional_stream_failure_is_observable_without_producing_video() {
    for cancel in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let peer = StalledPeer::new();
        let flag = Arc::new(AtomicBool::new(false));
        let config = EncoderConfig {
            stream_url: Some(peer.url("rtmp")),
            interrupted: Some(flag.clone()),
            network_timeout: Some(if cancel {
                Duration::from_secs(10)
            } else {
                Duration::from_millis(250)
            }),
            ..config()
        };
        let mut encoder = VideoEncoder::new(
            &dir.path().join("recording.mp4"),
            64,
            64,
            ffmpeg::Rational(30, 1),
            &config,
        )
        .unwrap();
        let monitor = encoder.stream_monitor().unwrap();
        if ffmpeg::format::version() >> 16 < 62 {
            assert!(
                matches!(monitor.status(), StreamStatus::Failed(reason) if reason.contains("FFmpeg 8+"))
            );
            encoder
                .write_nv12_frame(&[128; 64 * 64 * 3 / 2], None)
                .unwrap();
            encoder.finish().unwrap();
            continue;
        }
        if cancel {
            flag.store(true, Ordering::Relaxed);
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        while matches!(
            monitor.status(),
            StreamStatus::Connecting | StreamStatus::Connected
        ) && Instant::now() < deadline
        {
            thread::sleep(Duration::from_millis(5));
        }
        let failure = monitor.status();
        if cancel {
            assert_eq!(failure, StreamStatus::Cancelled);
        } else {
            assert!(matches!(failure, StreamStatus::Failed(_)), "{failure:?}");
        }
        // A transport error must not stop local encoding or be replaced by successful recording finalization.
        encoder
            .write_nv12_frame(&[128; 64 * 64 * 3 / 2], None)
            .unwrap();
        encoder.finish().unwrap();
        drop(encoder);
        assert_eq!(monitor.status(), failure);
    }
}

#[test]
fn rejected_optional_stream_keeps_a_failure_monitor_and_recording() {
    let dir = tempfile::tempdir().unwrap();
    let config = EncoderConfig {
        stream_url: Some("rtsp://invalid.example/private-key".into()),
        ..config()
    };
    let output = dir.path().join("recording.mp4");
    let monitor = video(&output, &config, 3).unwrap();
    let StreamStatus::Failed(reason) = monitor.status() else {
        panic!("rejected stream must be visible")
    };
    assert!(!reason.contains("private-key"));
    assert!(reason.contains("RTMP"));
    let mut decoder = VideoDecoder::open_software(&output).unwrap();
    for _ in 0..3 {
        assert!(decoder.next_frame().unwrap().is_some());
    }
    assert!(decoder.next_frame().unwrap().is_none());
}

fn with_stream(output: &mut OutputIo, input: &format::context::Input) -> Result<(), EncodeError> {
    let source = input.streams().best(ffmpeg::media::Type::Video).unwrap();
    let mut stream = output
        .add_stream(ffmpeg::encoder::find(ffmpeg::codec::Id::None))
        .unwrap();
    stream.set_parameters(source.parameters());
    stream.set_time_base(source.time_base());
    // SAFETY: uniquely owned newly added stream parameters.
    unsafe {
        (*stream.parameters().as_mut_ptr()).codec_tag = 0;
    }
    output.write_header(Dictionary::new())
}

#[test]
fn fifo_finalization_retains_failure_without_another_packet() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.mp4");
    video(&path, &config(), 2);
    let input = format::input(&path).unwrap();
    let peer = StalledPeer::new();
    let mut output = OutputIo::open(Path::new(&peer.url("rtmp")), "flv", &config(), true).unwrap();
    with_stream(&mut output, &input).unwrap();
    let start = Instant::now();
    assert!(
        output.finish().is_err(),
        "FIFO success must not hide its transport failure"
    );
    assert!(
        output.finish().is_err(),
        "terminal error must remain visible"
    );
    assert!(start.elapsed() < Duration::from_secs(2));
}

#[test]
fn blocked_packet_write_is_cancelled_and_local_finalization_is_not() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.mp4");
    video(&path, &config(), 2);
    let input = format::input(&path).unwrap();
    let peer = StalledPeer::new();
    let flag = Arc::new(AtomicBool::new(false));
    let config = EncoderConfig {
        interrupted: Some(flag.clone()),
        network_timeout: Some(Duration::from_secs(5)),
        ..config()
    };
    let mut output = OutputIo::open(Path::new(&peer.url("tcp")), "flv", &config, false).unwrap();
    with_stream(&mut output, &input).unwrap();
    let worker = {
        let flag = flag.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            flag.store(true, Ordering::Relaxed);
        })
    };
    let start = Instant::now();
    let mut cancelled = false;
    for i in 0..2000 {
        let mut packet = Packet::copy(&[0; 65536]);
        packet.set_stream(0);
        packet.set_pts(Some(i * 33));
        packet.set_dts(Some(i * 33));
        if let Err(error) = output.write_packet(&mut packet) {
            assert!(matches!(error, EncodeError::Cancelled));
            cancelled = true;
            break;
        }
    }
    assert!(cancelled);
    assert!(start.elapsed() < Duration::from_secs(2));
    assert!(matches!(output.finish(), Err(EncodeError::Cancelled)));
    worker.join().unwrap();
    drop(output);
    // The same cancelled signal must not corrupt a local recording's drain/trailer.
    video(&dir.path().join("local.mp4"), &config, 2);
}
