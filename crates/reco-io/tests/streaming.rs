//! Bounded, task-owned RTMP receiver exercising the shared job's recording and broadcast.

#![cfg(all(feature = "ffmpeg", target_os = "linux"))]

#[allow(dead_code)]
#[path = "../src/rtsp/tests/peer.rs"]
mod peer;

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use ffmpeg_next::{self as ffmpeg, Rescale};
use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::projection::LShape;
use reco_io::ffmpeg::encoder::{StreamMonitor, StreamStatus};

struct Receiver {
    url: String,
    path: PathBuf,
    log: PathBuf,
    child: Child,
}

impl Receiver {
    fn new(directory: &Path) -> Self {
        let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reservation.local_addr().unwrap();
        let url = format!("rtmp://{address}/live/fixture");
        let path = directory.join("received.flv");
        let log = directory.join("receiver.log");
        drop(reservation);
        let child = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-listen",
                "1",
                "-timeout",
                "5",
                "-rw_timeout",
                "5000000",
                "-i",
                &url,
                "-map",
                "0",
                "-c",
                "copy",
                "-copyts",
                "-fs",
                "16777216",
                "-f",
                "flv",
            ])
            .arg(&path)
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(&log).unwrap())
            .spawn()
            .unwrap();
        let mut receiver = Self {
            url,
            path,
            log,
            child,
        };
        receiver.await_listen(address);
        receiver
    }

    fn await_listen(&mut self, address: std::net::SocketAddr) {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "receiver exited before listening: {}",
                self.log()
            );
            // Do not make a readiness connection: FFmpeg accepts one publishing client.
            let sockets = Command::new("ss").args(["-H", "-ltnp"]).output().unwrap();
            assert!(sockets.status.success());
            let owner = format!("pid={},", self.child.id());
            let endpoint = format!("{address} ");
            if String::from_utf8_lossy(&sockets.stdout)
                .lines()
                .any(|line| line.contains(&owner) && line.contains(&endpoint))
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "receiver never listened: {}",
                self.log()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn log(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap()
    }

    fn finish(&mut self, interrupted: bool) {
        let deadline = Instant::now() + Duration::from_secs(6);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(
                    interrupted || status.success(),
                    "receiver failed: {}",
                    self.log()
                );
                return;
            }
            assert!(
                Instant::now() < deadline,
                "receiver did not finish: {}",
                self.log()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Receiver {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = Command::new("kill")
                .args(["-TERM", &self.child.id().to_string()])
                .status();
            let deadline = Instant::now() + Duration::from_secs(1);
            while self.child.try_wait().ok().flatten().is_none() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if self.child.try_wait().ok().flatten().is_none() {
                let _ = self.child.kill();
            }
        }
        let _ = self.child.wait();
    }
}

fn fixture(path: &Path) {
    let result = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x32:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=997:sample_rate=48000",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-preset",
            "ultrafast",
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

fn packets(path: &Path, kind: ffmpeg::media::Type) -> Vec<(i64, Vec<u8>)> {
    let mut input = ffmpeg::format::input(path).unwrap();
    input
        .packets()
        .filter_map(|(stream, packet)| {
            (stream.parameters().medium() == kind).then(|| {
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
#[ignore = "requires openssl, Linux ss and a TLS-enabled linked FFmpeg; rejects only a task-owned untrusted loopback certificate"]
fn rtmps_rejects_untrusted_certificate_before_sending_media() {
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("key.pem");
    let cert = dir.path().join("cert.pem");
    let certificate = Command::new("openssl")
        .args([
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
            "-keyout",
        ])
        .arg(&key)
        .arg("-out")
        .arg(&cert)
        .output()
        .unwrap();
    assert!(
        certificate.status.success(),
        "{}",
        String::from_utf8_lossy(&certificate.stderr)
    );
    let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let path = dir.path().join("application-bytes.bin");
    let log = dir.path().join("tls.log");
    let child = Command::new("openssl")
        .args([
            "s_server",
            "-accept",
            &address.to_string(),
            "-naccept",
            "1",
            "-quiet",
            "-state",
            "-ign_eof",
            "-cert",
        ])
        .arg(&cert)
        .arg("-key")
        .arg(&key)
        .stdin(Stdio::null())
        .stdout(std::fs::File::create(&path).unwrap())
        .stderr(std::fs::File::create(&log).unwrap())
        .spawn()
        .unwrap();
    let mut receiver = Receiver {
        url: format!("rtmps://{address}/live/fixture"),
        path,
        log,
        child,
    };
    receiver.await_listen(address);
    let config = reco_io::ffmpeg::encoder::EncoderConfig {
        encoder_name: Some("libx264".into()),
        software_only: true,
        container: reco_io::output::Container::Flv,
        network_timeout: Some(Duration::from_secs(2)),
        ..Default::default()
    };
    let result = reco_io::adapters::FfmpegFileEncoder::new(
        Path::new(&receiver.url),
        64,
        32,
        (30, 1),
        &config,
    );
    assert!(result.is_err(), "untrusted TLS peer was accepted");
    receiver.finish(true);
    assert_eq!(
        std::fs::metadata(&receiver.path).unwrap().len(),
        0,
        "RTMP application bytes crossed an unverified TLS connection"
    );
    // GnuTLS can reject the certificate after the TLS handshake, without an unknown-CA alert.
    assert!(
        receiver
            .log()
            .contains("SSL_accept:SSLv3/TLS read finished"),
        "TLS was not exercised: {}",
        receiver.log()
    );
}

#[test]
#[ignore = "requires linked FFmpeg 8+, ffmpeg with RTMP listen support and Linux ss; owns only a bounded loopback receiver"]
fn recording_and_rtmp_share_source_audio_video_timing_and_pacing() {
    run_workflow(false);
}

#[test]
#[ignore = "requires linked FFmpeg 8+, ffmpeg with RTMP listen support and Linux ss; terminates only its own loopback receiver"]
fn rtmp_disconnect_is_reported_without_truncating_recording() {
    run_workflow(true);
}

#[test]
#[ignore = "requires linked FFmpeg 8+, ffmpeg and Linux ss; owns bounded RTSP/RTMP loopback peers"]
fn rtsp_audio_recording_and_broadcast_share_packets_and_survive_receiver_failure() {
    use reco_io::output::AudioMode;
    use reco_io::rtsp::{RtspConfig, RtspInput};
    assert!(ffmpeg::format::version() >> 16 >= 62);
    for disconnect in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("rtsp.mp4");
        let mut receiver = Receiver::new(dir.path());
        let peers = [
            peer::Peer::video(70, None),
            peer::Peer::with_audio(70, 7_000_000, 125_000),
        ];
        let inputs = peers
            .iter()
            .zip([0, -7_000_000])
            .map(|(peer, timestamp_offset_us)| RtspInput {
                url: peer.url.clone(),
                timestamp_offset_us,
            })
            .collect();
        let config = RtspConfig {
            fps: (10, 1),
            synchronization: reco_io::frame_sync::Config {
                wait_us: 150_000,
                skew_us: 30_000,
                max_age_us: 350_000,
                capacity: 4,
            },
            open_timeout: Duration::from_secs(3),
            read_timeout: Duration::from_secs(1),
            software_decode: true,
        };
        let cal = Calibration::new(
            vec![Lens::fisheye(64, 32, 32.0, 32.0, 32.0, 16.0, [0.0; 4]); 2],
            LShape {
                intersect: 0.5,
                blend_width: 0.0,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
            },
            Framing {
                axis_offset: 0.25,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        let monitor = Arc::new(OnceLock::<StreamMonitor>::new());
        let ready = monitor.clone();
        let progress = monitor.clone();
        let receiver_pid = receiver.child.id();
        let mut killed = false;
        let result = reco_io::StitchJob::from_rtsp(inputs, config, cal, &output)
            .cpu()
            .resolution(64, 32)
            .encoder_name("libx264")
            .preset("ultrafast")
            .audio(AudioMode::CopyFrom(1))
            .lookahead(0.3)
            .lookahead_memory_limit(1024 * 1024)
            .max_frames(40)
            .stream_to(&receiver.url)
            .on_stream_ready(move |value| {
                ready.set(value).unwrap();
            })
            .on_progress(move |p| {
                if disconnect
                    && !killed
                    && p.frames_completed >= 15
                    && progress
                        .get()
                        .is_some_and(|m| m.status() == StreamStatus::Connected)
                {
                    assert!(
                        Command::new("kill")
                            .args(["-TERM", &receiver_pid.to_string()])
                            .status()
                            .unwrap()
                            .success()
                    );
                    killed = true;
                }
            })
            .on_session(|session, _| {
                session.core_mut().set_constrained_look(false);
                Ok(())
            })
            .run(&Arc::new(AtomicBool::new(false)))
            .unwrap();
        assert_eq!(result.frames_processed, 40);
        receiver.finish(disconnect);
        drop(peers);
        assert_eq!(result.stream_status, Some(monitor.get().unwrap().status()));
        let audio = packets(&output, ffmpeg::media::Type::Audio);
        assert!(
            audio.len() > 170,
            "selected source audio must span recording"
        );
        assert!(
            audio[0].0 < 30_000,
            "audio must follow first selected output slot"
        );
        assert!(audio.last().unwrap().0 > 3_800_000);
        let mut decoder = reco_io::ffmpeg::decoder::VideoDecoder::open_software(&output).unwrap();
        for _ in 0..40 {
            assert!(decoder.next_frame().unwrap().is_some());
        }
        assert!(decoder.next_frame().unwrap().is_none());
        if disconnect {
            assert!(
                matches!(result.stream_status, Some(StreamStatus::Failed(_))),
                "{:?}",
                result.stream_status
            );
        } else {
            assert_eq!(result.stream_status, Some(StreamStatus::Finished));
            let mut shift = None;
            for kind in [ffmpeg::media::Type::Audio, ffmpeg::media::Type::Video] {
                let recording = packets(&output, kind);
                let streamed = packets(&receiver.path, kind);
                assert_eq!(recording.len(), streamed.len());
                for ((expected_time, expected_bytes), (time, bytes)) in
                    recording.iter().zip(streamed)
                {
                    assert_eq!(*expected_bytes, bytes, "compressed {kind:?} packet changed");
                    let delta = time - expected_time;
                    assert!(
                        delta.abs_diff(*shift.get_or_insert(delta)) <= 1000,
                        "A/V phase changed"
                    );
                }
            }
        }
    }
}

fn run_workflow(disconnect: bool) {
    assert!(
        ffmpeg::format::version() >> 16 >= 62,
        "this workflow requires the fixed FIFO in FFmpeg 8+"
    );
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.mp4");
    fixture(&input);
    let output = dir.path().join("recording.mp4");
    let mut receiver = Receiver::new(dir.path());
    let cal = Calibration::new(
        vec![Lens::fisheye(64, 32, 32.0, 32.0, 32.0, 16.0, [0.0; 4]); 2],
        LShape {
            intersect: 0.5,
            blend_width: 0.0,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
        },
        Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let clock = Arc::new(Mutex::new(Vec::new()));
    let progress_clock = clock.clone();
    let observed = Arc::new(OnceLock::<StreamMonitor>::new());
    let ready = observed.clone();
    let progress_monitor = observed.clone();
    let disconnected = Arc::new(AtomicBool::new(false));
    let disconnect_once = disconnected.clone();
    let receiver_pid = receiver.child.id();
    let result = reco_io::StitchJob::with_calibration([input.clone(), input], cal, &output)
        .cpu()
        .resolution(64, 32)
        .encoder_name("libx264")
        .preset("ultrafast")
        .start_time(0.25)
        .max_frames(45)
        .lookahead(0.1)
        .stream_to(&receiver.url)
        .on_stream_ready(move |monitor| {
            ready.set(monitor).unwrap();
        })
        .on_progress(move |p| {
            progress_clock
                .lock()
                .unwrap()
                .push((p.frames_completed, Instant::now()));
            if disconnect
                && p.frames_completed >= 15
                && progress_monitor
                    .get()
                    .is_some_and(|monitor| monitor.status() == StreamStatus::Connected)
                && !disconnect_once.swap(true, Ordering::Relaxed)
            {
                assert!(
                    Command::new("kill")
                        .args(["-TERM", &receiver_pid.to_string()])
                        .status()
                        .unwrap()
                        .success()
                );
            }
        })
        .on_session(|session, _| {
            session.core_mut().set_constrained_look(false);
            Ok(())
        })
        .run(&Arc::new(AtomicBool::new(false)))
        .unwrap();
    assert_eq!(result.frames_processed, 45);
    assert_eq!(result.stream_status, Some(observed.get().unwrap().status()));
    receiver.finish(disconnect);
    if disconnect {
        assert!(
            disconnected.load(Ordering::Relaxed),
            "receiver was never connected and terminated"
        );
        assert!(
            matches!(result.stream_status, Some(StreamStatus::Failed(_))),
            "{:?}",
            result.stream_status
        );
        let mut decoder = reco_io::ffmpeg::decoder::VideoDecoder::open_software(&output).unwrap();
        for _ in 0..45 {
            assert!(decoder.next_frame().unwrap().is_some());
        }
        assert!(decoder.next_frame().unwrap().is_none());
        assert!(packets(&output, ffmpeg::media::Type::Audio).len() >= 45);
        return;
    }
    assert_eq!(result.stream_status, Some(StreamStatus::Finished));
    let clock = clock.lock().unwrap();
    let early = clock.iter().find(|(frame, _)| *frame >= 10).unwrap().1;
    let late = clock.iter().find(|(frame, _)| *frame >= 40).unwrap().1;
    assert!(
        late.duration_since(early) >= Duration::from_millis(850),
        "network pacing must apply during processing, not only at startup or finalization"
    );
    let mut shift = None;
    for kind in [ffmpeg::media::Type::Audio, ffmpeg::media::Type::Video] {
        let recording = packets(&output, kind);
        let stream = packets(&receiver.path, kind);
        assert!(recording.len() >= 45);
        assert_eq!(recording.len(), stream.len());
        for ((expected_time, expected_bytes), (time, bytes)) in recording.iter().zip(stream) {
            assert_eq!(*expected_bytes, bytes, "shared compressed {kind:?} packets");
            let delta = time - expected_time;
            assert!(
                delta.abs_diff(*shift.get_or_insert(delta)) <= 1000,
                "relative A/V timing changed"
            );
        }
    }
    let mut decoder =
        reco_io::ffmpeg::decoder::VideoDecoder::open_software(&receiver.path).unwrap();
    for _ in 0..45 {
        let image = decoder.next_frame().unwrap().unwrap();
        assert!(image.y.iter().any(|&pixel| pixel > 32));
    }
    assert!(decoder.next_frame().unwrap().is_none());
}
