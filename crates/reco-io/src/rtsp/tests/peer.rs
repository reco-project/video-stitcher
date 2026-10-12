//! Minimal owned RTSP interleaved-RTP test peer; no host services or sockets outside loopback.

#[path = "peer/audio.rs"]
mod audio;
use audio::Audio;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub(super) struct Peer {
    pub(super) url: String,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

fn fixture(frames: usize) -> Vec<Vec<Vec<u8>>> {
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x32:rate=10",
            "-frames:v",
            &frames.to_string(),
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-x264-params",
            "keyint=1:repeat-headers=1:aud=1",
            "-f",
            "h264",
            "pipe:1",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let data = output.stdout;
    let mut starts = Vec::new();
    let mut index = 0;
    while index + 3 < data.len() {
        let prefix = if data[index..].starts_with(&[0, 0, 0, 1]) {
            4
        } else if data[index..].starts_with(&[0, 0, 1]) {
            3
        } else {
            index += 1;
            continue;
        };
        starts.push((index, index + prefix));
        index += prefix;
    }
    let mut result: Vec<Vec<Vec<u8>>> = Vec::new();
    for (index, &(_, begin)) in starts.iter().enumerate() {
        let end = starts
            .get(index + 1)
            .map_or(data.len(), |&(start, _)| start);
        let nal = &data[begin..end];
        if nal[0] & 31 == 9 {
            result.push(Vec::new());
        }
        result
            .last_mut()
            .expect("AUD precedes image")
            .push(nal.to_vec());
    }
    assert_eq!(result.len(), frames);
    result
}

impl Peer {
    pub(super) fn silent() -> Self {
        Self::start(None, None, 0, None, None)
    }
    pub(super) fn video(frames: usize, reset: Option<usize>) -> Self {
        Self::start(Some(fixture(frames)), reset, 0, None, None)
    }

    pub(super) fn mapped(frames: usize, offset_us: i64, missing: Option<usize>) -> Self {
        Self::start(Some(fixture(frames)), None, offset_us, missing, None)
    }

    pub(super) fn with_audio(frames: usize, source_offset_us: i64, audio_offset_us: i64) -> Self {
        Self::start(
            Some(fixture(frames)),
            None,
            source_offset_us,
            None,
            Some(Audio::new(frames, audio_offset_us)),
        )
    }

    #[allow(dead_code)] // This fixture module is also reused by the GUI tests.
    pub(super) fn delayed_video_with_audio(frames: usize, delay: usize) -> Self {
        let mut images = fixture(frames);
        for image in images.iter_mut().take(delay) {
            image.clear();
        }
        Self::start(Some(images), None, 0, None, Some(Audio::new(frames, 0)))
    }

    fn start(
        frames: Option<Vec<Vec<Vec<u8>>>>,
        reset: Option<usize>,
        offset_us: i64,
        missing: Option<usize>,
        mut audio: Option<Audio>,
    ) -> Self {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("rtsp://{}/stream", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let control = stop.clone();
        let base = url.clone();
        let thread = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut stream = loop {
                if control.load(Ordering::Relaxed) || Instant::now() >= deadline {
                    return;
                }
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("accept: {error}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_millis(30)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_millis(300)))
                .unwrap();
            let Some(frames) = frames else {
                while !control.load(Ordering::Relaxed) && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(5));
                }
                return;
            };
            let mut request = Vec::new();
            loop {
                if control.load(Ordering::Relaxed) || Instant::now() >= deadline {
                    return;
                }
                let mut bytes = [0; 2048];
                match stream.read(&mut bytes) {
                    Ok(0) => return,
                    Ok(n) => request.extend_from_slice(&bytes[..n]),
                    Err(error)
                        if matches!(
                            error.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ) =>
                    {
                        continue;
                    }
                    Err(_) => return,
                }
                if !request.ends_with(b"\r\n\r\n") {
                    continue;
                }
                let text = String::from_utf8(std::mem::take(&mut request)).unwrap();
                let sequence = text
                    .lines()
                    .find_map(|line| line.strip_prefix("CSeq: "))
                    .unwrap();
                let method = text.split_whitespace().next().unwrap();
                let headers = match method {
                    "OPTIONS" => "Public: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN\r\n".into(),
                    "DESCRIBE" => {
                        let mut sdp = "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=Reco test\r\nt=0 0\r\na=control:*\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\na=fmtp:96 packetization-mode=1\r\na=control:trackID=0\r\n".to_string();
                        if let Some(audio) = &audio {
                            sdp.push_str(&audio.sdp());
                        }
                        format!(
                            "Content-Base: {base}/\r\nContent-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{sdp}",
                            sdp.len()
                        )
                    }
                    "SETUP" => {
                        if text.lines().next().unwrap().contains("trackID=1") {
                            audio.as_mut().unwrap().enabled = true;
                            "Session: 1\r\nTransport: RTP/AVP/TCP;unicast;interleaved=2-3\r\n"
                                .into()
                        } else {
                            "Session: 1\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n"
                                .into()
                        }
                    }
                    "PLAY" => format!(
                        "Session: 1\r\nRange: npt={:.6}-\r\nRTP-Info: url={base}/trackID=0;seq=0;rtptime=90000{}\r\n",
                        offset_us as f64 / 1e6,
                        if audio.as_ref().is_some_and(|a| a.enabled) {
                            format!(",url={base}/trackID=1;seq=0;rtptime=48000")
                        } else {
                            String::new()
                        },
                    ),
                    "TEARDOWN" => return,
                    _ => "Session: 1\r\n".into(),
                };
                let ending = if method == "DESCRIBE" { "" } else { "\r\n" };
                if write!(
                    stream,
                    "RTSP/1.0 200 OK\r\nCSeq: {sequence}\r\n{headers}{ending}"
                )
                .is_err()
                {
                    return;
                }
                if method == "PLAY" {
                    send_frames(
                        &mut stream,
                        &frames,
                        reset,
                        missing,
                        &control,
                        audio.as_mut(),
                    );
                }
            }
        });
        Self {
            url,
            stop,
            thread: Some(thread),
        }
    }
}

fn send_frames(
    stream: &mut TcpStream,
    frames: &[Vec<Vec<u8>>],
    reset: Option<usize>,
    missing: Option<usize>,
    stop: &AtomicBool,
    mut audio: Option<&mut Audio>,
) {
    let start = Instant::now();
    let mut sequence = 0u16;
    if let Some(audio) = audio.as_ref().filter(|audio| audio.enabled)
        && audio.send_reports(stream).is_err()
    {
        return;
    }
    for (index, nals) in frames.iter().enumerate() {
        while start.elapsed() < Duration::from_millis(index as u64 * 100) {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        if let Some(audio) = &mut audio
            && audio.send_until(stream, index as i64 * 100_000).is_err()
        {
            return;
        }
        if missing == Some(index) || nals.is_empty() {
            continue;
        }
        let timestamp = if reset.is_some_and(|at| index >= at) {
            (index - reset.unwrap()) as u32 * 9000
        } else {
            index as u32 * 9000
        };
        let mut packet = vec![0x80, 96 | 128];
        packet.extend_from_slice(&sequence.to_be_bytes());
        packet.extend_from_slice(&(timestamp + 90000).to_be_bytes());
        packet.extend_from_slice(&1234u32.to_be_bytes());
        packet.push(0x78); // STAP-A: one complete access unit per RTP packet.
        for nal in nals {
            packet.extend_from_slice(&u16::try_from(nal.len()).unwrap().to_be_bytes());
            packet.extend_from_slice(nal);
        }
        if interleaved(stream, 0, &packet).is_err() {
            return;
        }
        sequence = sequence.wrapping_add(1);
    }
}

fn interleaved(stream: &mut TcpStream, channel: u8, packet: &[u8]) -> std::io::Result<()> {
    let size = u16::try_from(packet.len()).unwrap();
    let mut bytes = vec![b'$', channel];
    bytes.extend_from_slice(&size.to_be_bytes());
    bytes.extend_from_slice(packet);
    stream.write_all(&bytes)
}

impl Drop for Peer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().unwrap();
    }
}
