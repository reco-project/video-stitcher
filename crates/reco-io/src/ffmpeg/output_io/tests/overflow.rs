//! Bounded loopback pressure with inter-frame video and interleaved audio.

use super::*;
use std::io::Read;

struct RecoveringPeer {
    address: String,
    paused: Arc<AtomicBool>,
    resume: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<Vec<u8>>>,
}

impl RecoveringPeer {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        socket2::SockRef::from(&listener)
            .set_recv_buffer_size(65536)
            .unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let paused = Arc::new(AtomicBool::new(false));
        let resume = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = {
            let paused = paused.clone();
            let resume = resume.clone();
            let stop = stop.clone();
            thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(15);
                let mut bytes = Vec::new();
                let mut socket = loop {
                    if stop.load(Ordering::Relaxed) || Instant::now() >= deadline {
                        return bytes;
                    }
                    match listener.accept() {
                        Ok((socket, _)) => break socket,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(2));
                        }
                        Err(error) => panic!("accept: {error}"),
                    }
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_millis(20)))
                    .unwrap();
                let mut buffer = [0; 16384];
                while !stop.load(Ordering::Relaxed) && Instant::now() < deadline {
                    if bytes.len() >= 32768 && !resume.load(Ordering::Relaxed) {
                        paused.store(true, Ordering::Relaxed);
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    match socket.read(&mut buffer) {
                        Ok(0) => return bytes,
                        Ok(count) => bytes.extend_from_slice(&buffer[..count]),
                        Err(error)
                            if matches!(
                                error.kind(),
                                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                            ) => {}
                        Err(error) => panic!("read: {error}"),
                    }
                    assert!(bytes.len() < 16 * 1024 * 1024, "unbounded output");
                }
                bytes
            })
        };
        Self {
            address,
            paused,
            resume,
            stop,
            worker: Some(worker),
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.resume.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap()
    }
}

impl Drop for RecoveringPeer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn texture(index: usize) -> Vec<u8> {
    let mut pixels = vec![128; 128 * 128 * 3 / 2];
    let mut state = index as u32 + 1;
    for (pixel, value) in pixels[..128 * 128].iter_mut().enumerate() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let noise = if pixel % 128 < 64 {
            state
        } else {
            // Translating texture depends on earlier pictures; random noise alone hid corruption.
            let mut position = ((pixel % 128 + index * 2) % 4096 + pixel / 128 * 4096 + 1) as u32;
            position ^= position << 13;
            position ^= position >> 17;
            position ^= position << 5;
            position
        };
        *value = 16 + (noise % 220) as u8;
    }
    pixels
}

#[test]
#[ignore = "requires an FFmpeg build with video-keyframe FIFO recovery and the ffmpeg command"]
fn fifo_overflow_resumes_on_video_keyframes_with_audio() {
    crate::ffmpeg::init();
    let dir = tempfile::tempdir().unwrap();
    let audio = dir.path().join("audio.m4a");
    let generated = std::process::Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "30",
            "-c:a",
            "aac",
        ])
        .arg(&audio)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = dir.path().join("source.mp4");
    let config = EncoderConfig {
        quality: crate::output::Quality::Value(100),
        gop_size: Some(30),
        audio_source: Some(vec![audio]),
        network_timeout: Some(Duration::from_secs(2)),
        ..config()
    };
    let mut encoder = VideoEncoder::new(&source, 128, 128, (30, 1).into(), &config).unwrap();
    for index in 0..900 {
        encoder
            .write_nv12_frame(&texture(index), Some(index as i64 * 1_000_000 / 30))
            .unwrap();
    }
    encoder.finish().unwrap();
    drop(encoder);

    let peer = RecoveringPeer::new();
    let mut input = format::input(&source).unwrap();
    let video = input
        .streams()
        .best(ffmpeg::media::Type::Video)
        .unwrap()
        .index();
    assert!(input.streams().best(ffmpeg::media::Type::Audio).is_some());
    let mut output = OutputIo::open(
        Path::new(&format!("tcp://{}", peer.address)),
        "flv",
        &config,
        true,
    )
    .unwrap();
    for stream in input.streams() {
        let mut target = output
            .add_stream(ffmpeg::encoder::find(ffmpeg::codec::Id::None))
            .unwrap();
        target.set_parameters(stream.parameters());
        target.set_time_base(stream.time_base());
        // SAFETY: each newly created stream owns independent writable parameters.
        unsafe {
            (*target.parameters().as_mut_ptr()).codec_tag = 0;
        }
    }
    output.write_header(Dictionary::new()).unwrap();
    let start = Instant::now();
    let mut pictures = 0;
    for (stream, mut packet) in input.packets() {
        let is_video = stream.index() == video;
        packet.rescale_ts(
            stream.time_base(),
            output.stream(stream.index()).unwrap().time_base(),
        );
        output.write_packet(&mut packet).unwrap();
        if is_video {
            pictures += 1;
            if pictures <= 60 {
                // Let the writer establish a decodable prefix before applying pressure.
                thread::sleep(Duration::from_millis(5));
            }
            if pictures == 60 {
                let deadline = Instant::now() + Duration::from_secs(2);
                while !peer.paused.load(Ordering::Relaxed) && Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(2));
                }
                assert!(peer.paused.load(Ordering::Relaxed), "receiver never paused");
            }
            if pictures >= 650 {
                peer.resume.store(true, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(5));
            }
        }
    }
    assert_eq!(pictures, 900);
    output.finish().unwrap();
    drop(output);
    let received = dir.path().join("received.flv");
    std::fs::write(&received, peer.finish()).unwrap();
    assert!(start.elapsed() < Duration::from_secs(10));

    let mut packets = format::input(&received).unwrap();
    assert!(packets.streams().best(ffmpeg::media::Type::Audio).is_some());
    let mut previous_pts = None;
    for (stream, packet) in packets.packets() {
        if stream.parameters().medium() != ffmpeg::media::Type::Video {
            continue;
        }
        use ffmpeg::Rescale;
        let pts = packet
            .pts()
            .unwrap()
            .rescale(stream.time_base(), (1, 1_000_000));
        if previous_pts.is_some_and(|last| pts - last > 50_000) {
            assert!(
                packet.is_key(),
                "recovery resumed on a predicted video picture"
            );
        }
        previous_pts = Some(pts);
    }

    let mut expected = VideoDecoder::open_software(&source).unwrap();
    let mut actual = VideoDecoder::open_software(&received).unwrap();
    let mut offset = None;
    let mut previous = None;
    let mut expected_index = 0;
    let mut gaps = 0;
    let mut count = 0;
    while let Some(frame) = actual.next_frame().unwrap() {
        let elapsed = frame.timestamp_us - *offset.get_or_insert(frame.timestamp_us);
        let index = ((elapsed * 30 + 500_000) / 1_000_000) as usize;
        assert!(index < 900 && previous.is_none_or(|last| index > last));
        gaps += usize::from(previous.is_some_and(|last| index > last + 1));
        while expected_index < index {
            expected.next_frame().unwrap().unwrap();
            expected_index += 1;
        }
        let reference = expected.next_frame().unwrap().unwrap();
        expected_index += 1;
        assert!(
            (frame.y, frame.u, frame.v) == (reference.y, reference.u, reference.v),
            "picture {index} differs from the intact recording"
        );
        previous = Some(index);
        count += 1;
    }
    assert!(gaps > 0 && count < 900, "queue overflow was not exercised");
    assert_eq!(previous, Some(899), "no complete recovery before EOF");
}
