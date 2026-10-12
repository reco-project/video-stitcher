//! Real loopback RTSP/TCP decode and source lifecycle checks.

mod audio;
mod peer;
mod pipeline;
use super::*;
use peer::Peer;

fn config() -> RtspConfig {
    RtspConfig {
        fps: (10, 1),
        synchronization: Config {
            wait_us: 150_000,
            skew_us: 30_000,
            max_age_us: 350_000,
            capacity: 4,
        },
        open_timeout: Duration::from_secs(3),
        read_timeout: Duration::from_millis(500),
        software_decode: true,
    }
}

fn input(peer: &Peer) -> RtspInput {
    RtspInput {
        url: peer.url.clone(),
        timestamp_offset_us: 0,
    }
}

#[test]
fn rational_slots_preserve_phase_when_skipping_and_reject_overflow() {
    for fps in [(30000, 1001), (24000, 1001), (120, 1)] {
        for origin in [0, 8_000_000_000_000_000] {
            let mut slot = origin;
            let mut remainder = 0;
            for tick in 1..=100_000i64 {
                (slot, remainder) = advance_slot(slot, remainder, fps, slot).unwrap();
                assert_eq!(
                    slot - origin,
                    tick * 1_000_000 * i64::from(fps.1) / i64::from(fps.0)
                );
            }
            let after = slot + 999_999;
            let (next, _) = advance_slot(slot, remainder, fps, after).unwrap();
            assert!(next > after);
            assert!(next - after <= (1_000_000 * i64::from(fps.1) / i64::from(fps.0)) + 1);
        }
    }
    assert!(advance_slot(i64::MAX, 0, (30, 1), i64::MAX).is_err());
    assert!(advance_slot(10, 0, (30, 1), 9).is_err());
}

#[test]
fn wrong_media_offsets_are_not_repaired_by_arrival_order() {
    let peers = [Peer::mapped(30, 0, None), Peer::mapped(30, 7_000_000, None)];
    let mut source = RtspSource::open(
        peers.iter().map(input).collect(),
        config(),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert!(
        source.next_frame().is_err(),
        "an unexplained seven-second clock difference is not synchronization"
    );
}

#[test]
fn validates_before_opening_and_does_not_echo_credentials() {
    let mut options = config();
    options.fps = (0, 1);
    let result = RtspSource::open(
        vec![RtspInput {
            url: "rtsp://private:secret@127.0.0.1:1/stream".into(),
            timestamp_offset_us: 0,
        }],
        options,
        Arc::new(AtomicBool::new(false)),
    );
    let Err(error) = result else {
        panic!("invalid cadence accepted")
    };
    assert!(error.to_string().contains("cadence"));
    assert!(!error.to_string().contains("secret"));
    for url in [
        "file:///tmp/video",
        "https://example.invalid",
        "rtsp://",
        "rtsp://127.0.0.1/\0",
    ] {
        assert!(
            RtspSource::open(
                vec![RtspInput {
                    url: url.into(),
                    timestamp_offset_us: 0
                }],
                config(),
                Arc::new(AtomicBool::new(false))
            )
            .is_err()
        );
    }
}

#[test]
fn opening_silent_peer_obeys_deadline_and_cancellation() {
    for cancel in [false, true] {
        let peer = Peer::silent();
        let interrupted = Arc::new(AtomicBool::new(false));
        let token = interrupted.clone();
        let canceller = cancel.then(|| {
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(60));
                token.store(true, Ordering::Relaxed);
            })
        });
        let mut options = config();
        options.open_timeout = Duration::from_millis(200);
        let start = Instant::now();
        assert!(RtspSource::open(vec![input(&peer)], options, interrupted).is_err());
        assert!(start.elapsed() < Duration::from_secs(2));
        if let Some(thread) = canceller {
            thread.join().unwrap();
        }
    }
}

#[test]
fn actual_rtsp_decode_preserves_cadence_and_cancels() {
    let peer = Peer::video(30, None);
    let interrupted = Arc::new(AtomicBool::new(false));
    let mut source = RtspSource::open(vec![input(&peer)], config(), interrupted.clone()).unwrap();
    assert_eq!(source.info().camera_sizes, [(64, 32)]);
    assert_eq!(source.info().fps_rational, Some((10, 1)));
    assert!(!source.is_gpu_resident());
    let mut last = None;
    for _ in 0..8 {
        let frame = source.next_frame().unwrap().unwrap();
        let pts = frame.pts_us.unwrap();
        assert!(last.is_none_or(|last| pts > last));
        let capture = frame.capture.unwrap();
        assert!(capture.at_us >= capture.times_us[0]);
        assert!(capture.at_us - capture.times_us[0] <= 350_000);
        let FrameSet::Yuv420p(images) = frame.images else {
            panic!("expected retained host images")
        };
        assert_eq!(images[0].y.len(), 64 * 32);
        assert!(images[0].y.iter().any(|&pixel| pixel > 32));
        last = Some(pts);
    }
    interrupted.store(true, Ordering::Relaxed);
    assert!(source.next_frame().unwrap().is_none());
    assert!(source.is_exhausted());
    let start = Instant::now();
    drop(source);
    assert!(start.elapsed() < Duration::from_secs(2));
}

#[test]
fn interactive_polling_keeps_pending_slots_and_reports_cancellation() {
    let peer = Peer::video(40, None);
    let interrupted = Arc::new(AtomicBool::new(false));
    let mut source = RtspSource::open(vec![input(&peer)], config(), interrupted.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(4);
    let mut frames = Vec::new();
    let mut pending = 0;
    while frames.len() < 8 {
        assert!(Instant::now() < deadline);
        let before = Instant::now();
        let frame = source.try_next_frame().unwrap();
        assert!(
            before.elapsed() < Duration::from_millis(100),
            "poll waited for media"
        );
        if let Some(frame) = frame {
            frames.push(frame.pts_us.unwrap());
        } else {
            pending += 1;
            assert!(!source.is_exhausted());
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    assert!(pending > 20);
    assert!(frames.windows(2).all(|pair| pair[1] > pair[0]));
    assert!(source.next_frame().unwrap().unwrap().pts_us.unwrap() > *frames.last().unwrap());
    interrupted.store(true, Ordering::Relaxed);
    assert!(source.try_next_frame().unwrap().is_none());
    assert!(source.is_exhausted());
    assert!(source.next_frame().unwrap().is_none());
}

#[test]
fn interactive_stall_errors_without_becoming_end_of_file() {
    let peer = Peer::video(5, None);
    let mut source = RtspSource::open(vec![input(&peer)], config(), Arc::default()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    let error = loop {
        assert!(Instant::now() < deadline);
        match source.try_next_frame() {
            Ok(_) => assert!(!source.is_exhausted()),
            Err(error) => break error.to_string(),
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    assert_eq!(source.next_frame().err().unwrap().to_string(), error);
    assert_eq!(source.try_next_frame().err().unwrap().to_string(), error);
}

#[test]
fn interactive_pause_expires_pending_slot_without_replaying_backlog() {
    let peer = Peer::video(50, None);
    let mut source = RtspSource::open(vec![input(&peer)], config(), Arc::default()).unwrap();
    let first = source.next_frame().unwrap().unwrap().pts_us.unwrap();
    // Leave a slot pending, just as a render callback can before pausing.
    while source.try_next_frame().unwrap().is_some() {}
    assert!(source.waiting);
    std::thread::sleep(Duration::from_millis(850));
    let resumed = source.next_frame().unwrap().unwrap();
    assert!(resumed.pts_us.unwrap() - first >= 800_000);
    let capture = resumed.capture.unwrap();
    assert!(source.now_us().unwrap() - capture.at_us < 200_000);
}

#[test]
fn interactive_drop_cancels_and_defers_worker_join_to_scope() {
    let peer = Peer::video(30, None);
    let mut source = RtspSource::open(vec![input(&peer)], config(), Arc::default()).unwrap();
    let stop = source.stop.clone();
    let (release, wait) = std::sync::mpsc::channel();
    source
        .workers
        .push(DecoderTask::new(std::thread::spawn(move || {
            wait.recv_timeout(Duration::from_secs(3)).unwrap();
        })));
    let cleanup = crate::DecoderCleanup::default();
    source.defer_cleanup(&cleanup);
    drop(source);
    assert!(stop.load(Ordering::Relaxed));
    let (joined, done) = std::sync::mpsc::channel();
    let owner = std::thread::spawn(move || {
        drop(cleanup);
        joined.send(()).unwrap();
    });
    assert!(done.recv_timeout(Duration::from_millis(30)).is_err());
    release.send(()).unwrap();
    done.recv_timeout(Duration::from_secs(2)).unwrap();
    owner.join().unwrap();
}

#[test]
fn connected_stall_fails_and_failure_is_sticky() {
    let peer = Peer::video(5, None);
    let mut source = RtspSource::open(
        vec![input(&peer)],
        config(),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let start = Instant::now();
    let error = loop {
        match source.next_frame() {
            Ok(Some(_)) => {}
            Ok(None) => panic!("connected stalled producer is not EOF"),
            Err(error) => break error,
        }
        assert!(start.elapsed() < Duration::from_secs(3));
    };
    assert_eq!(
        error.to_string(),
        source
            .next_frame()
            .err()
            .expect("sticky failure")
            .to_string()
    );
}

#[test]
fn disconnected_live_camera_is_not_successful_completion() {
    let peer = Peer::video(30, None);
    let mut source = RtspSource::open(
        vec![input(&peer)],
        config(),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert!(source.next_frame().unwrap().is_some());
    drop(peer);
    let start = Instant::now();
    loop {
        match source.next_frame() {
            Err(_) => break,
            Ok(None) => panic!("disconnected live camera silently reported normal completion"),
            Ok(Some(_)) => {}
        }
        assert!(start.elapsed() < Duration::from_secs(2));
    }
}

#[test]
fn cancellation_interrupts_an_active_socket_read_before_its_long_deadline() {
    // Supply enough timed images to finish demux probing before the peer stalls.
    let peer = Peer::video(30, None);
    let interrupted = Arc::new(AtomicBool::new(false));
    let mut options = config();
    options.read_timeout = Duration::from_secs(10);
    let source = RtspSource::open(vec![input(&peer)], options, interrupted.clone()).unwrap();
    std::thread::sleep(Duration::from_millis(3100));
    let start = Instant::now();
    interrupted.store(true, Ordering::Relaxed);
    drop(source);
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "socket read ignored cancellation"
    );
}

#[test]
fn mapped_cameras_preserve_missing_capture_and_age_bounds() {
    let peers = [
        Peer::mapped(40, 0, None),
        Peer::mapped(40, 7_000_000, Some(12)),
    ];
    let mut source = RtspSource::open(
        vec![
            input(&peers[0]),
            RtspInput {
                timestamp_offset_us: -7_000_000,
                ..input(&peers[1])
            },
        ],
        config(),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let mut selected = Vec::new();
    for _ in 0..20 {
        let frame = source.next_frame().unwrap().unwrap();
        let times = frame.capture.unwrap();
        assert_eq!(times.times_us.len(), 2);
        for &time in &times.times_us {
            assert!(times.at_us - time <= 350_000);
            assert_eq!((time + source.media_origin_us) % 100_000, 0);
        }
        assert_ne!(times.times_us[1] + source.media_origin_us, 1_200_000);
        selected.push(times.times_us);
    }
    // Decode latency may make the adapter skip a late slot instead of holding
    // its previous image. Either way, the missing capture must not be fabricated.
    assert!(selected.first().unwrap()[1] + source.media_origin_us < 1_200_000);
    assert!(selected.last().unwrap()[1] + source.media_origin_us > 1_200_000);
    assert!(
        selected
            .windows(2)
            .all(|pair| (0..2).all(|i| pair[1][i] >= pair[0][i]))
    );
}

#[test]
fn clock_reset_cannot_be_evicted_from_full_queue() {
    let peer = Peer::video(30, Some(7));
    let mut source = RtspSource::open(
        vec![input(&peer)],
        config(),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    std::thread::sleep(Duration::from_millis(1000));
    assert!(source.shared.lock().inputs[0].images.len() <= config().synchronization.capacity);
    let error = source
        .next_frame()
        .err()
        .expect("clock reset must survive dropped images");
    assert!(error.to_string().contains("clock reset"), "{error}");
}

#[test]
fn slow_consumer_skips_backlog_and_drop_releases_images() {
    let peer = Peer::video(30, None);
    let mut source = RtspSource::open(
        vec![input(&peer)],
        config(),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let first = source.next_frame().unwrap().unwrap();
    let first_pts = first.pts_us.unwrap();
    let FrameSet::Yuv420p(images) = first.images else {
        panic!("host images")
    };
    let owner = Arc::downgrade(&images[0]);
    drop(images);
    std::thread::sleep(Duration::from_millis(900));
    assert!(source.shared.lock().inputs[0].images.len() <= config().synchronization.capacity);
    let resumed = source.next_frame().unwrap().unwrap();
    assert!(resumed.pts_us.unwrap() - first_pts >= 800_000);
    assert!(owner.upgrade().is_none());
    let FrameSet::Yuv420p(images) = resumed.images else {
        panic!("host images")
    };
    let owner = Arc::downgrade(&images[0]);
    drop(images);
    drop(source);
    assert!(owner.upgrade().is_none());
}

#[test]
fn second_input_open_failure_stops_other_worker() {
    let peers = [Peer::video(30, None), Peer::silent()];
    let mut options = config();
    options.open_timeout = Duration::from_millis(350);
    let start = Instant::now();
    assert!(
        RtspSource::open(
            peers.iter().map(input).collect(),
            options,
            Arc::new(AtomicBool::new(false))
        )
        .is_err()
    );
    assert!(start.elapsed() < Duration::from_secs(2));
}
