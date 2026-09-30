//! Host timing survives delayed processing; consumers share enriched snapshots.

use super::*;
use crate::telemetry::{FrameTiming, TelemetrySink, TelemetrySnapshot};
use std::time::{Duration, Instant};

struct MeasuredDetector(Arc<Mutex<Duration>>);
impl Detector for MeasuredDetector {
    fn name(&self) -> &'static str {
        "measured"
    }
    fn detect(
        &mut self,
        camera: usize,
        _: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(2));
        *self.0.lock().unwrap() += start.elapsed();
        Ok(vec![Detection {
            camera,
            class_id: 1,
            confidence: 0.9,
            center_x: 0.5,
            center_y: 0.5,
            width: 0.1,
            height: 0.1,
        }])
    }
}

struct Ball(Arc<Mutex<Duration>>);
impl Tracker for Ball {
    fn class_id(&self) -> u16 {
        1
    }
    fn update(&mut self, _: &[MappedDetection], _: &[usize], _: f64) -> Vec<TrackedEntity> {
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(1));
        *self.0.lock().unwrap() += start.elapsed();
        vec![TrackedEntity {
            id: 0,
            class_id: 1,
            yaw: 0.0,
            pitch: 0.0,
            confidence: 1.0,
            state: TrackState::Tracking,
            age_frames: 1,
            origin: 0,
        }]
    }
}

struct Hold(Arc<Mutex<Duration>>);
impl Panner for Hold {
    fn decide(&mut self, _: &WorldState, ctx: &PanContext<'_>) -> WorldPose {
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(1));
        *self.0.lock().unwrap() += start.elapsed();
        ctx.previous_pose
    }
}

struct Snapshots(Arc<Mutex<Vec<TelemetrySnapshot>>>);
impl TelemetrySink for Snapshots {
    fn on_snapshot(&mut self, snapshot: &TelemetrySnapshot) {
        self.0.lock().unwrap().push(snapshot.clone());
    }
}

struct SlowOutput(Nv12CheckingSink);
impl OutputSink for SlowOutput {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn wants(&self) -> SinkInput {
        self.0.wants()
    }
    fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
        std::thread::sleep(Duration::from_millis(2));
        self.0.consume(frame)
    }
    fn finish(&mut self) -> Result<(), SinkError> {
        self.0.finish()
    }
}

#[test]
fn immediate_and_buffered_count_sparse_work_and_observations_once() {
    for depth in [0, 1, 12] {
        let mut session = build_cpu_session();
        session.core_mut().set_constrained_look(false);
        session.set_lookahead(depth, usize::MAX);
        let measured = Arc::new(Mutex::new(Duration::ZERO));
        session.set_detector(Box::new(MeasuredDetector(measured.clone())));
        session.set_detection_interval(2);
        let tracking = Arc::new(Mutex::new(Duration::ZERO));
        session.set_ball_tracker(Box::new(Ball(tracking.clone())));
        session.set_panner(Box::new(Hold(tracking.clone())));
        let snapshots = Arc::new(Mutex::new(Vec::new()));
        session.set_telemetry_sink(Box::new(Snapshots(snapshots.clone())), 1);
        session
            .run(
                &mut MockSource::new(6),
                u64::MAX,
                &AtomicBool::new(false),
                None,
            )
            .unwrap();
        let snapshot = session.telemetry_snapshot();
        let observed_ms = measured.lock().unwrap().as_secs_f32() * 1000.0;
        assert_eq!(snapshot.frames_processed, 6);
        assert_eq!(snapshot.total_detections, 12);
        assert_eq!(snapshot.ball_presence_pct, 100.0);
        assert!(snapshot.avg_detection_ms * 6.0 >= observed_ms - 0.001);
        assert!(snapshot.avg_tracking_ms > 0.0);
        assert!(
            snapshot.avg_tracking_ms * 6.0
                >= tracking.lock().unwrap().as_secs_f32() * 1000.0 - 0.001
        );
        assert!(snapshot.avg_total_ms >= snapshot.avg_detection_ms);
        let snapshots = snapshots.lock().unwrap();
        assert_eq!(snapshots.len(), 6);
        assert_eq!(snapshots.last().unwrap().total_detections, 12);
        // Known skipped work contributes zero rather than disappearing from the average.
        assert!(snapshots[1].avg_detection_ms < snapshots[0].avg_detection_ms);
    }
}

#[test]
fn buffered_frame_retains_injected_producer_timing() {
    let mut session = build_cpu_session();
    session.core_mut().set_constrained_look(false);
    let frame = super::super::frame_buffer::BufferedFrame {
        frame: Some(solid_frame()),
        pts_us: None,
        world_state: WorldState::default(),
        detections: Vec::new(),
        elapsed_ms: 0.0,
        retained_bytes: 0,
        timing: FrameTiming {
            decode: Some(Duration::from_millis(3)),
            upload: Some(Duration::from_millis(5)),
            detection: Some(Duration::from_millis(12)),
            tracking: Some(Duration::from_millis(7)),
            total: Some(Duration::from_millis(40)),
            ..Default::default()
        },
        #[cfg(feature = "gpu")]
        vram_slot: None,
    };
    session
        .process_frame_any(
            frame.frame.as_ref(),
            Duration::ZERO,
            frame.pts_us,
            frame.timing,
            Instant::now(),
            &FrameLoopContext {
                tracking_complete: true,
                frame_seconds: 1.0 / 30.0,
                ..Default::default()
            },
        )
        .unwrap();
    let snapshot = session.telemetry_snapshot();
    assert_eq!(snapshot.avg_decode_ms, 3.0);
    assert_eq!(snapshot.avg_upload_ms, 5.0);
    assert_eq!(snapshot.avg_detection_ms, 12.0);
    assert!(snapshot.avg_tracking_ms >= 7.0);
    assert!(snapshot.avg_total_ms >= 40.0);
}

#[test]
fn pushed_snapshots_include_output_worker_statistics() {
    let mut session = build_cpu_session();
    session.core_mut().set_constrained_look(false);
    session
        .add_sink(
            Box::new(SlowOutput(Nv12CheckingSink {
                frames: Arc::new(AtomicU64::new(0)),
            })),
            crate::session::SinkOptions::threaded(1),
        )
        .unwrap();
    let received = Arc::new(Mutex::new(Vec::new()));
    session.set_telemetry_sink(Box::new(Snapshots(received.clone())), 1);
    session
        .run(
            &mut MockSource::new(6),
            u64::MAX,
            &AtomicBool::new(false),
            None,
        )
        .unwrap();
    session.finish().unwrap();
    let polled = session.telemetry_snapshot();
    let snapshots = received.lock().unwrap();
    let pushed = snapshots.last().unwrap();
    assert_eq!(pushed.frames_processed, polled.frames_processed);
    assert!(pushed.avg_encode_worker_ms > 0.0);
    assert_eq!(pushed.avg_encode_worker_ms, polled.avg_encode_worker_ms);
    assert_eq!(pushed.backpressure_stalls, polled.backpressure_stalls);
    assert_eq!(pushed.backpressure_ms, polled.backpressure_ms);
    assert!(snapshots[..6].iter().any(|s| s.avg_encode_worker_ms > 0.0));
    drop(snapshots);
    session.finish().unwrap();
    let repeated = session.telemetry_snapshot();
    assert_eq!(repeated.backpressure_stalls, polled.backpressure_stalls);
    assert_eq!(repeated.backpressure_ms, polled.backpressure_ms);
}

#[test]
fn cancellation_and_output_limits_do_not_count_discarded_lookahead() {
    for cancelled in [false, true] {
        let mut session = build_cpu_session();
        session.core_mut().set_constrained_look(false);
        session.set_lookahead(12, usize::MAX);
        let result = session
            .run(
                &mut MockSource::new(6),
                2,
                &AtomicBool::new(cancelled),
                None,
            )
            .unwrap();
        assert_eq!(
            session.telemetry_snapshot().frames_processed,
            result.frames_processed
        );
        assert_eq!(result.frames_processed, if cancelled { 0 } else { 2 });
    }
}
