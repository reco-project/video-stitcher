//! Integration tests for [`StitchSession`] frame loop.
//!
//! GPU-dependent tests are marked `#[ignore]` so they can be skipped in CI
//! environments without a GPU. Run them explicitly with:
//! ```bash
//! cargo test -p reco-core -- --ignored
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::StitchSession;
mod capture_identity;
mod coverage;
mod cpu_detection;
mod lookahead;
#[cfg(all(feature = "gpu", target_os = "linux"))]
mod nvmm;
mod telemetry;
mod timing;
use super::types::*;
use crate::calibration::{Calibration, Framing, Lens};
use crate::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
use crate::detect::mapped_detection::MappedDetection;
use crate::detect::panner::{PanContext, Panner};
use crate::detect::tracker::{TrackState, TrackedEntity, Tracker, WorldState};
use crate::geometry::CameraIndex;
use crate::geometry::WorldPose;
use crate::projection::LShape;
use crate::render::viewport::ViewportSize;
use crate::sink::{OutputFrame, OutputSink, SinkError, SinkInput};
use crate::source::{FrameSet, FrameSource, SourceError, SourceInfo, Yuv420pData};

// ─── Helpers ───────────────────────────────────────────────────────────

/// Small test dimensions to keep GPU allocations minimal.
const W: u32 = 64;
const H: u32 = 64;

/// Create a minimal valid calibration for 64x64 frames.
fn test_calibration() -> Calibration {
    let cam = || Lens::fisheye(W, H, 32.0, 32.0, 32.0, 32.0, [0.0; 4]);
    Calibration::new(
        vec![cam(), cam()],
        LShape {
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        },
        Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        },
    )
}

/// Create a valid YUV420P stereo frame pair of solid gray.
fn solid_frame() -> FrameSet {
    let y_size = (W * H) as usize;
    let uv_size = ((W / 2) * (H / 2)) as usize;
    let yuv = Yuv420pData {
        size: (W, H),
        y: vec![128u8; y_size],
        u: vec![128u8; uv_size],
        v: vec![128u8; uv_size],
    };
    FrameSet::Yuv420p(vec![Arc::new(yuv.clone()), Arc::new(yuv)])
}

// ─── MockSource ────────────────────────────────────────────────────────

/// Mock frame source that returns N solid-color YUV420P frame pairs.
struct MockSource {
    remaining: u64,
}

impl MockSource {
    fn new(frame_count: u64) -> Self {
        Self {
            remaining: frame_count,
        }
    }
}

impl FrameSource for MockSource {
    fn info(&self) -> SourceInfo {
        SourceInfo {
            camera_sizes: vec![(W, H); 2],
            fps: 30.0,
            fps_rational: Some((30, 1)),
            total_frames: None,
        }
    }

    fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        Ok(Some(crate::source::SourceFrame::untimed(solid_frame())))
    }
}

#[test]
fn reused_session_applies_both_full_and_limited_source_ranges() {
    struct RangeSource(bool);
    impl FrameSource for RangeSource {
        fn info(&self) -> SourceInfo {
            MockSource::new(0).info()
        }
        fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
            Ok(None)
        }
        fn is_full_range(&self) -> bool {
            self.0
        }
    }
    let mut session = build_cpu_session();
    for full_range in [true, false] {
        session
            .run(
                &mut RangeSource(full_range),
                1,
                &AtomicBool::new(false),
                None,
            )
            .unwrap();
        assert!(matches!(
            &session.core.executor,
            crate::stitch::Executor::Cpu(executor) if executor.color.full_range == full_range
        ));
        assert_eq!(session.source_full_range, full_range);
    }
}

#[test]
#[ignore = "requires GPU readback"]
#[cfg(feature = "gpu")]
fn reused_session_replaces_source_rotation_without_double_flipping_host_frames() {
    struct EmptySource {
        native: bool,
        rotation: i32,
    }
    impl FrameSource for EmptySource {
        fn info(&self) -> SourceInfo {
            MockSource::new(0).info()
        }
        fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
            Ok(None)
        }
        fn is_gpu_resident(&self) -> bool {
            self.native
        }
        fn left_rotation(&self) -> i32 {
            self.rotation
        }
        fn right_rotation(&self) -> i32 {
            self.rotation
        }
    }
    fn image(session: &mut StitchSession) -> Vec<u8> {
        let mut frames = solid_frame();
        let FrameSet::Yuv420p(cameras) = &mut frames else {
            unreachable!()
        };
        for camera in cameras {
            for (i, y) in Arc::make_mut(camera).y.iter_mut().enumerate() {
                *y = 32 + (i % W as usize * 3) as u8;
            }
        }
        let pipeline = &mut session.gpu_exec().pipeline;
        let gpu = pipeline.gpu().clone();
        let cmd = pipeline
            .render_frame_set(&frames, crate::geometry::RenderPose::default())
            .unwrap();
        let mut readback = crate::gpu::rgba_readback::RgbaReadback::new(&gpu, W, H).unwrap();
        readback
            .readback(&gpu, pipeline.render_target(), cmd)
            .unwrap();
        readback.flush_pending(&gpu).unwrap().unwrap().to_vec()
    }
    let mut session = build_test_session(None, None, 1).unwrap();
    let baseline = image(&mut session);
    for (native, rotation) in [(true, 180), (true, 0), (true, 180), (false, 180)] {
        session
            .run(
                &mut EmptySource { native, rotation },
                1,
                &AtomicBool::new(false),
                None,
            )
            .unwrap();
        let flipped = native && rotation == 180;
        assert_eq!(
            image(&mut session) != baseline,
            flipped,
            "native={native}, rotation={rotation}"
        );
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        assert_eq!(
            (session.left_rotation, session.right_rotation),
            if native { (rotation, rotation) } else { (0, 0) }
        );
    }
}

// ─── MockDetector ──────────────────────────────────────────────────────

/// Mock detector that returns canned detections and counts calls.
struct MockDetector {
    detections: Vec<Detection>,
    call_count: Arc<AtomicU64>,
}

impl MockDetector {
    fn new(detections: Vec<Detection>, call_count: Arc<AtomicU64>) -> Self {
        Self {
            detections,
            call_count,
        }
    }
}

impl Detector for MockDetector {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn detect(
        &mut self,
        _camera: CameraIndex,
        _frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(self.detections.clone())
    }
}

// ─── MockTracker / MockPanner ──────────────────────────────────────────

/// Mock tracker that records how many detections it received each
/// frame and returns a single "ball" entity.
#[cfg(feature = "gpu")]
struct MockTracker {
    log: Arc<Mutex<Vec<TrackerSnapshot>>>,
    class_id: u16,
}

/// Snapshot of what a tracker saw on a single `update()` call.
#[derive(Debug, PartialEq)]
#[cfg(feature = "gpu")]
struct TrackerSnapshot {
    frame_index: u64,
    detection_count: usize,
    fresh: bool,
}

#[cfg(feature = "gpu")]
impl MockTracker {
    fn new(class_id: u16, log: Arc<Mutex<Vec<TrackerSnapshot>>>) -> Self {
        Self { class_id, log }
    }
}

#[cfg(feature = "gpu")]
impl Tracker for MockTracker {
    fn update(
        &mut self,
        detections: &[MappedDetection],
        fresh_cameras: &[crate::geometry::CameraIndex],
        _timestamp_ms: f64,
    ) -> Vec<TrackedEntity> {
        let fresh = !fresh_cameras.is_empty();
        // Session bumps frame_count after dispatch, so `log.len()` is
        // the per-call frame_index we see.
        let frame_index = self.log.lock().unwrap().len() as u64;
        self.log.lock().unwrap().push(TrackerSnapshot {
            frame_index,
            detection_count: detections.len(),
            fresh,
        });
        vec![TrackedEntity {
            id: 0,
            class_id: self.class_id,
            yaw: 0.0,
            pitch: 0.0,
            confidence: 1.0,
            state: TrackState::Tracking,
            age_frames: 1,
            origin: 0,
        }]
    }

    fn class_id(&self) -> u16 {
        self.class_id
    }
}

/// Mock panner that returns a fixed position regardless of the world.
struct MockPanner {
    position: WorldPose,
}

impl Panner for MockPanner {
    fn decide(&mut self, _world: &WorldState, _ctx: &PanContext<'_>) -> WorldPose {
        self.position
    }
}

// ─── MockEncoder ───────────────────────────────────────────────────────

/// Mock encoder that records submitted frames (counts and dimensions).
struct MockEncoder {
    submitted: Arc<AtomicU64>,
}

impl MockEncoder {
    fn new(counter: Arc<AtomicU64>) -> Self {
        Self { submitted: counter }
    }
}

impl OutputSink for MockEncoder {
    fn name(&self) -> &str {
        "mock-encoder"
    }

    fn wants(&self) -> SinkInput {
        SinkInput::CpuBytes(crate::sink::PixelFormat::Nv12)
    }

    fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
        assert!(frame.width > 0, "frame width must be positive");
        assert!(frame.height > 0, "frame height must be positive");
        assert!(!frame.data.is_empty(), "frame data must not be empty");
        self.submitted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

// ─── Recording Panner ──────────────────────────────────────────────────

/// Panner that logs the dispatch inputs it sees (frame index +
/// previous-pose threading) and answers a deterministic pose per
/// frame. Two entry points driving one AI stack must produce
/// identical logs.
#[cfg(feature = "gpu")]
struct RecordingPanner {
    log: Arc<Mutex<Vec<(u64, WorldPose)>>>,
}

#[cfg(feature = "gpu")]
impl Panner for RecordingPanner {
    fn decide(&mut self, _world: &WorldState, ctx: &PanContext<'_>) -> WorldPose {
        self.log
            .lock()
            .unwrap()
            .push((ctx.frame_index, ctx.previous_pose));
        WorldPose {
            yaw: 0.001 * ctx.frame_index as f32,
            pitch: -0.0005 * ctx.frame_index as f32,
            ..Default::default()
        }
    }
}

// ─── NaN Panner ────────────────────────────────────────────────────────

/// Panner that returns invalid yaw/pitch to exercise presentation validation.
struct NanPanner;

impl Panner for NanPanner {
    fn decide(&mut self, _world: &WorldState, _ctx: &PanContext<'_>) -> WorldPose {
        WorldPose {
            yaw: f32::NAN,
            pitch: f32::NAN,
            ..Default::default()
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

/// Build a session using the builder pattern with test defaults.
/// Trackers and panners are attached post-build (they have their own
/// setters and aren't part of the builder API).
#[cfg(feature = "gpu")]
fn build_test_session(
    encoder: Option<Box<dyn OutputSink>>,
    detector: Option<Box<dyn Detector>>,
    detection_interval: u64,
) -> Result<StitchSession, SessionError> {
    let mut builder = StitchSession::builder()
        .calibration(test_calibration())
        .source_sizes(vec![(W, H); 2])
        .viewport_size(ViewportSize {
            width: 64,
            height: 64,
        })
        .detection_interval(detection_interval);

    if let Some(enc) = encoder {
        builder = builder.sink(enc, crate::session::SinkOptions::threaded(2));
    }
    if let Some(det) = detector {
        builder = builder.detector(det);
    }

    builder.build()
}

#[test]
#[ignore = "requires GPU readback"]
#[cfg(feature = "gpu")]
fn finish_finalizes_all_sinks_when_pending_delivery_fails() {
    struct FinalizationProbe {
        fail_consume: bool,
        fail_finish: bool,
        finishes: Arc<AtomicU64>,
    }
    impl OutputSink for FinalizationProbe {
        fn name(&self) -> &str {
            "finalization-probe"
        }
        fn wants(&self) -> SinkInput {
            SinkInput::CpuBytes(crate::sink::PixelFormat::Nv12)
        }
        fn consume(&mut self, _: OutputFrame<'_>) -> Result<(), SinkError> {
            if self.fail_consume {
                Err(SinkError::Consume {
                    frame_index: None,
                    reason: "pending delivery failure".into(),
                })
            } else {
                Ok(())
            }
        }
        fn finish(&mut self) -> Result<(), SinkError> {
            self.finishes.fetch_add(1, Ordering::Relaxed);
            if self.fail_finish {
                Err(SinkError::Finish {
                    reason: "secondary finalization failure".into(),
                })
            } else {
                Ok(())
            }
        }
    }
    for fail_finish in [false, true] {
        let mut session = build_test_session(None, None, 1).unwrap();
        session.core_mut().set_constrained_look(false);
        let finishes = Arc::new(AtomicU64::new(0));
        for fail_consume in [true, false] {
            session
                .add_sink(
                    Box::new(FinalizationProbe {
                        fail_consume,
                        fail_finish,
                        finishes: finishes.clone(),
                    }),
                    super::SinkOptions {
                        delivery: super::SinkDelivery::Inline,
                        on_error: super::SinkErrorPolicy::Abort,
                    },
                )
                .unwrap();
        }
        session
            .run(&mut MockSource::new(2), 2, &AtomicBool::new(false), None)
            .unwrap();
        let error = session.finish().unwrap_err();
        assert!(
            error.to_string().contains("pending delivery failure"),
            "{error}"
        );
        assert_eq!(finishes.load(Ordering::Relaxed), 2);
        session.finish().unwrap();
        assert_eq!(finishes.load(Ordering::Relaxed), 2);
    }
}

#[test]
#[ignore] // requires GPU
#[cfg(feature = "gpu")]
fn basic_frame_loop() {
    let submitted = Arc::new(AtomicU64::new(0));
    let encoder = MockEncoder::new(Arc::clone(&submitted));

    let mut session = build_test_session(Some(Box::new(encoder)), None, 1).expect("session build");

    let mut source = MockSource::new(10);
    let interrupted = AtomicBool::new(false);

    let processed = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run");

    session.finish().expect("finish");

    assert_eq!(
        processed.frames_processed, 10,
        "should process all 10 frames"
    );

    // The async encoder receives frames with a 1-frame pipeline delay
    // (NV12 double-buffering), so we expect at least 9 frames submitted.
    let count = submitted.load(Ordering::Relaxed);
    assert!(
        (9..=10).contains(&count),
        "encoder should receive 9-10 frames, got {count}"
    );
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires GPU staging allocation"]
#[cfg(feature = "gpu")]
fn host_run_releases_native_history_before_budget_admission() {
    struct NativeEmptySource;
    impl FrameSource for NativeEmptySource {
        fn info(&self) -> SourceInfo {
            MockSource::new(0).info()
        }
        fn is_gpu_resident(&self) -> bool {
            true
        }
        fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
            Ok(None)
        }
    }

    for depth in [0, 2] {
        let mut session = build_test_session(None, None, 1).unwrap();
        session.core_mut().set_constrained_look(false);
        session.set_lookahead(5, usize::MAX);
        session
            .run(
                &mut NativeEmptySource,
                u64::MAX,
                &AtomicBool::new(false),
                None,
            )
            .unwrap();
        assert!(session.gpu_exec_ref().residency.pool.is_some());

        // Even a rejected host run must not retain the unused native history.
        session.set_lookahead(2, 1);
        assert!(matches!(
            session.run(
                &mut MockSource::new(3),
                u64::MAX,
                &AtomicBool::new(false),
                None
            ),
            Err(SessionError::Config(_))
        ));
        assert!(session.gpu_exec_ref().residency.pool.is_none());

        session.set_lookahead(depth, (W * H * 3 * 3) as usize);
        assert_eq!(
            session
                .run(
                    &mut MockSource::new(3),
                    u64::MAX,
                    &AtomicBool::new(false),
                    None
                )
                .unwrap()
                .frames_processed,
            3
        );
        assert!(session.gpu_exec_ref().residency.pool.is_none());
    }
}

#[test]
#[ignore] // requires GPU
#[cfg(feature = "gpu")]
fn tracker_receives_detections() {
    let call_count = Arc::new(AtomicU64::new(0));
    let canned = vec![
        Detection {
            camera: 0,
            class_id: 0,
            confidence: 0.9,
            center_x: 0.5,
            center_y: 0.5,
            width: 0.1,
            height: 0.1,
        },
        Detection {
            camera: 1,
            class_id: 1,
            confidence: 0.8,
            center_x: 0.3,
            center_y: 0.7,
            width: 0.05,
            height: 0.05,
        },
    ];

    let detector = MockDetector::new(canned, Arc::clone(&call_count));
    let tracker_log = Arc::new(Mutex::new(Vec::new()));
    let tracker = MockTracker::new(0, Arc::clone(&tracker_log));

    let submitted = Arc::new(AtomicU64::new(0));
    let encoder = MockEncoder::new(Arc::clone(&submitted));

    let mut session = build_test_session(Some(Box::new(encoder)), Some(Box::new(detector)), 1)
        .expect("session build");
    session.set_ball_tracker(Box::new(tracker));
    session.set_panner(Box::new(MockPanner {
        position: WorldPose::default(),
    }));

    let mut source = MockSource::new(3);
    let interrupted = AtomicBool::new(false);

    let processed = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run");

    session.finish().expect("finish");

    assert_eq!(processed.frames_processed, 3);

    let log = tracker_log.lock().unwrap();
    assert_eq!(log.len(), 3, "tracker should be updated once per frame");

    // The detector returns 2 detections per detect() call, and the session
    // calls detect() twice per frame (once for left, once for right camera).
    // So each frame produces 4 mapped detections the tracker observes.
    for (i, snapshot) in log.iter().enumerate() {
        assert_eq!(snapshot.frame_index, i as u64);
        assert!(snapshot.fresh);
        assert!(
            snapshot.detection_count > 0,
            "frame {i}: tracker should receive non-empty detections, got {}",
            snapshot.detection_count,
        );
    }
}

#[test]
#[ignore] // requires GPU
#[cfg(feature = "gpu")]
fn tracker_freshness_matches_inference_with_and_without_lookahead() {
    for lookahead in [0, 2] {
        let calls = Arc::new(AtomicU64::new(0));
        let detector = MockDetector::new(
            vec![Detection {
                camera: 0,
                class_id: 0,
                confidence: 0.9,
                center_x: 0.5,
                center_y: 0.5,
                width: 0.1,
                height: 0.1,
            }],
            calls.clone(),
        );
        let mut session = build_test_session(None, Some(Box::new(detector)), 3).unwrap();
        session.core_mut().set_constrained_look(false);
        session.set_lookahead(lookahead, usize::MAX);
        let log = Arc::new(Mutex::new(Vec::new()));
        session.set_ball_tracker(Box::new(MockTracker::new(0, log.clone())));
        session.set_panner(Box::new(MockPanner {
            position: WorldPose::default(),
        }));
        let processed = session
            .run(
                &mut MockSource::new(9),
                u64::MAX,
                &AtomicBool::new(false),
                None,
            )
            .unwrap();
        session.finish().unwrap();
        assert_eq!(processed.frames_processed, 9);
        assert_eq!(calls.load(Ordering::Relaxed), 6);
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 9);
        for (frame, observation) in log.iter().enumerate() {
            assert_eq!(
                observation.fresh,
                frame % 3 == 0,
                "lookahead={lookahead}, frame={frame}"
            );
            assert!(observation.detection_count > 0);
        }
    }
}

#[test]
#[ignore] // requires GPU
#[cfg(feature = "gpu")]
fn detection_interval_respected() {
    let call_count = Arc::new(AtomicU64::new(0));
    let canned = vec![Detection {
        camera: 0,
        class_id: 0,
        confidence: 0.9,
        center_x: 0.5,
        center_y: 0.5,
        width: 0.1,
        height: 0.1,
    }];

    let detector = MockDetector::new(canned, Arc::clone(&call_count));

    let submitted = Arc::new(AtomicU64::new(0));
    let encoder = MockEncoder::new(Arc::clone(&submitted));

    let mut session = build_test_session(Some(Box::new(encoder)), Some(Box::new(detector)), 3)
        .expect("session build");

    let mut source = MockSource::new(10);
    let interrupted = AtomicBool::new(false);

    let processed = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run");

    session.finish().expect("finish");

    assert_eq!(processed.frames_processed, 10);

    // With interval=3 and 10 frames (indices 0..9), detection runs at
    // frames 0, 3, 6, 9 = 4 times. Each detection call triggers detect()
    // twice (left + right camera), so total calls = 4 * 2 = 8.
    let calls = call_count.load(Ordering::Relaxed);
    // Allow some tolerance: 6..=10 calls covers 3-5 detection frames x 2 cameras.
    assert!(
        (6..=10).contains(&calls),
        "with interval=3 over 10 frames, expected 6-10 detector calls, got {calls}"
    );
}

#[test]
#[ignore] // requires GPU
#[cfg(feature = "gpu")]
fn nan_panner_is_rejected_before_output() {
    let submitted = Arc::new(AtomicU64::new(0));
    let encoder = MockEncoder::new(Arc::clone(&submitted));

    let mut session = build_test_session(Some(Box::new(encoder)), None, 1).expect("session build");
    session.set_panner(Box::new(NanPanner));

    let mut source = MockSource::new(5);
    let interrupted = AtomicBool::new(false);

    let error = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect_err("non-finite panner output must fail presentation validation");
    assert!(matches!(
        error,
        SessionError::Core(crate::core::types::StitchCoreError::Config(_))
    ));

    session.finish().expect("finish");

    assert_eq!(submitted.load(Ordering::Relaxed), 0);
}

#[test]
fn run_completion_records_the_observed_stop_reason() {
    for depth in [0, 2, 20] {
        for (frames, limit, cancel_after, expected, count) in [
            (3, 10, None, RunCompletion::Completed, 3),
            (10, 2, Some(2), RunCompletion::Completed, 2),
            (10, 10, Some(2), RunCompletion::Cancelled, 2),
            (10, 10, Some(0), RunCompletion::Cancelled, 0),
        ] {
            let mut session = build_cpu_session();
            session.set_lookahead(depth, usize::MAX);
            session.core_mut().set_constrained_look(false);
            let flag = Arc::new(AtomicBool::new(cancel_after == Some(0)));
            let signal = flag.clone();
            let result = session
                .run(
                    &mut MockSource::new(frames),
                    limit,
                    &flag,
                    Some(Box::new(move |p| {
                        if cancel_after == Some(p.frames_completed) {
                            signal.store(true, Ordering::Relaxed);
                        }
                    })),
                )
                .unwrap();
            flag.store(true, Ordering::Relaxed);
            session.finish().unwrap();
            assert_eq!(result.completion, expected);
            assert_eq!(result.frames_processed, count);
        }
    }
}

#[test]
fn cancellation_during_source_read_is_cancelled_without_extra_output() {
    struct InterruptedRead {
        source: MockSource,
        flag: Arc<AtomicBool>,
        output: Arc<AtomicU64>,
        at_stop: Arc<AtomicU64>,
        remaining: u64,
    }
    impl FrameSource for InterruptedRead {
        fn info(&self) -> SourceInfo {
            self.source.info()
        }
        fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
            if self.remaining == 0 {
                self.at_stop
                    .store(self.output.load(Ordering::Relaxed), Ordering::Relaxed);
                self.flag.store(true, Ordering::Relaxed);
                return Ok(None);
            }
            self.remaining -= 1;
            self.source.next_frame()
        }
    }
    for depth in [0, 1, 4] {
        for frames in [0, 2, 4, 7] {
            let mut session = build_cpu_session();
            session.core_mut().set_constrained_look(false);
            session.set_lookahead(depth, usize::MAX);
            let flag = Arc::new(AtomicBool::new(false));
            let output = Arc::new(AtomicU64::new(0));
            let at_stop = Arc::new(AtomicU64::new(u64::MAX));
            session
                .add_sink(
                    Box::new(MockEncoder::new(output.clone())),
                    super::SinkOptions::inline_lossy(),
                )
                .unwrap();
            let result = session
                .run(
                    &mut InterruptedRead {
                        source: MockSource::new(frames),
                        flag: flag.clone(),
                        output: output.clone(),
                        at_stop: at_stop.clone(),
                        remaining: frames,
                    },
                    u64::MAX,
                    &flag,
                    None,
                )
                .unwrap();
            assert_eq!(
                result.completion,
                RunCompletion::Cancelled,
                "depth={depth}, frames={frames}"
            );
            session.finish().unwrap();
            assert_eq!(result.frames_processed, at_stop.load(Ordering::Relaxed));
            assert_eq!(
                output.load(Ordering::Relaxed),
                at_stop.load(Ordering::Relaxed)
            );
        }
    }
}

#[test]
fn cancellation_does_not_hide_source_or_finalization_errors() {
    struct FailingSource(Arc<AtomicBool>);
    impl FrameSource for FailingSource {
        fn info(&self) -> SourceInfo {
            MockSource::new(0).info()
        }
        fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
            self.0.store(true, Ordering::Relaxed);
            Err(SourceError::Read {
                reason: "source failure".into(),
            })
        }
    }
    let flag = Arc::new(AtomicBool::new(false));
    for depth in [0, 4] {
        flag.store(false, Ordering::Relaxed);
        let mut session = build_cpu_session();
        session.set_lookahead(depth, usize::MAX);
        assert!(matches!(
            session.run(&mut FailingSource(flag.clone()), 10, &flag, None),
            Err(SessionError::Source(_))
        ));
    }

    struct FailingFinish;
    impl OutputSink for FailingFinish {
        fn name(&self) -> &str {
            "failing-finish"
        }
        fn wants(&self) -> SinkInput {
            SinkInput::CpuBytes(crate::sink::PixelFormat::Nv12)
        }
        fn consume(&mut self, _: OutputFrame<'_>) -> Result<(), SinkError> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), SinkError> {
            Err(SinkError::Finish {
                reason: "finish failure".into(),
            })
        }
    }
    let mut session = build_cpu_session();
    session
        .add_sink(Box::new(FailingFinish), super::SinkOptions::threaded(1))
        .unwrap();
    let result = session
        .run(&mut MockSource::new(10), 10, &flag, None)
        .unwrap();
    assert_eq!(result.completion, RunCompletion::Cancelled);
    assert!(session.finish().is_err());
}

#[test]
#[ignore] // requires GPU
#[cfg(feature = "gpu")]
fn gpu_run_completion_with_and_without_lookahead() {
    for lookahead in [0, 4] {
        for cancel_after in [None, Some(0), Some(2)] {
            let mut session = build_test_session(None, None, 1).unwrap();
            session.core_mut().set_constrained_look(false);
            session.set_lookahead(lookahead, usize::MAX);
            let flag = Arc::new(AtomicBool::new(cancel_after == Some(0)));
            let signal = flag.clone();
            let result = session
                .run(
                    &mut MockSource::new(12),
                    8,
                    &flag,
                    Some(Box::new(move |p| {
                        if cancel_after == Some(p.frames_completed) {
                            signal.store(true, Ordering::Relaxed);
                        }
                    })),
                )
                .unwrap();
            assert_eq!(
                result.completion,
                if cancel_after.is_some() {
                    RunCompletion::Cancelled
                } else {
                    RunCompletion::Completed
                }
            );
            assert_eq!(result.frames_processed, cancel_after.unwrap_or(8));
            session.finish().unwrap();
        }
    }
}

#[test]
#[ignore] // requires GPU; exercised by the GPU session completion CI step
#[cfg(feature = "gpu")]
fn gpu_run_completion_after_source_eof() {
    struct EofSource {
        source: MockSource,
        eof: Arc<AtomicBool>,
    }
    impl FrameSource for EofSource {
        fn info(&self) -> SourceInfo {
            self.source.info()
        }
        fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
            let frame = self.source.next_frame()?;
            if frame.is_none() {
                self.eof.store(true, Ordering::Relaxed);
            }
            Ok(frame)
        }
    }
    for frames in [3, 5, 7] {
        for cancel_after in [None, Some(frames - 1), Some(frames)] {
            let mut session = build_test_session(None, None, 1).unwrap();
            session.core_mut().set_constrained_look(false);
            session.set_lookahead(4, usize::MAX);
            let eof = Arc::new(AtomicBool::new(false));
            let observed_eof = eof.clone();
            let flag = Arc::new(AtomicBool::new(false));
            let signal = flag.clone();
            let result = session
                .run(
                    &mut EofSource {
                        source: MockSource::new(frames),
                        eof,
                    },
                    u64::MAX,
                    &flag,
                    Some(Box::new(move |p| {
                        if cancel_after == Some(p.frames_completed) {
                            assert!(
                                observed_eof.load(Ordering::Relaxed),
                                "cancellation must arrive after EOF"
                            );
                            signal.store(true, Ordering::Relaxed);
                        }
                    })),
                )
                .unwrap();
            assert_eq!(result.frames_processed, cancel_after.unwrap_or(frames));
            assert_eq!(
                result.completion,
                if cancel_after.is_some_and(|at| at < frames) {
                    RunCompletion::Cancelled
                } else {
                    RunCompletion::Completed
                }
            );
            session.finish().unwrap();
        }
    }
}

// ─── compute_frame_limit tests (no GPU required) ──────────────────────

#[test]
fn compute_frame_limit_both_none() {
    assert_eq!(compute_frame_limit(None, None, 30.0), u64::MAX);
}

#[test]
fn compute_frame_limit_negative_duration() {
    // Negative duration should be treated as "no duration limit".
    assert_eq!(compute_frame_limit(Some(-5.0), None, 30.0), u64::MAX);
}

#[test]
fn compute_frame_limit_zero_fps_uses_fallback() {
    // Zero fps should use the 30.0 fallback.
    let result = compute_frame_limit(Some(10.0), None, 0.0);
    // 10.0 * 30.0 = 300
    assert_eq!(result, 300);
}

#[test]
fn compute_frame_limit_both_set_min_wins() {
    // duration=2s at 30fps = 60 frames, max_frames=40 -> min(60,40) = 40
    assert_eq!(compute_frame_limit(Some(2.0), Some(40), 30.0), 40);

    // duration=1s at 30fps = 30 frames, max_frames=100 -> min(30,100) = 30
    assert_eq!(compute_frame_limit(Some(1.0), Some(100), 30.0), 30);
}

#[test]
fn compute_frame_limit_duration_only() {
    // 5s at 60fps = 300 frames
    assert_eq!(compute_frame_limit(Some(5.0), None, 60.0), 300);
}

#[test]
fn compute_frame_limit_max_frames_only() {
    assert_eq!(compute_frame_limit(None, Some(42), 30.0), 42);
}

#[test]
fn compute_frame_limit_negative_fps_uses_fallback() {
    // Negative fps should also trigger the 30.0 fallback.
    let result = compute_frame_limit(Some(10.0), None, -1.0);
    assert_eq!(result, 300);
}

/// The one-AI-stack guard, in two halves. (1) Entry-point parity:
/// the same scripted detector + tracker + panner, fed the same frames
/// through `StitchCore::submit_frame` and `StitchSession::run`,
/// must see identical inputs in identical order - frame indices,
/// previous-pose threading, detector call counts. (2) The pull side
/// must have driven the ENGINE's stack, observed through the core's
/// own detection cache: a session-side shadow component could still
/// produce a matching log, but it would leave the engine's state
/// untouched.
#[test]
#[ignore] // requires GPU
#[cfg(feature = "gpu")]
fn push_and_pull_share_one_ai_brain() {
    const FRAMES: u64 = 6;
    let canned = vec![Detection {
        camera: 0,
        class_id: 0,
        confidence: 0.9,
        center_x: 0.5,
        center_y: 0.5,
        width: 0.1,
        height: 0.1,
    }];

    // Pull side: session.run over a mock source.
    let pull_calls = Arc::new(AtomicU64::new(0));
    let pull_log = Arc::new(Mutex::new(Vec::new()));
    let pull_tracks = Arc::new(Mutex::new(Vec::new()));
    let mut session = build_test_session(
        None,
        Some(Box::new(MockDetector::new(
            canned.clone(),
            Arc::clone(&pull_calls),
        ))),
        3,
    )
    .expect("session build");
    session.set_ball_tracker(Box::new(MockTracker::new(0, pull_tracks.clone())));
    session.set_panner(Box::new(RecordingPanner {
        log: Arc::clone(&pull_log),
    }));
    let interrupted = AtomicBool::new(false);
    session
        .run(&mut MockSource::new(FRAMES), u64::MAX, &interrupted, None)
        .expect("run");

    // Half (2): the session's detection wrote the ENGINE's cache
    // (one entry per camera). A shadow session-side detection stack
    // would leave this empty even with a matching panner log.
    assert_eq!(
        session.core().last_detections().len(),
        2,
        "session detection must land in the engine's cache"
    );

    // Push side: the same frames submitted straight to an engine.
    let Some(gpu) = crate::stitch::test_support::gpu_or_skip() else {
        return;
    };
    let executor = crate::stitch::GpuExecutor::new(
        gpu,
        crate::stitch::GpuExecutorConfig {
            viewport_size: ViewportSize {
                width: 64,
                height: 64,
            },
            ..{
                let source_sizes = vec![(W, H); 2];
                crate::stitch::GpuExecutorConfig::new(
                    test_calibration(),
                    source_sizes,
                    crate::render::renderer::InputFormat::Yuv420p,
                )
            }
        },
    )
    .expect("gpu executor");
    let mut core = crate::core::StitchCore::new(crate::stitch::Executor::Gpu(Box::new(executor)))
        .expect("engine");
    let push_calls = Arc::new(AtomicU64::new(0));
    let push_log = Arc::new(Mutex::new(Vec::new()));
    let push_tracks = Arc::new(Mutex::new(Vec::new()));
    core.set_detector(Box::new(MockDetector::new(canned, Arc::clone(&push_calls))));
    core.set_detection_interval(3);
    core.set_ball_tracker(Box::new(MockTracker::new(0, push_tracks.clone())));
    core.set_panner(Box::new(RecordingPanner {
        log: Arc::clone(&push_log),
    }));

    for _ in 0..FRAMES {
        let frames = solid_frame();
        core.submit_frame(&frames, 1.0 / 30.0).expect("submit");
    }

    let pull = pull_log.lock().unwrap();
    let push = push_log.lock().unwrap();
    assert_eq!(*pull_tracks.lock().unwrap(), *push_tracks.lock().unwrap());
    for (frame, observation) in push_tracks.lock().unwrap().iter().enumerate() {
        assert_eq!(observation.fresh, frame % 3 == 0);
    }
    assert_eq!(pull.len() as u64, FRAMES, "pull dispatched once per frame");
    assert_eq!(
        *pull, *push,
        "push and pull panner inputs diverged - the AI stack has two brains again"
    );
    assert_eq!(
        pull_calls.load(Ordering::Relaxed),
        push_calls.load(Ordering::Relaxed),
        "detector call-count parity"
    );
}

// ─── CPU-executor session ──────────────────────────────────────────────

/// Build a session over the CPU executor - no GPU anywhere.
fn build_cpu_session() -> StitchSession {
    let executor = {
        let source_sizes = vec![(W, H); 2];
        crate::stitch::CpuExecutor::new(
            test_calibration(),
            ViewportSize {
                width: 64,
                height: 64,
            },
            source_sizes,
            false,
        )
    }
    .expect("cpu executor");
    StitchSession::with_executor(crate::stitch::Executor::Cpu(Box::new(executor)))
        .expect("cpu session")
}

/// Inline sink asserting every delivered frame is a well-formed NV12
/// buffer at the session's delivery dimensions.
struct Nv12CheckingSink {
    frames: Arc<AtomicU64>,
}

impl OutputSink for Nv12CheckingSink {
    fn name(&self) -> &str {
        "nv12-check"
    }
    fn wants(&self) -> SinkInput {
        SinkInput::CpuBytes(crate::sink::PixelFormat::Nv12)
    }
    fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
        assert_eq!(frame.format, crate::sink::PixelFormat::Nv12);
        assert_eq!(
            frame.data.len(),
            (frame.width * frame.height * 3 / 2) as usize,
            "NV12 layout: Y plane + half-size interleaved UV"
        );
        self.frames.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn finish(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

/// The full batch path over the software stitch: source -> engine
/// submit (detection + pose + CPU stitch) -> CPU NV12 -> sink fan-out.
/// Runs in the default suite - the session's first GPU-free
/// end-to-end test.
#[test]
fn cpu_session_runs_end_to_end_without_gpu() {
    const FRAMES: u64 = 5;
    let mut session = build_cpu_session();

    let encoded = Arc::new(AtomicU64::new(0));
    session
        .add_sink(
            Box::new(MockEncoder::new(Arc::clone(&encoded))),
            crate::session::SinkOptions::threaded(2),
        )
        .expect("attach threaded sink");
    let checked = Arc::new(AtomicU64::new(0));
    session
        .add_sink(
            Box::new(Nv12CheckingSink {
                frames: Arc::clone(&checked),
            }),
            crate::session::SinkOptions::inline_lossy(),
        )
        .expect("attach inline sink");

    let detections = Arc::new(AtomicU64::new(0));
    session.set_detector(Box::new(MockDetector::new(vec![], Arc::clone(&detections))));
    session.set_detection_interval(1);

    let mut source = MockSource::new(FRAMES);
    let interrupted = AtomicBool::new(false);
    let frames = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("cpu run");
    session.finish().expect("finish");

    assert_eq!(frames.frames_processed, FRAMES);
    assert_eq!(
        encoded.load(Ordering::SeqCst),
        FRAMES,
        "every frame reached the threaded sink"
    );
    assert_eq!(
        checked.load(Ordering::SeqCst),
        FRAMES,
        "every frame reached the inline sink"
    );
    assert!(
        detections.load(Ordering::SeqCst) >= FRAMES,
        "detection ran inside the CPU submit path"
    );
}

/// Zero-copy frames cannot be consumed by the software stitch; the
/// loop rejects them with a typed error instead of panicking.
#[test]
fn cpu_session_rejects_gpu_resident_frames() {
    struct ResidentSource;
    impl FrameSource for ResidentSource {
        fn info(&self) -> SourceInfo {
            SourceInfo {
                camera_sizes: vec![(W, H); 2],
                fps: 30.0,
                fps_rational: Some((30, 1)),
                total_frames: None,
            }
        }
        fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
            Ok(Some(crate::source::SourceFrame::untimed(
                FrameSet::GpuResident { slots: vec![0, 0] },
            )))
        }
    }

    let mut session = build_cpu_session();
    let interrupted = AtomicBool::new(false);
    let err = session
        .run(&mut ResidentSource, u64::MAX, &interrupted, None)
        .expect_err("resident frames must be rejected on the CPU executor");
    assert!(
        matches!(err, SessionError::Config(_)),
        "expected a typed configuration error, got: {err}"
    );
}

/// The mono cylinder path end to end without a GPU: single-input
/// source -> engine mono submit (CPU cylinder stitch) -> NV12 -> sink.
///
/// Doubles as the minimal mono-consumer example: a calibration built
/// from `Lens::flat` + `Cylinder` + a zero `Framing`, a
/// `FrameSource` yielding a one-camera `FrameSet`, and a plain session
/// run on the CPU executor - no GPU context is created on this path.
#[test]
fn mono_cylinder_session_runs_end_to_end_without_gpu() {
    struct MonoSource(u64);
    impl FrameSource for MonoSource {
        fn info(&self) -> SourceInfo {
            SourceInfo {
                camera_sizes: vec![(W, H)],
                fps: 30.0,
                fps_rational: Some((30, 1)),
                total_frames: None,
            }
        }
        fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
            if self.0 == 0 {
                return Ok(None);
            }
            self.0 -= 1;
            let y_size = (W * H) as usize;
            let uv_size = ((W / 2) * (H / 2)) as usize;
            Ok(Some(crate::source::SourceFrame::untimed(
                FrameSet::Yuv420p(vec![
                    Yuv420pData {
                        size: (W, H),
                        y: vec![128u8; y_size],
                        u: vec![128u8; uv_size],
                        v: vec![128u8; uv_size],
                    }
                    .into(),
                ]),
            )))
        }
    }

    const FRAMES: u64 = 4;
    let cal = Calibration::new(
        vec![crate::calibration::Lens::flat(W, H)],
        crate::projection::Cylinder {
            radius: f64::from(W),
            ..Default::default()
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let executor = {
        let source_sizes = vec![(W, H); cal.projection.camera_count()];
        crate::stitch::CpuExecutor::new(
            cal,
            ViewportSize {
                width: 64,
                height: 64,
            },
            source_sizes,
            false,
        )
    }
    .expect("mono cpu executor");
    let mut session =
        StitchSession::with_executor(crate::stitch::Executor::Cpu(Box::new(executor)))
            .expect("mono session");

    let encoded = Arc::new(AtomicU64::new(0));
    session
        .add_sink(
            Box::new(MockEncoder::new(Arc::clone(&encoded))),
            crate::session::SinkOptions::threaded(2),
        )
        .expect("attach sink");

    let interrupted = AtomicBool::new(false);
    let frames = session
        .run(&mut MonoSource(FRAMES), u64::MAX, &interrupted, None)
        .expect("mono run");
    session.finish().expect("finish");

    assert_eq!(frames.frames_processed, FRAMES);
    assert_eq!(
        encoded.load(Ordering::SeqCst),
        FRAMES,
        "every mono frame reached the sink"
    );
}
