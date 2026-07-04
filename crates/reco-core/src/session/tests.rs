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
use super::types::*;
use crate::calibration::{Calibration, Framing, Lens, Topology};
use crate::detect::detector::{Detection, DetectorError, DetectorFrame, UnifiedDetector};
use crate::detect::director::MappedDetection;
use crate::detect::panner::{PanContext, Panner};
use crate::detect::tracker::{TrackState, TrackedEntity, Tracker, WorldState};
use crate::geometry::CameraId;
use crate::geometry::ViewportPosition;
use crate::render::viewport::ViewportConfig;
use crate::sink::{OutputFrame, OutputSink, SinkError, SinkInput};
use crate::source::{FramePair, FrameSource, SourceError, SourceInfo, StereoFrame, YuvData};

// ─── Helpers ───────────────────────────────────────────────────────────

/// Small test dimensions to keep GPU allocations minimal.
const W: u32 = 64;
const H: u32 = 64;

/// Create a minimal valid calibration for 64x64 frames.
fn test_calibration() -> Calibration {
    let cam = || Lens::fisheye(W, H, 32.0, 32.0, 32.0, 32.0, [0.0; 4]);
    Calibration::new(
        vec![cam(), cam()],
        Topology {
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
fn solid_frame() -> StereoFrame {
    let y_size = (W * H) as usize;
    let uv_size = ((W / 2) * (H / 2)) as usize;
    let yuv = YuvData {
        y: vec![128u8; y_size],
        u: vec![128u8; uv_size],
        v: vec![128u8; uv_size],
    };
    StereoFrame::Yuv420p(FramePair {
        left: yuv.clone(),
        right: yuv,
    })
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
            width: W,
            height: H,
            fps: 30.0,
            fps_rational: Some((30, 1)),
            total_frames: None,
        }
    }

    fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        Ok(Some(solid_frame()))
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

impl UnifiedDetector for MockDetector {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn detect(
        &mut self,
        _camera: CameraId,
        _frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(self.detections.clone())
    }
}

// ─── MockTracker / MockPanner ──────────────────────────────────────────

/// Mock tracker that records how many detections it received each
/// frame and returns a single "ball" entity.
struct MockTracker {
    log: Arc<Mutex<Vec<TrackerSnapshot>>>,
    class_id: u16,
}

/// Snapshot of what a tracker saw on a single `update()` call.
#[derive(Debug)]
struct TrackerSnapshot {
    frame_index: u64,
    detection_count: usize,
}

impl MockTracker {
    fn new(class_id: u16, log: Arc<Mutex<Vec<TrackerSnapshot>>>) -> Self {
        Self { class_id, log }
    }
}

impl Tracker for MockTracker {
    fn update(&mut self, detections: &[MappedDetection], _timestamp_ms: f64) -> Vec<TrackedEntity> {
        // Session bumps frame_count after dispatch, so `log.len()` is
        // the per-call frame_index we see.
        let frame_index = self.log.lock().unwrap().len() as u64;
        self.log.lock().unwrap().push(TrackerSnapshot {
            frame_index,
            detection_count: detections.len(),
        });
        vec![TrackedEntity {
            id: 0,
            class_id: self.class_id,
            yaw: 0.0,
            pitch: 0.0,
            confidence: 1.0,
            state: TrackState::Tracking,
            age_frames: 1,
            origin: CameraId::Left,
        }]
    }

    fn class_id(&self) -> u16 {
        self.class_id
    }
}

/// Mock panner that returns a fixed position regardless of the world.
struct MockPanner {
    position: ViewportPosition,
}

impl Panner for MockPanner {
    fn decide(&mut self, _world: &WorldState, _ctx: &PanContext<'_>) -> ViewportPosition {
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
struct RecordingPanner {
    log: Arc<Mutex<Vec<(u64, ViewportPosition)>>>,
}

impl Panner for RecordingPanner {
    fn decide(&mut self, _world: &WorldState, ctx: &PanContext<'_>) -> ViewportPosition {
        self.log
            .lock()
            .unwrap()
            .push((ctx.frame_index, ctx.previous_position));
        ViewportPosition {
            yaw: 0.001 * ctx.frame_index as f32,
            pitch: -0.0005 * ctx.frame_index as f32,
            fov_degrees: None,
        }
    }
}

// ─── NaN Panner ────────────────────────────────────────────────────────

/// Panner that returns NaN yaw/pitch to test coverage clamping resilience.
struct NanPanner;

impl Panner for NanPanner {
    fn decide(&mut self, _world: &WorldState, _ctx: &PanContext<'_>) -> ViewportPosition {
        ViewportPosition {
            yaw: f32::NAN,
            pitch: f32::NAN,
            fov_degrees: None,
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

/// Build a session using the builder pattern with test defaults.
/// Trackers and panners are attached post-build (they have their own
/// setters and aren't part of the builder API).
fn build_test_session(
    encoder: Option<Box<dyn OutputSink>>,
    detector: Option<Box<dyn UnifiedDetector>>,
    detection_interval: u64,
) -> Result<StitchSession, SessionError> {
    let mut builder = StitchSession::builder()
        .calibration(test_calibration())
        .input_dimensions(W, H)
        .viewport(ViewportConfig {
            width: 64,
            height: 64,
            fov_degrees: 75.0,
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
#[ignore] // requires GPU
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

    assert_eq!(processed, 10, "should process all 10 frames");

    // The async encoder receives frames with a 1-frame pipeline delay
    // (NV12 double-buffering), so we expect at least 9 frames submitted.
    let count = submitted.load(Ordering::Relaxed);
    assert!(
        (9..=10).contains(&count),
        "encoder should receive 9-10 frames, got {count}"
    );
}

#[test]
#[ignore] // requires GPU
fn tracker_receives_detections() {
    let call_count = Arc::new(AtomicU64::new(0));
    let canned = vec![
        Detection {
            camera: CameraId::Left,
            class_id: 0,
            confidence: 0.9,
            center_x: 0.5,
            center_y: 0.5,
            width: 0.1,
            height: 0.1,
        },
        Detection {
            camera: CameraId::Right,
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
        position: ViewportPosition::default(),
    }));

    let mut source = MockSource::new(3);
    let interrupted = AtomicBool::new(false);

    let processed = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run");

    session.finish().expect("finish");

    assert_eq!(processed, 3);

    let log = tracker_log.lock().unwrap();
    assert_eq!(log.len(), 3, "tracker should be updated once per frame");

    // The detector returns 2 detections per detect() call, and the session
    // calls detect() twice per frame (once for left, once for right camera).
    // So each frame produces 4 mapped detections the tracker observes.
    for (i, snapshot) in log.iter().enumerate() {
        assert_eq!(snapshot.frame_index, i as u64);
        assert!(
            snapshot.detection_count > 0,
            "frame {i}: tracker should receive non-empty detections, got {}",
            snapshot.detection_count,
        );
    }
}

#[test]
#[ignore] // requires GPU
fn detection_interval_respected() {
    let call_count = Arc::new(AtomicU64::new(0));
    let canned = vec![Detection {
        camera: CameraId::Left,
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

    assert_eq!(processed, 10);

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
fn nan_panner_does_not_crash() {
    let submitted = Arc::new(AtomicU64::new(0));
    let encoder = MockEncoder::new(Arc::clone(&submitted));

    let mut session = build_test_session(Some(Box::new(encoder)), None, 1).expect("session build");
    session.set_panner(Box::new(NanPanner));

    let mut source = MockSource::new(5);
    let interrupted = AtomicBool::new(false);

    // This should not panic. Coverage clamping handles NaN gracefully.
    let processed = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run should succeed even with NaN panner output");

    session.finish().expect("finish");

    assert_eq!(processed, 5, "all 5 frames should be processed");
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
/// through `StitchCore::submit_frame_yuv` and `StitchSession::run`,
/// must see identical inputs in identical order - frame indices,
/// previous-pose threading, detector call counts. (2) The pull side
/// must have driven the ENGINE's stack, observed through the core's
/// own detection cache: a session-side shadow component could still
/// produce a matching log, but it would leave the engine's state
/// untouched.
#[test]
#[ignore] // requires GPU
fn push_and_pull_share_one_ai_brain() {
    const FRAMES: u64 = 6;
    let canned = vec![Detection {
        camera: CameraId::Left,
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
    let mut session = build_test_session(
        None,
        Some(Box::new(MockDetector::new(
            canned.clone(),
            Arc::clone(&pull_calls),
        ))),
        1,
    )
    .expect("session build");
    session.set_ball_tracker(Box::new(MockTracker::new(
        0,
        Arc::new(Mutex::new(Vec::new())),
    )));
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
            viewport: ViewportConfig {
                width: 64,
                height: 64,
                fov_degrees: 75.0,
            },
            ..crate::stitch::GpuExecutorConfig::new(
                test_calibration(),
                W,
                H,
                crate::render::renderer::InputFormat::Yuv420p,
            )
        },
    )
    .expect("gpu executor");
    let mut core = crate::core::StitchCore::new(crate::stitch::Executor::Gpu(Box::new(executor)))
        .expect("engine");
    let push_calls = Arc::new(AtomicU64::new(0));
    let push_log = Arc::new(Mutex::new(Vec::new()));
    core.set_detector(Box::new(MockDetector::new(canned, Arc::clone(&push_calls))));
    core.set_ball_tracker(Box::new(MockTracker::new(
        0,
        Arc::new(Mutex::new(Vec::new())),
    )));
    core.set_panner(Box::new(RecordingPanner {
        log: Arc::clone(&push_log),
    }));

    for _ in 0..FRAMES {
        let StereoFrame::Yuv420p(pair) = solid_frame() else {
            unreachable!("solid_frame is YUV420P")
        };
        let left = crate::render::planes::YuvPlanes {
            y: &pair.left.y,
            u: &pair.left.u,
            v: &pair.left.v,
        };
        let right = crate::render::planes::YuvPlanes {
            y: &pair.right.y,
            u: &pair.right.u,
            v: &pair.right.v,
        };
        core.submit_frame_yuv(&left, &right).expect("submit");
    }

    let pull = pull_log.lock().unwrap();
    let push = push_log.lock().unwrap();
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
    let executor = crate::stitch::CpuExecutor::new(
        Box::new(crate::projection::LShapeProjection),
        test_calibration(),
        ViewportConfig {
            width: 64,
            height: 64,
            fov_degrees: 75.0,
        },
        W,
        H,
        false,
    )
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

    assert_eq!(frames, FRAMES);
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
                width: W,
                height: H,
                fps: 30.0,
                fps_rational: Some((30, 1)),
                total_frames: None,
            }
        }
        fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
            Ok(Some(StereoFrame::GpuResident {
                left_slot: 0,
                right_slot: 0,
            }))
        }
    }

    let mut session = build_cpu_session();
    let interrupted = AtomicBool::new(false);
    let err = session
        .run(&mut ResidentSource, u64::MAX, &interrupted, None)
        .expect_err("resident frames must be rejected on the CPU executor");
    assert!(
        matches!(err, SessionError::Config(_)),
        "expected a typed config error, got: {err}"
    );
}
