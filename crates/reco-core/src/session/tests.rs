//! Integration tests for [`StitchSession`] frame loop.
//!
//! GPU-dependent tests are marked `#[ignore]` so they can be skipped in CI
//! environments without a GPU. Run them explicitly with:
//! ```bash
//! cargo test -p reco-core -- --ignored
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::*;
use crate::calibration::{CameraParams, MatchCalibration, PlaneLayout};
use crate::detector::{CameraId, Detection, Detector, RawFrame};
use crate::director::{Director, DirectorContext, ViewportPosition};
use crate::encoder::{EncodeError, Encoder, OutputFrame};
use crate::source::{FramePair, FrameSource, SourceError, SourceInfo, StereoFrame, YuvData};
use crate::viewport::ViewportConfig;

// ─── Helpers ───────────────────────────────────────────────────────────

/// Small test dimensions to keep GPU allocations minimal.
const W: u32 = 64;
const H: u32 = 64;

/// Create a minimal valid calibration for 64x64 frames.
fn test_calibration() -> MatchCalibration {
    let cam = CameraParams {
        width: W,
        height: H,
        fx: 32.0,
        fy: 32.0,
        cx: 32.0,
        cy: 32.0,
        d: [0.0; 4],
    };
    MatchCalibration {
        left: cam.clone(),
        right: cam,
        layout: PlaneLayout {
            camera_axis_offset: 0.25,
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
        },
        rig_tilt: 0.0,
        rig_roll: 0.0,
        sync_offset: 0,
        field_roi: None,
    }
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

impl Detector for MockDetector {
    fn detect(&mut self, _camera: CameraId, _frame: &RawFrame<'_>) -> Vec<Detection> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        self.detections.clone()
    }
}

// ─── MockDirector ──────────────────────────────────────────────────────

/// Mock director that records received contexts and returns a fixed position.
struct MockDirector {
    position: ViewportPosition,
    update_log: Arc<Mutex<Vec<DirectorSnapshot>>>,
}

/// Snapshot of what the director received in a single `update()` call.
#[derive(Debug)]
struct DirectorSnapshot {
    frame_index: u64,
    detection_count: usize,
    fresh_detection: bool,
}

impl MockDirector {
    fn new(position: ViewportPosition, log: Arc<Mutex<Vec<DirectorSnapshot>>>) -> Self {
        Self {
            position,
            update_log: log,
        }
    }
}

impl Director for MockDirector {
    fn update(&mut self, ctx: &DirectorContext<'_>) {
        self.update_log.lock().unwrap().push(DirectorSnapshot {
            frame_index: ctx.frame_index,
            detection_count: ctx.detections.len(),
            fresh_detection: ctx.fresh_detection,
        });
    }

    fn position(&self) -> ViewportPosition {
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

impl Encoder for MockEncoder {
    fn submit(&mut self, frame: OutputFrame<'_>) -> Result<(), EncodeError> {
        assert!(frame.width > 0, "frame width must be positive");
        assert!(frame.height > 0, "frame height must be positive");
        assert!(!frame.data.is_empty(), "frame data must not be empty");
        self.submitted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        Ok(())
    }
}

// ─── NaN Director ──────────────────────────────────────────────────────

/// Director that returns NaN yaw/pitch to test coverage clamping resilience.
struct NanDirector;

impl Director for NanDirector {
    fn update(&mut self, _ctx: &DirectorContext<'_>) {}

    fn position(&self) -> ViewportPosition {
        ViewportPosition {
            yaw: f32::NAN,
            pitch: f32::NAN,
            fov_degrees: None,
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

/// Build a session using the builder pattern with test defaults.
fn build_test_session(
    encoder: Option<Box<dyn Encoder + Send>>,
    detector: Option<Box<dyn Detector>>,
    director: Option<Box<dyn Director>>,
    detection_interval: u64,
) -> Result<StitchSession, SessionError> {
    let mut builder = StitchSession::builder()
        .calibration(test_calibration())
        .input_dimensions(W, H)
        .viewport(ViewportConfig {
            width: 64,
            height: 64,
            blend_width: 0.15,
            fov_degrees: 75.0,
            rig_tilt: 0.0,
        })
        .detection_interval(detection_interval);

    if let Some(enc) = encoder {
        builder = builder.encoder(enc, 2);
    }
    if let Some(det) = detector {
        builder = builder.detector(det);
    }
    if let Some(dir) = director {
        builder = builder.director(dir);
    }

    builder.build()
}

#[test]
#[ignore] // requires GPU
fn basic_frame_loop() {
    let submitted = Arc::new(AtomicU64::new(0));
    let encoder = MockEncoder::new(Arc::clone(&submitted));

    let mut session =
        build_test_session(Some(Box::new(encoder)), None, None, 1).expect("session build");

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
fn director_receives_detections() {
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

    let director_log = Arc::new(Mutex::new(Vec::new()));
    let director = MockDirector::new(ViewportPosition::default(), Arc::clone(&director_log));

    let submitted = Arc::new(AtomicU64::new(0));
    let encoder = MockEncoder::new(Arc::clone(&submitted));

    let mut session = build_test_session(
        Some(Box::new(encoder)),
        Some(Box::new(detector)),
        Some(Box::new(director)),
        1,
    )
    .expect("session build");

    let mut source = MockSource::new(3);
    let interrupted = AtomicBool::new(false);

    let processed = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run");

    session.finish().expect("finish");

    assert_eq!(processed, 3);

    let log = director_log.lock().unwrap();
    assert_eq!(log.len(), 3, "director should be updated once per frame");

    // The detector returns 2 detections per detect() call, and the session
    // calls detect() twice per frame (once for left, once for right camera).
    // So each frame produces 4 mapped detections.
    for (i, snapshot) in log.iter().enumerate() {
        assert_eq!(snapshot.frame_index, i as u64);
        assert!(
            snapshot.detection_count > 0,
            "frame {i}: director should receive non-empty detections, got {}",
            snapshot.detection_count,
        );
        assert!(
            snapshot.fresh_detection,
            "frame {i}: detection_interval=1 so every frame should be fresh"
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

    let mut session =
        build_test_session(Some(Box::new(encoder)), Some(Box::new(detector)), None, 3)
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
fn nan_director_does_not_crash() {
    let submitted = Arc::new(AtomicU64::new(0));
    let encoder = MockEncoder::new(Arc::clone(&submitted));

    let mut session = build_test_session(
        Some(Box::new(encoder)),
        None,
        Some(Box::new(NanDirector)),
        1,
    )
    .expect("session build");

    let mut source = MockSource::new(5);
    let interrupted = AtomicBool::new(false);

    // This should not panic. Coverage clamping handles NaN gracefully.
    let processed = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run should succeed even with NaN director output");

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
