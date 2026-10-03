//! Capture identity survives delivery cadence and is consumed only by successful inference.

use super::*;
use crate::source::{Nv12Data, SourceFrame};

fn timing(times_us: Vec<i64>) -> Option<crate::source::CaptureTiming> {
    Some(crate::source::CaptureTiming {
        at_us: *times_us.iter().max()?,
        times_us,
        max_age_us: None,
    })
}

type Calls = Arc<Mutex<Vec<(usize, u8)>>>;
struct DetectorProbe(Calls);

impl Detector for DetectorProbe {
    fn name(&self) -> &'static str {
        "capture-identity"
    }
    fn detect(
        &mut self,
        camera: usize,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let DetectorFrame::Yuv(view) = frame else {
            panic!("expected host image")
        };
        let marker = view.y[0];
        self.0.lock().unwrap().push((camera, marker));
        if marker == 255
            && self
                .0
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| **entry == (camera, marker))
                .count()
                == 1
        {
            return Err(DetectorError::InferenceFailed("injected".into()));
        }
        if marker == 254 {
            return Ok(Vec::new());
        }
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

struct FreshnessProbe(Arc<Mutex<Vec<Vec<usize>>>>);
impl Tracker for FreshnessProbe {
    fn class_id(&self) -> u16 {
        1
    }
    fn update(&mut self, _: &[MappedDetection], fresh: &[usize], _: f64) -> Vec<TrackedEntity> {
        self.0.lock().unwrap().push(fresh.to_vec());
        Vec::new()
    }
}

struct Image {
    markers: [u8; 2],
    nv12: bool,
    pts_us: Option<i64>,
    capture: Option<crate::source::CaptureTiming>,
}

struct Images(std::collections::VecDeque<Image>);
impl FrameSource for Images {
    fn info(&self) -> SourceInfo {
        MockSource::new(0).info()
    }
    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        Ok(self.0.pop_front().map(|frame| SourceFrame {
            images: image(frame.markers, frame.nv12),
            pts_us: frame.pts_us,
            capture: frame.capture,
        }))
    }
}

fn image(markers: [u8; 2], nv12: bool) -> FrameSet {
    if nv12 {
        FrameSet::Nv12(
            markers
                .into_iter()
                .map(|marker| {
                    Arc::new(Nv12Data {
                        size: (W, H),
                        y: vec![marker; (W * H) as usize],
                        uv: vec![128; (W * H / 2) as usize],
                    })
                })
                .collect(),
        )
    } else {
        FrameSet::Yuv420p(
            markers
                .into_iter()
                .map(|marker| {
                    Arc::new(Yuv420pData {
                        size: (W, H),
                        y: vec![marker; (W * H) as usize],
                        u: vec![128; (W * H / 4) as usize],
                        v: vec![128; (W * H / 4) as usize],
                    })
                })
                .collect(),
        )
    }
}

#[test]
fn reused_camera_images_are_inferred_once_in_immediate_and_lookahead_sessions() {
    check_reused_images(|_| build_cpu_session());
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "requires GPU rendering"]
fn gpu_reused_camera_images_are_inferred_once() {
    use crate::render::renderer::InputFormat;
    use crate::stitch::{Executor, GpuExecutor, GpuExecutorConfig};
    let gpu = crate::gpu::GpuContext::new_blocking().unwrap();
    check_reused_images(|nv12| {
        let executor = GpuExecutor::new(
            gpu.clone(),
            GpuExecutorConfig {
                viewport_size: ViewportSize {
                    width: 64,
                    height: 64,
                },
                ..GpuExecutorConfig::new(
                    test_calibration(),
                    vec![(W, H); 2],
                    if nv12 {
                        InputFormat::Nv12
                    } else {
                        InputFormat::Yuv420p
                    },
                )
            },
        )
        .unwrap();
        StitchSession::with_executor(Executor::Gpu(Box::new(executor))).unwrap()
    });
}

fn check_reused_images(build: impl Fn(bool) -> StitchSession) {
    for nv12 in [false, true] {
        for depth in [0, 1, 4, 16] {
            for reverse in [false, true] {
                let mut session = build(nv12);
                session.core_mut().set_constrained_look(false);
                session.set_lookahead(depth, usize::MAX);
                let calls = Arc::new(Mutex::new(Vec::new()));
                let freshness = Arc::new(Mutex::new(Vec::new()));
                session
                    .core_mut()
                    .set_detector(Box::new(DetectorProbe(calls.clone())));
                session.core_mut().set_detection_interval(2);
                session
                    .core_mut()
                    .set_player_tracker(Box::new(FreshnessProbe(freshness.clone())));
                session.set_panner(Box::new(MockPanner {
                    position: WorldPose::default(),
                }));
                // 11 is first displayed on a skipped tick. 24 is later reused on a due tick.
                let markers = [
                    [10, 20],
                    [11, 21],
                    [11, 22],
                    [11, 23],
                    [11, 24],
                    [11, 24],
                    [11, 24],
                    [12, 25],
                    [12, 25],
                ];
                let frames = markers
                    .into_iter()
                    .enumerate()
                    .map(|(index, mut markers)| {
                        if reverse {
                            markers.reverse();
                        }
                        Image {
                            markers,
                            nv12,
                            pts_us: Some(index as i64 * 33_333),
                            capture: timing(markers.map(|v| i64::from(v) * 1000).to_vec()),
                        }
                    })
                    .collect();
                let result = session
                    .run(&mut Images(frames), u64::MAX, &AtomicBool::new(false), None)
                    .unwrap();
                assert_eq!(result.frames_processed, 9);
                let snapshot = session.telemetry_snapshot();
                assert_eq!(snapshot.frames_processed, 9);
                assert_eq!(snapshot.total_detections, 18);
                let expected = if reverse {
                    vec![
                        (0, 20),
                        (1, 10),
                        (0, 22),
                        (1, 11),
                        (0, 24),
                        (0, 25),
                        (1, 12),
                    ]
                } else {
                    vec![
                        (0, 10),
                        (1, 20),
                        (0, 11),
                        (1, 22),
                        (1, 24),
                        (0, 12),
                        (1, 25),
                    ]
                };
                assert_eq!(*calls.lock().unwrap(), expected);
                assert_eq!(
                    *freshness.lock().unwrap(),
                    vec![
                        vec![0, 1],
                        vec![],
                        vec![0, 1],
                        vec![],
                        vec![usize::from(!reverse)],
                        vec![],
                        vec![],
                        vec![],
                        vec![0, 1]
                    ]
                );
            }
        }
    }
}

#[test]
fn capture_identity_handles_retry_empty_results_and_invalidation() {
    for failed in [0, 1] {
        let mut session = build_cpu_session();
        let core = session.core_mut();
        let calls = Arc::new(Mutex::new(Vec::new()));
        core.set_detector(Box::new(DetectorProbe(calls.clone())));
        let mut markers = [10, 20];
        markers[failed] = 255;
        core.run_cpu_detection(&image(markers, false), &[100, 100])
            .unwrap();
        core.update_trackers(0, 0.0);
        assert_eq!(core.detected_capture_times[failed], None);
        // Same image, same time: retry only the camera that failed.
        core.run_cpu_detection(&image(markers, false), &[100, 100])
            .unwrap();
        assert_eq!(calls.lock().unwrap().len(), 3);
        assert_eq!(core.detected_capture_times[failed], Some(100));
        assert_eq!(core.fresh_cameras, [failed]);
        core.update_trackers(1, 33.3);
        core.run_cpu_detection(&image(markers, false), &[100, 100])
            .unwrap();
        assert_eq!(
            calls.lock().unwrap().len(),
            3,
            "successful retry is consumed once"
        );
        markers[failed] = 254;
        let mut times = [100, 100];
        times[failed] = 200;
        core.run_cpu_detection(&image(markers, false), &times)
            .unwrap();
        assert_eq!(core.fresh_cameras, [failed]);
        core.update_trackers(2, 66.6);
        core.run_cpu_detection(&image(markers, false), &times)
            .unwrap();
        assert!(core.fresh_cameras.is_empty());
        assert_eq!(
            calls.lock().unwrap().len(),
            4,
            "successful empty is still analysed"
        );
        // Detector replacement and geometry invalidation permit re-analysis.
        core.set_detector(Box::new(DetectorProbe(calls.clone())));
        core.run_cpu_detection(&image(markers, false), &times)
            .unwrap();
        assert_eq!(calls.lock().unwrap().len(), 6);
        core.edit_calibration(|cal| cal.framing.roll += 0.1)
            .unwrap();
        core.run_cpu_detection(&image(markers, false), &times)
            .unwrap();
        assert_eq!(calls.lock().unwrap().len(), 8);
        for invalid in [vec![100], vec![-1, 200], vec![0, 0]] {
            assert!(
                core.run_cpu_detection(&image(markers, false), &invalid)
                    .is_err()
            );
        }
        assert_eq!(calls.lock().unwrap().len(), 8, "reject before inference");
    }
}

#[test]
fn invalid_capture_metadata_fails_even_on_skipped_detection_ticks() {
    for depth in [0, 4] {
        for times in [
            vec![-1, 101],
            vec![101, -1],
            vec![99, 101],
            vec![101, 99],
            vec![101],
            vec![],
        ] {
            let mut session = build_cpu_session();
            session.core_mut().set_constrained_look(false);
            session.set_lookahead(depth, usize::MAX);
            session.core_mut().set_detection_interval(10);
            let frames = [vec![100, 100], times]
                .into_iter()
                .enumerate()
                .map(|(index, capture_times_us)| Image {
                    markers: [10, 20],
                    nv12: false,
                    pts_us: Some(index as i64 * 33_333),
                    capture: timing(capture_times_us),
                })
                .collect();
            assert!(matches!(
                session.run(&mut Images(frames), u64::MAX, &AtomicBool::new(false), None),
                Err(SessionError::Source(_))
            ));
        }
    }
}

#[test]
fn new_source_run_does_not_reuse_previous_source_image_identity() {
    for depth in [0, 4] {
        let mut session = build_cpu_session();
        session.core_mut().set_constrained_look(false);
        session.set_lookahead(depth, usize::MAX);
        let calls = Arc::new(Mutex::new(Vec::new()));
        session
            .core_mut()
            .set_detector(Box::new(DetectorProbe(calls.clone())));
        for capture_time in [100, 100, 0] {
            let frames = [Image {
                markers: [10, 20],
                nv12: false,
                pts_us: Some(0),
                capture: timing(vec![capture_time; 2]),
            }]
            .into_iter()
            .collect();
            session
                .run(&mut Images(frames), u64::MAX, &AtomicBool::new(false), None)
                .unwrap();
        }
        assert_eq!(calls.lock().unwrap().len(), 6);
    }
}

type Observations = Arc<Mutex<Vec<(Vec<usize>, Vec<usize>)>>>;
struct ObservationProbe(Observations);
impl Tracker for ObservationProbe {
    fn class_id(&self) -> u16 {
        1
    }
    fn update(
        &mut self,
        detections: &[MappedDetection],
        fresh: &[usize],
        _: f64,
    ) -> Vec<TrackedEntity> {
        self.0.lock().unwrap().push((
            detections.iter().map(|d| d.camera).collect(),
            fresh.to_vec(),
        ));
        Vec::new()
    }
}

#[test]
fn observation_expiry_uses_source_age_before_immediate_and_lookahead_tracking() {
    for depth in [0, 1, 4] {
        for reverse in [false, true] {
            for slow in [false, true] {
                let mut session = build_cpu_session();
                session.core_mut().set_constrained_look(false);
                session.set_lookahead(depth, usize::MAX);
                session.core_mut().set_detection_interval(4);
                let calls = Arc::new(Mutex::new(Vec::new()));
                let observations = Arc::new(Mutex::new(Vec::new()));
                session
                    .core_mut()
                    .set_detector(Box::new(DetectorProbe(calls.clone())));
                session
                    .core_mut()
                    .set_player_tracker(Box::new(ObservationProbe(observations.clone())));
                session.set_panner(Box::new(MockPanner {
                    position: WorldPose::default(),
                }));
                let frames = [100_000, 133_333, 166_666, 233_335, 266_668]
                    .into_iter()
                    .enumerate()
                    .map(|(index, at_us)| {
                        let mut times_us = vec![at_us - 100_000, at_us];
                        if reverse {
                            times_us.reverse();
                        }
                        Image {
                            markers: [10 + index as u8, 20 + index as u8],
                            nv12: false,
                            pts_us: Some(at_us - 100_000),
                            capture: Some(crate::source::CaptureTiming {
                                times_us,
                                at_us,
                                max_age_us: Some(100_000),
                            }),
                        }
                    })
                    .collect();
                struct Delayed {
                    inner: Images,
                    slow: bool,
                }
                impl FrameSource for Delayed {
                    fn info(&self) -> SourceInfo {
                        self.inner.info()
                    }
                    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
                        if self.slow {
                            std::thread::sleep(std::time::Duration::from_millis(2));
                        }
                        self.inner.next_frame()
                    }
                }
                session
                    .run(
                        &mut Delayed {
                            inner: Images(frames),
                            slow,
                        },
                        u64::MAX,
                        &AtomicBool::new(false),
                        None,
                    )
                    .unwrap();
                assert_eq!(
                    *observations.lock().unwrap(),
                    vec![
                        (vec![0, 1], vec![0, 1]),
                        (vec![0, 1], vec![]),
                        (vec![0, 1], vec![]),
                        (vec![usize::from(!reverse)], vec![]),
                        (vec![0, 1], vec![0, 1]),
                    ],
                    "depth={depth}, reverse={reverse}, slow={slow}"
                );
                assert_eq!(calls.lock().unwrap().len(), 4);
            }
        }
    }
}

#[test]
fn expiry_preserves_identity_and_cannot_revive_an_old_measurement() {
    let mut session = build_cpu_session();
    let core = session.core_mut();
    let calls = Arc::new(Mutex::new(Vec::new()));
    core.set_detector(Box::new(DetectorProbe(calls.clone())));
    core.run_cpu_detection(&image([10, 20], false), &[0, 0])
        .unwrap();
    let capture = crate::source::CaptureTiming {
        times_us: vec![1_000_000; 2],
        at_us: 1_000_000,
        max_age_us: Some(0),
    };
    core.expire_observations(Some(&capture), 1.0 / 30.0)
        .unwrap();
    assert!(core.last_detections.is_empty());
    assert!(core.fresh_cameras.is_empty());
    assert_eq!(core.detected_capture_times, vec![Some(0), Some(0)]);
    core.run_cpu_detection(&image([10, 20], false), &[0, 0])
        .unwrap();
    assert_eq!(calls.lock().unwrap().len(), 2);
    assert!(core.last_detections.is_empty());
    assert!(core.expire_observations(Some(&capture), f32::NAN).is_err());
    let overflow = crate::source::CaptureTiming {
        max_age_us: Some(u64::MAX),
        ..capture
    };
    assert!(core.expire_observations(Some(&overflow), 1.0).is_err());
}

#[test]
fn manual_cpu_tracking_expires_observations_on_skipped_detection_ticks() {
    let mut session = build_cpu_session();
    let calls = Calls::default();
    let observations = Arc::new(Mutex::new(Vec::new()));
    session
        .core_mut()
        .set_detector(Box::new(DetectorProbe(calls.clone())));
    session
        .core_mut()
        .set_player_tracker(Box::new(ObservationProbe(observations.clone())));
    session.core_mut().set_detection_interval(4);
    session.set_panner(Box::new(MockPanner {
        position: WorldPose::default(),
    }));
    for (index, at_us) in [0, 200_000].into_iter().enumerate() {
        session.frame_count = index as u64;
        session
            .detect_and_update_world_pose(
                &image([10, 20], false),
                Some(&crate::source::CaptureTiming {
                    times_us: vec![at_us; 2],
                    at_us,
                    max_age_us: Some(0),
                }),
                1.0 / 30.0,
                std::time::Duration::ZERO,
            )
            .unwrap();
    }
    assert_eq!(
        *observations.lock().unwrap(),
        vec![(vec![0, 1], vec![0, 1]), (vec![], vec![])]
    );
    assert_eq!(calls.lock().unwrap().len(), 2);
}

#[test]
fn bounded_capture_timing_rejects_stale_future_and_nonadvancing_sets_before_inference() {
    use crate::source::CaptureTiming;
    for depth in [0, 4] {
        let valid = CaptureTiming {
            times_us: vec![100; 2],
            at_us: 100,
            max_age_us: Some(50),
        };
        for invalid in [
            CaptureTiming {
                at_us: 99,
                ..valid.clone()
            },
            CaptureTiming {
                at_us: 151,
                ..valid.clone()
            },
            valid.clone(),
            CaptureTiming {
                at_us: 101,
                max_age_us: None,
                ..valid.clone()
            },
            CaptureTiming {
                at_us: 101,
                times_us: vec![101],
                ..valid.clone()
            },
        ] {
            let mut session = build_cpu_session();
            session.core_mut().set_constrained_look(false);
            session.set_lookahead(depth, usize::MAX);
            session.core_mut().set_detection_interval(10);
            let calls = Arc::new(Mutex::new(Vec::new()));
            session
                .core_mut()
                .set_detector(Box::new(DetectorProbe(calls.clone())));
            let frames = [valid.clone(), invalid]
                .into_iter()
                .enumerate()
                .map(|(i, capture)| Image {
                    markers: [10, 20],
                    nv12: false,
                    pts_us: Some(i as i64),
                    capture: Some(capture),
                })
                .collect();
            assert!(matches!(
                session.run(&mut Images(frames), u64::MAX, &AtomicBool::new(false), None),
                Err(SessionError::Source(_))
            ));
            assert_eq!(
                calls.lock().unwrap().len(),
                2,
                "invalid skipped tick must not run inference"
            );
        }
    }
}
