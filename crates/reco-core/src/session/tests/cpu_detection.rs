//! CPU camera dispatch and mapped measurement parity across submission paths.

use super::*;
use crate::detect::detector::ChromaPlanes;
use crate::geometry::YawPitch;
use crate::source::Nv12Data;
use crate::stitch::{CpuExecutor, Executor};

struct FutureProbe(Arc<Mutex<Vec<usize>>>);

#[test]
fn shared_host_images_survive_source_release_and_match_owned_delivery() {
    struct QueuedSource<T> {
        frames: std::collections::VecDeque<Vec<Arc<T>>>,
        wrap: fn(Vec<Arc<T>>) -> FrameSet,
    }
    impl<T: Send + Sync> FrameSource for QueuedSource<T> {
        fn info(&self) -> SourceInfo {
            MockSource::new(0).info()
        }
        fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
            Ok(self
                .frames
                .pop_front()
                .map(|images| crate::source::SourceFrame::untimed((self.wrap)(images))))
        }
    }

    fn verify<T: Send + Sync + Clone + 'static>(images: [T; 2], wrap: fn(Vec<Arc<T>>) -> FrameSet) {
        let [left, right] = images.map(Arc::new);
        let weak = [Arc::downgrade(&left), Arc::downgrade(&right)];
        let mut baseline = None;
        for shared in [false, true] {
            for depth in [0, 1, 8] {
                let frames = (0..4)
                    .map(|_| {
                        let cameras = if shared {
                            vec![left.clone(), right.clone()]
                        } else {
                            vec![Arc::new((*left).clone()), Arc::new((*right).clone())]
                        };
                        assert_eq!(Arc::ptr_eq(&cameras[0], &left), shared);
                        assert_eq!(Arc::ptr_eq(&cameras[1], &right), shared);
                        cameras
                    })
                    .collect();
                let mut source = QueuedSource { frames, wrap };
                let mut session = build_cpu_session();
                session.core_mut().set_constrained_look(false);
                session.set_lookahead(depth, usize::MAX);
                let output = Arc::new(Mutex::new(Vec::new()));
                session
                    .add_sink(
                        Box::new(CapturedOutput(output.clone())),
                        crate::session::SinkOptions::threaded(1),
                    )
                    .unwrap();
                let result = session
                    .run(&mut source, u64::MAX, &AtomicBool::new(false), None)
                    .unwrap();
                assert_eq!(result.frames_processed, 4);
                assert!(source.frames.is_empty());
                drop(source);
                session.finish().unwrap();
                assert_eq!(Arc::strong_count(&left), 1);
                assert_eq!(Arc::strong_count(&right), 1);
                let output = output.lock().unwrap();
                assert_eq!(output.len(), 4);
                if let Some(expected) = &baseline {
                    assert_eq!(&*output, expected);
                } else {
                    baseline = Some(output.clone());
                }
            }
        }
        // A queued frame, independently of its producer, owns both images.
        let frame = wrap(vec![left, right]);
        assert!(weak.iter().all(|image| image.upgrade().is_some()));
        drop(frame);
        assert!(weak.iter().all(|image| image.upgrade().is_none()));
    }

    verify(
        [64, 192].map(|value| Nv12Data {
            y: vec![value; (W * H) as usize],
            uv: vec![128; (W * H / 2) as usize],
        }),
        FrameSet::Nv12,
    );
    verify(
        [64, 192].map(|value| Yuv420pData {
            y: vec![value; (W * H) as usize],
            u: vec![128; (W * H / 4) as usize],
            v: vec![128; (W * H / 4) as usize],
        }),
        FrameSet::Yuv420p,
    );
}

#[test]
fn full_circle_lookahead_follows_both_directions_across_the_yaw_seam() {
    struct SeamPanner(f32);
    impl Panner for SeamPanner {
        fn decide(&mut self, _: &WorldState, ctx: &PanContext<'_>) -> WorldPose {
            WorldPose {
                yaw: self.0
                    * if ctx.frame_index == 0 {
                        179.0_f32
                    } else {
                        -179.0
                    }
                    .to_radians(),
                pitch: 0.0,
                fov_degrees: 60.0,
            }
        }
    }
    for direction in [-1.0, 1.0] {
        for depth in [0, 1, 4] {
            let cal = Calibration::new(
                vec![Lens::flat(W, H)],
                crate::projection::Cylinder {
                    arc_deg: 360.0,
                    radius: 64.0,
                    surface_height: None,
                },
                Framing {
                    axis_offset: 0.0,
                    tilt: 0.0,
                    roll: 0.0,
                },
            );
            let executor = CpuExecutor::new(
                cal,
                ViewportSize {
                    width: 16,
                    height: 16,
                },
                W,
                H,
                false,
            )
            .unwrap();
            let mut session =
                StitchSession::with_executor(Executor::Cpu(Box::new(executor))).unwrap();
            session.core_mut().set_constrained_look(false);
            session
                .core_mut()
                .enable_replay_buffer(Some(std::time::Duration::from_secs(1)));
            session.set_panner(Box::new(SeamPanner(direction)));
            session.set_lookahead(depth, usize::MAX);
            session
                .run(
                    &mut Source {
                        cameras: 1,
                        nv12: false,
                        index: 4,
                    },
                    u64::MAX,
                    &AtomicBool::new(false),
                    None,
                )
                .unwrap();
            session.finish().unwrap();
            let replay = session.core().replay_buffer().unwrap();
            assert_eq!(replay.len(), 2);
            for frame in replay.iter() {
                assert!(
                    frame.pose.yaw.cos() < -0.99,
                    "direction={direction}, depth={depth}, pose={:?}",
                    frame.pose
                );
                assert_eq!(frame.pose.fov_degrees, 60.0);
            }
        }
    }
}

#[test]
fn lookahead_image_budget_covers_both_queues_and_preserves_short_inputs() {
    for nv12 in [false, true] {
        for depth in [1usize, 2, 3, 10] {
            let frame_bytes = (W * H * 3) as usize;
            let peak = (depth + (depth / 2).max(1) + 1).min(6);
            for short in [false, true] {
                let length = if short { 1 } else { 6 };
                let bytes = frame_bytes * if short { 1 } else { peak };
                let mut reference = None;
                for allowance in [usize::MAX, bytes, bytes - 1] {
                    let mut session = build_cpu_session();
                    session.core_mut().set_constrained_look(false);
                    session.set_lookahead(depth, allowance);
                    let output = Arc::new(Mutex::new(Vec::new()));
                    session
                        .add_sink(
                            Box::new(CapturedOutput(output.clone())),
                            crate::session::SinkOptions::inline_lossy(),
                        )
                        .unwrap();
                    let mut source = Source {
                        cameras: 2,
                        nv12,
                        index: 6 - length,
                    };
                    let result = session.run(&mut source, u64::MAX, &AtomicBool::new(false), None);
                    if allowance < bytes {
                        assert!(
                            matches!(result, Err(SessionError::Config(_))),
                            "depth={depth} short={short}"
                        );
                    } else {
                        assert_eq!(result.unwrap().frames_processed, u64::from(length));
                        let images = output.lock().unwrap().clone();
                        if let Some(expected) = &reference {
                            assert_eq!(&images, expected);
                        } else {
                            reference = Some(images);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn lookahead_image_budget_counts_capacity_and_all_cameras() {
    for cameras in [1, 2, 3] {
        let planes: Vec<_> = (0..cameras)
            .map(|_| {
                let mut y = Vec::with_capacity(100);
                y.resize(4, 0);
                Nv12Data { y, uv: vec![0; 2] }
            })
            .collect();
        let expected: usize = planes
            .iter()
            .map(|p| p.y.capacity() + p.uv.capacity())
            .sum();
        assert_eq!(
            FrameSet::Nv12(planes.into_iter().map(Arc::new).collect())
                .retained_image_bytes()
                .unwrap(),
            expected
        );
        assert!(expected >= cameras * 102);

        let planes: Vec<_> = (0..cameras)
            .map(|_| {
                let mut v = Vec::with_capacity(100);
                v.resize(1, 128);
                Yuv420pData {
                    y: vec![0; 4],
                    u: vec![128],
                    v,
                }
            })
            .collect();
        let expected: usize = planes
            .iter()
            .map(|p| p.y.capacity() + p.u.capacity() + p.v.capacity())
            .sum();
        assert_eq!(
            FrameSet::Yuv420p(planes.into_iter().map(Arc::new).collect())
                .retained_image_bytes()
                .unwrap(),
            expected
        );
        assert!(expected >= cameras * 105);
    }
}

#[test]
fn shared_images_do_not_bypass_the_lookahead_budget() {
    let nv12 = Arc::new(Nv12Data {
        y: vec![0; 4],
        uv: vec![0; 2],
    });
    let yuv = Arc::new(Yuv420pData {
        y: vec![0; 4],
        u: vec![0; 1],
        v: vec![0; 1],
    });
    for count in [1, 2, 3] {
        for _ in 0..2 {
            assert_eq!(
                FrameSet::Nv12(vec![nv12.clone(); count])
                    .retained_image_bytes()
                    .unwrap(),
                count * (nv12.y.capacity() + nv12.uv.capacity())
            );
            assert_eq!(
                FrameSet::Yuv420p(vec![yuv.clone(); count])
                    .retained_image_bytes()
                    .unwrap(),
                count * (yuv.y.capacity() + yuv.u.capacity() + yuv.v.capacity())
            );
        }
    }
}

#[test]
fn lookahead_budget_rejection_can_be_reconfigured_without_stale_accounting() {
    let mut session = build_cpu_session();
    session.core_mut().set_constrained_look(false);
    let mut completed = 0;
    for (depth, budget, accepted) in [
        (2, 0, false),
        (2, (W * H * 3) as usize, false),
        (2, usize::MAX, true),
        (0, 0, true),
    ] {
        session.set_lookahead(depth, budget);
        let mut source = Source {
            cameras: 2,
            nv12: true,
            index: 0,
        };
        let result = session.run(&mut source, u64::MAX, &AtomicBool::new(false), None);
        if accepted {
            completed += 6;
            assert_eq!(result.unwrap().frames_processed, completed);
        } else {
            assert!(matches!(result, Err(SessionError::Config(_))));
        }
    }
}

struct CapturedOutput(Arc<Mutex<Vec<Vec<u8>>>>);

struct RawRecorder(Arc<AtomicU64>);
impl crate::core::types::StackedReplayRecorder for RawRecorder {
    fn record_yuv(
        &mut self,
        _: &crate::render::planes::Yuv420pPlanes<'_>,
        _: &crate::render::planes::Yuv420pPlanes<'_>,
        _: u32,
        _: u32,
    ) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

impl OutputSink for CapturedOutput {
    fn name(&self) -> &str {
        "captured-output"
    }
    fn wants(&self) -> SinkInput {
        SinkInput::CpuBytes(crate::sink::PixelFormat::Nv12)
    }
    fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
        self.0.lock().unwrap().push(frame.data.to_vec());
        Ok(())
    }
    fn finish(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

struct LabelledSource {
    source: Source,
    full_range: bool,
}
impl FrameSource for LabelledSource {
    fn info(&self) -> SourceInfo {
        self.source.info()
    }
    fn is_full_range(&self) -> bool {
        self.full_range
    }
    fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
        let mut frame = self.source.next_frame()?;
        match frame.as_mut().map(|frame| &mut frame.images) {
            Some(FrameSet::Yuv420p(cameras)) => {
                for camera in cameras {
                    Arc::make_mut(camera).y.iter_mut().for_each(|v| *v += 32);
                }
            }
            Some(FrameSet::Nv12(cameras)) => {
                for camera in cameras {
                    Arc::make_mut(camera).y.iter_mut().for_each(|v| *v += 32);
                }
            }
            _ => {}
        }
        Ok(frame)
    }
}

#[test]
fn cpu_buffered_output_preserves_order_manual_pose_replay_and_limits() {
    for nv12 in [false, true] {
        for full_range in [false, true] {
            for length in [0, 1, 2, 6] {
                for limit in [0, 1, 4, u64::MAX] {
                    let mut baseline = None;
                    for depth in [0, 1, 2, 10] {
                        let mut session = build_cpu_session();
                        session.core_mut().set_constrained_look(false);
                        session.core.set_requested_world_pose(WorldPose {
                            yaw: 0.25,
                            pitch: 0.125,
                            fov_degrees: 64.0,
                        });
                        session
                            .core_mut()
                            .enable_replay_buffer(Some(std::time::Duration::from_secs(60)));
                        session.set_lookahead(depth, usize::MAX);
                        let raw_frames = Arc::new(AtomicU64::new(0));
                        session.set_stacked_recorder(Box::new(RawRecorder(raw_frames.clone())));
                        let images = Arc::new(Mutex::new(Vec::new()));
                        session
                            .add_sink(
                                Box::new(CapturedOutput(images.clone())),
                                crate::session::SinkOptions::inline_lossy(),
                            )
                            .unwrap();
                        let mut source = LabelledSource {
                            source: Source {
                                cameras: 2,
                                nv12,
                                index: 6 - length,
                            },
                            full_range,
                        };
                        let result = session
                            .run(&mut source, limit, &AtomicBool::new(false), None)
                            .unwrap();
                        session.finish().unwrap();
                        let count = u64::from(length).min(limit);
                        assert_eq!(result.frames_processed, count);
                        assert_eq!(session.core().frame_count(), count);
                        assert_eq!(
                            raw_frames.load(Ordering::Relaxed),
                            if nv12 { 0 } else { count }
                        );
                        let replay = session.core().replay_buffer().unwrap();
                        assert_eq!(replay.len(), count as usize);
                        let captured = images.lock().unwrap().clone();
                        assert_eq!(captured.len(), count as usize);
                        for pair in captured.windows(2) {
                            assert_ne!(pair[0], pair[1], "source labels must survive rendering");
                        }
                        let snapshot = (
                            captured,
                            replay
                                .iter()
                                .map(|f| (f.rgba.clone(), f.pose))
                                .collect::<Vec<_>>(),
                        );
                        if let Some(expected) = &baseline {
                            assert_eq!(&snapshot, expected, "depth {depth}");
                        } else {
                            baseline = Some(snapshot);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn cpu_buffered_failures_and_prefill_cancellation_leave_no_render_state() {
    struct FaultSource {
        remaining: usize,
        cancel: Option<Arc<AtomicBool>>,
    }
    impl FrameSource for FaultSource {
        fn info(&self) -> SourceInfo {
            MockSource::new(0).info()
        }
        fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
            if self.remaining == 0 {
                if let Some(flag) = &self.cancel {
                    flag.store(true, Ordering::Relaxed);
                    return Ok(Some(crate::source::SourceFrame::untimed(solid_frame())));
                }
                return Err(SourceError::Read {
                    reason: "injected source failure".into(),
                });
            }
            self.remaining -= 1;
            Ok(Some(crate::source::SourceFrame::untimed(solid_frame())))
        }
    }
    for cancel in [false, true] {
        for remaining in [0, 1, 4, 8] {
            let mut session = build_cpu_session();
            session.core_mut().set_constrained_look(false);
            session.set_lookahead(4, usize::MAX);
            let flag = Arc::new(AtomicBool::new(false));
            let result = session.run(
                &mut FaultSource {
                    remaining,
                    cancel: cancel.then(|| flag.clone()),
                },
                100,
                &flag,
                None,
            );
            if cancel {
                assert_eq!(result.unwrap().completion, RunCompletion::Cancelled);
            } else {
                assert!(matches!(result, Err(SessionError::Source(_))));
            }
            #[cfg(feature = "gpu")]
            assert!(session.current_vram_slot.is_none());
            session.finish().unwrap();
        }
    }
}

#[test]
fn cpu_buffered_rejects_bad_camera_count_before_inference_and_overflow_before_read() {
    for depth in [0, 2, usize::MAX] {
        let mut session = build_cpu_session();
        session.set_lookahead(depth, usize::MAX);
        let calls = Arc::new(AtomicU64::new(0));
        session.set_detector(Box::new(MockDetector::new(vec![], calls.clone())));
        let mut source = Source {
            cameras: 1,
            nv12: false,
            index: 0,
        };
        assert!(matches!(
            session.run(&mut source, 10, &AtomicBool::new(false), None),
            Err(SessionError::Config(_))
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        if depth == usize::MAX {
            assert_eq!(source.index, 0);
        }
    }
}

#[test]
fn cpu_buffered_propagates_sink_and_pose_failures_without_extra_output() {
    struct FailingSink(Arc<AtomicU64>);
    impl OutputSink for FailingSink {
        fn name(&self) -> &str {
            "failing"
        }
        fn wants(&self) -> SinkInput {
            SinkInput::CpuBytes(crate::sink::PixelFormat::Nv12)
        }
        fn consume(&mut self, _: OutputFrame<'_>) -> Result<(), SinkError> {
            if self.0.load(Ordering::Relaxed) == 2 {
                return Err(SinkError::Consume {
                    frame_index: Some(2),
                    reason: "injected sink failure".into(),
                });
            }
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        fn finish(&mut self) -> Result<(), SinkError> {
            Ok(())
        }
    }
    for depth in [0, 2, 10] {
        let mut session = build_cpu_session();
        session.core_mut().set_constrained_look(false);
        session.set_lookahead(depth, usize::MAX);
        let count = Arc::new(AtomicU64::new(0));
        session
            .add_sink(
                Box::new(FailingSink(count.clone())),
                crate::session::SinkOptions {
                    delivery: crate::session::SinkDelivery::Inline,
                    on_error: crate::session::SinkErrorPolicy::Abort,
                },
            )
            .unwrap();
        let error = session
            .run(&mut MockSource::new(12), 12, &AtomicBool::new(false), None)
            .unwrap_err();
        assert!(matches!(error, SessionError::Sink(_)));
        assert_eq!(count.load(Ordering::Relaxed), 2);
        assert_eq!(session.frame_count(), 2);
        #[cfg(feature = "gpu")]
        assert!(session.current_vram_slot.is_none());

        let mut session = build_cpu_session();
        session.core_mut().set_constrained_look(false);
        session.set_lookahead(depth, usize::MAX);
        session.set_panner(Box::new(NanPanner));
        assert!(matches!(
            session.run(&mut MockSource::new(12), 12, &AtomicBool::new(false), None),
            Err(SessionError::Core(_))
        ));
        assert_eq!(session.frame_count(), 0);
        assert_eq!(session.core().frame_count(), 0);
        #[cfg(feature = "gpu")]
        assert!(session.current_vram_slot.is_none());
    }
}

#[test]
fn cpu_session_receives_future_states_for_both_projections_and_formats() {
    for cameras in [1, 2] {
        for nv12 in [false, true] {
            for depth in [0, 1, 2, 10] {
                let cal = if cameras == 1 {
                    Calibration::new(
                        vec![Lens::flat(W, H)],
                        crate::projection::Cylinder::default(),
                        Framing {
                            axis_offset: 0.0,
                            tilt: 0.0,
                            roll: 0.0,
                        },
                    )
                } else {
                    test_calibration()
                };
                let executor = CpuExecutor::new(
                    cal,
                    ViewportSize {
                        width: 16,
                        height: 16,
                    },
                    W,
                    H,
                    false,
                )
                .unwrap();
                let mut session =
                    StitchSession::with_executor(Executor::Cpu(Box::new(executor))).unwrap();
                session.core_mut().set_constrained_look(false);
                let futures = Arc::new(Mutex::new(Vec::new()));
                let submitted = Arc::new(AtomicU64::new(0));
                session.set_panner(Box::new(FutureProbe(futures.clone())));
                session
                    .add_sink(
                        Box::new(MockEncoder::new(submitted.clone())),
                        crate::session::SinkOptions {
                            delivery: crate::session::SinkDelivery::Inline,
                            on_error: crate::session::SinkErrorPolicy::Abort,
                        },
                    )
                    .unwrap();
                session.set_lookahead(depth, usize::MAX);
                let result = session
                    .run(
                        &mut Source {
                            cameras,
                            nv12,
                            index: 0,
                        },
                        u64::MAX,
                        &AtomicBool::new(false),
                        None,
                    )
                    .unwrap();
                session.finish().unwrap();
                assert_eq!(result.frames_processed, 6);
                assert_eq!(session.frame_count(), 6);
                assert_eq!(session.core().frame_count(), 6);
                assert_eq!(submitted.load(Ordering::Relaxed), 6);
                let futures = futures.lock().unwrap();
                assert_eq!(futures.len(), 6);
                assert_eq!(futures.iter().any(|n| *n > 0), depth > 0, "depth {depth}");
                assert_eq!(futures.last(), Some(&0));
            }
        }
    }
}

impl Panner for FutureProbe {
    fn decide(&mut self, world: &WorldState, ctx: &PanContext<'_>) -> WorldPose {
        self.decide_with_lookahead(world, &[], ctx)
    }

    fn decide_with_lookahead(
        &mut self,
        _: &WorldState,
        future: &[WorldState],
        _: &PanContext<'_>,
    ) -> WorldPose {
        self.0.lock().unwrap().push(future.len());
        WorldPose {
            yaw: if future.is_empty() { -0.2 } else { 0.2 },
            ..Default::default()
        }
    }
}

#[test]
#[ignore] // requires a hardware GPU
#[cfg(feature = "gpu")]
fn cylinder_gpu_session_receives_future_states_and_drains_both_formats() {
    use crate::gpu::GpuContext;
    use crate::render::renderer::InputFormat;
    use crate::stitch::{GpuExecutor, GpuExecutorConfig};
    let gpu = GpuContext::new_blocking().unwrap();
    for nv12 in [false, true] {
        for depth in [0, 2, 10] {
            let cal = Calibration::new(
                vec![Lens::flat(W, H)],
                crate::projection::Cylinder {
                    radius: 64.0,
                    ..Default::default()
                },
                Framing {
                    axis_offset: 0.0,
                    tilt: 0.0,
                    roll: 0.0,
                },
            );
            let executor = GpuExecutor::new(
                gpu.clone(),
                GpuExecutorConfig {
                    viewport_size: ViewportSize {
                        width: 16,
                        height: 16,
                    },
                    ..GpuExecutorConfig::new(
                        cal,
                        W,
                        H,
                        if nv12 {
                            InputFormat::Nv12
                        } else {
                            InputFormat::Yuv420p
                        },
                    )
                },
            )
            .unwrap();
            let mut session =
                StitchSession::with_executor(Executor::Gpu(Box::new(executor))).unwrap();
            session.core_mut().set_constrained_look(false);
            let futures = Arc::new(Mutex::new(Vec::new()));
            let calls = Arc::new(Mutex::new(Vec::new()));
            let observations: Observations = Arc::new(Mutex::new(Vec::new()));
            let submitted = Arc::new(AtomicU64::new(0));
            session
                .add_sink(
                    Box::new(MockEncoder::new(submitted.clone())),
                    crate::session::SinkOptions {
                        delivery: crate::session::SinkDelivery::Inline,
                        on_error: crate::session::SinkErrorPolicy::Abort,
                    },
                )
                .unwrap();
            session.set_detector(Box::new(ProbeDetector {
                nv12,
                calls: calls.clone(),
            }));
            session.set_player_tracker(Box::new(ProbeTracker(observations.clone())));
            session.set_panner(Box::new(FutureProbe(futures.clone())));
            session.set_detection_interval(3);
            session.set_lookahead(depth, usize::MAX);
            let result = session
                .run(
                    &mut Source {
                        cameras: 1,
                        nv12,
                        index: 0,
                    },
                    u64::MAX,
                    &AtomicBool::new(false),
                    None,
                )
                .unwrap();
            session.finish().unwrap();
            assert_eq!(result.frames_processed, 6);
            assert_eq!(submitted.load(Ordering::Relaxed), 6);
            assert_eq!(calls.lock().unwrap().len(), 2);
            assert_eq!(observations.lock().unwrap().len(), 6);
            let futures = futures.lock().unwrap();
            assert_eq!(futures.len(), 6);
            assert_eq!(futures.iter().any(|n| *n > 0), depth > 0);
            assert_eq!(futures.last(), Some(&0));
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Measurement {
    camera: usize,
    source: (f32, f32),
    position: YawPitch,
}

type Observations = Arc<Mutex<Vec<(bool, Vec<Measurement>)>>>;

struct ProbeDetector {
    nv12: bool,
    calls: Arc<Mutex<Vec<(usize, u8)>>>,
}

impl Detector for ProbeDetector {
    fn name(&self) -> &'static str {
        "cpu-camera-probe"
    }

    fn detect(
        &mut self,
        camera: usize,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let DetectorFrame::Yuv(view) = frame else {
            panic!("expected CPU planes")
        };
        assert_eq!((view.width, view.height), (W, H));
        assert_eq!(matches!(view.chroma, ChromaPlanes::Nv12 { .. }), self.nv12);
        let value = view.y[0];
        assert_eq!(usize::from(value % 8), camera);
        self.calls.lock().unwrap().push((camera, value));
        if value / 8 == 3 {
            return Ok(Vec::new());
        }
        Ok(vec![Detection {
            camera,
            class_id: 1,
            confidence: 0.9,
            center_x: 0.3 + f32::from(value) / 200.0,
            center_y: 0.4,
            width: 0.05,
            height: 0.05,
        }])
    }
}

struct ProbeTracker(Observations);
impl Tracker for ProbeTracker {
    fn class_id(&self) -> u16 {
        1
    }
    fn update(
        &mut self,
        detections: &[MappedDetection],
        fresh_cameras: &[crate::geometry::CameraIndex],
        _: f64,
    ) -> Vec<TrackedEntity> {
        let fresh = !fresh_cameras.is_empty();
        self.0.lock().unwrap().push((
            fresh,
            detections
                .iter()
                .map(|d| Measurement {
                    camera: d.camera,
                    source: d.camera_center,
                    position: d.position.expect("mapped direction"),
                })
                .collect(),
        ));
        detections
            .iter()
            .enumerate()
            .map(|(id, d)| {
                let position = d.position.unwrap();
                TrackedEntity {
                    id: id as u64,
                    class_id: 1,
                    yaw: position.yaw,
                    pitch: position.pitch,
                    confidence: d.confidence,
                    state: TrackState::Tracking,
                    age_frames: 1,
                    origin: d.camera,
                }
            })
            .collect()
    }
}

struct Source {
    cameras: usize,
    nv12: bool,
    index: u8,
}
impl FrameSource for Source {
    fn info(&self) -> SourceInfo {
        SourceInfo {
            width: W,
            height: H,
            fps: 30.0,
            fps_rational: Some((30, 1)),
            total_frames: Some(6),
        }
    }
    fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
        if self.index == 6 {
            return Ok(None);
        }
        let index = self.index;
        self.index += 1;
        let y = |camera| vec![index * 8 + camera as u8; (W * H) as usize];
        Ok(Some(crate::source::SourceFrame::untimed(if self.nv12 {
            FrameSet::Nv12(
                (0..self.cameras)
                    .map(|camera| Nv12Data {
                        y: y(camera),
                        uv: vec![128; (W * H / 2) as usize],
                    })
                    .map(Arc::new)
                    .collect(),
            )
        } else {
            FrameSet::Yuv420p(
                (0..self.cameras)
                    .map(|camera| Yuv420pData {
                        y: y(camera),
                        u: vec![128; (W * H / 4) as usize],
                        v: vec![128; (W * H / 4) as usize],
                    })
                    .map(Arc::new)
                    .collect(),
            )
        })))
    }
}

#[test]
fn cpu_detection_matches_push_and_session_for_both_projections() {
    for cameras in [1, 2] {
        let cal = if cameras == 1 {
            Calibration::new(
                vec![Lens::flat(W, H)],
                crate::projection::Cylinder {
                    radius: 64.0,
                    ..Default::default()
                },
                Framing {
                    axis_offset: 0.0,
                    tilt: 0.13,
                    roll: -0.09,
                },
            )
        } else {
            test_calibration()
        };
        for nv12 in [false, true] {
            for interval in [1, 3] {
                let mut baseline = None;
                // None is direct push; Some is the requested session lookahead depth.
                for lookahead in [None, Some(0), Some(2)] {
                    let executor = CpuExecutor::new(
                        cal.clone(),
                        ViewportSize {
                            width: 16,
                            height: 16,
                        },
                        W,
                        H,
                        false,
                    )
                    .unwrap();
                    let mut session =
                        StitchSession::with_executor(Executor::Cpu(Box::new(executor))).unwrap();
                    let calls = Arc::new(Mutex::new(Vec::new()));
                    let observations: Observations = Arc::new(Mutex::new(Vec::new()));
                    session.set_detector(Box::new(ProbeDetector {
                        nv12,
                        calls: calls.clone(),
                    }));
                    session.set_detection_interval(interval);
                    session.set_player_tracker(Box::new(ProbeTracker(observations.clone())));
                    session.set_panner(Box::new(MockPanner {
                        position: WorldPose::default(),
                    }));
                    session.core_mut().set_constrained_look(false);
                    let mut source = Source {
                        cameras,
                        nv12,
                        index: 0,
                    };
                    if let Some(depth) = lookahead {
                        session.set_lookahead(depth, usize::MAX);
                        let result = session
                            .run(&mut source, u64::MAX, &AtomicBool::new(false), None)
                            .unwrap();
                        assert_eq!(result.frames_processed, 6);
                        session.finish().unwrap();
                    } else {
                        while let Some(frame) = source.next_frame().unwrap() {
                            session
                                .core_mut()
                                .submit_frame(&frame.images, 1.0 / 30.0)
                                .unwrap();
                        }
                    }
                    let calls = calls.lock().unwrap().clone();
                    let observations = observations.lock().unwrap().clone();
                    assert_eq!(observations.len(), 6);
                    assert_eq!(calls.len(), (6 / interval as usize) * cameras);
                    for (index, (fresh, measurements)) in observations.iter().enumerate() {
                        let measured_at = index / interval as usize * interval as usize;
                        assert_eq!(*fresh, (index as u64).is_multiple_of(interval));
                        assert_eq!(
                            measurements.len(),
                            if measured_at == 3 { 0 } else { cameras }
                        );
                        for (camera, measurement) in measurements.iter().enumerate() {
                            assert_eq!(measurement.camera, camera);
                            assert_eq!(
                                measurement.position,
                                cal.map_source_point(
                                    camera,
                                    [measurement.source.0, measurement.source.1]
                                )
                                .unwrap()
                            );
                        }
                    }
                    let snapshot = (calls, observations);
                    if let Some(expected) = &baseline {
                        assert_eq!(&snapshot, expected);
                    } else {
                        baseline = Some(snapshot);
                    }
                }
            }
        }
    }
}
