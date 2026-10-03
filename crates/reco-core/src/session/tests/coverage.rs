//! Presentation advances after immediate or buffered camera decisions.

use super::*;
use crate::detect::pipeline_event::{PipelineEvent, PipelineEventSink};
use crate::projection::coverage::controller::{Controller, MapConfig};
use crate::stitch::{CpuExecutor, Executor};

struct Events(Arc<Mutex<Vec<PipelineEvent>>>);
impl PipelineEventSink for Events {
    fn emit(&mut self, event: PipelineEvent) {
        self.0.lock().unwrap().push(event);
    }
}

struct Steps;

#[test]
fn source_rates_are_accepted_in_immediate_and_buffered_sessions() {
    struct RatedSource {
        source: MockSource,
        fps: f64,
    }
    impl FrameSource for RatedSource {
        fn info(&self) -> SourceInfo {
            SourceInfo {
                fps: self.fps,
                fps_rational: None,
                ..self.source.info()
            }
        }
        fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
            self.source.next_frame()
        }
    }
    for fps in [0.5, 300.0, 1000.0] {
        for lookahead in [0, 4] {
            let cal = test_calibration();
            let size = ViewportSize {
                width: W,
                height: H,
            };
            let executor = {
                let source_sizes = vec![(W, H); cal.projection.camera_count()];
                CpuExecutor::new(cal.clone(), size.clone(), source_sizes, false)
            }
            .unwrap();
            let mut session =
                StitchSession::with_executor(Executor::Cpu(Box::new(executor))).unwrap();
            session.set_lookahead(lookahead, usize::MAX);
            let mut source = RatedSource {
                source: MockSource::new(8),
                fps,
            };
            assert_eq!(
                session
                    .run(&mut source, 8, &AtomicBool::new(false), None)
                    .unwrap()
                    .frames_processed,
                8
            );
            session.finish().unwrap();
            assert!(session.core_mut().current_pose().yaw.is_finite());
        }
    }
}
impl Panner for Steps {
    fn decide(&mut self, _world: &WorldState, ctx: &PanContext<'_>) -> WorldPose {
        WorldPose {
            yaw: if ctx.frame_index < 60 { -0.5 } else { 0.5 },
            pitch: 0.05,
            fov_degrees: 100.0,
        }
    }
}

#[test]
fn coverage_runs_once_after_immediate_or_lookahead_decisions() {
    let mut cal = test_calibration();
    cal.framing.tilt = 0.15;
    cal.framing.roll = -0.12;
    let size = ViewportSize {
        width: W,
        height: H,
    };
    let config = MapConfig::default();
    for (gpu, lookahead) in [(false, 0), (false, 8), (true, 0), (true, 8)] {
        let executor = Executor::Cpu(Box::new(
            {
                let source_sizes = vec![(W, H); cal.projection.camera_count()];
                CpuExecutor::new(cal.clone(), size.clone(), source_sizes, false)
            }
            .unwrap(),
        ));
        #[cfg(not(feature = "gpu"))]
        if gpu {
            continue;
        }
        #[cfg(feature = "gpu")]
        let executor = if gpu {
            let Some(gpu) = crate::stitch::test_support::gpu_or_skip() else {
                continue;
            };
            Executor::Gpu(Box::new(
                crate::stitch::GpuExecutor::new(
                    gpu,
                    crate::stitch::GpuExecutorConfig {
                        viewport_size: size.clone(),
                        ..{
                            let source_sizes = vec![(W, H); cal.projection.camera_count()];
                            crate::stitch::GpuExecutorConfig::new(
                                cal.clone(),
                                source_sizes,
                                crate::render::renderer::InputFormat::Yuv420p,
                            )
                        }
                    },
                )
                .unwrap(),
            ))
        } else {
            executor
        };
        let mut session = StitchSession::with_executor(executor).unwrap();
        session.core_mut().prepare_coverage().unwrap();
        session.set_panner(Box::new(Steps));
        session.set_lookahead(lookahead, usize::MAX);
        let events = Arc::new(Mutex::new(Vec::new()));
        session.set_event_sink(Box::new(Events(events.clone())));
        assert_eq!(
            session
                .run(
                    &mut MockSource::new(180),
                    180,
                    &AtomicBool::new(false),
                    None
                )
                .unwrap()
                .frames_processed,
            180
        );
        session.finish().unwrap();
        let events = events.lock().unwrap();
        let requested: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                PipelineEvent::PanDecision { frame_index, pose } => Some((*frame_index, *pose)),
                _ => None,
            })
            .collect();
        let presented: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                PipelineEvent::RenderPoseResolved { frame_index, pose } => {
                    Some((*frame_index, *pose))
                }
                _ => None,
            })
            .collect();
        assert_eq!(requested.len(), 180);
        assert_eq!(presented.len(), 180);
        let mut expected = Controller::new(&cal, &size, config.clone()).unwrap();
        for (index, ((request_index, request), (actual_index, actual))) in
            requested.iter().zip(&presented).enumerate()
        {
            assert_eq!(*request_index, index as u64);
            assert_eq!(*actual_index, index as u64);
            assert_eq!(
                *actual,
                expected
                    .advance(&cal, &size, *request, index as u64, 1.0 / 30.0)
                    .unwrap()
            );
            assert_eq!(actual.fov_degrees, request.fov_degrees.min(expected.cap()));
        }
        let last = presented.last().unwrap().1;
        assert_eq!(session.core_mut().current_pose(), last);
        assert_eq!(session.core_mut().current_pose(), last);
        assert_eq!(
            session.core.requested_world_pose,
            requested.last().unwrap().1
        );
    }
}
