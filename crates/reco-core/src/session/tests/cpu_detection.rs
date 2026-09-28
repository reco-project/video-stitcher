//! CPU camera dispatch and mapped measurement parity across submission paths.

use super::*;
use crate::detect::detector::ChromaPlanes;
use crate::geometry::YawPitch;
use crate::source::Nv12Data;
use crate::stitch::{CpuExecutor, Executor};

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
        fresh: bool,
        _: f64,
    ) -> Vec<TrackedEntity> {
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
    fn next_frame(&mut self) -> Result<Option<FrameSet>, SourceError> {
        if self.index == 6 {
            return Ok(None);
        }
        let index = self.index;
        self.index += 1;
        let y = |camera| vec![index * 8 + camera as u8; (W * H) as usize];
        Ok(Some(if self.nv12 {
            FrameSet::Nv12(
                (0..self.cameras)
                    .map(|camera| Nv12Data {
                        y: y(camera),
                        uv: vec![128; (W * H / 2) as usize],
                    })
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
                    .collect(),
            )
        }))
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
                for lookahead in [None, Some(0)] {
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
                        session.set_lookahead(depth);
                        let result = session
                            .run(&mut source, u64::MAX, &AtomicBool::new(false), None)
                            .unwrap();
                        assert_eq!(result.frames_processed, 6);
                        session.finish().unwrap();
                    } else {
                        while let Some(frame) = source.next_frame().unwrap() {
                            session.core_mut().submit_frame(&frame, 1.0 / 30.0).unwrap();
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
