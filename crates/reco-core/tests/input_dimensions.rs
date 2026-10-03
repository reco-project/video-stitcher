//! Independent decoded extents preserve calibrated geometry and camera identity.

use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::core::StitchCore;
use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
use reco_core::geometry::RenderPose;
use reco_core::projection::LShape;
use reco_core::render::viewport::ViewportSize;
use reco_core::session::StitchSession;
use reco_core::source::{
    FrameSet, FrameSource, Nv12Data, SourceError, SourceFrame, SourceInfo, Yuv420pData,
};
use reco_core::stitch::{CpuExecutor, Executor};
use std::sync::{Arc, Mutex, atomic::AtomicBool};

fn calibration() -> Calibration {
    Calibration::new(
        vec![Lens::flat(64, 32); 2],
        LShape {
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.08,
        },
        Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        },
    )
}

fn viewport() -> ViewportSize {
    ViewportSize {
        width: 128,
        height: 72,
    }
}

fn image((width, height): (u32, u32)) -> Nv12Data {
    Nv12Data {
        size: (width, height),
        y: (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let u = (x as f64 + 0.5) / width as f64;
                    let v = (y as f64 + 0.5) / height as f64;
                    (30.0 + 100.0 * u + 90.0 * v) as u8
                })
            })
            .collect(),
        uv: vec![128; (width * height / 2) as usize],
    }
}

fn frames(sizes: &[(u32, u32)], planar: bool) -> FrameSet {
    if planar {
        FrameSet::Yuv420p(
            sizes
                .iter()
                .map(|&size| {
                    let image = image(size);
                    Arc::new(Yuv420pData {
                        size,
                        y: image.y,
                        u: vec![128; (size.0 * size.1 / 4) as usize],
                        v: vec![128; (size.0 * size.1 / 4) as usize],
                    })
                })
                .collect(),
        )
    } else {
        FrameSet::Nv12(sizes.iter().map(|&size| Arc::new(image(size))).collect())
    }
}

fn cpu(sizes: Vec<(u32, u32)>) -> CpuExecutor {
    CpuExecutor::new(calibration(), viewport(), sizes, false).unwrap()
}

fn error(actual: &[u8], expected: &[u8]) -> (f64, u8) {
    assert_eq!(actual.len(), expected.len());
    let mut sum = 0usize;
    let mut max = 0;
    for (&a, &b) in actual.iter().zip(expected) {
        let d = a.abs_diff(b);
        sum += usize::from(d);
        max = max.max(d);
    }
    (sum as f64 / actual.len() as f64, max)
}

#[test]
fn independent_sizes_preserve_normalized_sampling_in_both_orders() {
    let reference_images = [(256, 128); 2].map(image);
    let reference = cpu(vec![(256, 128); 2]);
    for sizes in [
        vec![(64, 32), (128, 64)],
        vec![(128, 64), (64, 32)],
        vec![(64, 32), (128, 16)],
        vec![(128, 16), (64, 32)],
    ] {
        let executor = cpu(sizes.clone());
        let data: Vec<_> = sizes.iter().copied().map(image).collect();
        for yaw in [-0.7, 0.0, 0.7] {
            let pose = RenderPose {
                yaw,
                pitch: 0.0,
                fov_degrees: 80.0,
            };
            let expected = reference
                .stitch_nv12(
                    &reference_images
                        .iter()
                        .map(Nv12Data::as_planes)
                        .collect::<Vec<_>>(),
                    pose,
                )
                .unwrap();
            let actual = executor
                .stitch_nv12(
                    &data.iter().map(Nv12Data::as_planes).collect::<Vec<_>>(),
                    pose,
                )
                .unwrap();
            let (mean, max) = error(&actual, &expected);
            assert!(
                mean < 0.6 && max <= 5,
                "sizes={sizes:?} yaw={yaw}: {mean}/{max}"
            );
        }
    }
}

type Calls = Arc<Mutex<Vec<(usize, (u32, u32))>>>;
struct ShapeDetector(Calls);
impl Detector for ShapeDetector {
    fn name(&self) -> &'static str {
        "shape-detector"
    }
    fn detect(
        &mut self,
        camera: usize,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let DetectorFrame::Yuv(view) = frame else {
            panic!("expected host image")
        };
        assert_eq!(view.y.len(), (view.width * view.height) as usize);
        self.0
            .lock()
            .unwrap()
            .push((camera, (view.width, view.height)));
        Ok(vec![])
    }
}

struct Source {
    sizes: Vec<(u32, u32)>,
    planar: bool,
    remaining: usize,
}
impl FrameSource for Source {
    fn info(&self) -> SourceInfo {
        SourceInfo {
            camera_sizes: self.sizes.clone(),
            fps: 30.0,
            fps_rational: Some((30, 1)),
            total_frames: Some(6),
        }
    }
    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        Ok(Some(SourceFrame::untimed(frames(&self.sizes, self.planar))))
    }
}

#[test]
fn immediate_and_lookahead_detection_receive_each_image_shape() {
    for planar in [false, true] {
        for sizes in [
            vec![(64, 32), (128, 64)],
            vec![(128, 64), (64, 32)],
            vec![(64, 32), (128, 16)],
        ] {
            for depth in [0, 3] {
                let mut session =
                    StitchSession::with_executor(Executor::Cpu(Box::new(cpu(sizes.clone()))))
                        .unwrap();
                session.core_mut().set_constrained_look(false);
                assert_eq!(session.core().calibration(), &calibration());
                let calls = Calls::default();
                session.set_detector(Box::new(ShapeDetector(calls.clone())));
                session.set_detection_interval(1);
                session.set_lookahead(depth, 1_000_000);
                let result = session
                    .run(
                        &mut Source {
                            sizes: sizes.clone(),
                            planar,
                            remaining: 6,
                        },
                        6,
                        &AtomicBool::new(false),
                        None,
                    )
                    .unwrap();
                assert_eq!(result.frames_processed, 6);
                assert_eq!(
                    *calls.lock().unwrap(),
                    (0..6)
                        .flat_map(|_| sizes.iter().copied().enumerate())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn equal_byte_count_wrong_shape_is_rejected_before_detection() {
    for planar in [false, true] {
        let mut core = StitchCore::new(Executor::Cpu(Box::new(cpu(vec![(64, 32); 2])))).unwrap();
        core.set_constrained_look(false);
        let calls = Calls::default();
        core.set_detector(Box::new(ShapeDetector(calls.clone())));
        assert!(
            core.submit_frame(&frames(&[(64, 32), (128, 16)], planar), 1.0 / 30.0)
                .is_err()
        );
        assert!(calls.lock().unwrap().is_empty());

        let mut session =
            StitchSession::with_executor(Executor::Cpu(Box::new(cpu(vec![(64, 32); 2])))).unwrap();
        assert!(
            session
                .render_frame(
                    &frames(&[(64, 32), (128, 16)], planar),
                    RenderPose::default(),
                    None,
                )
                .is_err()
        );
    }
}

#[test]
fn replay_rejects_unsupported_layout_at_attachment() {
    use reco_core::core::types::StackedReplayRecorder;
    use reco_core::render::planes::Yuv420pPlanes;

    struct Recorder;
    impl StackedReplayRecorder for Recorder {
        fn record_yuv(&mut self, _: &Yuv420pPlanes<'_>, _: &Yuv420pPlanes<'_>, _: u32, _: u32) {
            panic!("no frames submitted");
        }
    }

    for (sizes, accepted) in [
        (vec![(64, 32); 2], true),
        (vec![(64, 32), (128, 16)], false),
    ] {
        let mut core = StitchCore::new(Executor::Cpu(Box::new(cpu(sizes)))).unwrap();
        assert_eq!(
            core.set_stacked_recorder(Box::new(Recorder)).is_ok(),
            accepted
        );
    }
    let mono = Calibration::new(
        vec![Lens::flat(64, 32)],
        reco_core::projection::Cylinder {
            radius: 20.0,
            arc_deg: 180.0,
            surface_height: None,
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let executor = CpuExecutor::new(mono, viewport(), vec![(64, 32)], false).unwrap();
    let mut core = StitchCore::new(Executor::Cpu(Box::new(executor))).unwrap();
    assert!(core.set_stacked_recorder(Box::new(Recorder)).is_err());
}

#[test]
fn input_resolution_does_not_change_calibrated_geometry() {
    let mut reference = StitchCore::new(Executor::Cpu(Box::new(cpu(vec![(64, 32); 2])))).unwrap();
    let mut mixed =
        StitchCore::new(Executor::Cpu(Box::new(cpu(vec![(128, 64), (128, 16)])))).unwrap();
    assert_eq!(mixed.calibration(), reference.calibration());
    assert_eq!(
        mixed.prepare_coverage().unwrap(),
        reference.prepare_coverage().unwrap()
    );
    for camera in 0..2 {
        assert_eq!(
            mixed.calibration().map_source_point(camera, [0.3, 0.4]),
            reference.calibration().map_source_point(camera, [0.3, 0.4])
        );
    }
}

#[test]
fn source_layout_is_validated_before_processing() {
    for sizes in [
        vec![],
        vec![(64, 32)],
        vec![(64, 32), (0, 32)],
        vec![(64, 32), (63, 32)],
    ] {
        assert!(CpuExecutor::new(calibration(), viewport(), sizes, false).is_err());
    }
    let mut session =
        StitchSession::with_executor(Executor::Cpu(Box::new(cpu(vec![(64, 32); 2])))).unwrap();
    let mut source = Source {
        sizes: vec![(64, 32), (128, 16)],
        planar: true,
        remaining: 6,
    };
    assert!(
        session
            .run(&mut source, 6, &AtomicBool::new(false), None)
            .is_err()
    );
    assert_eq!(source.remaining, 6);
}

#[test]
#[cfg(feature = "gpu")]
fn gpu_upload_matches_independent_cpu_sampling() {
    use reco_core::gpu::GpuContext;
    use reco_core::render::renderer::InputFormat;
    use reco_core::stitch::{GpuExecutor, GpuExecutorConfig, StitchExecutor};
    let Ok(gpu) = GpuContext::new_blocking() else {
        return;
    };
    eprintln!("dimension oracle: {}", gpu.gpu_name());
    for sizes in [
        vec![(64, 32), (128, 64)],
        vec![(128, 64), (64, 32)],
        vec![(64, 32), (128, 16)],
        vec![(128, 16), (64, 32)],
    ] {
        let cpu = cpu(sizes.clone());
        let mut executor = GpuExecutor::new(
            gpu.clone(),
            GpuExecutorConfig {
                viewport_size: viewport(),
                ..GpuExecutorConfig::new(calibration(), sizes.clone(), InputFormat::Nv12)
            },
        )
        .unwrap();
        let data: Vec<_> = sizes.iter().copied().map(image).collect();
        let planes: Vec<_> = data.iter().map(Nv12Data::as_planes).collect();
        for yaw in [-0.7, 0.0, 0.7] {
            let pose = RenderPose {
                yaw,
                pitch: 0.0,
                fov_degrees: 80.0,
            };
            let expected = cpu.stitch_nv12(&planes, pose).unwrap();
            let actual = executor.stitch(&planes, pose).unwrap();
            let (mean, max) = error(&actual, &expected);
            assert!(mean < 0.1 && max <= 1, "sizes={sizes:?}: {mean}/{max}");
        }
    }
}
