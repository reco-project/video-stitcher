//! Explicit-pose submissions preserve caller ownership of inference and motion.

use super::StitchCore;
use crate::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
use crate::detect::panner::{PanContext, Panner};
use crate::detect::tracker::WorldState;
use crate::geometry::{CameraIndex, WorldPose};
use crate::render::viewport::ViewportSize;
use crate::source::{FrameSet, Yuv420pData};
use crate::stitch::{CpuExecutor, Executor, test_support::calib};

struct UnexpectedInference;
impl Detector for UnexpectedInference {
    fn name(&self) -> &'static str {
        "unexpected"
    }
    fn detect(
        &mut self,
        _: CameraIndex,
        _: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        panic!("explicit-pose rendering must not run inference")
    }
}

struct UnexpectedPanner;
impl Panner for UnexpectedPanner {
    fn decide(&mut self, _: &WorldState, _: &PanContext<'_>) -> WorldPose {
        panic!("explicit-pose rendering must not advance the panner")
    }
}

fn attach(core: &mut StitchCore) {
    core.set_detector(Box::new(UnexpectedInference));
    core.set_panner(Box::new(UnexpectedPanner));
    core.enable_replay_buffer(Some(std::time::Duration::from_secs(1)));
}

#[test]
fn malformed_host_layout_does_not_advance_panner() {
    let executor = CpuExecutor::new(
        calib(64, 64),
        ViewportSize {
            width: 64,
            height: 64,
        },
        vec![(64, 64); 2],
        false,
    )
    .unwrap();
    let mut core = StitchCore::new(Executor::Cpu(Box::new(executor))).unwrap();
    core.set_panner(Box::new(UnexpectedPanner));
    let image = crate::source::Nv12Data {
        size: (128, 32),
        y: vec![128; 4096],
        uv: vec![128; 2048],
    };
    let frames = crate::source::FrameSet::Nv12(vec![image.clone().into(), image.into()]);
    assert!(core.submit_frame(&frames, 1.0 / 30.0).is_err());
    assert_eq!(core.frame_count(), 0);
}

#[test]
fn explicit_cpu_yuv_does_not_run_ai() {
    let executor = CpuExecutor::new(
        calib(64, 64),
        ViewportSize {
            width: 64,
            height: 64,
        },
        vec![(64, 64); 2],
        false,
    )
    .unwrap();
    let mut core = StitchCore::new(Executor::Cpu(Box::new(executor))).unwrap();
    attach(&mut core);
    let images = stereo_images();
    for _ in 0..3 {
        core.submit_frame_at_pose(&images, Default::default())
            .unwrap();
    }
    assert_eq!(core.frame_count(), 3);
    assert!(core.last_detections().is_empty());
    assert_eq!(core.replay_buffer().unwrap().len(), 3);
}

fn stereo_images() -> FrameSet {
    let image = Yuv420pData {
        size: (64, 64),
        y: vec![128; 4096],
        u: vec![128; 1024],
        v: vec![128; 1024],
    };
    FrameSet::Yuv420p(vec![image.clone().into(), image.into()])
}

#[cfg(feature = "gpu")]
#[test]
fn explicit_gpu_submit_does_not_run_ai() {
    use crate::render::renderer::InputFormat;
    use crate::stitch::{GpuExecutor, GpuExecutorConfig};
    let Some(gpu) = crate::stitch::test_support::gpu_or_skip() else {
        return;
    };
    let config = GpuExecutorConfig {
        viewport_size: ViewportSize {
            width: 64,
            height: 64,
        },
        ..GpuExecutorConfig::new(calib(64, 64), vec![(64, 64); 2], InputFormat::Yuv420p)
    };
    let mut core = StitchCore::new(Executor::Gpu(Box::new(
        GpuExecutor::new(gpu, config).unwrap(),
    )))
    .unwrap();
    attach(&mut core);
    let images = stereo_images();
    for _ in 0..3 {
        core.submit_frame_at_pose(&images, Default::default())
            .unwrap();
    }
    assert_eq!(core.frame_count(), 3);
    assert!(core.last_detections().is_empty());
    assert_eq!(core.replay_buffer().unwrap().len(), 1);
}
