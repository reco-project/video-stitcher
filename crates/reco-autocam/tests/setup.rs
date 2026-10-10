//! Camera-control setup without an inference model or GPU executor.

use std::sync::{Arc, Mutex};

use reco_autocam::{AutocamConfig, TrackingMode, setup_autocam};
use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::core::StitchCore;
use reco_core::detect::pipeline_event::{PipelineEvent, PipelineEventSink};
use reco_core::geometry::WorldPose;
use reco_core::projection::Cylinder;
use reco_core::render::viewport::ViewportSize;
use reco_core::source::{FrameSet, Nv12Data};
use reco_core::stitch::{CpuExecutor, Executor};

struct Decisions(Arc<Mutex<Vec<WorldPose>>>);

impl PipelineEventSink for Decisions {
    fn emit(&mut self, event: PipelineEvent) {
        if let PipelineEvent::PanDecision { pose, .. } = event {
            self.0.lock().unwrap().push(pose);
        }
    }
}

#[test]
fn sweep_needs_no_model_or_detector_and_uses_source_fps() {
    let empty = tempfile::tempdir().unwrap();
    let config = AutocamConfig::new(empty.path().join("missing.onnx"))
        .with_tracking_mode(TrackingMode::Sweep);
    for fps in [24.0, 30.0, 60.0] {
        let calibration = Calibration::new(
            vec![Lens::flat(16, 8)],
            Cylinder {
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
        let executor = CpuExecutor::new(
            calibration,
            ViewportSize {
                width: 8,
                height: 4,
            },
            vec![(16, 8)],
            false,
        )
        .unwrap();
        let mut core = StitchCore::new(Executor::Cpu(Box::new(executor))).unwrap();
        core.set_constrained_look(false);
        assert!(setup_autocam(&mut core, &config, fps, false).unwrap());
        assert!(!core.detection_due(0));
        let decisions = Arc::new(Mutex::new(Vec::new()));
        core.set_event_sink(Box::new(Decisions(decisions.clone())));
        let frame = FrameSet::Nv12(vec![Arc::new(Nv12Data {
            size: (16, 8),
            y: vec![100; 128],
            uv: vec![128; 64],
        })]);
        let peak = (fps * 2.5) as usize;
        for _ in 0..=peak {
            core.submit_frame(&frame, 1.0 / fps).unwrap();
        }
        let decisions = decisions.lock().unwrap();
        assert_eq!(decisions.len(), peak + 1);
        assert!(decisions[0].yaw.abs() < 1e-6);
        assert!((decisions[peak].yaw - 0.8).abs() < 1e-6, "fps={fps}");
        assert!(
            decisions
                .iter()
                .all(|pose| (30.0..=90.0).contains(&pose.fov_degrees))
        );
    }
}
