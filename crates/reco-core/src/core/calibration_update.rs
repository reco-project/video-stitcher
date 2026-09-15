//! Acceptance boundary for live calibration edits.

use super::{StitchCore, StitchCoreError};
use crate::calibration::Calibration;

impl StitchCore {
    /// Apply an edited calibration without changing input layout or topology kind.
    ///
    /// Invalid documents, changed image dimensions or topology kind, and engines with an attached tracker or panner are rejected without changing the accepted state. Coverage and backend geometry are refreshed together on success; cached detections are cleared.
    pub fn update_calibration(&mut self, calibration: Calibration) -> Result<(), StitchCoreError> {
        calibration.validate()?;
        let current = self.calibration();
        if std::mem::discriminant(&current.topology)
            != std::mem::discriminant(&calibration.topology)
            || current.lenses.len() != calibration.lenses.len()
            || current
                .lenses
                .iter()
                .zip(&calibration.lenses)
                .any(|(old, new)| (old.width, old.height) != (new.width, new.height))
        {
            return Err(StitchCoreError::CalibrationRequiresRebuild(
                "topology kind or calibration image dimensions changed",
            ));
        }
        if self.ball_tracker.is_some() || self.player_tracker.is_some() || self.panner.is_some() {
            return Err(StitchCoreError::CalibrationRequiresRebuild(
                "camera control retains state from the current calibration",
            ));
        }
        let coverage = calibration.topology.projection().coverage(&calibration);
        self.executor.apply_calibration(calibration);
        self.coverage = Some(coverage);
        self.last_detections.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{Lens, Topology};
    use crate::detect::director::MappedDetection;
    use crate::detect::panner::{PanContext, Panner};
    use crate::detect::tracker::{TrackedEntity, Tracker, WorldState};
    use crate::geometry::Pose;
    use crate::projection::Cylinder;
    use crate::render::planes::Nv12Planes;
    use crate::render::viewport::ViewportSize;
    use crate::stitch::test_support::{calib, nv12};
    use crate::stitch::{CpuExecutor, Executor, StitchExecutor};

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 36;

    fn cpu(calibration: Calibration) -> StitchCore {
        let executor = CpuExecutor::new(
            calibration,
            ViewportSize {
                width: WIDTH,
                height: HEIGHT,
            },
            WIDTH,
            HEIGHT,
            false,
        )
        .unwrap();
        StitchCore::new(Executor::Cpu(Box::new(executor))).unwrap()
    }

    fn render(core: &mut StitchCore) -> Vec<u8> {
        let (y, uv) = nv12(WIDTH, HEIGHT, 20);
        let planes = vec![Nv12Planes { y: &y, uv: &uv }; core.camera_count()];
        core.executor
            .stitch(
                &planes,
                Pose {
                    yaw: 0.1,
                    pitch: -0.03,
                    fov_degrees: 70.0,
                },
            )
            .unwrap()
    }

    fn detection() -> MappedDetection {
        MappedDetection {
            camera: 0,
            class_id: 0,
            confidence: 1.0,
            camera_center: (0.5, 0.5),
            camera_size: (0.1, 0.1),
            position: Some(Pose::default()),
        }
    }

    fn rejection_preserves_state(make: impl Fn(Calibration) -> StitchCore) {
        let calibration = calib(WIDTH, HEIGHT);
        let mut core = make(calibration.clone());
        core.last_detections.push(detection());
        let accepted = serde_json::to_value(core.calibration()).unwrap();
        let pixels = render(&mut core);
        let coverage = core.coverage().unwrap().yaw_range();
        let mut drafts = Vec::new();
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
            let mut draft = calibration.clone();
            draft.framing.axis_offset = invalid;
            drafts.push(draft);
        }
        let mut draft = calibration.clone();
        draft.lenses[0].fx = 0.0;
        drafts.push(draft);
        let mut draft = calibration.clone();
        draft.lenses[1].correction = f32::NAN;
        drafts.push(draft);
        let mut draft = calibration.clone();
        draft.lenses.pop();
        drafts.push(draft);
        for draft in drafts {
            assert!(matches!(
                core.update_calibration(draft),
                Err(StitchCoreError::Calibration(_))
            ));
            assert_eq!(serde_json::to_value(core.calibration()).unwrap(), accepted);
            assert_eq!(core.coverage().unwrap().yaw_range(), coverage);
            assert_eq!(render(&mut core), pixels);
            assert_eq!(core.last_detections.len(), 1);
        }
        let mut resized = calibration.clone();
        resized.lenses[0].width *= 2;
        let mono = Calibration::new(
            vec![Lens::flat(WIDTH, HEIGHT)],
            Cylinder::default(),
            calibration.framing.clone(),
        );
        for draft in [resized, mono] {
            draft.validate().unwrap();
            assert!(matches!(
                core.update_calibration(draft),
                Err(StitchCoreError::CalibrationRequiresRebuild(_))
            ));
            assert_eq!(serde_json::to_value(core.calibration()).unwrap(), accepted);
            assert_eq!(render(&mut core), pixels);
        }
    }

    fn accepted_matches_fresh(make: impl Fn(Calibration) -> StitchCore, calibration: Calibration) {
        let mut core = make(calibration);
        // Exercise repeated edits, including correction and topology-specific geometry.
        for step in 0..4 {
            let mut draft = core.calibration().clone();
            match step {
                0 => {
                    draft.framing.tilt = 0.21;
                    draft.framing.roll = -0.08;
                }
                1 => {
                    draft.lenses[0].fx *= 1.1;
                    draft.lenses[0].cx += 3.0;
                }
                2 => {
                    for lens in &mut draft.lenses {
                        lens.correction = 0.4;
                    }
                }
                _ => match &mut draft.topology {
                    Topology::LShape(layout) => {
                        layout.intersect = 0.45;
                        layout.blend_width = 0.1;
                        draft.framing.axis_offset = 0.3;
                    }
                    Topology::Cylinder(cylinder) => cylinder.sweep_deg = 240.0,
                },
            }
            let mut fresh = make(draft.clone());
            core.last_detections.push(detection());
            core.update_calibration(draft).unwrap();
            assert!(core.last_detections.is_empty());
            assert_eq!(
                serde_json::to_value(core.calibration()).unwrap(),
                serde_json::to_value(fresh.calibration()).unwrap()
            );
            assert_eq!(
                core.coverage().unwrap().yaw_range(),
                fresh.coverage().unwrap().yaw_range()
            );
            assert_eq!(
                core.coverage().unwrap().pitch_range(),
                fresh.coverage().unwrap().pitch_range()
            );
            for yaw in [-3.0, 0.0, 3.0] {
                let pose = Pose {
                    yaw,
                    pitch: 1.0,
                    fov_degrees: 60.0,
                };
                let actual = core.safe_clamp(pose);
                let expected = fresh.safe_clamp(pose);
                assert_eq!((actual.yaw, actual.pitch), (expected.yaw, expected.pitch));
            }
            assert_eq!(render(&mut core), render(&mut fresh));
        }
    }

    #[test]
    fn cpu_rejected_edits_preserve_document_coverage_detections_and_render() {
        rejection_preserves_state(cpu);
    }

    #[test]
    fn cpu_live_edits_match_fresh_engines() {
        accepted_matches_fresh(cpu, calib(WIDTH, HEIGHT));
        accepted_matches_fresh(
            cpu,
            Calibration::new(
                vec![Lens::flat(WIDTH, HEIGHT)],
                Cylinder {
                    focal_length: 32.0,
                    ..Default::default()
                },
                calib(WIDTH, HEIGHT).framing,
            ),
        );
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_live_edits_are_atomic_and_match_fresh_engines() {
        use crate::render::renderer::InputFormat;
        use crate::stitch::{GpuExecutor, GpuExecutorConfig};
        let Some(gpu) = crate::stitch::test_support::gpu_or_skip() else {
            return;
        };
        let make = |calibration| {
            let executor = GpuExecutor::new(
                gpu.clone(),
                GpuExecutorConfig {
                    viewport_size: ViewportSize {
                        width: WIDTH,
                        height: HEIGHT,
                    },
                    ..GpuExecutorConfig::new(calibration, WIDTH, HEIGHT, InputFormat::Nv12)
                },
            )
            .unwrap();
            StitchCore::new(Executor::Gpu(Box::new(executor))).unwrap()
        };
        rejection_preserves_state(make);
        accepted_matches_fresh(make, calib(WIDTH, HEIGHT));
    }

    struct NoopControl;

    impl Tracker for NoopControl {
        fn update(&mut self, _: &[MappedDetection], _: f64) -> Vec<TrackedEntity> {
            vec![]
        }
        fn class_id(&self) -> u16 {
            0
        }
    }

    impl Panner for NoopControl {
        fn decide(&mut self, _: &WorldState, _: &PanContext<'_>) -> Pose {
            Pose::default()
        }
    }

    #[test]
    fn attached_camera_control_requires_reconstruction() {
        for slot in 0..3 {
            let mut core = cpu(calib(WIDTH, HEIGHT));
            match slot {
                0 => core.set_ball_tracker(Box::new(NoopControl)),
                1 => core.set_player_tracker(Box::new(NoopControl)),
                _ => core.set_panner(Box::new(NoopControl)),
            }
            let accepted = serde_json::to_value(core.calibration()).unwrap();
            let mut draft = core.calibration().clone();
            draft.framing.tilt = 0.2;
            assert!(matches!(
                core.update_calibration(draft),
                Err(StitchCoreError::CalibrationRequiresRebuild(_))
            ));
            assert_eq!(serde_json::to_value(core.calibration()).unwrap(), accepted);
        }
    }
}
