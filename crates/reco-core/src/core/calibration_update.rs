//! Acceptance boundary for live calibration edits.

use super::{StitchCore, StitchCoreError};
use crate::calibration::Calibration;

impl StitchCore {
    /// Edit a private draft and accept it through [`Self::update_calibration`].
    ///
    /// The closure runs once. Rejected edits leave the engine unchanged.
    pub fn edit_calibration(
        &mut self,
        edit: impl FnOnce(&mut Calibration),
    ) -> Result<(), StitchCoreError> {
        let mut draft = self.calibration().clone();
        edit(&mut draft);
        self.update_calibration(draft)
    }

    /// Apply an edited calibration without changing input layout or topology kind.
    ///
    /// Invalid documents and changed image dimensions or topology kind are rejected without changing the accepted state. With a tracker or panner attached, only no-op and L-shape blend-width edits are accepted: neither changes mapping, framing, timing or ROI. These preserve coverage and cached detections; other accepted edits refresh coverage and clear detections. Backend geometry is refreshed on every changed document. See this module's CPU/GPU regression tests.
    pub fn update_calibration(&mut self, calibration: Calibration) -> Result<(), StitchCoreError> {
        calibration.validate()?;
        let current = self.calibration();
        if current == &calibration {
            return Ok(());
        }
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
        // Compare the complete document except for the one tracking-neutral field.
        // Newly added fields conservatively require invalidation too.
        let mut tracking_calibration = calibration.clone();
        if let (Some(old), Some(new)) = (
            current.topology.l_shape(),
            tracking_calibration.topology.l_shape_mut(),
        ) {
            new.blend_width = old.blend_width;
        }
        let tracking_unchanged = current == &tracking_calibration;
        if !tracking_unchanged
            && (self.ball_tracker.is_some()
                || self.player_tracker.is_some()
                || self.panner.is_some())
        {
            return Err(StitchCoreError::CalibrationRequiresRebuild(
                "camera control retains state from the current calibration",
            ));
        }
        let coverage = if tracking_unchanged {
            None
        } else {
            Some(calibration.topology.projection().coverage(&calibration))
        };
        self.executor.apply_calibration(calibration);
        if let Some(coverage) = coverage {
            self.coverage = Some(coverage);
            self.last_detections.clear();
        }
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
            if step % 2 == 0 {
                core.edit_calibration(|cal| *cal = draft).unwrap();
            } else {
                core.update_calibration(draft).unwrap();
            }
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
    fn rejected_closure_runs_once_without_publishing_its_draft() {
        let mut core = cpu(calib(WIDTH, HEIGHT));
        let accepted = core.calibration().clone();
        let pixels = render(&mut core);
        core.last_detections.push(detection());
        let mut calls = 0;
        assert!(
            core.edit_calibration(|cal| {
                calls += 1;
                cal.lenses[0].fx = 0.0;
            })
            .is_err()
        );
        assert_eq!(calls, 1);
        assert_eq!(core.calibration(), &accepted);
        assert_eq!(render(&mut core), pixels);
        assert_eq!(core.last_detections.len(), 1);
    }

    fn blend_edits_preserve_tracking(make: impl Fn(Calibration) -> StitchCore) {
        for slot in 0..3 {
            let mut core = make(calib(WIDTH, HEIGHT));
            match slot {
                0 => core.set_ball_tracker(Box::new(NoopControl)),
                1 => core.set_player_tracker(Box::new(NoopControl)),
                _ => core.set_panner(Box::new(NoopControl)),
            }
            core.last_detections.push(detection());
            let coverage = core.coverage().unwrap().yaw_range();
            let initial_pixels = render(&mut core);
            let mut changed_pixels = false;
            core.edit_calibration(|_| {}).unwrap();
            for blend_width in [0.0, 0.15, 0.5] {
                core.edit_calibration(|cal| {
                    cal.topology.l_shape_mut().unwrap().blend_width = blend_width;
                })
                .unwrap();
                assert_eq!(core.last_detections.len(), 1);
                assert_eq!(core.coverage().unwrap().yaw_range(), coverage);
                assert_eq!(core.ball_tracker.is_some(), slot == 0);
                assert_eq!(core.player_tracker.is_some(), slot == 1);
                assert_eq!(core.panner.is_some(), slot == 2);
                let mut fresh = make(core.calibration().clone());
                let pixels = render(&mut core);
                changed_pixels |= pixels != initial_pixels;
                assert_eq!(pixels, render(&mut fresh));
            }
            assert!(changed_pixels, "fixture must exercise the blend seam");
        }
    }

    #[test]
    fn cpu_blend_edits_preserve_tracking_and_match_fresh_render() {
        blend_edits_preserve_tracking(cpu);
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
        blend_edits_preserve_tracking(make);
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
            let accepted = core.calibration().clone();
            let edits: [fn(&mut Calibration); 6] = [
                |cal| cal.framing.tilt = 0.2,
                |cal| cal.lenses[0].fx += 1.0,
                |cal| cal.lenses[0].correction = 0.5,
                |cal| cal.topology.l_shape_mut().unwrap().intersect = 0.3,
                |cal| cal.sync_offset = 1,
                |cal| {
                    cal.field_roi = Some(crate::calibration::FieldRoi {
                        left: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                        right: vec![],
                    })
                },
            ];
            for edit in edits {
                assert!(matches!(
                    core.edit_calibration(|cal| {
                        cal.topology.l_shape_mut().unwrap().blend_width = 0.2;
                        edit(cal);
                    }),
                    Err(StitchCoreError::CalibrationRequiresRebuild(_))
                ));
                assert_eq!(core.calibration(), &accepted);
            }
        }
    }

    #[test]
    fn cylinder_noop_preserves_tracking_but_geometry_edits_require_reconstruction() {
        let mut core = cpu(Calibration::new(
            vec![Lens::flat(WIDTH, HEIGHT)],
            Cylinder::default(),
            calib(WIDTH, HEIGHT).framing,
        ));
        core.set_panner(Box::new(NoopControl));
        core.last_detections.push(detection());
        let accepted = core.calibration().clone();
        core.update_calibration(accepted.clone()).unwrap();
        assert_eq!(core.last_detections.len(), 1);
        assert!(matches!(
            core.edit_calibration(|cal| {
                let Topology::Cylinder(cylinder) = &mut cal.topology else {
                    unreachable!()
                };
                cylinder.sweep_deg = 240.0;
            }),
            Err(StitchCoreError::CalibrationRequiresRebuild(_))
        ));
        assert_eq!(core.calibration(), &accepted);
    }
}
