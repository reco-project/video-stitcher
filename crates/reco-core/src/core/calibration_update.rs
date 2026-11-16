//! Acceptance boundary for live calibration edits.

use super::{StitchCore, StitchCoreError};
use crate::calibration::Calibration;
use crate::projection::coverage::controller::Controller;

/// Compare geometry while ignoring blending. Detection policies may depend on blending independently.
pub(super) fn same_geometry(current: &Calibration, updated: &Calibration) -> bool {
    if current == updated {
        return true;
    }
    // Newly added fields conservatively require invalidation too.
    let mut geometry = updated.clone();
    if let (Some(old), Some(new)) = (
        current.projection.l_shape(),
        geometry.projection.l_shape_mut(),
    ) {
        new.blend_width = old.blend_width;
    }
    current == &geometry
}

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

    /// Apply an edited calibration without changing input layout or projection kind.
    ///
    /// Invalid documents and changed image dimensions or projection kind are rejected without changing the accepted state. With a tracker or panner attached, only no-op edits and, without a detection filter, L-shape blend-width edits are accepted. Filters can use blending to select camera ownership. These preserve coverage and cached detections; other accepted edits refresh coverage and clear detections. Backend geometry is refreshed on every changed document. See this module's CPU/GPU regression tests.
    pub fn update_calibration(&mut self, calibration: Calibration) -> Result<(), StitchCoreError> {
        let tracking_unchanged = self.validate_calibration_update(&calibration)?;
        if self.calibration() == &calibration {
            return Ok(());
        }
        let rebuilt = if tracking_unchanged {
            None
        } else {
            self.coverage_motion
                .as_ref()
                .map(|controller| controller.rebuild(&calibration, self.executor.viewport_size()))
                .transpose()
                .map_err(StitchCoreError::Config)?
        };
        self.accept_calibration(calibration, tracking_unchanged, rebuilt);
        Ok(())
    }

    /// Apply validated preview geometry immediately without waiting for coverage.
    ///
    /// Geometry changes invalidate coverage; use [`Self::prepare_coverage_preview`] to rebuild asynchronously. A valid document can have no usable coverage: that preparation error does not roll back the edited document. Invalid documents and edits incompatible with attached tracking are rejected without mutation. Offline export and automatic tracking should use [`Self::update_calibration`].
    pub fn update_calibration_preview(
        &mut self,
        calibration: Calibration,
    ) -> Result<(), StitchCoreError> {
        let tracking_unchanged = self.validate_calibration_update(&calibration)?;
        if self.calibration() == &calibration {
            return Ok(());
        }
        if !tracking_unchanged {
            self.invalidate_preview_coverage();
        }
        self.accept_calibration(calibration, tracking_unchanged, None);
        Ok(())
    }

    fn validate_calibration_update(
        &self,
        calibration: &Calibration,
    ) -> Result<bool, StitchCoreError> {
        calibration.validate()?;
        let current = self.calibration();
        if current == calibration {
            return Ok(true);
        }
        if std::mem::discriminant(&current.projection)
            != std::mem::discriminant(&calibration.projection)
            || current.lenses.len() != calibration.lenses.len()
            || current
                .lenses
                .iter()
                .zip(&calibration.lenses)
                .any(|(old, new)| (old.width, old.height) != (new.width, new.height))
        {
            return Err(StitchCoreError::CalibrationRequiresRebuild(
                "projection kind or calibration image dimensions changed",
            ));
        }
        let tracking_unchanged = same_geometry(current, calibration)
            && (self.detection_filter.is_none() || current == calibration);
        if !tracking_unchanged
            && (self.ball_tracker.is_some()
                || self.player_tracker.is_some()
                || self.panner.is_some())
        {
            return Err(StitchCoreError::CalibrationRequiresRebuild(
                "camera control retains state from the current calibration",
            ));
        }
        Ok(tracking_unchanged)
    }

    fn accept_calibration(
        &mut self,
        calibration: Calibration,
        tracking_unchanged: bool,
        rebuilt: Option<Controller>,
    ) {
        self.executor.apply_calibration(calibration);
        if let Some(controller) = rebuilt {
            self.coverage_motion = Some(controller);
        }
        if !tracking_unchanged {
            self.clear_detection_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{Lens, Projection};
    use crate::detect::mapped_detection::MappedDetection;
    use crate::detect::panner::{PanContext, Panner};
    use crate::detect::tracker::{TrackedEntity, Tracker, WorldState};
    use crate::geometry::RenderPose;
    use crate::projection::Cylinder;
    use crate::render::planes::Nv12Planes;
    use crate::render::viewport::ViewportSize;
    use crate::stitch::test_support::{calib, nv12};
    use crate::stitch::{CpuExecutor, Executor, StitchExecutor};

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 36;

    #[test]
    fn preview_edits_render_immediately_and_publish_only_current_coverage() {
        let mut core = cpu(calib(WIDTH, HEIGHT));
        core.prepare_coverage().unwrap();
        let original = core.calibration().clone();
        let original_pixels = render(&mut core);
        let mut first = original.clone();
        first.framing.tilt += 0.03;
        core.update_calibration_preview(first.clone()).unwrap();
        assert_eq!(core.calibration(), &first);
        assert_eq!(core.coverage_max_fov_degrees(), None);
        assert_ne!(render(&mut core), original_pixels);
        assert!(core.prepare_coverage_preview().unwrap().is_none());

        let mut latest = original.clone();
        latest.framing.roll -= 0.04;
        latest.lenses[0].correction = 0.8;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        core.update_calibration_preview(latest.clone()).unwrap();
        while core.prepare_coverage_preview().unwrap().is_none() {
            assert!(std::time::Instant::now() < deadline);
            assert_eq!(core.calibration(), &latest);
            assert_eq!(core.coverage_max_fov_degrees(), None);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(core.calibration(), &latest);
        let mut fresh = cpu(latest.clone());
        assert_eq!(
            core.coverage_max_fov_degrees(),
            Some(fresh.prepare_coverage().unwrap())
        );
        assert_eq!(render(&mut core), render(&mut fresh));
        let accepted_pixels = render(&mut core);
        let mut invalid = latest;
        invalid.lenses[0].width = 0;
        assert!(core.update_calibration_preview(invalid).is_err());
        assert_eq!(render(&mut core), accepted_pixels);
    }

    #[test]
    fn preview_edit_rejects_changes_with_attached_camera_control() {
        let mut core = cpu(calib(WIDTH, HEIGHT));
        core.prepare_coverage().unwrap();
        let original = core.calibration().clone();
        let mut draft = original.clone();
        draft.framing.roll += 0.05;
        core.panner = Some(Box::new(NoopControl));
        assert!(core.update_calibration_preview(draft).is_err());
        assert_eq!(core.calibration(), &original);
    }

    fn cpu(calibration: Calibration) -> StitchCore {
        let executor = {
            let source_sizes = vec![(WIDTH, HEIGHT); calibration.projection.camera_count()];
            CpuExecutor::new(
                calibration,
                ViewportSize {
                    width: WIDTH,
                    height: HEIGHT,
                },
                source_sizes,
                false,
            )
        }
        .unwrap();
        StitchCore::new(Executor::Cpu(Box::new(executor))).unwrap()
    }

    fn render(core: &mut StitchCore) -> Vec<u8> {
        let (y, uv) = nv12(WIDTH, HEIGHT, 20);
        let planes = vec![Nv12Planes { y: &y, uv: &uv }; core.camera_count()];
        core.executor
            .stitch(
                &planes,
                RenderPose {
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
            position: Some(crate::geometry::YawPitch {
                yaw: 0.0,
                pitch: 0.0,
            }),
        }
    }

    fn rejection_preserves_state(make: impl Fn(Calibration) -> StitchCore) {
        let calibration = calib(WIDTH, HEIGHT);
        let mut core = make(calibration.clone());
        core.last_detections.push(detection());
        let accepted = serde_json::to_value(core.calibration()).unwrap();
        let pixels = render(&mut core);
        let coverage = core.prepare_coverage().unwrap();
        let mut drafts = Vec::new();
        let mut draft = calibration.clone();
        draft.schema_version = u32::MAX;
        drafts.push(draft);
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
            assert_eq!(core.prepare_coverage().unwrap(), coverage);
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
        // Exercise repeated edits, including correction and projection-specific geometry.
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
                _ => match &mut draft.projection {
                    Projection::LShape(layout) => {
                        layout.intersect = 0.45;
                        layout.blend_width = 0.1;
                        draft.framing.axis_offset = 0.3;
                    }
                    Projection::Cylinder(cylinder) => cylinder.arc_deg = 240.0,
                    Projection::Panorama(panorama) => panorama.arc_deg = 240.0,
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
                core.prepare_coverage().unwrap(),
                fresh.prepare_coverage().unwrap()
            );
            for yaw in [-3.0, 0.0, 3.0] {
                let pose = crate::geometry::WorldPose {
                    yaw,
                    pitch: 1.0,
                    fov_degrees: 60.0,
                };
                let requested = core.calibration().world_to_render_pose(pose);
                let actual = core.constrain_render_pose(requested, requested).unwrap();
                let expected = fresh.constrain_render_pose(requested, requested).unwrap();
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
            let coverage = core.prepare_coverage().unwrap();
            let initial_pixels = render(&mut core);
            let mut changed_pixels = false;
            core.edit_calibration(|_| {}).unwrap();
            for blend_width in [0.0, 0.15, 0.5] {
                core.edit_calibration(|cal| {
                    cal.projection.l_shape_mut().unwrap().blend_width = blend_width;
                })
                .unwrap();
                assert_eq!(core.last_detections.len(), 1);
                assert_eq!(core.prepare_coverage().unwrap(), coverage);
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
        for camera in 0..2 {
            let mut mixed = calib(WIDTH, HEIGHT);
            mixed.lenses[camera] = mixed.lenses[camera].rescaled(WIDTH, HEIGHT * 2).unwrap();
            accepted_matches_fresh(cpu, mixed);
        }
        accepted_matches_fresh(
            cpu,
            Calibration::new(
                vec![Lens::flat(WIDTH, HEIGHT)],
                Cylinder {
                    radius: 32.0,
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
        let make = |calibration: Calibration| {
            let executor = GpuExecutor::new(
                gpu.clone(),
                GpuExecutorConfig {
                    viewport_size: ViewportSize {
                        width: WIDTH,
                        height: HEIGHT,
                    },
                    ..{
                        let source_sizes =
                            vec![(WIDTH, HEIGHT); calibration.projection.camera_count()];
                        GpuExecutorConfig::new(calibration, source_sizes, InputFormat::Nv12)
                    }
                },
            )
            .unwrap();
            StitchCore::new(Executor::Gpu(Box::new(executor))).unwrap()
        };
        rejection_preserves_state(make);
        accepted_matches_fresh(make, calib(WIDTH, HEIGHT));
        for camera in 0..2 {
            let mut mixed = calib(WIDTH, HEIGHT);
            mixed.lenses[camera] = mixed.lenses[camera].rescaled(WIDTH, HEIGHT * 2).unwrap();
            accepted_matches_fresh(make, mixed);
        }
        accepted_matches_fresh(
            make,
            Calibration::new(
                vec![Lens::flat(WIDTH, HEIGHT)],
                crate::projection::Cylinder {
                    radius: 64.0,
                    ..Default::default()
                },
                crate::calibration::Framing {
                    axis_offset: 0.0,
                    tilt: 0.0,
                    roll: 0.0,
                },
            ),
        );
        blend_edits_preserve_tracking(make);
    }

    struct NoopControl;

    impl Tracker for NoopControl {
        fn update(
            &mut self,
            _: &[MappedDetection],
            _: &[crate::geometry::CameraIndex],
            _: f64,
        ) -> Vec<TrackedEntity> {
            vec![]
        }
        fn class_id(&self) -> u16 {
            0
        }
    }

    impl Panner for NoopControl {
        fn decide(&mut self, _: &WorldState, _: &PanContext<'_>) -> crate::geometry::WorldPose {
            crate::geometry::WorldPose::default()
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
                |cal| cal.projection.l_shape_mut().unwrap().intersect = 0.3,
                |cal| cal.sync_offset = 1,
                |cal| {
                    cal.field_roi = Some(crate::calibration::FieldRoi {
                        corners: [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]].map(|position| {
                            crate::projection::SourcePoint {
                                camera: 0,
                                position,
                            }
                        }),
                    })
                },
            ];
            for edit in edits {
                assert!(matches!(
                    core.edit_calibration(|cal| {
                        cal.projection.l_shape_mut().unwrap().blend_width = 0.2;
                        edit(cal);
                    }),
                    Err(StitchCoreError::CalibrationRequiresRebuild(_))
                ));
                assert_eq!(core.calibration(), &accepted);
            }
        }
    }

    #[test]
    fn a_blend_sensitive_policy_prevents_stale_tracking_state() {
        struct Filter;
        impl crate::detect::DetectionFilter for Filter {
            fn retain(
                &mut self,
                _: &Calibration,
                _: &mut Vec<crate::detect::detector::Detection>,
            ) -> Result<(), crate::calibration::CalibrationError> {
                Ok(())
            }
        }
        let mut core = cpu(calib(WIDTH, HEIGHT));
        core.set_detection_filter(Some(Box::new(Filter)));
        core.set_panner(Box::new(NoopControl));
        let original = core.calibration().clone();
        assert!(matches!(
            core.edit_calibration(|c| c.projection.l_shape_mut().unwrap().blend_width += 0.1),
            Err(StitchCoreError::CalibrationRequiresRebuild(_))
        ));
        assert_eq!(core.calibration(), &original);
        core.set_detection_filter(None);
        core.edit_calibration(|c| c.projection.l_shape_mut().unwrap().blend_width += 0.1)
            .unwrap();
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
                let Projection::Cylinder(cylinder) = &mut cal.projection else {
                    unreachable!()
                };
                cylinder.arc_deg = 240.0;
            }),
            Err(StitchCoreError::CalibrationRequiresRebuild(_))
        ));
        assert_eq!(core.calibration(), &accepted);
    }
}
