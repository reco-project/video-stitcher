//! Selected detector metadata drives one tracking and ROI policy.

use super::*;
use reco_core::detect::{
    DetectionTarget,
    detector::{Detection, Detector, DetectorError, DetectorFrame},
    panner::Panner,
    tracker::Tracker,
};

#[derive(Default)]
struct Target {
    detector: Option<Box<dyn Detector>>,
    filter: Option<Box<dyn reco_core::detect::DetectionFilter>>,
    interval: u64,
    ball: Option<u16>,
    players: Option<u16>,
    panner: Option<Box<dyn Panner>>,
}

impl DetectionTarget for Target {
    fn set_detector(&mut self, detector: Box<dyn Detector>) {
        self.detector = Some(detector);
    }
    fn set_detection_filter(
        &mut self,
        filter: Option<Box<dyn reco_core::detect::DetectionFilter>>,
    ) {
        self.filter = filter;
    }
    fn set_detection_interval(&mut self, interval: u64) {
        self.interval = interval;
    }
    fn set_ball_tracker(&mut self, tracker: Box<dyn Tracker>) {
        self.ball = Some(tracker.class_id());
    }
    fn set_player_tracker(&mut self, tracker: Box<dyn Tracker>) {
        self.players = Some(tracker.class_id());
    }
    fn set_panner(&mut self, panner: Box<dyn Panner>) {
        self.panner = Some(panner);
    }
    fn source_sizes(&self) -> &[(u32, u32)] {
        &[(16, 16)]
    }
    fn gpu(&self) -> Option<&reco_core::gpu::GpuContext> {
        None
    }
}

struct Labeled(Vec<String>);

impl Detector for Labeled {
    fn name(&self) -> &'static str {
        "labeled-fixture"
    }
    fn class_names(&self) -> Option<&[String]> {
        Some(&self.0)
    }
    fn detect(
        &mut self,
        camera: usize,
        _: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        Ok([0, 1, 32]
            .into_iter()
            .map(|class_id| Detection {
                camera,
                class_id,
                confidence: 0.9,
                center_x: 0.5,
                center_y: 0.7,
                width: 0.1,
                height: 0.4,
            })
            .collect())
    }
}

#[test]
fn loaded_detector_drives_tracking_and_roi_without_another_model_load() {
    for (names, ball, person, rejected) in [
        (vec!["ball", "person"], 0, Some(1), Some(1)),
        (vec!["person", "sports ball"], 1, Some(0), Some(0)),
        (vec!["ball"], 0, None, None),
        (vec!["other"], 32, None, None),
        (vec![], 32, None, Some(0)),
    ] {
        for mode in [TrackingMode::Field, TrackingMode::Ball] {
            let mut target = Target {
                interval: 7,
                ..Default::default()
            };
            let names: Vec<_> = names.iter().map(|name| (*name).into()).collect();
            let config = AutocamConfig::new("nonexistent-model").with_tracking_mode(mode);
            let mut cal = reco_core::calibration::Calibration::new(
                vec![reco_core::calibration::Lens::flat(1920, 1080)],
                reco_core::projection::Cylinder::default(),
                reco_core::calibration::Framing {
                    axis_offset: 0.0,
                    tilt: 0.0,
                    roll: 0.0,
                },
            );
            cal.field_roi = Some(reco_core::calibration::FieldRoi {
                corners: [[0.4, 0.2], [0.6, 0.2], [0.6, 0.8], [0.4, 0.8]].map(|position| {
                    reco_core::projection::SourcePoint {
                        camera: 0,
                        position,
                    }
                }),
            });
            attach_control(&mut target, &config, 30.0, Box::new(Labeled(names.clone())));
            assert_eq!(target.interval, 1);
            assert_eq!(target.ball, Some(ball));
            assert_eq!(
                target.players,
                if mode == TrackingMode::Field {
                    person
                } else {
                    None
                }
            );
            assert!(target.panner.is_some());
            let detector = target.detector.as_mut().unwrap();
            assert_eq!(detector.name(), "labeled-fixture");
            assert_eq!(detector.class_names(), Some(names.as_slice()));
            let mut boxes = detector
                .detect(
                    0,
                    &DetectorFrame::Rgba {
                        data: &[0; 4],
                        width: 1,
                        height: 1,
                    },
                )
                .unwrap();
            target
                .filter
                .as_mut()
                .unwrap()
                .retain(&cal, &mut boxes)
                .unwrap();
            assert_eq!(
                boxes.iter().map(|d| d.class_id).collect::<Vec<_>>(),
                [0, 1, 32]
                    .into_iter()
                    .filter(|class| Some(*class) != rejected)
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn engine_sidecar_labels_are_consumed_through_the_detector() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("detector.labels");
    std::fs::write(&path, "ball\nperson\n").unwrap();
    let detector = Labeled(reco_detect::read_labels_file(&path));
    let mut target = Target::default();
    attach_control(
        &mut target,
        &AutocamConfig::new("not-reopened.engine"),
        30.0,
        Box::new(detector),
    );
    assert_eq!(target.ball, Some(0));
    assert_eq!(target.players, Some(1));
}
