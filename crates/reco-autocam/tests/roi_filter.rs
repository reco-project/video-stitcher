//! Public post-inference ROI policy and synchronous decorator agreement.

use reco_autocam::{RoiAnchor, RoiFilter, RoiFilteredDetector};
use reco_core::calibration::FieldRoi;
use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
use reco_core::geometry::CameraIndex;

struct Completed {
    result: Result<Vec<Detection>, DetectorError>,
    labels: Vec<String>,
}

impl Detector for Completed {
    fn name(&self) -> &'static str {
        "completed-fixture"
    }

    fn detect(
        &mut self,
        _: CameraIndex,
        _: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        self.result.clone()
    }

    fn class_names(&self) -> Option<&[String]> {
        Some(&self.labels)
    }
}

fn policy(default: RoiAnchor) -> RoiFilter {
    RoiFilter::new(FieldRoi {
        left: vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        right: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    })
    .with_default_anchor(default)
    .with_class_anchor(1, RoiAnchor::Bottom)
    .with_class_anchor(2, RoiAnchor::Top)
}

fn samples() -> Vec<Detection> {
    (0..4)
        .flat_map(|camera| {
            (0..4).flat_map(move |class_id| {
                [0.1, 0.3, 0.5, 0.7, 0.9]
                    .into_iter()
                    .map(move |center_y| Detection {
                        camera,
                        class_id,
                        confidence: 0.73,
                        center_x: 0.5,
                        center_y,
                        width: 0.1,
                        height: 0.4,
                    })
            })
        })
        .collect()
}

fn fields(d: &Detection) -> (usize, u16, [u32; 5]) {
    (
        d.camera,
        d.class_id,
        [d.confidence, d.center_x, d.center_y, d.width, d.height].map(f32::to_bits),
    )
}

#[test]
fn completed_batches_and_detector_share_policy_without_altering_metadata() {
    for anchor in [RoiAnchor::Center, RoiAnchor::Bottom, RoiAnchor::Top] {
        let original = samples();
        let mut completed = original.clone();
        let allocation = completed.as_ptr();
        let capacity = completed.capacity();
        policy(anchor).retain(&mut completed);
        assert_eq!(completed.as_ptr(), allocation);
        assert_eq!(completed.capacity(), capacity);
        assert!(completed.len() < original.len());
        assert_eq!(completed.iter().filter(|d| d.camera >= 2).count(), 40);

        let mut decorated = RoiFilteredDetector::new(
            Box::new(Completed {
                result: Ok(original.clone()),
                labels: vec!["ball".into(), "person".into()],
            }),
            policy(anchor),
        );
        assert_eq!(decorated.name(), "completed-fixture");
        assert_eq!(decorated.class_names().unwrap(), ["ball", "person"]);
        let frame = DetectorFrame::Rgba {
            data: &[0; 4],
            width: 1,
            height: 1,
        };
        let sync = decorated.detect(0, &frame).unwrap();
        assert_eq!(
            sync.iter().map(fields).collect::<Vec<_>>(),
            completed.iter().map(fields).collect::<Vec<_>>()
        );
        let mut source = original.iter();
        for item in &completed {
            assert!(source.any(|candidate| fields(candidate) == fields(item)));
        }
        let once = completed.iter().map(fields).collect::<Vec<_>>();
        policy(anchor).retain(&mut completed);
        assert_eq!(completed.iter().map(fields).collect::<Vec<_>>(), once);
    }
}

#[test]
fn anchors_apply_to_the_same_box_and_class_overrides_win() {
    let mut boxes: Vec<_> = (0..3)
        .map(|class_id| Detection {
            camera: 0,
            class_id,
            confidence: 0.9,
            center_x: 0.5,
            center_y: 0.7,
            width: 0.1,
            height: 0.4,
        })
        .collect();
    policy(RoiAnchor::Center).retain(&mut boxes);
    assert_eq!(boxes.iter().map(|d| d.class_id).collect::<Vec<_>>(), [0, 2]);

    let mut boxes = samples();
    let original = boxes.iter().map(fields).collect::<Vec<_>>();
    RoiFilter::new(FieldRoi {
        left: vec![],
        right: vec![[0.0, 0.0], [1.0, 1.0]],
    })
    .retain(&mut boxes);
    assert_eq!(boxes.iter().map(fields).collect::<Vec<_>>(), original);
    let mut empty = Vec::new();
    policy(RoiAnchor::Top).retain(&mut empty);
    assert!(empty.is_empty());
}

#[test]
fn decorator_preserves_inference_failures() {
    for error in [
        DetectorError::InferenceFailed("fixture".into()),
        DetectorError::Timeout {
            after: std::time::Duration::from_millis(40),
        },
        DetectorError::UnsupportedFrameKind,
        DetectorError::Transport("disconnected".into()),
        DetectorError::Canceled,
    ] {
        let expected = error.to_string();
        let mut detector = RoiFilteredDetector::new(
            Box::new(Completed {
                result: Err(error),
                labels: Vec::new(),
            }),
            policy(RoiAnchor::Bottom),
        );
        let frame = DetectorFrame::Rgba {
            data: &[0; 4],
            width: 1,
            height: 1,
        };
        assert_eq!(
            detector.detect(0, &frame).unwrap_err().to_string(),
            expected
        );
    }
}
