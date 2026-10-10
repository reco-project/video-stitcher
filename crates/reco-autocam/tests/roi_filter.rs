//! Source-attached field policy for completed detections, independent of detector backend.

use reco_autocam::{RoiAnchor, RoiFilter};
use reco_core::calibration::{Calibration, FieldRoi, Framing, Lens};
use reco_core::detect::detector::Detection;
use reco_core::projection::{Cylinder, SourcePoint};

fn calibration() -> Calibration {
    let mut cal = Calibration::new(
        vec![Lens::flat(1920, 1080)],
        Cylinder::default(),
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    cal.field_roi = Some(FieldRoi {
        corners: [[0.4, 0.2], [0.6, 0.2], [0.6, 0.8], [0.4, 0.8]].map(|position| SourcePoint {
            camera: 0,
            position,
        }),
    });
    cal.validate().unwrap();
    cal
}

fn detection(class_id: u16, center_y: f32) -> Detection {
    Detection {
        camera: 0,
        class_id,
        confidence: 0.73,
        center_x: 0.5,
        center_y,
        width: 0.1,
        height: 0.4,
    }
}

fn policy() -> RoiFilter {
    RoiFilter::default()
        .with_class_anchor(1, RoiAnchor::Bottom)
        .with_class_anchor(2, RoiAnchor::Top)
}

fn fields(d: &Detection) -> (usize, u16, [u32; 5]) {
    (
        d.camera,
        d.class_id,
        [d.confidence, d.center_x, d.center_y, d.width, d.height].map(f32::to_bits),
    )
}

#[test]
fn anchors_preserve_batch_allocation_order_and_metadata() {
    let cal = calibration();
    let original: Vec<_> = (0..3).map(|class| detection(class, 0.7)).collect();
    let mut boxes = original.clone();
    let allocation = boxes.as_ptr();
    let capacity = boxes.capacity();
    let mut policy = policy();
    policy.retain(&cal, &mut boxes).unwrap();
    assert_eq!(boxes.as_ptr(), allocation);
    assert_eq!(boxes.capacity(), capacity);
    assert_eq!(
        boxes.iter().map(fields).collect::<Vec<_>>(),
        [fields(&original[0]), fields(&original[2])]
    );
    policy.retain(&cal, &mut boxes).unwrap();
    assert_eq!(boxes.len(), 2);
    let mut empty = vec![];
    policy.retain(&cal, &mut empty).unwrap();
    assert!(empty.is_empty());
}

#[test]
fn accepted_document_changes_replace_cached_geometry_and_none_disables_policy() {
    let mut cal = calibration();
    let mut policy = policy();
    let mut boxes = vec![detection(0, 0.7)];
    policy.retain(&cal, &mut boxes).unwrap();
    assert_eq!(boxes.len(), 1);
    for corner in &mut cal.field_roi.as_mut().unwrap().corners {
        corner.position[1] *= 0.5;
    }
    policy.retain(&cal, &mut boxes).unwrap();
    assert!(boxes.is_empty());
    cal.field_roi = None;
    boxes.push(detection(0, 0.7));
    policy.retain(&cal, &mut boxes).unwrap();
    assert_eq!(boxes.len(), 1);
}

#[test]
fn malformed_regions_error_and_unsupported_observations_are_rejected() {
    let mut cal = calibration();
    let mut policy = policy();
    let mut boxes = vec![detection(0, 0.5); 4];
    boxes[1].camera = 99;
    boxes[2].center_x = f32::NAN;
    boxes[3].center_x = -0.1;
    policy.retain(&cal, &mut boxes).unwrap();
    assert_eq!(boxes.len(), 1);
    cal.field_roi.as_mut().unwrap().corners.swap(1, 2);
    assert!(policy.retain(&cal, &mut boxes).is_err());
    assert_eq!(
        boxes.len(),
        1,
        "failure does not partially mutate the batch"
    );
    cal = calibration();
    policy.retain(&cal, &mut boxes).unwrap();
    assert_eq!(boxes.len(), 1);
}
