//! Real NCNN tensor and inference regressions, using tiny generated models.

use super::*;

fn model(param: &str, weights: &[f32]) -> tempfile::TempDir {
    let model = tempfile::tempdir().unwrap();
    std::fs::write(model.path().join("model.ncnn.param"), param).unwrap();
    let bytes: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(model.path().join("model.ncnn.bin"), bytes).unwrap();
    model
}

fn detector() -> NcnnYoloDetector {
    let model = model(
        "7767517\n2 2\nInput input 0 1 in0\nMemoryData output 0 1 out0 0=1 1=5\n",
        &[4.0, 4.0, 2.0, 2.0, 0.9],
    );
    NcnnYoloDetector::new(model.path(), 8, 0.25, Vec::new()).unwrap()
}

#[test]
fn preprocess_preserves_aspect_and_integer_padding() {
    let mut detector = detector();
    for (width, height, content, offset) in [
        (6, 4, (8, 5), (0, 1)),
        (4, 6, (5, 8), (1, 0)),
        (8, 8, (8, 8), (0, 0)),
    ] {
        for planar in [false, true] {
            let frame = YuvView {
                y: &[100; 64],
                chroma: if planar {
                    ChromaPlanes::Yuv420p {
                        u: &[148; 16],
                        v: &[168; 16],
                    }
                } else {
                    ChromaPlanes::Nv12 {
                        uv: &[148, 168].repeat(16),
                    }
                },
                width,
                height,
            };
            let (mat, _) = detector.preprocess(&frame).unwrap();
            // 8x8 float planes are aligned, so channel stride equals pixel count.
            let tensor = unsafe { std::slice::from_raw_parts(ncnn_mat_get_data(mat.0), 3 * 64) };
            for (channel, rgb) in [162.0_f32, 77.0, 137.0].into_iter().enumerate() {
                for y in 0..8 {
                    for x in 0..8 {
                        let inside = x >= offset.0
                            && x < offset.0 + content.0
                            && y >= offset.1
                            && y < offset.1 + content.1;
                        let expected = if inside { rgb } else { 114.0 } / 255.0;
                        assert!(
                            (tensor[channel * 64 + y * 8 + x] - expected).abs() < 1e-6,
                            "{width}x{height}, planar={planar}, channel={channel}, ({x},{y})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn repeated_inference() {
    let mut detector = detector();
    let frame = DetectorFrame::Yuv(YuvView {
        y: &[200; 24],
        chroma: ChromaPlanes::Nv12 { uv: &[128; 12] },
        width: 6,
        height: 4,
    });
    for camera in 0..32 {
        let detections = detector.detect(camera, &frame).unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].camera, camera);
    }
}

#[test]
fn alternating_cameras_recover_coordinates_and_reuse_rgb_storage() {
    let mut detector = detector();
    let mut allocation = None;
    for _ in 0..8 {
        for (camera, width, height, center, extent) in [
            (2, 8, 8, (0.5, 0.5), (0.25, 0.25)),
            (0, 6, 4, (0.5, 0.5625), (0.25, 0.375)),
            (1, 4, 6, (0.5625, 0.5), (0.375, 0.25)),
        ] {
            let frame = DetectorFrame::Yuv(YuvView {
                y: &[200; 64],
                chroma: ChromaPlanes::Nv12 { uv: &[128; 32] },
                width,
                height,
            });
            let detections = detector.detect(camera, &frame).unwrap();
            assert_eq!(detections.len(), 1);
            let d = detections[0];
            assert_eq!(d.camera, camera);
            for (actual, expected) in [
                (d.center_x, center.0),
                (d.center_y, center.1),
                (d.width, extent.0),
                (d.height, extent.1),
            ] {
                assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
            }
            let current = (detector.rgb.as_ptr(), detector.rgb.capacity());
            assert_eq!(*allocation.get_or_insert(current), current);
        }
    }
}

#[test]
fn invalid_frames_are_errors_and_do_not_poison_the_detector() {
    let mut detector = detector();
    for (width, height, y, chroma) in [
        (0, 4, &[][..], ChromaPlanes::Nv12 { uv: &[] }),
        (3, 4, &[0; 12][..], ChromaPlanes::Nv12 { uv: &[0; 6] }),
        (4, 3, &[0; 12][..], ChromaPlanes::Nv12 { uv: &[0; 6] }),
        (6, 4, &[0; 23][..], ChromaPlanes::Nv12 { uv: &[0; 12] }),
        (6, 4, &[0; 24][..], ChromaPlanes::Nv12 { uv: &[0; 11] }),
        (
            6,
            4,
            &[0; 24][..],
            ChromaPlanes::Yuv420p {
                u: &[0; 5],
                v: &[0; 6],
            },
        ),
        (
            6,
            4,
            &[0; 24][..],
            ChromaPlanes::Yuv420p {
                u: &[0; 6],
                v: &[0; 5],
            },
        ),
        (
            u32::MAX - 1,
            u32::MAX - 1,
            &[][..],
            ChromaPlanes::Nv12 { uv: &[] },
        ),
    ] {
        assert!(
            detector
                .detect(
                    0,
                    &DetectorFrame::Yuv(YuvView {
                        y,
                        chroma,
                        width,
                        height
                    })
                )
                .is_err()
        );
    }
    assert!(
        detector
            .detect(
                0,
                &DetectorFrame::Yuv(YuvView {
                    y: &[100; 24],
                    chroma: ChromaPlanes::Nv12 { uv: &[128; 12] },
                    width: 6,
                    height: 4,
                })
            )
            .is_ok()
    );
}

#[test]
fn invalid_configuration_is_rejected_before_loading() {
    for input in [0, u32::MAX, i32::MAX as u32, 65536] {
        assert!(
            NcnnYoloDetector::new("unused", input, 0.25, vec![])
                .err()
                .unwrap()
                .to_string()
                .contains("input tensor")
        );
    }
    for threshold in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        assert!(
            NcnnYoloDetector::new("unused", 8, threshold, vec![])
                .err()
                .unwrap()
                .to_string()
                .contains("confidence")
        );
    }
}

#[test]
fn inference_error_paths_release_native_resources() {
    for param in [
        "7767517\n2 2\nInput input 0 1 wrong_input\nSplit output 1 1 wrong_input out0\n",
        "7767517\n2 2\nInput input 0 1 in0\nSplit output 1 1 in0 wrong_output\n",
        "7767517\n2 2\nInput input 0 1 in0\nSplit output 1 1 in0 out0\n",
    ] {
        let model = model(param, &[]);
        let mut detector = NcnnYoloDetector::new(model.path(), 8, 0.25, vec![]).unwrap();
        for _ in 0..4 {
            let frame = DetectorFrame::Yuv(YuvView {
                y: &[100; 24],
                chroma: ChromaPlanes::Nv12 { uv: &[128; 12] },
                width: 6,
                height: 4,
            });
            assert!(detector.detect(0, &frame).is_err());
        }
    }
}

#[test]
fn malformed_boxes_are_discarded_and_nms_remains_class_aware() {
    let rows = [
        [4.0, 4.0, 4.0, f32::NAN, 4.0, 4.0, 4.0, 20.0],
        [4.0; 8],
        [2.0, 2.0, 2.0, 2.0, -2.0, 2.0, 2.0, 2.0],
        [2.0; 8],
        [0.9, 0.8, 0.0, 0.9, 0.9, f32::INFINITY, f32::NAN, 0.9],
        [0.0, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let model = model(
        "7767517\n2 2\nInput input 0 1 in0\nMemoryData output 0 1 out0 0=8 1=6\n",
        rows.as_flattened(),
    );
    let mut detector = NcnnYoloDetector::new(model.path(), 8, 0.25, vec![]).unwrap();
    let detections = detector
        .detect(
            1,
            &DetectorFrame::Yuv(YuvView {
                y: &[100; 24],
                chroma: ChromaPlanes::Nv12 { uv: &[128; 12] },
                width: 6,
                height: 4,
            }),
        )
        .unwrap();
    assert_eq!(detections.len(), 2);
    assert_eq!(detections[0].class_id, 0);
    assert_eq!(detections[0].confidence, 0.9);
    assert_eq!(detections[1].class_id, 1);
    assert_eq!(detections[1].confidence, 0.85);
}

#[test]
fn invalid_output_shape_is_reported() {
    let model = model(
        "7767517\n2 2\nInput input 0 1 in0\nMemoryData output 0 1 out0 0=1 1=4\n",
        &[4.0, 4.0, 2.0, 2.0],
    );
    let mut detector = NcnnYoloDetector::new(model.path(), 8, 0.25, vec![]).unwrap();
    let error = detector
        .detect(
            0,
            &DetectorFrame::Yuv(YuvView {
                y: &[100; 24],
                chroma: ChromaPlanes::Nv12 { uv: &[128; 12] },
                width: 6,
                height: 4,
            }),
        )
        .unwrap_err();
    assert!(error.to_string().contains("classes"));
}
