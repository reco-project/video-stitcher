//! Native tensor contents and per-camera coordinate mapping through actual inference.

use super::*;
use reco_core::interop::cuda::cuda_memcpy_htod_2d;

use crate::detectors::tensor_probe::MODEL;

#[test]
fn invalid_source_dimensions_fail_before_model_or_cuda_initialization() {
    for sizes in [
        vec![],
        vec![(0, 2)],
        vec![(2, 0)],
        vec![(3, 2)],
        vec![(2, 2), (2, 3)],
    ] {
        let error = OrtGpuDetector::try_new("nonexistent.onnx", &sizes, 0.1, Vec::new(), false)
            .err()
            .expect("invalid dimensions must fail")
            .to_string();
        assert!(
            error.contains(if sizes.is_empty() {
                "at least one"
            } else {
                "positive and even"
            }),
            "{error}"
        );
    }
}

fn upload(bytes: &[u8]) -> Allocation {
    let buffer = Allocation::new(bytes.len()).unwrap();
    cuda_memcpy_htod_2d(
        buffer.0,
        bytes.len(),
        bytes.as_ptr(),
        bytes.len(),
        bytes.len(),
        1,
    )
    .unwrap();
    buffer
}

#[test]
fn integer_letterbox_placement_round_trips_source_coordinates() {
    for size in [(6, 4), (4, 6), (128, 96), (192, 64)] {
        let input = InputLayout::new(size, 16);
        let x = |value| value * size.0 as f32 * input.scale + input.pad_x as f32;
        let y = |value| value * size.1 as f32 * input.scale + input.pad_y as f32;
        let result = postprocess(
            &[x(0.3), y(0.5), x(0.5), y(0.7), 1.0, 0.0],
            1,
            0,
            0.1,
            input.scale,
            input.pad_x as f32,
            input.pad_y as f32,
            size.0,
            size.1,
        );
        assert_eq!(result.len(), 1);
        assert!((result[0].center_x - 0.4).abs() < 1e-6);
        assert!((result[0].center_y - 0.6).abs() < 1e-6);
    }
}

#[test]
#[ignore = "requires NVIDIA CUDA and a TensorRT/CUDA ONNX Runtime provider"]
fn mixed_sizes_reuse_storage_and_map_detections_after_real_inference() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("tensor-probe.onnx");
    std::fs::write(&model, MODEL).unwrap();
    let sizes = [(8, 4), (4, 8), (8, 8), (6, 4), (4, 6)];
    for is_10bit in [false, true] {
        let mut detector =
            OrtGpuDetector::try_new(&model, &sizes, 0.0, vec!["object".into()], is_10bit)
                .unwrap()
                .expect("GPU provider required");
        let tensor_ptr = detector.tensor_f32.0;
        let scratch_ptr = detector
            .p010_scratch
            .as_ref()
            .map(|(allocation, _)| allocation.0);
        for camera in [0, 1, 2, 3, 4, 3, 2, 1, 0] {
            let (width, height) = sizes[camera];
            let luma = 40 + camera as u8 * 30;
            let y_stride = width as usize + 8;
            let uv_stride = width as usize + 16;
            let y = vec![luma; y_stride * height as usize];
            let uv = vec![128; uv_stride * height as usize / 2];
            let (y, uv, bytes_per_sample) = if is_10bit {
                let y16: Vec<u16> = y.iter().map(|&v| u16::from(v) << 8).collect();
                let uv16: Vec<u16> = uv.iter().map(|&v| u16::from(v) << 8).collect();
                (
                    upload(bytemuck::cast_slice(&y16)),
                    upload(bytemuck::cast_slice(&uv16)),
                    2,
                )
            } else {
                (upload(&y), upload(&uv), 1)
            };
            let mut frame = CudaNv12Frame {
                y_ptr: y.0,
                uv_ptr: uv.0,
                y_pitch: y_stride * bytes_per_sample,
                uv_pitch: uv_stride * bytes_per_sample,
                width,
                height,
                rotation: 0,
                is_10bit,
            };
            let detections = detector
                .detect(camera, &DetectorFrame::CudaNv12(frame))
                .unwrap();
            assert_eq!(detections.len(), 1);
            let detection = &detections[0];
            let scale = 16.0 / width.max(height) as f32;
            let content_w = (width as f32 * scale).round() as u32;
            let content_h = (height as f32 * scale).round() as u32;
            let expected = (content_w * content_h) as f32 * f32::from(luma)
                + (256 - content_w * content_h) as f32 * 114.0;
            assert!(
                (detection.confidence - expected / (256.0 * 255.0)).abs() < 0.002,
                "camera {camera}, P010={is_10bit}: {}",
                detection.confidence
            );
            let x = (7.5 - ((16 - content_w) / 2) as f32) / scale / width as f32;
            let y = (8.5 - ((16 - content_h) / 2) as f32) / scale / height as f32;
            assert!((detection.center_x - x).abs() < 1e-6);
            assert!((detection.center_y - y).abs() < 1e-6);
            assert_eq!(detection.camera, camera);
            assert_eq!(detector.tensor_f32.0, tensor_ptr);
            assert_eq!(
                detector.p010_scratch.as_ref().map(|(a, _)| a.0),
                scratch_ptr
            );
            assert!(
                detector
                    .detect(sizes.len(), &DetectorFrame::CudaNv12(frame))
                    .is_err()
            );
            frame.width += 2;
            assert!(
                detector
                    .detect(camera, &DetectorFrame::CudaNv12(frame))
                    .is_err()
            );
        }
    }
}
