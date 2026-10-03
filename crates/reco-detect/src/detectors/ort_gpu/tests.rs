//! Native tensor contents and per-camera coordinate mapping through actual inference.

use super::*;
use reco_core::interop::cuda::cuda_memcpy_htod_2d;

// ONNX IR 8, opset 13. ReduceMean(input[1,3,16,16]) supplies confidence in
// output[1,1,6] = [5,6,10,11,mean,0]. The model observes the entire tensor.
const MODEL: &[u8] = &[
    8, 8, 58, 158, 2, 10, 42, 10, 5, 105, 110, 112, 117, 116, 18, 4, 109, 101, 97, 110, 34, 10, 82,
    101, 100, 117, 99, 101, 77, 101, 97, 110, 42, 15, 10, 8, 107, 101, 101, 112, 100, 105, 109,
    115, 24, 0, 160, 1, 2, 10, 34, 10, 4, 109, 101, 97, 110, 10, 5, 115, 104, 97, 112, 101, 18, 10,
    99, 111, 110, 102, 105, 100, 101, 110, 99, 101, 34, 7, 82, 101, 115, 104, 97, 112, 101, 10, 53,
    10, 3, 98, 111, 120, 10, 10, 99, 111, 110, 102, 105, 100, 101, 110, 99, 101, 10, 5, 99, 108,
    97, 115, 115, 18, 6, 111, 117, 116, 112, 117, 116, 34, 6, 67, 111, 110, 99, 97, 116, 42, 11,
    10, 4, 97, 120, 105, 115, 24, 2, 160, 1, 2, 18, 12, 116, 101, 110, 115, 111, 114, 95, 112, 114,
    111, 98, 101, 42, 16, 8, 3, 16, 7, 58, 3, 1, 1, 1, 66, 5, 115, 104, 97, 112, 101, 42, 31, 8, 1,
    8, 1, 8, 4, 16, 1, 34, 16, 0, 0, 160, 64, 0, 0, 192, 64, 0, 0, 32, 65, 0, 0, 48, 65, 66, 3, 98,
    111, 120, 42, 21, 8, 1, 8, 1, 8, 1, 16, 1, 34, 4, 0, 0, 0, 0, 66, 5, 99, 108, 97, 115, 115, 90,
    31, 10, 5, 105, 110, 112, 117, 116, 18, 22, 10, 20, 8, 1, 18, 16, 10, 2, 8, 1, 10, 2, 8, 3, 10,
    2, 8, 16, 10, 2, 8, 16, 98, 28, 10, 6, 111, 117, 116, 112, 117, 116, 18, 18, 10, 16, 8, 1, 18,
    12, 10, 2, 8, 1, 10, 2, 8, 1, 10, 2, 8, 6, 66, 4, 10, 0, 16, 13,
];

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
