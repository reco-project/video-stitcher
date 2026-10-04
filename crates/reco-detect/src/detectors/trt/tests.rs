//! Binding validation and input-sensitive native inference regressions.

use super::engine::BindingInfo;
use super::*;
use reco_core::detect::detector::YuvView;

fn bindings() -> Vec<BindingInfo> {
    vec![
        BindingInfo {
            name: "input".into(),
            is_input: true,
            dims: vec![1, 3, 16, 16],
            data_type: 0,
            linear_device: true,
        },
        BindingInfo {
            name: "output".into(),
            is_input: false,
            dims: vec![1, 1, 6],
            data_type: 0,
            linear_device: true,
        },
    ]
}

#[test]
fn model_contract_checks_both_bindings_before_allocating() {
    let mut valid = bindings();
    let layout = ModelLayout::new(&valid).unwrap();
    assert_eq!(
        (layout.size, layout.input_bytes, layout.output_elements),
        (16, 3072, 6)
    );
    valid.swap(0, 1);
    assert_eq!(ModelLayout::new(&valid).unwrap().input_index, 1);
    for dims in [
        vec![1, 3, 16],
        vec![2, 3, 16, 16],
        vec![1, 4, 16, 16],
        vec![1, 3, 16, 32],
        vec![1, 3, -1, -1],
        vec![1, 3, 0, 0],
        vec![1, 3, i64::MAX, i64::MAX],
    ] {
        let mut values = bindings();
        values[0].dims = dims;
        assert!(ModelLayout::new(&values).is_err());
    }
    for dims in [
        vec![1, 6],
        vec![2, 1, 6],
        vec![1, 1, 7],
        vec![1, 0, 6],
        vec![1, -1, 6],
        vec![1, i64::MAX, 6],
    ] {
        let mut values = bindings();
        values[1].dims = dims;
        assert!(ModelLayout::new(&values).is_err());
    }
    for index in 0..2 {
        let mut values = bindings();
        values[index].data_type = 1;
        assert!(ModelLayout::new(&values).is_err());
        values[index].data_type = 0;
        values[index].linear_device = false;
        assert!(ModelLayout::new(&values).is_err());
        values[index].linear_device = true;
        values[index].is_input = !values[index].is_input;
        assert!(ModelLayout::new(&values).is_err());
    }
    assert!(ModelLayout::new(&[]).is_err());
    assert!(ModelLayout::new(&bindings()[..1]).is_err());
    let mut extra = bindings();
    extra.extend(bindings());
    assert!(ModelLayout::new(&extra).is_err());
}

#[test]
fn invalid_dimensions_and_thresholds_fail_without_cuda_or_model() {
    for sizes in [
        vec![],
        vec![(0, 2)],
        vec![(2, 0)],
        vec![(2, 2), (u32::MAX, 2)],
    ] {
        let error = TrtGpuDetector::try_new("missing.engine", &sizes, 0.0, vec![])
            .err()
            .unwrap();
        assert!(error.to_string().contains("source sizes"));
    }
    for confidence in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        let error = TrtGpuDetector::try_new("missing.engine", &[(2, 2)], confidence, vec![])
            .err()
            .unwrap();
        assert!(error.to_string().contains("threshold"));
    }
    assert!(validate_plane(0, 8, 8, 4).is_err());
    assert!(validate_plane(1, 7, 8, 4).is_err());
    assert!(validate_plane(1, usize::MAX, 8, 4).is_err());
    assert!(validate_plane(1, i32::MAX as usize, 8, 4).is_err());
    assert!(validate_plane(1, 16, 8, 4).is_ok());
    assert!(validate_nv12((3, 4)).is_err());
    assert!(validate_nv12((4, 3)).is_err());
}

#[test]
fn integer_letterbox_mapping_matches_written_roi() {
    for size in [(6, 4), (4, 6), (8, 4), (4, 8), (8192, 1)] {
        let layout = InputLayout::new(size, 16);
        assert!(layout.resized.0 > 0 && layout.resized.1 > 0);
        let x = |v| v * size.0 as f32 * layout.scale + layout.padding.0 as f32;
        let y = |v| v * size.1 as f32 * layout.scale + layout.padding.1 as f32;
        let detections = super::super::postprocess(
            &[x(0.3), y(0.5), x(0.5), y(0.7), 1.0, 0.0],
            1,
            0,
            0.1,
            layout.scale,
            layout.padding.0 as f32,
            layout.padding.1 as f32,
            size.0,
            size.1,
        );
        assert!((detections[0].center_x - 0.4).abs() < 1e-4);
        assert!((detections[0].center_y - 0.6).abs() < 1e-4);
    }
}

fn upload(bytes: &[u8]) -> CudaBuffer {
    let buffer = CudaBuffer::new(bytes.len()).unwrap();
    cuda_memcpy_htod_2d(
        buffer.ptr(),
        bytes.len(),
        bytes.as_ptr(),
        bytes.len(),
        bytes.len(),
        1,
    )
    .unwrap();
    cuda::synchronize(std::ptr::null_mut()).unwrap();
    buffer
}

fn tensor(detector: &TrtGpuDetector) -> Vec<f32> {
    let mut data = vec![0.0; detector.model.input_bytes / size_of::<f32>()];
    detector
        .tensor
        .copy_to_host(&mut data, &detector.stream)
        .unwrap();
    data
}

fn check_output(
    detector: &mut TrtGpuDetector,
    camera: usize,
    frame: &DetectorFrame<'_>,
) -> Vec<f32> {
    let detections = detector.detect(camera, frame).unwrap_or_else(|error| {
        panic!(
            "camera {camera} {:?} {}: {error}",
            detector.inputs[camera].size,
            frame.variant_name()
        )
    });
    assert_eq!(detections.len(), 1);
    let layout = detector.inputs[camera];
    assert_eq!(detections[0].camera, camera);
    assert!(
        (detections[0].center_x
            - (7.5 - layout.padding.0 as f32) / layout.scale / layout.size.0 as f32)
            .abs()
            < 1e-6
    );
    assert!(
        (detections[0].center_y
            - (8.5 - layout.padding.1 as f32) / layout.scale / layout.size.1 as f32)
            .abs()
            < 1e-6
    );
    let data = tensor(detector);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    assert!((detections[0].confidence - mean).abs() < 1e-5);
    data
}

#[track_caller]
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (index, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() < 1e-6, "tensor[{index}]: {a} != {b}");
    }
}

/// Build the portable ONNX fixture for the local GPU. RECO_TRTEXEC overrides PATH lookup.
fn engine(dir: &Path, output_format: &str) -> std::path::PathBuf {
    let onnx = dir.join("probe.onnx");
    let engine = dir.join(format!("probe-{output_format}.engine"));
    std::fs::write(&onnx, crate::detectors::tensor_probe::MODEL).unwrap();
    let output = std::process::Command::new(
        std::env::var_os("RECO_TRTEXEC").unwrap_or_else(|| "trtexec".into()),
    )
    .arg(format!("--onnx={}", onnx.display()))
    .arg(format!("--saveEngine={}", engine.display()))
    .arg(format!("--outputIOFormats={output_format}:chw"))
    .arg("--skipInference")
    .output()
    .expect("trtexec required");
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    engine
}

#[test]
#[ignore = "requires NVIDIA CUDA, NPP, TensorRT and trtexec (or RECO_TRTEXEC)"]
fn letterboxed_rgba_honors_allocation_pitch() {
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(dir.path(), "fp32");
    let mut detector = TrtGpuDetector::try_new(&engine, &[(8, 4)], 0.0, vec![])
        .unwrap()
        .unwrap();
    let pitch = 16 * 4 + 32;
    let mut rgba = vec![0xA5; pitch * 16];
    let mut expected = vec![0.0; 3 * 16 * 16];
    for y in 0..16 {
        for x in 0..16 {
            for channel in 0..3 {
                let value = (y * 7 + x * 3 + channel * 20) as u8;
                rgba[y * pitch + x * 4 + channel] = value;
                expected[channel * 256 + y * 16 + x] = value as f32 / 255.0;
            }
        }
    }
    let source = upload(&rgba);
    detector.normalize_letterboxed(source.ptr(), pitch).unwrap();
    detector.infer().unwrap();
    let actual = tensor(&detector);
    close(&actual, &expected);
    let mean = expected.iter().sum::<f32>() / expected.len() as f32;
    assert!((detector.output_host[4] - mean).abs() < 1e-5);
}

#[test]
#[ignore = "requires NVIDIA CUDA, NPP, TensorRT and trtexec (or RECO_TRTEXEC)"]
fn native_inference_preserves_layout_pixels_and_storage_across_cameras() {
    let dir = tempfile::tempdir().unwrap();
    let incompatible = engine(dir.path(), "fp16");
    let error = TrtGpuDetector::try_new(&incompatible, &[(8, 4)], 0.0, vec![])
        .err()
        .unwrap();
    assert!(error.to_string().contains("FP32"), "{error}");
    let engine = engine(dir.path(), "fp32");
    let sizes = [(8, 4), (4, 8), (32, 16), (16, 32), (8, 8), (12, 8), (8, 12)];
    let mut detector = TrtGpuDetector::try_new(&engine, &sizes, 0.0, vec![])
        .unwrap()
        .unwrap();
    let inference_ptr = detector.tensor.ptr();
    let mut scratch_ptrs = None;
    for camera in [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
        .into_iter()
        .cycle()
        .take(13 * 8)
    {
        let (width, height) = sizes[camera];
        let layout = detector.inputs[camera];
        let value = 40 + camera as u8 * 25;
        let pixels = width as usize * height as usize;
        let rgba = vec![value; pixels * 4];
        let host = DetectorFrame::Rgba {
            data: &rgba,
            width,
            height,
        };
        let data = check_output(&mut detector, camera, &host);
        // Independently construct the whole expected letterboxed tensor.
        let mut expected = vec![114.0 / 255.0; 3 * 16 * 16];
        for channel in 0..3 {
            for y in layout.padding.1..layout.padding.1 + layout.resized.1 {
                for x in layout.padding.0..layout.padding.0 + layout.resized.0 {
                    expected[channel * 256 + y as usize * 16 + x as usize] = value as f32 / 255.0;
                }
            }
        }
        close(&data, &expected);
        let pitch = width as usize * 4 + 16;
        let mut padded = vec![0; pitch * height as usize];
        for row in padded.chunks_mut(pitch) {
            row[..width as usize * 4].fill(value);
        }
        let rgba_gpu = upload(&padded);
        close(
            &check_output(
                &mut detector,
                camera,
                &DetectorFrame::CudaRgba {
                    ptr: rgba_gpu.ptr(),
                    pitch,
                    width,
                    height,
                },
            ),
            &expected,
        );

        let y = vec![value; pixels];
        let uv = vec![128; pixels / 2];
        let nv12 = DetectorFrame::Yuv(YuvView {
            y: &y,
            chroma: ChromaPlanes::Nv12 { uv: &uv },
            width,
            height,
        });
        let expected_nv12 = check_output(&mut detector, camera, &nv12);
        let pitch = width as usize + 8;
        let y_gpu = upload(&vec![value; pitch * height as usize]);
        let uv_gpu = upload(&vec![128; pitch * height as usize / 2]);
        let mut frame = CudaNv12Frame {
            y_ptr: y_gpu.ptr(),
            uv_ptr: uv_gpu.ptr(),
            y_pitch: pitch,
            uv_pitch: pitch,
            width,
            height,
            rotation: 0,
            is_10bit: false,
        };
        close(
            &check_output(&mut detector, camera, &DetectorFrame::CudaNv12(frame)),
            &expected_nv12,
        );
        // P010 permits independent Y/UV pitches; each plane is converted to tight NV12.
        let uv_pitch = pitch + 8;
        let y16 = vec![u16::from(value) << 8; pitch * height as usize];
        let uv16 = vec![128u16 << 8; uv_pitch * height as usize / 2];
        let y16 = upload(bytemuck::cast_slice(&y16));
        let uv16 = upload(bytemuck::cast_slice(&uv16));
        frame.y_ptr = y16.ptr();
        frame.uv_ptr = uv16.ptr();
        frame.y_pitch = pitch * 2;
        frame.uv_pitch = uv_pitch * 2;
        frame.is_10bit = true;
        frame.rotation = 180;
        close(
            &check_output(&mut detector, camera, &DetectorFrame::CudaNv12(frame)),
            &expected_nv12,
        );
        assert_eq!(detector.tensor.ptr(), inference_ptr);
        let n = detector.nv12.as_ref().unwrap();
        let current = [
            n.rgb.ptr(),
            n.mirrored.ptr(),
            n.resized.ptr(),
            detector.p010.as_ref().unwrap().ptr(),
            detector.upload_nv12.as_ref().unwrap().ptr(),
            detector.upload_rgba.as_ref().unwrap().ptr(),
            detector.resized_rgba.as_ref().unwrap().ptr(),
        ];
        assert_eq!(*scratch_ptrs.get_or_insert(current), current);
    }

    // Nonuniform RGBA pins nearest-neighbour sampling rather than merely its mean.
    let mut rgba = Vec::new();
    for y in 0..4u8 {
        for x in 0..8u8 {
            rgba.extend([x * 31, y * 63, (x + y) * 21, 255]);
        }
    }
    let actual = check_output(
        &mut detector,
        0,
        &DetectorFrame::Rgba {
            data: &rgba,
            width: 8,
            height: 4,
        },
    );
    let mut expected = vec![114.0 / 255.0; 3 * 16 * 16];
    for channel in 0..3 {
        for y in 0..8 {
            for x in 0..16 {
                expected[channel * 256 + (y + 4) * 16 + x] =
                    rgba[((y / 2) * 8 + x / 2) * 4 + channel] as f32 / 255.0;
            }
        }
    }
    close(&actual, &expected);

    // Preserve native NPP colour conversion, independently of the ORT converter.
    let y = [81; 32];
    let uv = [90, 240].repeat(8);
    let actual = check_output(
        &mut detector,
        0,
        &DetectorFrame::Yuv(YuvView {
            y: &y,
            chroma: ChromaPlanes::Nv12 { uv: &uv },
            width: 8,
            height: 4,
        }),
    );
    for (channel, value) in [208, 30, 3].into_iter().enumerate() {
        for y in 4..12 {
            for x in 0..16 {
                expected[channel * 256 + y * 16 + x] = value as f32 / 255.0;
            }
        }
    }
    close(&actual, &expected);

    // Rotation must change source pixels without changing postprocess geometry.
    let y: Vec<u8> = (0..32).map(|v| 30 + v * 5).collect();
    let uv = vec![128; 16];
    let y_gpu = upload(&y);
    let uv_gpu = upload(&uv);
    let actual = check_output(
        &mut detector,
        0,
        &DetectorFrame::CudaNv12(CudaNv12Frame {
            y_ptr: y_gpu.ptr(),
            uv_ptr: uv_gpu.ptr(),
            y_pitch: 8,
            uv_pitch: 8,
            width: 8,
            height: 4,
            rotation: 180,
            is_10bit: false,
        }),
    );
    let reversed: Vec<u8> = y.into_iter().rev().collect();
    let expected = check_output(
        &mut detector,
        0,
        &DetectorFrame::Yuv(YuvView {
            y: &reversed,
            chroma: ChromaPlanes::Nv12 { uv: &uv },
            width: 8,
            height: 4,
        }),
    );
    close(&actual, &expected);

    // Dropping one runtime must not invalidate another runtime's logger or engine.
    let mut letterboxed = TrtGpuDetector::try_new(&engine, &sizes, 0.0, vec![])
        .unwrap()
        .unwrap();
    drop(detector);
    let square = upload(&vec![200; 16 * 16 * 4]);
    for _ in 0..4 {
        letterboxed
            .normalize_letterboxed(square.ptr(), 16 * 4)
            .unwrap();
        letterboxed.infer().unwrap();
        let data = tensor(&letterboxed);
        close(&data, &vec![200.0 / 255.0; 3 * 16 * 16]);
        assert!(
            letterboxed.nv12.is_none()
                && letterboxed.resized_rgba.is_none()
                && letterboxed.p010.is_none()
                && letterboxed.upload_nv12.is_none()
                && letterboxed.upload_rgba.is_none()
        );
    }
    // Layout errors are rejected before touching even deliberately invalid device addresses.
    for frame in [
        DetectorFrame::CudaNv12(CudaNv12Frame {
            y_ptr: 2,
            uv_ptr: 2,
            y_pitch: 17,
            uv_pitch: 16,
            width: 8,
            height: 4,
            rotation: 0,
            is_10bit: true,
        }),
        DetectorFrame::CudaRgba {
            ptr: 1,
            pitch: 32,
            width: 4,
            height: 8,
        },
        DetectorFrame::CudaRgba {
            ptr: 0,
            pitch: 32,
            width: 8,
            height: 4,
        },
        DetectorFrame::CudaRgba {
            ptr: 1,
            pitch: 31,
            width: 8,
            height: 4,
        },
        DetectorFrame::Rgba {
            data: &[0; 4],
            width: 8,
            height: 4,
        },
        DetectorFrame::Yuv(YuvView {
            y: &[],
            chroma: ChromaPlanes::Nv12 { uv: &[] },
            width: 8,
            height: 4,
        }),
        DetectorFrame::CudaNv12(CudaNv12Frame {
            y_ptr: 1,
            uv_ptr: 1,
            y_pitch: 8,
            uv_pitch: 16,
            width: 8,
            height: 4,
            rotation: 0,
            is_10bit: false,
        }),
        DetectorFrame::CudaNv12(CudaNv12Frame {
            y_ptr: 1,
            uv_ptr: 1,
            y_pitch: 8,
            uv_pitch: 8,
            width: 8,
            height: 4,
            rotation: 90,
            is_10bit: false,
        }),
    ] {
        assert!(letterboxed.detect(0, &frame).is_err());
    }
    assert!(
        letterboxed
            .detect(
                99,
                &DetectorFrame::CudaRgba {
                    ptr: 1,
                    pitch: 8 * 4,
                    width: 8,
                    height: 4
                }
            )
            .is_err()
    );
    for (ptr, pitch) in [(0, 64), (1, 63)] {
        assert!(letterboxed.normalize_letterboxed(ptr, pitch).is_err());
    }
    assert!(matches!(
        letterboxed.detect(
            0,
            &DetectorFrame::Yuv(YuvView {
                y: &[],
                chroma: ChromaPlanes::Yuv420p { u: &[], v: &[] },
                width: 8,
                height: 4
            })
        ),
        Err(DetectorError::UnsupportedFrameKind)
    ));
    // Move ownership, infer, and clean up with the CUDA context made current on that thread.
    std::thread::spawn(move || {
        let rgba = vec![200; 8 * 4 * 4];
        assert_eq!(
            letterboxed
                .detect(
                    0,
                    &DetectorFrame::Rgba {
                        data: &rgba,
                        width: 8,
                        height: 4
                    }
                )
                .unwrap()
                .len(),
            1
        );
    })
    .join()
    .unwrap();
}
