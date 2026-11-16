//! Provider failures must be isolated because ORT caches loaded libraries globally.

use super::*;

// ONNX IR 8, opset 13: Identity(input: float[1,3,2,2]) -> output.
const IDENTITY: &[u8] = &[
    8, 8, 58, 104, 10, 25, 10, 5, 105, 110, 112, 117, 116, 18, 6, 111, 117, 116, 112, 117, 116, 34,
    8, 73, 100, 101, 110, 116, 105, 116, 121, 18, 8, 105, 100, 101, 110, 116, 105, 116, 121, 90,
    31, 10, 5, 105, 110, 112, 117, 116, 18, 22, 10, 20, 8, 1, 18, 16, 10, 2, 8, 1, 10, 2, 8, 3, 10,
    2, 8, 2, 10, 2, 8, 2, 98, 32, 10, 6, 111, 117, 116, 112, 117, 116, 18, 22, 10, 20, 8, 1, 18,
    16, 10, 2, 8, 1, 10, 2, 8, 3, 10, 2, 8, 2, 10, 2, 8, 2, 66, 4, 10, 0, 16, 13,
];

fn host_inference(model: &Path) {
    let (mut session, size, labels) = create_ort_session(model, vec!["object".into()]).unwrap();
    assert_eq!(size, 2);
    assert_eq!(labels, ["object"]);
    assert_identity(&mut session);
}

fn assert_identity(session: &mut Session) {
    let values: Vec<f32> = (0..12).map(|n| n as f32).collect();
    let tensor = ort::value::Tensor::from_array(([1, 3, 2, 2], values.clone())).unwrap();
    let output = session.run(ort::inputs![tensor]).unwrap();
    assert_eq!(output[0].try_extract_tensor::<f32>().unwrap().1, values);
}

#[test]
fn host_session_runs_identity() {
    if ort_runtime_available().is_err() {
        return; // Missing-runtime behavior is exercised in a fresh subprocess below.
    }
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("identity.onnx");
    std::fs::write(&model, IDENTITY).unwrap();
    host_inference(&model);
}

#[test]
#[ignore = "requires a working native ONNX Runtime"]
fn required_host_session_runs_identity() {
    ort_runtime_available().expect("ONNX Runtime must load for native acceptance");
    host_session_runs_identity();
}

#[test]
fn concurrent_sessions_run_independently() {
    const CASE: &str = "concurrent_sessions_run_independently";
    if !child_case(CASE) {
        run_child(CASE, false);
        return;
    }
    if ort_runtime_available().is_err() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("identity.onnx");
    std::fs::write(&model, IDENTITY).unwrap();
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..2)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    let result = create_ort_session(&model, vec!["object".into()]);
                    // Both loads must finish before inference, including a failed load.
                    barrier.wait();
                    let (mut session, size, labels) = result.unwrap();
                    assert_eq!(size, 2);
                    assert_eq!(labels, ["object"]);
                    assert_identity(&mut session);
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
    });
}

#[test]
fn failed_model_load_does_not_block_next_session() {
    const CASE: &str = "failed_model_load_does_not_block_next_session";
    if !child_case(CASE) {
        run_child(CASE, false);
        return;
    }
    if ort_runtime_available().is_err() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.onnx");
    assert!(create_ort_session(&model, vec![]).is_err());
    std::fs::write(&model, b"not an ONNX model").unwrap();
    assert!(create_ort_session(&model, vec![]).is_err());
    std::fs::write(&model, IDENTITY).unwrap();
    host_inference(&model);
}

fn child_case(case: &str) -> bool {
    std::env::var("RECO_ORT_TEST_CHILD").as_deref() == Ok(case)
}

fn run_child(case: &str, copy_binary: bool) {
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    let dir = tempfile::tempdir().unwrap();
    let exe = std::env::current_exe().unwrap();
    let exe = if copy_binary {
        let copied = dir.path().join(exe.file_name().unwrap());
        #[cfg(target_os = "linux")]
        if case == "missing_tensorrt_uses_cuda" {
            for name in [
                "libonnxruntime_providers_shared.so",
                "libonnxruntime_providers_cuda.so",
            ] {
                let original = exe.parent().unwrap().join(name).canonicalize().unwrap();
                std::os::unix::fs::symlink(original, dir.path().join(name)).unwrap();
            }
        }
        std::fs::copy(exe, &copied).unwrap();
        copied
    } else {
        exe
    };
    let log_path = dir.path().join("child.log");
    let log = std::fs::File::create(&log_path).unwrap();
    let mut command = Command::new(exe);
    command
        .args([
            "--exact",
            &format!("ort_session::provider_tests::{case}"),
            "--nocapture",
            "--include-ignored",
        ])
        .env("RECO_ORT_TEST_CHILD", case)
        .stdout(Stdio::from(log.try_clone().unwrap()))
        .stderr(Stdio::from(log));
    if case == "missing_runtime_returns_errors" {
        command.env("ORT_DYLIB_PATH", dir.path().join("missing-runtime"));
    }
    let mut child = command.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            let output = std::fs::read_to_string(log_path).unwrap();
            assert!(
                status.success() && output.contains("1 passed; 0 failed"),
                "{status}: {output}"
            );
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!(
                "provider initialization did not terminate: {}",
                std::fs::read_to_string(log_path).unwrap()
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(all(
    target_os = "linux",
    feature = "tensorrt",
    feature = "cuda",
    not(feature = "load-dynamic")
))]
#[test]
#[ignore = "requires CUDA runtime and provider libraries"]
fn missing_tensorrt_uses_cuda() {
    const CASE: &str = "missing_tensorrt_uses_cuda";
    if !child_case(CASE) {
        run_child(CASE, true);
        return;
    }
    let report = crate::probe::probe_execution_providers();
    assert_eq!(report.providers, ["CUDA", "CPU"]);
    assert_eq!(report.errors.len(), 1);
    assert!(report.errors[0].starts_with("TensorRT:"));
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("identity.onnx");
    std::fs::write(&model, IDENTITY).unwrap();
    assert!(create_cuda_ort_session(&model, vec![]).is_ok());
    host_inference(&model);
}

#[cfg(all(
    target_os = "linux",
    any(feature = "tensorrt", feature = "cuda"),
    not(feature = "load-dynamic")
))]
#[test]
fn missing_providers_reject_cuda_but_allow_host_inference() {
    use reco_core::detect::detector::{CudaNv12Frame, Detector, DetectorError, DetectorFrame};

    const CASE: &str = "missing_providers_reject_cuda_but_allow_host_inference";
    if !child_case(CASE) {
        run_child(CASE, true);
        return;
    }
    let report = crate::probe::probe_execution_providers();
    assert_eq!(report.providers, ["CPU"]);
    assert!(!report.errors.is_empty());
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("identity.onnx");
    std::fs::write(&model, IDENTITY).unwrap();
    let error = create_cuda_ort_session(&model, vec![]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("CUDA input requires a registered")
    );
    host_inference(&model);
    let mut detector = crate::CpuYoloDetector::from_file(&model).unwrap();
    let unsupported = DetectorFrame::CudaNv12(CudaNv12Frame {
        y_ptr: 0,
        uv_ptr: 0,
        y_pitch: 2,
        uv_pitch: 2,
        width: 2,
        height: 2,
        rotation: 0,
        is_10bit: false,
        is_full_range: true,
    });
    assert!(matches!(
        detector.detect(0, &unsupported),
        Err(DetectorError::UnsupportedFrameKind)
    ));

    // Valid ONNX graph, but Gather index 5 exceeds the input's first dimension.
    let invalid_inference = [
        8, 8, 58, 143, 1, 10, 43, 10, 5, 105, 110, 112, 117, 116, 10, 5, 105, 110, 100, 101, 120,
        18, 6, 111, 117, 116, 112, 117, 116, 34, 6, 71, 97, 116, 104, 101, 114, 42, 11, 10, 4, 97,
        120, 105, 115, 24, 0, 160, 1, 2, 18, 13, 105, 110, 118, 97, 108, 105, 100, 95, 105, 110,
        100, 101, 120, 42, 14, 8, 1, 16, 7, 58, 1, 5, 66, 5, 105, 110, 100, 101, 120, 90, 31, 10,
        5, 105, 110, 112, 117, 116, 18, 22, 10, 20, 8, 1, 18, 16, 10, 2, 8, 1, 10, 2, 8, 3, 10, 2,
        8, 2, 10, 2, 8, 2, 98, 32, 10, 6, 111, 117, 116, 112, 117, 116, 18, 22, 10, 20, 8, 1, 18,
        16, 10, 2, 8, 1, 10, 2, 8, 3, 10, 2, 8, 2, 10, 2, 8, 2, 66, 4, 10, 0, 16, 13,
    ];
    let model = dir.path().join("invalid-inference.onnx");
    std::fs::write(&model, invalid_inference).unwrap();
    assert!(create_ort_session(&model, vec![]).is_ok());
    assert!(crate::CpuYoloDetector::from_file(&model).is_err());
}

#[cfg(feature = "load-dynamic")]
#[test]
fn missing_runtime_returns_errors() {
    const CASE: &str = "missing_runtime_returns_errors";
    if !child_case(CASE) {
        run_child(CASE, false);
        return;
    }
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    for sizes in [vec![], vec![(0, 2)], vec![(3, 2)]] {
        let error = crate::OrtGpuDetector::try_new("unused.onnx", &sizes, 0.1, vec![], false)
            .err()
            .expect("invalid dimensions must fail without loading the runtime");
        assert!(error.to_string().contains(if sizes.is_empty() {
            "at least one"
        } else {
            "positive and even"
        }));
    }
    for _ in 0..2 {
        let report = crate::probe::probe_execution_providers();
        assert!(!report.is_available());
        assert!(!report.errors.is_empty());
        assert!(matches!(
            create_ort_session(Path::new("unused.onnx"), vec![]),
            Err(SessionError::RuntimeUnavailable(_))
        ));
    }
}

#[cfg(not(any(feature = "cuda", feature = "tensorrt")))]
#[test]
fn host_providers_do_not_admit_cuda_input() {
    if ort_runtime_available().is_ok() {
        let error = session_builder(true).err().expect("CUDA input must fail");
        assert!(
            error
                .to_string()
                .contains("CUDA input requires a registered")
        );
    }
}
