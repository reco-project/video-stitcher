//! Missing models must fail before input opening, even without an inference backend.

#![cfg(feature = "autocam")]

use std::process::Command;

#[test]
fn missing_model_is_reported_before_opening_sources() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "reco-model-preflight-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&dir).unwrap();
    let calibration = dir.join("calibration.json");
    let lens = reco_core::calibration::Lens::fisheye(64, 64, 32.0, 32.0, 32.0, 32.0, [0.0; 4]);
    reco_core::calibration::Calibration::new(
        vec![lens.clone(), lens],
        reco_core::projection::LShape {
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        },
        reco_core::calibration::Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        },
    )
    .to_file(&calibration)
    .unwrap();
    let destination = dir.join("output.mp4");
    std::fs::write(&destination, b"previous output").unwrap();
    let model = dir.join("missing.onnx");
    for tracking in ["field", "ball", "sweep"] {
        let output = Command::new(env!("CARGO_BIN_EXE_reco"))
            .args([
                "stitch",
                "missing-left.mp4",
                "missing-right.mp4",
                "--cpu",
                "--tracking",
                tracking,
                "--model",
            ])
            .arg(&model)
            .arg("--calibration")
            .arg(&calibration)
            .arg("--output")
            .arg(&destination)
            .current_dir(&dir)
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{output:?}");
        assert_eq!(std::fs::read(&destination).unwrap(), b"previous output");
        if tracking == "sweep" {
            assert!(!error.contains("AI model path does not exist"), "{error}");
            assert!(
                error.contains("missing-left.mp4"),
                "sweep must proceed to input opening: {error}"
            );
        } else {
            assert!(
                error.contains("AI model path does not exist") && error.contains("missing.onnx"),
                "missing-model error must precede missing-source error: {error}"
            );
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
