//! Command-line parsing must not create or truncate profiling output.

#[test]
fn argument_only_exits_leave_trace_untouched() {
    let directory = std::env::temp_dir().join(format!("reco-cli-startup-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let trace = directory.join("reco-trace.json");
    for arguments in [
        vec!["--help"],
        vec!["--version"],
        vec!["stitch", "--help"],
        vec!["--invalid-argument-for-test"],
    ] {
        for existing in [false, true] {
            if existing {
                std::fs::write(&trace, "previous trace").unwrap();
            }
            let output = std::process::Command::new(env!("CARGO_BIN_EXE_reco"))
                .args(&arguments)
                .current_dir(&directory)
                .output()
                .unwrap();
            assert_eq!(
                output.status.success(),
                arguments[0] != "--invalid-argument-for-test",
                "{output:?}"
            );
            if existing {
                assert_eq!(std::fs::read_to_string(&trace).unwrap(), "previous trace");
                std::fs::remove_file(&trace).unwrap();
            } else {
                assert!(!trace.exists(), "{arguments:?} created a trace");
            }
        }
    }
    std::fs::remove_dir(directory).unwrap();
}

#[cfg(feature = "profiling")]
#[test]
fn unusable_trace_path_is_a_command_error_not_a_panic() {
    let directory =
        std::env::temp_dir().join(format!("reco-cli-trace-error-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let trace = directory.join("reco-trace.json");
    std::fs::create_dir(&trace).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_reco"))
        .arg("info")
        .current_dir(&directory)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        error.contains("create profiling trace reco-trace.json"),
        "{error}"
    );
    assert!(!error.contains("panicked"), "{error}");
    std::fs::remove_dir(trace).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn export_collisions_leave_inputs_intact() {
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::projection::Cylinder;

    let dir = std::env::temp_dir().join(format!("reco-cli-export-paths-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let calibration = dir.join("calibration.json");
    Calibration::new(
        vec![Lens::flat(64, 36)],
        Cylinder::default(),
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    )
    .to_file(&calibration)
    .unwrap();
    let bytes = std::fs::read(&calibration).unwrap();
    let input = dir.join("input.mp4");
    std::fs::write(&input, b"preserve recording").unwrap();
    let url = format!("file:{}", input.display());
    for (source, events) in [
        (input.as_path(), &calibration),
        (std::path::Path::new(&url), &input),
    ] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_reco"))
            .current_dir(&dir)
            .arg("stitch")
            .arg(source)
            .arg("-c")
            .arg(&calibration)
            .args(["-o", "out.mp4", "--cpu", "--events"])
            .arg(events)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("must be distinct"),
            "{output:?}"
        );
        assert_eq!(std::fs::read(&calibration).unwrap(), bytes);
        assert_eq!(std::fs::read(&input).unwrap(), b"preserve recording");
        assert!(!dir.join("out.mp4").exists());
    }
    std::fs::remove_dir_all(dir).unwrap();
}
