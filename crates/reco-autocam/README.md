# reco-autocam

AI camera control: trackers interpret detections, and panners request the virtual camera's orientation and FOV.

## What it owns

- **Trackers**: `BallTracker` follows a ball; `ClassProvider` supplies per-class entities.
- **Panners**: `FieldPanner` follows ball/player state, `SweepPanner` generates a test pan, and `FilePanner` reads a pose trajectory. Core supplies future world states and resolves coverage.
- **ROI filtering**: `RoiFilter` tests completed detections against the calibration’s four source-attached field corners and renderer-derived camera ownership. The engine applies it after inference on every frame backend; no image transfer or detector wrapper is needed. Anchors, invalidation and batch preservation are exercised in `tests/roi_filter.rs`.
- **`setup_autocam`** — orchestrator: accepts `AutocamConfig`, picks the detector backend for the platform + model file, wires tracker/panner, attaches to a `StitchSession`.
- **`AutocamConfig`** — builder-style configuration. `FieldPannerConfig` exposes all AI tracking tuning parameters with safe defaults.

## Safety policy

Zero `unsafe` code (`#![forbid(unsafe_code)]`). All FFI / platform crossings live in `reco-core` or `reco-detect`; the intelligence layer stays in safe Rust.

## Build

```bash
cargo build -p reco-autocam
cargo build -p reco-autocam --no-default-features  # ROI, trackers and panners without GPU or inference dependencies
cargo build -p reco-autocam --no-default-features --features tensorrt-native  # Jetson
```

The `detectors` feature exposes GPU preprocessing adapters; inference backend features enable it automatically. `profiling` alone does not pull in detector or GPU dependencies. The isolated dependency graph is checked in `.github/workflows/rust.yml`.

Without any of `ort`, `tensorrt-native`, `ncnn`, detected tracking modes log a warning and return `Ok(false)` from `setup_autocam`. Sweep mode needs no model or detector and uses the supplied source FPS, exercised by `tests/setup.rs`.
