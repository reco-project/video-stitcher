# reco-autocam

AI camera control: trackers interpret detections, and panners request the virtual camera's orientation and FOV.

## What it owns

- **Trackers**: `BallTracker` follows a ball; `ClassProvider` supplies per-class entities.
- **Panners**: `FieldPanner` follows ball/player state, `SweepPanner` generates a test pan, and `FilePanner` reads a pose trajectory. Core owns lookahead smoothing and coverage resolution.
- **ROI filtering**: `RoiFilter` applies camera-space field polygons to completed detections in place; `RoiFilteredDetector` applies that same policy after synchronous inference. Public policy/decorator agreement is tested in `tests/roi_filter.rs`.
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
