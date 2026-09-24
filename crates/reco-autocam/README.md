# reco-autocam

AI camera control: trackers interpret detections, and panners request the virtual camera's orientation and FOV.

## What it owns

- **Trackers**: `BallTracker` follows a ball; `ClassProvider` supplies per-class entities.
- **Panners**: `FieldPanner` follows ball/player state, `SweepPanner` generates a test pan, and `FilePanner` reads a pose trajectory. Core owns lookahead smoothing and coverage resolution.
- **RoiFilteredDetector**: wraps any `Detector` with a polygonal field-ROI mask so detections outside the play area never reach tracking.
- **`setup_autocam`** — orchestrator: accepts `AutocamConfig`, picks the detector backend for the platform + model file, wires tracker/panner, attaches to a `StitchSession`.
- **`AutocamConfig`** — builder-style configuration. `FieldPannerConfig` exposes all AI tracking tuning parameters with safe defaults.

## Safety policy

Zero `unsafe` code (`#![forbid(unsafe_code)]`). All FFI / platform crossings live in `reco-core` or `reco-detect`; the intelligence layer stays in safe Rust.

## Build

```bash
cargo build -p reco-autocam
cargo build -p reco-autocam --no-default-features --features tensorrt-native  # Jetson
```

Without any of `ort`, `tensorrt-native`, `ncnn`, `setup_autocam` logs a warning and returns `Ok(false)` — the session keeps running without AI control.
