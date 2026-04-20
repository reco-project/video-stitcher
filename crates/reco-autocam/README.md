# reco-autocam

AI camera control: detectors are plugged in, directors decide where the virtual camera points.

## What it owns

- **Directors** — `BallDirector` (single-ball following + plausibility), `FieldDirector` (ball + player DBSCAN clustering, broadcast-style), `SweepDirector` (constrained pan sweep for venues without trackable subjects).
- **SmoothedDirector** — One Euro filter decorator that smooths any underlying director's yaw/pitch/FOV output.
- **RoiFilteredDetector** — wraps any `UnifiedDetector` with a polygonal field-ROI mask so detections outside the play area never reach the director.
- **`setup_autocam`** — orchestrator: picks the appropriate detector backend for the current platform + model file, wires the director, attaches to a `StitchSession`.
- **`AutocamConfig`** — builder-style config surface over the 11-arg `setup_autocam` for new consumers (see FRICTION N10 for the consumer-facing API evolution).

## Safety policy

Zero `unsafe` code (`#![forbid(unsafe_code)]`). All FFI / platform crossings live in `reco-core` or `reco-detect`; the intelligence layer stays in safe Rust.

## Build

```bash
cargo build -p reco-autocam
cargo build -p reco-autocam --no-default-features --features tensorrt-native  # Jetson
```

Without any of `ort`, `tensorrt-native`, `ncnn`, `setup_autocam` logs a warning and returns `Ok(false)` — the session keeps running without AI control.
