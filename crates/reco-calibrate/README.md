# reco-calibrate

Stereo camera calibration for the reco L-shape projection.

## What it owns

- **AKAZE feature detection + matching** with Lowe-ratio and RANSAC filtering for outlier rejection.
- **Non-linear optimizer** (argmin) that finds `cameraAxisOffset`, `intersect`, `xTy`, `xRz`, `zRx`, `zRz` parameters against the L-shape projection model.
- **Lens profile loader** — Gyroflow-style JSON (camera matrix + fisheye distortion coeffs + optional radial limit); bundled lens database covers 4200+ Gyroflow profiles plus manual entries (e.g. GoPro HERO10 Linear).
- **Temporal sync** — IMU cross-correlation (Gyroflow telemetry-parser) + audio cross-correlation fallback.
- **Live calibration** (`live` module) — `LiveFramePairSource` trait + `calibrate_from_live` function for streaming sources (OBS plugin, Jetson CSI, WebRTC); no `io` feature required.
- **`calibrate_videos`** — file-based one-shot calibration with `--skip-start`, `--frames`, `--akaze-threshold`, `--lock-cam-d`, `--lock-z-rx` controls.

## Features

| Feature | Enables |
|---|---|
| `io` | FFmpeg-backed video file reading for `calibrate_videos` |
| `profiling` | `tracing` spans around AKAZE + optimizer |

## Build

```bash
cargo build -p reco-calibrate --features io   # file-based calibration (default for reco-cli)
cargo build -p reco-calibrate                  # live-only (OBS plugin, Jetson)
```
