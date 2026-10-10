# reco-calibrate

Stereo camera calibration for the reco L-shape projection.

## What it owns

- Direct image alignment on bounded luma images, using canonical core geometry. Pixel matching initializes placement; AKAZE provides a fallback when initialization lacks support.
- Fixed supplied optics and estimation of missing optics. Estimates describe stitching alignment, not independently certified lens profiles. Native ownership is exercised by `direct::solver::tests::fitting_parameters_never_modify_a_supplied_lens`.
- Independent-frame overlap checks produce `Confirmed` or `NeedsReview`; contradictory measured alignment fails. `direct::confirmation::tests` exercises the assessment rule. An application must obtain explicit acceptance before applying or saving a reviewable result.
- Lens profile loading from Gyroflow-style JSON, embedded camera metadata and the bundled database. No profile is required.
- **Temporal sync** — IMU cross-correlation (Gyroflow telemetry-parser) + audio cross-correlation fallback.
- **Live calibration** (`live` module) — `LiveFramePairSource` trait + `calibrate_from_live` function for streaming sources (OBS plugin, Jetson CSI, WebRTC); no `io` feature required.
- `calibrate_videos` wraps probing, synchronization, frame extraction and native calibration. The first sampled pair fits the model; later pairs verify it. `reco calibrate --frames 4` selects four pairs. A reviewable CLI result is retained in an explicit `.review.json` report; `--accept-unconfirmed` permits saving it as a calibration.
- `--estimate-optics` explicitly refits optics even when profiles are available. The GUI exposes the same choice and previews reviewable candidates without saving, covered by `explicit_candidate_preview_does_not_save_until_requested` in its calibration editor tests.

## Features

| Feature | Enables |
|---|---|
| `io` | FFmpeg-backed video file reading for `calibrate_videos` |
| `profiling` | Opt-in core tracing instrumentation |

## Build

```bash
cargo build -p reco-calibrate --features io   # file-based calibration (default for reco-cli)
cargo build -p reco-calibrate                  # live-only (OBS plugin, Jetson)
```
