# reco-calibrate

Stereo camera calibration for the reco L-shape projection.

## What it owns

- Direct image alignment on bounded luma images, using canonical core geometry. Placement starts with local pixel matching. When local alignment fails, the fitter scores upright and inverted left-camera placements on low-resolution luminance, then refines a candidate using bidirectional pixel matches. The search is bounded and checks later synchronized pairs separately. See [overlap search](src/direct/overlap.rs) and [proposal fitting](src/direct/workflow.rs).
- Fixed supplied optics and estimation of missing optics. Estimates describe stitching alignment, not independently certified lens profiles. Native ownership is exercised by `direct::solver::tests::fitting_parameters_never_modify_a_supplied_lens`.
- Independent-frame overlap checks produce `Confirmed` or `NeedsReview`; contradictory measured alignment fails. `direct::confirmation::tests` exercises the assessment rule. An application must obtain explicit acceptance before applying or saving a reviewable result.
- Optional fixed optics from a Reco or Gyroflow profile file or the manual database picker. Each camera without a supplied lens is estimated from overlap.
- **Temporal sync** — IMU cross-correlation (Gyroflow telemetry-parser) + audio cross-correlation fallback.
- **Live calibration** (`live` module) — `LiveFramePairSource` trait + `calibrate_from_live` function for streaming sources (OBS plugin, Jetson CSI, WebRTC); no `io` feature required.
- `calibrate_videos` wraps probing, synchronization, frame extraction and native calibration. The first sampled pair fits the model; later pairs verify it. `reco calibrate --frames 4` selects four pairs. A reviewable CLI result is retained in an explicit `.review.json` report; `--accept-unconfirmed` permits saving it as a calibration.
- `--left-profile` and `--right-profile` independently fix each camera’s optics. The GUI previews reviewable candidates without saving, covered by `explicit_candidate_preview_does_not_save_until_requested` in its calibration editor tests.

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
