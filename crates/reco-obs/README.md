# reco-obs

OBS Studio source plugin. Embeds the reco stitching engine inside an OBS source so live producers get a panoramic feed without a separate pipeline.

## What it does

- Accepts async I420 or packed BGRA/BGRX/RGBA camera sources. Choose the matching input format in source properties. Each camera's dimensions come from its frames, not manual settings; the cameras may have different sizes. Input-layout validation is exercised by `pipeline_uses_each_camera_size_and_rejects_invalid_replacements` in `src/source_tests.rs`.
- Loads a calibration JSON file and supports drag-to-pan and wheel-to-zoom through OBS's Interact window.
- Optional stacked-video recording (`--features replay`) follows OBS Record/Stream by default when the frontend API is available. Independent recording is selectable in source properties. This records camera pairs, not the composited OBS output.

Stacked recording requires equal-sized I420 camera images and uses OBS's configured frame rate. An incompatible pair leaves stitching running without a raw recording. Choose a new `.mkv` path for each recording; existing files are never overwritten. Output-only resizing keeps the recorder attached, while an input-layout or calibration rebuild stops it. Changing the path finalizes the previous file. Lifecycle regressions are in `src/source_tests.rs`. Missing camera ticks are not duplicated to preserve elapsed time, so this raw recording is not a screen capture.

Open integration gaps are tracked in [FRICTION.md](FRICTION.md).

## Build

```bash
cargo build --release -p reco-obs
```

Output: `libreco_obs.so` (Linux), `reco_obs.dll` (Windows), `libreco_obs.dylib` (macOS). Drop into OBS's `obs-plugins/64bit/` directory.

## Not yet shipped

- NV12 input and GPU sharing with OBS
- AI auto-pan from OBS controls
- Sync-source support (Browser / Screen / WebRTC)
- Keyboard/controller mapping and live calibration capture

## Architecture

`StitchCore` owns rendering and coverage. UI callbacks accumulate pending input; video callbacks resolve it and publish a CPU RGBA image for upload to an OBS texture. Interaction, stitching and presentation have separate ownership so UI input does not wait for rendering. Callback ownership and pending-input behavior are exercised by `src/source_tests.rs` and `src/interaction.rs` tests. Strict coverage preparation still blocks video ticking after geometry changes.

Source selection rejects unsupported source types, including Reco itself, and retries cameras that are created later. Its standalone libobs regression can be run in a desktop session:

```bash
cargo test -p reco-obs native_selection_rejects_self_and_releases_late_sources -- --ignored --test-threads=1
```
