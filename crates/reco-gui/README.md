# reco-gui

Slint 1.15 desktop consumer with wgpu 28 zero-copy preview.

## What you get

- Interactive panorama preview with pan/zoom via mouse + keyboard, driven by `reco-control::PoseControl` and core's shared render-space coverage constraints.
- One Setup/Adjust inspector for sources, calibration, camera framing and lens controls, with dark and light themes.
- Calibration with progress and match-support reporting. Match support is not an optical-quality score.
- Single-camera lens preview with projected ROI vertices. Region editing opens a browser editor; paste its result into the workspace.
- Export job orchestration: source selection, output codec / quality / container, optional AI tracking, replay recording, progress bar, cancelable.
- Persisted recent files, export and recording defaults, model path, window geometry, preview aspect and theme (`src/settings.rs`).

## Render bridge

Slint on wgpu 28 drives its own render loop. The `BeforeRendering` notifier renders the preview on Slint's shared device, and the resulting texture is imported without CPU readback (`src/main.rs`, `src/preview.rs`). Display pacing depends on the windowing backend and presentation environment.

Quick recording is a fixed-rate capture path, not an exact screen recording. Missing interactive frames can compress elapsed time. Use Export for source-timeline processing; the outstanding capture boundaries are described in [FRICTION.md](FRICTION.md).

`src/ui_tests.rs` exercises layout, visible notifications, dialog focus, keyboard shortcuts and ROI letterboxing without a desktop session. `src/calibration_edit_tests.rs` checks document acceptance, background-result freshness, failed saves and preservation of unsaved edits through source changes and preview restoration. GPU lens-preview mapping is checked against rendered landmarks in `reco-core/src/lens/preview.rs`.

## Build

```bash
cargo build --release -p reco-gui
cargo build --release -p reco-gui --features tensorrt-native   # Jetson desktop
cargo build --release -p reco-gui --features profiling         # tracing
./target/release/reco-gui
```

## License note

Slint is tri-licensed (GPL-3.0-only / royalty-free / commercial). reco-gui distributes under the Slint royalty-free path for desktop, not GPL. See [../../deny.toml](../../deny.toml) for the workspace license gate.
