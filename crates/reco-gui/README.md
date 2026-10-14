# reco-gui

Slint 1.15 desktop consumer with wgpu 28 zero-copy preview.

## What you get

- Interactive panorama preview with pan/zoom via mouse + keyboard, driven by `reco-control::PoseControl` and core's shared render-space coverage constraints.
- One Setup/Adjust inspector for sources, calibration, tracking region and camera/lens controls, with dark and light themes. Loaded file details collapse to keep calibration and field setup reachable on small windows; live inputs have their own connection summary and preview controls.
- Calibration with progress and match-support reporting. Match support is not an optical-quality score.
- Single-camera lens preview and four-corner field selection directly in the stitched preview. Drag corners to adjust, drag elsewhere to pan, and scroll to zoom. Apply accepts the region; Save persists it with the calibration.
- Export job orchestration: source selection, output codec / quality / container, optional AI tracking, replay recording, progress bar, cancelable.
- Persisted recent files, export and recording defaults, model path, window geometry, preview aspect and theme (`src/settings.rs`).

## Render bridge

Slint on wgpu 28 drives its own render loop. The `BeforeRendering` notifier renders the preview on Slint's shared device, and the resulting texture is imported without CPU readback (`src/main.rs`, `src/preview.rs`). Display pacing depends on the windowing backend and presentation environment.

Quick recording is a fixed-rate capture path, not an exact screen recording. Missing interactive frames can compress elapsed time. Use Export for source-timeline processing; the outstanding capture boundaries are described in [FRICTION.md](FRICTION.md).

`src/ui_tests.rs` exercises layout, visible notifications, dialog focus, keyboard shortcuts and corner letterboxing without a desktop session. `src/ui_tests/workflow.rs` covers file/live setup, active-work guards, corner actions and live output without file-trim controls at small and normal window sizes. `src/field_editor.rs` tests source-anchored edits, unsupported picks, Apply/Cancel and explicit save. `src/calibration_edit_tests.rs` checks document acceptance, background-result freshness, failed saves and preservation of unsaved edits through source changes and preview restoration. GPU lens-preview mapping is checked against rendered landmarks in `reco-core/src/lens/preview.rs`.

## Build

```bash
cargo build --release -p reco-gui
cargo build --release -p reco-gui --features tensorrt-native   # Jetson desktop
cargo build --release -p reco-gui --features profiling         # tracing
./target/release/reco-gui
```

## License note

Slint is tri-licensed (GPL-3.0-only / royalty-free / commercial). reco-gui distributes under the Slint royalty-free path for desktop, not GPL. See [../../deny.toml](../../deny.toml) for the workspace license gate.
