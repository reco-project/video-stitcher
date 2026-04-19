# reco-gui

Slint 1.15 desktop consumer with wgpu 28 zero-copy preview.

## What you get

- Interactive panorama preview with pan/zoom via mouse + keyboard (routed through `reco-control::KeyboardTransport` and `reco-core::PoseControl`).
- Calibration wizard with live progress (AKAZE matches, reprojection error, confidence).
- Export job orchestration: source selection, output codec / quality / container, optional AI tracking, replay recording, progress bar, cancelable.
- Session persistence: recent files, last-used calibration JSON, FOV / director / detection interval settings.

## Render bridge

Slint on wgpu 28 drives its own render loop; reco-gui sources panorama frames via the `BeforeRendering` notifier (not a timer) to stay vsync-paced. The preview is a wgpu texture imported into Slint's render thread.

## Build

```bash
cargo build --release -p reco-gui
cargo build --release -p reco-gui --features tensorrt-native   # Jetson desktop
cargo build --release -p reco-gui --features profiling         # tracing
./target/release/reco-gui
```

## License note

Slint is tri-licensed (GPL-3.0-only / royalty-free / commercial). reco-gui distributes under the Slint royalty-free path for desktop, not GPL. See [../../deny.toml](../../deny.toml) for the workspace license gate.
