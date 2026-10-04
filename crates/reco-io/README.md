# reco-io

Pluggable frame I/O backends + end-to-end orchestration for `reco-core`.

## What it owns

- **FFmpeg decode / encode** (`ffmpeg`, default). Hardware codecs remain available without wgpu; FFmpeg chooses and opens its own devices.
- **Native textures and GPU stitching** (`gpu`, default). CUDA/Vulkan imports on Linux, D3D11VA/DX12 on Windows and VideoToolbox/Metal on macOS. Disabling this feature retains host frame I/O and CPU stitching, not native texture delivery.
- **GStreamer camera ingest** (`gstreamer` feature) — `nvarguscamerasrc` CSI pipelines for Jetson, V4L2 for desktop cameras, dual-source NV12 appsink.
- **`StitchJob`** — one-call batch orchestrator: source, core, encoder, panner, replay, sinks. Consumed by reco-cli and reco-gui for file-to-file processing.
- **Stacked-video** (`stacked-output` feature) — `GridLayout` + `pack_yuv420p` / `unpack_yuv420p` CPU primitives, `StackedEncoder` (ffmpeg-backed), `StackedSource` reader, `ReplayRecordingSource` decorator, `GpuAtlasRecorder` sink for GPU-resident replay.
- **Settings persistence** (`config` feature) — opt-in per-user preferences via `directories` + serde.

## Dependencies

Depends on `reco-core`. No other workspace crates. Disable default features to omit wgpu; enable only the required I/O backends. The isolated `reco-io wgpu-free` lanes in `.github/workflows/rust.yml` check this dependency boundary and run host behavior tests.

## Build

```bash
cargo build -p reco-io
cargo build -p reco-io --features gstreamer,stacked-output,config,profiling
cargo build -p reco-io --no-default-features --features ffmpeg,stacked-output
```
