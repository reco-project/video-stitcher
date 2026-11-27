# reco-io

Pluggable frame I/O backends + end-to-end orchestration for `reco-core`.

## What it owns

- **FFmpeg decode / encode** (`ffmpeg`, default). Hardware codecs remain available without wgpu; FFmpeg chooses and opens its own devices.
- **Native textures and GPU stitching** (`gpu`, default). CUDA/Vulkan imports on Linux, D3D11VA/DX12 on Windows and VideoToolbox/Metal on macOS. Disabling this feature retains host frame I/O and CPU stitching, not native texture delivery.
- **GStreamer camera ingest** (`gstreamer` feature) — `nvarguscamerasrc` CSI pipelines for Jetson, V4L2 for desktop cameras, dual-source NV12 appsink.
- **`StitchJob`** — one-call batch orchestrator: source, core, encoder, panner, replay, sinks. Consumed by reco-cli and reco-gui for file-to-file processing.
- **Stacked-video** (CPU pack/unpack primitives unconditional; FFmpeg wrappers under `ffmpeg`, GPU atlas recorder under `ffmpeg` + `gpu`) - `GridLayout` + `pack_yuv420p` / `unpack_yuv420p`, `StackedEncoder`, `StackedSource` reader, `ReplayRecordingSource` decorator, `GpuAtlasRecorder` sink for GPU-resident replay.
- **Settings persistence** (unconditional) - per-user preferences via `directories` + serde (`settings` module).
- **Capture selection**: `frame_sync::Selector` pairs timestamped image handles with explicit wait, skew and reuse limits. A caller can report a missing output slot and call `skip_pending` after its deadline without flushing future captures. Ordering, owner release and recovery are exercised in `src/frame_sync/tests.rs`; this does not change a source adapter's error policy.

## Dependencies

Depends on `reco-core`. No other workspace crates. Disable default features to omit wgpu; enable only the required I/O backends. The isolated `reco-io wgpu-free` lanes in `.github/workflows/rust-extended.yml` check this dependency boundary and run host behavior tests.

## Build

```bash
cargo build -p reco-io
cargo build -p reco-io --features gstreamer,profiling
cargo build -p reco-io --no-default-features --features ffmpeg
```
