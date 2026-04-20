# reco-io

Pluggable frame I/O backends + end-to-end orchestration for `reco-core`.

## What it owns

- **FFmpeg decode / encode** (default). NVDEC zero-copy on Linux/Windows, VideoToolbox zero-copy on macOS, NVENC / h264_v4l2m2m / libx264 / libx265 encoders.
- **GStreamer camera ingest** (`gstreamer` feature) — `nvarguscamerasrc` CSI pipelines for Jetson, V4L2 for desktop cameras, dual-source NV12 appsink.
- **`StitchJob`** — one-call batch orchestrator: source, core, encoder, director, replay, sinks. Consumed by reco-cli and reco-gui for file-to-file processing.
- **Stacked-video** (`stacked-output` feature) — `GridLayout` + `pack_yuv420p` / `unpack_yuv420p` CPU primitives, `StackedEncoder` (ffmpeg-backed), `StackedSource` reader, `ReplayRecordingSource` decorator, `GpuAtlasRecorder` sink for GPU-resident replay.
- **Settings persistence** (`config` feature) — opt-in per-user preferences via `directories` + serde.

## Dependencies

Depends on `reco-core`. No other workspace crates.

## Build

```bash
cargo build -p reco-io
cargo build -p reco-io --features gstreamer,stacked-output,config,profiling
```
