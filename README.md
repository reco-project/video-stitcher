# Reco Video Stitcher

**Open-source GPU-accelerated panoramic sports camera software. Desktop App with AI tracking, CLI, and OBS Plugin.**

[![CI](https://github.com/reco-project/video-stitcher/actions/workflows/rust.yml/badge.svg)](https://github.com/reco-project/video-stitcher/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

Stitch two camera feeds into a seamless panoramic sports view with AI-powered automatic camera control. No subscriptions, no vendor lock-in. Ships as a CLI, a Slint-based desktop app, and an OBS Studio source plugin that all consume the same Rust engine.

## Features

- **GPU-first pipeline** - Real-time stitching via wgpu 28 (Vulkan / Metal / DX12), 500+ fps on modern desktop GPUs
- **Zero-copy decode** - NVDEC (Linux/Windows) and VideoToolbox (macOS) frames go directly to GPU textures, no CPU bounce
- **AI ball + player tracking** - YOLO detection on ONNX (CPU / CUDA / CoreML / NCNN) or native TensorRT `.engine`, with anticipatory lookahead smoothing and density-peak player clustering
- **Automatic calibration** - AKAZE feature matching with IMU / audio cross-correlation for temporal sync, Gyroflow lens profiles
- **Live production** - GStreamer camera ingest (Linux / Jetson CSI), push-based `StitchCore::submit_frame_*` for OBS and live streams, replay ring buffer, stacked-video pack/unpack
- **Cross-platform** - Linux / macOS / Windows desktop, NVIDIA Jetson Orin, cloud workers; mobile (iOS / Android) trait points land in this release, concrete impls follow

## Quick start

```bash
# Build the CLI
cargo build --release -p reco-cli

# Calibrate two cameras (AKAZE features + IMU/audio sync)
./target/release/reco calibrate left.mp4 right.mp4 -o match.json

# Stitch a panoramic video
./target/release/reco stitch left.mp4 right.mp4 -c match.json -o panorama.mp4

# Stitch with AI tracking (field mode: ball + players)
./target/release/reco stitch left.mp4 right.mp4 -c match.json -o tracked.mp4 \
    --model yolo.onnx --tracking field

# Pick a framing preset (broadcast | action | frame_all); --panner-config
# JSON overlays individual knobs on top
./target/release/reco stitch left.mp4 right.mp4 -c match.json -o tracked.mp4 \
    --model yolo.onnx --tracking field --panner-preset action

# Interactive preview with pan/zoom
./target/release/reco preview left.mp4 right.mp4 -c match.json

# Live camera stitching (GStreamer, Jetson CSI)
./target/release/reco camera --left-device 0 --right-device 1 \
    -c match.json -o live.mkv --container mkv \
    --model yolo.engine --tracking field
```

## Architecture

Nine Rust crates. Strict dependency direction keeps the engine reusable as a library.

```
reco-core        GPU stitching engine (wgpu). No I/O, no domain logic.
reco-io          FFmpeg decode/encode, GStreamer cameras, StitchJob API, stacked-video pack/unpack.
reco-detect      Inference backends: ORT (CPU/CUDA/CoreML), NCNN, native TensorRT.
reco-autocam     Panners (field, ball, sweep), lookahead smoothing, ROI filters.
reco-calibrate   AKAZE features, stereo optimization, lens database, live-calibration source trait.
reco-control     Operator intent vocabulary (keyboard today; gopro/mobile/websocket scaffolds).
reco-cli         Terminal consumer: stitch / calibrate / preview / camera / analyze / info.
reco-gui         Slint desktop consumer with wgpu preview + export UI.
reco-obs         OBS Studio source plugin (async-frame ingestion, BGRA, interactive pan/zoom).
```

**Dependency direction:** consumers (`cli` / `gui` / `obs`) depend on the four library crates (`autocam`, `calibrate`, `detect`, `io`); all four depend on `reco-core`. `reco-control` is consumed by `cli` / `gui` / `obs`.

Push-based is the canonical ingestion path: consumers call `StitchCore::submit_frame_yuv` / `submit_frame_bgra` per frame. Batch file processing (`StitchSession::run`) is a thin pull-adapter on top.

## Building

Prerequisites: Rust **1.92+** (workspace MSRV), FFmpeg development libraries, pkg-config, clang.

```bash
# Ubuntu / Debian (apt)
sudo apt install libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libavdevice-dev libavfilter-dev libswresample-dev \
    pkg-config clang

# macOS (Homebrew)
brew install ffmpeg pkg-config

# Windows: FFmpeg 7.x binaries + LLVM/Clang, set FFMPEG_DIR env var

# Build everything
cargo build --release
```

### Feature flags

| Feature | Crate | Purpose |
|---|---|---|
| `ort` (default) | reco-detect | ONNX Runtime inference (CPU + CUDA + CoreML EPs) |
| `tensorrt-native` | reco-detect | Native TensorRT for `.engine` files (Jetson, no ORT glibc dep) |
| `ncnn` | reco-detect | NCNN backend for mobile / embedded |
| `load-dynamic` | reco-detect | Dynamically load `libonnxruntime.so` at runtime |
| `gstreamer` | reco-io | GStreamer camera ingest (Linux / Jetson CSI) |
| `stacked-output` | reco-io | FFmpeg-backed stacked-video encoder / source |
| `profiling` | workspace | `tracing` + `tracing-chrome` instrumentation, zero-cost when off |
| `keyboard` (default) | reco-control | Keyboard transport for operator intents |
| `gopro` / `mobile` / `websocket` | reco-control | Placeholder transports for future work |

Example combined build:

```bash
cargo build --release -p reco-cli --features tensorrt-native,gstreamer,profiling
```

## Development

```bash
cargo test --all              # Run all tests
cargo clippy --all-targets -- -D warnings   # Lint (zero warnings policy)
cargo fmt --all -- --check    # Format check
cargo doc --no-deps --open    # Generate docs
cargo deny check              # Supply-chain gates (advisories / licenses / bans / sources)
```

Profiling a 300-frame stitch:

```bash
cargo run --release -p reco-cli --features profiling -- \
    stitch left.mp4 right.mp4 -c match.json -o out.mp4 --max-frames 300
# Produces reco-trace.json — open in https://ui.perfetto.dev
```

## Privacy

Reco collects no usage data, sends no telemetry, and makes no network requests except those you explicitly configure (RTMP streaming, etc.). All performance statistics shown in the Stats panel are computed locally in memory and never leave your machine.

## License

**AGPL-3.0-only.** See [LICENSE](LICENSE). Contributor terms are in [CONTRIBUTING.md](CONTRIBUTING.md#contributor-license-agreement): by submitting a PR you grant the maintainer the right to relicense contributions, which keeps dual-licensing on the table for future commercial distribution.

Questions about commercial licensing can be directed to the project maintainer; there is no public paid tier today.

## Community

- [Website](https://www.reco-project.org) - Landing page
- [Forum](https://forum.reco-project.org) - Questions, showcase, feature requests
- [Issues](https://github.com/reco-project/video-stitcher/issues) - Bug reports + feature tracking
- [v1 archive](https://github.com/reco-project/video-stitcher-v1) - Legacy Electron / Python version (predecessor to this Rust rewrite)
