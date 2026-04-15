# Reco Video Stitcher

**Open-source GPU-accelerated panoramic sports camera software.**

[![CI](https://github.com/reco-project/video-stitcher/actions/workflows/rust.yml/badge.svg)](https://github.com/reco-project/video-stitcher/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

Stitch two camera feeds into a seamless panoramic sports view with AI-powered automatic camera control. No subscriptions, no vendor lock-in.

## Features

- **GPU-first pipeline** - Real-time stitching via wgpu (Vulkan/Metal/DX12), 500+ fps on modern GPUs
- **Zero-copy decode** - NVDEC/VideoToolbox frames go directly to GPU textures, no CPU copies
- **AI ball tracking** - YOLO-based detection with One Euro trajectory smoothing
- **Field tracking** - DBSCAN player clustering for broadcast-style camera motion
- **Automatic calibration** - AKAZE feature matching with IMU/audio temporal sync
- **Cross-platform** - Linux, macOS, Windows. Desktop, NVIDIA Jetson, cloud

## Quick start

```bash
# Build
cargo build --release -p reco-cli

# Calibrate two cameras
reco calibrate left.mp4 right.mp4 -o match.json

# Stitch a panoramic video
reco stitch left.mp4 right.mp4 -c match.json -o panorama.mp4

# Stitch with AI tracking (requires ONNX model)
reco stitch left.mp4 right.mp4 -c match.json -o tracked.mp4 \
    --model ball.onnx --tracking field

# Interactive preview
reco preview left.mp4 right.mp4 -c match.json

# Live camera stitching (GStreamer, Jetson)
reco camera /dev/video0 /dev/video1 -c match.json -o live.mp4
```

## Architecture

Five Rust crates with clean separation of concerns:

```
reco-core        GPU stitching engine (wgpu). No I/O, no domain logic.
reco-io          FFmpeg decode/encode, GStreamer cameras, StitchJob API.
reco-autocam     YOLO detector, ball/field directors, trajectory smoothing.
reco-calibrate   AKAZE features, stereo optimization, lens database.
reco-cli         Thin CLI binary. Each command is ~100 lines.
```

**Dependency direction:** `cli -> {autocam, io, calibrate} -> core`

The core engine is usable as a standalone Rust library. GUI apps, OBS plugins, and cloud services consume the same API as the CLI.

## Building

Prerequisites: Rust 1.88+, FFmpeg development libraries, pkg-config.

```bash
# Ubuntu/Debian
sudo apt install libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libavdevice-dev libavfilter-dev libswresample-dev \
    pkg-config clang

# macOS
brew install ffmpeg pkg-config

# Build all crates
cargo build --release

# Optional features
cargo build --release --features tensorrt   # TensorRT GPU detection
cargo build --release --features gstreamer  # Live camera support
cargo build --release --features profiling  # Tracing instrumentation
```

## Development

```bash
cargo test --all              # Run all tests
cargo clippy --all-targets -- -D warnings   # Lint
cargo fmt --all -- --check    # Format check
cargo doc --no-deps --open    # Generate docs
```

## License

AGPL-3.0-only. See [LICENSE](LICENSE).

## Community

- [Forum](https://discourse.reco-project.org) - Questions, showcase, feature requests
- [Issues](https://github.com/reco-project/video-stitcher/issues) - Bug reports
- [v1 archive](https://github.com/reco-project/video-stitcher-v1) - Legacy Electron/Python version
