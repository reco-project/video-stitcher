# Reco Video Stitcher

**Open-source GPU-accelerated panoramic sports camera software. Desktop App with AI tracking, CLI, and OBS Plugin.**

[![CI](https://github.com/reco-project/video-stitcher/actions/workflows/rust.yml/badge.svg)](https://github.com/reco-project/video-stitcher/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

Stitch two camera feeds into a seamless panoramic sports view with AI-powered automatic camera control. No subscriptions, no vendor lock-in. Ships as a CLI, a Slint-based desktop app, and an OBS Studio source plugin that all consume the same Rust engine.

## Features

- **GPU-first pipeline** - Real-time stitching via wgpu 28 (Vulkan / Metal / DX12), 500+ fps on modern desktop GPUs
- **Zero-copy decode** - NVDEC (Linux/Windows) and VideoToolbox (macOS) frames go directly to GPU textures, no CPU bounce
- **AI ball + player tracking** - YOLO detection on ONNX (CPU / CUDA / CoreML / NCNN) or native TensorRT `.engine`, with anticipatory lookahead smoothing and density-peak player clustering
- **Automatic calibration** - Direct image alignment with optional lens profiles, estimated optics, independent overlap checks and IMU / audio temporal sync
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

# Preview a pre-stitched panorama using a cylinder calibration
./target/release/reco preview panorama.mp4 -c cylinder.json

# Live camera stitching (GStreamer, Jetson CSI)
./target/release/reco camera --left-device 0 --right-device 1 \
    -c match.json -o live.mkv --container mkv \
    --model yolo.engine --tracking field

# RTSP video inputs whose decoded media timestamps already share an origin
./target/release/reco stitch rtsp://camera-a/live rtsp://camera-b/live \
    -c match.json -o live.mkv --rtsp-fps 30 --rtsp-offsets-us=0,0

# Keep a recording while publishing the same encoded feed at media speed
./target/release/reco stitch left.mp4 right.mp4 -c match.json -o match.mp4 \
    --codec h264 --stream-url rtmp://127.0.0.1/live/example
```

RTSP input uses TCP. Compatible Linux CUDA/Vulkan jobs retain GPU images through synchronization and lookahead; other configurations and interactive GUI preview use host-backed images. `--no-zero-copy` requests host delivery without disabling hardware decoding. Supply one measured media timestamp offset per input; `0,0` is only correct for an already shared timeline, not independently started cameras. `stitch --help` exposes bounded wait, skew, reuse age and timeout controls. Unexpected disconnects, clock resets and expired images fail visibly instead of reconnecting or reporting success. `--audio-source 0` (the default) copies available audio from the first input; select another zero-based input index or `none`. Live audio uses that camera's existing connection and the selected video timeline, with bounded buffering and no mixing or transcoding. The output container must support the source audio codec; RTMP copy requires AAC or MP3. The loopback tests in `crates/reco-io/src/rtsp/tests.rs` cover source boundaries and A/V phase through startup and lookahead; its CLI integration test uses `RECO_TEST_CLI` to select the binary.

Optional RTMP/RTMPS publishing requires FFmpeg 8+ for safe audio/video queue recovery; bundled builds include a suitable runtime. Recording continues if optional publishing fails, and the CLI reports stream state separately. H.264 video and selected AAC/MP3 source audio share compressed packets; without source audio the stream gets silent AAC, not microphone capture. Network output is paced; ordinary file exports remain unrestricted. The loopback workflow in `crates/reco-io/tests/streaming.rs` checks shared packets, A/V timing and pacing. The FIFO recovery regression is in `crates/reco-io/src/ffmpeg/output_io/tests/overflow.rs`. External-service authentication and playback are separate acceptance checks.

In the GUI, load the rig calibration, then choose **Setup → Connect RTSP** and enter each camera's URL, timestamp offset and output frame rate. **Record / publish** uses the same processed-output job with optional source audio and RTMP/RTMPS publishing; file export also offers publishing. Preview pauses during output. **Stop and save** finalizes the live recording; reconnect explicitly to resume preview. Live input does not support seeking, file-frame sync offsets or quick recording. The GUI lifecycle tests in `crates/reco-gui/src/network/tests.rs` cover opening cancellation, stale results, polling and recorded audio; `crates/reco-gui/src/ui_tests.rs` covers form editing and publishing controls.

## Logging

Reco uses `RUST_LOG` for its own CLI/library logs. If `RUST_LOG` is unset,
the CLI defaults to `info`; set it explicitly when you want more or less
output.

```bash
# Show debug messages from Reco and dependencies
RUST_LOG=debug ./target/release/reco stitch left.mp4 right.mp4 -c match.json -o out.mp4

# Set 'info' level debug for default, 'debug' only for certain crates
RUST_LOG='info,reco_io=debug,reco_autocam=debug' \
    ./target/release/reco stitch ...

# Quieter normal run
RUST_LOG=warn ./target/release/reco stitch left.mp4 right.mp4 -c match.json -o out.mp4
```

Raw FFmpeg and libva messages are controlled separately because they write
outside Rust's logger. Reco keeps them quiet by default; opt in when debugging
codec, hardware decode, or driver setup issues:

```bash
# Show FFmpeg's own diagnostics
RECO_FFMPEG_LOG=debug ./target/release/reco stitch left.mp4 right.mp4 -c match.json -o out.mp4

# Linux: show libva driver messages again
LIBVA_MESSAGING_LEVEL=2 ./target/release/reco stitch left.mp4 right.mp4 -c match.json -o out.mp4
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

Push-based is the canonical ingestion path: consumers call `StitchCore::submit_frame` with a per-camera `FrameSet` each frame (interactive consumers use the `submit_frame_*_at_pose` variants). Batch file processing (`StitchSession::run`) is a thin pull-adapter on top.

## Building

Linux x86-64 desktop archives produced by the [release workflow](.github/workflows/release.yml) bundle FFmpeg and CPU ONNX Runtime. They target Ubuntu 24.04 or compatible systems with glibc 2.39; graphics drivers remain system dependencies. Keep the extracted directory together. Matching source archives include Reco, vendored Rust dependencies and native dependency sources. See the [SDK and packaging checks](scripts/SDK.md) for validation and platform boundaries. Source builds can use their own native FFmpeg installation; this desktop baseline does not apply to ARM/Jetson builds.

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
| `gopro` | reco-control | Optional OpenGoPro device helper (recording control) |

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
