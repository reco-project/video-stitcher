<p align="center">
  <img src="https://raw.githubusercontent.com/reco-project/video-stitcher/main/electron/resources/icon.png" alt="Reco Logo" width="120" />
</p>

<h1 align="center">Reco Video Stitcher</h1>

<p align="center">
  <strong>Open-source GPU-accelerated panoramic sports camera software</strong>
</p>

<p align="center">
  <a href="https://github.com/reco-project/video-stitcher/releases">
    <img src="https://img.shields.io/github/v/release/reco-project/video-stitcher?style=flat-square" alt="Release" />
  </a>
  <a href="https://github.com/reco-project/video-stitcher/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" alt="License" />
  </a>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=flat-square" alt="Platform" />
</p>

<p align="center">
  Stitch two camera feeds into a seamless panoramic sports view — no subscriptions, no vendor lock-in.
</p>

---

## Why Reco Exists

Sports camera systems like Veo and Pixellot provide powerful automated filming, but they come with expensive hardware, recurring subscriptions, and proprietary lock-in.

Reco takes a different approach:

- **Open source** — the code is yours, forever
- **Hardware flexible** — use GoPros, action cameras, DSLRs, or any pair of cameras
- **Local processing** — your footage stays on your machine
- **Community driven** — built by and for the sports filming community

The long-term goal is an **open platform for sports video capture and analysis**.

---

## v2 — The Rust Engine

Reco v2 is a ground-up rewrite in **Rust** with a **GPU-first pipeline** using [wgpu](https://wgpu.rs). It replaces the original Electron/Python prototype with a fast, portable, and maintainable architecture.

### What it does today

- **GPU-accelerated stitching** — fisheye undistortion, depth-tested compositing, and viewport rendering on the GPU
- **Hardware video encoding** — auto-detects NVENC, QSV, VideoToolbox, VAAPI; falls back to libx264
- **Multithreaded decode** — parallel left/right video decoding with frame synchronization
- **Interactive preview** — real-time preview window with mouse drag-to-pan, scroll-to-zoom, and smooth camera animation
- **CLI-first** — scriptable `reco stitch` and `reco preview` commands
- **Profiling built in** — opt-in `tracing-chrome` instrumentation for performance analysis with [Perfetto](https://ui.perfetto.dev)

### Architecture

```
left.mp4  ──► decode (thread) ──┐
                                ├──► pair ──► GPU stitch ──► encode ──► output.mp4
right.mp4 ──► decode (thread) ──┘
```

| Crate | Purpose |
|---|---|
| `reco-core` | GPU stitching engine (wgpu) — usable as a standalone library |
| `reco-ffmpeg` | FFmpeg decode/encode with hardware acceleration |
| `reco-cli` | CLI binary: `reco stitch`, `reco preview`, `reco info` |

### Quick start (v2)

**Prerequisites:** Rust toolchain, FFmpeg development libraries

```bash
# Ubuntu/Debian
sudo apt install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libavdevice-dev libavfilter-dev libswresample-dev pkg-config clang

# macOS
brew install ffmpeg pkg-config

# Build
git clone https://github.com/reco-project/video-stitcher.git
cd video-stitcher
git checkout v2
cargo build --release
```

### Usage

```bash
# Stitch two videos into a panorama
cargo run --release -p reco-cli -- stitch left.mp4 right.mp4 \
  --calibration match.json -o output.mp4

# Interactive preview with pan/zoom
cargo run --release -p reco-cli -- preview left.mp4 right.mp4 \
  --calibration match.json

# Show GPU and encoder info
cargo run --release -p reco-cli -- info
```

**Preview controls:**

| Input | Action |
|---|---|
| Mouse drag | Pan (yaw/pitch) |
| Scroll wheel | Zoom (FOV) |
| Arrow keys | Pan (keyboard) |
| `+` / `-` | Zoom (keyboard) |
| Space | Play / pause |
| N | Step one frame |
| P | Skip 30 frames |
| Q / Escape | Quit |

### Development

```bash
cargo build                   # Build all crates
cargo test --all              # Run tests
cargo clippy --all-targets -- -D warnings   # Lint (zero warnings policy)
cargo fmt --all -- --check    # Format check
cargo doc --no-deps --open    # Browse API documentation

# Profile a stitch run (outputs reco-trace.json, open in ui.perfetto.dev)
cargo run --release -p reco-cli --features profiling -- stitch \
  left.mp4 right.mp4 -c match.json -o out.mp4 --max-frames 300
```

---

## v1 — The Prototype (Legacy)

The original version (Electron + React + Python) validated the concept of an open-source sports camera system. It works, but the architecture made it difficult to extend and optimize.

v1 is on the `main` branch and should be considered **experimental**. Future development focuses entirely on the Rust-based v2 engine.

<details>
<summary>v1 details</summary>

### Features

- GPU stitching via WebGL and Three.js shaders
- Automatic calibration with feature matching
- Lens profiles for GoPro, DJI, Insta360, Sony, and more
- Cross-platform desktop app (Windows, macOS, Linux)

### Getting started

```bash
# Prerequisites: Node.js 20+, Python 3.10+, FFmpeg in PATH
git clone https://github.com/reco-project/video-stitcher.git
cd video-stitcher
npm run setup
npm run dev
```

### Architecture

| Component | Stack | Purpose |
|---|---|---|
| Frontend | React, Three.js, Vite | UI and WebGL stitching |
| Backend | FastAPI, OpenCV, FFmpeg | Video processing and calibration |
| Desktop | Electron | Cross-platform shell |

</details>

---

## Roadmap

**v2 engine** (current)
- [x] GPU stitching pipeline (wgpu)
- [x] Hardware-accelerated encode/decode
- [x] Interactive preview window
- [x] Profiling infrastructure
- [ ] Ctrl-C resilience and crash-safe output
- [ ] Desktop GUI (Tauri)
- [ ] AI-powered auto-tracking director

**Future**
- Automatic ball/player tracking
- Live streaming
- Mobile companion app
- Cloud processing
- Plugin ecosystem

---

## Community

- **Forum:** [forum.reco-project.org](https://forum.reco-project.org) — discussions, feature requests, showcase
- **Website:** [reco-project.org](https://reco-project.org)
- **Issues:** [GitHub Issues](https://github.com/reco-project/video-stitcher/issues)

Contributions welcome — whether you're a developer, coach, or camera enthusiast.

---

## License

Licensed under **AGPL-3.0** — all derived versions must remain open-source.

See [LICENSE](LICENSE) for details.

---

<p align="center">
  <a href="https://github.com/reco-project/video-stitcher">Star the repo</a> to follow the development of v2
</p>

<p align="center">
  Made for the sports filming community
</p>
