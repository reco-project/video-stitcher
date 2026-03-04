<p align="center">
  <img src="https://raw.githubusercontent.com/reco-project/video-stitcher/main/electron/resources/icon.png" alt="Reco Logo" width="120" />
</p>

<h1 align="center">Reco Video Stitcher</h1>

<p align="center">
  <strong>Open-source panoramic sports camera software</strong>
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
  A GPU-first open-source solution for sports filming</a>. Stitch two camera feeds into a seamless panoramic sports view without subscriptions.
</p>

---

## Why Reco Exists

Modern sports camera systems like Veo and Pixellot provide powerful automated filming, but they are often expensive and tied to proprietary hardware and subscriptions.

Reco aims to explore a different approach:

- open-source software
- hardware flexibility
- local processing
- community-driven innovation

The long-term goal is to build an **open platform for sports video capture and analysis**.

---

## Project Status

Reco Video Stitcher is currently undergoing a [**major rewrite**](https://forum.reco-project.org/t/v2-rebuilding-for-something-better/10)

The current version (Electron + React + Python) was built as an **early prototype** to validate the concept of an open-source sports camera system.

While the prototype works, the architecture makes it difficult to extend and maintain.

Because of this, development is shifting toward **Video Stitcher v2**, a new architecture built in **Rust with a GPU-first pipeline**. This transition had been planned from the beginning, even before the v1 prototype.

The goal of v2 is to provide:

- significantly faster processing
- a cleaner media pipeline
- real-time stitching capabilities
- a better foundation for future features like tracking and live streaming

The existing Python/Electron version should be considered **experimental and unstable**.

Future development will focus on the new Rust-based engine.

Follow development progress and discussions on the [forum](https://forum.reco-project.org).

---

## Features (Prototype v1)

- **GPU-Accelerated Stitching** — Real-time panoramic rendering using WebGL and Three.js shaders
- **Automatic Calibration** — Feature matching and position optimization for seamless blending
- **Lens Profile Support** — Pre-built profiles for GoPro, DJI, Insta360, Sony, and more
- **Works with Any Camera** — Use action cameras, DSLRs, or even mobile devices
- **Cross-Platform Desktop App** — Windows, macOS, and Linux
- **No Subscriptions** — One-time setup, your data stays local

---

## How It Works

1. **Import Videos** — Select your left and right camera recordings
2. **Assign Lens Profiles** — Choose calibration profiles for each camera
3. **Process** — The app transcodes, extracts frames, and calibrates alignment
4. **View & Export** — Watch the stitched panorama in the built-in viewer

The processing pipeline combines backend video processing (FFmpeg + OpenCV) with frontend GPU rendering (Three.js) for optimal performance.

---

## Video Stitcher v2 (In Development)

The next generation of Reco Video Stitcher is being built around a **Rust-based GPU pipeline**.

Core technologies:

| Layer               | Technology |
| ------------------- | ---------- |
| Core engine         | Rust       |
| GPU rendering       | wgpu       |
| Desktop application | Tauri      |
| CLI tool            | `reco`     |

The new architecture will look like:

```

decode → synchronize → stitch → render → encode

```

This design enables:

- faster exports
- real-time stitching
- improved stability
- easier future extensions
- better cross-platform support

Development is currently in the early stages.

---

## Roadmap

**v1 (prototype)**  
Electron + React + Python implementation used to validate the concept.

**v2 (in development)**  
Rust-based stitching engine using wgpu and a Tauri desktop application.

**Future directions**

- automatic ball tracking
- live streaming
- mobile companion app
- plugin ecosystem

---

## Getting Started (Prototype v1)

### Prerequisites

- **Node.js** 20+ and npm
- **Python** 3.10+ (3.11+ recommended)
- **FFmpeg** (should be accessible through PATH)

### Installation

```bash
git clone https://github.com/reco-project/video-stitcher.git
cd video-stitcher

npm run setup
npm run dev
```

The app will launch with the Electron desktop interface, React frontend, and FastAPI backend running together.

---

## Architecture (Prototype v1)

```
video-stitcher/
├── frontend/          # React + Three.js UI and GPU rendering
├── backend/           # FastAPI server for video processing
├── electron/          # Desktop app shell and system integration
├── docs/              # Documentation
└── scripts/           # Build and development utilities
```

| Component | Technology              | Purpose                           |
| --------- | ----------------------- | --------------------------------- |
| Frontend  | React, Three.js, Vite   | UI, WebGL stitching               |
| Backend   | FastAPI, OpenCV, FFmpeg | Video transcoding and calibration |
| Desktop   | Electron                | Cross-platform application        |

---

## Documentation

- [Backend API](backend/README.md) — API endpoints and development guide
- [Telemetry](docs/TELEMETRY.md) — Privacy-focused, opt-in analytics
- [Releases & Auto-Updates](docs/RELEASES.md) — Release workflow and updates

---

## Privacy & Telemetry

This app can include **optional, opt-in telemetry** to help improve the software:

- Disabled by default
- No personal data, filenames, or video content collected
- All data stored locally first
- Can be enabled or disabled anytime in Settings

See [TELEMETRY.md](docs/TELEMETRY.md) for full details.

---

## License

Licensed under **AGPL-3.0** — all derived versions must remain open-source.

See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome!

Whether you're a developer, designer, coach, or camera enthusiast:

- 🐛 Report bugs or request features via [GitHub Issues](https://github.com/reco-project/video-stitcher/issues)
- 💬 Join discussions on the [Reco Project Forum](https://forum.reco-project.org)
- 🔧 Submit pull requests

---

## Contact

- **Website:** [https://reco-project.org](https://reco-project.org)
- **Forum:** [https://forum.reco-project.org](https://forum.reco-project.org)
- **Email:** [mohamedtahaguelzim@gmail.com](mailto:mohamedtahaguelzim@gmail.com)
- **GitHub:** [https://github.com/reco-project/video-stitcher](https://github.com/reco-project/video-stitcher)

---

<p align="center">
  ⭐ Star the project to follow the development of the Rust rewrite
</p>

<p align="center">
  Made with ❤️ for the sports filming community
</p>
