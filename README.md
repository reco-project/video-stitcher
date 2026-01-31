<p align="center">
  <img src="https://raw.githubusercontent.com/reco-project/video-stitcher/main/electron/resources/icon.png" alt="Reco Logo" width="120" />
</p>

<h1 align="center">Reco Video Stitcher</h1>

<p align="center">
  <strong>Open-source, GPU-accelerated video stitching for sports filming</strong>
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
  An open-source alternative to <a href="https://www.veo.co/">Veo</a> — stitch two camera feeds into a seamless panoramic view without subscriptions.
</p>

---

## ✨ Features

- **GPU-Accelerated Stitching** — Real-time panoramic rendering using WebGL and Three.js shaders
- **Automatic Calibration** — Feature matching and position optimization for seamless blending
- **Lens Profile Support** — Pre-built profiles for GoPro, DJI, Insta360, Sony, and more
- **Works with Any Camera** — Use action cameras, DSLRs, or even mobile devices
- **Cross-Platform** — Native desktop app for Windows, macOS, and Linux
- **Auto-Updates** — Automatic update checks keep your app current with the latest features and fixes
- **No Subscriptions** — One-time setup, no recurring fees, your data stays local

## 🎬 How It Works

1. **Import Videos** — Select your left and right camera recordings
2. **Assign Lens Profiles** — Choose calibration profiles for each camera
3. **Process** — The app transcodes, extracts frames, and calibrates alignment
4. **View & Export** — Watch the stitched panorama in the built-in viewer

The processing pipeline combines backend video processing (FFmpeg + OpenCV) with frontend GPU rendering (Three.js) for optimal performance.

## 🚀 Getting Started

### Prerequisites

- **Node.js** 20+ and npm
- **Python** 3.10+ (3.11+ recommended)
- **FFmpeg** (automatically downloaded during setup)

### Installation

```bash
# Clone the repository
git clone https://github.com/reco-project/video-stitcher.git
cd video-stitcher

# Install all dependencies (frontend, backend, electron)
npm run setup

# Start the development environment
npm run dev
```

The app will launch with the Electron desktop interface, React frontend, and FastAPI backend running together.

### Building for Production

```bash
# Package the app for your platform
npm run electron-make
```

## 🏗️ Architecture

```
video-stitcher/
├── frontend/          # React + Three.js UI and GPU rendering
├── backend/           # FastAPI server for video processing
├── electron/          # Desktop app shell and system integration
├── docs/              # Documentation
└── scripts/           # Build and development utilities
```

| Component    | Technology               | Purpose                               |
| ------------ | ------------------------ | ------------------------------------- |
| **Frontend** | React, Three.js, Vite    | UI, WebGL stitching, frame extraction |
| **Backend**  | FastAPI, OpenCV, FFmpeg  | Video transcoding, feature matching   |
| **Desktop**  | Electron, Electron Forge | Cross-platform native app             |

## 📖 Documentation

- [Backend API](backend/README.md) — API endpoints and development guide
- [Telemetry](docs/TELEMETRY.md) — Privacy-focused, opt-in analytics
- [Releases & Auto-Updates](docs/RELEASES.md) — How releases work and auto-update system

## 🔒 Privacy & Telemetry

This app includes **optional, opt-in telemetry** to help improve the software:

- Disabled by default
- No personal data, filenames, or video content collected
- All data stored locally first
- Can be enabled/disabled anytime in Settings

See [TELEMETRY.md](docs/TELEMETRY.md) for full details.

## 📄 License

Licensed under **[AGPL-3.0](LICENSE)** — all derived versions must remain open-source.

## 🤝 Contributing

Contributions are welcome! Whether you're a developer, designer, coach, or camera enthusiast:

- 🐛 Report bugs and request features via [GitHub Issues](https://github.com/reco-project/video-stitcher/issues)
- 💬 Join the discussion on the [Reco Project Forum](https://forum.reco-project.org)
- 🔧 Submit pull requests for improvements

## 📬 Contact

- **Website:** [reco-project.org](https://reco-project.org)
- **Forum:** [forum.reco-project.org](https://forum.reco-project.org)
- **Email:** mohamedtahaguelzim@gmail.com
- **GitHub:** [reco-project/video-stitcher](https://github.com/reco-project/video-stitcher)

---

<p align="center">
  Made with ❤️ for the sports filming community
</p>
