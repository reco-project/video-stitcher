# video-stitcher

An open-source, GPU-accelerated alternative to [Veo](https://www.veo.co/) for affordable sports filming.

**Status:** Imminent beta release  
**License:** AGPL-3.0

---

## Overview

**video-stitcher** stitches two camera feeds into a seamless panoramic view — similar to Veo, but open-source and without subscriptions.

Tested with GoPros and designed to support various cameras, including mobile devices.

**Tech stack:**

- **Frontend:** React + Three.js (GPU-accelerated WebGL stitching)
- **Backend:** Python FastAPI (OpenCV + FFmpeg)
- **Platforms:** Windows, macOS, Linux

**Features:**

- Real-time stitching with GPU acceleration
- Automatic camera calibration
- Works with any camera setup
- Recording and playback controls

More context: [Reddit post](https://www.reddit.com/r/VeoCamera/comments/1nr0ic7/how_would_you_design_your_veo/)

---

## Getting Started

### Prerequisites

- **Node.js** 20+ and npm
- **Python** 3.9+ (3.11+ recommended)
- **FFmpeg** installed and available in PATH

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/reco-project/video-stitcher.git
    cd video-stitcher
    ```

2. Install dependencies:

    ```bash
    npm run setup
    ```

3. Run the development environment:
    ```bash
    npm run dev
    ```

The app will start with frontend, backend, and Electron desktop app.

---

## Telemetry

This app collects minimal anonymous usage data to help improve the software (e.g., feature usage, errors). No personal information or video content is collected. Telemetry is **opt-in** and disabled by default. You can enable it in Settings.

---

## License

Licensed under **AGPL-3.0** — all derived versions must remain open-source.

---

## Community & Contributions

Contributions, feedback, and testing are welcome from developers, designers, coaches, and camera enthusiasts.

Guidelines will be released soon, along with a dedicated community forum.

---

## Feedback

Share feedback in the [Reddit thread](https://www.reddit.com/r/VeoCamera/comments/1nr0ic7/how_would_you_design_your_veo/) or contact mohamedtahaguelzim@gmail.com
