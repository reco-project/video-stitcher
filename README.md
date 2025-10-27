# video-stitcher  
### An open-source, GPU-accelerated alternative to [Veo](https://www.veo.co/) for affordable sports filming

**Status:** Imminent beta release  
**License:** AGPL-3.0  
**Progress:** see [timeline](#development-progress) below

---

## Overview

**video-stitcher** is an experimental open-source project aimed at making sports filming affordable and accessible to everyone — especially small clubs.  
It can **live-stitch two camera feeds directly in the browser** (after calibration), producing a seamless panoramic view similar to Veo or Veo Go — but without subscriptions or proprietary hardware.

The project has been tested using two GoPros and is designed to support a wide range of cameras, including mobile devices.

More context: [Reddit post](https://www.reddit.com/r/VeoCamera/comments/1nr0ic7/how_would_you_design_your_veo/)

**Current tech stack:**  
- **Frontend:** React + Vite (Three.js / React Three Fiber for WebGL and shaders)  
- **Backend:** Python Flask (OpenCV / NumPy + FFmpeg for stitching and transcoding)  
- **Platforms:** Windows, macOS, Linux (x86_64 and arm64)

---

## Current Features

- Live stitching of two video feeds (GPU-accelerated)  
- Browser-based stitching, even on mobile devices  
- Works with arbitrary camera models and positions  
- Experimental livestreaming support (in progress)

---

## Goals

- Provide a non-subscription, self-hosted alternative to Veo  
- Allow local clubs and communities to film and analyze games affordably  
- Build a sustainable, community-driven open-source ecosystem  

### Notes on Current Approach


This browser-based approach comes with some quirks, mainly related to:

- Frame synchronization between cameras  
- Limited control over video playback, as browsers don’t provide frame-precise rendering  

To mitigate these, a pre-transcoding step is required: both videos are first stacked vertically into a single file.  
This introduces a delay before stitching but ensures smoother operation and synchronization in-browser.

If the project gains traction, a more robust stack will be explored, likely Rust + wgpu similarly to [Gyroflow](https://github.com/gyroflow/gyroflow),  enabling:

- Faster-than-real-time processing  
- Frame-precise synchronization  
- Multiple simultaneous streams without pre-transcoding  
- High-resolution and low-latency operation  
- Cloud deployment  
- Seamless AI-based football tracking (although auto-panning via ball coordinates is already feasible)

*Note: Just found out about [Webcodecs](https://developer.chrome.com/docs/web-platform/best-practices/webcodecs) which could be useful to mitigate the above mentionned quirks, without having to rebuild in a new tech stack. This shall be considered soon.*

---

## Development Progress

All core functionality already exists in a private repository.  
The following steps mainly involve refactoring and integration work before the public beta.

- [x] Setup project repository  
- [x] Implement initial stitcher with hardcoded settings  
- [x] Add controls to pan across the panorama  
- [x] Introduce state management to replace hardcoded settings  
- [x] Add video player controls
- Integrate backend processing pipeline  
- [ ] Design and implement a clean UI  
- [ ] Prepare and publish beta release  

---

## License

This project is licensed under the **AGPL-3.0** license.  
All derived versions must remain open-source.  
A contributor license agreement (CLA) will be published soon.

---

## Community & Contributions

This project welcomes everyone: developers, designers, coaches, and camera enthusiasts alike.  
Contributions, feedback, and testing are all appreciated, whether technical or not.

Guidelines and contribution details will be released very soon.  
A dedicated community forum will also be launched to centralize discussions and ideas.

---

## Feedback

Feedback on design priorities, usability, or the general direction of the project is highly encouraged.  
You can share thoughts in the [Reddit thread](https://www.reddit.com/r/VeoCamera/comments/1nr0ic7/how_would_you_design_your_veo/) for now. Issues and discussions will open once the beta is public.

You can also contact me at mohamedtahaguelzim@gmail.com
