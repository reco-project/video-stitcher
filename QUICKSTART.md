# Quick Start Guide

## 1. Download & Open

Download the latest **GUI** release for your platform from the [Releases page](https://github.com/reco-project/video-stitcher/releases). Also download `yolo26n.onnx` from the same release (needed for AI tracking).

Extract and run `reco-gui`. On Mac, right-click > Open the first time (unsigned app).

## 2. Import & Calibrate

- Click **Left Video** and select your left camera file(s)
- Click **Right Video** and select your right camera file(s)
- Click **Auto Calibrate** and wait for it to finish

Your stitched panorama should appear in the preview. Pan with mouse drag, zoom with scroll wheel.

## 3. Fix the Lens (if needed)

If the image looks warped (curved lines that should be straight):

- Open the right panel (button in the top bar)
- Go to **Lens** > **Browse Profiles** and pick your camera model
- Use **Lens Preview** to check that lines look straight
- Click **Save Calibration** (re-calibration runs automatically)

## 4. Set the Field Area

- Click **Set ROI** - a browser page opens after a few seconds
- Draw a polygon covering the field visible in the **left camera**
- Switch to **Right Camera** in the dropdown and do the same for the right
- Click **Copy ROI**, go back to the app, click **Paste ROI**

## 5. Export

- Click **Export** in the top bar
- Pick resolution (1080p recommended) and output location
- To enable AI tracking:
  - Toggle **AI Tracking**
  - Select the `yolo26n.onnx` model you downloaded
  - Set mode to **Field** and detection interval to **15**
- Click **Start**

## Tips

- Try a short clip first (set start/end times) before exporting the full video
- 10-bit footage (e.g. DJI Action 4) may show green on Windows - use CPU Decode in settings as a workaround
- Share your calibration file with us if something looks wrong
- Can't find a lens profile for your camera? Let us know
