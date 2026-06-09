# Quick Start Guide

## 1. Download & Open

Download the latest **GUI** release for your platform from the [Releases page](https://github.com/reco-project/video-stitcher/releases). Also download `yolo26n.onnx` from the same release (needed for AI tracking).

Extract and run `reco-gui`.

- **Windows**: If SmartScreen blocks the app, right-click the exe > Properties > check "Unblock" > OK, then run again.
- **Mac**: Right-click > Open the first time (unsigned app).

For AI tracking, also download `yolo26n.onnx` (1280, higher accuracy) or `yolo26n_640.onnx` (640, faster on integrated GPUs) from the same release.

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

- Click **Set ROI**
- Draw the field boundary on the stitched preview
- Click **Save ROI** to write it into the calibration JSON
- Use **Copy JSON** only when you want the raw ROI JSON on the clipboard

## 5. Export

- Click **Export** in the top bar
- Pick resolution (1080p recommended) and output location
- To enable AI tracking:
  - Toggle **AI Tracking**
  - Select the `yolo26n.onnx` model you downloaded
  - Set mode to **Field** and detection interval to **15**
- Click **Start**

## 6. Debug AI Tracking (optional)

To see what the AI detected, export pipeline events and visualize them:

```bash
reco stitch left.mp4 right.mp4 -c cal.json --model yolo26n.onnx \
    --tracking field --detection-interval 15 --events detections.jsonl -o output.mp4

python3 scripts/visualize_detections.py export detections.jsonl left.mp4 right.mp4 \
    -c cal.json -o annotated.mp4
```

This produces a side-by-side video with detection boxes, ROI boundaries, tracking state, and panner decisions overlaid.

## Tips

- Try a short clip first (set start/end times) before exporting the full video
- For football, use **Field** tracking mode with detection interval **10-15** for smooth panning
- Use the **Sync offset** slider in Stitching to fine-tune camera alignment after calibration
- Share your calibration file with us if something looks wrong
- Can't find a lens profile for your camera? Let us know
- Join the community at [forum.reco-project.org](https://forum.reco-project.org)
