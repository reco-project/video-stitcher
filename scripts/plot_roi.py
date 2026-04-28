#!/usr/bin/env python3
"""Plot field ROI from a calibration JSON file.

Extracts one frame from each video and overlays the ROI polygon.
Usage: python scripts/plot_roi.py calibration.json [left_video] [right_video]

If videos are not provided, plots the polygon on a blank canvas.
"""
import json
import sys
import subprocess
import tempfile
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None


def extract_frame(video_path, time_sec=1.0):
    """Extract a single frame from a video using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(time_sec), "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", tmp_path],
            capture_output=True, check=True
        )
        if Image:
            return np.array(Image.open(tmp_path))
    except Exception as e:
        print(f"Warning: frame extraction failed: {e}")
    return None


def plot_roi(cal_path, left_video=None, right_video=None):
    with open(cal_path) as f:
        cal = json.load(f)

    roi = cal.get("field_roi")
    if not roi:
        print("No field_roi in calibration file.")
        sys.exit(1)

    left_pts = np.array(roi.get("left", []))
    right_pts = np.array(roi.get("right", []))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Field ROI - {Path(cal_path).name}", fontsize=14)

    for ax, pts, label, video in [
        (axes[0], left_pts, "Left", left_video),
        (axes[1], right_pts, "Right", right_video),
    ]:
        frame = extract_frame(video) if video else None
        if frame is not None:
            ax.imshow(frame)
            h, w = frame.shape[:2]
        else:
            w, h = 1920, 1080
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)

        ax.set_title(f"{label} ({len(pts)} points)")

        if len(pts) > 0:
            if frame is not None:
                # Scale from [0,1] normalized to pixel coordinates
                xs = pts[:, 0] * w
                ys = pts[:, 1] * h
            else:
                xs = pts[:, 0]
                ys = pts[:, 1]

            # Draw polygon
            poly = plt.Polygon(
                list(zip(xs, ys)),
                fill=True, facecolor=(0, 1, 0, 0.15),
                edgecolor="lime", linewidth=2
            )
            ax.add_patch(poly)

            # Draw vertices
            ax.scatter(xs, ys, c="lime", s=40, zorder=5, edgecolors="white", linewidth=0.5)
            for i, (x, y) in enumerate(zip(xs, ys)):
                ax.annotate(str(i+1), (x, y), textcoords="offset points",
                           xytext=(6, -8), fontsize=8, color="white",
                           fontweight="bold")

        if frame is None:
            ax.set_facecolor("#333")
            ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_roi.py calibration.json [left_video] [right_video]")
        sys.exit(1)

    cal = sys.argv[1]
    left = sys.argv[2] if len(sys.argv) > 2 else None
    right = sys.argv[3] if len(sys.argv) > 3 else None
    plot_roi(cal, left, right)
