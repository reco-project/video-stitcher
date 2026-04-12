#!/usr/bin/env python3
"""Generate field_roi polygons for a calibration JSON file.

Opens a frame from each camera video. Click polygon vertices to define
the playing field boundary. Press Enter to confirm, Escape to redo.
The field_roi is saved in normalized [0,1] coordinates matching the
FieldRoi format in reco-core's calibration.rs.

Usage:
    python3 scripts/field_roi.py left.mp4 right.mp4 calibration.json

Requirements:
    pip install opencv-python
"""

import argparse
import json
import sys

import cv2


def get_frame(video_path: str, frame_num: int = 30) -> any:
    """Extract a single frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error: cannot read frame {frame_num} from {video_path}")
        sys.exit(1)
    return frame


def collect_polygon(window_name: str, frame) -> list:
    """Show a frame and let the user click polygon vertices.

    Left click: add point
    Right click: remove last point
    Enter: confirm
    Escape: clear and restart
    """
    points = []
    display = frame.copy()
    h, w = frame.shape[:2]

    def on_mouse(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    def redraw():
        nonlocal display
        display = frame.copy()
        for i, (px, py) in enumerate(points):
            cv2.circle(display, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(display, str(i), (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(display, points[i], points[i + 1], (0, 255, 0), 2)
            # Close the polygon visually
            cv2.line(display, points[-1], points[0], (0, 255, 0), 1)

        info = f"Click field boundary points. Enter=confirm, Esc=redo, RClick=undo ({len(points)} pts)"
        cv2.putText(display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window_name, display)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(w, 1280), min(h, 720))
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter
            if len(points) >= 3:
                break
            print("Need at least 3 points. Keep clicking.")
        elif key == 27:  # Escape
            points.clear()
            redraw()

    cv2.destroyWindow(window_name)

    # Normalize to [0, 1]
    normalized = [[round(x / w, 4), round(y / h, 4)] for x, y in points]
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Generate field_roi for a reco calibration file"
    )
    parser.add_argument("left", help="Path to left camera video")
    parser.add_argument("right", help="Path to right camera video")
    parser.add_argument("calibration", help="Path to calibration JSON")
    parser.add_argument("--frame", type=int, default=30,
                        help="Which frame to show (default: 30)")
    parser.add_argument("--output", help="Output JSON path (default: overwrite input)")
    args = parser.parse_args()

    # Load calibration
    with open(args.calibration) as f:
        cal = json.load(f)

    print("=== Left Camera ===")
    print("Click the field boundary polygon, then press Enter.")
    left_frame = get_frame(args.left, args.frame)
    left_roi = collect_polygon("Left Camera - Field ROI", left_frame)
    print(f"Left ROI: {len(left_roi)} points")

    print("\n=== Right Camera ===")
    print("Click the field boundary polygon, then press Enter.")
    right_frame = get_frame(args.right, args.frame)
    right_roi = collect_polygon("Right Camera - Field ROI", right_frame)
    print(f"Right ROI: {len(right_roi)} points")

    # Update calibration
    cal["field_roi"] = {
        "left": left_roi,
        "right": right_roi,
    }

    output_path = args.output or args.calibration
    with open(output_path, "w") as f:
        json.dump(cal, f, indent=4)
    print(f"\nSaved field_roi to {output_path}")
    print(f"  Left:  {len(left_roi)} vertices")
    print(f"  Right: {len(right_roi)} vertices")


if __name__ == "__main__":
    main()
