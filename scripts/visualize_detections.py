#!/usr/bin/env python3
"""Visualize AI detections from a reco pipeline events JSONL file.

Two modes:
  --export: Render an annotated video with detection boxes, tracking info,
            frame counter, and timestamp overlay. Output is 1080p h264.
  --browse: Interactive frame-by-frame browser. Arrow keys to step,
            click to seek, 'q' to quit.

Usage:
  # Generate the events file with reco CLI:
  reco stitch left.mp4 right.mp4 -c cal.json --model yolo26n.onnx \
      --events detections.jsonl -o output.mp4

  # Export annotated video:
  python3 visualize_detections.py --export detections.jsonl left.mp4

  # Browse frame by frame:
  python3 visualize_detections.py --browse detections.jsonl left.mp4

  # Use right camera instead:
  python3 visualize_detections.py --export detections.jsonl right.mp4 --camera right
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

COCO_NAMES = {
    0: "person", 32: "sports ball", 37: "skateboard",
    38: "surfboard", 39: "tennis racket",
}

CLASS_COLORS = {
    0: (0, 200, 0),      # person - green
    32: (0, 100, 255),    # ball - orange
}
DEFAULT_COLOR = (200, 200, 200)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_events(path):
    """Load JSONL events grouped by frame_index."""
    frames = defaultdict(lambda: {
        "detections_raw": [],
        "detection_filter": [],
        "world_state": None,
        "pan_decision": None,
        "pose_presented": None,
        "frame_complete": None,
        "timestamp_ms": 0.0,
    })
    max_frame = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            kind = ev.get("kind", "")
            idx = ev.get("frame_index", -1)
            if idx < 0:
                continue
            max_frame = max(max_frame, idx)
            if kind == "frame_start":
                frames[idx]["timestamp_ms"] = ev.get("timestamp_ms", 0.0)
            elif kind == "detections_raw":
                frames[idx]["detections_raw"] = ev.get("detections", [])
            elif kind == "detection_filter":
                frames[idx]["detection_filter"].append(ev)
            elif kind == "world_state":
                frames[idx]["world_state"] = ev
            elif kind == "pan_decision":
                frames[idx]["pan_decision"] = ev
            elif kind == "pose_presented":
                frames[idx]["pose_presented"] = ev
            elif kind == "frame_complete":
                frames[idx]["frame_complete"] = ev
    return frames, max_frame


def draw_detections(img, detections, camera_filter, h, w):
    """Draw bounding boxes from DetectionsRaw."""
    count = 0
    for det in detections:
        cam = det.get("camera", "").lower()
        if cam != camera_filter:
            continue
        cx, cy = det["camera_center"]
        sw, sh = det["camera_size"]
        conf = det.get("confidence", 0)
        class_id = det.get("class_id", -1)

        x1 = int((cx - sw / 2) * w)
        y1 = int((cy - sh / 2) * h)
        x2 = int((cx + sw / 2) * w)
        y2 = int((cy + sh / 2) * h)

        color = CLASS_COLORS.get(class_id, DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = COCO_NAMES.get(class_id, f"c{class_id}")
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 6),
                    FONT, 0.45, color, 1, cv2.LINE_AA)
        count += 1
    return count


def draw_filter_removed(img, filter_events, camera_filter, h, w):
    """Draw detections removed by ROI/filter stages as grey dashed boxes."""
    for fev in filter_events:
        before_ids = {(d["camera_center"][0], d["camera_center"][1])
                      for d in fev.get("before", [])}
        after_ids = {(d["camera_center"][0], d["camera_center"][1])
                     for d in fev.get("after", [])}
        removed = before_ids - after_ids

        for det in fev.get("before", []):
            cam = det.get("camera", "").lower()
            if cam != camera_filter:
                continue
            key = (det["camera_center"][0], det["camera_center"][1])
            if key not in removed:
                continue
            cx, cy = det["camera_center"]
            sw, sh = det["camera_size"]
            x1 = int((cx - sw / 2) * w)
            y1 = int((cy - sh / 2) * h)
            x2 = int((cx + sw / 2) * w)
            y2 = int((cy + sh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 1)
            cv2.putText(img, "filtered", (x1, y1 - 6),
                        FONT, 0.35, (100, 100, 100), 1, cv2.LINE_AA)


def draw_overlay(img, frame_idx, frame_data, total_frames, fps):
    """Draw frame counter, timestamp, and tracking info."""
    h, w = img.shape[:2]
    ts_ms = frame_data["timestamp_ms"]
    ts_sec = ts_ms / 1000.0
    minutes = int(ts_sec // 60)
    seconds = ts_sec % 60

    # Frame counter + timestamp (top left)
    text = f"Frame {frame_idx}/{total_frames}  {minutes:02d}:{seconds:05.2f}"
    cv2.putText(img, text, (10, 28), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (10, 28), FONT, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    # Tracking state (top right)
    ws = frame_data.get("world_state")
    if ws:
        n_players = len(ws.get("players", []))
        ball = ws.get("ball")
        ball_text = f"ball: ({ball['yaw']:.2f}, {ball['pitch']:.2f})" if ball else "ball: -"
        info = f"players: {n_players}  {ball_text}"
        tw = cv2.getTextSize(info, FONT, 0.5, 1)[0][0]
        cv2.putText(img, info, (w - tw - 10, 28),
                    FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, info, (w - tw - 10, 28),
                    FONT, 0.5, (0, 200, 200), 1, cv2.LINE_AA)

    # Pan decision (bottom left)
    pose = frame_data.get("pose_presented")
    if pose and "pose" in pose:
        p = pose["pose"]
        yaw_deg = p.get("yaw", 0) * 57.2958
        pitch_deg = p.get("pitch", 0) * 57.2958
        fov = p.get("fov_deg", 0)
        pan_text = f"pan: yaw={yaw_deg:.1f} pitch={pitch_deg:.1f} fov={fov:.0f}"
        cv2.putText(img, pan_text, (10, h - 12),
                    FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, pan_text, (10, h - 12),
                    FONT, 0.5, (200, 200, 0), 1, cv2.LINE_AA)

    # Timing (bottom right)
    fc = frame_data.get("frame_complete")
    if fc and "timing" in fc:
        t = fc["timing"]
        det_ms = (t.get("detect_us") or 0) / 1000
        total_ms = (t.get("total_us") or 0) / 1000
        timing_text = f"detect: {det_ms:.0f}ms  total: {total_ms:.0f}ms"
        tw = cv2.getTextSize(timing_text, FONT, 0.45, 1)[0][0]
        cv2.putText(img, timing_text, (w - tw - 10, h - 12),
                    FONT, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, timing_text, (w - tw - 10, h - 12),
                    FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def export_mode(events_path, video_path, output_path, camera, max_frames):
    """Export an annotated video."""
    frames, total = load_events(events_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}", file=sys.stderr)
        return 1

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_w, out_h = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    print(f"Source: {src_w}x{src_h} @ {fps:.1f} fps")
    print(f"Events: {total + 1} frames")
    print(f"Output: {output_path} ({out_w}x{out_h})")

    frame_idx = 0
    limit = max_frames if max_frames else total + 1
    while frame_idx <= min(total, limit - 1):
        ret, raw = cap.read()
        if not ret:
            break

        img = cv2.resize(raw, (out_w, out_h))
        fd = frames[frame_idx]

        n = draw_detections(img, fd["detections_raw"], camera, out_h, out_w)
        draw_filter_removed(img, fd["detection_filter"], camera, out_h, out_w)
        draw_overlay(img, frame_idx, fd, total, fps)

        out.write(img)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total + 1} frames...", flush=True)

    out.release()
    cap.release()
    print(f"Done: {frame_idx} frames written to {output_path}")
    return 0


def browse_mode(events_path, video_path, camera):
    """Interactive frame-by-frame browser."""
    frames, total = load_events(events_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}", file=sys.stderr)
        return 1

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    disp_w, disp_h = 1920, 1080
    current = 0
    needs_redraw = True

    print(f"Source: {src_w}x{src_h} @ {fps:.1f} fps, {total + 1} event frames")
    print("Controls: Right/D = next, Left/A = prev, PgDn = +30, PgUp = -30")
    print("          Home = first, End = last, Q = quit")

    cv2.namedWindow("Reco Detection Browser", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reco Detection Browser", disp_w, disp_h)

    def seek_and_draw(idx):
        idx = max(0, min(idx, total))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, raw = cap.read()
        if not ret:
            return None, idx
        img = cv2.resize(raw, (disp_w, disp_h))
        fd = frames[idx]
        draw_detections(img, fd["detections_raw"], camera, disp_h, disp_w)
        draw_filter_removed(img, fd["detection_filter"], camera, disp_h, disp_w)
        draw_overlay(img, idx, fd, total, fps)
        return img, idx

    while True:
        if needs_redraw:
            img, current = seek_and_draw(current)
            if img is not None:
                cv2.imshow("Reco Detection Browser", img)
            needs_redraw = False

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("d") or key == 83:  # right arrow
            current += 1
            needs_redraw = True
        elif key == ord("a") or key == 81:  # left arrow
            current -= 1
            needs_redraw = True
        elif key == 85:  # page up
            current -= 30
            needs_redraw = True
        elif key == 86:  # page down
            current += 30
            needs_redraw = True
        elif key == 80:  # home
            current = 0
            needs_redraw = True
        elif key == 87:  # end
            current = total
            needs_redraw = True

    cv2.destroyAllWindows()
    cap.release()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Visualize reco AI detection events on source video."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    exp = sub.add_parser("export", help="Export annotated video")
    exp.add_argument("events", type=Path, help="Pipeline events JSONL file")
    exp.add_argument("video", type=Path, help="Source video (left or right)")
    exp.add_argument("-o", "--output", type=Path, default=Path("annotated.mp4"),
                     help="Output video path (default: annotated.mp4)")
    exp.add_argument("--camera", choices=["left", "right"], default="left",
                     help="Which camera's detections to show (default: left)")
    exp.add_argument("--max-frames", type=int, default=0,
                     help="Limit frames to process (0 = all)")

    brw = sub.add_parser("browse", help="Interactive frame browser")
    brw.add_argument("events", type=Path, help="Pipeline events JSONL file")
    brw.add_argument("video", type=Path, help="Source video (left or right)")
    brw.add_argument("--camera", choices=["left", "right"], default="left",
                     help="Which camera's detections to show (default: left)")

    args = parser.parse_args()

    if args.mode == "export":
        return export_mode(args.events, args.video, args.output,
                           args.camera, args.max_frames)
    elif args.mode == "browse":
        return browse_mode(args.events, args.video, args.camera)


if __name__ == "__main__":
    sys.exit(main())
