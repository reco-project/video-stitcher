#!/usr/bin/env python3
"""Visualize AI detections from a reco pipeline events JSONL file.

Two modes:
  export: Render an annotated side-by-side video (left + right cameras)
          with detection boxes, tracking info, frame counter, and timestamp.
  browse: Interactive frame-by-frame browser with keyboard navigation.

Usage:
  # Generate the events file with reco CLI:
  reco stitch left.mp4 right.mp4 -c cal.json --model yolo26n.onnx \
      --events detections.jsonl -o output.mp4

  # Export annotated side-by-side video:
  python3 visualize_detections.py export detections.jsonl left.mp4 right.mp4

  # Single camera only:
  python3 visualize_detections.py export detections.jsonl left.mp4 --camera left

  # Browse frame by frame:
  python3 visualize_detections.py browse detections.jsonl left.mp4 right.mp4
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


def load_calibration(cal_path):
    """Load ROI polygons and sync_offset from a calibration JSON."""
    if not cal_path:
        return None, None, 0
    try:
        with open(cal_path) as f:
            cal = json.load(f)
        roi = cal.get("field_roi", {})
        left = np.array(roi["left"], dtype=np.float32) if roi.get("left") else None
        right = np.array(roi["right"], dtype=np.float32) if roi.get("right") else None
        sync_offset = cal.get("sync_offset", 0)
        return left, right, sync_offset
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        return None, None, 0


def draw_roi(img, roi_points, h, w):
    """Draw ROI polygon boundary on the image."""
    if roi_points is None:
        return
    pts = (roi_points * np.array([w, h])).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(255, 200, 0), thickness=2)
    cv2.putText(img, "ROI", (pts[0][0], pts[0][1] - 8),
                FONT, 0.4, (255, 200, 0), 1, cv2.LINE_AA)


def load_events(path):
    """Load JSONL events grouped by frame_index."""
    frames = defaultdict(lambda: {
        "detections_raw": [],
        "detection_filter": [],
        "world_state": None,
        "pan_decision": None,
        "panner_debug": None,
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
            elif kind == "panner_debug":
                frames[idx]["panner_debug"] = ev
            elif kind == "pan_decision":
                frames[idx]["pan_decision"] = ev
            elif kind == "pose_presented":
                frames[idx]["pose_presented"] = ev
            elif kind == "frame_complete":
                frames[idx]["frame_complete"] = ev
    return frames, max_frame


def draw_detections(img, detections, camera_filter, h, w):
    """Draw bounding boxes from DetectionsRaw. Returns detection count."""
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


def draw_camera_label(img, label, h, w):
    """Draw camera label (L/R) in the top corner."""
    cv2.putText(img, label, (w - 40, 30), FONT, 0.8,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, label, (w - 40, 30), FONT, 0.8,
                (0, 200, 255), 2, cv2.LINE_AA)


def draw_shared_overlay(img, frame_idx, frame_data, total_frames, fps):
    """Draw frame counter, timestamp, tracking, and pan info on combined image."""
    h, w = img.shape[:2]
    ts_ms = frame_data["timestamp_ms"]
    ts_sec = ts_ms / 1000.0
    minutes = int(ts_sec // 60)
    seconds = ts_sec % 60

    # Frame counter + timestamp (top left)
    text = f"Frame {frame_idx}/{total_frames}  {minutes:02d}:{seconds:05.2f}"
    cv2.putText(img, text, (10, 28), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (10, 28), FONT, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    # Detection count vs track count (top center)
    n_dets = len(frame_data["detections_raw"])
    ws = frame_data.get("world_state")
    n_tracks = len(ws.get("players", [])) if ws else 0
    ball = ws.get("ball") if ws else None
    ball_state = ""
    if ball:
        state = ball.get("state", "")
        ball_state = f"ball: {state} ({ball['yaw']:.2f}, {ball['pitch']:.2f})"
    else:
        ball_state = "ball: -"

    info = f"det: {n_dets}  tracks: {n_tracks}  {ball_state}"
    tw = cv2.getTextSize(info, FONT, 0.5, 1)[0][0]
    cx = (w - tw) // 2
    cv2.putText(img, info, (cx, 28), FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, info, (cx, 28), FONT, 0.5, (0, 200, 200), 1, cv2.LINE_AA)

    # Panner debug (second line from top)
    pd = frame_data.get("panner_debug")
    if pd:
        ball_near = "NEAR" if pd.get("ball_near_cluster") else "far"
        eff_w = pd.get("effective_ball_weight", 0)
        bp = pd.get("ball_presence", 0)
        spread = pd.get("cluster_spread", 0)
        fov_t = pd.get("fov_target", 0)
        n_p = pd.get("n_players", 0)
        debug_text = (f"cluster: {n_p}p spread={spread:.2f}  "
                      f"ball: {ball_near} presence={bp:.2f} weight={eff_w:.2f}  "
                      f"fov_target={fov_t:.0f}")
        cv2.putText(img, debug_text, (10, 48), FONT, 0.4,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, debug_text, (10, 48), FONT, 0.4,
                    (255, 150, 50), 1, cv2.LINE_AA)

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


def draw_stale_detections(img, detections, camera_filter, h, w):
    """Draw last-known detections dimmed (for non-detection interval frames)."""
    for det in detections:
        cam = det.get("camera", "").lower()
        if cam != camera_filter:
            continue
        cx, cy = det["camera_center"]
        sw, sh = det["camera_size"]
        x1 = int((cx - sw / 2) * w)
        y1 = int((cy - sh / 2) * h)
        x2 = int((cx + sw / 2) * w)
        y2 = int((cy + sh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), 1)


def resize_keep_aspect(img, target_w):
    """Resize to target width, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = target_w / w
    return cv2.resize(img, (target_w, int(h * scale)))


def is_window_open(window_name):
    """Return False once an OpenCV UI window has been closed."""
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def annotate_frame(left_img, right_img, frame_idx, frame_data, last_detections,
                   total, fps, panel_w, roi_left, roi_right, hstack):
    """Annotate and combine left+right into a single frame."""
    left = resize_keep_aspect(left_img, panel_w)
    panel_h = left.shape[0]
    fresh = len(frame_data["detections_raw"]) > 0
    if fresh:
        draw_detections(left, frame_data["detections_raw"], "left", panel_h, panel_w)
    elif last_detections:
        draw_stale_detections(left, last_detections, "left", panel_h, panel_w)
    draw_filter_removed(left, frame_data["detection_filter"], "left", panel_h, panel_w)
    draw_roi(left, roi_left, panel_h, panel_w)
    draw_camera_label(left, "L", panel_h, panel_w)

    if right_img is not None:
        right = resize_keep_aspect(right_img, panel_w)
        rh = right.shape[0]
        if fresh:
            draw_detections(right, frame_data["detections_raw"], "right", rh, panel_w)
        elif last_detections:
            draw_stale_detections(right, last_detections, "right", rh, panel_w)
        draw_filter_removed(right, frame_data["detection_filter"], "right", rh, panel_w)
        draw_roi(right, roi_right, rh, panel_w)
        draw_camera_label(right, "R", rh, panel_w)
        combined = np.hstack([left, right]) if hstack else np.vstack([left, right])
    else:
        combined = left

    draw_shared_overlay(combined, frame_idx, frame_data, total, fps)

    # Detection freshness indicator
    h, w = combined.shape[:2]
    status = "DETECT" if fresh else "coast"
    color = (0, 255, 0) if fresh else (100, 100, 100)
    cv2.putText(combined, status, (10, 50), FONT, 0.45, color, 1, cv2.LINE_AA)

    return combined


def export_mode(events_path, left_path, right_path, output_path, max_frames,
                cal_path, hstack):
    """Export an annotated video (vstack by default, hstack with --hstack)."""
    frames, total = load_events(events_path)
    roi_left, roi_right, sync_offset = load_calibration(cal_path)
    if roi_left is not None:
        print(f"ROI: left={len(roi_left)} pts, right={len(roi_right) if roi_right is not None else 0} pts")
    if sync_offset != 0:
        print(f"Sync offset: {sync_offset} frames (positive = skip right)")

    cap_l = cv2.VideoCapture(str(left_path))
    if not cap_l.isOpened():
        print(f"Error: cannot open {left_path}", file=sys.stderr)
        return 1

    cap_r = None
    if right_path:
        cap_r = cv2.VideoCapture(str(right_path))
        if not cap_r.isOpened():
            print(f"Warning: cannot open {right_path}, single camera mode", file=sys.stderr)
            cap_r = None

    # Apply sync offset: skip frames on the appropriate camera
    if sync_offset > 0 and cap_r:
        cap_r.set(cv2.CAP_PROP_POS_FRAMES, sync_offset)
    elif sync_offset < 0:
        cap_l.set(cv2.CAP_PROP_POS_FRAMES, abs(sync_offset))

    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if hstack:
        panel_w = 960
    else:
        panel_w = min(1920, src_w)
    panel_h = int(src_h * panel_w / src_w)
    if not hstack and cap_r and panel_h * 2 > 1080:
        panel_h = 540
        panel_w = int(src_w * panel_h / src_h)
    if cap_r:
        out_w = panel_w * 2 if hstack else panel_w
        out_h = panel_h if hstack else panel_h * 2
    else:
        out_w, out_h = panel_w, panel_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    print(f"Events: {total + 1} frames")
    print(f"Output: {output_path} ({out_w}x{out_h})")

    frame_idx = 0
    last_detections = []
    limit = max_frames if max_frames else total + 1
    while frame_idx <= min(total, limit - 1):
        ret_l, raw_l = cap_l.read()
        if not ret_l:
            break

        raw_r = None
        if cap_r:
            ret_r, raw_r = cap_r.read()
            if not ret_r:
                raw_r = None

        fd = frames[frame_idx]
        if fd["detections_raw"]:
            last_detections = fd["detections_raw"]

        combined = annotate_frame(raw_l, raw_r, frame_idx, fd, last_detections,
                                  total, fps, panel_w, roi_left, roi_right, hstack)
        out.write(combined)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total + 1} frames...", flush=True)

    out.release()
    cap_l.release()
    if cap_r:
        cap_r.release()
    print(f"Done: {frame_idx} frames written to {output_path}")
    return 0


def browse_mode(events_path, left_path, right_path, cal_path, hstack):
    """Interactive frame-by-frame browser."""
    frames, total = load_events(events_path)
    roi_left, roi_right, sync_offset = load_calibration(cal_path)

    cap_l = cv2.VideoCapture(str(left_path))
    if not cap_l.isOpened():
        print(f"Error: cannot open {left_path}", file=sys.stderr)
        return 1

    cap_r = None
    if right_path:
        cap_r = cv2.VideoCapture(str(right_path))
        if not cap_r.isOpened():
            cap_r = None

    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = 960 if hstack else 1920
    panel_h = int(src_h * panel_w / src_w)
    if cap_r:
        out_w = panel_w * 2 if hstack else panel_w
        out_h = panel_h if hstack else panel_h * 2
    else:
        out_w, out_h = panel_w, panel_h

    print(f"{total + 1} event frames")
    print("Controls: Right/D = next, Left/A = prev, PgDn = +30, PgUp = -30")
    print("          Home = first, End = last, Q = quit")

    win = "Reco Detection Browser"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, out_w, out_h)

    current = 0
    needs_redraw = True
    last_det = []

    # Compute per-camera frame offsets
    left_offset = abs(sync_offset) if sync_offset < 0 else 0
    right_offset = sync_offset if sync_offset > 0 else 0

    def seek_and_draw(idx, last_detections):
        idx = max(0, min(idx, total))
        cap_l.set(cv2.CAP_PROP_POS_FRAMES, idx + left_offset)
        ret_l, raw_l = cap_l.read()
        if not ret_l:
            return None, idx, last_detections

        raw_r = None
        if cap_r:
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, idx + right_offset)
            ret_r, raw_r = cap_r.read()
            if not ret_r:
                raw_r = None

        fd = frames[idx]
        if fd["detections_raw"]:
            last_detections = fd["detections_raw"]
        combined = annotate_frame(raw_l, raw_r, idx, fd, last_detections,
                                  total, fps, panel_w, roi_left, roi_right, hstack)
        return combined, idx, last_detections

    def wait_for_key_or_close():
        while is_window_open(win):
            key = cv2.waitKey(50)
            if key != -1:
                return key & 0xFF
        return ord("q")

    while True:
        if needs_redraw:
            img, current, last_det = seek_and_draw(current, last_det)
            if img is not None:
                cv2.imshow(win, img)
            if not is_window_open(win):
                break
            needs_redraw = False

        key = wait_for_key_or_close()
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
    cap_l.release()
    if cap_r:
        cap_r.release()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Visualize reco AI detection events on source video."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    exp = sub.add_parser("export", help="Export annotated side-by-side video")
    exp.add_argument("events", type=Path, help="Pipeline events JSONL file")
    exp.add_argument("left_video", type=Path, help="Left camera video")
    exp.add_argument("right_video", type=Path, nargs="?", default=None,
                     help="Right camera video (omit for single camera)")
    exp.add_argument("-o", "--output", type=Path, default=Path("annotated.mp4"),
                     help="Output path (default: annotated.mp4)")
    exp.add_argument("--max-frames", type=int, default=0,
                     help="Limit frames (0 = all)")
    exp.add_argument("-c", "--calibration", type=Path, default=None,
                     help="Calibration JSON (for ROI polygon overlay)")
    exp.add_argument("--hstack", action="store_true",
                     help="Side-by-side layout instead of vertical stack")

    brw = sub.add_parser("browse", help="Interactive frame browser")
    brw.add_argument("events", type=Path, help="Pipeline events JSONL file")
    brw.add_argument("left_video", type=Path, help="Left camera video")
    brw.add_argument("right_video", type=Path, nargs="?", default=None,
                     help="Right camera video (omit for single camera)")
    brw.add_argument("-c", "--calibration", type=Path, default=None,
                     help="Calibration JSON (for ROI polygon overlay)")
    brw.add_argument("--hstack", action="store_true",
                     help="Side-by-side layout instead of vertical stack")

    args = parser.parse_args()

    if args.mode == "export":
        return export_mode(args.events, args.left_video, args.right_video,
                           args.output, args.max_frames, args.calibration,
                           args.hstack)
    elif args.mode == "browse":
        return browse_mode(args.events, args.left_video, args.right_video,
                           args.calibration, args.hstack)


if __name__ == "__main__":
    sys.exit(main())
