"""Evaluate panner quality from a reco pipeline events JSONL file.

Computes standardized metrics for comparing panners:
  - Smoothness: velocity std, P95, acceleration std, direction reversals
  - Tracking: ball coverage, ball offset from viewport center
  - Stability: idle ratio, total arc traveled

Usage:
    python3 scripts/eval_panner.py /tmp/events.jsonl
    python3 scripts/eval_panner.py a.jsonl b.jsonl --labels "baseline" "lookahead"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PannerMetrics:
    name: str
    n_frames: int = 0
    fps: float = 29.97
    # Smoothness
    vel_mean: float = 0.0
    vel_std: float = 0.0
    vel_p95: float = 0.0
    vel_max: float = 0.0
    accel_std: float = 0.0
    reversals_per_sec: float = 0.0
    # Tracking
    ball_coverage: float = 0.0
    ball_offset_rms: float = 0.0
    ball_offset_p95: float = 0.0
    # Stability
    idle_ratio: float = 0.0
    total_arc_deg: float = 0.0
    yaw_range_deg: float = 0.0
    # Anticipation
    lead_lag_frames: float = 0.0
    lead_lag_ms: float = 0.0

    def print_table(self):
        print(f"\n{'=' * 50}")
        print(f"  {self.name} ({self.n_frames} frames)")
        print(f"{'=' * 50}")
        print(f"  Smoothness")
        print(f"    vel_mean:       {self.vel_mean:6.2f} deg/s")
        print(f"    vel_std:        {self.vel_std:6.2f} deg/s  {'OK' if self.vel_std < 3.5 else 'HIGH'}")
        print(f"    vel_p95:        {self.vel_p95:6.2f} deg/s  {'OK' if self.vel_p95 < 7.5 else 'HIGH'}")
        print(f"    vel_max:        {self.vel_max:6.2f} deg/s")
        print(f"    accel_std:      {self.accel_std:6.2f} deg/s^2")
        print(f"    reversals/s:    {self.reversals_per_sec:6.2f}       {'OK' if self.reversals_per_sec < 2.0 else 'HIGH'}")
        print(f"  Tracking")
        print(f"    ball_coverage:  {self.ball_coverage:6.1f} %    {'OK' if self.ball_coverage > 90 else 'LOW'}")
        print(f"    offset_rms:     {self.ball_offset_rms:6.2f} deg  {'OK' if self.ball_offset_rms < 4 else 'HIGH'}")
        print(f"    offset_p95:     {self.ball_offset_p95:6.2f} deg")
        lag = self.lead_lag_frames
        label = "LEADS" if lag < -0.5 else ("LAGS" if lag > 0.5 else "SYNC")
        print(f"    lead/lag:      {self.lead_lag_ms:+.0f} ms ({lag:+.1f} frames) {label}")
        print(f"  Stability")
        print(f"    idle_ratio:     {self.idle_ratio:6.1f} %")
        print(f"    total_arc:      {self.total_arc_deg:6.1f} deg")
        print(f"    yaw_range:      {self.yaw_range_deg:6.1f} deg")


def evaluate(path: Path, name: str, fps: float) -> PannerMetrics:
    poses = []
    ball_positions = []
    has_ball = []

    with open(path) as f:
        current_ball = None
        for line in f:
            ev = json.loads(line)
            kind = ev.get("kind")
            if kind == "world_state":
                current_ball = ev.get("ball")
            elif kind == "pan_decision":
                p = ev["pose"]
                poses.append((p["yaw"], p["pitch"], p["fov_degrees"]))
                if current_ball is not None:
                    has_ball.append(True)
                    ball_positions.append((current_ball["yaw"], current_ball["pitch"]))
                else:
                    has_ball.append(False)
                    ball_positions.append((0.0, 0.0))

    if len(poses) < 2:
        print(f"  {name}: not enough frames ({len(poses)})")
        return PannerMetrics(name=name)

    m = PannerMetrics(name=name, n_frames=len(poses), fps=fps)

    yaws = [p[0] for p in poses]
    pitches = [p[1] for p in poses]

    # Velocities (deg/s)
    vels = []
    for i in range(1, len(yaws)):
        dy = math.degrees(yaws[i] - yaws[i - 1])
        dp = math.degrees(pitches[i] - pitches[i - 1])
        v = math.sqrt(dy * dy + dp * dp) * fps
        vels.append(v)

    yaw_vels = [math.degrees(yaws[i] - yaws[i - 1]) * fps for i in range(1, len(yaws))]

    m.vel_mean = sum(vels) / len(vels)
    m.vel_std = math.sqrt(sum((v - m.vel_mean) ** 2 for v in vels) / len(vels))
    sorted_vels = sorted(vels)
    m.vel_p95 = sorted_vels[int(0.95 * len(sorted_vels))]
    m.vel_max = max(vels)

    # Acceleration
    accels = [(vels[i] - vels[i - 1]) * fps for i in range(1, len(vels))]
    if accels:
        accel_mean = sum(accels) / len(accels)
        m.accel_std = math.sqrt(sum((a - accel_mean) ** 2 for a in accels) / len(accels))

    # Direction reversals (yaw sign changes)
    reversals = 0
    for i in range(1, len(yaw_vels)):
        if yaw_vels[i] * yaw_vels[i - 1] < 0 and abs(yaw_vels[i]) > 0.5:
            reversals += 1
    duration_s = len(poses) / fps
    m.reversals_per_sec = reversals / duration_s if duration_s > 0 else 0

    # Ball coverage and offset
    n_ball = sum(1 for b in has_ball if b)
    m.ball_coverage = 100.0 * n_ball / len(has_ball) if has_ball else 0

    offsets = []
    for i, (by, bp) in enumerate(ball_positions):
        if not has_ball[i]:
            continue
        dy = math.degrees(by - yaws[i])
        dp = math.degrees(bp - pitches[i])
        offsets.append(math.sqrt(dy * dy + dp * dp))

    if offsets:
        m.ball_offset_rms = math.sqrt(sum(o * o for o in offsets) / len(offsets))
        sorted_offsets = sorted(offsets)
        m.ball_offset_p95 = sorted_offsets[int(0.95 * len(sorted_offsets))]

    # Lead/lag: cross-correlate ball yaw velocity with camera yaw velocity.
    # Negative lag = camera leads (anticipates). Positive = camera lags.
    ball_yaws = [bp[0] for i, bp in enumerate(ball_positions) if has_ball[i]]
    cam_yaws_for_ball = [yaws[i] for i in range(len(yaws)) if has_ball[i]]
    if len(ball_yaws) > 30:
        ball_dy = [ball_yaws[i] - ball_yaws[i - 1] for i in range(1, len(ball_yaws))]
        cam_dy = [cam_yaws_for_ball[i] - cam_yaws_for_ball[i - 1] for i in range(1, len(cam_yaws_for_ball))]
        n_sig = min(len(ball_dy), len(cam_dy))
        ball_dy = ball_dy[:n_sig]
        cam_dy = cam_dy[:n_sig]
        # Normalize
        b_mean = sum(ball_dy) / n_sig
        c_mean = sum(cam_dy) / n_sig
        b_std = max(1e-9, math.sqrt(sum((x - b_mean) ** 2 for x in ball_dy) / n_sig))
        c_std = max(1e-9, math.sqrt(sum((x - c_mean) ** 2 for x in cam_dy) / n_sig))
        ball_n = [(x - b_mean) / b_std for x in ball_dy]
        cam_n = [(x - c_mean) / c_std for x in cam_dy]
        # Cross-correlate over +/- 30 frames
        max_lag = min(30, n_sig // 3)
        best_r, best_lag = -1.0, 0
        for lag in range(-max_lag, max_lag + 1):
            s = 0.0
            count = 0
            for j in range(n_sig):
                k = j + lag
                if 0 <= k < n_sig:
                    s += ball_n[j] * cam_n[k]
                    count += 1
            r = s / max(1, count)
            if r > best_r:
                best_r = r
                best_lag = lag
        m.lead_lag_frames = float(best_lag)
        m.lead_lag_ms = best_lag * 1000.0 / fps

    # Stability
    idle_count = sum(1 for v in vels if v < 0.5)
    m.idle_ratio = 100.0 * idle_count / len(vels)
    m.total_arc_deg = sum(math.degrees(abs(yaws[i] - yaws[i - 1])) for i in range(1, len(yaws)))
    m.yaw_range_deg = math.degrees(max(yaws) - min(yaws))

    return m


def main():
    ap = argparse.ArgumentParser(description="Evaluate panner quality from JSONL events")
    ap.add_argument("files", nargs="+", type=Path, help="JSONL event files")
    ap.add_argument("--labels", nargs="+", help="Names for each file (default: filename)")
    ap.add_argument("--fps", type=float, default=29.97, help="Source FPS (default: 29.97)")
    args = ap.parse_args()

    labels = args.labels or [p.stem for p in args.files]
    if len(labels) < len(args.files):
        labels.extend(p.stem for p in args.files[len(labels):])

    metrics = []
    for path, label in zip(args.files, labels):
        m = evaluate(path, label, args.fps)
        m.print_table()
        metrics.append(m)

    if len(metrics) > 1:
        print(f"\n{'=' * 60}")
        print("  Comparison")
        print(f"{'=' * 60}")
        header = f"  {'Metric':<20}" + "".join(f"{m.name:>14}" for m in metrics)
        print(header)
        print("  " + "-" * (20 + 14 * len(metrics)))
        rows = [
            ("vel_std (deg/s)", [f"{m.vel_std:.2f}" for m in metrics]),
            ("vel_p95 (deg/s)", [f"{m.vel_p95:.2f}" for m in metrics]),
            ("accel_std", [f"{m.accel_std:.2f}" for m in metrics]),
            ("reversals/s", [f"{m.reversals_per_sec:.2f}" for m in metrics]),
            ("ball_coverage %", [f"{m.ball_coverage:.1f}" for m in metrics]),
            ("offset_rms (deg)", [f"{m.ball_offset_rms:.2f}" for m in metrics]),
            ("lead/lag (ms)", [f"{m.lead_lag_ms:+.0f}" for m in metrics]),
            ("idle_ratio %", [f"{m.idle_ratio:.1f}" for m in metrics]),
            ("total_arc (deg)", [f"{m.total_arc_deg:.1f}" for m in metrics]),
        ]
        for label, vals in rows:
            print(f"  {label:<20}" + "".join(f"{v:>14}" for v in vals))


if __name__ == "__main__":
    main()
