#!/usr/bin/env python3
"""Generate lookahead trajectory from events JSONL."""
import json, sys
import numpy as np
from scipy.ndimage import uniform_filter1d

FPS = 30.0

def load_world_states(path):
    ws = {}
    with open(path) as f:
        for line in f:
            ev = json.loads(line)
            fi = ev.get('frame_index')
            if fi is None or ev['kind'] != 'world_state': continue
            ball = ev.get('ball')
            players = [(p['yaw'], p['pitch'], p['confidence'])
                       for p in ev.get('players', []) if p.get('state') != 'Lost']
            ws[fi] = {
                'ball_yaw': ball['yaw'] if ball and ball.get('state') != 'Lost' else None,
                'ball_pitch': ball['pitch'] if ball and ball.get('state') != 'Lost' else None,
                'players': players,
            }
    return ws

def compute_targets(ws, ball_weight=0.20, edge_push=0.15, pitch_bias=0.05):
    players = ws['players']
    if len(players) < 2: return None, None, None
    total_c = sum(c for _,_,c in players)
    if total_c <= 0: return None, None, None
    cy = sum(y*c for y,_,c in players) / total_c
    cp = sum(p*c for _,p,c in players) / total_c
    spread = max((((y-cy)**2 + (p-cp)**2)**0.5 for y,p,_ in players), default=0)
    ty = cy * (1.0 + edge_push)
    tp = cp + pitch_bias
    if ws['ball_yaw'] is not None:
        ty = ty * (1.0 - ball_weight) + ws['ball_yaw'] * ball_weight
        tp = tp * (1.0 - ball_weight) + ws['ball_pitch'] * ball_weight
    return ty, tp, spread

def lookahead_blend(targets, N=25, decay_factor=0.6):
    n = len(targets)
    out = np.zeros(n)
    smooth = uniform_filter1d(targets, 5)
    for k in range(n):
        end = min(k + N, n)
        window = smooth[k:end]
        weights = np.exp(-np.arange(len(window)) / (N * decay_factor))
        blended = np.average(window, weights=weights)
        if len(window) >= 5:
            slope = np.polyfit(np.arange(min(10, len(window))), window[:min(10, len(window))], 1)[0]
            blended += slope * N * 0.3
        out[k] = blended
    return out

def generate(jsonl_path, out_path, dead_zone_rad=0.0):
    world_states = load_world_states(jsonl_path)
    frames = sorted(world_states.keys())
    n = len(frames)
    print(f'{jsonl_path}: {n} frames ({n/FPS:.0f}s)')

    raw_yaw, raw_pitch, raw_spread = [], [], []
    for fi in frames:
        ty, tp, sp = compute_targets(world_states[fi])
        raw_yaw.append(ty if ty is not None else np.nan)
        raw_pitch.append(tp if tp is not None else np.nan)
        raw_spread.append(sp if sp is not None else np.nan)

    raw_yaw = np.array(raw_yaw)
    raw_pitch = np.array(raw_pitch)
    raw_spread = np.array(raw_spread)

    for arr in [raw_yaw, raw_pitch, raw_spread]:
        mask = np.isfinite(arr)
        if mask.sum() > 0:
            arr[:] = np.interp(np.arange(n), np.arange(n)[mask], arr[mask])

    blend_yaw = lookahead_blend(raw_yaw, 25, 0.6)
    blend_pitch = lookahead_blend(raw_pitch, 25, 0.4)
    blend_spread = lookahead_blend(raw_spread, 25, 0.4)

    cam_yaw = np.zeros(n)
    cam_pitch = np.zeros(n)
    cam_fov = np.zeros(n)
    yaw, pitch, fov = blend_yaw[0], blend_pitch[0], 40.0

    for i in range(n):
        target_yaw = blend_yaw[i]
        target_pitch = blend_pitch[i]

        # Dead zone: only update target if it moved beyond the radius
        if dead_zone_rad > 0:
            dist = ((target_yaw - yaw)**2 + (target_pitch - pitch)**2)**0.5
            if dist < dead_zone_rad:
                target_yaw = yaw
                target_pitch = pitch

        yaw += 0.04 * (target_yaw - yaw)
        pitch += 0.015 * (target_pitch - pitch)
        spread_deg = np.degrees(blend_spread[i])
        target_fov = np.clip(2.0 * spread_deg, 22.0, 58.0)
        fov += 0.008 * (target_fov - fov)
        cam_yaw[i] = yaw
        cam_pitch[i] = pitch
        cam_fov[i] = fov

    cam_yaw = uniform_filter1d(cam_yaw, 9)
    cam_pitch = uniform_filter1d(cam_pitch, 15)
    cam_fov = uniform_filter1d(cam_fov, 21)

    vel = np.degrees(np.std(np.diff(cam_yaw) * FPS))
    dz_label = f' dead_zone={np.degrees(dead_zone_rad):.1f}deg' if dead_zone_rad > 0 else ''
    print(f'  -> {out_path} | vel_std={vel:.2f} deg/s{dz_label}')

    with open(out_path, 'w') as f:
        f.write('frame,yaw,pitch,fov\n')
        for i, fi in enumerate(frames):
            f.write(f'{fi},{cam_yaw[i]:.6f},{cam_pitch[i]:.6f},{cam_fov[i]:.2f}\n')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <events.jsonl> <output.csv> [dead_zone_deg]")
        sys.exit(1)
    dz = float(sys.argv[3]) * (3.14159 / 180.0) if len(sys.argv) > 3 else 0.0
    generate(sys.argv[1], sys.argv[2], dead_zone_rad=dz)
