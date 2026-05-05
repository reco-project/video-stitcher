#!/usr/bin/env python3
"""Game-day web control panel for Reco on Jetson.

Serves an HTML control page on port 8080. Open from your phone:
    http://<jetson-ip>:8080

Controls:
- Start/stop recording (local MKV file + optional RTMP stream)
- Set RTMP key, capture resolution, CRF, detection interval
- View live status: fps, CPU%, temperature, recording duration

When an RTMP key is set before recording, reco streams directly
via --stream-url (single encode, zero extra CPU). No separate
ffmpeg process needed.

Usage:
    python3 gameday.py [--port 8080] [--calibration match.json]
                       [--left-device 0] [--right-device 1]

Requires: reco CLI built at ~/video-stitcher/target/release/reco
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from datetime import datetime, timedelta

# ── State ────────────────────────────────────────────────────────────

RECO_BIN = Path.home() / "video-stitcher" / "target" / "release" / "reco"

state = {
    "recording": False,
    "streaming": False,
    "reco_proc": None,
    "ffmpeg_proc": None,
    "start_time": None,
    "fps": 0.0,
    "output_path": "",
    "rtmp_key": "",
    "calibration": "",
    "left_device": "0",
    "right_device": "1",
    "capture_width": 4032,
    "capture_height": 3040,
    "output_width": 1920,
    "output_height": 1080,
    "crf": 28,
    "detection_interval": 15,
    "model": str(Path.home() / "yolo26n.engine"),
    "tracking": "field",
    "blend": 0.05,
    "log_lines": [],
    "calibrating": False,
    "last_calibration_result": None,
}
lock = threading.Lock()


def read_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read().strip()) / 1000
    except Exception:
        return 0.0


def read_cpu():
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        idle = int(parts[4])
        total = sum(int(x) for x in parts[1:])
        if not hasattr(read_cpu, "prev"):
            read_cpu.prev = (idle, total)
            return 0.0
        prev_idle, prev_total = read_cpu.prev
        read_cpu.prev = (idle, total)
        d_idle = idle - prev_idle
        d_total = total - prev_total
        if d_total == 0:
            return 0.0
        return (1 - d_idle / d_total) * 100
    except Exception:
        return 0.0


def tail_log(proc):
    """Read reco stdout lines and extract fps."""
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").strip()
        with lock:
            state["log_lines"] = state["log_lines"][-50:] + [text]
            if "fps)" in text:
                try:
                    # "Processed 30 frames (29.9 / 30.0 fps)"
                    parts = text.split("(")[1].split("/")
                    state["fps"] = float(parts[0].strip())
                except (IndexError, ValueError):
                    pass


def start_recording():
    with lock:
        if state["reco_proc"] is not None:
            return {"error": "already recording"}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"/tmp/game_{ts}.mkv"
        state["output_path"] = output

        cal = state["calibration"]
        if not cal or not Path(cal).exists():
            return {"error": f"calibration not found: {cal}"}

        cmd = [
            str(RECO_BIN), "camera",
            "--left-device", state["left_device"],
            "--right-device", state["right_device"],
            "--capture-width", str(state["capture_width"]),
            "--capture-height", str(state["capture_height"]),
            "--width", str(state["output_width"]),
            "--height", str(state["output_height"]),
            "-c", cal,
            "-o", output,
            "--container", "mkv",
            "--crf", str(state["crf"]),
            "--preset", "superfast",
            "--blend", str(state["blend"]),
            "--tracking", state["tracking"],
            "--detection-interval", str(state["detection_interval"]),
            "--unconstrained",
        ]

        model = state["model"]
        if model and Path(model).exists():
            cmd.extend(["--model", model])

        rtmp_key = state["rtmp_key"]
        if rtmp_key:
            url = f"rtmp://a.rtmp.youtube.com/live2/{rtmp_key}"
            cmd.extend(["--stream-url", url])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            state["reco_proc"] = proc
            state["recording"] = True
            state["streaming"] = bool(rtmp_key)
            state["start_time"] = time.time()
            state["fps"] = 0.0
            state["log_lines"] = []

            threading.Thread(target=tail_log, args=(proc,), daemon=True).start()

            msg = "recording + streaming started" if rtmp_key else "recording started"
            return {"status": msg, "output": output}
        except Exception as e:
            return {"error": str(e)}


def stop_recording():
    with lock:
        proc = state["reco_proc"]
        if proc is None:
            return {"error": "not recording"}

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

        state["reco_proc"] = None
        state["recording"] = False
        state["streaming"] = False
        output = state["output_path"]
        return {"status": "stopped", "output": output}


def start_streaming():
    """Legacy: start streaming via separate ffmpeg process.

    With --stream-url support in reco, streaming is built into
    start_recording() when an RTMP key is set. This fallback is
    kept for older reco binaries that lack --stream-url.
    """
    with lock:
        if state["streaming"]:
            return {"error": "already streaming"}
        if not state["recording"]:
            return {"error": "start recording first"}

        rtmp_key = state["rtmp_key"]
        if not rtmp_key:
            return {"error": "set RTMP key first"}

        output = state["output_path"]
        url = f"rtmp://a.rtmp.youtube.com/live2/{rtmp_key}"

        cmd = [
            "ffmpeg", "-re", "-i", output,
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-f", "flv", url,
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            state["ffmpeg_proc"] = proc
            state["streaming"] = True
            return {"status": "streaming started (legacy ffmpeg)"}
        except Exception as e:
            return {"error": str(e)}


def stop_streaming():
    with lock:
        if not state["streaming"]:
            return {"error": "not streaming"}

        # Legacy ffmpeg process path.
        proc = state["ffmpeg_proc"]
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            state["ffmpeg_proc"] = None
            state["streaming"] = False
            return {"status": "stream stopped"}

        # Built-in --stream-url: streaming stops with recording.
        return {"info": "stream is managed by reco, stop recording to stop stream"}


def run_calibration():
    """Run live calibration via reco camera --live-calibrate."""
    with lock:
        if state["reco_proc"] is not None:
            return {"error": "stop recording first"}
        state["calibrating"] = True

    cal_output = str(Path.home() / f"calib_{datetime.now().strftime('%H%M%S')}.json")

    cmd = [
        str(RECO_BIN), "camera",
        "--left-device", state["left_device"],
        "--right-device", state["right_device"],
        "--capture-width", str(state["capture_width"]),
        "--capture-height", str(state["capture_height"]),
        "-c", cal_output,
        "-o", "/dev/null",
        "--live-calibrate",
        "--calibrate-frames", "8",
    ]

    # Lens profile (auto-loaded by reco if ~/imx477_profile.json exists).
    lens = Path.home() / "imx477_profile.json"
    if lens.exists():
        cmd.extend(["--left-profile", str(lens)])

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        output = proc.communicate(timeout=180)[0].decode("utf-8", errors="replace")
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        with lock:
            state["calibrating"] = False
        return {"error": "calibration timed out (180s)"}
    except Exception as e:
        with lock:
            state["calibrating"] = False
        return {"error": str(e)}

    with lock:
        state["calibrating"] = False

    if returncode != 0:
        return {"error": f"calibration failed (exit {returncode})", "log": output[-500:]}

    # Parse machine-readable JSON from stdout (last line).
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                result = json.loads(line)
                with lock:
                    state["calibration"] = result.get("path", cal_output)
                return result
            except json.JSONDecodeError:
                pass

    with lock:
        state["calibration"] = cal_output
    return {"status": "ok", "path": cal_output, "log": output[-500:]}


def capture_snapshot(device_id):
    """Capture a single JPEG frame from a camera via gstreamer."""
    tmp = f"/tmp/snapshot_{device_id}.jpg"
    cmd = (
        f"gst-launch-1.0 -e nvarguscamerasrc sensor-id={device_id} num-buffers=5 "
        f"! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=21/1' "
        f"! nvjpegenc ! filesink location={tmp}"
    )
    try:
        subprocess.run(cmd, shell=True, timeout=15, capture_output=True)
        if Path(tmp).exists():
            return Path(tmp).read_bytes()
    except Exception:
        pass
    return None


def save_roi(roi_data):
    """Merge field_roi into the current calibration JSON."""
    with lock:
        cal_path = state["calibration"]
    if not cal_path or not Path(cal_path).exists():
        return {"error": "no calibration file loaded"}

    try:
        cal = json.loads(Path(cal_path).read_text())
        cal["field_roi"] = roi_data
        Path(cal_path).write_text(json.dumps(cal, indent=2))
        return {"status": "ok", "path": cal_path}
    except Exception as e:
        return {"error": str(e)}


ROI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>ROI Editor</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a2e; color: #eee; font-family: sans-serif; padding: 8px; }
h2 { text-align: center; margin: 8px 0; font-size: 1.1em; }
canvas { display: block; width: 100%; border: 2px solid #333; border-radius: 8px; margin-bottom: 8px; touch-action: none; }
.controls { display: flex; gap: 8px; margin-bottom: 8px; }
.controls button { flex: 1; padding: 12px; border: none; border-radius: 8px; font-size: 1em; font-weight: bold; cursor: pointer; }
.btn-save { background: #27ae60; color: white; }
.btn-undo { background: #e67e22; color: white; }
.btn-clear { background: #e74c3c; color: white; }
.info { text-align: center; color: #888; font-size: 0.85em; margin-bottom: 8px; }
</style>
</head>
<body>
<h2 id="title">Left Camera ROI</h2>
<p class="info">Tap to add vertices. Draw a polygon around the playing field.</p>
<canvas id="canvas"></canvas>
<div class="controls">
    <button class="btn-undo" onclick="undo()">Undo</button>
    <button class="btn-clear" onclick="clearPoly()">Clear</button>
    <button class="btn-save" onclick="save()">Save ROI</button>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let img = new Image();
let side = 'left';
let polys = {left: [], right: []};
let imgW = 1, imgH = 1;

function loadImage(s) {
    side = s;
    document.getElementById('title').textContent = (s === 'left' ? 'Left' : 'Right') + ' Camera ROI';
    img = new Image();
    img.onload = () => { imgW = img.naturalWidth; imgH = img.naturalHeight; draw(); };
    img.src = '/api/snapshot/' + s + '?' + Date.now();
}

function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = (rect.width * 9 / 16) * window.devicePixelRatio;
    canvas.style.height = (rect.width * 9 / 16) + 'px';
    draw();
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img.complete && img.naturalWidth > 0) {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
    const pts = polys[side];
    if (pts.length === 0) return;
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 3;
    ctx.fillStyle = 'rgba(0, 255, 136, 0.15)';
    ctx.beginPath();
    ctx.moveTo(pts[0][0] * canvas.width, pts[0][1] * canvas.height);
    for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i][0] * canvas.width, pts[i][1] * canvas.height);
    }
    if (pts.length > 2) { ctx.closePath(); ctx.fill(); }
    ctx.stroke();
    // Draw vertices
    for (const p of pts) {
        ctx.beginPath();
        ctx.arc(p[0] * canvas.width, p[1] * canvas.height, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#00ff88';
        ctx.fill();
    }
}

canvas.addEventListener('pointerdown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    polys[side].push([Math.round(x * 1000) / 1000, Math.round(y * 1000) / 1000]);
    draw();
});

function undo() {
    polys[side].pop();
    draw();
}

function clearPoly() {
    polys[side] = [];
    draw();
}

function save() {
    if (side === 'left') {
        loadImage('right');
        return;
    }
    // Both sides done, POST to server.
    fetch('/api/roi', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({left: polys.left, right: polys.right}),
    }).then(r => r.json()).then(r => {
        if (r.status === 'ok') {
            document.body.innerHTML = '<h2 style="text-align:center;margin-top:40vh;color:#00ff88">ROI saved! Close this tab.</h2>';
        } else {
            alert('Error: ' + (r.error || 'unknown'));
        }
    });
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();
loadImage('left');
</script>
</body>
</html>
"""


def get_status():
    with lock:
        elapsed = 0
        if state["start_time"]:
            elapsed = int(time.time() - state["start_time"])
        size_mb = 0
        if state["output_path"] and Path(state["output_path"]).exists():
            size_mb = Path(state["output_path"]).stat().st_size // (1024 * 1024)

        return {
            "recording": state["recording"],
            "streaming": state["streaming"],
            "fps": round(state["fps"], 1),
            "elapsed_s": elapsed,
            "elapsed_str": str(timedelta(seconds=elapsed)),
            "output": state["output_path"],
            "size_mb": size_mb,
            "cpu_pct": round(read_cpu(), 1),
            "temp_c": round(read_temp(), 1),
            "rtmp_key_set": bool(state["rtmp_key"]),
            "calibration": state["calibration"],
            "calibrating": state["calibrating"],
            "resolution": f'{state["capture_width"]}x{state["capture_height"]}',
            "crf": state["crf"],
            "log": state["log_lines"][-10:],
        }


# ── HTML UI ──────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reco Game Control</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, system-ui, sans-serif; background: #1a1a2e; color: #eee; padding: 16px; max-width: 480px; margin: 0 auto; }
h1 { font-size: 1.4em; margin-bottom: 12px; text-align: center; }
.status { background: #16213e; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
.status-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1a1a3e; }
.status-row:last-child { border-bottom: none; }
.label { color: #888; }
.value { font-weight: bold; }
.value.green { color: #00ff88; }
.value.red { color: #ff4444; }
.value.yellow { color: #ffaa00; }
.buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }
button { padding: 16px; border: none; border-radius: 12px; font-size: 1.1em; font-weight: bold; cursor: pointer; transition: transform 0.1s; }
button:active { transform: scale(0.95); }
.btn-record { background: #e74c3c; color: white; }
.btn-record.active { background: #27ae60; }
.btn-stream { background: #3498db; color: white; }
.btn-stream.active { background: #27ae60; }
.btn-stop { background: #7f8c8d; color: white; }
input { width: 100%; padding: 12px; border: 1px solid #333; border-radius: 8px; background: #0f0f23; color: #eee; font-size: 1em; margin-bottom: 8px; }
.settings { background: #16213e; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
.settings h3 { margin-bottom: 8px; font-size: 1em; color: #888; }
.log { background: #0a0a1a; border-radius: 8px; padding: 8px; font-family: monospace; font-size: 0.75em; color: #888; max-height: 120px; overflow-y: auto; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.dot.on { background: #00ff88; animation: pulse 1s infinite; }
.dot.off { background: #555; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }

/* Score overlay controls */
.score-section { background: #16213e; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
.score-row { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
.score-row input { flex: 1; }
.score-btn { padding: 8px 16px; font-size: 1.2em; }
</style>
</head>
<body>
<h1>Reco Game Control</h1>

<div class="status" id="status">
    <div class="status-row">
        <span class="label">Recording</span>
        <span class="value" id="s-rec"><span class="dot off"></span>Off</span>
    </div>
    <div class="status-row">
        <span class="label">Streaming</span>
        <span class="value" id="s-stream"><span class="dot off"></span>Off</span>
    </div>
    <div class="status-row">
        <span class="label">FPS</span>
        <span class="value" id="s-fps">-</span>
    </div>
    <div class="status-row">
        <span class="label">Duration</span>
        <span class="value" id="s-elapsed">-</span>
    </div>
    <div class="status-row">
        <span class="label">File size</span>
        <span class="value" id="s-size">-</span>
    </div>
    <div class="status-row">
        <span class="label">CPU / Temp</span>
        <span class="value" id="s-hw">-</span>
    </div>
</div>

<div class="buttons">
    <button class="btn-record" id="btn-rec" onclick="toggleRecord()">Start Recording</button>
    <button class="btn-stream" id="btn-stream" onclick="toggleStream()">Start Stream</button>
    <button style="background:#8e44ad;color:white" id="btn-cal" onclick="calibrate()">Calibrate</button>
    <button style="background:#2c3e50;color:white" onclick="window.open('/roi','_blank')">Edit ROI</button>
</div>

<div class="settings">
    <h3>YouTube RTMP Key</h3>
    <input type="text" id="rtmp-key" placeholder="xxxx-xxxx-xxxx-xxxx" oninput="setConfig('rtmp_key', this.value)">
</div>

<div class="settings">
    <h3>Settings</h3>
    <div class="status-row">
        <span class="label">Resolution</span>
        <span class="value" id="s-res">-</span>
    </div>
    <div class="status-row">
        <span class="label">CRF</span>
        <span class="value" id="s-crf">-</span>
    </div>
    <div class="status-row">
        <span class="label">Calibration</span>
        <span class="value" id="s-cal" style="font-size:0.8em">-</span>
    </div>
</div>

<div class="score-section">
    <h3>Score Overlay</h3>
    <div class="score-row">
        <input type="text" id="team-home" placeholder="Home" value="Home">
        <button class="score-btn" onclick="adjustScore('home',-1)">-</button>
        <span id="score-home" style="font-size:1.5em;min-width:30px;text-align:center">0</span>
        <button class="score-btn" onclick="adjustScore('home',1)">+</button>
    </div>
    <div class="score-row">
        <input type="text" id="team-away" placeholder="Away" value="Away">
        <button class="score-btn" onclick="adjustScore('away',-1)">-</button>
        <span id="score-away" style="font-size:1.5em;min-width:30px;text-align:center">0</span>
        <button class="score-btn" onclick="adjustScore('away',1)">+</button>
    </div>
    <div class="score-row">
        <button onclick="toggleTimer()" id="btn-timer" style="flex:1;padding:12px;border-radius:8px;background:#2c3e50;color:white;border:none;font-size:1em">Start Match Timer</button>
        <span id="match-time" style="font-size:1.3em;min-width:60px;text-align:center">00:00</span>
    </div>
</div>

<div class="log" id="log"></div>

<script>
let scores = {home: 0, away: 0};
let timerRunning = false;
let timerStart = 0;
let timerInterval = null;

function api(path, body) {
    return fetch('/api/' + path, {
        method: body ? 'POST' : 'GET',
        headers: body ? {'Content-Type': 'application/json'} : {},
        body: body ? JSON.stringify(body) : null,
    }).then(r => r.json());
}

function updateStatus() {
    api('status').then(s => {
        const rec = document.getElementById('s-rec');
        const stream = document.getElementById('s-stream');
        rec.innerHTML = s.recording
            ? '<span class="dot on"></span><span class="green">Recording</span>'
            : '<span class="dot off"></span>Off';
        stream.innerHTML = s.streaming
            ? '<span class="dot on"></span><span class="green">Live</span>'
            : '<span class="dot off"></span>Off';
        document.getElementById('s-fps').textContent = s.fps + ' fps';
        document.getElementById('s-fps').className = 'value ' + (s.fps >= 28 ? 'green' : s.fps >= 20 ? 'yellow' : 'red');
        document.getElementById('s-elapsed').textContent = s.elapsed_str;
        document.getElementById('s-size').textContent = s.size_mb + ' MB';
        document.getElementById('s-hw').textContent = s.cpu_pct + '% / ' + s.temp_c + 'C';
        document.getElementById('s-res').textContent = s.resolution;
        document.getElementById('s-crf').textContent = 'CRF ' + s.crf;
        document.getElementById('s-cal').textContent = s.calibrating ? 'Running...' : (s.calibration || 'none');
        document.getElementById('log').textContent = (s.log || []).join('\\n');

        const btnCal = document.getElementById('btn-cal');
        if (s.calibrating) { btnCal.textContent = 'Calibrating...'; btnCal.disabled = true; }
        else { btnCal.textContent = 'Calibrate'; btnCal.disabled = false; }

        const btnRec = document.getElementById('btn-rec');
        btnRec.textContent = s.recording ? 'Stop Recording' : 'Start Recording';
        btnRec.className = s.recording ? 'btn-record active' : 'btn-record';
        const btnStream = document.getElementById('btn-stream');
        btnStream.textContent = s.streaming ? 'Stop Stream' : 'Start Stream';
        btnStream.className = s.streaming ? 'btn-stream active' : 'btn-stream';
    }).catch(() => {});
}

function toggleRecord() {
    api('status').then(s => {
        if (s.recording) api('stop', {}).then(updateStatus);
        else api('start', {}).then(updateStatus);
    });
}

function toggleStream() {
    api('status').then(s => {
        if (s.streaming) api('stream/stop', {}).then(updateStatus);
        else api('stream/start', {}).then(updateStatus);
    });
}

function setConfig(key, value) {
    api('config', {[key]: value});
}

function calibrate() {
    const btn = document.getElementById('btn-cal');
    btn.textContent = 'Calibrating...';
    btn.disabled = true;
    api('calibrate', {}).then(r => {
        btn.disabled = false;
        btn.textContent = 'Calibrate';
    });
}

function adjustScore(team, delta) {
    scores[team] = Math.max(0, scores[team] + delta);
    document.getElementById('score-' + team).textContent = scores[team];
    api('score', {
        home: scores.home, away: scores.away,
        home_name: document.getElementById('team-home').value,
        away_name: document.getElementById('team-away').value,
    });
}

function toggleTimer() {
    if (timerRunning) {
        clearInterval(timerInterval);
        timerRunning = false;
        document.getElementById('btn-timer').textContent = 'Resume Timer';
    } else {
        if (!timerStart) timerStart = Date.now();
        timerRunning = true;
        document.getElementById('btn-timer').textContent = 'Pause Timer';
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - timerStart) / 1000);
            const m = Math.floor(elapsed / 60);
            const s = elapsed % 60;
            document.getElementById('match-time').textContent =
                String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
        }, 1000);
    }
}

setInterval(updateStatus, 2000);
updateStatus();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default logging

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == "/roi":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(ROI_HTML.encode())
        elif self.path == "/api/status":
            self.json_response(get_status())
        elif self.path.startswith("/api/snapshot/"):
            side = self.path.split("/")[-1].split("?")[0]
            device = state["left_device"] if side == "left" else state["right_device"]
            data = capture_snapshot(device)
            if data:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(500, "Failed to capture snapshot")
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}

        if self.path == "/api/start":
            self.json_response(start_recording())
        elif self.path == "/api/stop":
            self.json_response(stop_recording())
        elif self.path == "/api/stream/start":
            self.json_response(start_streaming())
        elif self.path == "/api/stream/stop":
            self.json_response(stop_streaming())
        elif self.path == "/api/config":
            with lock:
                for k, v in body.items():
                    if k in state:
                        state[k] = v
            self.json_response({"status": "ok"})
        elif self.path == "/api/calibrate":
            # Run in a thread so the HTTP response returns immediately.
            threading.Thread(target=self._run_calibrate, daemon=True).start()
            self.json_response({"status": "calibration started"})
        elif self.path == "/api/roi":
            self.json_response(save_roi(body))
        elif self.path == "/api/score":
            # Score data received - could be used for overlay in the future
            self.json_response({"status": "ok"})
        else:
            self.send_error(404)

    def _run_calibrate(self):
        result = run_calibration()
        with lock:
            state["last_calibration_result"] = result

    def json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


def main():
    parser = argparse.ArgumentParser(description="Reco game-day control panel")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calibration", "-c", default="")
    parser.add_argument("--left-device", default="0")
    parser.add_argument("--right-device", default="1")
    args = parser.parse_args()

    state["calibration"] = args.calibration
    state["left_device"] = args.left_device
    state["right_device"] = args.right_device

    # Auto-find calibration if not specified
    if not args.calibration:
        for p in [Path.home() / "match_4032_outdoor.json",
                  Path.home() / "match_daylight.json",
                  Path.home() / "match.json"]:
            if p.exists():
                state["calibration"] = str(p)
                break

    # Enable jetson_clocks and fan
    os.system("sudo jetson_clocks 2>/dev/null")
    os.system("sudo sh -c 'echo 255 > /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1' 2>/dev/null")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    ip = subprocess.getoutput("hostname -I").split()[0] if subprocess.getoutput("hostname -I") else "localhost"
    print(f"\n  Reco Game Control Panel")
    print(f"  Open on your phone: http://{ip}:{args.port}")
    print(f"  Calibration: {state['calibration'] or 'NONE - set via --calibration'}")
    print(f"  Cameras: left={args.left_device} right={args.right_device}")
    print(f"  Resolution: {state['capture_width']}x{state['capture_height']}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_streaming()
        stop_recording()
        server.shutdown()


if __name__ == "__main__":
    main()
