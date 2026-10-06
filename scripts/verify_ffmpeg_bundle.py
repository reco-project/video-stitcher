#!/usr/bin/env python3
"""Verify a relocated FFmpeg package without developer loader environment variables."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from bundle_linux import SYSTEM_LIBRARIES, inspect


def verify(directory: Path):
    directory = directory.resolve(strict=True)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("LD_", "FFMPEG_", "PKG_CONFIG", "ORT_"))}
    env["PATH"] = str(directory) + os.pathsep + env.get("PATH", os.defpath)
    manifest = json.loads(Path(__file__).with_name("ffmpeg-sdk.json").read_text())
    ffmpeg, ffprobe = directory / "ffmpeg", directory / "ffprobe"

    def run(binary, *args, input=None):
        return subprocess.check_output([str(binary), *args], input=input, env=env, cwd="/", stderr=subprocess.PIPE, timeout=120)

    io_tests = directory / "reco-io-tests"
    binaries = [ffmpeg, ffprobe]
    if io_tests.is_file():
        binaries.append(io_tests)
    detect_tests = directory / "reco-detect-tests"
    if detect_tests.is_file():
        binaries.append(detect_tests)
    reco = directory / "reco"
    if reco.is_file():
        binaries.append(reco)
    gui = directory / "reco-gui"
    if gui.is_file():
        binaries.append(gui)
    for binary in binaries:
        interpreter, = inspect("--print-interpreter", binary)
        linked = run(interpreter, "--list", str(binary)).decode()
        for line in linked.splitlines():
            if "=>" not in line:
                continue
            name, target = line.strip().split(" => ", 1)
            path = Path(target.rsplit(" (", 1)[0])
            if name not in SYSTEM_LIBRARIES and path != directory / name:
                raise ValueError(f"library escaped the bundle: {line.strip()}")
        if binary in (ffmpeg, ffprobe):
            version = run(binary, "-version").decode().splitlines()[0]
            if not version.startswith(f"{binary.name} version {manifest['version']} "):
                raise ValueError(f"unexpected runtime: {version}")
    protocols = run(ffmpeg, "-hide_banner", "-protocols").decode().split()
    for protocol in ("file", "tcp", "rtmp", "tls", "https", "rtmps"):
        if protocol not in protocols:
            raise ValueError(f"missing protocol: {protocol}")
    png = run(ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=size=8x8:color=red",
              "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-")
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("PNG support is missing")
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / "roundtrip.mp4"
        source = ["-v", "error", "-f", "lavfi", "-i", "testsrc2=size=192x108:rate=30", "-frames:v", "12"]
        expected = run(ffmpeg, *source, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-")
        run(ffmpeg, *source, "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", str(output))
        actual = run(ffmpeg, "-v", "error", "-i", str(output), "-pix_fmt", "yuv420p", "-f", "rawvideo", "-")
        if actual != expected or len(actual) != 192 * 108 * 3 // 2 * 12:
            raise ValueError("lossless decode/encode round trip changed pixels or frame count")
        probe = json.loads(run(ffprobe, "-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "json", str(output)))
        if probe["streams"][0]["nb_read_frames"] != "12":
            raise ValueError("ffprobe frame count mismatch")
        if reco.is_file():
            # Distinct solid colours expose missing/reordered frames and black borders
            # without making codec pixel rounding a projection-geometry oracle.
            colours = [(40 + 12 * i, 180 - 8 * i, 80 + 7 * i) for i in range(12)]
            raw = b"".join(bytes(colour) * (192 * 108) for colour in colours)
            input_video = Path(scratch) / "colours.mp4"
            run(ffmpeg, "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", "192x108", "-framerate", "30", "-i", "pipe:0",
                "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-colorspace", "bt709", "-color_range", "tv",
                "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(input_video), input=raw)
            calibration = Path(scratch) / "cylinder.json"
            calibration.write_text(json.dumps({
                "lenses": [{"width": 192, "height": 108, "fx": 192.0, "fy": 108.0,
                            "cx": 96.0, "cy": 54.0, "distortion": [0.0] * 4, "correction": 0.0}],
                "projection": {"type": "cylinder", "radius": 48.0},
                "framing": {"axis_offset": 0.0},
            }))
            rendered = Path(scratch) / "rendered.mp4"
            run(reco, "stitch", str(input_video), "-c", str(calibration), "-o", str(rendered),
                "--cpu", "--width", "192", "--height", "108", "--encoder", "libx264",
                "--preset", "ultrafast", "--quality-value", "100", "--max-frames", "12")
            pixels = run(ffmpeg, "-v", "error", "-i", str(rendered), "-pix_fmt", "rgb24", "-f", "rawvideo", "-")
            if len(pixels) != len(raw) or any(abs(a - b) > 8 for a, b in zip(pixels, raw)):
                raise ValueError("packaged CLI changed frame order, colour or coverage in the CPU export")
            print("Packaged CLI: 12 ordered CPU-rendered colour frames passed (8/255 channel tolerance).")
    print("Pinned runtime, bundled dependency resolution, protocol availability and 12-frame lossless round trip passed.")
    if io_tests.is_file():
        case = "ffmpeg::output_io::tests::overflow::fifo_overflow_resumes_on_video_keyframes_with_audio"
        listed = run(io_tests, case, "--exact", "--list").decode().splitlines()
        if f"{case}: test" not in listed:
            raise ValueError("the packaged Reco I/O recovery regression is missing")
        print(run(io_tests, case, "--exact", "--ignored", "--test-threads=1").decode())
    if detect_tests.is_file():
        case = "ort_session::provider_tests::required_host_session_runs_identity"
        if f"{case}: test" not in run(detect_tests, case, "--exact", "--list").decode().splitlines():
            raise ValueError("the packaged mandatory ONNX inference test is missing")
        print(run(detect_tests, case, "--exact", "--ignored", "--test-threads=1").decode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        verify(args.directory)
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr.decode(errors="replace"), file=sys.stderr)
        parser.exit(1, f"Bundle verification failed: {error}\n")
