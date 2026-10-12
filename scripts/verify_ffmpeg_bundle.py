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
from bundle_macos import verify_loaded_libraries
from fetch_windows_sources import digest, relative_path


def verify_media_features(run, ffmpeg):
    """Require the transport and codecs used by recording and live publishing."""
    protocols = run(ffmpeg, "-hide_banner", "-protocols").decode().split()
    for protocol in ("file", "tcp", "rtmp", "tls", "https", "rtmps"):
        if protocol not in protocols:
            raise ValueError(f"missing protocol: {protocol}")
    for category, required in (
        ("demuxers", ("rtsp", "mov")),
        ("muxers", ("fifo", "flv", "mp4")),
        ("decoders", ("h264", "aac")),
        ("encoders", ("libx264", "aac")),
    ):
        available = set()
        for line in run(ffmpeg, "-hide_banner", f"-{category}").decode().splitlines():
            fields = line.split()
            if len(fields) >= 2:
                available.update(fields[1].split(","))
        for name in required:
            if name not in available:
                raise ValueError(f"missing {category}: {name}")


def verify(directory: Path):
    directory = directory.resolve(strict=True)
    windows = sys.platform == "win32"
    macos = sys.platform == "darwin"
    suffix = ".exe" if windows else ""
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("LD_", "DYLD_", "FFMPEG_", "PKG_CONFIG", "ORT_"))}
    if windows:
        # Exclude SDK/developer directories, retaining Windows and its driver stack.
        env = {key: value for key, value in env.items() if key.upper() != "PATH"}
        env["PATH"] = str(directory) + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32")
        for name, checksum in json.loads((directory / "files.json").read_text()).items():
            if digest(directory / relative_path(name)) != checksum:
                raise ValueError(f"package checksum mismatch: {name}")
    elif macos:
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        env["DYLD_PRINT_LIBRARIES"] = "1"
        for name, record in json.loads((directory / "mach-o.json").read_text()).items():
            path = directory / relative_path(name)
            if path.is_symlink() or digest(path) != record["output_sha256"]:
                raise ValueError(f"package checksum mismatch: {name}")
    else:
        env["PATH"] = str(directory) + os.pathsep + env.get("PATH", os.defpath)
    manifest = json.loads(Path(__file__).with_name("ffmpeg-sdk.json").read_text())
    ffmpeg, ffprobe = directory / f"ffmpeg{suffix}", directory / f"ffprobe{suffix}"

    def run(binary, *args, input=None):
        result = subprocess.run([str(binary), *args], input=input, env=env, cwd=directory.parent,
                                capture_output=True, timeout=120, check=True)
        if macos:
            verify_loaded_libraries(result.stderr.decode(errors="replace"), directory)
        return result.stdout

    io_tests = directory / f"reco-io-tests{suffix}"
    binaries = [ffmpeg, ffprobe]
    if io_tests.is_file():
        binaries.append(io_tests)
    detect_tests = directory / f"reco-detect-tests{suffix}"
    if detect_tests.is_file():
        binaries.append(detect_tests)
    reco = directory / f"reco{suffix}"
    if reco.is_file():
        binaries.append(reco)
    gui = directory / f"reco-gui{suffix}"
    if gui.is_file():
        binaries.append(gui)
    for binary in binaries:
        if not windows and not macos:
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
            if not version.startswith(f"{binary.stem if windows else binary.name} version {manifest['version']} "):
                raise ValueError(f"unexpected runtime: {version}")
    verify_media_features(run, ffmpeg)
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
    print("Pinned runtime, protocol availability and 12-frame lossless round trip passed.")
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
