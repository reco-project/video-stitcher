"""
Video transcoding service for synchronization and stacking.

Handles audio-based video synchronization and vertical stacking
of left/right camera feeds into a single video file.
"""

import os
import subprocess
import uuid
import shutil
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
from scipy.signal import correlate


# Error messages
ERROR_MESSAGES = {
    "FFMPEG_NOT_FOUND": "FFmpeg not installed. Please install from https://ffmpeg.org/download.html",
    "VIDEO_NOT_FOUND": "Video file not found",
    "AUDIO_EXTRACTION_FAILED": "Failed to extract audio from video",
    "SYNC_FAILED": "Failed to compute audio synchronization",
    "STACKING_FAILED": "Failed to stack videos",
}


def check_ffmpeg() -> None:
    """
    Check if FFmpeg is available.

    Raises:
        RuntimeError: If FFmpeg is not found
    """
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        raise RuntimeError(ERROR_MESSAGES["FFMPEG_NOT_FOUND"])


def transcode_and_stack(
    video1_path: str, video2_path: str, output_path: str, temp_dir: Optional[str] = None
) -> Tuple[str, float]:
    """
    Synchronize and stack two videos vertically.

    Args:
        video1_path: Path to first (top) video
        video2_path: Path to second (bottom) video
        output_path: Path for output stacked video
        temp_dir: Optional temporary directory (auto-created if None)

    Returns:
        Tuple of (output_path, offset_seconds)

    Raises:
        RuntimeError: If transcoding fails
        FileNotFoundError: If input videos not found
    """
    # Check FFmpeg availability
    check_ffmpeg()

    # Validate input files
    if not os.path.exists(video1_path):
        raise FileNotFoundError(f"Video not found: {video1_path}")
    if not os.path.exists(video2_path):
        raise FileNotFoundError(f"Video not found: {video2_path}")

    # Create temp directory if not provided
    cleanup_temp = False
    if temp_dir is None:
        temp_dir = os.path.join("backend", "temp", str(uuid.uuid4()))
        cleanup_temp = True

    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Extract audio from both videos
        audio1_path = os.path.join(temp_dir, "audio1.wav")
        audio2_path = os.path.join(temp_dir, "audio2.wav")

        _extract_audio(video1_path, audio1_path)
        _extract_audio(video2_path, audio2_path)

        # Compute synchronization offset
        offset = _compute_offset(audio1_path, audio2_path)

        # Stack videos with computed offset
        _stack_videos(video1_path, video2_path, offset, output_path)

        return output_path, offset

    except Exception as e:
        # Clean up output file if it was created
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Transcoding failed: {str(e)}") from e

    finally:
        # Clean up temp directory if we created it
        if cleanup_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def _extract_audio(video_path: str, output_path: str) -> None:
    """Extract audio from video as mono 16kHz WAV."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Audio extraction failed: {e}")


def _compute_offset(audio1_path: str, audio2_path: str) -> float:
    """
    Compute time offset between two audio files using cross-correlation.

    Returns:
        Offset in seconds (positive if audio2 should be delayed)
    """
    try:
        a1, sr1 = sf.read(audio1_path)
        a2, sr2 = sf.read(audio2_path)

        if sr1 != sr2:
            raise ValueError(f"Sample rate mismatch: {sr1} vs {sr2}")

        # Remove DC offset
        a1 = a1 - np.mean(a1)
        a2 = a2 - np.mean(a2)

        # Compute cross-correlation
        corr = correlate(a1, a2, mode="full")
        lag = np.argmax(corr) - len(a2)

        return lag / sr1

    except Exception as e:
        raise RuntimeError(f"Sync computation failed: {e}") from e


def _detect_gpu_encoder() -> str:
    """
    Detect available GPU encoder.

    Returns:
        Encoder name (h264_nvenc, h264_qsv, h264_amf, or libx264)
    """
    encoders = {"h264_nvenc": "NVIDIA", "h264_qsv": "Intel", "h264_amf": "AMD"}

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )

        available = [enc for enc in encoders if enc in result.stdout]

        # Prefer NVIDIA > Intel > AMD
        for enc in ["h264_nvenc", "h264_qsv", "h264_amf"]:
            if enc in available:
                return enc

    except subprocess.SubprocessError:
        pass

    return "libx264"


def _stack_videos(video1_path: str, video2_path: str, offset: float, output_path: str) -> None:
    """Stack two videos vertically with audio sync offset."""
    offset_str = f"{offset:.2f}"
    encoder = _detect_gpu_encoder()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try with detected encoder first, fallback to libx264 if it fails
    encoders_to_try = [encoder] if encoder != "libx264" else ["libx264"]
    if encoder != "libx264":
        encoders_to_try.append("libx264")  # Always have software fallback

    last_error = None

    for enc in encoders_to_try:
        # Clean up any failed output from previous attempt
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            video1_path,
            "-itsoffset",
            offset_str,
            "-i",
            video2_path,
            "-filter_complex",
            "[0:v][1:v]vstack=inputs=2[vout];[vout]scale=1920:2160[vscaled]",
            "-map",
            "[vscaled]",
            "-c:v",
            enc,
            "-preset",
            "fast",
            "-b:v",
            "30M",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-shortest",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, timeout=3600, text=True)

            # Verify output file was created and has content
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file was not created: {output_path}")

            file_size = os.path.getsize(output_path)
            if file_size == 0:
                raise RuntimeError(f"Output file is empty (0 bytes)")

            print(f"Successfully encoded with {enc} ({file_size} bytes)")
            return  # Success!
        except subprocess.CalledProcessError as e:
            last_error = e
            stderr = e.stderr if e.stderr else ""
            print(f"Encoding with {enc} failed: {stderr}")

            # If this was a hardware encoder and it failed, try software
            if enc != "libx264":
                print(f"Falling back to software encoding (libx264)...")
                continue
            else:
                # Software encoding also failed, raise the error
                raise RuntimeError(f"Video stacking failed: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Video stacking timed out after 1 hour: {e}") from e

    # If we got here, all encoders failed
    if last_error:
        raise RuntimeError(f"Video stacking failed with all encoders: {last_error}") from last_error
