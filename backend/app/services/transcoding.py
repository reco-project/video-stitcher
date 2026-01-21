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
from typing import Tuple, Optional, List
import numpy as np
import soundfile as sf
from scipy.signal import correlate
from app.utils.logger import get_logger
from app.config import get_settings

logger = get_logger(__name__)

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
    video1_path: str,
    video2_path: str,
    output_path: str,
    temp_dir: Optional[str] = None,
    progress_callback=None,
) -> Tuple[str, float]:
    """
    Synchronize and stack two videos vertically.

    Args:
        video1_path: Path to first (top) video
        video2_path: Path to second (bottom) video
        output_path: Path for output stacked video
        temp_dir: Optional temporary directory (auto-created if None)
        progress_callback: Optional callback function(dict) for progress updates

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

        if progress_callback:
            progress_callback({'stage': 'audio_extraction', 'message': 'Extracting audio from left video...'})
        _extract_audio(video1_path, audio1_path)

        if progress_callback:
            progress_callback({'stage': 'audio_extraction', 'message': 'Extracting audio from right video...'})
        _extract_audio(video2_path, audio2_path)

        # Compute synchronization offset
        if progress_callback:
            progress_callback({'stage': 'audio_sync', 'message': 'Computing audio synchronization...'})
        offset = _compute_offset(audio1_path, audio2_path)

        if progress_callback:
            progress_callback(
                {'stage': 'encoding', 'message': 'Encoding stacked video...', 'offset_seconds': round(offset, 2)}
            )

        # Stack videos with computed offset
        _stack_videos(video1_path, video2_path, offset, output_path, progress_callback)

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


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        pass
    return 0.0


def _get_video_fps(video_path: str) -> float:
    """Get video frame rate using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            # Parse fraction like "30000/1001" or simple number like "30"
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = fps_str.split('/')
                return float(num) / float(den)
            return float(fps_str)
    except (subprocess.SubprocessError, ValueError):
        pass
    return 30.0  # Default fallback


def _get_available_encoders() -> List[str]:
    """
    Get list of available GPU encoders on this system.
    Actually tests each encoder to ensure hardware is present.

    Returns:
        List of available encoder names
    """
    encoders = ["h264_nvenc", "h264_qsv", "h264_amf"]
    available = []

    for enc in encoders:
        # Test if encoder actually works by trying to initialize it
        # Generate a tiny 1-frame test video (256x256 minimum for hardware encoders)
        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1", "-c:v", enc, "-f", "null", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )

            # If the command succeeded, the encoder is available
            if result.returncode == 0:
                available.append(enc)
                logger.info(f"Encoder {enc} is available and working")
            else:
                logger.debug(f"Encoder {enc} failed test: {result.stderr[:200]}")

        except subprocess.SubprocessError as e:
            logger.debug(f"Encoder {enc} test error: {e}")
            pass

    return available


def _detect_gpu_encoder() -> str:
    """
    Detect available GPU encoder based on user settings.

    Returns:
        Encoder name (h264_nvenc, h264_qsv, h264_amf, or libx264)
    """
    settings = get_settings()
    encoder_pref = settings.encoder

    # If user specified a specific encoder, try to use it
    if encoder_pref != "auto":
        available = _get_available_encoders()
        if encoder_pref in available:
            logger.info(f"Using user-preferred encoder: {encoder_pref}")
            return encoder_pref
        elif encoder_pref == "libx264":
            logger.info("Using CPU encoder (libx264) as per user preference")
            return "libx264"
        else:
            logger.warning(f"Preferred encoder {encoder_pref} not available, falling back to auto-detect")

    # Auto-detect: Prefer NVIDIA > Intel > AMD
    available = _get_available_encoders()
    for enc in ["h264_nvenc", "h264_qsv", "h264_amf"]:
        if enc in available:
            logger.info(f"Auto-detected encoder: {enc}")
            return enc

    logger.info("No GPU encoder available, using CPU encoder (libx264)")
    return "libx264"


def _stack_videos(
    video1_path: str,
    video2_path: str,
    offset: float,
    output_path: str,
    progress_callback=None,
) -> None:
    """Stack two videos vertically with audio sync offset.

    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        offset: Audio sync offset in seconds
        output_path: Path for output video
        progress_callback: Optional callback function(dict) for progress updates
    """
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
            "-progress",
            "pipe:1",
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
            # Get total duration and frame rate for progress calculation
            total_duration = _get_video_duration(video1_path)
            frame_rate = _get_video_fps(video1_path)
            total_frames = int(total_duration * frame_rate) if frame_rate > 0 else 0

            print(f"[TRANSCODE] Video: {total_duration:.2f}s @ {frame_rate:.2f} fps = ~{total_frames} frames")
            logger.info(
                f"Video duration detected: {total_duration:.2f}s @ {frame_rate:.2f} fps = ~{total_frames} frames"
            )
            logger.info("Will attempt using " + enc)

            # Run FFmpeg with progress monitoring
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Read stderr in a separate thread to prevent blocking
            import threading

            stderr_lines = []

            def read_stderr():
                for line in process.stderr:
                    stderr_lines.append(line)

            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            # Parse progress output in real-time from stdout
            progress_data = {}
            progress_count = 0
            last_frame_count = 0

            for line in process.stdout:
                line = line.strip()

                if '=' in line:
                    key, value = line.split('=', 1)
                    progress_data[key] = value

                    # When we get 'progress' key, we have a complete progress update
                    if key == 'progress':
                        progress_count += 1

                        if progress_callback and (total_duration > 0 or total_frames > 0):
                            try:
                                # Extract frame count - this is usually reliable
                                frame_str = progress_data.get('frame', '0')
                                current_frame = int(frame_str) if frame_str.isdigit() else 0

                                # Calculate progress based on frames if available
                                if total_frames > 0 and current_frame > 0:
                                    progress_percent = min(100, (current_frame / total_frames) * 100)
                                    current_time = current_frame / frame_rate if frame_rate > 0 else 0
                                else:
                                    # Fallback to time-based if we have it
                                    out_time_ms = progress_data.get('out_time_ms', 'N/A')
                                    if out_time_ms != 'N/A' and out_time_ms.replace('.', '').isdigit():
                                        current_time = float(out_time_ms) / 1000.0
                                        progress_percent = (
                                            min(100, (current_time / total_duration) * 100) if total_duration > 0 else 0
                                        )
                                    else:
                                        current_time = 0
                                        progress_percent = 0

                                # Extract FPS and speed - these show encoding performance
                                fps_str = progress_data.get('fps', '0.00')
                                encoding_fps = (
                                    float(fps_str) if fps_str != 'N/A' and fps_str.replace('.', '').isdigit() else 0.0
                                )

                                speed = progress_data.get('speed', '0x')
                                if speed == 'N/A':
                                    speed = '0x'
                                speed = speed.rstrip('x')

                                bitrate = progress_data.get('bitrate', '0')
                                if bitrate == 'N/A':
                                    bitrate = '0kbits/s'

                                # Only send updates with meaningful progress or first few updates
                                if current_frame != last_frame_count or progress_count <= 3:
                                    last_frame_count = current_frame

                                    progress_callback(
                                        {
                                            'stage': 'encoding',
                                            'progress_percent': round(progress_percent, 1),
                                            'current_time': round(current_time, 1),
                                            'total_duration': round(total_duration, 1),
                                            'fps': round(encoding_fps, 1),
                                            'speed': speed,
                                            'frame': str(current_frame),
                                            'total_frames': str(total_frames),
                                            'bitrate': bitrate,
                                            'encoder': enc,
                                        }
                                    )
                            except (ValueError, ZeroDivisionError) as e:
                                if progress_count <= 3:
                                    print(f"[TRANSCODE] Parse error: {e}")
                                    logger.warning(f"Progress parse error: {e}")
                        elif not progress_callback:
                            print(f"[TRANSCODE] WARNING: No progress_callback provided")
                            logger.warning(f"No progress_callback provided")
                        elif total_duration <= 0:
                            print(f"[TRANSCODE] WARNING: Invalid total_duration: {total_duration}")
                            logger.warning(f"Invalid total_duration: {total_duration}")

                        progress_data = {}  # Reset for next update

            print(f"[TRANSCODE] Encoding complete. Total progress updates: {progress_count}")
            logger.info(f"FFmpeg encoding complete. Total progress updates: {progress_count}")

            # Wait for process to complete
            process.wait(timeout=3600)
            stderr_thread.join(timeout=5)

            if process.returncode != 0:
                stderr = ''.join(stderr_lines)
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr=stderr)

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


def extract_preview_frame(video_path: str, output_path: str, timestamp: float = 1.0) -> str:
    """
    Extract a single frame from video as a preview thumbnail.

    Args:
            video_path: Path to input video
            output_path: Path for output preview image (JPEG)
            timestamp: Time in seconds to extract frame from (default: 1.0 second)

    Returns:
            Path to output preview image

    Raises:
            RuntimeError: If frame extraction fails
            FileNotFoundError: If video not found
    """
    check_ffmpeg()

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        # Use ffmpeg to extract frame at specified timestamp
        # -ss: seek to timestamp (fast seeking)
        # -vframes 1: extract only 1 frame
        # -q:v 2: quality (1-31, lower is better)
        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-vframes",
            "1",
            "-q:v",
            "2",
            "-y",  # Overwrite output file
            output_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        if not os.path.exists(output_path):
            raise RuntimeError("Preview image was not created")

        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("Frame extraction timed out")
    except Exception as e:
        # Clean up partial output
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        raise RuntimeError(f"Frame extraction failed: {str(e)}") from e
