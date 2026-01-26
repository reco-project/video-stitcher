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


def _get_encoding_params(quality_settings):
    """Get encoding parameters from quality settings.

    Args:
        quality_settings: Dict with quality settings from frontend

    Returns:
        Tuple of (bitrate, speed_preset, resolution)
    """
    # Default values if no quality settings provided
    bitrate = "50M"
    speed_preset = "medium"
    resolution = "1080p"

    if quality_settings:
        bitrate = quality_settings.get("bitrate", "50M")
        speed_preset = quality_settings.get("speed_preset", "medium")
        resolution = quality_settings.get("resolution", "1080p")

    return bitrate, speed_preset, resolution


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


def concat_videos(video_list: List[str], output_path: str, progress_callback=None) -> str:
    """
    Concatenate multiple video files into a single file using FFmpeg concat demuxer.

    This is useful for GoPro-style split recordings where a single session
    is automatically split into multiple files.

    Args:
        video_list: List of video file paths to concatenate (in order)
        output_path: Path for the concatenated output video
        progress_callback: Optional callback for progress updates

    Returns:
        Path to the concatenated video

    Raises:
        RuntimeError: If concatenation fails
        FileNotFoundError: If any input video is not found
    """
    if not video_list:
        raise ValueError("video_list cannot be empty")

    # If only one video, just return its path (no concatenation needed)
    if len(video_list) == 1:
        return video_list[0]

    # Validate all input files exist
    for video_path in video_list:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

    # Create concat list file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for video_path in video_list:
            # Use absolute paths and escape single quotes
            abs_path = os.path.abspath(video_path)
            f.write(f"file '{abs_path}'\n")
        list_path = f.name

    try:
        if progress_callback:
            progress_callback({'stage': 'concatenation', 'message': f'Concatenating {len(video_list)} video files...'})

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",  # Stream copy (no re-encoding) for speed
            output_path,
        ]

        logger.info(f"Concatenating {len(video_list)} videos to {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        logger.info(f"Concatenation complete: {output_path}")
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("Video concatenation timed out")
    finally:
        # Clean up temp concat list file
        if os.path.exists(list_path):
            os.remove(list_path)


def transcode_and_stack_multiple(
    left_videos: List[str],
    right_videos: List[str],
    output_path: str,
    temp_dir: Optional[str] = None,
    progress_callback=None,
    cancellation_check=None,
    process_callback=None,
    quality_settings=None,
) -> Tuple[str, float]:
    """
    Concatenate and then synchronize and stack multiple videos from each side.

    This is the main entry point for processing matches with multiple video files
    per camera (e.g., GoPro split recordings).

    Args:
        left_videos: List of paths to left/top camera videos (in order)
        right_videos: List of paths to right/bottom camera videos (in order)
        output_path: Path for output stacked video
        temp_dir: Optional temporary directory (auto-created if None)
        progress_callback: Optional callback function(dict) for progress updates
        cancellation_check: Optional callback function() -> bool to check if cancelled
        process_callback: Optional callback function(pid: int) to receive FFmpeg PID
        quality_settings: Optional dict with quality settings

    Returns:
        Tuple of (output_path, offset_seconds)

    Raises:
        RuntimeError: If processing fails
        FileNotFoundError: If input videos not found
        ValueError: If video lists are empty
    """
    if not left_videos:
        raise ValueError("left_videos list cannot be empty")
    if not right_videos:
        raise ValueError("right_videos list cannot be empty")

    # Check FFmpeg availability
    check_ffmpeg()

    # Create temp directory if not provided
    cleanup_temp = False
    if temp_dir is None:
        temp_dir = os.path.join("backend", "temp", str(uuid.uuid4()))
        cleanup_temp = True

    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Step 1: Concatenate videos if multiple files per side
        left_concat_path = left_videos[0]  # Default to first if only one
        right_concat_path = right_videos[0]

        if len(left_videos) > 1:
            if progress_callback:
                progress_callback(
                    {
                        'stage': 'concatenation',
                        'message': f'Concatenating {len(left_videos)} left camera videos...',
                        'concat_side': 'left',
                        'concat_count': len(left_videos),
                    }
                )

            if cancellation_check and cancellation_check():
                raise RuntimeError("Processing cancelled during left video concatenation")

            left_concat_path = os.path.join(temp_dir, "left_concat.mp4")
            concat_videos(left_videos, left_concat_path, progress_callback)
            logger.info(f"Left videos concatenated: {len(left_videos)} files -> {left_concat_path}")

        if len(right_videos) > 1:
            if progress_callback:
                progress_callback(
                    {
                        'stage': 'concatenation',
                        'message': f'Concatenating {len(right_videos)} right camera videos...',
                        'concat_side': 'right',
                        'concat_count': len(right_videos),
                    }
                )

            if cancellation_check and cancellation_check():
                raise RuntimeError("Processing cancelled during right video concatenation")

            right_concat_path = os.path.join(temp_dir, "right_concat.mp4")
            concat_videos(right_videos, right_concat_path, progress_callback)
            logger.info(f"Right videos concatenated: {len(right_videos)} files -> {right_concat_path}")

        # Step 2: Now use the existing transcode_and_stack with the concatenated videos
        return transcode_and_stack(
            left_concat_path,
            right_concat_path,
            output_path,
            temp_dir,
            progress_callback,
            cancellation_check,
            process_callback,
            quality_settings,
        )

    except Exception as e:
        # Clean up output file if it was created
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

    finally:
        # Clean up temp directory if we created it
        if cleanup_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def transcode_and_stack(
    video1_path: str,
    video2_path: str,
    output_path: str,
    temp_dir: Optional[str] = None,
    progress_callback=None,
    cancellation_check=None,
    process_callback=None,
    quality_settings=None,
) -> Tuple[str, float]:
    """
    Synchronize and stack two videos vertically.

    Args:
        video1_path: Path to first (top) video
        video2_path: Path to second (bottom) video
        output_path: Path for output stacked video
        temp_dir: Optional temporary directory (auto-created if None)
        progress_callback: Optional callback function(dict) for progress updates
        cancellation_check: Optional callback function() -> bool to check if cancelled
        process_callback: Optional callback function(pid: int) to receive FFmpeg PID
        quality_settings: Optional dict with quality settings (preset, codec, crf, bitrate, speed_preset, use_gpu_decode)

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
        _stack_videos(
            video1_path,
            video2_path,
            offset,
            output_path,
            progress_callback,
            cancellation_check,
            process_callback,
            quality_settings,
        )

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
        Offset in seconds.

        Sign convention (as used by `_stack_videos`):
        - offset > 0 means audio2 starts later than audio1 (audio1 is "early")
        - offset < 0 means audio1 starts later than audio2 (audio2 is "early")
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


def _get_video_resolution(video_path: str) -> tuple[int, int]:
    """Get video resolution (width, height) using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            w_str, h_str = result.stdout.strip().split("x", 1)
            return int(w_str), int(h_str)
    except (subprocess.SubprocessError, ValueError):
        pass
    return 0, 0


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


def _map_preset_for_encoder(speed_preset, encoder):
    """Map software encoder presets to hardware encoder presets.

    Args:
        speed_preset: Software preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower)
        encoder: Encoder name (h264_nvenc, hevc_nvenc, h264_qsv, etc.)

    Returns:
        Mapped preset suitable for the encoder
    """
    # Software encoders (libx264, libx265) use the preset as-is
    if encoder.startswith("lib"):
        return speed_preset

    # NVENC (NVIDIA) preset mapping
    if "_nvenc" in encoder:
        nvenc_map = {
            "ultrafast": "p1",
            "superfast": "p2",
            "veryfast": "p3",
            "faster": "p4",
            "fast": "p5",
            "medium": "p6",
            "slow": "p7",
            "slower": "p7",
        }
        return nvenc_map.get(speed_preset, "p6")

    # QSV (Intel) preset mapping
    if "_qsv" in encoder:
        qsv_map = {
            "ultrafast": "veryfast",
            "superfast": "veryfast",
            "veryfast": "veryfast",
            "faster": "faster",
            "fast": "fast",
            "medium": "medium",
            "slow": "slow",
            "slower": "slower",
        }
        return qsv_map.get(speed_preset, "medium")

    # AMF (AMD) preset mapping
    if "_amf" in encoder:
        amf_map = {
            "ultrafast": "speed",
            "superfast": "speed",
            "veryfast": "speed",
            "faster": "balanced",
            "fast": "balanced",
            "medium": "balanced",
            "slow": "quality",
            "slower": "quality",
        }
        return amf_map.get(speed_preset, "balanced")

    return speed_preset


def _stack_videos(
    video1_path: str,
    video2_path: str,
    offset: float,
    output_path: str,
    progress_callback=None,
    cancellation_check=None,
    process_callback=None,
    quality_settings=None,
) -> None:
    """Stack two videos vertically with audio sync offset.

    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        offset: Audio sync offset in seconds
        output_path: Path for output video
        progress_callback: Optional callback function(dict) for progress updates
        cancellation_check: Optional callback function() -> bool to check if cancelled
        process_callback: Optional callback function(pid: int) to receive FFmpeg PID
        quality_settings: Optional dict with quality settings
    """
    offset_str = f"{offset:.2f}"
    encoder = _detect_gpu_encoder()

    # Get encoding parameters from quality settings (simplified - no presets/CRF/lanczos)
    bitrate, speed_preset, resolution_preset = _get_encoding_params(quality_settings)

    # Debug logging
    logger.info(f"Quality settings received: {quality_settings}")
    logger.info(f"Parsed encoding params - bitrate: {bitrate}, speed: {speed_preset}, resolution: {resolution_preset}")

    # Check if GPU decoding should be used (default: True)
    use_gpu_decode = True
    if quality_settings and "use_gpu_decode" in quality_settings:
        use_gpu_decode = quality_settings.get("use_gpu_decode", True)

    # Map resolution preset to dimensions for each individual video (before stacking)
    # These are the dimensions each video will be scaled to before vertical stacking
    single_video_resolution_map = {
        "720p": "1280:720",  # Each video scaled to 720p
        "1080p": "1920:1080",  # Each video scaled to 1080p
        "1440p": "2560:1440",  # Each video scaled to 1440p
        "4k": "3840:2160",  # Each video scaled to 4K
    }
    single_resolution = single_video_resolution_map.get(resolution_preset, "1920:1080")

    # If both inputs are already the same resolution, skip scaling entirely (faster path).
    # Otherwise, scale each input individually to the requested preset resolution.
    v1_w, v1_h = _get_video_resolution(video1_path)
    v2_w, v2_h = _get_video_resolution(video2_path)
    inputs_same_resolution = v1_w > 0 and v1_h > 0 and v1_w == v2_w and v1_h == v2_h

    # IMPORTANT: vstack defaults to extending to the longest input.
    # Using shortest=1 ensures the output ends at the overlap (intersection).
    if inputs_same_resolution:
        logger.info(
            "Using fast stack path (no scaling): inputs=%sx%s, preset_target=%s",
            v1_w,
            v1_h,
            single_resolution,
        )
        filter_complex = f"[0:v][1:v]vstack=inputs=2:shortest=1[vout]"
    else:
        logger.info(
            "Using scale+stack path: input1=%sx%s input2=%sx%s target=%s",
            v1_w,
            v1_h,
            v2_w,
            v2_h,
            single_resolution,
        )
        filter_complex = (
            f"[0:v]scale={single_resolution}[v0];"
            f"[1:v]scale={single_resolution}[v1];"
            f"[v0][v1]vstack=inputs=2:shortest=1[vout]"
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Always use H.264 with detected encoder (NVENC/QSV/AMF or software fallback)
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

        # Determine hardware acceleration for decoding
        hwaccel_args = []
        if use_gpu_decode:
            if enc.endswith("_nvenc"):
                hwaccel_args = ["-hwaccel", "cuda"]
            elif enc.endswith("_qsv"):
                hwaccel_args = ["-hwaccel", "qsv"]
            elif enc.endswith("_amf"):
                hwaccel_args = ["-hwaccel", "auto"]

        # Map preset to encoder-specific preset
        mapped_preset = _map_preset_for_encoder(speed_preset, enc)

        # Calculate trim points to only encode when both videos are playing
        # When offset > 0: video2 starts later, so trim the beginning of video1
        # When offset < 0: video1 starts later, so trim the beginning of video2
        video1_trim = max(0, offset)  # Trim from start of video1 if offset positive
        video2_trim = max(0, -offset)  # Trim from start of video2 if offset negative

        # Build input arguments with trimming
        input_args = []

        # Video 1 input with optional trim
        if video1_trim > 0:
            input_args.extend(["-ss", f"{video1_trim:.3f}"])
        input_args.extend(["-i", video1_path])

        # Video 2 input with optional trim (no itsoffset needed since we trimmed)
        if video2_trim > 0:
            input_args.extend(["-ss", f"{video2_trim:.3f}"])
        input_args.extend(["-i", video2_path])

        cmd = (
            [
                "ffmpeg",
                "-y",
            ]
            + hwaccel_args
            + [
                "-progress",
                "pipe:1",
                "-loglevel",
                "error",
            ]
            + input_args
            + [
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "0:a?",  # Map audio from first input if available
                "-c:v",
                enc,
                "-preset",
                mapped_preset,
                "-c:a",
                "aac",  # Encode audio as AAC
                "-b:a",
                "192k",  # 192 kbps audio bitrate
            ]
        )

        # Add bitrate control (simplified - always bitrate mode)
        cmd.extend(["-b:v", bitrate])

        cmd.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-shortest",
                output_path,
            ]
        )

        # Log the FFmpeg command
        logger.info(f"FFmpeg command: {' '.join(cmd)}")

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
                bufsize=0,  # Unbuffered for immediate output
                universal_newlines=True,
            )
            # Store process PID for direct termination if needed
            if process_callback:
                process_callback(process.pid)
                print(f"[TRANSCODE] FFmpeg process started with PID: {process.pid}")
                logger.info(f"FFmpeg process started with PID: {process.pid}")
            # Read stderr in a separate thread to prevent blocking
            import threading
            import select
            import sys

            stderr_lines = []

            def read_stderr():
                if process.stderr:
                    for line in process.stderr:
                        stderr_lines.append(line)

            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            # Parse progress output in real-time from stdout with cancellation checks
            progress_data = {}
            progress_count = 0
            last_frame_count = 0

            # Use select for non-blocking reads with timeout (Unix only)
            # On Windows, fall back to blocking reads
            use_select = sys.platform != 'win32'

            while process.poll() is None:
                # Check for cancellation even if no data available
                if cancellation_check and cancellation_check():
                    logger.info("Transcoding cancelled, terminating FFmpeg process")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning("FFmpeg did not terminate gracefully, killing process")
                        process.kill()
                    raise RuntimeError("Transcoding cancelled by user")

                # Read available data with timeout
                if use_select:
                    ready, _, _ = select.select([process.stdout], [], [], 0.5)
                    if not ready:
                        continue  # Timeout - loop back to check cancellation

                line = process.stdout.readline()
                if not line:
                    continue

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
                                # Cap at 95% to leave room for calibration step
                                if total_frames > 0 and current_frame > 0:
                                    progress_percent = min(95, (current_frame / total_frames) * 95)
                                    current_time = current_frame / frame_rate if frame_rate > 0 else 0
                                else:
                                    # Fallback to time-based if we have it
                                    out_time_ms = progress_data.get('out_time_ms', 'N/A')
                                    if out_time_ms != 'N/A' and out_time_ms.replace('.', '').isdigit():
                                        current_time = float(out_time_ms) / 1000.0
                                        progress_percent = (
                                            min(95, (current_time / total_duration) * 95) if total_duration > 0 else 0
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

                                current_bitrate = progress_data.get('bitrate', '0')
                                if current_bitrate == 'N/A':
                                    current_bitrate = '0kbits/s'

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
                                            'bitrate': current_bitrate,
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
            stderr = e.stderr if e.stderr else "No error output captured"
            print(f"Encoding with {enc} failed: {stderr}")
            logger.error(f"FFmpeg encoding failed with {enc}: {stderr}")

            # If this was a hardware encoder and it failed, try software
            if enc != "libx264":
                print(f"Falling back to software encoding (libx264)...")
                continue
            else:
                # Software encoding also failed, raise the error with stderr
                raise RuntimeError(f"Video stacking failed: {e}\nFFmpeg stderr: {stderr}") from e
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
