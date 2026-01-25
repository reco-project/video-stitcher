"""
Processing router for match video processing.

Handles the /matches/{id}/process endpoint that orchestrates
transcoding, calibration, and position optimization.
"""

import threading
import base64
import io
import os
import signal
import time
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
import cv2
import numpy as np

from app.utils.logger import get_logger
from app.repositories.match_store import MatchStore
from app.repositories.file_match_store import FileMatchStore
from app.data_paths import MATCHES_DIR, VIDEOS_DIR, TEMP_DIR


logger = get_logger(__name__)
router = APIRouter(prefix="/matches", tags=["processing"])

# In-memory cache for live progress updates (avoids excessive disk I/O)
# Structure: {match_id: {transcode_fps, transcode_speed, transcode_progress, ...}}
_progress_cache = {}

# Track cancellation requests
_cancellation_flags = {}

# Track FFmpeg process PIDs for direct termination
_ffmpeg_processes = {}


def get_store() -> MatchStore:
    """Dependency injection for match store."""
    return FileMatchStore(str(MATCHES_DIR))


@router.get("/{match_id}/status")
async def get_processing_status(match_id: str, match_store: MatchStore = Depends(get_store)):
    """
    Get the current processing status of a match.

    Args:
        match_id: Match identifier

    Returns:
        Dictionary with status, step, and error info if applicable

    Raises:
        404: Match not found
    """
    match_data = match_store.get_by_id(match_id)
    if not match_data:
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' not found")

    # Detect stale processing state (app was quit/crashed during processing)
    # If status is active but no cache entry exists, the process was interrupted
    active_statuses = ["transcoding", "calibrating"]
    if match_data.processing and match_data.processing.status in active_statuses:
        if match_id not in _progress_cache and match_id not in _ffmpeg_processes:
            # Processing was interrupted - reset to error state
            logger.warning(f"Detected stale processing state for {match_id}, status was {match_data.processing.status}")
            match_data.update_processing(
                status="error",
                step=None,
                message="Processing interrupted (app was closed)",
                error_code="INTERRUPTED",
                error_message="Processing was interrupted. Please retry.",
            )
            match_store.update(match_id, match_data.model_dump(exclude_none=False))

    # Return nested structure directly
    response = {
        "match_id": match_id,
        "processing": match_data.processing.model_dump(exclude_none=False) if match_data.processing else {},
        "transcode": match_data.transcode.model_dump(exclude_none=False) if match_data.transcode else None,
    }

    # Merge live progress from cache if available
    if match_id in _progress_cache:
        cache = _progress_cache[match_id]

        # Update processing.message from cache
        if "processing_message" in cache:
            response["processing"]["message"] = cache["processing_message"]

        # Update transcode fields from cache
        cache_to_response = {
            "transcode_progress": "progress",
            "transcode_fps": "fps",
            "transcode_speed": "speed",
            "transcode_current_time": "current_time",
            "transcode_total_duration": "total_duration",
        }

        for cache_key, response_key in cache_to_response.items():
            if cache_key in cache:
                if response["transcode"] is None:
                    response["transcode"] = {}
                response["transcode"][response_key] = cache[cache_key]

    return JSONResponse(content=response)


@router.post("/{match_id}/cancel")
async def cancel_processing(match_id: str, match_store: MatchStore = Depends(get_store)):
    """
    Cancel ongoing processing for a match.

    Args:
        match_id: Match identifier
        match_store: Match store dependency

    Returns:
        Success message

    Raises:
        404: Match not found
    """
    match_data = match_store.get_by_id(match_id)
    if not match_data:
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' not found")

    # Set cancellation flag
    _cancellation_flags[match_id] = True
    logger.info(f"Cancellation requested for match {match_id}")

    # Kill FFmpeg process directly if running
    if match_id in _ffmpeg_processes:
        pid = _ffmpeg_processes[match_id]
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to FFmpeg process {pid}")
            time.sleep(0.5)

            # Check if still alive, only then send SIGKILL
            try:
                os.kill(pid, 0)  # Check if process exists
                os.kill(pid, signal.SIGKILL)
                logger.info(f"Process still alive, sent SIGKILL to {pid}")
            except ProcessLookupError:
                logger.info(f"FFmpeg process {pid} terminated cleanly")
        except ProcessLookupError:
            logger.info(f"FFmpeg process {pid} not found")
        except Exception as e:
            logger.warning(f"Failed to kill FFmpeg process {pid}: {e}")
        finally:
            _ffmpeg_processes.pop(match_id, None)

    # Update match status to reflect cancellation
    current_status = match_data.get_status()
    if current_status in ["transcoding", "calibrating"]:
        match_data.update_processing(status="pending", step=None, message="Processing cancelled by user")
        match_store.update(match_id, match_data.model_dump(exclude_none=False))

    # Clear progress cache
    if match_id in _progress_cache:
        del _progress_cache[match_id]

    return {"message": "Processing cancelled", "match_id": match_id}


@router.post("/{match_id}/transcode")
async def transcode_match_endpoint(match_id: str, match_store: MatchStore = Depends(get_store)):
    """
    Transcode and stack videos only (no calibration).

    Used when frontend will handle frame extraction and warping.

    Args:
        match_id: Match identifier
        match_store: Match store dependency

    Returns:
        Success response with video path

    Raises:
        404: Match not found
        400: Match validation failed
    """
    from app.services.transcoding import transcode_and_stack, check_ffmpeg

    # Get match
    match_data = match_store.get_by_id(match_id)
    if not match_data:
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' not found")

    # Validate match has videos
    left_videos = match_data.left_videos
    right_videos = match_data.right_videos

    if not left_videos or not right_videos:
        raise HTTPException(status_code=400, detail="Match must have at least one left and one right video")

    # Get first video from each side
    left_video_path = left_videos[0].path
    right_video_path = right_videos[0].path

    # Check FFmpeg
    try:
        check_ffmpeg()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg not available: {e}")

    # Update match status
    from datetime import datetime, timezone

    try:
        match_data.update_processing(
            status="transcoding",
            step="transcoding",
            message="Preparing to sync and stack videos...",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        match_store.update(match_id, match_data.model_dump(exclude_none=False))
    except Exception as e:
        logger.error(f"Failed to update match status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize transcoding: {str(e)}")

    # Initialize progress cache with starting message
    _progress_cache[match_id] = {'processing_message': 'Starting transcoding process...'}

    try:
        # Transcode in background thread
        def _transcode_background():
            import os
            import tempfile
            import time

            # Create a fresh store instance for the background thread
            bg_store = get_store()

            # Re-fetch match data to ensure we have the latest version
            bg_match_data = bg_store.get_by_id(match_id)
            if not bg_match_data:
                logger.error(f"Match {match_id} not found in background thread")
                return

            temp_dir = os.path.join("temp", match_id)
            os.makedirs(temp_dir, exist_ok=True)

            def update_progress(progress_info):
                """Update in-memory progress cache (no disk I/O during encoding)."""
                try:
                    stage = progress_info.get('stage', 'transcoding')

                    # Initialize cache entry if needed
                    if match_id not in _progress_cache:
                        _progress_cache[match_id] = {}

                    # Update based on stage
                    if stage == 'audio_extraction':
                        _progress_cache[match_id]['processing_message'] = progress_info.get(
                            'message', 'Extracting audio...'
                        )
                    elif stage == 'audio_sync':
                        _progress_cache[match_id]['processing_message'] = progress_info.get(
                            'message', 'Syncing audio...'
                        )
                    elif stage == 'encoding':
                        # Detailed encoding progress - stored in memory only
                        fps = progress_info.get('fps', 0)
                        speed = progress_info.get('speed', '0')
                        progress_percent = progress_info.get('progress_percent', 0)
                        current_time = progress_info.get('current_time', 0)
                        total_duration = progress_info.get('total_duration', 0)
                        encoder = progress_info.get('encoder', 'h264')

                        _progress_cache[match_id].update(
                            {
                                'processing_message': f"Encoding video ({encoder})...",
                                'transcode_progress': float(round(progress_percent, 1)),
                                'transcode_fps': float(round(fps, 1)),
                                'transcode_speed': str(speed),
                                'transcode_current_time': float(round(current_time, 1)),
                                'transcode_total_duration': float(round(total_duration, 1)),
                                'transcode_encoder': encoder,
                            }
                        )

                        # Store offset if provided (convert numpy types to Python native)
                        if 'offset_seconds' in progress_info:
                            _progress_cache[match_id]['offset_seconds'] = float(progress_info['offset_seconds'])
                except Exception as e:
                    logger.warning(f"Failed to update progress cache for {match_id}: {e}")

            try:
                output_video_path = str(VIDEOS_DIR / f"{match_id}.mp4")

                # Check for cancellation before starting
                if _cancellation_flags.get(match_id, False):
                    logger.info(f"Transcoding cancelled before start for {match_id}")
                    bg_match_data.update_processing(status="pending", step=None, message="Processing cancelled")
                    bg_store.update(match_id, bg_match_data.model_dump(exclude_none=False))
                    if match_id in _progress_cache:
                        del _progress_cache[match_id]
                    if match_id in _cancellation_flags:
                        del _cancellation_flags[match_id]
                    return

                # Cancellation checker function
                def is_cancelled():
                    cancelled = _cancellation_flags.get(match_id, False)
                    if cancelled:
                        logger.info(f"Cancellation flag detected for {match_id}")
                    return cancelled

                # Store process info callback
                def store_process_pid(pid):
                    _ffmpeg_processes[match_id] = pid
                    logger.info(f"Stored FFmpeg PID {pid} for match {match_id}")

                # Get quality settings from match (convert to dict if needed)
                quality_settings = None
                if bg_match_data.quality_settings:
                    quality_settings = bg_match_data.quality_settings.model_dump(exclude_none=False)
                    logger.info(f"Using quality settings from match data: {quality_settings}")
                else:
                    logger.warning(f"No quality_settings found in match data for {match_id}")

                stacked_path, offset = transcode_and_stack(
                    left_video_path,
                    right_video_path,
                    output_video_path,
                    temp_dir,
                    progress_callback=update_progress,
                    cancellation_check=is_cancelled,
                    process_callback=store_process_pid,
                    quality_settings=quality_settings,
                )

                # Check for cancellation after transcoding
                if _cancellation_flags.get(match_id, False):
                    logger.info(f"Transcoding cancelled after completion for {match_id}")
                    bg_match_data.update_processing(status="pending", step=None, message="Processing cancelled")
                    bg_store.update(match_id, bg_match_data.model_dump(exclude_none=False))
                    if match_id in _progress_cache:
                        del _progress_cache[match_id]
                    if match_id in _cancellation_flags:
                        del _cancellation_flags[match_id]
                    if match_id in _ffmpeg_processes:
                        del _ffmpeg_processes[match_id]
                    return

                # Extract preview frame from the stacked video
                try:
                    from app.services.transcoding import extract_preview_frame

                    preview_path = str(VIDEOS_DIR / f"{match_id}_preview.jpg")
                    extract_preview_frame(stacked_path, preview_path, timestamp=1.0)
                    logger.info(f"Preview generated for {match_id} at {preview_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate preview for {match_id}: {e}")
                    # Continue even if preview generation fails - not critical

                # Get final progress values from cache before clearing
                final_progress = _progress_cache.get(match_id, {})
                logger.info(f"Final progress cache for {match_id}: {final_progress}")

                # Update match with video path and final progress
                from datetime import datetime, timezone

                bg_match_data.src = f"videos/{match_id}.mp4"

                # Update processing status
                bg_match_data.update_processing(
                    status="pending",
                    step="awaiting_frames",
                    message="Video ready, awaiting frame extraction",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )

                # Persist final transcode metrics using nested structure
                bg_match_data.update_transcode(
                    progress=final_progress.get('transcode_progress'),
                    fps=final_progress.get('transcode_fps'),
                    speed=final_progress.get('transcode_speed'),
                    current_time=final_progress.get('transcode_current_time'),
                    total_duration=final_progress.get('transcode_total_duration'),
                    offset_seconds=round(offset, 2),
                )
                logger.info(
                    f"Saved transcode metrics - fps: {bg_match_data.get_transcode_fps()}, progress: {bg_match_data.transcode.progress if bg_match_data.transcode else None}"
                )

                # Persist quality settings used for this transcode
                if quality_settings:
                    from app.models.match import QualitySettings

                    bg_match_data.quality_settings = QualitySettings(**quality_settings)
                    logger.info(f"Saved quality settings: {quality_settings}")
                else:
                    logger.warning(f"No quality settings to save for {match_id}")

                bg_store.update(match_id, bg_match_data.model_dump(exclude_none=False))

                # Clear progress cache, cancellation flag, and PID tracking
                if match_id in _progress_cache:
                    del _progress_cache[match_id]
                if match_id in _cancellation_flags:
                    del _cancellation_flags[match_id]
                if match_id in _ffmpeg_processes:
                    del _ffmpeg_processes[match_id]

            except RuntimeError as e:
                # Check if this is a cancellation error
                if "cancelled" in str(e).lower():
                    logger.info(f"Transcoding cancelled for {match_id}: {e}")
                    bg_match_data.update_processing(status="pending", step=None, message="Processing cancelled")
                    bg_store.update(match_id, bg_match_data.model_dump(exclude_none=False))
                else:
                    # Other runtime errors
                    logger.error(f"Transcoding failed for {match_id}: {e}", exc_info=True)
                    bg_match_data.update_processing(status="error", step=None, error_message=str(e))
                    bg_store.update(match_id, bg_match_data.model_dump(exclude_none=False))

                # Clear progress cache and cancellation flag
                if match_id in _progress_cache:
                    del _progress_cache[match_id]
                if match_id in _cancellation_flags:
                    del _cancellation_flags[match_id]
                if match_id in _ffmpeg_processes:
                    del _ffmpeg_processes[match_id]

            except Exception as e:
                logger.error(f"Transcoding failed for {match_id}: {e}", exc_info=True)
                bg_match_data.update_processing(status="error", step=None, error_message=str(e))
                bg_store.update(match_id, bg_match_data.model_dump(exclude_none=False))

                # Clear progress cache and cancellation flag
                if match_id in _progress_cache:
                    del _progress_cache[match_id]
                if match_id in _cancellation_flags:
                    del _cancellation_flags[match_id]
                if match_id in _ffmpeg_processes:
                    del _ffmpeg_processes[match_id]

        thread = threading.Thread(target=_transcode_background, daemon=True)
        thread.start()

        return {"message": "Transcoding started", "match_id": match_id}

    except Exception as e:
        logger.error(f"Failed to start transcoding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to start transcoding")


@router.post("/{match_id}/process-with-frames")
async def process_match_with_frames(
    match_id: str,
    left_frame: UploadFile = File(...),
    right_frame: UploadFile = File(...),
    match_store: MatchStore = Depends(get_store),
):
    """
    Process a match using pre-warped frames from the frontend.

    The frontend extracts frames from the video, applies fisheye correction,
    and sends the corrected frames here for feature matching and optimization.

    Args:
        match_id: Match identifier
        left_frame: Warped left camera frame (PNG/JPEG)
        right_frame: Warped right camera frame (PNG/JPEG)
        match_store: Match store dependency

    Returns:
        Processing results with calibration parameters

    Raises:
        404: Match not found
        400: Invalid frame data
    """
    from app.services.feature_matching import match_features
    from app.services.position_optimization import optimize_position

    # Get match
    match_data = match_store.get_by_id(match_id)
    if not match_data:
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' not found")

    try:
        # Read uploaded images
        left_bytes = await left_frame.read()
        right_bytes = await right_frame.read()

        logger.info(f"Received frames: left={len(left_bytes)} bytes, right={len(right_bytes)} bytes")

        # Save frames for debugging
        import os

        debug_dir = os.path.join("temp", match_id, "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)

        with open(os.path.join(debug_dir, "left_received.png"), "wb") as f:
            f.write(left_bytes)
        with open(os.path.join(debug_dir, "right_received.png"), "wb") as f:
            f.write(right_bytes)

        logger.info(f"Debug frames saved to: {debug_dir}")

        # Convert to numpy arrays
        left_np = np.frombuffer(left_bytes, dtype=np.uint8)
        right_np = np.frombuffer(right_bytes, dtype=np.uint8)

        # Decode images
        img_left = cv2.imdecode(left_np, cv2.IMREAD_COLOR)
        img_right = cv2.imdecode(right_np, cv2.IMREAD_COLOR)

        if img_left is None or img_right is None:
            raise HTTPException(status_code=400, detail="Failed to decode image data")

        logger.info(f"Decoded images: left={img_left.shape}, right={img_right.shape}")

        # Update match status
        match_data.update_processing(
            status="calibrating", step="feature_matching", message="Matching features in warped frames..."
        )
        match_store.update(match_id, match_data.model_dump(exclude_none=False))

        # Step 1: Match features
        match_result = match_features(img_left, img_right)

        # Debug: Draw feature matches visualization
        try:
            # Resize images for visualization
            h, w = img_left.shape[:2]
            target_w = 1920
            scale = target_w / w

            img_left_vis = cv2.resize(img_left, (int(w * scale), int(h * scale)))
            img_right_vis = cv2.resize(img_right, (int(w * scale), int(h * scale)))

            # Get points from match result (they're in normalized plane coords)
            left_points = np.array(match_result["left_points"])
            right_points = np.array(match_result["right_points"])

            # Convert back to image coordinates for visualization
            img_h, img_w = img_left_vis.shape[:2]
            plane_w = 1.0
            plane_h = plane_w * (img_h / img_w)

            # Reverse the normalization
            left_pts_img = np.zeros_like(left_points)
            left_pts_img[:, 0] = (left_points[:, 0] / plane_w + 0.5) * img_w
            left_pts_img[:, 1] = (left_points[:, 1] / plane_h + 0.5) * img_h

            right_pts_img = np.zeros_like(right_points)
            right_pts_img[:, 0] = (right_points[:, 0] / plane_w + 0.5) * img_w
            right_pts_img[:, 1] = (right_points[:, 1] / plane_h + 0.5) * img_h

            # Draw matches on concatenated image
            vis_height = max(img_left_vis.shape[0], img_right_vis.shape[0])
            vis_img = np.zeros((vis_height, img_left_vis.shape[1] + img_right_vis.shape[1], 3), dtype=np.uint8)
            vis_img[: img_left_vis.shape[0], : img_left_vis.shape[1]] = img_left_vis
            vis_img[: img_right_vis.shape[0], img_left_vis.shape[1] :] = img_right_vis

            # Draw lines between matched points
            offset = img_left_vis.shape[1]
            for i in range(len(left_pts_img)):
                pt1 = tuple(left_pts_img[i].astype(int))
                pt2 = tuple((right_pts_img[i] + [offset, 0]).astype(int))

                # Draw circles at keypoints
                cv2.circle(vis_img, pt1, 5, (0, 255, 0), 2)
                cv2.circle(vis_img, pt2, 5, (0, 255, 0), 2)

                # Draw line connecting them
                cv2.line(vis_img, pt1, pt2, (255, 0, 255), 1)

            # Add text overlay
            text = f"Matches: {match_result['num_matches']} | Confidence: {match_result['confidence']:.2%}"
            cv2.putText(vis_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # Save visualization
            vis_path = os.path.join(debug_dir, "feature_matches.png")
            cv2.imwrite(vis_path, vis_img)
            logger.info(f"Feature matching visualization saved to: {vis_path}")

        except Exception as viz_error:
            logger.warning(f"Failed to create feature matching visualization: {viz_error}")

        # Update status
        match_data.update_processing(
            step="optimizing", message=f"Found {match_result['num_matches']} features, optimizing..."
        )
        match_store.update(match_id, match_data.model_dump(exclude_none=False))

        # Step 2: Optimize camera positions
        # Note: Swap left/right to match viewer's coordinate system
        params = optimize_position(match_result["right_points"], match_result["left_points"])

        # Update match with results
        from datetime import datetime, timezone

        match_data.params = params
        match_data.num_matches = match_result["num_matches"]
        match_data.confidence = match_result["confidence"]

        # Update processing info
        match_data.update_processing(
            status="ready", step="complete", message="Processing complete", error_code=None, error_message=None
        )

        # Don't overwrite processing_completed_at if it was already set during transcoding
        if match_data.processing and not match_data.processing.completed_at:
            match_data.update_processing(completed_at=datetime.now(timezone.utc).isoformat())

        match_store.update(match_id, match_data.model_dump(exclude_none=False))

        return {
            "success": True,
            "params": params,
            "num_matches": match_result["num_matches"],
            "confidence": match_result["confidence"],
        }

    except ValueError as e:
        # Feature matching or optimization error
        match_data.update_processing(status="error", step=None, error_message=str(e))
        match_store.update(match_id, match_data.model_dump(exclude_none=False))
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected error
        logger.error(f"Error processing frames for match {match_id}: {e}", exc_info=True)
        match_data.update_processing(status="error", step=None, error_message="Failed to process frames")
        match_store.update(match_id, match_data.model_dump(exclude_none=False))
        raise HTTPException(status_code=500, detail="Internal processing error")
