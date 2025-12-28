"""
Processing router for match video processing.

Handles the /matches/{id}/process endpoint that orchestrates
transcoding, calibration, and position optimization.
"""

import logging
import threading
import base64
import io
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
import cv2
import numpy as np

from app.repositories.match_store import MatchStore
from app.repositories.file_match_store import FileMatchStore


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/matches", tags=["processing"])

# Match store path
MATCHES_DIR = Path(__file__).parent.parent.parent / "data" / "matches"


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

    return {
        "match_id": match_id,
        "status": match_data.get("status", "pending"),
        "processing_step": match_data.get("processing_step"),
        "error_code": match_data.get("error_code"),
        "error_message": match_data.get("error_message"),
        "processing_started_at": match_data.get("processing_started_at"),
        "processing_completed_at": match_data.get("processing_completed_at"),
    }


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
    left_videos = match_data.get("left_videos", [])
    right_videos = match_data.get("right_videos", [])

    if not left_videos or not right_videos:
        raise HTTPException(status_code=400, detail="Match must have at least one left and one right video")

    # Get first video from each side
    left_video_path = left_videos[0]["path"]
    right_video_path = right_videos[0]["path"]

    # Check FFmpeg
    try:
        check_ffmpeg()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg not available: {e}")

    # Update match status
    match_data["status"] = "transcoding"
    match_data["processing_step"] = "transcoding"
    match_data["processing_message"] = "Syncing and stacking videos..."
    match_store.update(match_id, match_data)

    try:
        # Transcode in background thread
        def _transcode_background():
            import os
            import tempfile

            temp_dir = os.path.join("temp", match_id)
            os.makedirs(temp_dir, exist_ok=True)

            try:
                output_video_path = os.path.join("data/videos", f"{match_id}.mp4")
                stacked_path, offset = transcode_and_stack(
                    left_video_path, right_video_path, output_video_path, temp_dir
                )

                # Extract preview frame from the stacked video
                try:
                    from app.services.transcoding import extract_preview_frame
                    preview_path = os.path.join("data/videos", f"{match_id}_preview.jpg")
                    extract_preview_frame(stacked_path, preview_path, timestamp=1.0)
                    logger.info(f"Preview generated for {match_id} at {preview_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate preview for {match_id}: {e}")
                    # Continue even if preview generation fails - not critical

                # Update match with video path
                match_data["src"] = f"videos/{match_id}.mp4"
                match_data["offset_seconds"] = round(offset, 2)
                match_data["status"] = "pending"  # Ready for frontend processing
                match_data["processing_step"] = "awaiting_frames"
                match_data["processing_message"] = "Video ready, awaiting frame extraction"
                match_store.update(match_id, match_data)

            except Exception as e:
                logger.error(f"Transcoding failed for {match_id}: {e}", exc_info=True)
                match_data["status"] = "error"
                match_data["error_message"] = str(e)
                match_data["processing_step"] = None
                match_store.update(match_id, match_data)

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
        match_data["status"] = "calibrating"
        match_data["processing_step"] = "feature_matching"
        match_data["processing_message"] = "Matching features in warped frames..."
        match_store.update(match_id, match_data)

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
        match_data["processing_step"] = "optimizing"
        match_data["processing_message"] = f"Found {match_result['num_matches']} features, optimizing..."
        match_store.update(match_id, match_data)

        # Step 2: Optimize camera positions
        # Note: Swap left/right to match viewer's coordinate system
        params = optimize_position(match_result["right_points"], match_result["left_points"])

        # Update match with results
        match_data["status"] = "ready"
        match_data["params"] = params
        match_data["num_matches"] = match_result["num_matches"]
        match_data["confidence"] = match_result["confidence"]
        match_data["processing_step"] = "complete"
        match_data["processing_message"] = "Processing complete"
        match_data["processing_completed_at"] = datetime.now(timezone.utc).isoformat()
        match_data["error_code"] = None
        match_data["error_message"] = None
        match_store.update(match_id, match_data)

        return {
            "success": True,
            "params": params,
            "num_matches": match_result["num_matches"],
            "confidence": match_result["confidence"],
        }

    except ValueError as e:
        # Feature matching or optimization error
        match_data["status"] = "error"
        match_data["error_message"] = str(e)
        match_data["processing_step"] = None
        match_store.update(match_id, match_data)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected error
        logger.error(f"Error processing frames for match {match_id}: {e}", exc_info=True)
        match_data["status"] = "error"
        match_data["error_message"] = "Failed to process frames"
        match_data["processing_step"] = None
        match_store.update(match_id, match_data)
        raise HTTPException(status_code=500, detail="Internal processing error")
