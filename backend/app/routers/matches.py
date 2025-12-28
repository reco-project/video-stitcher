"""FastAPI router for match API endpoints."""

from typing import List, Dict
from fastapi import APIRouter, HTTPException, Depends, status

from app.repositories.match_store import MatchStore
from app.models.match import MatchModel


router = APIRouter(prefix="/matches")


# Dependency injection - will be configured in main.py
def get_store() -> MatchStore:
    """
    Dependency to get the match store instance.
    This will be overridden in main.py with the actual store.
    """
    raise NotImplementedError("Store dependency not configured")


@router.get("", response_model=List[Dict])
def list_all_matches(store: MatchStore = Depends(get_store)):
    """
    List all matches.

    Returns:
        List of all match dictionaries
    """
    return store.list_all()


@router.get("/{match_id}", response_model=Dict)
def get_match(match_id: str, store: MatchStore = Depends(get_store)):
    """
    Get a specific match by ID.

    Args:
        match_id: Unique match identifier

    Returns:
        Match dictionary

    Raises:
        404: Match not found
    """
    match = store.get_by_id(match_id)
    if match is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Match with ID '{match_id}' not found")
    return match


@router.post("", response_model=Dict, status_code=status.HTTP_201_CREATED)
def create_match(match: MatchModel, store: MatchStore = Depends(get_store)):
    """
    Create a new match.

    Args:
        match: Match data to create

    Returns:
        Created match dictionary

    Raises:
        400: Validation error
        409: Match with same ID already exists
    """
    try:
        match_dict = match.model_dump()
        created = store.create(match_dict)
        return created
    except ValueError as e:
        error_msg = str(e)
        # Check if it's an ID conflict
        if "already exists" in error_msg:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error_msg)
        # Otherwise it's a validation error
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)


@router.put("/{match_id}", response_model=Dict)
def update_match(match_id: str, match: MatchModel, store: MatchStore = Depends(get_store)):
    """
    Update an existing match.

    Args:
        match_id: ID of match to update
        match: Updated match data

    Returns:
        Updated match dictionary

    Raises:
        400: Validation error or ID mismatch
        404: Match not found
    """
    try:
        match_dict = match.model_dump()
        updated = store.update(match_id, match_dict)
        return updated
    except ValueError as e:
        error_msg = str(e)
        # Check if it's a not found error
        if "not found" in error_msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)
        # Otherwise it's a validation or mismatch error
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)


@router.delete("/{match_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_match(match_id: str, store: MatchStore = Depends(get_store)):
    """
    Delete a match and its associated video files.

    Args:
        match_id: ID of match to delete

    Raises:
        404: Match not found
    """
    import os
    import logging

    logger = logging.getLogger(__name__)

    # Get match data first to find video files
    match = store.get_by_id(match_id)
    if match is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Match with ID '{match_id}' not found")

    # Delete from store
    deleted = store.delete(match_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Match with ID '{match_id}' not found")

    # Delete associated video files
    if match.get("src"):
        video_path = match["src"]
        # Remove 'videos/' prefix if present to get relative path from data directory
        if video_path.startswith("videos/"):
            video_path = video_path[7:]  # Remove 'videos/' prefix

        full_video_path = os.path.join("data", "videos", video_path)

        # Delete main video file
        if os.path.exists(full_video_path):
            try:
                os.remove(full_video_path)
                logger.info(f"Deleted video file: {full_video_path}")
            except Exception as e:
                logger.warning(f"Failed to delete video file {full_video_path}: {e}")

        # Delete preview image if it exists
        preview_path = full_video_path.rsplit(".", 1)[0] + "_preview.jpg"
        if os.path.exists(preview_path):
            try:
                os.remove(preview_path)
                logger.info(f"Deleted preview file: {preview_path}")
            except Exception as e:
                logger.warning(f"Failed to delete preview file {preview_path}: {e}")

    # Delete temp directory if it exists
    temp_dir = os.path.join("temp", match_id)
    if os.path.exists(temp_dir):
        try:
            import shutil

            shutil.rmtree(temp_dir)
            logger.info(f"Deleted temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to delete temp directory {temp_dir}: {e}")

    return None
