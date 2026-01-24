"""
Helper utilities for working with match dict structures.

These are thin wrappers for when working with raw dicts from storage,
before they're validated into MatchModel instances.
"""

from typing import Dict, Any, Optional


def update_processing(match_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Update processing fields in match dict."""
    if "processing" not in match_data:
        match_data["processing"] = {}

    for key, value in kwargs.items():
        if value is not None:
            match_data["processing"][key] = value

    return match_data


def update_transcode(match_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Update transcode fields in match dict."""
    if "transcode" not in match_data or match_data["transcode"] is None:
        match_data["transcode"] = {}

    for key, value in kwargs.items():
        if value is not None:
            match_data["transcode"][key] = value

    return match_data


def get_processing_status(match_data: Dict[str, Any]) -> str:
    """Get processing status from match dict."""
    if "processing" in match_data and isinstance(match_data["processing"], dict):
        return match_data["processing"].get("status", "pending")
    return "pending"


def get_transcode_fps(match_data: Dict[str, Any]) -> Optional[float]:
    """Get transcode FPS from match dict."""
    if "transcode" in match_data and isinstance(match_data["transcode"], dict):
        return match_data["transcode"].get("fps")
    return None
