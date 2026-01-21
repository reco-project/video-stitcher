"""
Settings API router.

Handles application settings endpoints including GPU encoder preferences.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
from app.config import get_settings, EncoderType
from app.services.transcoding import _get_available_encoders

router = APIRouter(prefix="/settings", tags=["settings"])


class EncoderSettingsUpdate(BaseModel):
    """Request model for updating encoder settings."""

    encoder: EncoderType


class EncoderInfo(BaseModel):
    """Response model for encoder information."""

    available_encoders: list[str]
    current_encoder: EncoderType
    encoder_descriptions: dict[str, str]


@router.get("/encoders", response_model=EncoderInfo)
async def get_encoder_settings():
    """
    Get current encoder settings and available encoders.
    Validates current encoder and resets to auto if unavailable.

    Returns:
        EncoderInfo: Current encoder preference and available encoders
    """
    settings = get_settings()
    available = _get_available_encoders()

    # Validate current encoder - reset to auto if it's a hardware encoder that's not available
    current = settings.encoder
    if current not in ["auto", "libx264"] and current not in available:
        settings.update_encoder("auto")
        current = "auto"

    descriptions = {
        "auto": "Auto",
        "h264_nvenc": "NVIDIA GPU (h264_nvenc)",
        "h264_qsv": "Intel GPU (Quick Sync Video)",
        "h264_amf": "AMD GPU (Advanced Media Framework)",
        "libx264": "CPU (libx264 - software encoding)",
    }

    return EncoderInfo(
        available_encoders=["auto", "libx264"] + available,
        current_encoder=current,
        encoder_descriptions=descriptions,
    )


@router.put("/encoders")
async def update_encoder_settings(update: EncoderSettingsUpdate):
    """
    Update encoder preference.

    Args:
        update: New encoder preference

    Returns:
        dict: Success message with new encoder
    """
    settings = get_settings()
    settings.update_encoder(update.encoder)

    return {"message": "Encoder settings updated successfully", "encoder": update.encoder}


@router.get("/")
async def get_all_settings():
    """
    Get all application settings.

    Returns:
        dict: All current settings
    """
    settings = get_settings()
    return settings.get_all()
