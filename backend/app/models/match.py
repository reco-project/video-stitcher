"""
Pydantic models for match data.

A match represents a pair of videos (left/right) with their assigned lens profiles.
This is the first step in the video stitching pipeline.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class VideoInput(BaseModel):
    """Video file input with lens profile."""
    
    path: str = Field(..., description="Local filesystem path to video file")
    profile_id: Optional[str] = Field(None, description="Lens profile ID assigned to this video")
    
    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate that path is not empty."""
        if not v or not v.strip():
            raise ValueError("Video path cannot be empty")
        return v.strip()


class MatchModel(BaseModel):
    """
    Match model representing videos for stitching.
    
    Attributes:
        id: Unique match identifier
        name: User-friendly match name
        left_videos: Left camera videos with lens profiles
        right_videos: Right camera videos with lens profiles
        params: Calibration parameters (default values for now)
        uniforms: Camera uniforms (default values for now)
        created_at: Match creation timestamp
        metadata: Additional metadata (future use)
    """
    
    id: str = Field(..., description="Unique match identifier")
    name: str = Field(..., description="Match display name")
    left_videos: List[VideoInput] = Field(..., description="Left camera videos")
    right_videos: List[VideoInput] = Field(..., description="Right camera videos")
    params: Dict[str, float] = Field(
        default_factory=lambda: {
            "cameraAxisOffset": 0.2335068393564666,
            "intersect": 0.5472022558355283,
            "zRx": -0.035271498000659006,
            "xTy": -0.0014468249987039522,
            "xRz": 0.00836074850140415,
        },
        description="Calibration parameters"
    )
    uniforms: Dict[str, Any] = Field(
        default_factory=lambda: {
            "width": 3840,
            "height": 2160,
            "fx": 1796.3208206894308,
            "fy": 1797.22277342282,
            "cx": 1919.372365976781,
            "cy": 1063.171593155705,
            "d": [0.03421388, 0.0676732, -0.0740897, 0.02994442],
        },
        description="Camera uniforms"
    )
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate that ID is not empty and contains only valid characters."""
        if not v or not v.strip():
            raise ValueError("Match ID cannot be empty")
        
        # Allow alphanumeric, hyphens, underscores
        cleaned = v.strip()
        if not all(c.isalnum() or c in ('-', '_') for c in cleaned):
            raise ValueError("Match ID can only contain alphanumeric characters, hyphens, and underscores")
        
        return cleaned
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is not empty."""
        if not v or not v.strip():
            raise ValueError("Match name cannot be empty")
        return v.strip()
