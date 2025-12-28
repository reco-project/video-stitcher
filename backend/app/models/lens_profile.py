"""
Lens profile data model with validation.

This module defines the validation schema for lens profiles.
Pydantic is used as a validator only - models are immediately converted to dicts.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class LensProfileModel(BaseModel):
    """
    Lens profile validation model.

    Validates camera calibration data for video stitching.
    Treats calibration data (camera_matrix, distortion_coeffs) as opaque numeric data.
    """

    # Required fields
    id: str = Field(..., description="Unique identifier in slug format")
    camera_brand: str = Field(..., min_length=1, description="Camera manufacturer")
    camera_model: str = Field(..., min_length=1, description="Camera model name")
    resolution: dict = Field(..., description="Video resolution {width: int, height: int}")
    distortion_model: Literal["fisheye_kb4"] = Field(
        ..., description="Distortion model type (only fisheye_kb4 supported in v1)"
    )
    camera_matrix: dict = Field(..., description="Camera intrinsic matrix {fx, fy, cx, cy}")
    distortion_coeffs: list[float] = Field(..., description="Distortion coefficients (4 floats for fisheye_kb4)")

    # Optional fields
    lens_model: Optional[str] = Field(None, description="Lens mode (e.g., Linear, Wide)")
    metadata: Optional[dict] = Field(
        None, description="Optional provenance info (calibrated_by, calibration_date, notes)"
    )

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Validate ID follows slug format rules."""
        if not v:
            raise ValueError("id cannot be empty")

        if len(v) > 100:
            raise ValueError("id cannot exceed 100 characters")

        # Must be lowercase alphanumeric + hyphens only
        if not re.match(r"^[a-z0-9\-]+$", v):
            raise ValueError("id must contain only lowercase letters, numbers, and hyphens")

        # Cannot start or end with hyphen
        if v.startswith("-") or v.endswith("-"):
            raise ValueError("id cannot start or end with hyphen")

        return v

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: dict) -> dict:
        """Validate resolution has required fields with positive integers."""
        if not isinstance(v, dict):
            raise ValueError("resolution must be an object")

        if "width" not in v or "height" not in v:
            raise ValueError("resolution must contain 'width' and 'height'")

        width = v["width"]
        height = v["height"]

        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("resolution width and height must be integers")

        if width <= 0 or height <= 0:
            raise ValueError("resolution width and height must be positive")

        return v

    @field_validator("camera_matrix")
    @classmethod
    def validate_camera_matrix(cls, v: dict) -> dict:
        """Validate camera_matrix has required fields with positive floats."""
        if not isinstance(v, dict):
            raise ValueError("camera_matrix must be an object")

        required_fields = ["fx", "fy", "cx", "cy"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"camera_matrix must contain '{field}'")

        for field in required_fields:
            val = v[field]
            if not isinstance(val, (int, float)):
                raise ValueError(f"camera_matrix.{field} must be a number")
            if val <= 0:
                raise ValueError(f"camera_matrix.{field} must be positive")

        return v

    @field_validator("distortion_coeffs")
    @classmethod
    def validate_distortion_coeffs(cls, v: list[float]) -> list[float]:
        """Validate distortion_coeffs has exactly 4 floats for fisheye_kb4."""
        if not isinstance(v, list):
            raise ValueError("distortion_coeffs must be an array")

        if len(v) != 4:
            raise ValueError("distortion_coeffs must contain exactly 4 values for fisheye_kb4 model")

        for i, coeff in enumerate(v):
            if not isinstance(coeff, (int, float)):
                raise ValueError(f"distortion_coeffs[{i}] must be a number")

        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Optional[dict]) -> Optional[dict]:
        """Validate metadata is optional and opaque (no field validation)."""
        if v is not None and not isinstance(v, dict):
            raise ValueError("metadata must be an object if provided")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "gopro-hero10black-linear-3840x2160",
                "camera_brand": "GoPro",
                "camera_model": "HERO10 Black",
                "lens_model": "Linear",
                "resolution": {"width": 3840, "height": 2160},
                "distortion_model": "fisheye_kb4",
                "camera_matrix": {
                    "fx": 1796.3208206894308,
                    "fy": 1797.22277342282,
                    "cx": 1919.372365976781,
                    "cy": 1063.171593155705,
                },
                "distortion_coeffs": [0.03421388, 0.0676732, -0.0740897, 0.02994442],
                "metadata": {
                    "calibrated_by": "John Doe",
                    "calibration_date": "2025-12-27",
                    "notes": "Calibrated using 28 images",
                },
            }
        }
    }
