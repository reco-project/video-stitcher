"""
Application configuration and settings.

Manages user preferences and system configuration including
GPU encoder selection for FFmpeg transcoding.
"""

import os
import json
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field

# Settings file location
SETTINGS_DIR = Path(__file__).parent.parent / "data"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

# Encoder types
EncoderType = Literal["auto", "h264_nvenc", "h264_qsv", "h264_amf", "libx264"]


class AppSettings(BaseModel):
    """Application settings model."""

    # GPU Encoder preference
    encoder: EncoderType = Field(
        default="auto",
        description="Preferred video encoder (auto=detect, h264_nvenc=NVIDIA, h264_qsv=Intel, h264_amf=AMD, libx264=CPU)",
    )

    # Future settings can go here
    # max_processing_threads: int = 4
    # temp_dir_path: Optional[str] = None


class Settings:
    """Settings manager with persistence."""

    def __init__(self):
        self._settings: AppSettings = self._load()

    def _load(self) -> AppSettings:
        """Load settings from disk or return defaults."""
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                return AppSettings(**data)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Failed to load settings: {e}")
                return AppSettings()
        return AppSettings()

    def _save(self) -> None:
        """Save settings to disk."""
        try:
            os.makedirs(SETTINGS_DIR, exist_ok=True)
            with open(SETTINGS_FILE, "w") as f:
                json.dump(self._settings.model_dump(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

    @property
    def encoder(self) -> EncoderType:
        """Get encoder preference."""
        return self._settings.encoder

    def update_encoder(self, encoder: EncoderType) -> None:
        """Update encoder preference."""
        self._settings.encoder = encoder
        self._save()

    def get_all(self) -> dict:
        """Get all settings as dict."""
        return self._settings.model_dump()

    def update(self, **kwargs) -> None:
        """Update multiple settings."""
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
        self._save()

    def reset(self) -> None:
        """Reset to default settings."""
        self._settings = AppSettings()
        self._save()


# Global settings instance
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
