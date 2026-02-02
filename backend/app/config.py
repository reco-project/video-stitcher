"""
Application configuration and settings.

Manages user preferences and system configuration including
GPU encoder selection for FFmpeg transcoding.

The backend reads/writes to the same settings.json as Electron to avoid
having separate settings files.
"""

import os
import sys
import json
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field

# Settings file location - use Electron's settings.json directly (not in backend_data subfolder)
# In production: userData/settings.json (same as Electron)
# In development: devData/settings.json
USER_DATA_PATH = os.environ.get('USER_DATA_PATH')
IS_PRODUCTION = getattr(sys, 'frozen', False)

if IS_PRODUCTION and USER_DATA_PATH:
    SETTINGS_FILE = Path(USER_DATA_PATH) / "settings.json"
else:
    SETTINGS_FILE = Path(__file__).parent.parent.parent / "devData" / "settings.json"

# Encoder types
EncoderType = Literal["auto", "h264_nvenc", "h264_qsv", "h264_amf", "libx264"]


class Settings:
    """
    Settings manager that shares the settings.json with Electron.

    Electron uses 'encoderPreference' field, we read/write that same field.
    """

    def __init__(self):
        self._cache = self._load()

    def _load(self) -> dict:
        """Load full settings from disk or return empty dict."""
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Failed to load settings: {e}")
                return {}
        return {}

    def _save(self) -> None:
        """Save settings to disk."""
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

    @property
    def encoder(self) -> EncoderType:
        """Get encoder preference (reads 'encoderPreference' from shared settings)."""
        # Reload to get latest (in case Electron changed it)
        self._cache = self._load()
        value = self._cache.get('encoderPreference', 'auto')
        # Validate it's a valid encoder type
        if value in ('auto', 'h264_nvenc', 'h264_qsv', 'h264_amf', 'libx264'):
            return value
        return 'auto'

    def update_encoder(self, encoder: EncoderType) -> None:
        """Update encoder preference (writes to 'encoderPreference' in shared settings)."""
        self._cache = self._load()  # Reload to preserve other settings
        self._cache['encoderPreference'] = encoder
        self._save()

    def get_all(self) -> dict:
        """Get all settings as dict."""
        self._cache = self._load()
        return self._cache

    def reset(self) -> None:
        """Reset encoder to default."""
        self._cache = self._load()
        self._cache['encoderPreference'] = 'auto'
        self._save()


# Global settings instance
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
