"""
Data paths configuration

Centralizes all data directory paths and supports using user data folder
via USER_DATA_PATH environment variable (set by Electron).
"""

import os
import sys
from pathlib import Path

# Get user data path from environment (set by Electron)
# Only use it in production (PyInstaller bundle), ignore in development
USER_DATA_PATH = os.environ.get('USER_DATA_PATH')
IS_PRODUCTION = getattr(sys, 'frozen', False)

if IS_PRODUCTION and USER_DATA_PATH:
    # Production: Use Electron user data folder
    BASE_DATA_DIR = Path(USER_DATA_PATH) / 'backend_data'
else:
    # Development: Use devData/ at project root (gitignored, not packaged)
    # Ignores USER_DATA_PATH to avoid disturbing production installation
    BASE_DATA_DIR = Path(__file__).parent.parent.parent / 'devData'

# Ensure base directory exists
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# User-generated data (should be in user data folder)
MATCHES_DIR = BASE_DATA_DIR / 'matches'
VIDEOS_DIR = BASE_DATA_DIR / 'videos'
TEMP_DIR = BASE_DATA_DIR / 'temp'
LOGS_DIR = BASE_DATA_DIR / 'logs'

# Static app resources
# In production (PyInstaller), lens_profiles.sqlite is in the dist_bundle/data directory
# In development, use JSON files from backend/data/lens_profiles


def _get_profiles_db_path() -> Path:
    """Get the path to lens_profiles.sqlite database."""
    # Check if running from PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Production: Look in the bundle's data directory
        bundle_dir = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(sys.executable).parent
        db_path = bundle_dir / 'data' / 'lens_profiles.sqlite'
        if db_path.exists():
            return db_path
        # Fallback: check parent directory (for different PyInstaller modes)
        alt_path = Path(sys.executable).parent / 'data' / 'lens_profiles.sqlite'
        if alt_path.exists():
            return alt_path

    # Development: Look for SQLite in electron/resources (if built)
    dev_db_path = Path(__file__).parent.parent.parent / 'electron' / 'resources' / 'lens_profiles.sqlite'
    if dev_db_path.exists():
        return dev_db_path

    # No SQLite database found - will need to fall back to JSON files
    return None


def _get_profiles_json_dir() -> Path:
    """Get the path to lens profile JSON files (development fallback)."""
    return Path(__file__).parent.parent / 'data' / 'lens_profiles'


# Lens profile paths
PROFILES_DB_PATH = _get_profiles_db_path()
PROFILES_DIR = _get_profiles_json_dir()

# User-created profiles directory (in user data folder)
USER_PROFILES_DIR = BASE_DATA_DIR / 'user_profiles'

# Favorites file (stores IDs of favorited profiles from any source)
FAVORITES_FILE = BASE_DATA_DIR / 'favorites.json'

# Flag to indicate if SQLite database is available
USE_SQLITE_PROFILES = PROFILES_DB_PATH is not None and PROFILES_DB_PATH.exists()

# Ensure directories exist
for directory in [MATCHES_DIR, VIDEOS_DIR, TEMP_DIR, LOGS_DIR, USER_PROFILES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_ffmpeg_path() -> str:
    """Get the path to the FFmpeg binary.

    Always uses the bundled FFmpeg (which includes GPU encoder support).
    Falls back to system PATH only if bundled binary is not found.

    Returns:
        Path to ffmpeg executable (or just 'ffmpeg' for system PATH)
    """
    import shutil

    ffmpeg_name = 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg'

    # In production (PyInstaller bundle)
    if IS_PRODUCTION:
        if hasattr(sys, '_MEIPASS'):
            bundled = Path(sys._MEIPASS) / 'bin' / ffmpeg_name
        else:
            bundled = Path(sys.executable).parent / 'bin' / ffmpeg_name
        if bundled.exists():
            return str(bundled)
    else:
        # In development: use bundled FFmpeg from backend/bin
        bundled = Path(__file__).parent.parent / 'bin' / ffmpeg_name
        if bundled.exists():
            return str(bundled)

    # Fallback to system PATH
    system_ffmpeg = shutil.which('ffmpeg')
    if system_ffmpeg:
        return system_ffmpeg

    return 'ffmpeg'  # Final fallback


def get_ffprobe_path() -> str:
    """Get the path to the FFprobe binary.

    Returns:
        Path to ffprobe executable (or just 'ffprobe' for system PATH)
    """
    import shutil

    ffmpeg_path = get_ffmpeg_path()

    # If using a specific FFmpeg path, use the corresponding ffprobe
    if ffmpeg_path != 'ffmpeg':
        ffmpeg_dir = Path(ffmpeg_path).parent
        ffprobe_name = 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'
        ffprobe_path = ffmpeg_dir / ffprobe_name
        if ffprobe_path.exists():
            return str(ffprobe_path)

    # Fallback to system PATH
    return shutil.which('ffprobe') or 'ffprobe'
