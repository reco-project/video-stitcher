"""
Data paths configuration

Centralizes all data directory paths and supports using user data folder
via USER_DATA_PATH environment variable (set by Electron).
"""

import os
from pathlib import Path

# Get user data path from environment (set by Electron) or use backend/data as fallback
USER_DATA_PATH = os.environ.get('USER_DATA_PATH')

if USER_DATA_PATH:
    # Production: Use Electron user data folder
    BASE_DATA_DIR = Path(USER_DATA_PATH) / 'backend_data'
else:
    # Development: Use backend/data
    BASE_DATA_DIR = Path(__file__).parent.parent / 'data'

# Ensure base directory exists
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# User-generated data (should be in user data folder)
MATCHES_DIR = BASE_DATA_DIR / 'matches'
VIDEOS_DIR = BASE_DATA_DIR / 'videos'
TEMP_DIR = BASE_DATA_DIR / 'temp'
LOGS_DIR = BASE_DATA_DIR / 'logs'

# Static app resources (can stay in backend folder)
PROFILES_DIR = Path(__file__).parent.parent / 'data' / 'lens_profiles'

# Ensure directories exist
for directory in [MATCHES_DIR, VIDEOS_DIR, TEMP_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
