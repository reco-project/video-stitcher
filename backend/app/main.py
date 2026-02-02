# backend/app/main.py

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import sys
import os

# Ensure unbuffered output for subprocess logging (important for PyInstaller)
os.environ.setdefault('PYTHONUNBUFFERED', '1')
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

from app.utils.logger import get_logger, configure_uvicorn_logging, info
from app.repositories.hybrid_lens_profile_store import HybridLensProfileStore
from app.repositories.file_match_store import FileMatchStore
from app.repositories.match_store import MatchStore
from app.data_paths import (
    PROFILES_DIR,
    PROFILES_DB_PATH,
    USE_SQLITE_PROFILES,
    USER_PROFILES_DIR,
    FAVORITES_FILE,
    MATCHES_DIR,
    VIDEOS_DIR,
)
import app.routers.profiles as profiles_router
import app.routers.matches as matches_router
import app.routers.processing as processing_router
import app.routers.settings as settings_router

# Fix for Windows: Use SelectorEventLoop instead of ProactorEventLoop
# This prevents timeout issues when running uvicorn on Windows
# Must be set AFTER all imports to avoid issues with scipy and other libraries
if sys.platform == 'win32':
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize logging
logger = get_logger(__name__)
configure_uvicorn_logging()
info("=" * 60)
info("VIDEO STITCHER BACKEND STARTING")
info("=" * 60)

# Initialize lens profile store using hybrid architecture:
# - SQLite database for bundled official profiles (read-only)
# - User JSON files for user-created profiles (read-write)
# - Separate favorites file for managing favorites across both sources
sqlite_path = str(PROFILES_DB_PATH) if USE_SQLITE_PROFILES else None
if sqlite_path:
    info(f"Official profiles: SQLite database at {PROFILES_DB_PATH}")
else:
    info(f"Official profiles: JSON files at {PROFILES_DIR}")

info(f"User profiles: {USER_PROFILES_DIR}")
info(f"Favorites file: {FAVORITES_FILE}")

profile_store = HybridLensProfileStore(
    sqlite_db_path=sqlite_path,
    user_profiles_dir=str(USER_PROFILES_DIR),
    favorites_file=str(FAVORITES_FILE),
    official_profiles_dir=str(PROFILES_DIR) if not sqlite_path else None,
)

# Initialize match store (file-based, persistent)
match_store = FileMatchStore(str(MATCHES_DIR))

# Videos directory for static file serving
# (Already created by data_paths module)


def get_profile_store() -> HybridLensProfileStore:
    """Dependency injection for profile store."""
    return profile_store


def get_match_store() -> MatchStore:
    """Dependency injection for match store."""
    return match_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup: Check for stale processing states and inconsistent status
    try:
        logger.info("Checking for stale processing states...")
        matches = match_store.list_all()
        active_statuses = ["transcoding", "calibrating"]
        stale_count = 0

        for match in matches:
            # Fix stale processing states (interrupted during transcoding/calibrating)
            if match.processing and match.processing.status in active_statuses:
                logger.warning(f"Found stale processing state for match {match.id}: status={match.processing.status}")
                match.update_processing(
                    status="error",
                    step=None,
                    message="Processing interrupted (app was closed)",
                    error_code="INTERRUPTED",
                    error_message="Processing was interrupted. Please retry.",
                )
                match_store.update(match.id, match.model_dump(exclude_none=False))
                stale_count += 1

        if stale_count > 0:
            logger.info(f"Reset {stale_count} stale processing state(s)")
        else:
            logger.info("No stale processing states found")
    except Exception as e:
        logger.error(f"Error checking stale processing states: {e}", exc_info=True)

    yield

    # Shutdown (if needed)
    logger.info("Application shutting down")


app = FastAPI(title="Video Stitcher Backend", lifespan=lifespan)

# Allow requests from Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers with dependency override
logger.info("Registering API routers...")
app.include_router(profiles_router.router, prefix="/api", tags=["profiles"])
app.dependency_overrides[profiles_router.get_store] = get_profile_store

app.include_router(matches_router.router, prefix="/api", tags=["matches"])
app.dependency_overrides[matches_router.get_store] = get_match_store

app.include_router(processing_router.router, prefix="/api", tags=["processing"])

app.include_router(settings_router.router, prefix="/api", tags=["settings"])

# Mount static files for video serving
# Ensure the videos directory exists
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")
logger.info(f"Static files mounted at /videos -> {VIDEOS_DIR}")


@app.get("/")
async def root():
    return {"message": "FastAPI backend is running!"}


@app.get("/api/health")
async def health_check():
    logger.debug("Health check requested")
    return {"status": "ok"}


def run_server():
    """Run the server without reload (for production/Electron)."""
    # Use workers=1 and no reload to prevent process spawning issues with PyInstaller
    # Enable access_log to see incoming requests
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=True,
        workers=1,
    )


if __name__ == "__main__":
    import os
    import multiprocessing

    # Required for PyInstaller to prevent infinite process spawning
    multiprocessing.freeze_support()

    # Use reload only in development (when run via npm run backend-dev)
    # Don't use reload when started by Electron (USER_DATA_PATH is set)
    use_reload = "USER_DATA_PATH" not in os.environ

    if use_reload:
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
    else:
        run_server()
