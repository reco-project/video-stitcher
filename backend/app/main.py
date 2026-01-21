# backend/app/main.py

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.utils.logger import get_logger, configure_uvicorn_logging, info
from app.repositories.file_lens_profile_store import FileLensProfileStore
from app.repositories.lens_profile_store import LensProfileStore
from app.repositories.file_match_store import FileMatchStore
from app.repositories.match_store import MatchStore
import app.routers.profiles as profiles_router
import app.routers.matches as matches_router
import app.routers.processing as processing_router
import app.routers.settings as settings_router

# Initialize logging
logger = get_logger(__name__)
configure_uvicorn_logging()
info("=" * 60)
info("VIDEO STITCHER BACKEND STARTING")
info("=" * 60)

# Initialize lens profile store
PROFILES_DIR = Path(__file__).parent.parent / "data" / "lens_profiles"
profile_store = FileLensProfileStore(str(PROFILES_DIR))

# Initialize match store (file-based, persistent)
MATCHES_DIR = Path(__file__).parent.parent / "data" / "matches"
match_store = FileMatchStore(str(MATCHES_DIR))

# Videos directory for static file serving
VIDEOS_DIR = Path(__file__).parent.parent / "data" / "videos"


def get_profile_store() -> LensProfileStore:
    """Dependency injection for profile store."""
    return profile_store


def get_match_store() -> MatchStore:
    """Dependency injection for match store."""
    return match_store


app = FastAPI(title="Video Stitcher Backend")

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
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
