# backend/app/main.py

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.repositories.file_lens_profile_store import FileLensProfileStore
from app.repositories.lens_profile_store import LensProfileStore
from app.repositories.match_store import InMemoryMatchStore
import app.routers.profiles as profiles_router
import app.routers.matches as matches_router

# Initialize lens profile store
PROFILES_DIR = Path(__file__).parent.parent / "lens_profiles"
profile_store = FileLensProfileStore(str(PROFILES_DIR))

# Initialize match store (in-memory, session-based)
match_store = InMemoryMatchStore()


def get_profile_store() -> LensProfileStore:
    """Dependency injection for profile store."""
    return profile_store


def get_match_store() -> InMemoryMatchStore:
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
app.include_router(profiles_router.router, prefix="/api", tags=["profiles"])
app.dependency_overrides[profiles_router.get_store] = get_profile_store

app.include_router(matches_router.router, prefix="/api", tags=["matches"])
app.dependency_overrides[matches_router.get_store] = get_match_store


@app.get("/")
async def root():
    return {"message": "FastAPI backend is running!"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
