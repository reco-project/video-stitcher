"""
Additional tests for match processing fields (status, error_code, etc.)
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from app.repositories.file_match_store import FileMatchStore


@pytest.fixture
def temp_matches_dir():
    """Create a temporary directory for test matches."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def store(temp_matches_dir):
    """Create a FileMatchStore with temporary directory."""
    return FileMatchStore(base_path=str(temp_matches_dir))


@pytest.fixture
def sample_match():
    """Sample valid match data."""
    return {
        "id": "match-status-test",
        "name": "Status Test Match",
        "left_videos": [{"path": "/test/left.mp4"}],
        "right_videos": [{"path": "/test/right.mp4"}],
    }


def test_create_match_with_default_status(store):
    """Test that new matches default to 'pending' status in nested structure."""
    match = {
        "id": "match-status-test",
        "name": "Status Test Match",
        "left_videos": [{"path": "/test/left.mp4"}],
        "right_videos": [{"path": "/test/right.mp4"}],
    }

    created = store.create(match)
    # Check nested structure
    assert created.processing.status == "pending"
    assert created.processing.error_code is None
    assert created.processing.error_message is None
    assert created.processing.step is None


def test_update_match_to_transcoding(store, sample_match):
    """Test updating match status to transcoding."""
    created = store.create(sample_match)

    # Update using model method
    created.update_processing(
        status="transcoding",
        step="transcoding",
        started_at="2025-01-01T00:00:00Z",
    )

    updated = store.update(created.id, created.model_dump(exclude_none=False))

    assert updated.processing.status == "transcoding"
    assert updated.processing.step == "transcoding"
    assert updated.processing.started_at == "2025-01-01T00:00:00Z"


def test_update_match_to_error_state(store, sample_match):
    """Test updating match to error state."""
    created = store.create(sample_match)

    # Update using model method
    created.update_processing(
        status="error",
        step=None,
        completed_at="2025-01-01T00:10:00Z",
        error_code="TRANSCODING_FAILED",
        error_message="Failed to sync videos",
    )

    updated = store.update(created.id, created.model_dump(exclude_none=False))

    assert updated.processing.status == "error"
    assert updated.processing.error_code == "TRANSCODING_FAILED"
    assert updated.processing.error_message == "Failed to sync videos"
    assert updated.processing.completed_at == "2025-01-01T00:10:00Z"


def test_update_match_to_ready(store, sample_match):
    """Test updating match to ready state with src."""
    created = store.create(sample_match)

    created.src = "videos/match-test-123.mp4"
    created.update_processing(
        status="ready",
        step=None,
        completed_at="2025-01-01T00:15:00Z",
    )

    updated = store.update(created.id, created.model_dump(exclude_none=False))

    assert updated.processing.status == "ready"
    assert updated.src == "videos/match-test-123.mp4"
    assert updated.processing.completed_at == "2025-01-01T00:15:00Z"
    assert updated.processing.step is None


def test_create_match_with_explicit_status(store):
    """Test creating match with explicit status."""
    match = {
        "id": "match-explicit-status",
        "name": "Explicit Status",
        "left_videos": [{"path": "/test/left.mp4"}],
        "right_videos": [{"path": "/test/right.mp4"}],
        "src": "videos/existing.mp4",
        "processing": {
            "status": "ready",
            "step": None,
            "message": None,
            "started_at": None,
            "completed_at": None,
            "error_code": None,
            "error_message": None,
        },
    }

    created = store.create(match)
    assert created.processing.status == "ready"
    assert created.src == "videos/existing.mp4"


def test_processing_fields_persist_across_reload(store, temp_matches_dir, sample_match):
    """Test that processing fields persist when reloading from disk."""
    created = store.create(sample_match)

    # Update with nested processing fields
    created.update_processing(
        status="calibrating",
        step="feature_matching",
        started_at="2025-01-01T00:00:00Z",
    )
    store.update(created.id, created.model_dump(exclude_none=False))

    # Create new store instance (simulates app restart)
    new_store = FileMatchStore(base_path=str(temp_matches_dir))
    loaded = new_store.get_by_id(created.id)

    assert loaded.processing.status == "calibrating"
    assert loaded.processing.step == "feature_matching"
    assert loaded.processing.started_at == "2025-01-01T00:00:00Z"
