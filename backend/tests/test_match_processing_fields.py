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
    """Test that new matches default to 'pending' status."""
    match = {
        "id": "match-status-test",
        "name": "Status Test Match",
        "left_videos": [{"path": "/test/left.mp4"}],
        "right_videos": [{"path": "/test/right.mp4"}],
    }

    created = store.create(match)
    assert created["status"] == "pending"
    assert created["error_code"] is None
    assert created["error_message"] is None
    assert created["processing_step"] is None


def test_update_match_to_transcoding(store, sample_match):
    """Test updating match status to transcoding."""
    store.create(sample_match)

    sample_match["status"] = "transcoding"
    sample_match["processing_step"] = "transcoding"
    sample_match["processing_started_at"] = "2025-01-01T00:00:00Z"

    updated = store.update(sample_match["id"], sample_match)

    assert updated["status"] == "transcoding"
    assert updated["processing_step"] == "transcoding"
    assert updated["processing_started_at"] == "2025-01-01T00:00:00Z"


def test_update_match_to_error_state(store, sample_match):
    """Test updating match to error state."""
    store.create(sample_match)

    sample_match["status"] = "error"
    sample_match["error_code"] = "TRANSCODING_FAILED"
    sample_match["error_message"] = "Failed to sync videos"
    sample_match["processing_completed_at"] = "2025-01-01T00:10:00Z"

    updated = store.update(sample_match["id"], sample_match)

    assert updated["status"] == "error"
    assert updated["error_code"] == "TRANSCODING_FAILED"
    assert updated["error_message"] == "Failed to sync videos"
    assert updated["processing_completed_at"] == "2025-01-01T00:10:00Z"


def test_update_match_to_ready(store, sample_match):
    """Test updating match to ready state with src."""
    store.create(sample_match)

    sample_match["status"] = "ready"
    sample_match["src"] = "videos/match-test-123.mp4"
    sample_match["processing_completed_at"] = "2025-01-01T00:15:00Z"
    sample_match["processing_step"] = None

    updated = store.update(sample_match["id"], sample_match)

    assert updated["status"] == "ready"
    assert updated["src"] == "videos/match-test-123.mp4"
    assert updated["processing_completed_at"] == "2025-01-01T00:15:00Z"
    assert updated["processing_step"] is None


def test_create_match_with_explicit_status(store):
    """Test creating match with explicit status."""
    match = {
        "id": "match-explicit-status",
        "name": "Explicit Status",
        "status": "ready",
        "left_videos": [{"path": "/test/left.mp4"}],
        "right_videos": [{"path": "/test/right.mp4"}],
        "src": "videos/existing.mp4",
    }

    created = store.create(match)
    assert created["status"] == "ready"
    assert created["src"] == "videos/existing.mp4"


def test_processing_fields_persist_across_reload(store, temp_matches_dir, sample_match):
    """Test that processing fields persist when reloading from disk."""
    store.create(sample_match)

    # Update with processing fields
    sample_match["status"] = "calibrating"
    sample_match["processing_step"] = "feature_matching"
    sample_match["processing_started_at"] = "2025-01-01T00:00:00Z"
    store.update(sample_match["id"], sample_match)

    # Create new store instance (simulates app restart)
    new_store = FileMatchStore(base_path=str(temp_matches_dir))
    loaded = new_store.get_by_id(sample_match["id"])

    assert loaded["status"] == "calibrating"
    assert loaded["processing_step"] == "feature_matching"
    assert loaded["processing_started_at"] == "2025-01-01T00:00:00Z"
