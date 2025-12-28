"""
Unit tests for FileMatchStore repository implementation.

Tests CRUD operations, file I/O, validation, and error handling.
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
        "id": "match-test-123",
        "name": "Test Match",
        "left_videos": [{"path": "/videos/left1.mp4", "profile_id": "gopro-hero10-linear"}],
        "right_videos": [{"path": "/videos/right1.mp4", "profile_id": "gopro-hero9-wide"}],
        "params": {
            "cameraAxisOffset": 0.23,
            "intersect": 0.55,
            "zRx": 0.0,
            "xTy": 0.0,
            "xRz": 0.0,
        },
        "left_uniforms": {
            "width": 3840,
            "height": 2160,
            "fx": 2532.61,
            "fy": 2537.19,
            "cx": 2658.31,
            "cy": 1501.14,
            "d": [0.3503, 0.0307, 0.2982, -0.159],
        },
        "right_uniforms": {
            "width": 2704,
            "height": 1520,
            "fx": 1796.32,
            "fy": 1797.22,
            "cx": 1919.37,
            "cy": 1063.17,
            "d": [0.0342, 0.0677, -0.0741, 0.0299],
        },
        "metadata": {
            "left_profile_id": "gopro-hero10-linear",
            "right_profile_id": "gopro-hero9-wide",
        },
    }


@pytest.fixture
def sample_match_with_src(sample_match):
    """Sample match with output src URL."""
    match = sample_match.copy()
    match["src"] = "https://storage.example.com/output.mp4"
    return match


class TestFileMatchStore:
    """Tests for FileMatchStore."""

    def test_create_match(self, store, sample_match):
        """Test creating a new match."""
        created = store.create(sample_match)

        assert created["id"] == sample_match["id"]
        assert created["name"] == sample_match["name"]
        assert len(created["left_videos"]) == 1
        assert len(created["right_videos"]) == 1
        assert "created_at" in created  # Auto-generated timestamp

    def test_create_match_with_src(self, store, sample_match_with_src):
        """Test creating match with src URL."""
        created = store.create(sample_match_with_src)

        assert created["src"] == sample_match_with_src["src"]

    def test_create_duplicate_raises_error(self, store, sample_match):
        """Test creating duplicate match raises ValueError."""
        store.create(sample_match)

        with pytest.raises(ValueError) as exc_info:
            store.create(sample_match)

        assert "already exists" in str(exc_info.value)

    def test_create_validates_video_paths(self, store, sample_match):
        """Test creating match with empty video path raises error."""
        invalid_match = sample_match.copy()
        invalid_match["left_videos"] = [{"path": ""}]

        with pytest.raises(ValueError):
            store.create(invalid_match)

    def test_get_by_id(self, store, sample_match):
        """Test retrieving match by ID."""
        store.create(sample_match)

        retrieved = store.get_by_id(sample_match["id"])

        assert retrieved is not None
        assert retrieved["id"] == sample_match["id"]
        assert retrieved["name"] == sample_match["name"]

    def test_get_by_id_nonexistent(self, store):
        """Test retrieving non-existent match returns None."""
        result = store.get_by_id("non-existent-match")
        assert result is None

    def test_list_all(self, store, sample_match):
        """Test listing all matches."""
        # Create multiple matches
        match1 = sample_match.copy()
        match1["id"] = "match-1"
        match1["name"] = "Match 1"

        match2 = sample_match.copy()
        match2["id"] = "match-2"
        match2["name"] = "Match 2"

        store.create(match1)
        store.create(match2)

        all_matches = store.list_all()

        assert len(all_matches) == 2
        assert any(m["id"] == "match-1" for m in all_matches)
        assert any(m["id"] == "match-2" for m in all_matches)

    def test_list_all_empty(self, store):
        """Test listing matches when none exist."""
        result = store.list_all()
        assert result == []

    def test_update_match(self, store, sample_match):
        """Test updating an existing match."""
        store.create(sample_match)

        # Update the match
        updated_data = sample_match.copy()
        updated_data["name"] = "Updated Match Name"
        updated_data["src"] = "https://storage.example.com/updated.mp4"
        updated_data["params"]["intersect"] = 0.6

        updated = store.update(sample_match["id"], updated_data)

        assert updated["name"] == "Updated Match Name"
        assert updated["src"] == "https://storage.example.com/updated.mp4"
        assert updated["params"]["intersect"] == 0.6

        # Verify persistence
        retrieved = store.get_by_id(sample_match["id"])
        assert retrieved["name"] == "Updated Match Name"

    def test_update_nonexistent_raises_error(self, store, sample_match):
        """Test updating non-existent match raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            store.update("non-existent-match", sample_match)

        assert "not found" in str(exc_info.value)

    def test_delete_match(self, store, sample_match):
        """Test deleting a match."""
        store.create(sample_match)

        result = store.delete(sample_match["id"])

        assert result is True
        assert store.get_by_id(sample_match["id"]) is None

    def test_delete_nonexistent(self, store):
        """Test deleting non-existent match returns False."""
        result = store.delete("non-existent-match")
        assert result is False

    def test_exists(self, store, sample_match):
        """Test checking if match exists."""
        assert store.exists(sample_match["id"]) is False

        store.create(sample_match)

        assert store.exists(sample_match["id"]) is True

    def test_file_persistence(self, store, sample_match, temp_matches_dir):
        """Test that matches are persisted to disk."""
        store.create(sample_match)

        # Check file exists
        match_file = temp_matches_dir / f"{sample_match['id']}.json"
        assert match_file.exists()

        # Create new store instance with same directory
        new_store = FileMatchStore(base_path=str(temp_matches_dir))
        retrieved = new_store.get_by_id(sample_match["id"])

        assert retrieved is not None
        assert retrieved["id"] == sample_match["id"]

    def test_multiple_videos(self, store, sample_match):
        """Test match with multiple videos per camera."""
        multi_video_match = sample_match.copy()
        multi_video_match["id"] = "match-multi"
        multi_video_match["left_videos"] = [
            {"path": "/videos/left1.mp4", "profile_id": "gopro-hero10-linear"},
            {"path": "/videos/left2.mp4", "profile_id": "gopro-hero10-linear"},
            {"path": "/videos/left3.mp4", "profile_id": "gopro-hero10-linear"},
        ]
        multi_video_match["right_videos"] = [
            {"path": "/videos/right1.mp4", "profile_id": "gopro-hero9-wide"},
            {"path": "/videos/right2.mp4", "profile_id": "gopro-hero9-wide"},
        ]

        created = store.create(multi_video_match)

        assert len(created["left_videos"]) == 3
        assert len(created["right_videos"]) == 2

    def test_optional_fields(self, store, sample_match):
        """Test match creation with optional fields."""
        minimal_match = {
            "id": "match-minimal",
            "left_videos": [{"path": "/video1.mp4"}],
            "right_videos": [{"path": "/video2.mp4"}],
            "params": {"cameraAxisOffset": 0.23, "intersect": 0.55},
            "left_uniforms": {
                "width": 1920,
                "height": 1080,
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 960.0,
                "cy": 540.0,
                "d": [0.0, 0.0, 0.0, 0.0],
            },
            "right_uniforms": {
                "width": 1920,
                "height": 1080,
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 960.0,
                "cy": 540.0,
                "d": [0.0, 0.0, 0.0, 0.0],
            },
        }

        created = store.create(minimal_match)

        assert created["name"] is None  # Optional field not provided
        assert created["src"] is None  # Optional field not provided
        assert "created_at" in created

    def test_legacy_label_field(self, store, sample_match):
        """Test match with legacy label field."""
        legacy_match = sample_match.copy()
        legacy_match["label"] = "Legacy Label"

        created = store.create(legacy_match)

        assert created["label"] == "Legacy Label"

        retrieved = store.get_by_id(legacy_match["id"])
        assert retrieved["label"] == "Legacy Label"
