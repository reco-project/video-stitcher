"""
Tests for nested match structure (processing, transcode) and model methods.
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


class TestNestedStructure:
    """Tests for nested processing and transcode structures."""

    def test_create_match_has_nested_processing(self, store):
        """Test that new matches have nested processing structure."""
        match = {
            "id": "match-nested-test",
            "name": "Nested Test",
            "left_videos": [{"path": "/test/left.mp4"}],
            "right_videos": [{"path": "/test/right.mp4"}],
        }

        created = store.create(match)

        # Nested structure exists
        assert hasattr(created, 'processing')
        assert created.processing is not None
        assert created.processing.status == "pending"
        assert created.processing.step is None
        assert created.processing.message is None
        assert created.processing.started_at is None
        assert created.processing.completed_at is None
        assert created.processing.error_code is None
        assert created.processing.error_message is None

    def test_update_processing_creates_nested_and_flat(self, store):
        """Test that update_processing updates nested structure."""
        match = {
            "id": "match-update-test",
            "name": "Update Test",
            "left_videos": [{"path": "/test/left.mp4"}],
            "right_videos": [{"path": "/test/right.mp4"}],
        }

        created = store.create(match)

        # Update using model method
        created.update_processing(
            status="transcoding",
            step="transcoding",
            message="Encoding video...",
            started_at="2026-01-01T00:00:00Z",
        )

        store.update(created.id, created.model_dump(exclude_none=False))
        updated = store.get_by_id(created.id)

        # Nested structure updated
        assert updated.processing.status == "transcoding"
        assert updated.processing.step == "transcoding"
        assert updated.processing.message == "Encoding video..."
        assert updated.processing.started_at == "2026-01-01T00:00:00Z"

    def test_update_transcode_creates_nested_and_flat(self, store):
        """Test that update_transcode updates nested structure."""
        match = {
            "id": "match-transcode-test",
            "name": "Transcode Test",
            "left_videos": [{"path": "/test/left.mp4"}],
            "right_videos": [{"path": "/test/right.mp4"}],
        }

        created = store.create(match)

        # Update using model method
        created.update_transcode(
            fps=45.2,
            speed="14.5x",
            progress=75.0,
            current_time=30.5,
            total_duration=40.0,
            offset_seconds=0.25,
        )

        store.update(created.id, created.model_dump(exclude_none=False))
        updated = store.get_by_id(created.id)

        # Nested structure created
        assert updated.transcode is not None
        assert updated.transcode.fps == 45.2
        assert updated.transcode.speed == "14.5x"
        assert updated.transcode.progress == 75.0
        assert updated.transcode.current_time == 30.5
        assert updated.transcode.total_duration == 40.0
        assert updated.transcode.offset_seconds == 0.25

    def test_quality_settings_saved(self, store):
        """Test that quality settings are properly saved."""
        match = {
            "id": "match-quality-test",
            "name": "Quality Test",
            "left_videos": [{"path": "/test/left.mp4"}],
            "right_videos": [{"path": "/test/right.mp4"}],
            "quality_settings": {
                "preset": "1080p",
                "bitrate": "50M",
                "speed_preset": "veryfast",
                "resolution": "1080p",
                "use_gpu_decode": True,
            },
        }

        created = store.create(match)

        assert created.quality_settings is not None
        assert created.quality_settings.preset == "1080p"
        assert created.quality_settings.bitrate == "50M"
        assert created.quality_settings.speed_preset == "veryfast"
        assert created.quality_settings.resolution == "1080p"
        assert created.quality_settings.use_gpu_decode is True


class TestMigration:
    """Tests for helper functions that read from nested structures."""

    def test_get_processing_status_from_nested(self, store):
        """Test reading status from nested structure."""
        match = {
            "id": "test-status-read",
            "name": "Status Test",
            "left_videos": [{"path": "/test/left.mp4"}],
            "right_videos": [{"path": "/test/right.mp4"}],
            "processing": {"status": "transcoding"},
        }

        created = store.create(match)
        status = created.get_status()
        assert status == "transcoding"

    def test_get_transcode_fps_from_nested(self, store):
        """Test reading FPS from nested structure."""
        match = {
            "id": "test-fps-read",
            "name": "FPS Test",
            "left_videos": [{"path": "/test/left.mp4"}],
            "right_videos": [{"path": "/test/right.mp4"}],
        }

        created = store.create(match)
        created.update_transcode(fps=45.5)
        store.update(created.id, created.model_dump(exclude_none=False))

        updated = store.get_by_id(created.id)
        fps = updated.get_transcode_fps()
        assert fps == 45.5

    def test_match_with_nested_structure_persists(self, store):
        """Test that matches with nested structure persist correctly."""
        match = {
            "id": "test-nested-match",
            "name": "Nested Test",
            "left_videos": [{"path": "/test/left.mp4"}],
            "right_videos": [{"path": "/test/right.mp4"}],
            "processing": {"status": "ready", "step": "complete"},
            "transcode": {
                "fps": 42.0,
                "speed": "12.3x",
                "offset_seconds": 0.5,
            },
        }

        # Store should accept it
        created = store.create(match)

        # Should work with model methods
        status = created.get_status()
        fps = created.get_transcode_fps()

        assert status == "ready"
        assert fps == 42.0


class TestErrorHandling:
    """Tests for error handling in nested structure."""

    def test_error_state_in_nested_structure(self, store):
        """Test that error states work in nested structure."""
        match = {
            "id": "match-error-test",
            "name": "Error Test",
            "left_videos": [{"path": "/test/left.mp4"}],
            "right_videos": [{"path": "/test/right.mp4"}],
        }

        created = store.create(match)

        # Update to error state using model method
        created.update_processing(
            status="error",
            step=None,
            error_code="TRANSCODE_FAILED",
            error_message="FFmpeg crashed",
        )

        store.update(created.id, created.model_dump(exclude_none=False))
        updated = store.get_by_id(created.id)

        # Nested structure
        assert updated.processing.status == "error"
        assert updated.processing.step is None
        assert updated.processing.error_code == "TRANSCODE_FAILED"
        assert updated.processing.error_message == "FFmpeg crashed"
