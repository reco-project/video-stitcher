"""
Tests for the SQLite lens profile store and build script.

Includes tests for:
- SQLite database generation from JSON files
- Query functions (list, get by id, filters)
- FTS search functionality
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


# Sample lens profile data for testing
SAMPLE_PROFILES = [
    {
        "id": "gopro-hero10-black-linear-3840x2160",
        "camera_brand": "GoPro",
        "camera_model": "HERO10 Black",
        "lens_model": "Linear",
        "resolution": {"width": 3840, "height": 2160},
        "distortion_model": "fisheye_kb4",
        "camera_matrix": {"fx": 1796.32, "fy": 1797.22, "cx": 1919.37, "cy": 1063.17},
        "distortion_coeffs": [0.0342, 0.0677, -0.0741, 0.0299],
        "metadata": {
            "official": True,
            "source": "GoPro Labs",
            "source_file": "HERO10_Black_Linear_4K.json",
            "notes": "Official GoPro calibration for 4K Linear mode",
        },
    },
    {
        "id": "dji-mavic-3-wide-5120x2700",
        "camera_brand": "DJI",
        "camera_model": "Mavic 3",
        "lens_model": "Wide",
        "resolution": {"width": 5120, "height": 2700},
        "distortion_model": "fisheye_kb4",
        "camera_matrix": {"fx": 2500.0, "fy": 2500.0, "cx": 2560.0, "cy": 1350.0},
        "distortion_coeffs": [0.1, 0.05, -0.03, 0.01],
        "metadata": {
            "official": False,
            "source": "Community",
            "notes": "Community calibration for Mavic 3",
        },
    },
    {
        "id": "insta360-one-r-4k-wide-3840x2160",
        "camera_brand": "Insta360",
        "camera_model": "ONE R",
        "lens_model": "4K Wide",
        "resolution": {"width": 3840, "height": 2160},
        "distortion_model": "fisheye_kb4",
        "camera_matrix": {"fx": 1800.0, "fy": 1800.0, "cx": 1920.0, "cy": 1080.0},
        "distortion_coeffs": [0.05, 0.02, -0.01, 0.005],
        "metadata": {
            "official": False,
            "source": "Gyroflow",
            "notes": "Insta360 ONE R 4K calibration",
        },
    },
]


@pytest.fixture
def temp_profiles_dir(tmp_path):
    """Create a temporary directory with sample JSON profile files."""
    profiles_dir = tmp_path / "lens_profiles"
    profiles_dir.mkdir()

    for profile in SAMPLE_PROFILES:
        # Create brand/model subdirectories
        brand_dir = profiles_dir / profile["camera_brand"].lower()
        model_dir = brand_dir / profile["camera_model"].lower().replace(" ", "-")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON file
        json_file = model_dir / f"{profile['id']}.json"
        json_file.write_text(json.dumps(profile, indent=2))

    return profiles_dir


@pytest.fixture
def temp_sqlite_db(temp_profiles_dir, tmp_path):
    """Build a SQLite database from the temp profiles directory."""
    output_path = tmp_path / "test_profiles.sqlite"

    # Run the build script
    script_path = Path(__file__).parent.parent.parent / "scripts" / "build-lens-profiles-sqlite.cjs"

    result = subprocess.run(
        ["node", str(script_path), "--src", str(temp_profiles_dir), "--output", str(output_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip(f"Build script failed (may need sqlite3 CLI): {result.stderr}")

    return output_path


class TestBuildScript:
    """Tests for the SQLite build script."""

    def test_build_creates_database(self, temp_sqlite_db):
        """Test that the build script creates a valid SQLite database."""
        assert temp_sqlite_db.exists()
        assert temp_sqlite_db.stat().st_size > 0

    def test_build_correct_row_count(self, temp_sqlite_db):
        """Test that the database contains the expected number of profiles."""
        conn = sqlite3.connect(str(temp_sqlite_db))
        cursor = conn.execute("SELECT COUNT(*) FROM profiles")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == len(SAMPLE_PROFILES)

    def test_build_correct_schema(self, temp_sqlite_db):
        """Test that the database has the correct schema."""
        conn = sqlite3.connect(str(temp_sqlite_db))
        cursor = conn.execute("PRAGMA table_info(profiles)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        # Check required columns
        assert "id" in columns
        assert "camera_brand" in columns
        assert "camera_model" in columns
        assert "lens_model" in columns
        assert "w" in columns
        assert "h" in columns
        assert "distortion_model" in columns
        assert "official" in columns
        assert "source" in columns
        assert "source_file" in columns
        assert "notes" in columns
        assert "json" in columns

    def test_build_creates_indexes(self, temp_sqlite_db):
        """Test that the database has the expected indexes."""
        conn = sqlite3.connect(str(temp_sqlite_db))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "idx_brand_model_lens" in indexes
        assert "idx_resolution" in indexes
        assert "idx_official" in indexes

    def test_build_creates_fts_table(self, temp_sqlite_db):
        """Test that the FTS5 virtual table is created."""
        conn = sqlite3.connect(str(temp_sqlite_db))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='profiles_fts'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "profiles_fts" in tables


class TestSqliteLensProfileStore:
    """Tests for the SqliteLensProfileStore class."""

    def test_list_all(self, temp_sqlite_db):
        """Test listing all profiles."""
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(temp_sqlite_db))
        profiles = store.list_all()

        assert len(profiles) == len(SAMPLE_PROFILES)
        assert all("id" in p for p in profiles)
        assert all("camera_brand" in p for p in profiles)

    def test_get_by_id(self, temp_sqlite_db):
        """Test getting a profile by ID."""
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(temp_sqlite_db))

        # Get existing profile
        profile = store.get_by_id("gopro-hero10-black-linear-3840x2160")
        assert profile is not None
        assert profile["id"] == "gopro-hero10-black-linear-3840x2160"
        assert profile["camera_brand"] == "GoPro"
        assert profile["camera_model"] == "HERO10 Black"
        assert profile["lens_model"] == "Linear"
        assert profile["resolution"]["width"] == 3840
        assert profile["resolution"]["height"] == 2160

        # Get non-existent profile
        profile = store.get_by_id("non-existent-id")
        assert profile is None

    def test_list_all_metadata(self, temp_sqlite_db):
        """Test listing profile metadata (lightweight)."""
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(temp_sqlite_db))
        profiles = store.list_all_metadata()

        assert len(profiles) == len(SAMPLE_PROFILES)

        # Check that metadata format is correct (no full calibration data)
        for p in profiles:
            assert "id" in p
            assert "camera_brand" in p
            assert "w" in p
            assert "h" in p
            assert "camera_matrix" not in p  # Full calibration data should not be present
            assert "distortion_coeffs" not in p

    def test_list_all_metadata_with_filters(self, temp_sqlite_db):
        """Test filtering profiles."""
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(temp_sqlite_db))

        # Filter by brand
        profiles = store.list_all_metadata(filters={"brand": "GoPro"})
        assert len(profiles) == 1
        assert profiles[0]["camera_brand"] == "GoPro"

        # Filter by resolution
        profiles = store.list_all_metadata(filters={"w": 3840, "h": 2160})
        assert len(profiles) == 2

        # Filter by official status
        profiles = store.list_all_metadata(filters={"official": True})
        assert len(profiles) == 1
        assert profiles[0]["official"] is True

    def test_list_all_metadata_pagination(self, temp_sqlite_db):
        """Test pagination."""
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(temp_sqlite_db))

        # Get first 2
        profiles = store.list_all_metadata(limit=2)
        assert len(profiles) == 2

        # Get with offset
        profiles = store.list_all_metadata(limit=1, offset=1)
        assert len(profiles) == 1

    def test_list_brands(self, temp_sqlite_db):
        """Test listing unique brands."""
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(temp_sqlite_db))
        brands = store.list_brands()

        assert len(brands) == 3
        assert "GoPro" in brands
        assert "DJI" in brands
        assert "Insta360" in brands

    def test_list_models(self, temp_sqlite_db):
        """Test listing models for a brand."""
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(temp_sqlite_db))

        models = store.list_models("GoPro")
        assert len(models) == 1
        assert "HERO10 Black" in models

    def test_list_by_brand_model(self, temp_sqlite_db):
        """Test listing profiles by brand and model."""
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(temp_sqlite_db))

        profiles = store.list_by_brand_model("GoPro", "HERO10 Black")
        assert len(profiles) == 1
        assert profiles[0]["id"] == "gopro-hero10-black-linear-3840x2160"

    def test_fts_search(self, temp_sqlite_db):
        """Test full-text search."""
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(temp_sqlite_db))

        # Search for GoPro
        results = store.search_fts("gopro")
        assert len(results) >= 1
        assert any(r["camera_brand"] == "GoPro" for r in results)

        # Search for Mavic
        results = store.search_fts("mavic")
        assert len(results) >= 1
        assert any(r["camera_model"] == "Mavic 3" for r in results)

    def test_count(self, temp_sqlite_db):
        """Test counting profiles."""
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(temp_sqlite_db))

        # Total count
        count = store.count()
        assert count == len(SAMPLE_PROFILES)

        # Count with filter
        count = store.count(filters={"brand": "GoPro"})
        assert count == 1

    def test_read_only_mode(self, temp_sqlite_db):
        """Test that write operations fail in read-only mode."""
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(temp_sqlite_db), read_only=True)

        with pytest.raises(RuntimeError, match="read-only"):
            store.create(SAMPLE_PROFILES[0])

        with pytest.raises(RuntimeError, match="read-only"):
            store.update("gopro-hero10-black-linear-3840x2160", SAMPLE_PROFILES[0])

        with pytest.raises(RuntimeError, match="read-only"):
            store.delete("gopro-hero10-black-linear-3840x2160")


class TestSqliteLensProfileStoreIntegration:
    """Integration tests that use the actual lens profiles database."""

    @pytest.fixture
    def real_db_path(self):
        """Get path to the real lens profiles database if it exists."""
        db_path = Path(__file__).parent.parent.parent / "electron" / "resources" / "lens_profiles.sqlite"
        if not db_path.exists():
            pytest.skip("Real lens profiles database not found (run build script first)")
        return db_path

    def test_real_database_loads(self, real_db_path):
        """Test that the real database can be loaded."""
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(real_db_path))
        count = store.count()
        assert count > 0
        print(f"Real database has {count} profiles")

    def test_real_database_query_performance(self, real_db_path):
        """Test that queries are fast on the real database."""
        import time
        from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore

        store = SqliteLensProfileStore(str(real_db_path))

        # Time metadata listing
        start = time.time()
        profiles = store.list_all_metadata(limit=100)
        elapsed = time.time() - start
        print(f"list_all_metadata (100 results): {elapsed:.3f}s")
        assert elapsed < 1.0  # Should be very fast

        # Time get_by_id
        if profiles:
            start = time.time()
            profile = store.get_by_id(profiles[0]["id"])
            elapsed = time.time() - start
            print(f"get_by_id: {elapsed:.3f}s")
            assert elapsed < 0.1

        # Time FTS search
        start = time.time()
        results = store.search_fts("gopro")
        elapsed = time.time() - start
        print(f"FTS search 'gopro': {elapsed:.3f}s, {len(results)} results")
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
