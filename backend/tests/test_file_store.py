"""
Unit tests for FileLensProfileStore repository implementation.

Tests CRUD operations, file I/O, validation, and hierarchy queries.
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from app.repositories.file_lens_profile_store import FileLensProfileStore


@pytest.fixture
def temp_profiles_dir():
    """Create a temporary directory for test profiles."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def store(temp_profiles_dir):
    """Create a FileLensProfileStore with temporary directory."""
    return FileLensProfileStore(base_path=temp_profiles_dir)


@pytest.fixture
def sample_profile():
    """Sample valid profile data."""
    return {
        "id": "test-camera-model-1920x1080",
        "camera_brand": "TestBrand",
        "camera_model": "TestModel",
        "lens_model": "Wide",
        "resolution": {"width": 1920, "height": 1080},
        "distortion_model": "fisheye_kb4",
        "camera_matrix": {
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 960.0,
            "cy": 540.0
        },
        "distortion_coeffs": [0.1, 0.01, -0.05, 0.02],
        "metadata": {
            "source": "Test",
            "notes": "Test profile"
        }
    }


class TestFileStore:
    """Tests for FileLensProfileStore."""
    
    def test_create_profile(self, store, sample_profile):
        """Test creating a new profile."""
        created = store.create(sample_profile)
        
        assert created["id"] == sample_profile["id"]
        assert created["camera_brand"] == sample_profile["camera_brand"]
        assert created["camera_model"] == sample_profile["camera_model"]
    
    def test_create_duplicate_raises_error(self, store, sample_profile):
        """Test creating duplicate profile raises ValueError."""
        store.create(sample_profile)
        
        with pytest.raises(ValueError) as exc_info:
            store.create(sample_profile)
        
        assert "already exists" in str(exc_info.value)
    
    def test_get_by_id(self, store, sample_profile):
        """Test retrieving profile by ID."""
        store.create(sample_profile)
        
        retrieved = store.get_by_id(sample_profile["id"])
        
        assert retrieved is not None
        assert retrieved["id"] == sample_profile["id"]
        assert retrieved["camera_brand"] == sample_profile["camera_brand"]
    
    def test_get_by_id_nonexistent(self, store):
        """Test retrieving non-existent profile returns None."""
        result = store.get_by_id("non-existent-id")
        assert result is None
    
    def test_update_profile(self, store, sample_profile):
        """Test updating an existing profile."""
        store.create(sample_profile)
        
        # Update the profile
        updated_data = sample_profile.copy()
        updated_data["metadata"]["notes"] = "Updated notes"
        updated_data["lens_model"] = "UltraWide"
        
        updated = store.update(sample_profile["id"], updated_data)
        
        assert updated["metadata"]["notes"] == "Updated notes"
        assert updated["lens_model"] == "UltraWide"
        
        # Verify persistence
        retrieved = store.get_by_id(sample_profile["id"])
        assert retrieved["metadata"]["notes"] == "Updated notes"
    
    def test_update_nonexistent_raises_error(self, store, sample_profile):
        """Test updating non-existent profile raises ValueError."""
        # Use matching ID to avoid ID mismatch error
        nonexistent_profile = sample_profile.copy()
        nonexistent_profile["id"] = "non-existent-id"
        
        with pytest.raises(ValueError) as exc_info:
            store.update("non-existent-id", nonexistent_profile)
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_update_id_mismatch_raises_error(self, store, sample_profile):
        """Test updating with ID mismatch raises ValueError."""
        store.create(sample_profile)
        
        mismatched_data = sample_profile.copy()
        mismatched_data["id"] = "different-id"
        
        with pytest.raises(ValueError) as exc_info:
            store.update(sample_profile["id"], mismatched_data)
        
        assert "mismatch" in str(exc_info.value).lower()
    
    def test_delete_profile(self, store, sample_profile):
        """Test deleting a profile."""
        store.create(sample_profile)
        
        result = store.delete(sample_profile["id"])
        assert result is True
        
        # Verify deletion
        retrieved = store.get_by_id(sample_profile["id"])
        assert retrieved is None
    
    def test_delete_nonexistent(self, store):
        """Test deleting non-existent profile returns False."""
        result = store.delete("non-existent-id")
        assert result is False
    
    def test_list_all_profiles(self, store, sample_profile):
        """Test listing all profiles."""
        # Create multiple profiles
        profile1 = sample_profile.copy()
        profile1["id"] = "test1-camera-1920x1080"
        
        profile2 = sample_profile.copy()
        profile2["id"] = "test2-camera-1920x1080"
        profile2["camera_brand"] = "AnotherBrand"
        
        store.create(profile1)
        store.create(profile2)
        
        all_profiles = store.list_all()
        
        assert len(all_profiles) == 2
        ids = [p["id"] for p in all_profiles]
        assert "test1-camera-1920x1080" in ids
        assert "test2-camera-1920x1080" in ids
    
    def test_list_brands(self, store, sample_profile):
        """Test listing unique brands."""
        # Create profiles with different brands
        profile1 = sample_profile.copy()
        profile1["id"] = "brand1-model-1920x1080"
        profile1["camera_brand"] = "BrandA"
        
        profile2 = sample_profile.copy()
        profile2["id"] = "brand2-model-1920x1080"
        profile2["camera_brand"] = "BrandB"
        
        profile3 = sample_profile.copy()
        profile3["id"] = "brand3-model-1920x1080"
        profile3["camera_brand"] = "BrandA"  # Duplicate brand
        
        store.create(profile1)
        store.create(profile2)
        store.create(profile3)
        
        brands = store.list_brands()
        
        assert len(brands) == 2
        assert "BrandA" in brands
        assert "BrandB" in brands
        assert brands == sorted(brands)  # Should be sorted
    
    def test_list_models(self, store, sample_profile):
        """Test listing models for a brand."""
        # Create profiles with same brand, different models
        profile1 = sample_profile.copy()
        profile1["id"] = "gopro-hero9-1920x1080"
        profile1["camera_brand"] = "GoPro"
        profile1["camera_model"] = "HERO9 Black"
        
        profile2 = sample_profile.copy()
        profile2["id"] = "gopro-hero10-1920x1080"
        profile2["camera_brand"] = "GoPro"
        profile2["camera_model"] = "HERO10 Black"
        
        profile3 = sample_profile.copy()
        profile3["id"] = "dji-action2-1920x1080"
        profile3["camera_brand"] = "DJI"
        profile3["camera_model"] = "Action 2"
        
        store.create(profile1)
        store.create(profile2)
        store.create(profile3)
        
        gopro_models = store.list_models("GoPro")
        
        assert len(gopro_models) == 2
        assert "HERO9 Black" in gopro_models
        assert "HERO10 Black" in gopro_models
        assert gopro_models == sorted(gopro_models)  # Should be sorted
        
        dji_models = store.list_models("DJI")
        assert len(dji_models) == 1
        assert "Action 2" in dji_models
    
    def test_list_by_brand_model(self, store, sample_profile):
        """Test listing profiles by brand and model."""
        # Create multiple profiles for same brand/model
        profile1 = sample_profile.copy()
        profile1["id"] = "gopro-hero10-linear-3840x2160"
        profile1["camera_brand"] = "GoPro"
        profile1["camera_model"] = "HERO10 Black"
        profile1["lens_model"] = "Linear"
        profile1["resolution"] = {"width": 3840, "height": 2160}
        
        profile2 = sample_profile.copy()
        profile2["id"] = "gopro-hero10-wide-2704x1520"
        profile2["camera_brand"] = "GoPro"
        profile2["camera_model"] = "HERO10 Black"
        profile2["lens_model"] = "Wide"
        profile2["resolution"] = {"width": 2704, "height": 1520}
        
        profile3 = sample_profile.copy()
        profile3["id"] = "gopro-hero9-wide-1920x1080"
        profile3["camera_brand"] = "GoPro"
        profile3["camera_model"] = "HERO9 Black"
        
        store.create(profile1)
        store.create(profile2)
        store.create(profile3)
        
        hero10_profiles = store.list_by_brand_model("GoPro", "HERO10 Black")
        
        assert len(hero10_profiles) == 2
        ids = [p["id"] for p in hero10_profiles]
        assert "gopro-hero10-linear-3840x2160" in ids
        assert "gopro-hero10-wide-2704x1520" in ids
        
        hero9_profiles = store.list_by_brand_model("GoPro", "HERO9 Black")
        assert len(hero9_profiles) == 1
    
    def test_file_structure(self, store, sample_profile, temp_profiles_dir):
        """Test correct file structure is created."""
        store.create(sample_profile)
        
        # Check directory structure: brand_slug/model_slug/profile_id.json
        expected_path = temp_profiles_dir / "testbrand" / "testmodel" / "test-camera-model-1920x1080.json"
        
        assert expected_path.exists()
        assert expected_path.is_file()
    
    def test_invalid_profile_raises_error(self, store):
        """Test creating profile with invalid data raises ValueError."""
        invalid_profile = {
            "id": "test-invalid",
            "camera_brand": "Test",
            "camera_model": "Test",
            # Missing required fields
        }
        
        with pytest.raises(ValueError):
            store.create(invalid_profile)
