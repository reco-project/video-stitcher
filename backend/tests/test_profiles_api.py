"""
Comprehensive test suite for the Lens Profile API endpoints.

Tests all CRUD operations, hierarchy endpoints, validation, and error handling.

The hybrid store architecture supports:
- Official profiles from SQLite (read-only) or JSON files (fallback)
- User-created profiles stored in user data directory (read-write)
- Favorites stored in a separate JSON file
"""

import sys
import os
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_list_all_profiles():
    """Test GET /api/profiles - List all profiles"""
    response = client.get("/api/profiles")
    assert response.status_code == 200
    profiles = response.json()
    assert len(profiles) >= 1, "Should have at least 1 profile"
    print(f"Found {len(profiles)} profile(s)")


def test_list_profiles_metadata():
    """Test GET /api/profiles/list - List profiles with metadata only"""
    response = client.get("/api/profiles/list?limit=10")
    assert response.status_code == 200
    profiles = response.json()
    assert len(profiles) <= 10, "Should respect limit"
    
    # Check that metadata format is correct
    if profiles:
        p = profiles[0]
        assert "id" in p
        assert "camera_brand" in p
        assert "camera_model" in p
        assert "w" in p
        assert "h" in p
        # Should NOT have full calibration data
        assert "camera_matrix" not in p
        assert "distortion_coeffs" not in p


def test_list_profiles_with_filters():
    """Test GET /api/profiles/list with filters"""
    # Filter by brand
    response = client.get("/api/profiles/list?brand=GoPro")
    assert response.status_code == 200
    profiles = response.json()
    for p in profiles:
        assert "gopro" in p["camera_brand"].lower()


def test_search_profiles_fuzzy():
    """Test GET /api/profiles/list with fuzzy search"""
    # Search for "gopro hero10" should match "GoPro HERO10 Black"
    response = client.get("/api/profiles/list?search=gopro%20hero10")
    assert response.status_code == 200
    profiles = response.json()
    
    # Should have at least one result
    assert len(profiles) > 0
    
    # All results should have both "gopro" and "hero10" somewhere
    for p in profiles:
        searchable = f"{p['camera_brand']} {p['camera_model']} {p.get('lens_model', '')}".lower()
        assert "gopro" in searchable
        assert "hero10" in searchable or "hero 10" in searchable
    print(f"Found {len(profiles)} profiles matching 'gopro hero10'")


def test_search_profiles_empty():
    """Test GET /api/profiles/list with search that matches nothing"""
    response = client.get("/api/profiles/list?search=nonexistentcamera12345")
    assert response.status_code == 200
    profiles = response.json()
    assert len(profiles) == 0


def test_search_profiles_case_insensitive():
    """Test that search is case-insensitive"""
    response1 = client.get("/api/profiles/list?search=GOPRO")
    response2 = client.get("/api/profiles/list?search=gopro")
    response3 = client.get("/api/profiles/list?search=GoPro")
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200
    
    # All should return same number of results
    assert len(response1.json()) == len(response2.json()) == len(response3.json())


def test_count_profiles():
    """Test GET /api/profiles/count"""
    response = client.get("/api/profiles/count")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert data["count"] >= 1


def test_get_profile_by_id():
    """Test GET /api/profiles/{id} - Get specific profile"""
    response = client.get("/api/profiles/gopro-hero10-black-linear-3840x2160")
    assert response.status_code == 200
    profile = response.json()
    assert profile['id'] == 'gopro-hero10-black-linear-3840x2160'
    assert profile['camera_brand'] == 'GoPro'
    assert profile['camera_model'] == 'HERO10 Black'


def test_get_nonexistent_profile():
    """Test GET /api/profiles/{id} - 404 for non-existent profile"""
    response = client.get("/api/profiles/non-existent-id-12345")
    assert response.status_code == 404


def test_list_brands():
    """Test GET /api/profiles/hierarchy/brands - List all brands"""
    response = client.get("/api/profiles/hierarchy/brands")
    assert response.status_code == 200
    brands = response.json()
    assert 'GoPro' in brands
    print(f"Brands: {brands}")


def test_list_models_for_brand():
    """Test GET /api/profiles/hierarchy/brands/{brand}/models"""
    response = client.get("/api/profiles/hierarchy/brands/GoPro/models")
    assert response.status_code == 200
    models = response.json()
    assert 'HERO10 Black' in models
    print(f"GoPro models: {models}")


def test_list_profiles_by_brand_model():
    """Test GET /api/profiles/hierarchy/brands/{brand}/models/{model}"""
    response = client.get("/api/profiles/hierarchy/brands/GoPro/models/HERO10%20Black")
    assert response.status_code == 200
    profiles = response.json()
    assert len(profiles) >= 1
    print(f"Found {len(profiles)} profile(s) for GoPro HERO10 Black")


# Write operation tests - only run with file-based store
import pytest



def test_create_profile():
    """Test POST /api/profiles - Create new profile"""
    new_profile = {
        "id": "test-brand-model-1920x1080",
        "camera_brand": "TestBrand",
        "camera_model": "TestModel",
        "lens_model": "Standard",
        "resolution": {"width": 1920, "height": 1080},
        "distortion_model": "fisheye_kb4",
        "camera_matrix": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0},
        "distortion_coeffs": [0.1, 0.01, -0.05, 0.02],
        "metadata": {"calibrated_by": "Test Suite", "calibration_date": "2025-12-27", "notes": "Test profile"},
    }
    response = client.post("/api/profiles", json=new_profile)
    assert response.status_code == 201
    created = response.json()
    assert created['id'] == new_profile['id']
    print(f"Created profile: {created['id']}")



def test_create_duplicate_profile():
    """Test POST /api/profiles - Reject duplicate (409)"""
    duplicate_profile = {
        "id": "test-brand-model-1920x1080",
        "camera_brand": "TestBrand",
        "camera_model": "TestModel",
        "resolution": {"width": 1920, "height": 1080},
        "distortion_model": "fisheye_kb4",
        "camera_matrix": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0},
        "distortion_coeffs": [0.1, 0.01, -0.05, 0.02],
    }
    response = client.post("/api/profiles", json=duplicate_profile)
    assert response.status_code == 409



def test_update_profile():
    """Test PUT /api/profiles/{id} - Update existing profile"""
    updated_profile = {
        "id": "test-brand-model-1920x1080",
        "camera_brand": "TestBrand",
        "camera_model": "TestModel",
        "lens_model": "Standard",
        "resolution": {"width": 1920, "height": 1080},
        "distortion_model": "fisheye_kb4",
        "camera_matrix": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0},
        "distortion_coeffs": [0.1, 0.01, -0.05, 0.02],
        "metadata": {"calibrated_by": "Test Suite", "calibration_date": "2025-12-27", "notes": "Updated by test suite"},
    }
    response = client.put("/api/profiles/test-brand-model-1920x1080", json=updated_profile)
    assert response.status_code == 200
    updated = response.json()
    assert updated['metadata']['notes'] == "Updated by test suite"
    print("Profile updated successfully")



def test_update_nonexistent_profile():
    """Test PUT /api/profiles/{id} - Fail on non-existent (400/404)"""
    profile = {
        "id": "non-existent-id",
        "camera_brand": "Test",
        "camera_model": "Test",
        "resolution": {"width": 1920, "height": 1080},
        "distortion_model": "fisheye_kb4",
        "camera_matrix": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0},
        "distortion_coeffs": [0.1, 0.01, -0.05, 0.02],
    }
    response = client.put("/api/profiles/non-existent-id", json=profile)
    assert response.status_code in [400, 404]



def test_delete_profile():
    """Test DELETE /api/profiles/{id} - Delete profile"""
    response = client.delete("/api/profiles/test-brand-model-1920x1080")
    assert response.status_code == 204
    print("Profile deleted successfully")


def test_verify_deletion():
    """Test GET /api/profiles/{id} - Verify profile no longer exists"""
    response = client.get("/api/profiles/test-brand-model-1920x1080")
    assert response.status_code == 404



def test_delete_nonexistent_profile():
    """Test DELETE /api/profiles/{id} - 404 on non-existent"""
    response = client.delete("/api/profiles/non-existent-id-xyz")
    assert response.status_code == 404


def test_validation_invalid_distortion_model():
    """Test POST /api/profiles - Reject invalid distortion model (422)"""
    invalid_profile = {
        "id": "invalid-test",
        "camera_brand": "Test",
        "camera_model": "Test",
        "resolution": {"width": 1920, "height": 1080},
        "distortion_model": "invalid_model",
        "camera_matrix": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0},
        "distortion_coeffs": [0.1, 0.01, -0.05, 0.02],
    }
    response = client.post("/api/profiles", json=invalid_profile)
    assert response.status_code == 422


def test_validation_wrong_coefficient_count():
    """Test POST /api/profiles - Reject wrong coefficient count (422)"""
    invalid_profile = {
        "id": "invalid-test-2",
        "camera_brand": "Test",
        "camera_model": "Test",
        "resolution": {"width": 1920, "height": 1080},
        "distortion_model": "fisheye_kb4",
        "camera_matrix": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0},
        "distortion_coeffs": [0.1, 0.2],  # Only 2 instead of 4
    }
    response = client.post("/api/profiles", json=invalid_profile)
    assert response.status_code == 422


def test_toggle_favorite_add():
    """Test PATCH /api/profiles/{id}/favorite - Add to favorites"""
    response = client.patch(
        "/api/profiles/gopro-hero10-black-linear-3840x2160/favorite",
        json={"is_favorite": True}
    )
    assert response.status_code == 200
    profile = response.json()
    assert profile["is_favorite"] is True
    print("Added profile to favorites")


def test_toggle_favorite_remove():
    """Test PATCH /api/profiles/{id}/favorite - Remove from favorites"""
    # First add
    client.patch(
        "/api/profiles/gopro-hero10-black-linear-3840x2160/favorite",
        json={"is_favorite": True}
    )
    
    # Then remove
    response = client.patch(
        "/api/profiles/gopro-hero10-black-linear-3840x2160/favorite",
        json={"is_favorite": False}
    )
    assert response.status_code == 200
    profile = response.json()
    assert profile["is_favorite"] is False
    print("Removed profile from favorites")


def test_list_favorite_ids():
    """Test GET /api/profiles/favorites/ids - List favorite IDs"""
    # Add some favorites first
    client.patch(
        "/api/profiles/gopro-hero10-black-linear-3840x2160/favorite",
        json={"is_favorite": True}
    )
    
    response = client.get("/api/profiles/favorites/ids")
    assert response.status_code == 200
    favorite_ids = response.json()
    assert isinstance(favorite_ids, list)
    assert "gopro-hero10-black-linear-3840x2160" in favorite_ids
    print(f"Favorite IDs: {favorite_ids}")


def test_list_favorite_profiles():
    """Test GET /api/profiles/favorites/list - List favorite profiles"""
    # Add a favorite
    client.patch(
        "/api/profiles/gopro-hero10-black-linear-3840x2160/favorite",
        json={"is_favorite": True}
    )
    
    response = client.get("/api/profiles/favorites/list")
    assert response.status_code == 200
    favorites = response.json()
    assert isinstance(favorites, list)
    assert len(favorites) > 0
    
    # Check that all returned profiles are favorites
    for profile in favorites:
        assert profile.get("is_favorite") is True
    print(f"Found {len(favorites)} favorite profile(s)")


def test_pagination_with_limit():
    """Test GET /api/profiles/list with limit parameter"""
    response = client.get("/api/profiles/list?limit=5")
    assert response.status_code == 200
    profiles = response.json()
    assert len(profiles) <= 5


def test_pagination_with_offset():
    """Test GET /api/profiles/list with offset parameter"""
    # Get first page
    response1 = client.get("/api/profiles/list?limit=5&offset=0")
    assert response1.status_code == 200
    page1 = response1.json()
    
    # Get second page
    response2 = client.get("/api/profiles/list?limit=5&offset=5")
    assert response2.status_code == 200
    page2 = response2.json()
    
    # Pages should be different (if we have more than 5 profiles)
    if len(page1) == 5 and len(page2) > 0:
        page1_ids = [p["id"] for p in page1]
        page2_ids = [p["id"] for p in page2]
        assert page1_ids != page2_ids
        print(f"Page 1: {len(page1)} profiles, Page 2: {len(page2)} profiles")


if __name__ == "__main__":
    """Run all tests when executed directly"""
    print("=== Comprehensive Lens Profile API Test ===\n")
    print("Using hybrid store (SQLite/JSON for official + user profiles)\n")

    test_functions = [
        ("List all profiles", test_list_all_profiles),
        ("List profiles metadata", test_list_profiles_metadata),
        ("List profiles with filters", test_list_profiles_with_filters),
        ("Search profiles fuzzy", test_search_profiles_fuzzy),
        ("Search profiles empty", test_search_profiles_empty),
        ("Search case-insensitive", test_search_profiles_case_insensitive),
        ("Count profiles", test_count_profiles),
        ("Get profile by ID", test_get_profile_by_id),
        ("Get non-existent profile (404)", test_get_nonexistent_profile),
        ("List brands", test_list_brands),
        ("List models for brand", test_list_models_for_brand),
        ("List profiles by brand/model", test_list_profiles_by_brand_model),
        ("Create profile", test_create_profile),
        ("Create duplicate (409)", test_create_duplicate_profile),
        ("Update profile", test_update_profile),
        ("Update non-existent (400/404)", test_update_nonexistent_profile),
        ("Delete profile", test_delete_profile),
        ("Verify deletion", test_verify_deletion),
        ("Delete non-existent (404)", test_delete_nonexistent_profile),
        ("Validation - invalid model", test_validation_invalid_distortion_model),
        ("Validation - wrong coeff count", test_validation_wrong_coefficient_count),
        ("Toggle favorite - add", test_toggle_favorite_add),
        ("Toggle favorite - remove", test_toggle_favorite_remove),
        ("List favorite IDs", test_list_favorite_ids),
        ("List favorite profiles", test_list_favorite_profiles),
        ("Pagination with limit", test_pagination_with_limit),
        ("Pagination with offset", test_pagination_with_offset),
    ]

    passed = 0
    failed = 0

    for i, (name, test_func) in enumerate(test_functions, 1):
        try:
            print(f"{i}. {name}")
            test_func()
            print("   OK\n")
            passed += 1
        except AssertionError as e:
            print(f"   FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"   ERROR: {e}\n")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\nALL TESTS PASSED!")
