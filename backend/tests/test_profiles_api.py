"""
Comprehensive test suite for the Lens Profile API endpoints.

Tests all CRUD operations, hierarchy endpoints, validation, and error handling.
"""

import sys
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


def test_get_profile_by_id():
    """Test GET /api/profiles/{id} - Get specific profile"""
    response = client.get("/api/profiles/gopro-hero10black-linear-3840x2160")
    assert response.status_code == 200
    profile = response.json()
    assert profile['id'] == 'gopro-hero10black-linear-3840x2160'
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


if __name__ == "__main__":
    """Run all tests when executed directly"""
    print("=== Comprehensive Lens Profile API Test ===\n")

    test_functions = [
        ("List all profiles", test_list_all_profiles),
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
        print("\nAPI Summary:")
        print("  * Read endpoints: 6 endpoints, all working")
        print("  * Write endpoints: 3 endpoints, all working")
        print("  * Error handling: 404, 409, 422 correctly returned")
        print("  * Validation: Pydantic validation working correctly")
