"""
Unit tests for lens profile utilities and models.

Tests the slug generation, Pydantic validation, and data integrity.
"""

import pytest
from pydantic import ValidationError

from app.utils.slug import slugify
from app.models.lens_profile import LensProfileModel


class TestSlugify:
    """Tests for the slugify utility function."""

    def test_basic_lowercase(self):
        """Test basic lowercase conversion."""
        assert slugify("GoPro") == "gopro"
        assert slugify("HERO10") == "hero10"

    def test_spaces_to_hyphens(self):
        """Test space replacement with hyphens."""
        assert slugify("HERO10 Black") == "hero10-black"
        assert slugify("ONE X2") == "one-x2"

    def test_special_characters_removed(self):
        """Test removal of special characters."""
        assert slugify("Action@2!") == "action2"
        assert slugify("Model#123$") == "model123"

    def test_multiple_spaces_hyphens(self):
        """Test multiple consecutive spaces/hyphens normalized."""
        assert slugify("GoPro   HERO10") == "gopro-hero10"
        assert slugify("GoPro--HERO10") == "gopro-hero10"

    def test_strip_hyphens(self):
        """Test leading/trailing hyphens stripped."""
        assert slugify("-GoPro-") == "gopro"
        assert slugify("--HERO10--") == "hero10"

    def test_unicode_removed(self):
        """Test Unicode characters removed (ASCII-only)."""
        assert slugify("GoPro™") == "gopro"
        assert slugify("Héro10") == "hro10"

    def test_empty_string(self):
        """Test empty string handling."""
        assert slugify("") == ""
        assert slugify("   ") == ""
        assert slugify("@#$%") == ""


class TestLensProfileModel:
    """Tests for the LensProfileModel Pydantic validation."""

    def test_valid_profile(self):
        """Test creation of a valid profile."""
        profile_data = {
            "id": "gopro-hero10black-linear-3840x2160",
            "camera_brand": "GoPro",
            "camera_model": "HERO10 Black",
            "lens_model": "Linear",
            "resolution": {"width": 3840, "height": 2160},
            "distortion_model": "fisheye_kb4",
            "camera_matrix": {"fx": 1796.32, "fy": 1797.22, "cx": 1919.37, "cy": 1063.17},
            "distortion_coeffs": [0.0342, 0.0677, -0.0741, 0.0299],
            "metadata": {"source": "Gyroflow", "notes": "Test profile"},
        }
        profile = LensProfileModel(**profile_data)
        assert profile.id == "gopro-hero10black-linear-3840x2160"
        assert profile.camera_brand == "GoPro"
        assert profile.resolution["width"] == 3840
        assert len(profile.distortion_coeffs) == 4

    def test_id_format_validation(self):
        """Test ID format validation (lowercase alphanumeric + hyphens)."""
        valid_ids = ["gopro-hero10", "dji-action2-wide-4k", "insta360-onex2", "a1-b2-c3"]
        for valid_id in valid_ids:
            profile = LensProfileModel(
                id=valid_id,
                camera_brand="Test",
                camera_model="Test",
                resolution={"width": 1920, "height": 1080},
                distortion_model="fisheye_kb4",
                camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
                distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
            )
            assert profile.id == valid_id

    def test_id_invalid_format(self):
        """Test rejection of invalid ID formats."""
        invalid_ids = [
            "GoPro-HERO10",  # Uppercase
            "gopro hero10",  # Space
            "gopro_hero10",  # Underscore
            "gopro/hero10",  # Slash
            "gopro.hero10",  # Dot
            "gopro@hero10",  # Special char
        ]
        for invalid_id in invalid_ids:
            with pytest.raises(ValidationError) as exc_info:
                LensProfileModel(
                    id=invalid_id,
                    camera_brand="Test",
                    camera_model="Test",
                    resolution={"width": 1920, "height": 1080},
                    distortion_model="fisheye_kb4",
                    camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
                    distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
                )
            assert "id" in str(exc_info.value).lower()

    def test_id_too_long(self):
        """Test rejection of IDs exceeding 100 characters."""
        long_id = "a" * 101
        with pytest.raises(ValidationError):
            LensProfileModel(
                id=long_id,
                camera_brand="Test",
                camera_model="Test",
                resolution={"width": 1920, "height": 1080},
                distortion_model="fisheye_kb4",
                camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
                distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
            )

    def test_distortion_model_validation(self):
        """Test only fisheye_kb4 is accepted."""
        # Valid model
        profile = LensProfileModel(
            id="test-profile",
            camera_brand="Test",
            camera_model="Test",
            resolution={"width": 1920, "height": 1080},
            distortion_model="fisheye_kb4",
            camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
            distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
        )
        assert profile.distortion_model == "fisheye_kb4"

        # Invalid model
        with pytest.raises(ValidationError) as exc_info:
            LensProfileModel(
                id="test-profile",
                camera_brand="Test",
                camera_model="Test",
                resolution={"width": 1920, "height": 1080},
                distortion_model="brown_conrady",
                camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
                distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
            )
        assert "distortion_model" in str(exc_info.value).lower()

    def test_distortion_coeffs_count(self):
        """Test exactly 4 distortion coefficients required."""
        # Valid: 4 coefficients
        profile = LensProfileModel(
            id="test-profile",
            camera_brand="Test",
            camera_model="Test",
            resolution={"width": 1920, "height": 1080},
            distortion_model="fisheye_kb4",
            camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
            distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
        )
        assert len(profile.distortion_coeffs) == 4

        # Invalid: 2 coefficients
        with pytest.raises(ValidationError) as exc_info:
            LensProfileModel(
                id="test-profile",
                camera_brand="Test",
                camera_model="Test",
                resolution={"width": 1920, "height": 1080},
                distortion_model="fisheye_kb4",
                camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
                distortion_coeffs=[0.1, 0.01],
            )
        assert "distortion_coeffs" in str(exc_info.value).lower()

        # Invalid: 5 coefficients
        with pytest.raises(ValidationError):
            LensProfileModel(
                id="test-profile",
                camera_brand="Test",
                camera_model="Test",
                resolution={"width": 1920, "height": 1080},
                distortion_model="fisheye_kb4",
                camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
                distortion_coeffs=[0.1, 0.01, -0.05, 0.02, 0.03],
            )

    def test_resolution_positive(self):
        """Test resolution width/height must be positive."""
        # Valid
        profile = LensProfileModel(
            id="test-profile",
            camera_brand="Test",
            camera_model="Test",
            resolution={"width": 1920, "height": 1080},
            distortion_model="fisheye_kb4",
            camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
            distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
        )
        assert profile.resolution["width"] == 1920

        # Invalid: zero width
        with pytest.raises(ValidationError):
            LensProfileModel(
                id="test-profile",
                camera_brand="Test",
                camera_model="Test",
                resolution={"width": 0, "height": 1080},
                distortion_model="fisheye_kb4",
                camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
                distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
            )

        # Invalid: negative height
        with pytest.raises(ValidationError):
            LensProfileModel(
                id="test-profile",
                camera_brand="Test",
                camera_model="Test",
                resolution={"width": 1920, "height": -1080},
                distortion_model="fisheye_kb4",
                camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
                distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
            )

    def test_camera_matrix_positive(self):
        """Test camera matrix values must be positive."""
        # Valid
        profile = LensProfileModel(
            id="test-profile",
            camera_brand="Test",
            camera_model="Test",
            resolution={"width": 1920, "height": 1080},
            distortion_model="fisheye_kb4",
            camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
            distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
        )
        assert profile.camera_matrix["fx"] == 1000

        # Invalid: negative focal length
        with pytest.raises(ValidationError):
            LensProfileModel(
                id="test-profile",
                camera_brand="Test",
                camera_model="Test",
                resolution={"width": 1920, "height": 1080},
                distortion_model="fisheye_kb4",
                camera_matrix={"fx": -1000, "fy": 1000, "cx": 960, "cy": 540},
                distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
            )

    def test_optional_fields(self):
        """Test optional fields (lens_model, metadata)."""
        # Without optional fields
        profile = LensProfileModel(
            id="test-profile",
            camera_brand="Test",
            camera_model="Test",
            resolution={"width": 1920, "height": 1080},
            distortion_model="fisheye_kb4",
            camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
            distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
        )
        assert profile.lens_model is None
        assert profile.metadata is None

        # With optional fields
        profile_with_opts = LensProfileModel(
            id="test-profile",
            camera_brand="Test",
            camera_model="Test",
            lens_model="Wide",
            resolution={"width": 1920, "height": 1080},
            distortion_model="fisheye_kb4",
            camera_matrix={"fx": 1000, "fy": 1000, "cx": 960, "cy": 540},
            distortion_coeffs=[0.1, 0.01, -0.05, 0.02],
            metadata={"source": "Test", "notes": "Testing"},
        )
        assert profile_with_opts.lens_model == "Wide"
        assert profile_with_opts.metadata["source"] == "Test"
