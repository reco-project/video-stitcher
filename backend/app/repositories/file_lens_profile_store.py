"""
File-based implementation of LensProfileStore.

Stores profiles as JSON files in a hierarchical directory structure:
  {base_path}/{brand_slug}/{model_slug}/{profile_id}.json

Design notes:
- Filesystem is an index, not source of truth (JSON content is authoritative)
- get_by_id() derives path directly (no scanning for performance)
- Scanning is only used for discovery operations (list_all, list_brands, etc.)
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from app.repositories.lens_profile_store import LensProfileStore
from app.models.lens_profile import LensProfileModel
from app.utils.slug import slugify


class FileLensProfileStore(LensProfileStore):
    """File-based lens profile storage implementation."""

    def __init__(self, base_path: str):
        """
        Initialize file store.

        Args:
            base_path: Root directory for profile storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _validate_profile(self, profile_dict: Dict) -> Dict:
        """
        Validate profile using Pydantic model.

        Args:
            profile_dict: Profile dictionary to validate

        Returns:
            Validated profile dictionary

        Raises:
            ValueError: If validation fails
        """
        try:
            model = LensProfileModel(**profile_dict)
            return model.model_dump()
        except Exception as e:
            raise ValueError(f"Profile validation failed: {str(e)}")

    def _derive_file_path(self, profile_dict: Dict) -> Path:
        """
        Derive filesystem path from profile data.

        Path format: {base}/{brand_slug}/{model_slug}/{profile_id}.json

        Args:
            profile_dict: Profile dictionary containing brand, model, and id

        Returns:
            Full path to profile file
        """
        brand_slug = slugify(profile_dict["camera_brand"])
        model_slug = slugify(profile_dict["camera_model"])
        profile_id = profile_dict["id"]

        return self.base_path / brand_slug / model_slug / f"{profile_id}.json"

    def _load_profile_file(self, path: Path) -> Dict:
        """
        Load profile from JSON file.

        Args:
            path: Path to profile JSON file

        Returns:
            Profile dictionary

        Raises:
            ValueError: If JSON is malformed
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in {path}: {str(e)}")

    def list_all(self) -> List[Dict]:
        """Return all profiles by scanning filesystem."""
        profiles = []

        # Walk entire directory tree looking for .json files
        for json_file in self.base_path.rglob("*.json"):
            try:
                profile = self._load_profile_file(json_file)
                profiles.append(profile)
            except (ValueError, KeyError):
                # Skip malformed or invalid files
                continue

        return profiles

    def get_by_id(self, profile_id: str) -> Optional[Dict]:
        """
        Get profile by ID using direct path derivation.

        Note: This requires scanning to find the brand/model,
        but we optimize by caching the path structure.
        For now, we scan to find the file.
        """
        # Scan for file matching the profile ID
        for json_file in self.base_path.rglob(f"{profile_id}.json"):
            try:
                profile = self._load_profile_file(json_file)
                if profile.get("id") == profile_id:
                    return profile
            except (ValueError, KeyError):
                continue

        return None

    def create(self, profile: Dict) -> Dict:
        """
        Create new profile.

        Validates profile, checks for ID conflict, writes to filesystem.
        """
        # Validate first
        validated_profile = self._validate_profile(profile)

        # Derive file path
        file_path = self._derive_file_path(validated_profile)

        # Check for ID conflict
        if file_path.exists():
            raise ValueError(f"Profile with ID '{validated_profile['id']}' already exists")

        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON with pretty formatting
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(validated_profile, f, indent=2, ensure_ascii=False)

        return validated_profile

    def update(self, profile_id: str, profile: Dict) -> Dict:
        """
        Update existing profile.

        Validates profile, ensures it exists, overwrites file.
        """
        # Validate first
        validated_profile = self._validate_profile(profile)

        # Ensure ID matches
        if validated_profile["id"] != profile_id:
            raise ValueError(f"Profile ID mismatch: URL has '{profile_id}', " f"body has '{validated_profile['id']}'")

        # Find existing file
        existing_path = None
        for json_file in self.base_path.rglob(f"{profile_id}.json"):
            try:
                existing = self._load_profile_file(json_file)
                if existing.get("id") == profile_id:
                    existing_path = json_file
                    break
            except (ValueError, KeyError):
                continue

        if not existing_path:
            raise ValueError(f"Profile with ID '{profile_id}' not found")

        # Derive new path (in case brand/model changed)
        new_path = self._derive_file_path(validated_profile)

        # If path changed, remove old file
        if existing_path != new_path:
            existing_path.unlink()
            # Cleanup empty parent directories
            try:
                existing_path.parent.rmdir()
                existing_path.parent.parent.rmdir()
            except OSError:
                pass  # Directory not empty, that's fine

        # Write to new location
        new_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_path, "w", encoding="utf-8") as f:
            json.dump(validated_profile, f, indent=2, ensure_ascii=False)

        return validated_profile

    def delete(self, profile_id: str) -> bool:
        """
        Delete profile by ID.

        Removes file and cleans up empty parent directories.
        """
        # Find file
        for json_file in self.base_path.rglob(f"{profile_id}.json"):
            try:
                profile = self._load_profile_file(json_file)
                if profile.get("id") == profile_id:
                    # Delete file
                    json_file.unlink()

                    # Cleanup empty parent directories
                    try:
                        json_file.parent.rmdir()  # model dir
                        json_file.parent.parent.rmdir()  # brand dir
                    except OSError:
                        pass  # Directory not empty, that's fine

                    return True
            except (ValueError, KeyError):
                continue

        return False

    def list_brands(self) -> List[str]:
        """
        Return unique camera brands from all profiles.

        Returns brands in original casing from profiles (not slugs).
        """
        brands = set()

        for profile in self.list_all():
            if "camera_brand" in profile:
                brands.add(profile["camera_brand"])

        return sorted(list(brands))

    def list_models(self, brand: str) -> List[str]:
        """
        Return models for given brand.

        Matches brand case-insensitively, returns models in original casing.
        """
        models = set()
        brand_lower = brand.lower()

        for profile in self.list_all():
            profile_brand = profile.get("camera_brand", "")
            if profile_brand.lower() == brand_lower:
                models.add(profile["camera_model"])

        return sorted(list(models))

    def list_by_brand_model(self, brand: str, model: str) -> List[Dict]:
        """
        Return all profiles matching brand and model.

        Matches case-insensitively.
        """
        brand_lower = brand.lower()
        model_lower = model.lower()

        matching = []
        for profile in self.list_all():
            profile_brand = profile.get("camera_brand", "").lower()
            profile_model = profile.get("camera_model", "").lower()

            if profile_brand == brand_lower and profile_model == model_lower:
                matching.append(profile)

        return matching
