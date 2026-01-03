"""
File-based implementation of match storage.

Stores matches as JSON files in a flat directory structure:
  {base_path}/{match_id}.json

Design notes:
- Each match is a single JSON file identified by match_id
- Simpler structure than profiles (no hierarchical organization needed)
- Persists across app restarts
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from app.models.match import MatchModel
from app.repositories.match_store import MatchStore


class FileMatchStore(MatchStore):
    """File-based match storage implementation."""

    def __init__(self, base_path: str):
        """
        Initialize file store.

        Args:
            base_path: Root directory for match storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _validate_match(self, match_dict: Dict) -> Dict:
        """
        Validate match using Pydantic model.

        Args:
            match_dict: Match dictionary to validate

        Returns:
            Validated match dictionary with defaults applied

        Raises:
            ValueError: If validation fails
        """
        try:
            model = MatchModel(**match_dict)
            return model.model_dump()
        except Exception as e:
            raise ValueError(f"Match validation failed: {str(e)}")

    def _get_file_path(self, match_id: str) -> Path:
        """
        Get filesystem path for a match.

        Args:
            match_id: Match identifier

        Returns:
            Full path to match file
        """
        return self.base_path / f"{match_id}.json"

    def _load_match_file(self, path: Path) -> Dict:
        """
        Load match from JSON file.

        Args:
            path: Path to match JSON file

        Returns:
            Match dictionary

        Raises:
            ValueError: If JSON is malformed
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in {path}: {str(e)}")

    def _save_match_file(self, path: Path, match_dict: Dict) -> None:
        """
        Save match to JSON file.

        Args:
            path: Path to match JSON file
            match_dict: Match dictionary to save
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(match_dict, f, indent=2, ensure_ascii=False)

    def list_all(self) -> List[Dict]:
        """
        Return all matches by scanning filesystem, validated and normalized.

        Returns:
            List of all match dictionaries
        """
        matches = []

        for json_file in self.base_path.glob("*.json"):
            try:
                match = self._load_match_file(json_file)
                # Validate and normalize using Pydantic model
                match = self._validate_match(match)
                matches.append(match)
            except (ValueError, KeyError):
                # Skip malformed files
                continue

        return matches

    def get_by_id(self, match_id: str) -> Optional[Dict]:
        """
        Get match by ID, validated and normalized.

        Args:
            match_id: Unique match identifier

        Returns:
            Match dictionary or None if not found
        """
        path = self._get_file_path(match_id)

        if not path.exists():
            return None

        try:
            match = self._load_match_file(path)
            # Validate and normalize using Pydantic model
            return self._validate_match(match)
        except ValueError:
            return None

    def create(self, match_dict: Dict) -> Dict:
        """
        Create a new match.

        Args:
            match_dict: Match data to create

        Returns:
            Created match dictionary with defaults applied

        Raises:
            ValueError: If match already exists or validation fails
        """
        # Validate and apply defaults
        validated = self._validate_match(match_dict)
        match_id = validated["id"]

        # Check if already exists
        path = self._get_file_path(match_id)
        if path.exists():
            raise ValueError(f"Match with ID '{match_id}' already exists")

        # Save to file
        self._save_match_file(path, validated)

        return validated

    def update(self, match_id: str, match_dict: Dict) -> Dict:
        """
        Update an existing match.

        Args:
            match_id: Match ID to update
            match_dict: New match data

        Returns:
            Updated match dictionary

        Raises:
            ValueError: If match not found or validation fails
        """
        path = self._get_file_path(match_id)

        if not path.exists():
            raise ValueError(f"Match with ID '{match_id}' not found")

        # Ensure ID consistency
        match_dict["id"] = match_id

        # Validate and apply defaults
        validated = self._validate_match(match_dict)

        # Save to file
        self._save_match_file(path, validated)

        return validated

    def delete(self, match_id: str) -> bool:
        """
        Delete a match.

        Args:
            match_id: Match ID to delete

        Returns:
            True if deleted, False if not found
        """
        path = self._get_file_path(match_id)

        if not path.exists():
            return False

        path.unlink()
        return True

    def exists(self, match_id: str) -> bool:
        """
        Check if match exists.

        Args:
            match_id: Match ID to check

        Returns:
            True if exists, False otherwise
        """
        return self._get_file_path(match_id).exists()
