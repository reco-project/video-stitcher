"""
File-based implementation of match storage.

Stores matches as JSON files in a flat directory structure:
  {base_path}/{match_id}.json

Design notes:
- Each match is a single JSON file identified by match_id
- Simpler structure than profiles (no hierarchical organization needed)
- Persists across app restarts
- Uses atomic writes to prevent file corruption
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

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

    def _get_file_path(self, match_id: str) -> Path:
        """
        Get filesystem path for a match.

        Args:
            match_id: Match identifier

        Returns:
            Full path to match file
        """
        return self.base_path / f"{match_id}.json"

    def _load_match_file(self, path: Path) -> MatchModel:
        """
        Load match from JSON file.

        Args:
            path: Path to match JSON file

        Returns:
            Match model instance

        Raises:
            ValueError: If JSON is malformed
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return MatchModel(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in {path}: {str(e)}")

    def _save_match_file(self, path: Path, match: MatchModel) -> None:
        """
        Save match to JSON file atomically.

        Uses write-to-temp-then-rename pattern to prevent corruption
        from concurrent writes or crashes during write.

        Args:
            path: Path to match JSON file
            match: Match model to save
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file in the same directory (same filesystem for atomic rename)
        fd, tmp_path = tempfile.mkstemp(suffix='.tmp', prefix=f'{path.stem}_', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(match.model_dump(exclude_none=False), f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk

            # Atomic rename (on POSIX systems)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def list_all(self) -> List[MatchModel]:
        """
        Return all matches by scanning filesystem.

        Returns:
            List of all match model instances
        """
        matches = []

        for json_file in self.base_path.glob("*.json"):
            try:
                match = self._load_match_file(json_file)
                matches.append(match)
            except (ValueError, KeyError):
                # Skip malformed files
                continue

        return matches

    def get_by_id(self, match_id: str) -> Optional[MatchModel]:
        """
        Get match by ID.

        Args:
            match_id: Unique match identifier

        Returns:
            Match model or None if not found
        """
        path = self._get_file_path(match_id)

        if not path.exists():
            return None

        try:
            return self._load_match_file(path)
        except ValueError:
            return None

    def create(self, match_dict: dict) -> MatchModel:
        """
        Create a new match.

        Args:
            match_dict: Match data to create

        Returns:
            Created match model with defaults applied

        Raises:
            ValueError: If match already exists or validation fails
        """
        # Validate and create model
        match = MatchModel(**match_dict)
        match_id = match.id

        # Check if already exists
        path = self._get_file_path(match_id)
        if path.exists():
            raise ValueError(f"Match with ID '{match_id}' already exists")

        # Save to file
        self._save_match_file(path, match)

        return match

    def update(self, match_id: str, match_dict: dict) -> MatchModel:
        """
        Update an existing match.

        Args:
            match_id: Match ID to update
            match_dict: New match data

        Returns:
            Updated match model

        Raises:
            ValueError: If match not found or validation fails
        """
        path = self._get_file_path(match_id)

        if not path.exists():
            raise ValueError(f"Match with ID '{match_id}' not found")

        # Ensure ID consistency
        match_dict["id"] = match_id

        # Validate and create model
        match = MatchModel(**match_dict)

        # Save to file
        self._save_match_file(path, match)

        return match

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
