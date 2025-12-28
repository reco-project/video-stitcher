"""
Abstract interface for match storage operations.

This interface defines the contract for match storage implementations.
Storage backends (file-based, database, etc.) must implement all methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class MatchStore(ABC):
    """
    Abstract interface for match storage operations.

    All methods work with plain dicts (JSON-serializable).
    Implementations must handle validation internally.
    """

    @abstractmethod
    def list_all(self) -> List[Dict]:
        """
        Return all matches as list of dicts.

        Returns:
            List of match dictionaries
        """
        pass

    @abstractmethod
    def get_by_id(self, match_id: str) -> Optional[Dict]:
        """
        Return single match by ID, or None if not found.

        Args:
            match_id: Unique match identifier

        Returns:
            Match dict if found, None otherwise
        """
        pass

    @abstractmethod
    def create(self, match: Dict) -> Dict:
        """
        Create new match.

        Args:
            match: Match dictionary to create

        Returns:
            Created match dictionary with validation applied
        """
        pass

    @abstractmethod
    def update(self, match_id: str, match: Dict) -> Dict:
        """
        Update existing match.

        Args:
            match_id: Match ID to update
            match: Updated match data

        Returns:
            Updated match dictionary

        Raises:
            ValueError: If match not found
        """
        pass

    @abstractmethod
    def delete(self, match_id: str) -> bool:
        """
        Delete match by ID.

        Args:
            match_id: Match ID to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def exists(self, match_id: str) -> bool:
        """
        Check if match exists.

        Args:
            match_id: Match ID to check

        Returns:
            True if exists, False otherwise
        """
        pass
