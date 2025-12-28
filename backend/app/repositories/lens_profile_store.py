"""
Abstract interface for lens profile storage operations.

This interface defines the contract for profile storage implementations.
Storage backends (file-based, database, etc.) must implement all methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class LensProfileStore(ABC):
    """
    Abstract interface for lens profile storage operations.

    All methods work with plain dicts (JSON-serializable).
    Implementations must handle validation internally.
    """

    @abstractmethod
    def list_all(self) -> List[Dict]:
        """
        Return all profiles as list of dicts.

        Returns:
            List of profile dictionaries
        """
        pass

    @abstractmethod
    def get_by_id(self, profile_id: str) -> Optional[Dict]:
        """
        Return single profile by ID, or None if not found.

        Args:
            profile_id: Unique profile identifier

        Returns:
            Profile dict if found, None otherwise
        """
        pass

    @abstractmethod
    def create(self, profile: Dict) -> Dict:
        """
        Create new profile.

        Args:
            profile: Profile dictionary to create

        Returns:
            Created profile dictionary

        Raises:
            ValueError: If ID exists or validation fails
        """
        pass

    @abstractmethod
    def update(self, profile_id: str, profile: Dict) -> Dict:
        """
        Update existing profile.

        Args:
            profile_id: ID of profile to update
            profile: Updated profile dictionary

        Returns:
            Updated profile dictionary

        Raises:
            ValueError: If not found or validation fails
        """
        pass

    @abstractmethod
    def delete(self, profile_id: str) -> bool:
        """
        Delete profile by ID.

        Args:
            profile_id: ID of profile to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def list_brands(self) -> List[str]:
        """
        Return list of unique camera brands.

        Returns:
            Sorted list of brand names (original casing from profiles)
        """
        pass

    @abstractmethod
    def list_models(self, brand: str) -> List[str]:
        """
        Return list of models for given brand.

        Args:
            brand: Camera brand name

        Returns:
            Sorted list of model names for the brand
        """
        pass

    @abstractmethod
    def list_by_brand_model(self, brand: str, model: str) -> List[Dict]:
        """
        Return all profiles for given brand and model.

        Args:
            brand: Camera brand name
            model: Camera model name

        Returns:
            List of profile dictionaries matching brand and model
        """
        pass
