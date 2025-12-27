"""
In-memory match store for session-based match management.

This is a temporary storage solution. Matches are not persisted to disk.
When the application restarts, all matches are lost.
"""

from typing import List, Dict, Optional
from app.models.match import MatchModel


class InMemoryMatchStore:
    """In-memory storage for match data."""
    
    def __init__(self):
        """Initialize empty match store."""
        self._matches: Dict[str, Dict] = {}
    
    def list_all(self) -> List[Dict]:
        """
        Get all matches.
        
        Returns:
            List of all match dictionaries
        """
        return list(self._matches.values())
    
    def get_by_id(self, match_id: str) -> Optional[Dict]:
        """
        Get match by ID.
        
        Args:
            match_id: Match identifier
            
        Returns:
            Match dictionary if found, None otherwise
        """
        return self._matches.get(match_id)
    
    def create(self, match: Dict) -> Dict:
        """
        Create new match.
        
        Args:
            match: Match dictionary to create
            
        Returns:
            Created match dictionary
            
        Raises:
            ValueError: If match with same ID already exists
        """
        # Validate using Pydantic model
        validated_match = MatchModel(**match).model_dump()
        
        match_id = validated_match["id"]
        
        if match_id in self._matches:
            raise ValueError(f"Match with ID '{match_id}' already exists")
        
        self._matches[match_id] = validated_match
        return validated_match
    
    def update(self, match_id: str, match: Dict) -> Dict:
        """
        Update existing match.
        
        Args:
            match_id: ID of match to update
            match: Updated match data
            
        Returns:
            Updated match dictionary
            
        Raises:
            ValueError: If match not found or ID mismatch
        """
        if match_id not in self._matches:
            raise ValueError(f"Match with ID '{match_id}' not found")
        
        # Validate using Pydantic model
        validated_match = MatchModel(**match).model_dump()
        
        # Ensure ID consistency
        if validated_match["id"] != match_id:
            raise ValueError(f"Match ID mismatch: URL has '{match_id}', body has '{validated_match['id']}'")
        
        self._matches[match_id] = validated_match
        return validated_match
    
    def delete(self, match_id: str) -> bool:
        """
        Delete match.
        
        Args:
            match_id: ID of match to delete
            
        Returns:
            True if deleted, False if not found
        """
        if match_id in self._matches:
            del self._matches[match_id]
            return True
        return False
    
    def clear(self):
        """Clear all matches. Useful for testing."""
        self._matches.clear()
