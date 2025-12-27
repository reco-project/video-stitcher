"""FastAPI router for match API endpoints."""

from typing import List, Dict
from fastapi import APIRouter, HTTPException, Depends, status

from app.repositories.match_store import InMemoryMatchStore
from app.models.match import MatchModel


router = APIRouter(prefix="/matches")


# Dependency injection - will be configured in main.py
def get_store() -> InMemoryMatchStore:
    """
    Dependency to get the match store instance.
    This will be overridden in main.py with the actual store.
    """
    raise NotImplementedError("Store dependency not configured")


@router.get("", response_model=List[Dict])
def list_all_matches(store: InMemoryMatchStore = Depends(get_store)):
    """
    List all matches.
    
    Returns:
        List of all match dictionaries
    """
    return store.list_all()


@router.get("/{match_id}", response_model=Dict)
def get_match(match_id: str, store: InMemoryMatchStore = Depends(get_store)):
    """
    Get a specific match by ID.
    
    Args:
        match_id: Unique match identifier
        
    Returns:
        Match dictionary
        
    Raises:
        404: Match not found
    """
    match = store.get_by_id(match_id)
    if match is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match with ID '{match_id}' not found"
        )
    return match


@router.post("", response_model=Dict, status_code=status.HTTP_201_CREATED)
def create_match(match: MatchModel, store: InMemoryMatchStore = Depends(get_store)):
    """
    Create a new match.
    
    Args:
        match: Match data to create
        
    Returns:
        Created match dictionary
        
    Raises:
        400: Validation error
        409: Match with same ID already exists
    """
    try:
        match_dict = match.model_dump()
        created = store.create(match_dict)
        return created
    except ValueError as e:
        error_msg = str(e)
        # Check if it's an ID conflict
        if "already exists" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_msg
            )
        # Otherwise it's a validation error
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )


@router.put("/{match_id}", response_model=Dict)
def update_match(
    match_id: str,
    match: MatchModel,
    store: InMemoryMatchStore = Depends(get_store)
):
    """
    Update an existing match.
    
    Args:
        match_id: ID of match to update
        match: Updated match data
        
    Returns:
        Updated match dictionary
        
    Raises:
        400: Validation error or ID mismatch
        404: Match not found
    """
    try:
        match_dict = match.model_dump()
        updated = store.update(match_id, match_dict)
        return updated
    except ValueError as e:
        error_msg = str(e)
        # Check if it's a not found error
        if "not found" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
            )
        # Otherwise it's a validation or mismatch error
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )


@router.delete("/{match_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_match(match_id: str, store: InMemoryMatchStore = Depends(get_store)):
    """
    Delete a match.
    
    Args:
        match_id: ID of match to delete
        
    Raises:
        404: Match not found
    """
    deleted = store.delete(match_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match with ID '{match_id}' not found"
        )
    return None
