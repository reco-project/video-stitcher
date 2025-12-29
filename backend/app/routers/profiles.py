"""FastAPI router for lens profile API endpoints."""

from typing import List, Dict
from fastapi import APIRouter, HTTPException, Depends, status, Body

from app.repositories.lens_profile_store import LensProfileStore
from app.models.lens_profile import LensProfileModel


router = APIRouter(prefix="/profiles")


# Dependency injection placeholder - will be configured in main.py
def get_store() -> LensProfileStore:
    """
    Dependency to get the profile store instance.
    This will be overridden in main.py with the actual store.
    """
    raise NotImplementedError("Store dependency not configured")


@router.get("", response_model=List[Dict])
def list_all_profiles(store: LensProfileStore = Depends(get_store)):
    """
    List all lens profiles.

    Returns:
        List of all profile dictionaries
    """
    return store.list_all()


@router.get("/{profile_id}", response_model=Dict)
def get_profile(profile_id: str, store: LensProfileStore = Depends(get_store)):
    """
    Get a specific lens profile by ID.

    Args:
        profile_id: Unique profile identifier

    Returns:
        Profile dictionary

    Raises:
        404: Profile not found
    """
    profile = store.get_by_id(profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Profile with ID '{profile_id}' not found")
    return profile


@router.get("/hierarchy/brands", response_model=List[str])
def list_brands(store: LensProfileStore = Depends(get_store)):
    """
    List all camera brands.

    Returns:
        Sorted list of unique brand names
    """
    return store.list_brands()


@router.get("/hierarchy/brands/{brand}/models", response_model=List[str])
def list_models(brand: str, store: LensProfileStore = Depends(get_store)):
    """
    List all models for a given brand.

    Args:
        brand: Camera brand name

    Returns:
        Sorted list of model names for the brand
    """
    return store.list_models(brand)


@router.get("/hierarchy/brands/{brand}/models/{model}", response_model=List[Dict])
def list_profiles_by_brand_model(brand: str, model: str, store: LensProfileStore = Depends(get_store)):
    """
    List all profiles for a given brand and model.

    Args:
        brand: Camera brand name
        model: Camera model name

    Returns:
        List of profile dictionaries matching brand and model
    """
    return store.list_by_brand_model(brand, model)


@router.post("", response_model=Dict, status_code=status.HTTP_201_CREATED)
def create_profile(profile: LensProfileModel, store: LensProfileStore = Depends(get_store)):
    """
    Create a new lens profile.

    Args:
        profile: Profile data to create

    Returns:
        Created profile dictionary

    Raises:
        400: Validation error
        409: Profile with same ID already exists
    """
    try:
        profile_dict = profile.model_dump()
        created = store.create(profile_dict)
        return created
    except ValueError as e:
        error_msg = str(e)
        # Check if it's an ID conflict
        if "already exists" in error_msg:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error_msg)
        # Otherwise it's a validation error
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)


@router.put("/{profile_id}", response_model=Dict)
def update_profile(profile_id: str, profile: LensProfileModel, store: LensProfileStore = Depends(get_store)):
    """
    Update an existing lens profile.

    Args:
        profile_id: ID of profile to update
        profile: Updated profile data

    Returns:
        Updated profile dictionary

    Raises:
        400: Validation error or ID mismatch
        404: Profile not found
    """
    try:
        profile_dict = profile.model_dump()
        updated = store.update(profile_id, profile_dict)
        return updated
    except ValueError as e:
        error_msg = str(e)
        # Check if it's a not found error
        if "not found" in error_msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)
        # Otherwise it's a validation or mismatch error
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_profile(profile_id: str, store: LensProfileStore = Depends(get_store)):
    """
    Delete a lens profile.

    Args:
        profile_id: ID of profile to delete

    Raises:
        404: Profile not found
    """
    deleted = store.delete(profile_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Profile with ID '{profile_id}' not found")
    return None


@router.patch("/{profile_id}/favorite", response_model=Dict)
def toggle_favorite(profile_id: str, is_favorite: bool = Body(..., embed=True), store: LensProfileStore = Depends(get_store)):
    """
    Toggle favorite status for a lens profile.

    Args:
        profile_id: ID of profile to update
        is_favorite: New favorite status

    Returns:
        Updated profile dictionary

    Raises:
        404: Profile not found
    """
    profile = store.get_by_id(profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Profile with ID '{profile_id}' not found")
    
    # Update favorite status
    profile["is_favorite"] = is_favorite
    
    try:
        updated = store.update(profile_id, profile)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/favorites/list", response_model=List[Dict])
def list_favorite_profiles(store: LensProfileStore = Depends(get_store)):
    """
    List all favorite profiles.

    Returns:
        List of favorite profile dictionaries
    """
    all_profiles = store.list_all()
    favorites = [p for p in all_profiles if p.get("is_favorite", False)]
    return favorites
