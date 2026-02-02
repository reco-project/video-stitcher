"""FastAPI router for lens profile API endpoints."""

from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, Depends, status, Body, Query

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


@router.get("/list", response_model=List[Dict])
def list_profiles_metadata(
    brand: Optional[str] = Query(None, description="Filter by camera brand (substring match)"),
    model: Optional[str] = Query(None, description="Filter by camera model (substring match)"),
    lens: Optional[str] = Query(None, description="Filter by lens model (substring match)"),
    w: Optional[int] = Query(None, description="Filter by exact width"),
    h: Optional[int] = Query(None, description="Filter by exact height"),
    official: Optional[bool] = Query(None, description="Filter by official status"),
    search: Optional[str] = Query(None, description="Full-text search query"),
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Maximum results to return"),
    offset: Optional[int] = Query(None, ge=0, description="Number of results to skip"),
    store: LensProfileStore = Depends(get_store),
):
    """
    List lens profiles with efficient metadata-only response.

    Returns lightweight profile metadata (no full calibration data).
    Use GET /profiles/{id} to get full profile with calibration data.

    Supports filtering by brand, model, lens, resolution, official status,
    and full-text search. Pagination via limit/offset.
    """
    # Build filters dict
    filters = {}
    if brand:
        filters["brand"] = brand
    if model:
        filters["model"] = model
    if lens:
        filters["lens"] = lens
    if w is not None:
        filters["w"] = w
    if h is not None:
        filters["h"] = h
    if official is not None:
        filters["official"] = official
    if search:
        filters["search"] = search

    # Check if store supports efficient metadata listing
    if hasattr(store, 'list_all_metadata'):
        return store.list_all_metadata(filters=filters, limit=limit, offset=offset)

    # Fallback for file-based store: load all and filter in Python
    all_profiles = store.list_all()

    # Apply filters
    result = []
    for p in all_profiles:
        if brand and brand.lower() not in p.get("camera_brand", "").lower():
            continue
        if model and model.lower() not in p.get("camera_model", "").lower():
            continue
        if lens and lens.lower() not in (p.get("lens_model") or "").lower():
            continue
        if w is not None and p.get("resolution", {}).get("width") != w:
            continue
        if h is not None and p.get("resolution", {}).get("height") != h:
            continue
        if official is not None:
            p_official = p.get("metadata", {}).get("official", False)
            if p_official != official:
                continue
        if search:
            # Simple text search
            search_lower = search.lower()
            searchable = " ".join(
                [
                    p.get("id", ""),
                    p.get("camera_brand", ""),
                    p.get("camera_model", ""),
                    p.get("lens_model") or "",
                    p.get("metadata", {}).get("notes") or "",
                ]
            ).lower()
            if search_lower not in searchable:
                continue

        # Convert to metadata-only dict
        result.append(
            {
                "id": p["id"],
                "camera_brand": p["camera_brand"],
                "camera_model": p["camera_model"],
                "lens_model": p.get("lens_model"),
                "w": p.get("resolution", {}).get("width"),
                "h": p.get("resolution", {}).get("height"),
                "distortion_model": p.get("distortion_model"),
                "official": p.get("metadata", {}).get("official", False),
                "source": p.get("metadata", {}).get("source"),
            }
        )

    # Apply pagination
    if offset:
        result = result[offset:]
    if limit:
        result = result[:limit]

    return result


@router.get("/count")
def count_profiles(
    brand: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    lens: Optional[str] = Query(None),
    w: Optional[int] = Query(None),
    h: Optional[int] = Query(None),
    official: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    store: LensProfileStore = Depends(get_store),
):
    """
    Get count of profiles matching filters.
    """
    filters = {}
    if brand:
        filters["brand"] = brand
    if model:
        filters["model"] = model
    if lens:
        filters["lens"] = lens
    if w is not None:
        filters["w"] = w
    if h is not None:
        filters["h"] = h
    if official is not None:
        filters["official"] = official
    if search:
        filters["search"] = search

    if hasattr(store, 'count'):
        return {"count": store.count(filters=filters)}

    # Fallback
    all_profiles = store.list_all()
    return {"count": len(all_profiles)}


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


@router.post("/cache/invalidate", status_code=status.HTTP_204_NO_CONTENT)
def invalidate_cache(store: LensProfileStore = Depends(get_store)):
    """
    Invalidate the profile store cache.

    Call this after creating/updating/deleting profiles to ensure
    fresh data is loaded on next request.
    """
    if hasattr(store, 'invalidate_cache'):
        store.invalidate_cache()
    return None


@router.post("/{profile_id}/duplicate", response_model=Dict, status_code=status.HTTP_201_CREATED)
def duplicate_profile(
    profile_id: str,
    new_id: str = Body(..., embed=True, description="New unique ID for the duplicated profile"),
    store: LensProfileStore = Depends(get_store),
):
    """
    Duplicate an existing profile (official or user) as a new user profile.

    This is the way to "edit" official profiles - duplicate them first,
    then modify the user copy.

    Args:
        profile_id: ID of the profile to duplicate
        new_id: New unique ID for the duplicated profile

    Returns:
        The newly created profile dictionary

    Raises:
        404: Source profile not found
        409: Profile with new_id already exists
    """
    # Get the source profile
    source = store.get_by_id(profile_id)
    if source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Profile with ID '{profile_id}' not found")

    # Check if new_id already exists
    if store.get_by_id(new_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Profile with ID '{new_id}' already exists")

    # Create a copy with the new ID
    new_profile = dict(source)
    new_profile['id'] = new_id

    # Update metadata to indicate it's a user copy
    if new_profile.get('metadata') is None:
        new_profile['metadata'] = {}
    new_profile['metadata']['duplicated_from'] = profile_id
    new_profile['metadata']['source'] = 'user'
    # Remove favorite status (user can re-favorite if needed)
    new_profile.pop('is_favorite', None)

    try:
        created = store.create(new_profile)
        if hasattr(store, 'invalidate_cache'):
            store.invalidate_cache()
        return created
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("", response_model=Dict, status_code=status.HTTP_201_CREATED)
def create_profile(profile: LensProfileModel, store: LensProfileStore = Depends(get_store)):
    """
    Create a new user lens profile.

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
        # Invalidate cache after creating
        if hasattr(store, 'invalidate_cache'):
            store.invalidate_cache()
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

    - For user-created profiles: Updates in place
    - For official profiles: Auto-creates a user copy with the changes

    The response includes `metadata.source` to indicate if it's 'user' or 'official',
    and `metadata.duplicated_from` if it was auto-duplicated from an official profile.

    Args:
        profile_id: ID of profile to update
        profile: Updated profile data

    Returns:
        Updated (or newly created) profile dictionary

    Raises:
        400: Validation error or ID mismatch
        404: Profile not found
    """
    profile_dict = profile.model_dump()

    try:
        # Try to update directly (works for user profiles)
        updated = store.update(profile_id, profile_dict)
        if hasattr(store, 'invalidate_cache'):
            store.invalidate_cache()
        return updated
    except RuntimeError as e:
        # Cannot modify official profile - auto-duplicate it
        if "official" in str(e).lower() or "Cannot modify" in str(e):
            # Get the original profile
            original = store.get_by_id(profile_id)
            if original is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=f"Profile with ID '{profile_id}' not found"
                )

            # Create the new profile with user's changes
            new_profile = profile_dict.copy()

            # Generate a unique ID for the user copy
            # Use the original ID as base, append "-user-copy" and a counter if needed
            base_id = f"{profile_id}-user-copy"
            new_id = base_id
            counter = 1
            while store.get_by_id(new_id) is not None:
                counter += 1
                new_id = f"{base_id}-{counter}"
            new_profile['id'] = new_id

            # Ensure metadata exists and mark as user profile
            if new_profile.get('metadata') is None:
                new_profile['metadata'] = {}
            new_profile['metadata']['source'] = 'user'
            new_profile['metadata']['duplicated_from'] = profile_id

            try:
                created = store.create(new_profile)
                if hasattr(store, 'invalidate_cache'):
                    store.invalidate_cache()
                return created
            except ValueError as create_error:
                error_msg = str(create_error)
                if "already exists" in error_msg:
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error_msg)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
        else:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_profile(profile_id: str, store: LensProfileStore = Depends(get_store)):
    """
    Delete a lens profile. Only user-created profiles can be deleted.

    Args:
        profile_id: ID of profile to delete

    Raises:
        403: Cannot delete official profiles
        404: Profile not found
    """
    try:
        deleted = store.delete(profile_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Profile with ID '{profile_id}' not found"
            )
        # Invalidate cache after deleting
        if hasattr(store, 'invalidate_cache'):
            store.invalidate_cache()
        return None
    except RuntimeError as e:
        # Cannot delete official profiles
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.patch("/{profile_id}/favorite", response_model=Dict)
def toggle_favorite(
    profile_id: str, is_favorite: bool = Body(..., embed=True), store: LensProfileStore = Depends(get_store)
):
    """
    Toggle favorite status for a lens profile.
    Works for both official (bundled) and user-created profiles.

    Args:
        profile_id: ID of profile to update
        is_favorite: New favorite status

    Returns:
        Updated profile dictionary

    Raises:
        404: Profile not found
    """
    # Use the hybrid store's set_favorite method if available
    if hasattr(store, 'set_favorite'):
        try:
            return store.set_favorite(profile_id, is_favorite)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    # Fallback for non-hybrid stores
    profile = store.get_by_id(profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Profile with ID '{profile_id}' not found")

    profile["is_favorite"] = is_favorite

    try:
        updated = store.update(profile_id, profile)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/favorites/ids", response_model=List[str])
def list_favorite_ids(store: LensProfileStore = Depends(get_store)):
    """
    List IDs of favorite profiles (fast).

    Returns:
        List of favorite profile IDs
    """
    # Use hybrid store's efficient method if available
    if hasattr(store, 'list_favorite_ids'):
        return store.list_favorite_ids()

    # Fallback
    all_profiles = store.list_all()
    favorite_ids = [p["id"] for p in all_profiles if p.get("is_favorite", False)]
    return favorite_ids


@router.get("/favorites/list", response_model=List[Dict])
def list_favorite_profiles(store: LensProfileStore = Depends(get_store)):
    """
    List all favorite profiles.

    Returns:
        List of favorite profile dictionaries
    """
    # Use hybrid store's efficient method if available
    if hasattr(store, 'list_favorites'):
        return store.list_favorites()

    # Fallback
    all_profiles = store.list_all()
    favorites = [p for p in all_profiles if p.get("is_favorite", False)]
    return favorites
