"""
Hybrid lens profile store combining SQLite (official) and file-based (user) profiles.

Architecture:
- SQLite database: Read-only official bundled profiles (~2700 from gyroflow)
- User JSON files: Read-write user-created profiles stored in userData
- Favorites JSON: Separate file tracking favorite profile IDs (works for both)

This allows:
- Fast read access to bundled profiles via SQLite
- User profile creation/editing stored in user data directory
- Favorites that work regardless of profile source
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from app.repositories.lens_profile_store import LensProfileStore
from app.repositories.sqlite_lens_profile_store import SqliteLensProfileStore
from app.repositories.file_lens_profile_store import FileLensProfileStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HybridLensProfileStore(LensProfileStore):
    """
    Hybrid store combining read-only SQLite and read-write file stores.

    Profiles from SQLite are marked with source='official'.
    User-created profiles are stored in the file store with source='user'.
    Favorites are stored separately and merged into profile data on read.
    """

    def __init__(
        self,
        sqlite_db_path: Optional[str] = None,
        user_profiles_dir: str = None,
        favorites_file: str = None,
        official_profiles_dir: Optional[str] = None,
    ):
        """
        Initialize hybrid store.

        Args:
            sqlite_db_path: Path to SQLite database (optional, can be None if not available)
            user_profiles_dir: Directory for user-created profiles
            favorites_file: Path to favorites JSON file
            official_profiles_dir: Fallback directory for official profiles when SQLite not available
        """
        self.sqlite_store = None
        self.official_file_store = None
        self.file_store = None
        self.favorites_file = Path(favorites_file) if favorites_file else None
        self._favorites_cache = None
        self._favorites_cache_time = None

        # Initialize SQLite store if database exists
        if sqlite_db_path and Path(sqlite_db_path).exists():
            try:
                self.sqlite_store = SqliteLensProfileStore(sqlite_db_path, read_only=True)
                logger.info(f"SQLite store initialized: {sqlite_db_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize SQLite store: {e}")
                self.sqlite_store = None

        # If SQLite not available, use file-based fallback for official profiles
        if not self.sqlite_store and official_profiles_dir and Path(official_profiles_dir).exists():
            self.official_file_store = FileLensProfileStore(official_profiles_dir)
            logger.info(f"Official profiles fallback (JSON): {official_profiles_dir}")

        # Initialize file store for user profiles
        if user_profiles_dir:
            self.file_store = FileLensProfileStore(user_profiles_dir)
            logger.info(f"User profile store initialized: {user_profiles_dir}")

        # Ensure favorites file directory exists
        if self.favorites_file:
            self.favorites_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_favorites(self) -> set:
        """Load favorite profile IDs from file."""
        if self._favorites_cache is not None:
            # Check if cache is still valid (1 minute TTL)
            if self._favorites_cache_time and (datetime.now() - self._favorites_cache_time).seconds < 60:
                return self._favorites_cache

        if not self.favorites_file or not self.favorites_file.exists():
            self._favorites_cache = set()
            self._favorites_cache_time = datetime.now()
            return self._favorites_cache

        try:
            with open(self.favorites_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._favorites_cache = set(data.get('favorites', []))
                self._favorites_cache_time = datetime.now()
                return self._favorites_cache
        except Exception as e:
            logger.warning(f"Failed to load favorites: {e}")
            self._favorites_cache = set()
            self._favorites_cache_time = datetime.now()
            return self._favorites_cache

    def _save_favorites(self, favorites: set) -> None:
        """Save favorite profile IDs to file."""
        if not self.favorites_file:
            return

        try:
            with open(self.favorites_file, 'w', encoding='utf-8') as f:
                json.dump({'favorites': list(favorites)}, f, indent=2)
            self._favorites_cache = favorites
            self._favorites_cache_time = datetime.now()
        except Exception as e:
            logger.error(f"Failed to save favorites: {e}")

    def _add_favorite_status(self, profile: Dict) -> Dict:
        """Add is_favorite field to profile based on favorites file."""
        favorites = self._load_favorites()
        profile = dict(profile)  # Don't mutate original
        profile['is_favorite'] = profile.get('id') in favorites
        return profile

    def _add_source_marker(self, profile: Dict, source: str) -> Dict:
        """Add source marker to profile metadata."""
        profile = dict(profile)
        if profile.get('metadata') is None:
            profile['metadata'] = {}
        else:
            profile['metadata'] = dict(profile['metadata'])  # Don't mutate original
        profile['metadata']['source'] = source
        return profile

    def _get_official_store(self):
        """Get the store for official profiles (SQLite or fallback file store)."""
        return self.sqlite_store or self.official_file_store

    # ========== Read Operations ==========

    def list_all(self) -> List[Dict]:
        """List all profiles from all stores."""
        profiles = []

        # Get official profiles from SQLite or fallback
        official_store = self._get_official_store()
        if official_store:
            for p in official_store.list_all():
                p = self._add_source_marker(p, 'official')
                p = self._add_favorite_status(p)
                profiles.append(p)

        # Get user profiles from file store
        if self.file_store:
            for p in self.file_store.list_all():
                p = self._add_source_marker(p, 'user')
                p = self._add_favorite_status(p)
                profiles.append(p)

        return profiles

    def list_all_metadata(
        self,
        filters: Optional[Dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict]:
        """List profiles with metadata only (efficient for large lists)."""
        profiles = []
        favorites = self._load_favorites()

        # Helper to convert full profile to metadata format
        def to_metadata(p: Dict, source: str) -> Dict:
            return {
                'id': p['id'],
                'camera_brand': p['camera_brand'],
                'camera_model': p['camera_model'],
                'lens_model': p.get('lens_model'),
                'w': p.get('resolution', {}).get('width'),
                'h': p.get('resolution', {}).get('height'),
                'distortion_model': p.get('distortion_model'),
                'official': source == 'official',
                'source': source,
                'is_favorite': p.get('id') in favorites,
            }

        # Helper to filter profile
        def matches_filters(p: Dict, filters: Dict) -> bool:
            if not filters:
                return True
            if 'brand' in filters and filters['brand'].lower() not in p.get('camera_brand', '').lower():
                return False
            if 'model' in filters and filters['model'].lower() not in p.get('camera_model', '').lower():
                return False
            if 'lens' in filters and filters['lens'].lower() not in (p.get('lens_model') or '').lower():
                return False
            if 'w' in filters and p.get('resolution', {}).get('width') != filters['w']:
                return False
            if 'h' in filters and p.get('resolution', {}).get('height') != filters['h']:
                return False
            # Full-text search: each word must appear somewhere
            if 'search' in filters and filters['search']:
                search_text = ' '.join(
                    [
                        p.get('id', ''),
                        p.get('camera_brand', ''),
                        p.get('camera_model', ''),
                        p.get('lens_model') or '',
                    ]
                ).lower()
                words = filters['search'].lower().split()
                for word in words:
                    if word not in search_text:
                        return False
            return True

        # Get from SQLite (has efficient metadata query)
        if self.sqlite_store:
            sqlite_profiles = self.sqlite_store.list_all_metadata(filters=filters)
            for p in sqlite_profiles:
                p = dict(p)
                p['source'] = 'official'
                p['is_favorite'] = p.get('id') in favorites
                profiles.append(p)
        elif self.official_file_store:
            # Fallback to file store for official profiles
            for p in self.official_file_store.list_all():
                if matches_filters(p, filters):
                    profiles.append(to_metadata(p, 'official'))

        # Get user profiles from file store
        if self.file_store:
            for p in self.file_store.list_all():
                if matches_filters(p, filters):
                    profiles.append(to_metadata(p, 'user'))

        # Apply pagination
        if offset:
            profiles = profiles[offset:]
        if limit:
            profiles = profiles[:limit]

        return profiles

    def get_by_id(self, profile_id: str) -> Optional[Dict]:
        """Get profile by ID, checking user store first then official stores."""
        # Check user profiles first (allows overriding bundled profiles)
        if self.file_store:
            profile = self.file_store.get_by_id(profile_id)
            if profile:
                profile = self._add_source_marker(profile, 'user')
                return self._add_favorite_status(profile)

        # Check official stores (SQLite or fallback file store)
        official_store = self._get_official_store()
        if official_store:
            profile = official_store.get_by_id(profile_id)
            if profile:
                profile = self._add_source_marker(profile, 'official')
                return self._add_favorite_status(profile)

        return None

    def list_brands(self) -> List[str]:
        """List all unique camera brands."""
        brands = set()

        official_store = self._get_official_store()
        if official_store:
            brands.update(official_store.list_brands())

        if self.file_store:
            brands.update(self.file_store.list_brands())

        return sorted(brands)

    def list_models(self, brand: str) -> List[str]:
        """List camera models for a brand."""
        models = set()

        official_store = self._get_official_store()
        if official_store:
            models.update(official_store.list_models(brand))

        if self.file_store:
            models.update(self.file_store.list_models(brand))

        return sorted(models)

    def list_by_brand_model(self, brand: str, model: str) -> List[Dict]:
        """List profiles for a specific brand and model."""
        profiles = []

        official_store = self._get_official_store()
        if official_store:
            for p in official_store.list_by_brand_model(brand, model):
                p = self._add_source_marker(p, 'official')
                p = self._add_favorite_status(p)
                profiles.append(p)

        if self.file_store:
            for p in self.file_store.list_by_brand_model(brand, model):
                p = self._add_source_marker(p, 'user')
                p = self._add_favorite_status(p)
                profiles.append(p)

        return profiles

    def count(self, filters: Optional[Dict] = None) -> int:
        """Count profiles matching filters."""
        total = 0

        official_store = self._get_official_store()
        if official_store:
            if hasattr(official_store, 'count'):
                total += official_store.count(filters)
            else:
                # Fallback for file store
                total += len(official_store.list_all())

        if self.file_store:
            # File store doesn't have count method, use list_all
            profiles = self.file_store.list_all()
            if filters:
                # Apply filters
                for p in profiles:
                    match = True
                    if 'brand' in filters and filters['brand'].lower() not in p.get('camera_brand', '').lower():
                        match = False
                    if (
                        match
                        and 'model' in filters
                        and filters['model'].lower() not in p.get('camera_model', '').lower()
                    ):
                        match = False
                    if match:
                        total += 1
            else:
                total += len(profiles)

        return total

    # ========== Write Operations (User Profiles Only) ==========

    def create(self, profile: Dict) -> Dict:
        """Create a new user profile."""
        if not self.file_store:
            raise RuntimeError("No file store configured for user profiles")

        # Check if ID already exists in either store
        if self.get_by_id(profile['id']):
            raise ValueError(f"Profile with ID '{profile['id']}' already exists")

        # Mark as user profile
        profile = self._add_source_marker(profile, 'user')

        # Create in file store
        created = self.file_store.create(profile)
        return self._add_favorite_status(created)

    def update(self, profile_id: str, profile: Dict) -> Dict:
        """Update a profile. Only user profiles can be modified."""
        if not self.file_store:
            raise RuntimeError("No file store configured for user profiles")

        # Check if this is a user profile
        existing = self.file_store.get_by_id(profile_id)
        if existing:
            # Update existing user profile
            updated = self.file_store.update(profile_id, profile)
            return self._add_favorite_status(updated)

        # Check if it's an official profile (SQLite or fallback file store)
        official_store = self._get_official_store()
        if official_store and official_store.get_by_id(profile_id):
            raise RuntimeError("Cannot modify official bundled profiles. Create a copy with a different ID instead.")

        raise ValueError(f"Profile with ID '{profile_id}' not found")

    def delete(self, profile_id: str) -> bool:
        """Delete a profile. Only user profiles can be deleted."""
        if not self.file_store:
            raise RuntimeError("No file store configured for user profiles")

        # Check if this is a user profile
        existing = self.file_store.get_by_id(profile_id)
        if existing:
            # Also remove from favorites
            favorites = self._load_favorites()
            favorites.discard(profile_id)
            self._save_favorites(favorites)

            return self.file_store.delete(profile_id)

        # Check if it's an official profile (SQLite or fallback file store)
        official_store = self._get_official_store()
        if official_store and official_store.get_by_id(profile_id):
            raise RuntimeError("Cannot delete official bundled profiles")

        return False

    # ========== Favorites Management ==========

    def set_favorite(self, profile_id: str, is_favorite: bool) -> Dict:
        """Set favorite status for a profile."""
        # Verify profile exists
        profile = self.get_by_id(profile_id)
        if not profile:
            raise ValueError(f"Profile with ID '{profile_id}' not found")

        # Update favorites file
        favorites = self._load_favorites()
        if is_favorite:
            favorites.add(profile_id)
        else:
            favorites.discard(profile_id)
        self._save_favorites(favorites)

        # Return updated profile
        profile['is_favorite'] = is_favorite
        return profile

    def list_favorites(self) -> List[Dict]:
        """List all favorite profiles."""
        favorites = self._load_favorites()
        result = []

        for profile_id in favorites:
            profile = self.get_by_id(profile_id)
            if profile:
                result.append(profile)

        return result

    def list_favorite_ids(self) -> List[str]:
        """List IDs of favorite profiles."""
        return list(self._load_favorites())

    # ========== Cache Management ==========

    def invalidate_cache(self) -> None:
        """Invalidate all caches."""
        self._favorites_cache = None
        self._favorites_cache_time = None

        if self.file_store:
            self.file_store._cache = None
            self.file_store._cache_time = None
        if self.official_file_store:
            self.official_file_store._cache = None
            self.official_file_store._cache_time = None
