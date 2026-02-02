"""
SQLite-based implementation of LensProfileStore.

Reads lens profiles from a single SQLite database file for efficient
access and to avoid Windows Defender slowness with many small files.

The database is read-only in production (built at CI time).
Write operations are only supported in development mode with JSON fallback.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

from app.repositories.lens_profile_store import LensProfileStore
from app.models.lens_profile import LensProfileModel


class SqliteLensProfileStore(LensProfileStore):
    """SQLite-based lens profile storage implementation (read-only)."""

    def __init__(self, db_path: str, read_only: bool = True):
        """
        Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file
            read_only: If True, opens database in read-only mode (default)
                      If False and DB doesn't exist, creates a new one
        """
        self.db_path = Path(db_path)
        self.read_only = read_only

        if not self.db_path.exists():
            if read_only:
                raise FileNotFoundError(f"Lens profile database not found: {db_path}")
            else:
                # Create new database with schema
                self._create_schema()

    def _create_schema(self):
        """Create database schema for tests."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    camera_brand TEXT NOT NULL,
                    camera_model TEXT NOT NULL,
                    lens_model TEXT,
                    w INTEGER,
                    h INTEGER,
                    distortion_model TEXT,
                    official INTEGER DEFAULT 1,
                    source TEXT DEFAULT 'official',
                    json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS profiles_fts USING fts5(
                    id, camera_brand, camera_model, lens_model,
                    content=profiles,
                    content_rowid=rowid
                )
            """)
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper isolation."""
        # Use read-only URI for read-only mode
        if self.read_only:
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _row_to_metadata_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a lightweight metadata dict (without full JSON)."""
        return {
            "id": row["id"],
            "camera_brand": row["camera_brand"],
            "camera_model": row["camera_model"],
            "lens_model": row["lens_model"],
            "w": row["w"],
            "h": row["h"],
            "distortion_model": row["distortion_model"],
            "official": bool(row["official"]),
            "source": row["source"],
        }

    def _row_to_full_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a full profile dict (parse JSON)."""
        return json.loads(row["json"])

    def list_all(self) -> List[Dict]:
        """Return all profiles (full JSON parsed)."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT json FROM profiles ORDER BY id")
            return [json.loads(row["json"]) for row in cursor.fetchall()]

    def list_all_metadata(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict]:
        """
        Return lightweight metadata for all profiles (no full JSON).

        This is much faster for listing/searching as it doesn't parse JSON.

        Args:
            filters: Optional dict with filter criteria:
                - brand: substring match on camera_brand (case-insensitive)
                - model: substring match on camera_model (case-insensitive)
                - lens: substring match on lens_model (case-insensitive)
                - w: exact match on width
                - h: exact match on height
                - official: boolean filter
                - search: full-text search query
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of metadata dicts (id, camera_brand, camera_model, lens_model, w, h, distortion_model, official, source)
        """
        filters = filters or {}
        conditions = []
        params = []

        # Build WHERE conditions
        if "brand" in filters and filters["brand"]:
            conditions.append("camera_brand LIKE ?")
            params.append(f"%{filters['brand']}%")

        if "model" in filters and filters["model"]:
            conditions.append("camera_model LIKE ?")
            params.append(f"%{filters['model']}%")

        if "lens" in filters and filters["lens"]:
            conditions.append("lens_model LIKE ?")
            params.append(f"%{filters['lens']}%")

        if "w" in filters and filters["w"] is not None:
            conditions.append("w = ?")
            params.append(filters["w"])

        if "h" in filters and filters["h"] is not None:
            conditions.append("h = ?")
            params.append(filters["h"])

        if "official" in filters and filters["official"] is not None:
            conditions.append("official = ?")
            params.append(1 if filters["official"] else 0)

        # Full-text search using LIKE (more flexible than FTS5 for partial matches)
        if "search" in filters and filters["search"]:
            search_term = filters["search"].strip()
            if search_term:
                # Split into words and search for each word in any field
                words = search_term.lower().split()
                for word in words:
                    # Each word must appear in at least one of the searchable fields
                    conditions.append(
                        "(LOWER(camera_brand) LIKE ? OR LOWER(camera_model) LIKE ? OR LOWER(COALESCE(lens_model, '')) LIKE ? OR LOWER(id) LIKE ?)"
                    )
                    like_pattern = f"%{word}%"
                    params.extend([like_pattern, like_pattern, like_pattern, like_pattern])

        # Build SQL query
        sql = """
            SELECT id, camera_brand, camera_model, lens_model, w, h, 
                   distortion_model, official, source
            FROM profiles
        """

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY camera_brand, camera_model, lens_model, id"

        if limit is not None:
            sql += f" LIMIT {int(limit)}"
            if offset is not None:
                sql += f" OFFSET {int(offset)}"

        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            return [self._row_to_metadata_dict(row) for row in cursor.fetchall()]

    def get_by_id(self, profile_id: str) -> Optional[Dict]:
        """Return single profile by ID (full JSON parsed), or None if not found."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT json FROM profiles WHERE id = ?", (profile_id,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row["json"])
            return None

    def create(self, profile: Dict) -> Dict:
        """Create new profile (not supported in read-only mode)."""
        if self.read_only:
            raise RuntimeError(
                "Cannot create profiles in read-only SQLite store. "
                "Use JSON source files and rebuild the database."
            )

        # Validate using Pydantic model
        model = LensProfileModel(**profile)
        validated = model.model_dump()

        # Extract metadata fields
        metadata = validated.get("metadata", {}) or {}
        w = validated.get("resolution", {}).get("width")
        h = validated.get("resolution", {}).get("height")

        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO profiles 
                    (id, camera_brand, camera_model, lens_model, w, h, 
                     distortion_model, official, source, source_file, notes, json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        validated["id"],
                        validated["camera_brand"],
                        validated["camera_model"],
                        validated.get("lens_model"),
                        w,
                        h,
                        validated.get("distortion_model"),
                        1 if metadata.get("official") else 0,
                        metadata.get("source"),
                        metadata.get("source_file"),
                        metadata.get("notes"),
                        json.dumps(validated),
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError(f"Profile with ID '{validated['id']}' already exists")

        return validated

    def update(self, profile_id: str, profile: Dict) -> Dict:
        """Update existing profile (not supported in read-only mode)."""
        if self.read_only:
            raise RuntimeError(
                "Cannot update profiles in read-only SQLite store. "
                "Use JSON source files and rebuild the database."
            )

        # Validate using Pydantic model
        model = LensProfileModel(**profile)
        validated = model.model_dump()

        if validated["id"] != profile_id:
            raise ValueError(
                f"Profile ID mismatch: URL has '{profile_id}', body has '{validated['id']}'"
            )

        # Extract metadata fields
        metadata = validated.get("metadata", {}) or {}
        w = validated.get("resolution", {}).get("width")
        h = validated.get("resolution", {}).get("height")

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE profiles SET
                    camera_brand = ?,
                    camera_model = ?,
                    lens_model = ?,
                    w = ?,
                    h = ?,
                    distortion_model = ?,
                    official = ?,
                    source = ?,
                    source_file = ?,
                    notes = ?,
                    json = ?
                WHERE id = ?
                """,
                (
                    validated["camera_brand"],
                    validated["camera_model"],
                    validated.get("lens_model"),
                    w,
                    h,
                    validated.get("distortion_model"),
                    1 if metadata.get("official") else 0,
                    metadata.get("source"),
                    metadata.get("source_file"),
                    metadata.get("notes"),
                    json.dumps(validated),
                    profile_id,
                ),
            )
            conn.commit()

            if cursor.rowcount == 0:
                raise ValueError(f"Profile with ID '{profile_id}' not found")

        return validated

    def delete(self, profile_id: str) -> bool:
        """Delete profile by ID (not supported in read-only mode)."""
        if self.read_only:
            raise RuntimeError(
                "Cannot delete profiles in read-only SQLite store. "
                "Use JSON source files and rebuild the database."
            )

        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
            conn.commit()
            return cursor.rowcount > 0

    def list_brands(self) -> List[str]:
        """Return list of unique camera brands (sorted)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT camera_brand FROM profiles ORDER BY camera_brand"
            )
            return [row["camera_brand"] for row in cursor.fetchall()]

    def list_models(self, brand: str) -> List[str]:
        """Return list of models for given brand (case-insensitive match, sorted)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT camera_model FROM profiles 
                WHERE LOWER(camera_brand) = LOWER(?)
                ORDER BY camera_model
                """,
                (brand,),
            )
            return [row["camera_model"] for row in cursor.fetchall()]

    def list_by_brand_model(self, brand: str, model: str) -> List[Dict]:
        """Return all profiles for given brand and model (case-insensitive, full JSON)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT json FROM profiles 
                WHERE LOWER(camera_brand) = LOWER(?) AND LOWER(camera_model) = LOWER(?)
                ORDER BY id
                """,
                (brand, model),
            )
            return [json.loads(row["json"]) for row in cursor.fetchall()]

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Return total count of profiles matching filters.

        Args:
            filters: Same filter options as list_all_metadata

        Returns:
            Total count of matching profiles
        """
        filters = filters or {}
        conditions = []
        params = []

        if "brand" in filters and filters["brand"]:
            conditions.append("camera_brand LIKE ?")
            params.append(f"%{filters['brand']}%")

        if "model" in filters and filters["model"]:
            conditions.append("camera_model LIKE ?")
            params.append(f"%{filters['model']}%")

        if "lens" in filters and filters["lens"]:
            conditions.append("lens_model LIKE ?")
            params.append(f"%{filters['lens']}%")

        if "w" in filters and filters["w"] is not None:
            conditions.append("w = ?")
            params.append(filters["w"])

        if "h" in filters and filters["h"] is not None:
            conditions.append("h = ?")
            params.append(filters["h"])

        if "official" in filters and filters["official"] is not None:
            conditions.append("official = ?")
            params.append(1 if filters["official"] else 0)

        if "search" in filters and filters["search"]:
            search_term = filters["search"].strip()
            if search_term:
                conditions.append(
                    "id IN (SELECT id FROM profiles_fts WHERE profiles_fts MATCH ?)"
                )
                escaped = search_term.replace('"', '""')
                params.append(f'"{escaped}"*')

        sql = "SELECT COUNT(*) as count FROM profiles"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            row = cursor.fetchone()
            return row["count"] if row else 0

    def search_fts(self, query: str, limit: int = 50) -> List[Dict]:
        """
        Full-text search across id, brand, model, lens, and notes.

        Args:
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of metadata dicts matching the query
        """
        if not query or not query.strip():
            return []

        # Escape special characters and add wildcard for prefix matching
        escaped = query.strip().replace('"', '""')
        fts_query = f'"{escaped}"*'

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT p.id, p.camera_brand, p.camera_model, p.lens_model, 
                       p.w, p.h, p.distortion_model, p.official, p.source
                FROM profiles p
                JOIN profiles_fts fts ON p.id = fts.id
                WHERE profiles_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            )
            return [self._row_to_metadata_dict(row) for row in cursor.fetchall()]
