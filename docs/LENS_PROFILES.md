# Lens Profile Management

Backend system for managing camera lens calibration profiles used for distortion correction and video stitching.

## Overview

Inspired by [Gyroflow's lens_profiles repository](https://github.com/gyroflow/lens_profiles). Provides:

- File-based storage with hierarchical structure (`brand/model/profile.json`)
- RESTful API for CRUD operations
- Pydantic validation
- Real Gyroflow calibration data (CC0 1.0 Universal license)

## Architecture

**Components:**

- `app/models/lens_profile.py` - Pydantic validation models
- `app/repositories/lens_profile_store.py` - Abstract storage interface
- `app/repositories/file_lens_profile_store.py` - File implementation
- `app/routers/profiles.py` - FastAPI REST endpoints
- `app/utils/slug.py` - Slug generation utility

**Storage:** `backend/data/lens_profiles/{brand-slug}/{model-slug}/{profile-id}.json`

## Profile Schema

```json
{
	"id": "gopro-hero10black-linear-3840x2160",
	"camera_brand": "GoPro",
	"camera_model": "HERO10 Black",
	"lens_model": "Linear",
	"resolution": { "width": 3840, "height": 2160 },
	"distortion_model": "fisheye_kb4",
	"camera_matrix": { "fx": 2532.61, "fy": 2537.19, "cx": 2658.31, "cy": 1501.14 },
	"distortion_coeffs": [0.3503, 0.0307, 0.2982, -0.159],
	"calib_dimension": { "width": 5312, "height": 2988 },
	"note": "Optional metadata"
}
```

**Required fields:** `id`, `camera_brand`, `camera_model`, `resolution`, `distortion_model`, `camera_matrix`, `distortion_coeffs`

**Constraints:**

- `id`: Lowercase alphanumeric + hyphens, max 100 chars
- `distortion_model`: Must be `"fisheye_kb4"`
- `distortion_coeffs`: Exactly 4 numbers
- All camera matrix values must be positive

## API Endpoints

Base path: `/api/profiles`

### CRUD Operations

- `GET /api/profiles` - List all profiles
- `GET /api/profiles/{id}` - Get profile by ID
- `POST /api/profiles` - Create profile (body: profile JSON)
- `PUT /api/profiles/{id}` - Update profile (body: profile JSON)
- `DELETE /api/profiles/{id}` - Delete profile

### Hierarchy

- `GET /api/profiles/hierarchy/brands` - List all brands
- `GET /api/profiles/hierarchy/brands/{brand}/models` - List models for brand
- `GET /api/profiles/hierarchy/brands/{brand}/models/{model}` - List profiles for brand/model

### Response Codes

- `200` OK, `201` Created, `204` No Content
- `404` Not Found, `409` Conflict (duplicate ID), `422` Validation Error

## Usage Examples

### Python

```python
import httpx
response = httpx.get("http://localhost:8000/api/profiles")
profiles = response.json()
```

### JavaScript

```javascript
import { listProfiles, useProfiles } from '@/features/profiles';

// Direct API call
const profiles = await listProfiles();

// React hook
const { profiles, loading, error } = useProfiles();
```

### cURL

```bash
curl http://localhost:8000/api/profiles
curl http://localhost:8000/api/profiles/gopro-hero10black-linear-3840x2160
```

## Testing

```bash
cd backend
pytest tests/ -v                          # Run all tests
pytest tests/test_profiles_api.py -v     # API tests only
```

Tests cover: slug generation, Pydantic validation, file storage CRUD, API endpoints, hierarchy navigation.

## Data Sources

Included profiles are from [Gyroflow lens_profiles](https://github.com/gyroflow/lens_profiles) (CC0 1.0 Universal):

- GoPro HERO10 Black (Linear, 4K) - 3840×2160
- GoPro HERO9 Black (Wide, 2.7K) - 2704×1520
- Insta360 ONE X2 (Ultrawide 148°) - 2880×2880
- DJI Action 2 (Standard, 4K) - 4000×3000

### Adding Profiles

**Via API:** Use `POST /api/profiles` with profile JSON

**Manual:** Create `backend/data/lens_profiles/{brand-slug}/{model-slug}/{profile-id}.json` and restart server

## Development

### Extending Storage

Implement `LensProfileStore` interface for alternative backends (database, S3, etc.):

```python
from app.repositories.lens_profile_store import LensProfileStore

class DatabaseLensProfileStore(LensProfileStore):
    async def create(self, profile: LensProfile) -> LensProfile:
        # Your implementation
        pass
```

Then update `app/main.py` to use the new store.

---

**License:** Calibration data is CC0 1.0 Universal (Public Domain)
