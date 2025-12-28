# Backend API

FastAPI backend for lens profile management and match orchestration.

## Quick Start

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

Server: `http://localhost:8000` | API docs: `http://localhost:8000/docs`

**Tests:** `pytest tests/ -v`

## Project Structure

```
backend/
├── app/
│   ├── main.py                          # FastAPI app, CORS, dependency injection
│   ├── models/
│   │   ├── lens_profile.py              # LensProfile Pydantic model
│   │   └── match.py                     # MatchModel, VideoInput Pydantic models
│   ├── repositories/
│   │   ├── lens_profile_store.py        # Abstract LensProfileStore interface
│   │   ├── file_lens_profile_store.py   # File-based profile storage
│   │   ├── match_store.py               # Abstract MatchStore interface
│   │   └── file_match_store.py          # File-based match storage
│   ├── routers/
│   │   ├── profiles.py                  # /api/profiles endpoints
│   │   └── matches.py                   # /api/matches endpoints
│   └── utils/
│       └── slug.py                      # Slug generation utility
├── data/
│   ├── lens_profiles/                   # Lens calibration files
│   │   └── {brand}/{model}/{id}.json
│   └── matches/                         # Match storage
│       └── {match-id}.json
├── docs/
│   ├── LENS_PROFILES.md                 # Profile system documentation
│   └── MATCHES.md                       # Match system documentation
├── tests/
│   ├── test_profiles_api.py             # Profile API tests
│   ├── test_matches_api.py              # Match API tests
│   └── conftest.py                      # Pytest fixtures
├── pyproject.toml                       # Project metadata
└── requirements.txt                     # Python dependencies
```

## API Endpoints

- **Health:** `GET /`, `GET /api/health`
- **Profiles:** `/api/profiles` - Full CRUD + hierarchy navigation
- **Matches:** `/api/matches` - Full CRUD operations

See [LENS_PROFILES.md](./docs/LENS_PROFILES.md) and [MATCHES.md](./docs/MATCHES.md) for details.

## Storage

**Profiles:** `backend/data/lens_profiles/{brand}/{model}/{id}.json`  
**Matches:** `backend/data/matches/{match-id}.json`

User-created matches are git-ignored (except m1-m5 samples).

## Development

### Adding New Endpoints

1. Create route handler in `app/routers/`
2. Import router in `app/main.py`
3. Register with `app.include_router()`
4. Add tests in `tests/`

Example:

```python
# app/routers/new_feature.py
from fastapi import APIRouter

router = APIRouter(prefix="/new-feature")

@router.get("")
def list_items():
    return {"items": []}

# app/main.py
import app.routers.new_feature as new_feature_router
app.include_router(new_feature_router.router, prefix="/api", tags=["new-feature"])
```

### Implementing New Storage Backend

1. Create class implementing `MatchStore` or `LensProfileStore`
2. Implement all abstract methods
3. Update dependency injection in `app/main.py`

Example:

```python
from app.repositories.match_store import MatchStore

class RedisMatchStore(MatchStore):
    def __init__(self, redis_client):
        self.client = redis_client

    def list_all(self) -> List[Dict]:
        # Implementation
        pass

    # ... implement other methods

# In app/main.py
from redis import Redis
from app.repositories.redis_match_store import RedisMatchStore

redis_client = Redis(host='localhost', port=6379)
match_store = RedisMatchStore(redis_client)
## Architecture

Storage backends implement abstract interfaces (`MatchStore`, `LensProfileStore`). Swap implementations by updating dependency injection in `app/main.py`.

**Key dependencies:** FastAPI, Uvicorn, Pydantic, pytest
# Kill process or use different port
uvicorn app.main:app --reload --port 8001
```

### Import errors

```bash
# Ensure you're in backend directory
cd backend

# Reinstall dependencies
pip install -r requirements.txt

# Check Python version (3.10+ required)
python --version
```

### CORS issues

Check `allow_origins` in `app/main.py`. For development, use `["*"]`. For production, specify exact frontend URL.

### Data not persisting

- Ensure `backend/data/` directories exist
- Check file permissions

## Status

---

**Detailed docs:** [Lens Profiles](./docs/LENS_PROFILES.md) | [Matches](./docs/MATCHES.md)  
**License:** AGPL-3.0 | Gyroflow data: CC0 1.0 Universal
