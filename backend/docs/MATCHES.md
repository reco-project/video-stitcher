# Match Management

Backend system for managing video stitching projects with multiple video inputs and lens profiles.

## Overview

- **Multi-video input**: Lists of raw videos from left/right cameras
- **Profile assignment**: Link lens calibration to video sources
- **Calibration params**: Store alignment and stitching parameters
- **Persistent storage**: File-based JSON in `backend/data/matches/`
- **RESTful API**: Full CRUD operations

## Match Lifecycle

```
1. Select raw videos → VideoImportStep
2. Assign lens profiles → ProfileAssignmentStep
3. Create match (videos + uniforms) → API
4. [Backend stitching - NOT IMPLEMENTED]
5. Backend adds `src` URL → Match viewable
6. View stitched result → Viewer
```

**Note:** Backend video processing (step 4) not implemented. Only pre-stitched matches (with `src` URLs) can be viewed.

## Match Schema

### Input (Creation)

```json
{
  "id": "match-1735344000000",
  "name": "Match 2024-12-27",
  "left_videos": [
    {"path": "/path/to/left1.mp4", "profile_id": "gopro-hero10-..."}
  ],
  "right_videos": [
    {"path": "/path/to/right1.mp4", "profile_id": "gopro-hero9-..."}
  ],
  "params": {
    "cameraAxisOffset": 0.23,
    "intersect": 0.55,
    "zRx": 0.0,
    "xTy": 0.0,
    "xRz": 0.0
  },
  "left_uniforms": {
    "width": 3840, "height": 2160,
    "fx": 2532.61, "fy": 2537.19, "cx": 2658.31, "cy": 1501.14,
    "d": [0.3503, 0.0307, 0.2982, -0.159]
  },
  "right_uniforms": {...},
  "metadata": {"left_profile_id": "...", "right_profile_id": "..."}
}
```

### Output (After Processing)

Same as input + `"src": "https://storage.../stitched_output.mp4"` + `"created_at": "2024-12-27T10:30:00Z"`

## API

- `GET /api/matches` - List all
- `GET /api/matches/{id}` - Get by ID
- `POST /api/matches` - Create
- `PUT /api/matches/{id}` - Update
- `DELETE /api/matches/{id}` - Delete

**Status codes:** 200 OK, 201 Created, 204 No Content, 400 Bad Request, 404 Not Found, 409 Conflict

## Usage Examples

### Python (Direct API)

```python
import httpx

# List all matches
response = httpx.get("http://localhost:8000/api/matches")
matches = response.json()

# Get specific match
match = httpx.get("http://localhost:8000/api/matches/match-1735344000000").json()

# Create match
new_match = {
    "id": "match-1735344000001",
    "name": "Test Match",
    "left_videos": [{"path": "/video1.mp4"}],
    "right_videos": [{"path": "/video2.mp4"}],
    "params": {...},
    "left_uniforms": {...},
    "right_uniforms": {...}
}
httpx.post("http://localhost:8000/api/matches", json=new_match)
```

### JavaScript (Frontend)

````javascript
import { useMatches, useMatchMutations } from '@/features/matches/hooks/useMatches';

// React hooks
## Usage

### Python

```python
import httpx
matches = httpx.get("http://localhost:8000/api/matches").json()
match = httpx.get("http://localhost:8000/api/matches/match-123").json()
httpx.post("http://localhost:8000/api/matches", json=match_data)
````

### JavaScript

```javascript
const { matches } = useMatches();
const { create, update, delete: del } = useMatchMutations();
await create(matchData);
```

### cURL

```bash
curl http://localhost:8000/api/matches
curl -X POST http://localhost:8000/api/matches -H "Content-Type: application/json" -d @match.json
```

from app.repositories.match_store import MatchStore

class DatabaseMatchStore(MatchStore):
def list_all(self) -> List[Dict]: # Database implementation
pass

    def create(self, match: Dict) -> Dict:
        # Database implementation
        pass

    # ... implement all abstract methods

````

Update `app/main.py` to use the new implementation:

```python
from app.repositories.database_match_store import DatabaseMatchStore

match_store = DatabaseMatchStore(connection_string)
````

### Adding Backend Processing

Future implementation should:

## Frontend Integration

**Creation:** VideoImportStep → ProfileAssignmentStep → MatchWizard (extracts uniforms, creates match without `src`)

**Viewing:** Requires `match.src`, `match.left_uniforms`, `match.right_uniforms`, `match.params`

Matches without `src` cannot be viewed (backend processing pending).

## Storage

Implement `MatchStore` interface for alternative backends (database, S3, etc.). Update `app/main.py` dependency injection.

---

**Status:** Storage and API complete. Video stitching pending.
