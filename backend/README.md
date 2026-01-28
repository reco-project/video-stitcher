# Backend API

FastAPI backend for lens profile management, match orchestration, and video processing.

## Quick Start

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

- **Server:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Tests:** `pytest tests/ -v`

## Requirements

- Python 3.10+
- FFmpeg (auto-downloaded via `npm run setup`)

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration settings
│   ├── models/              # Pydantic data models
│   ├── repositories/        # Data storage interfaces
│   ├── routers/             # API endpoint handlers
│   ├── services/            # Business logic (transcoding, matching)
│   └── utils/               # Helper utilities
├── data/
│   ├── lens_profiles/       # Lens calibration profiles
│   ├── matches/             # Match project storage
│   ├── temp/                # Temporary processing files
│   └── logs/                # Application logs
└── tests/                   # Pytest test suite
```

## API Endpoints

### Health

- `GET /` — Root health check
- `GET /api/health` — API health status

### Profiles

- `GET /api/profiles` — List all lens profiles
- `GET /api/profiles/{id}` — Get profile by ID
- `POST /api/profiles` — Create new profile
- `PUT /api/profiles/{id}` — Update profile
- `DELETE /api/profiles/{id}` — Delete profile

### Matches

- `GET /api/matches` — List all matches
- `GET /api/matches/{id}` — Get match by ID
- `POST /api/matches` — Create new match
- `PUT /api/matches/{id}` — Update match
- `DELETE /api/matches/{id}` — Delete match

### Processing

- `POST /api/transcode` — Stack and transcode videos
- `POST /api/process-with-frames` — Calibrate using warped frames

## Data Storage

**Profiles:** `data/lens_profiles/{brand}/{model}/{id}.json`  
**Matches:** `data/matches/{match-id}.json`

Storage backends implement abstract interfaces (`MatchStore`, `LensProfileStore`), making it easy to swap file-based storage for databases.

## Development

### Adding New Endpoints

1. Create route handler in `app/routers/`
2. Import and register router in `app/main.py`
3. Add tests in `tests/`

### Running Tests

```bash
pytest tests/ -v
pytest tests/test_profiles_api.py -v  # Specific test file
```

## Troubleshooting

### Port already in use

```bash
# Use a different port
python -m app.main --port 8001
```

### Import errors

```bash
# Ensure you're in backend/ and venv is activated
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

---

**License:** AGPL-3.0  
**Lens profiles data:** CC0 1.0 Universal (derived from [Gyroflow](https://github.com/gyroflow/gyroflow))
