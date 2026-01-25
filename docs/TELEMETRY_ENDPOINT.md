# Telemetry Endpoint

Simple "dumb inbox" backend that accepts anonymous opt-in telemetry.

**Deployed at**: `https://telemetry.reco-project.org/telemetry`

## Endpoint

`POST /telemetry`

**Request**:
```json
{
  "schema_version": 1,
  "client_id": "uuid-v4",
  "app": {
    "name": "video-stitcher",
    "version": "0.0.0",
    "environment": "dev" | "production"
  },
  "sent_at": "2025-01-01T00:00:00.000Z",
  "batch_id": "uuid-v4",
  "hardware_changed": true,
  "events": [...]
}
```

**Response**:
- `200 OK` for accepted batches
- `400` for invalid payloads
- `429` for rate limiting (100 req/min per client)

## Validation

- JSON is well-formed
- `schema_version` is `1`
- `events` is an array
- Each event has: `schema_version`, `ts`, `name`, `client_id`
- `name` is allowlisted
- `props` is `null` or small JSON object
