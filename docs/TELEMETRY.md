# Telemetry

Optional, anonymous, event-based usage stats stored locally.

## Principles

- **Opt-in** (disabled by default)
- **Event-based** (not continuous tracking)
- **No personal data** (no filenames, paths, match names)
- **Local-first** (events written to local JSONL file)
- **Optional system info** (OS/CPU/RAM/GPU) behind separate toggle

## Storage

Location: `app.getPath('userData')/telemetry/events.jsonl`

Open from Settings: **Telemetry → Open folder**

## Events

- `app_open`
- `match_created`
- `processing_start` / `processing_cancel` / `processing_success` / `processing_error`
- `context` (OS + hardware snapshot, if enabled)

## Not Collected

- Video filenames or paths
- Match names
- Lens profile names
- Full error logs

## Uploading

Automatic upload to `https://telemetry.reco-project.org/telemetry`

When telemetry is enabled:
- Events are uploaded 2 seconds after app start
- Then uploaded every 30 seconds in the background (configurable)
- Best-effort upload on app close (3s timeout)

You can also manually trigger upload from **Settings → Telemetry → Upload now**.

The uploader:
- Reads only telemetry files (not matches, videos, logs)
- Posts batches to `POST /telemetry`
- Tracks upload progress to avoid re-sending
- Detects hardware changes and includes `hardware_changed` flag in batch
- Only sends `context` events when hardware actually changes
- Retries 3 times with exponential backoff
