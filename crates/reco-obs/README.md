# reco-obs

OBS Studio source plugin. Embeds the reco stitching engine inside an OBS source so live producers get a panoramic feed without a separate pipeline.

## What it does

- **Async-frame ingestion** — OBS delivers raw frames via `obs_source_get_frame`; reco-obs pushes them into `StitchCore::submit_frame_bgra` / `submit_frame_yuv`.
- **Interactive pan/zoom** — mouse drag + wheel + keyboard translate to `ControlIntent`s → `PoseControl` → renderer.
- **Live calibration** (M6 A14) — "Calibrate from current sources" action drives `reco_calibrate::calibrate_from_live` off the plugin's dual-source buffer.
- **Replay recording** — opt-in stacked-video recording to disk (the M6.5 A18 push-API path), independent of OBS Record/Stream.

Tier 1 (PR #267) shipped 2026-04-18 with BGRA input, interactive pan/zoom, and async-frame ingestion. Per-session open friction is tracked in [FRICTION.md](FRICTION.md).

## Build

```bash
cargo build --release -p reco-obs
```

Output: `libreco_obs.so` (Linux), `reco_obs.dll` (Windows), `libreco_obs.dylib` (macOS). Drop into OBS's `obs-plugins/64bit/` directory.

## Not yet shipped

- Non-BGRA input formats (see FRICTION for the NV12 zero-copy path)
- AI auto-pan from OBS controls
- Sync-source support (Browser / Screen / WebRTC)

## Architecture

reco-obs is intentionally a thin plugin shell around `StitchCore`. Substantive logic stays in reco-core so the OBS path, GUI path, and CLI path all converge on the same engine — the plugin is a real proof of API modularity, not a fork.
