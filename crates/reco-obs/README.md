# reco-obs

OBS Studio source plugin. Embeds the reco stitching engine inside an OBS source so live producers get a panoramic feed without a separate pipeline.

## What it does

- **Async-frame ingestion**: OBS delivers raw frames via `obs_source_get_frame`; reco-obs converts packed input to RGBA and submits through `StitchCore::submit_frame_rgba_at_pose` or `submit_frame_yuv420p_at_pose`.
- **Interactive pan/zoom** — mouse drag + wheel + keyboard translate to `ControlIntent`s → `PoseControl` → renderer.
- **Live calibration**: "Calibrate from current sources" drives `reco_calibrate::calibrate_from_live` from the plugin's dual-source buffer.
- **Replay recording**: opt-in stacked-video recording to disk, independent of OBS Record/Stream.

Open integration gaps are tracked in [FRICTION.md](FRICTION.md).

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
