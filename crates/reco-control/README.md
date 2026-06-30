# reco-control

Operator pose-control state machine and intent dispatch.

## Why it exists

The 2026-04-18 deep review found consumer input paths duplicated three ways across reco-cli/preview, reco-gui, and reco-obs: different key mappings, different pan sensitivity, different units. This crate removes that duplication so every surface drives the viewport the same way.

## What it owns

- **`PoseControl`** - the unified mouse / drag / wheel / keyboard -> yaw / pitch / FOV state machine. Stores the pose in world space and eases `current` toward `target` each tick. The single source of truth for viewport panning across consumers. The rig tilt/roll correction that keeps the horizon level is applied at the render site by `reco_core::render::stitch_renderer::StitchRenderer::orient_pose`, not here.
- **`ControlIntent`** - a small intent vocabulary (`Hotkey`, `Pose`) for surfaces that speak a higher level than raw mouse deltas.
- **`IntentTranslator`** - dispatches a `ControlIntent` stream onto a borrowed `PoseControl`.

## Optional: GoPro device helper

The `gopro` feature (off by default) gates an OpenGoPro HTTP helper to drive a GoPro as a command target (start/stop recording, sync settings, query status). It is a device helper, not part of the pose-control core, and pulls in `reqwest`/`tokio`.

## Build

```bash
cargo build -p reco-control                  # pose control + intent dispatch
cargo build -p reco-control --features gopro # adds the GoPro device helper
```
