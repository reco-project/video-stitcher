# reco-control

Operator pose-control state machine and intent dispatch.

## Why it exists

CLI preview and GUI share drag/zoom translation and coverage feedback. Native key bindings and presentation scheduling stay in their consumers; geometry and calibration validation stay in core.

## What it owns

- **`PoseControl`** - the shared mouse / drag / wheel / keyboard -> yaw / pitch / FOV state machine. Stores render-space poses and accepts direct input through `StitchCore::constrain_render_pose`, without tracking inertia. Convert world-space reset directions with `StitchCore::world_to_render_pose` before giving them to the control; do not orient the accepted pose again when rendering.
- **`ControlIntent`** - signed pan/FOV deltas, absolute FOV, configured keyboard zoom and reset. `PoseControl::apply_intent` applies these directly. Coverage toggles use core's explicit setting.

`PoseControl::new` rejects invalid zoom bounds, sensitivity and resting poses. Input methods ignore non-finite values or overflowing results atomically. Controls retain both pending and accepted poses so input can accumulate before coverage fitting without accumulating blocked motion afterward. The external tests in `tests/pose_input.rs` exercise these contracts.

Use `constrain_preview` for direct interactive editing and preview recording: core prepares coverage asynchronously and temporarily permits black edges. `constrain` requires coverage before accepting a pose. Offline export and automatic tracking retain strict coverage resolution. Core's preview invalidation and strict-resolution regressions live in `core/coverage_motion/tests.rs`.

Commands serialize as a flat action, for example `{"action":"delta_yaw_rad","value":0.05}` or `{"action":"zoom_in"}`. They do not carry native key codes or calibration edits.

## Optional: GoPro device helper

The `gopro` feature (off by default) gates an OpenGoPro HTTP helper to drive a GoPro as a command target (start/stop recording, sync settings, query status). It enables Reqwest and the helper's logging, error and JSON dependencies. Viewport controls do not require a GPU or HTTP client.

## Build

```bash
cargo build -p reco-control                  # pose control + intent dispatch
cargo build -p reco-control --features gopro # adds the GoPro device helper
```
