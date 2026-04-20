# reco-control

Transport-agnostic operator-intent vocabulary.

## Why it exists

Every operator surface — keyboard shortcuts, GoPro USB buttons, a mobile companion app, a WebSocket bridge — translates raw input into the same set of intents: pan a bit, zoom in, toggle constrained-look, kick off capture, switch detection model. Duplicating that translation three times is the friction reco-control removes.

## What it owns

- **`ControlIntent` enum** — `Hotkey`, `Pose`, `Quality`, `Capture`, `ModelSelect`, plus `Extension(Box<dyn Any>)` for forward-compat.
- **`ControlTransport` trait** — any transport (keyboard, USB, mobile, WebSocket) is a stream of `ControlIntent`.
- **`KeyboardTransport`** (`keyboard` feature, default) — the one shipping transport today: reco-obs and reco-gui key events become `ControlIntent`s, dispatched to `PoseControl` / `StitchCore` / encoder.

## Placeholder transports

`gopro`, `mobile`, `websocket` features each gate a module whose functions are `todo!()`. They exist so the feature-combo CI matrix exercises the gates and so future work has an obvious target path.

## Build

```bash
cargo build -p reco-control                                    # keyboard only
cargo build -p reco-control --features keyboard,gopro,mobile,websocket  # all placeholders
```
