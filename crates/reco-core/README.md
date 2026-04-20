# reco-core

GPU stitching engine. The no-I/O, no-domain-logic foundation of the Reco workspace.

## What it owns

- The push-based [`StitchCore`] — canonical frame-submission entry point for consumers.
- The wgpu 28 pipeline: fisheye undistort, two-plane L-shape composite, viewport crop, NV12 converter, RGBA readback.
- Traits: `Projection`, `CameraInput`, `PipelineStage`, `UnifiedDetector`, `FrameSource`, `Encoder`, `Director`, `DetectionSink`.
- Platform interop: `cuda_interop` (CUDA 12 FFI on Linux/Windows), `metal_interop` (CVPixelBuffer import on macOS/iOS), `vulkan_interop` (DMA-BUF import on Linux), `zero_copy` shared-texture sets.
- `PoseControl` — unified mouse-drag / wheel / keyboard -> yaw / pitch / FOV primitive shared by every consumer.
- `framesync::TimestampedIngestBuffer` — N-source timestamp pairing for live calibration.
- `yuv_stack_packer` — GPU-resident stacked-video pack / unpack (pairs with the replay path in reco-io).

## What it does NOT own

No file I/O, no FFmpeg, no GStreamer, no ONNX / TensorRT. Those live in `reco-io` and `reco-detect`. reco-core is usable headless — the GUI, OBS plugin, and cloud workers all plug in the same engine.

## Build

```bash
cargo build -p reco-core
cargo build -p reco-core --features profiling   # tracing-chrome instrumentation
cargo doc --no-deps -p reco-core --open
```

[`StitchCore`]: src/core/mod.rs
