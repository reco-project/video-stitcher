# reco-core

GPU stitching engine. The no-I/O, no-domain-logic foundation of the Reco workspace.

## Module structure

```
src/
  core/        - StitchCore push API (submit frames, get rendered output)
  session/     - StitchSession pull API (batch processing with encoder wiring)
  render/      - GPU rendering pipeline, planes, viewport, scene geometry
  gpu/         - GpuContext, NV12 converter, RGBA readback, YUV stack packer
  detect/      - Detection/tracking vocabulary (detector, tracker, panner traits)
  interop/     - Platform zero-copy (CUDA, Vulkan, Metal, D3D11, DMA-buf)
  projection/  - Coordinate math, coverage boundary, virtual camera
  lens/        - KB4 fisheye model, undistortion, rig correction
  calibration.rs - MatchCalibration (stereo camera parameters)
  source.rs      - FrameSource trait, StereoFrame enum
  encoder.rs     - Encoder trait
  telemetry.rs   - Per-frame timing collection
```

## What it does NOT own

No file I/O, no FFmpeg, no GStreamer, no ONNX/TensorRT. Those live in `reco-io` and `reco-detect`. reco-core is usable headless.

## Build

```bash
cargo build -p reco-core
cargo build -p reco-core --features profiling   # tracing-chrome instrumentation
cargo doc --no-deps -p reco-core --open
```
