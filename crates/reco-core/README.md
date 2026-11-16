# reco-core

Calibrated panoramic rendering with CPU and GPU executors, shared pose resolution and session orchestration.

## Module structure

```
src/
  core/        - StitchCore push API (submit frames, get rendered output)
  session/     - StitchSession pull API (sources, lookahead and output sinks)
  render/      - GPU pipeline, shared image-plane views and viewport dimensions
  gpu/         - GpuContext, NV12 converter, RGBA readback, YUV stack packer
  detect/      - Detection/tracking vocabulary (detector, tracker, panner traits)
  interop/     - Platform zero-copy (CUDA, Vulkan, Metal, D3D11, DMA-buf)
  projection/  - Coordinate math, coverage boundary, virtual camera
  lens/        - KB4 fisheye model and undistortion
  stitch/      - Projection-agnostic CPU/GPU stitch backends + agreement oracle
  calibration.rs - Calibration (lenses, projection, framing)
  source.rs      - FrameSource trait, FrameSet enum
  sink.rs        - OutputSink trait
  telemetry.rs   - Per-frame timing collection
```

## What it does NOT own

Source acquisition, codecs and network delivery live in `reco-io` or injected consumers. Concrete inference backends live in `reco-detect`. The core has no FFmpeg, GStreamer or ONNX/TensorRT dependency.

## Build

```bash
cargo build -p reco-core
cargo test -p reco-core --no-default-features  # CPU engine, session and rendering
cargo build -p reco-core --features profiling   # tracing spans; the application selects a subscriber
cargo doc --no-deps -p reco-core --open
```

Without the default `gpu` feature, construct `StitchSession::with_executor` with `Executor::Cpu`. Detection/tracking, coverage, immediate and lookahead scheduling, and NV12 sink delivery use the same session implementation. GPU constructors, imports and readback methods require `gpu`. The isolated CPU test lane in `.github/workflows/rust.yml` exercises this configuration and checks that its dependency graph excludes wgpu.

For fixed dual-camera cylindrical output, see [Cylindrical stereo panorama](PANORAMA.md).
