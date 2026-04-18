# OBS Plugin Consumer API Friction

Friction points encountered while building `reco-obs` as a plugin
consumer of `reco-core` / `reco-io`. Active items at the top;
resolved items archived at the bottom with the PR that fixed them.

## Active

### A3. No OBS-level wgpu interop

**Impact**: Fundamental to OBS architecture, not a reco-core bug.

OBS uses its own graphics context (OpenGL / D3D11) and reco-core uses
wgpu. There's no interop path, so rendered frames must be copied
through CPU (GPU → staging → CPU → OBS texture). At 1080p that's
~8 MB per frame; at 60fps = ~480 MB/s memory bandwidth wasted.

Platform-specific solutions (DMA-BUF on Linux, shared D3D11 textures
on Windows) would need new interop code in reco-core. The
`GpuContext::from_device_queue()` method would help if OBS moved to
wgpu, which it hasn't.

Tracking here as a known limit; not actionable at the reco-core
level without a specific interop target.

## Resolved (archived)

- **R1. No RGBA readback helper on StitchPipeline**
  Resolved by Batch A (#223): `StitchRenderer::render_and_readback_rgba()`
  + `flush_rgba()` with triple-buffered staging, same pattern as
  `Nv12Converter`. Earlier ~40 lines of boilerplate in
  `reco-obs/source.rs` can be replaced.
- **R2. GpuContext::new() was async**
  Resolved by Batch H: `GpuContext::new_blocking()` added in reco-core.
  `reco-obs` dropped its direct `pollster` dep; plugin callbacks can
  now init the GPU synchronously without a runtime.
- **R3. YuvPlanes required tight packing, not stride-aware**
  Resolved by Batch I: `reco_core::pipeline::{FramePlaneView,
  StridedYuvPlanes}` added alongside the existing tight `YuvPlanes`.
  `StridedYuvPlanes::copy_into(&mut Vec<u8>)` repacks padded rows into
  a caller-owned buffer (tight-fast-path when `stride == width`) and
  returns a borrowed `YuvPlanes` ready for
  `StitchPipeline::render_to_target`. Buffer is reusable across frames
  to avoid per-frame allocation.
- **R4. No live camera input consumer helper**
  Resolved by Batch I: `reco_core::session::{LiveSessionConfig,
  LiveStitchSession}`. Bundles `StitchPipeline` + `RgbaReadback` with
  `submit_frame(left, right, yaw, pitch) -> Option<&[u8]>` for push-
  based compositor consumers (OBS, V4L2, WebRTC). `reco-obs/src/source.rs`
  migrated to it - net removal of ~30 lines of per-consumer plumbing.

## Notes on plugin status

reco-obs is still scaffolding PoC - it renders a green test pattern
to verify the OBS plugin hook-up works. Real frame ingestion from
OBS callbacks has not been implemented yet. When that happens, A1
and A4 will become blockers and likely motivate a Batch H in
reco-io.
