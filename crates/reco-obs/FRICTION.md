# OBS Plugin Consumer API Friction

Friction points encountered while building `reco-obs` as a plugin
consumer of `reco-core` / `reco-io`. Active items at the top;
resolved items archived at the bottom with the PR that fixed them.

## Active

### A5. No temporal frame pairing helper in reco-core / reco-io

**Impact**: Medium for any dual-source live consumer (OBS now, V4L2
later). Hit 2026-04-18 while wiring OBS Tier 1 ingestion.

`obs_source_get_frame` returns the "current" async frame for a given
source: whatever the upstream producer last handed OBS. With two
independent cameras feeding two independent OBS sources, the two
calls return frames whose timestamps may disagree by one or more
frame intervals (upstream producers run on separate threads with
their own pacing).

Tier 1 of reco-obs just polls both sides every `video_tick` and
submits the pair unconditionally - if drift grows, we start stitching
temporally mismatched frames with no warning.

A proper solution is a reusable ring-buffer pairing helper in reco-io
(or reco-core) that:
- Takes `(timestamp, StereoSide, FrameBufferHandle)` submissions from
  each side,
- Emits `(left, right)` pairs with the closest timestamps when both
  sides have recent frames,
- Drops older unpaired frames,
- Surfaces a "last pairing delta" so consumers can warn when drift
  exceeds a threshold.

The same helper would be reused by a future WebRTC ingest or a live
V4L2 consumer. Living in a consumer crate means every new consumer
reinvents it.

### A6. No obs_source_get_output_flags binding to filter source picker

**Impact**: Low (UX papercut). The reco-obs source-picker dropdowns
currently show *every* OBS source - scenes, audio-only inputs,
transitions, filters. Picking a scene just causes
`obs_source_get_frame` to return null, which we handle silently,
but it's a bad UX.

Properly filtering to async-video-capable sources (inputs that set
`OBS_SOURCE_ASYNC_VIDEO`) requires a new FFI binding for
`obs_source_get_output_flags`. Noted here so we don't lose track;
the fix is 5 lines of FFI + a conditional in the enumeration
callback, but it's OBS-FFI scope only.

### A7. No auto-resize when input frame dimensions don't match

**Impact**: Low (UX papercut). When an upstream source delivers
frames whose dimensions differ from the plugin's `input_width /
input_height` properties, we log once and skip all submissions
until the user edits the properties. A nicer flow would be to
reinitialize the `LiveStitchSession` on the first received frame
whose dimensions differ, picking up the new size automatically.

Not actionable at the reco-core level - `LiveStitchSession::new`
already supports being rebuilt - but the reco-obs callback plumbing
has to detect the size change and reset cached state (repack
buffers, OBS texture, etc.). Tier 2 item.

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
- **R5. reco-core only accepted YUV inputs (blocked Browser Source / WebRTC)**
  Resolved 2026-04-18 by Batch J: `InputFormat::Bgra` variant in
  `reco_core::renderer`, `BgraPlanes` input struct + `render_to_target_bgra`
  / `LiveStitchSession::submit_frame_bgra`. Shader branches on flags.y==2
  to sample a single `Rgba8Unorm` texture and skip YUV conversion. Bind
  group layout unchanged (1x1 dummy textures for u/v in BGRA mode).
  reco-obs now exposes an "Input format" dropdown (I420 / BGRA); Browser
  Source, screen capture and WebRTC feeds that deliver BGRA/BGRX/RGBA
  are accepted, swizzle-to-RGBA happens once per frame into a cached
  buffer.

## Notes on plugin status

Tier 1 (2026-04-18): real dual-source frame ingestion landed. The
plugin now exposes two source pickers in its properties UI (left /
right), resolves them via `obs_get_source_by_name`, and polls
`obs_source_get_frame` every `video_tick`. I420 input is routed
through `StridedYuvPlanes::copy_into` (Batch I) into
`LiveStitchSession::submit_frame`; other formats (NV12, YUY2,
UYVY, packed RGB) are logged once and skipped. Tier 2 target is
NV12 + temporal pairing (blocked on A5).
