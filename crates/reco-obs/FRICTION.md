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

### A9. obs_source_get_frame only sees async video sources

**Impact**: High. Discovered 2026-04-18 while trying to route an OBS
Browser Source (VDO Ninja) into Tier 1 ingestion.

OBS has two source rendering models:

- **Async video sources** (Media Source / ffmpeg_source, V4L2 Device,
  NDI, Decklink): produce frames via `obs_source_output_video()`. OBS
  buffers them and our current ingestion path reads via
  `obs_source_get_frame()`.
- **Sync video sources** (Browser Source, Screen Capture, Window
  Capture, Game Capture, Video Composite): render directly via a
  `video_render` callback. OBS never buffers their output; there is
  no async frame queue for `obs_source_get_frame` to pull from.

Tier 1 of reco-obs uses `obs_source_get_frame`, so sync video sources
are silently invisible - `get_frame` returns NULL forever, our diag
log shows `submitted=0 missed_*` ticking up indefinitely with no
warning (it looks healthy but no frames arrive).

**Suggested direction** (reco-obs, not reco-core):

Add a render-to-texture fallback. In `render_and_readback`, after
`obs_source_get_frame` returns NULL for a slot, render the source to
our own `gs_texture_t`:

```c
obs_enter_graphics();
gs_texture_t *my_tex = /* pre-allocated per side */;
gs_viewport_push();
gs_projection_push();
gs_set_render_target(my_tex, NULL);
gs_ortho(0.0f, cx, 0.0f, cy, -100.0f, 100.0f);
gs_clear(GS_CLEAR_COLOR, &zero, 0.0f, 0);
obs_source_video_render(left_source);
gs_set_render_target(NULL, NULL);
gs_projection_pop();
gs_viewport_pop();
// download my_tex -> CPU staging -> BgraPlanes (Batch J)
obs_leave_graphics();
```

Needs new FFI bindings for `gs_set_render_target`,
`gs_viewport_push/pop`, `gs_projection_push/pop`, `gs_ortho`,
`gs_clear`, `obs_source_video_render`, and a GPU-to-CPU texture
readback path (staging buffer + `gs_stagesurface_map`). ~2-3 hours.

Once implemented, the existing Batch J BGRA path (R5) is what
consumes the readback, so no further reco-core work needed.

**Workaround for now**: use any async source (Media Source pointed at
a file, V4L2 camera, NDI input) while sync-source support is on
backlog. VDO Ninja can also expose a local v4l2loopback virtual
camera via `v4l2loopback-dkms` + an ffmpeg pipeline - that path is
async and works with Tier 1 I420 today.

### A10. LiveStitchSession doesn't expose director / detector hooks for AI panning

**Impact**: Medium. AI ball-tracking / auto-pan works in `reco-cli
stitch --model` and in the GUI via `reco_core::session::StitchSession`
(the file-stitching pull-based path), but `LiveStitchSession` (the
push path used by reco-obs and any future live consumer) has no API
to plug in a detector + director.

Manual panning is live in reco-obs via the yaw/pitch property sliders
(2026-04-18), which gives basic usability, but there's no path to
"follow the ball automatically" in the OBS plugin today.

**Suggested direction** (reco-core):

Mirror `StitchSession`'s detector/director hooks on
`LiveStitchSession`:

- `LiveStitchSession::set_detector(Box<dyn Detector>)`
- `LiveStitchSession::set_director(Box<dyn Director>)`
- In `submit_frame` / `submit_frame_bgra`: after receiving the frame,
  run detector -> update director -> use director's `ViewportPosition`
  for yaw/pitch instead of the caller-provided values (or blend
  director output with manual offsets).

Consumer side (reco-obs Batch L):

- Add `reco-autocam` + `reco-detect` as deps (plugin binary grows
  ~30-50 MB with ORT bundled).
- New properties: "Auto-pan model" (path to .onnx), "Tracking mode"
  (ball / field), "Detection interval" (every N frames).
- Run detection on a worker thread (not the video_tick thread) so
  inference latency doesn't stall frame submission at 30fps+.
- Expose tracking-quality feedback somehow (overlay? log? status
  indicator?) so users see why auto-pan landed somewhere.

Estimated ~4-6 hours minimum for a workable v1, plus meaningful
testing time to validate tracking on real camera pairs.

### A11. Visible upstream sources steal async frames from our poller

**Impact**: High (UX footgun). When an upstream Media Source is
visible in the scene (eye icon ON), OBS's scene renderer pulls its
async frames for compositing, and our subsequent
`obs_source_get_frame` polls return NULL every tick. Hiding the
upstream with the eye icon (while our `inc_showing` / `inc_active`
refs keep its decoder running) is what lets frames land in our poll.

Discovered 2026-04-18 after a long debugging session where picking
Media Sources appeared to do nothing. Toggling their visibility off
immediately produced `first stitched frame ready` and `submitted`
counters started growing at 30 fps.

**Suggested direction** (reco-obs, plugin-level): when the user
picks an upstream source in our properties, automatically hide its
scene item (if any) via `obs_sceneitem_set_visible(item, false)`.
Would need new FFI bindings for:

- `obs_source_get_scene` or equivalent (walk scene graph to find the
  scene item referencing our picked source)
- `obs_sceneitem_set_visible`
- Likely `obs_scene_enum_items` to locate the scene item

Undo on `dec_showing` / destroy would be nice but not required - the
user can toggle visibility back themselves if they move the source
to a different scene.

Alternate direction: surface a UI hint in the properties dialog
("Hide upstream sources in the scene panel for correct frame
capture"), or add a dedicated help string on each dropdown.

Documenting here so the next person hitting this has an immediate
answer; the pure-UX fix is small but OBS scene-graph walking is
fiddly so it's backlog, not urgent.

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
