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

### A12. No keyboard / controller mapping for pan-zoom

**Impact**: Medium. Drag-to-pan and scroll-to-zoom work in OBS
"Interact" mode, but that requires opening a separate interact
window every session. Live operators using a keyboard, or game-pad
operators with a PS5/Xbox controller, have no fast path.

**Suggested direction**:

- **Keyboard (short-term)**: use OBS's `obs_hotkey_register_source`
  API to declare hotkeys on the Reco source (yaw_left,
  yaw_right, pitch_up, pitch_down, zoom_in, zoom_out, reset).
  Users bind them in File -> Settings -> Hotkeys. Requires new FFI
  bindings for the hotkey registration API; ~1 hour of work.
- **Game pad (medium-term)**: OBS does not expose controller input
  directly. Options: (a) a sidecar process using SDL3 that reads
  controller axes and sends them to the plugin via a Unix socket /
  named pipe; (b) use a general-purpose virtual-joystick-to-keyboard
  utility (QJoyPad, antimicro) and bind to the hotkeys from path
  (a). Plan to document the SDL3 sidecar approach in a separate
  PR so game-pad support is reproducible.

### A13. No "constrained look" toggle in the OBS plugin

**Impact**: Medium. `reco_core::projection::CoverageBoundary::safe_clamp`
exists (it's used by the GUI's constrained-look toggle, Tier 2b) but
the OBS plugin applies raw yaw/pitch straight to the renderer. At
extreme pan angles the view shows black borders outside the stitched
coverage.

**Suggested direction**: add a `constrained_look: bool` property to
reco-obs, default true. When set, `render_and_readback` should clamp
yaw/pitch via `CoverageBoundary::safe_clamp(yaw, pitch, fov, aspect,
rig_tilt)` before passing to `submit_frame`. The calibration file
already carries the boundary definition - just need to expose the
clamp function through `LiveStitchSession` or plumb it at the
reco-obs layer. Small: ~30 min including property wiring.

### A14. No live calibration workflow for non-file sources

**Impact**: High for production live use. The current calibration
path (`reco_calibrate::video::calibrate_videos`) requires two video
*files* on disk. For live OBS sources (V4L2 cameras, NDI, WebRTC)
there's no equivalent that captures from a live source to produce
calibration JSON.

**Suggested direction** (reco-calibrate + reco-obs):

- reco-calibrate already exposes the lower-level
  `calibrate(&gpu, &frame_pairs)` that takes in-memory frame pairs -
  the algorithmic side is ready.
- New reco-io helper: a "capture N frame pairs from two live
  FrameSources" utility that wraps the `StereoFrame` delivery and
  hands back `Vec<(YuvFrame, YuvFrame)>`.
- reco-obs UI: a "Calibrate from current sources" button in the
  properties that grabs ~30 frame pairs from the picked async
  sources via our existing `obs_source_get_frame` loop, feeds to
  `calibrate()`, writes the JSON next to the configured calibration
  path, then reloads.

Estimated 3-4 hours of work across the two crates.

### A15. OBS Interact window is a poor primary UX

**Impact**: Medium. Users expect to pan the panorama in the main
scene view, not a separate interact window that must be kept open.
OBS's interaction model is designed for web source clicks, not
continuous compositor control.

**Suggested direction**: the nicer shape is an embedded dock panel
(like Transition Override, Scene Filters) that gives a dedicated
control surface: yaw / pitch sliders with live preview, FOV slider,
constrained-look toggle, detector-status LED, recenter button. OBS
exposes dock registration via `obs_frontend_add_dock_by_id` (needs
qt bindings, typically via a small C++ shim). This is substantial
plumbing work but dramatically better UX than per-property sliders +
Interact mode.

Alternate short-term: offer a full-screen projector ("Windowed
Projector (Source)" in OBS, already built in) that shows just our
panorama, and keep driving the pose via hotkeys (A12).

### A16. No built-in scene / replay mode

**Impact**: Low to medium. Users want to switch scenes to a playback
view and enable replay instantly - replay isn't a plugin feature
today, and scene switching is regular OBS behavior but OBS doesn't
know our plugin's internal ring-buffer state.

**Suggested direction** (two parts):

- **Reco-core**: add an optional "rolling buffer" helper on
  `LiveStitchSession` - a ring of the last N stitched RGBA frames
  with timestamps, configurable duration (e.g. "keep last 30s").
  Memory cost: 1920*1080*4 * 30fps * 30s = ~7 GB, so needs to be
  opt-in.
- **Reco-obs**: expose a "Replay mode" toggle + hotkey. When
  activated, the source's `submit_frame` path switches from "stitch
  current" to "replay from buffer", advancing the buffer pointer
  based on wall time (or scrubbable via hotkey). Scene switch
  orthogonal to this - the user uses standard OBS scene transitions
  and the source just plays back from the buffer.

Estimated 4-6 hours plus memory-budget discussion.

### A19. Replay recording has no tile-downscale / quality override

**Impact**: Medium. Surfaced 2026-04-19 during first in-OBS test. Replay tiles are written at the exact input resolution (the stacked composite is `width × 2*height` for N=2), with no way to request a smaller recording. Consequences:

- Two 5K cameras produce a 5120×5760 composite. libx264 software encode at this size struggles to keep up on modest hardware, the file grows at several GB/min, and disk I/O may stall the OBS video_tick.
- A user who wants replay for review purposes (not archival) has no way to trade quality for size.

Today we emit a `warn!` when the input exceeds 4K per tile so the user isn't silently blindsided, but the proper fix is a UI-visible `Replay scale` dropdown (Full / 1080p / 720p) and a `Replay quality` dropdown (Fast / Balanced / High). Both map onto `StackedEncoderConfig.inner.{resolution, quality, crf}` which already exist.

**Suggested direction**: add two dropdowns in the reco-obs properties UI and thread them through `StackedEncoderConfig`. A GPU-resident downscale (once the future wgpu pack path lands) would avoid the CPU hit entirely.

### A20. Replay records concurrently with OBS recording/streaming without warning

**Impact**: Low. Surfaced 2026-04-19 during first in-OBS test. Toggling "Record replay" while OBS itself is recording or streaming results in two parallel encode paths (the OBS encoder and our stacked encoder) that have nothing to do with each other. Not harmful, but surprising to a user who expects "OBS is recording" to imply "reco-obs is also saving the panorama via OBS".

**Suggested direction**: one-shot log info at replay start time noting whether OBS is currently recording/streaming, and possibly a properties-UI hint text near the toggle. Cheap fix; mostly a documentation / transparency concern.

### A21. Live Matroska replay is hard to scrub in general-purpose players

**Impact**: Low. Surfaced 2026-04-19 during first in-OBS test. A Matroska file that's still being written has unstable duration metadata - VLC, mpv, and OBS itself treat the duration as `N/A` and won't scrub backwards past a few seconds. This is a generic streaming-container limitation, not a reco bug: finalized files (after the user unticks the toggle) scrub normally.

**Suggested direction**: either accept it (replay is replay, not scrub), or write a parallel "chunked" file (rotate every N seconds into a new .mkv) so recent chunks are always complete and seekable. Low priority - real-time replay in a stitcher UI is future work anyway.

### A17. No AI auto-pan access in the OBS plugin

See also A10 (deeper architectural notes). Short version: reco-obs
needs a way to plug in `reco_autocam::Director` + `reco_detect`
execution. Requires `LiveStitchSession::set_detector` /
`set_director` hooks in reco-core, then adding
`reco-autocam` + `reco-detect` as reco-obs deps with a worker-thread
inference scheduler so the 30fps tick doesn't stall on ORT
inference. User has flagged AI access as important; worth
prioritizing once the friction backlog is worked.

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
- **A18. No push-API entry point for disk-based replay recording**
  Resolved 2026-04-19 alongside M6.5 item 3 wiring. Added
  `reco_core::core::StackedReplayRecorder` trait and
  `StitchSession::{set,clear,flush}_stacked_recorder` on the push
  API. reco-io provides `stacked_video::replay::session_recorder(path,
  config, width, height)` that returns a `Box<dyn StackedReplayRecorder>`
  ready for `session.set_stacked_recorder(...)`. Mirrors the pull-side
  `StitchJob::with_replay_recording` builder: one line for the
  consumer, the session owns the per-frame tap + encoder lifecycle.
  Integration test `session_recorder_records_planes` in
  `reco-io/tests/stacked_video_roundtrip.rs` verifies a round trip
  through the trait path.

## Notes on plugin status

Tier 1 (2026-04-18): real dual-source frame ingestion landed. The
plugin now exposes two source pickers in its properties UI (left /
right), resolves them via `obs_get_source_by_name`, and polls
`obs_source_get_frame` every `video_tick`. I420 input is routed
through `StridedYuvPlanes::copy_into` (Batch I) into
`LiveStitchSession::submit_frame`; other formats (NV12, YUY2,
UYVY, packed RGB) are logged once and skipped. Tier 2 target is
NV12 + temporal pairing (blocked on A5).
