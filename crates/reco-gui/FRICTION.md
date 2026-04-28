# GUI Consumer API Friction

Friction points encountered while building the Slint GUI consumer of
reco-core and reco-io. Active items are at the top; resolved items
archived at the bottom with the PR that fixed them, so we can tell
"this used to be painful" without re-litigating solved problems.

## Active

### ~~A3. Preview + export CUDA context race~~ RESOLVED

Resolved in v0.5.0: preview rendering is paused during export via
`is_exporting()` guard in `vsync_render_tick`. Playback is paused
on export start and rendering resumes on completion.

### ~~A4. No clean pipeline unload~~ RESOLVED

Resolved in v0.5.0: `reset_pipeline()` on `AppState` drops bridge,
resets playback, pose, calibration baselines, and pending seeks.
Called from `unload_pipeline()` and all file-swap paths.

**Impact**: Medium. Manifested as a real state-corruption bug in Tier 3.

Once a `PreviewBridge` / `StitchPipeline` is built, the only way to
swap to different source files is to drop the whole struct and rebuild
from scratch. The GUI had to add its own `unload_pipeline()` helper
that sets `bridge = None`, `calibration = None`, `playback = new()`,
etc. - a half-dozen-field reset that reco-core doesn't own.

Without this, swapping a file while a pipeline is live left the preview
rendering the OLD source while `AppState.left_path/right_path` pointed
elsewhere - and export read the new paths and produced garbage.

**Suggested direction**: either a reco-core "session state" helper that
bundles pipeline + source + calibration with a single `unload()`, or
document the pattern in reco-io's StitchJob docs so the next consumer
doesn't reinvent it.

### A5-residual. No intra-step heartbeat during very slow steps

**Impact**: Low after the A5 primary fix (R22). The step label now
matches the work (DetectingProfiles step) and telemetry isn't parsed
twice, but individual steps can still run 20-60s silently: AKAZE
feature detection on dense scenes, the optimizer on 100+ frame pairs,
audio PCM extraction on long clips. A consumer can't distinguish
"making progress" from "hung" inside a single step.

**Suggested direction**: a heartbeat event every 2s during any step
that exceeds a short threshold, so consumers can show a spinner or
elapsed-time hint without needing intra-step instrumentation in every
slow routine.

### A7. render_to_target() still returns CommandBuffer

**Impact**: Low. The Batch A `RenderTarget` / `RenderOutcome` enums
simplified the surface-vs-internal asymmetry, but the lower-level
`render_to_target()` still returns a `wgpu::CommandBuffer` the caller
must submit. For the zero-copy Slint path this is fine (we render
directly to a Slint-owned texture via `render_yuv`); future consumers
that want to batch our render with their own copy commands already
work with the current API. Leaving this as a note for anyone hitting
the old asymmetry - the Batch A outcome enum was the right abstraction.

### N5. ExportOutcome::Failed loses structured StitchError variants

**Impact**: Medium. Hit while wiring the export-error toast/dialog.

The cross-thread channel from the export worker back to the UI
flattens `StitchError` into `ExportOutcome::Failed(String)`. The UI
can show "something broke" but can't branch on the variant (e.g.
retry-friendly FFmpegDecode vs hard-fail BadCalibration). Fixing it
requires `StitchError: Clone + Send + Sync` so a rich variant can
cross the channel boundary.

Plan disposition: K3 / E5 (cross-thread error propagation cleanup).

### N6. Calibration thread loses reco_calibrate::Error variants

**Impact**: Medium. Sibling of N5 for the calibrate path.

The calibration worker surfaces failures as a flat string via
`CalibrationProgress::Failed`. Low-level context (AKAZE threshold
too strict, RANSAC under-8 pairs, telemetry parse failure) gets
collapsed. Users see "calibration failed" without actionable
guidance. Same fix vector as N5: Clone+Send+Sync on the error enum.

### N7. FfmpegFileSource opens the whole container just to read total_frames

**Impact**: Low. Hit in the export dialog init.

The GUI wants total_frames up front to render the progress bar
scale. Today it constructs a full `FfmpegFileSource`, reads
`.total_frames()`, and drops it. That's two muxer opens per export
(init + real read). A lightweight `reco_io::probe_duration(path)`
that only opens the muxer, reads stream metadata, and closes would
remove one open.

Plan disposition: K6 (E6-adjacent).

### N8. Slint wgpu 28 rendering notifier does not expose AdapterInfo

**Impact**: Low (diagnostics). Hit while writing the "GPU info" footer.

`reco-core::GpuContext::adapter_info()` is available everywhere
_except_ inside the Slint rendering notifier closure, because Slint
1.15 hands consumers a `wgpu::Device` without the parent `Adapter`.
The GUI currently falls back to "unknown GPU" in its diagnostics
panel. Upstream Slint feature request, blocked on them exposing
the adapter reference.

Plan disposition: K8 (Slint upstream; out of scope for this branch).

### ~~N10. setup_autocam positional-11-arg signature~~ RESOLVED

Resolved: the 11-arg function was deleted. `setup_autocam(&mut session, &config, fps)` with `AutocamConfig` is now the only API. All consumers migrated. `FieldPannerConfig` exposes all tuning parameters with safe defaults.

### N11. Codec / Quality string parsing duplicated across consumers

**Impact**: Low. Hit when a new codec variant lands.

Each consumer (reco-cli `stitch`/`camera`/`preview`, reco-gui export
dropdown) hand-rolls a `match` for `"h264" | "x264" | ...` to build
`reco_io::Codec`. Four copies, and they drift on obscure names. A
`Codec::from_str_loose` / `Quality::from_str_loose` helper in reco-io
would collapse them into one.

Plan disposition: K6 / E9.

### N12. YuvPlanes hand-constructed in every render call site

**Impact**: Low. API ergonomics.

To call `StitchCore::submit_frame_yuv`, each consumer builds
`YuvPlanes { y: &[..], u: &[..], v: &[..], width, height }` from
whatever shape its source delivers. That's 6 field accesses and an
aspect-ratio compute spread across the GUI playback path and the CLI
stitch path. A `YuvData::planes()` method that returns
`YuvPlanes<'_>` directly would remove the per-call boilerplate.

Plan disposition: K6 / E7.

### N13. StereoFrame::into_yuv420p is missing

**Impact**: Low. Sibling of N12.

`StereoFrame` has `Yuv420p`, `Nv12`, and `Bgra` variants but no
accessor to produce a `(YuvPlanes, YuvPlanes)` tuple from a Yuv420p
variant. Consumers match on the variant themselves and destructure
the inner pair — which is fine once, awkward when five sites do it.
`as_yuv420p()` returning `Option<(YuvPlanes, YuvPlanes)>` would
centralize the destructure.

### ~~N14. StitchJob single on_session callback~~ RESOLVED

Resolved: `on_session` now pushes to a `Vec<SessionCallback>`.
Multiple hooks compose cleanly. A trait-based hook system would
be preferred for complex future composition.

### N15. PoseControl requires manual rig_tilt threading

**Impact**: High. Caused a constrained-look regression in the GUI.

Consumers must pass rig_tilt to both `clamp_via_coverage` and
`render_pose` on every tick, and must remember to use `render_pose`
instead of `current_pose` for the renderer. The CLI got this right;
the GUI got it wrong.

The renderer should own the pose state machine so rig_tilt, coverage
clamping, and render compensation are automatic. Consumers would call
`renderer.set_constrained_look(bool)` and `renderer.tick()` instead
of manually coordinating PoseControl + coverage + rig_tilt.

Plan disposition: K6 / E7. Lands with N12.

### N16. Recording lags preview and drops frames during panning

**Impact**: High. User-visible stutter during preview recording.

The GUI's recording path runs `render_and_readback_nv12` on the UI
thread. The NV12 readback stalls the GPU pipeline (triple-buffer
latency), blocking the Slint compositor. Even with async encoder
threads, the readback itself is synchronous and takes ~5-8ms per
frame on the UI thread.

**Current mitigation**: NV12 render every frame for the encoder,
display preview only every 5th frame. This prioritizes encoding
smoothness over preview responsiveness.

**Proper fix**: Recording should use the export pipeline (`StitchSession`
on a background thread with its own decoder), matching the CLI's
`run_immediate` loop. The preview continues independently at its own
rate. This is the Phase 11 recording architecture - "Rec" becomes
a lightweight export job. Requires sharing the calibration and viewport
state but NOT the GPU pipeline between preview and recording threads.

**Alternative**: Accept the preview-loop model but move the NV12
readback to a separate GPU queue or use compute-shader NV12 without
readback (encode from GPU-resident data via VAAPI/NVENC hardware
encoder path).

### N17. No ROI visualization or editing in GUI

**Impact**: Medium. The calibration produces `field_roi` (bounding box
of the playing field in panoramic coordinates) but the GUI never shows
or lets users adjust it. The ROI gates which detections the AI tracker
considers - a wrong ROI means the tracker ignores the ball or tracks
spectators. Users have no way to verify the ROI is correct without
looking at raw calibration JSON.

**Suggested direction**: overlay the ROI rectangle on the preview
(transparent colored border). An edit mode where users can drag the
ROI corners. Requires `panorama_to_screen` projection (inverse of the
stitch projection) to map ROI coordinates to screen pixels.

### N18. Slint slider loses pointer tracking inside ScrollView

**Impact**: Low. Reproducible with the correction slider in the Lens
section. When the user drags a Slint Slider inside a ScrollView, the
ScrollView can capture the pointer and the slider stops tracking.
The slider "drops" from the cursor and blocks until re-clicked.

Likely a Slint bug or interaction between ScrollView's pointer
handling and Slider's drag gesture. Workaround: move critical sliders
outside the ScrollView, or use a custom drag-based control.

## Resolved (archived)

Items pruned from Active once the reco-core / reco-io API added what
we asked for. Kept as an audit trail of which consumer friction items
drove which upstream changes.

- **R1. wgpu version mismatch blocking zero-copy rendering**
  Resolved 2026-04-16 by downgrading reco-core to wgpu 28 and using
  Slint 1.15's `unstable-wgpu-28` feature. `PreviewBridge` now shares
  Slint's device/queue via `GpuContext::from_device_queue()`.
- **R2. No RGBA readback API on StitchRenderer**
  Resolved by Batch A (#223): `StitchRenderer::render_and_readback_rgba()`
  + `flush_rgba()` with triple-buffered staging.
- **R3. StitchRenderer hardcoded InputFormat::Yuv420p**
  Resolved by Batch A (#223): `StitchRenderer::new` now takes an
  `input_format: InputFormat` parameter.
- **R4. resize() didn't notify consumers of new dimensions**
  Resolved by Batch C: `StitchPipeline::resize()` returns
  `Option<(u32, u32)>`.
- **R5. FrameSource::try_next_frame() EOF ambiguity**
  Resolved by Batch A: `FrameSource::is_exhausted()` trait method.
- **R6. render_to_view vs render_to_target asymmetry**
  Resolved by Batch A: `RenderTarget` enum + `RenderOutcome`.
- **R7. CalibrationResult didn't expose detected lens profile**
  Resolved by Batch B (#224): `CalibrationResult.left_lens_profile` /
  `right_lens_profile` carry `LensProfileInfo`.
- **R8. No API to list available lens profiles**
  Resolved by Batch B: `LensDatabase::iter_profiles()` +
  `candidates(width, height)` returning `LensProfileSummary`.
- **R9. Slider value binding echoes `changed` on programmatic updates**
  Resolved by seek slider using `released` callback.
- **R10. slint::Image::try_from(wgpu::Texture) per-frame allocation**
  Investigated and closed upstream. Slint maintainer (tronical)
  confirmed wgpu barriers handle the aliasing concern; no perf
  difference measured.
- **R11. No runtime API to probe ONNX Runtime execution providers**
  Resolved by Batch E: `reco_detect::probe_execution_providers()`.
- **R12. reco-calibrate didn't re-export lens profile types at crate root**
  Resolved by Tier 1 PR (#236): `LensProfileInfo`,
  `LensProfileSummary`, `ProfileSource` now re-exported from
  `reco_calibrate` crate root.
- **R13. Live lens tweaking had no fast CameraParams update path**
  Resolved by Batch F (#238): `StitchPipeline::update_camera_params`
  + matching method on `StitchRenderer`. Cheap enough for per-slider-
  drag updates (no GPU pipeline rebuild).
- **R14. No API to clamp camera pose to coverage boundary**
  False friend - `CoverageBoundary::safe_clamp(yaw, pitch, fov, aspect,
  rig_tilt)` was already present. Tier 2b constrained-look toggle uses
  it directly.
- **R15. Cannot re-run auto-calibrate after files_loaded**
  Resolved by Tier 2 (#237) dropping the `!files_loaded` gate on the
  Auto Calibrate button. Tier 3 hotfixes follow up with pipeline
  unload on failure so the user doesn't end up in a mixed state.
- **R16. Silent ffmpeg "Invalid argument" on bad input paths**
  Resolved by Batch G (#254): `reco_core::source::validate_input_path`
  with structured `InvalidPathReason` variants. The GUI maps each
  reason to a toast ("File not found", "Permission denied", etc).
- **R17. Silent encoder failures produce audio-only output**
  Resolved by Batch G (#254): `StitchJob::run` re-opens the output
  file after `session.finish()` and returns
  `StitchError::EmptyOutput` if the video stream is empty. Catches
  AV1-on-pre-Ada silent failure.
- **R18. No shared settings persistence utility across consumers**
  Resolved by reco-io PR #255: `reco_io::settings::{load, save,
  config_dir, RecentFiles}` behind an opt-in `config` feature.
  Per-consumer namespacing (`gui` / `cli` / `obs`) keeps each
  consumer's settings independent.
- **R19. CalibrateVideosOptions only accepted lens profiles via paths**
  Resolved by Batch H: `CalibrateVideosOptions { left_params,
  right_params: Option<CameraParams> }` added. Lens profile resolution
  now goes params > path > auto-detect. Consumers with a pre-resolved
  `CameraParams` (e.g. from `LensDatabase::candidates()`) can skip the
  file round-trip.
- **R20. StitchJob had no start_frame for partial exports**
  Resolved by Batch H: `StitchJob::start_frame(u64)` builder. Combines
  with `max_frames`/`duration` to select a time window. Implemented as
  drain-and-discard (no seek), so exports like "0:15 - 0:30" decode
  from 0:00 but session progress reflects only the exported window.
- **R21. GpuContext::new() was async, consumers had to pollster-wrap**
  Resolved by Batch H: `GpuContext::new_blocking()`. reco-cli, reco-io,
  reco-calibrate, reco-obs, and reco-gui dropped their direct `pollster`
  deps (except reco-cli, which still needs it for
  `GpuContext::for_surface` on the preview path).
- **R22. Wrong step label + double telemetry parse during calibration**
  Resolved 2026-04-18: new `CalibrationStep::DetectingProfiles` variant
  emitted before `pipeline.detect_profiles()` so the GUI label matches
  the work. `CalibrationPipeline` caches parsed `TelemetryData` for each
  side on first use; `imu_sync()` reuses what `detect_profiles()` already
  read. On DJI Action 4 4K/10-bit this halves wall time on the first
  half of calibrate_videos (~175s → ~85s). Heartbeat emission during
  very slow individual steps is tracked as A5-residual.
