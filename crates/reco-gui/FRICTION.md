# GUI Consumer API Friction

Friction points encountered while building the Slint GUI consumer of
reco-core and reco-io. Active items are at the top; resolved items
archived at the bottom with the PR that fixed them, so we can tell
"this used to be painful" without re-litigating solved problems.

## Active

### A3. Preview pipeline and export StitchJob share CUDA context and
racereliably when both active

**Impact**: High. Known to crash (#243) or hang (#247) the GUI when
the user pauses/plays preview while an export is encoding.

Both paths create their own NVDEC decoders on the same CUDA context
(the process-wide default). Contention on `cuCtxPopCurrent` during
frame handoff produces `cu->cuCtxPopCurrent(&dummy) failed` and a
subsequent segfault, or a mutex-like hang depending on timing.

reco-core / reco-io don't currently expose any "is an export active"
guard or "acquire-exclusive" flow, so consumers can't gate preview
on export state.

**Suggested direction** (needs design):
- Either serialize: `StitchJob::run_exclusive` drops a mutex that the
  preview pipeline must acquire too.
- Or share: pass the preview's `GpuContext` + decoder handles into
  `StitchJob` so they reuse the same resources.
- Or isolate: spawn export in a subprocess with its own CUDA ctx.

### A4. No way for consumers to unload a pipeline cleanly

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

### N10. setup_autocam positional-11-arg signature

**Impact**: Low-Medium. Pain on every autocam feature addition.

`reco_autocam::setup_autocam(session, model, w, h, fps, use_zero_copy,
interval, lead, mode, roi, is_10bit)` is an 11-positional-arg call.
Consumers that only want to change the detection interval have to
re-type every arg. `setup_autocam_from_config(&mut session, &config)`
with `AutocamConfig::new(path).with_*()` landed alongside it, but the
old signature is still the public path for the CLI/OBS.

Plan disposition: K6 / E11. Deprecate `setup_autocam` once all
consumers migrate to `AutocamConfig`.

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

Plan disposition: K6 / E7. Lands with N12.

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
