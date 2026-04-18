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

### A5. No runtime progress callback granularity during heavy calibration steps

**Impact**: Low to medium. Users on DJI 10-bit footage see the status
bar frozen on one step for 60-90 seconds and assume hang.

`reco_calibrate::video::calibrate_videos` emits `CalibrationProgress`
events at the STEP level (probing, telemetry, audio_sync, akaze,
optimize) but not INTRA-step. For DJI 4K 10-bit footage the
audio-sync and telemetry-extraction steps each take 10-30s with no
feedback.

**Suggested addition**: per-frame progress emission within
`extract_frame_pairs`, or at least a "heartbeat" event every 2s so
consumers know the worker is still alive.

### A7. render_to_target() still returns CommandBuffer

**Impact**: Low. The Batch A `RenderTarget` / `RenderOutcome` enums
simplified the surface-vs-internal asymmetry, but the lower-level
`render_to_target()` still returns a `wgpu::CommandBuffer` the caller
must submit. For the zero-copy Slint path this is fine (we render
directly to a Slint-owned texture via `render_yuv`); future consumers
that want to batch our render with their own copy commands already
work with the current API. Leaving this as a note for anyone hitting
the old asymmetry - the Batch A outcome enum was the right abstraction.

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
