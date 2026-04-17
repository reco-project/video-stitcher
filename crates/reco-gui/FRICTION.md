# GUI Consumer API Friction

Friction points encountered while building the Slint GUI consumer of
reco-core. Active items are at the top; resolved items are archived
at the bottom with the PR that fixed them so we can tell "this used to
be painful" without re-litigating solved problems.

## Active

### A1. reco-calibrate doesn't re-export lens profile types at crate root

**Impact**: Minor, one-line fix per consumer.

`LensProfileInfo`, `LensProfileSummary`, `ProfileSource` live in
`reco_calibrate::types::*` but were not re-exported in `lib.rs` alongside
`CalibrationResult` and friends. The Tier 1 GUI had to either use
fully-qualified paths or add the re-export itself.

**Fixed in passing** during Tier 1 (reco-gui PR #236), but the pattern
suggests a general "every public type visible in a CalibrationResult
field should be re-exported from the crate root" guideline worth adding
to the next API-principles pass.

### A2. CalibrateVideosOptions only accepts lens profiles via file paths

**Impact**: Medium. Blocks a functional lens-profile picker in the GUI.

`reco_calibrate::video::CalibrateVideosOptions::{left_profile, right_profile}`
are `Option<PathBuf>`. The consumer who already has a `LensProfileSummary`
from `LensDatabase::candidates()` (Batch B) has no way to pass it
directly. Workaround: look up the profile in the DB again via
`find(camera, model, w, h, lens_info)`, which returns
`(CameraParams, LensProfileInfo)`, then... there's no way to hand
`CameraParams` to `calibrate_videos` either. So the consumer must either
serialize the `CameraParams` to a temp JSON file just to re-load it, or
drop down to `CalibrationPipeline::set_profiles()` and reimplement the
video-frame extraction loop.

**Suggested addition**:
```rust
pub struct CalibrateVideosOptions {
    // ... existing ...
    pub left_params: Option<CameraParams>,
    pub right_params: Option<CameraParams>,
}
```
If both `_path` and `_params` are set, `_params` wins.

Blocked the Tier 1 picker from being functional (it's read-only); will
block Tier 2's "pick an alternate profile and re-run" interaction too.

### A3. Live lens tweaking has no "just update CameraParams" path

**Impact**: High for Tier 2 live lens fine-tuning.

To let the user tweak fx/fy/cx/cy/distortion sliders and see results in
the preview, the GUI needs an API to replace the `CameraParams` inside
an existing `MatchCalibration` and have the stitch pipeline re-undistort
without rebuilding everything from scratch. Currently the stitch
pipeline caches undistort LUTs at construction; changing camera params
means recreating the pipeline.

**Suggested addition** in reco-core:
```rust
impl StitchPipeline {
    /// Replace camera intrinsics for one or both cameras and rebuild
    /// the undistort LUTs. Cheap enough to call on slider drag.
    pub fn update_camera_params(
        &mut self,
        left: Option<CameraParams>,
        right: Option<CameraParams>,
    ) -> Result<(), PipelineError>;
}
```

Will raise as a Batch F candidate if Tier 2 hits this wall.

### A4. No API to clamp camera pose to coverage boundary

**Impact**: Medium. Tier 2 "constrained look" toggle needs this.

`CoverageBoundary` (reco-core) knows the valid yaw/pitch ranges of the
stitched output. The GUI yaw/pitch inputs feed the renderer directly,
with no clamping. To prevent the preview from panning into black
margins, the consumer would need a helper like
`CoverageBoundary::clamp(yaw, pitch, fov_rad) -> (yaw, pitch)` that
respects the effective viewport half-angle.

**Suggested addition**:
```rust
impl CoverageBoundary {
    /// Given a desired camera pose and viewport FOV, return the pose
    /// clamped so the entire viewport fits within the coverage region.
    pub fn clamp(&self, yaw: f32, pitch: f32, hfov_rad: f32, aspect: f32)
        -> (f32, f32);
}
```

### A5. Cannot re-run auto-calibrate after files_loaded

**Impact**: UX bug, not an API gap, but maintained here because it
surfaces a design question the API should answer.

The GUI disables Auto Calibrate once calibration completes
(`!root.files-loaded` in main.slint). The user who wants to try
different calibration options (different frames, IMU seeds toggled,
different profiles once A2 is solved) has to restart the app. The fix
is client-side (drop the `!files_loaded` gate), but it raises the
question of whether reco-core should expose a "clear previous
calibration" helper or document that consumers can freely re-run.

Not blocking; tracking here so we revisit after Tier 2 touches the
calibration flow.

### A6. GpuContext::new() is async, consumers always pollster-wrap

**Impact**: Very minor, but every consumer does the exact same thing.

Deliberate (wgpu adapter creation is async), but every call site in
reco-cli, reco-gui, reco-obs wraps it in `pollster::block_on`. A
blocking convenience constructor in reco-core would let consumers avoid
taking a direct dep on pollster:

```rust
impl GpuContext {
    pub fn new_blocking() -> Result<Self, PipelineError> {
        pollster::block_on(Self::new())
    }
}
```

### A7. render_to_target() still returns CommandBuffer

**Impact**: Low. The Batch A `RenderTarget`/`RenderOutcome` enums
simplified the surface-vs-internal asymmetry, but the lower-level
`render_to_target()` still returns a `wgpu::CommandBuffer` the caller
must submit. For the zero-copy Slint path this is fine (we render
directly to a Slint-owned texture via `render_yuv`); for any future
consumer that wants to batch our render with their own copy commands,
the current API already works. Leaving this here as a note for anyone
hitting the old asymmetry - the Batch A outcome enum was the right
abstraction.

## Resolved (archived)

Items pruned from Active once the reco-core API added what we asked for.
Kept as an audit trail of which consumer friction items drove which
upstream changes.

- **R1. wgpu version mismatch blocking zero-copy rendering**
  Resolved 2026-04-16 by downgrading reco-core to wgpu 28 and using
  Slint 1.15's `unstable-wgpu-28` feature. `PreviewBridge` now shares
  Slint's device/queue via `GpuContext::from_device_queue()`.
- **R2. No RGBA readback API on StitchRenderer**
  Resolved by Batch A (#223): `StitchRenderer::render_and_readback_rgba()`
  + `flush_rgba()` with triple-buffered staging. Lifted ~120 lines of
  readback boilerplate out of the GUI.
- **R3. StitchRenderer hardcoded InputFormat::Yuv420p**
  Resolved by Batch A (#223): `StitchRenderer::new` now takes an
  `input_format: InputFormat` parameter so NV12 live-camera consumers
  can construct a renderer without dropping to `StitchPipeline`.
- **R4. resize() didn't notify consumers of new dimensions**
  Resolved by Batch C (#222 companion): `StitchPipeline::resize()` now
  returns `Option<(u32, u32)>` so external staging buffers can be
  recreated when the internal render target changes size.
- **R5. FrameSource::try_next_frame() EOF ambiguity**
  Resolved by Batch A (#223): `FrameSource` trait gained
  `fn is_exhausted(&self) -> bool` with a default impl. Replaced the
  1-second timeout heuristic in the GUI.
- **R6. render_to_view vs render_to_target asymmetry**
  Resolved by Batch A (#223): `RenderTarget` enum + `RenderOutcome`
  unify the surface and internal-texture paths.
- **R7. CalibrationResult didn't expose detected lens profile**
  Resolved by Batch B (#224): `CalibrationResult.left_lens_profile`
  and `right_lens_profile` now carry `LensProfileInfo` with camera /
  lens / source / optional path.
- **R8. No API to list available lens profiles**
  Resolved by Batch B (#224): `LensDatabase::iter_profiles()` and
  `candidates(width, height)` return `LensProfileSummary` suitable
  for picker UIs. The Tier 1 GUI uses `candidates()` for the
  alternate-profile count.
- **R9. Slider value binding echoes `changed` on programmatic updates**
  Resolved by follow-up cleanup PR: switched seek slider to `released`
  callback. No more feedback loop between Rust state and UI updates,
  debounce no longer strictly needed but kept for drag coalescing.
- **R10. slint::Image::try_from(wgpu::Texture) per-frame allocation**
  Investigated and closed upstream. Slint maintainer (tronical) confirmed
  wgpu barriers handle the aliasing concern; measured no perf
  difference from per-frame allocation on our workload. Kept current
  per-frame-allocation pattern.
- **R11. No runtime API to probe ONNX Runtime execution providers**
  Resolved by Batch E (#222): `reco_detect::probe_execution_providers()`
  returns `AiProbeResult { providers, can_run_on_gpu_frames, errors }`.
  The Tier 1 GUI uses it to show an honest runtime AI status instead
  of lying with compile-time `cfg!()`.
