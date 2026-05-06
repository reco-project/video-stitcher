# OBS Plugin Consumer API Friction

Active friction points building `reco-obs` against `reco-core` / `reco-io`.

## Active

### A3. No OBS-level wgpu interop

**Impact**: Fundamental. OBS uses OpenGL/D3D11, reco-core uses wgpu.
Rendered frames must roundtrip through CPU (~8 MB/frame at 1080p).
Platform-specific solutions (DMA-BUF on Linux, shared D3D11 on Windows)
need new interop code. Known limit, not actionable without a specific target.

### A5. No temporal frame pairing helper

**Impact**: Medium. Two independent OBS sources return frames with
mismatched timestamps. A reusable ring-buffer pairing helper in
reco-io would emit closest-timestamp pairs and surface drift warnings.
Same helper reused by future WebRTC/V4L2 consumers.

### A6. No source filtering by OBS_SOURCE_ASYNC_VIDEO

**Impact**: Low. Source-picker shows every OBS source including scenes
and audio-only. Fix: 5 lines of FFI + a conditional.

### A7. No auto-resize on input dimension change

**Impact**: Low. Dimension mismatch between properties and actual frames
silently freezes output. Consumer-side rebuild works but isn't automatic.

### A9. obs_source_get_frame only sees async video sources

**Impact**: High. Sync sources (Browser Source, screen capture) are
invisible. Needs a render-to-texture fallback path via OBS graphics API.

### A10. No detector/director hooks on push API

**Impact**: Medium. AI tracking works in StitchSession (pull) but not
in StitchCore (push, used by OBS). Need set_detector/set_director on
the push API, plus a worker-thread inference scheduler.

### A11. Visible upstream sources steal async frames

**Impact**: High UX. When an upstream source is visible in the OBS scene,
its renderer consumes the async frame before our poller can read it.
Hiding the source via the eye icon fixes it. Needs auto-hide on pick.

### A12. No keyboard/controller mapping for pan-zoom

**Impact**: Medium. Only OBS Interact window works for pan/zoom. Need
obs_hotkey_register_source for keyboard, SDL3 sidecar for gamepad.

### A13. No constrained-look toggle

**Impact**: Medium. CoverageBoundary::safe_clamp exists but the OBS
plugin applies raw yaw/pitch. Extreme pan shows black borders. Fix:
add a constrained_look property, ~30 min.

### A14. No live calibration workflow for non-file sources

**Impact**: High. Calibration requires video files. Live OBS sources
need a "Calibrate from current sources" button that captures N frame
pairs and runs the solver.

### A15. OBS Interact window is a poor primary UX

**Impact**: Medium. Users expect to pan in the main scene view, not a
separate interact window. A dock panel with sliders + preview would be
better but requires Qt bindings.

### A16. No built-in replay mode

**Impact**: Low-Medium. Users want scene-switch to replay. Needs an
opt-in rolling buffer on StitchCore (~7 GB at 1080p/30fps/30s).

### A17. No AI auto-pan in the OBS plugin

**Impact**: Medium. See A10. Requires reco-autocam + reco-detect as
deps with worker-thread inference.

### A20. Replay toggle decoupled from OBS Record/Stream buttons

**Impact**: Medium. The replay checkbox is independent of OBS's global
recording state. Users expect OBS's Record button to control everything.

### A21. Live Matroska replay hard to scrub in players

**Impact**: Low. Matroska files being written have unstable duration
metadata. Finalized files scrub normally.

### A22. Input dimensions declared, not detected

**Impact**: High UX. User must manually set input W/H in properties.
Mismatched dims silently freeze output. Should auto-detect from first
frame and rebuild the pipeline.

### N9. FOV accessors require pipeline_mut()

**Impact**: Low. No StitchCore::set_fov/fov_degrees shortcut. Consumers
detour through core.pipeline_mut().set_fov(v). Should match the
set_rig_tilt pattern.
