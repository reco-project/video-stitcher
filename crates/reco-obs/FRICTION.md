# OBS Plugin Consumer API Friction

Active friction points building `reco-obs` against `reco-core` / `reco-io`.

## Active

### A3. No OBS-level wgpu interop

**Impact**: Fundamental. OBS uses OpenGL/D3D11, reco-core uses wgpu.
Rendered frames must roundtrip through CPU (~8 MB/frame at 1080p).
Platform-specific solutions (DMA-BUF on Linux, shared D3D11 on Windows)
need new interop code. Known limit, not actionable without a specific target.

### A5. OBS capture is not integrated with bounded frame selection

**Impact**: Medium. Two independent OBS sources can return frames with mismatched timestamps. `reco_io::frame_sync::Selector` already supplies transport-independent bounded wait/reuse and capture-age checks, covered by its module tests. OBS still needs explicit timestamp-domain mapping, polling and retained-frame ownership to use it. The helper does not infer clock alignment or own OBS frame lifetimes.

### A9. obs_source_get_frame only sees async video sources

**Impact**: High. Sync sources (Browser Source, screen capture) are
invisible. Needs a render-to-texture fallback path via OBS graphics API.

### A10. OBS tracking integration and inference scheduling

**Impact**: Medium. StitchCore exposes detector, tracker and panner attachment, but OBS does not wire them. OBS integration still needs an inference scheduling/threading decision that keeps the source render loop responsive. This is not a missing attachment API.

### A11. Visible upstream sources steal async frames

**Impact**: High UX. When an upstream source is visible in the OBS scene,
its renderer consumes the async frame before our poller can read it.
Hiding the source via the eye icon fixes it. Needs auto-hide on pick.

### A12. No keyboard/controller mapping for pan-zoom

**Impact**: Medium. Only OBS Interact window works for pan/zoom. Need
obs_hotkey_register_source for keyboard, SDL3 sidecar for gamepad.

### A14. No live calibration workflow for non-file sources

**Impact**: High. Calibration requires video files. Live OBS sources
need a "Calibrate from current sources" button that captures N frame
pairs and runs the solver.

### A15. OBS Interact window is a poor primary UX

**Impact**: Medium. Users expect to pan in the main scene view, not a
separate interact window. A dock panel with sliders + preview would be
better but requires Qt bindings.

### A16. No built-in replay mode

**Impact**: Low-Medium. Users want scene-switch to replay. `StitchCore::enable_replay_buffer` already supplies opt-in RGBA replay storage, covered by the core replay-buffer tests. The OBS plugin does not wire replay selection and timed delivery; enabling storage alone does not provide a replay mode.

### A17. No AI auto-pan in the OBS plugin

**Impact**: Medium. See A10. Requires reco-autocam + reco-detect as
deps with worker-thread inference.

### A21. Live Matroska replay hard to scrub in players

**Impact**: Low. Matroska files being written have unstable duration
metadata. Finalized files scrub normally.
