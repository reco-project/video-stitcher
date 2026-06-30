# GUI Consumer API Friction

Active friction points building the Slint GUI consumer against
reco-core and reco-io.

## Active

### A5-residual. No intra-step heartbeat during very slow steps

**Impact**: Low. Individual calibration steps can run 20-60s silently
(AKAZE on dense scenes, optimizer on 100+ pairs, audio PCM extraction).
A consumer can't distinguish "making progress" from "hung."

### N7. FfmpegFileSource opens full container just to read total_frames

**Impact**: Low. Two muxer opens per export (probe + real read).
A lightweight `reco_io::probe_duration(path)` would remove one.

### N8. Slint wgpu 28 rendering notifier does not expose AdapterInfo

**Impact**: Low. Blocked on Slint upstream exposing the adapter reference.

### N12. YuvPlanes hand-constructed in every render call site

**Impact**: Low. A `YuvData::planes()` returning `YuvPlanes<'_>` would
remove per-call boilerplate across GUI and CLI.

### N13. StereoFrame::into_yuv420p is missing

**Impact**: Low. `as_yuv420p()` returning `Option<(YuvPlanes, YuvPlanes)>`
would centralize the destructure done in 5+ sites.

### N16. Recording lags preview and drops frames during panning

**Impact**: High. NV12 readback runs on the UI thread, stalling the
Slint compositor. Recording should use a background StitchSession
(the export pipeline), not UI-thread readback.

### N17. No ROI visualization or editing in GUI

**Impact**: Medium. The calibration produces `field_roi` but the GUI
never shows or lets users adjust it. Users can't verify the ROI
without reading raw JSON.

### N18. Slint slider loses pointer tracking inside ScrollView

**Impact**: Low. Likely a Slint bug. Workaround: move critical sliders
outside the ScrollView.

### N19. FOV reaches the renderer via a cached push, not a render param

**Impact**: Medium. `render_yuv` takes yaw/pitch as live per-frame
parameters, but FOV only through a separate cached `pipeline.set_fov`.
A pose change applied outside the smoothing tick (e.g. re-enabling
constrained look, which clamps `current_fov` directly) updates yaw/pitch
live but leaves the cached FOV stale, so the view ignores the clamp while
the slider shows it. The consumer has to remember to push FOV after any
out-of-tick clamp. FOV should be a `render_yuv` parameter like yaw/pitch
(or the renderer should own the full pose state).
