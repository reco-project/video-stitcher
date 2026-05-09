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

### N15. PoseControl requires manual rig_tilt threading

**Impact**: High. Caused a constrained-look regression. Consumers must
pass rig_tilt to both `clamp_via_coverage` and `render_pose` on every
tick. The renderer should own the pose state machine.

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
