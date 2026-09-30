# GUI Consumer API Friction

Active friction points building the Slint GUI consumer against
reco-core and reco-io.

## Active

### A5-residual. No intra-step heartbeat during very slow steps

**Impact**: Low. Individual calibration steps can run 20-60s silently
(AKAZE on dense scenes, optimizer on 100+ pairs, audio PCM extraction).
A consumer can't distinguish "making progress" from "hung."

### N8. Slint wgpu 28 rendering notifier does not expose AdapterInfo

**Impact**: Low. Blocked on Slint upstream exposing the adapter reference.

### N16. Recording lags preview and drops frames during panning

**Impact**: High. Recording rerenders the scene for NV12 readback on the UI thread and reduces display updates to every fifth recording frame. Readback may block despite triple buffering. An independent background StitchSession would not capture the image selected for display, including paused camera edits or lens preview. Interactive recording needs an owned rendered-frame capture boundary with bounded background conversion/delivery and explicit failure reporting, preserving the displayed frame's timestamp through readback. Encoding must not require a second stitch or a platform-specific desktop-capture API.

### N20. Preview recording has no image-associated presentation timeline

**Impact**: High. Preview recording uses fixed-rate output, so missing interactive frames compress time rather than preserving elapsed duration. The file sink accepts explicit presentation timestamps (`reco-io/tests/encoder_timing.rs`), and session delivery retains them through GPU readback (`reco-core/src/session/tests/timing.rs`). Preview uses the engine's separate `render_and_readback_nv12` path, which returns bytes without their originating presentation time. Correct recording needs the displayed image and its time captured together, including paused camera edits; substituting the current callback time after readback would timestamp an older image.

### N17. No ROI visualization or editing in GUI

**Impact**: Medium. The calibration produces `field_roi` but the GUI
never shows or lets users adjust it. Users can't verify the ROI
without reading raw JSON.

### N18. Slint slider loses pointer tracking inside ScrollView

**Impact**: Low. Likely a Slint bug. Workaround: move critical sliders
outside the ScrollView.
