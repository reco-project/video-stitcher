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

### N17. No embedded ROI polygon editor

**Impact**: Medium. Lens preview shows ROI vertices using `LensPreviewRenderer::source_to_preview`, including the image's letterboxing. Editing still requires the browser editor and a paste-back operation. An embedded editor needs a shared source-coordinate editing contract; do not approximate it with viewport coordinates. Mapping is exercised by the lens-preview GPU landmark test and `src/ui_tests.rs`.

### N18. Reported native slider pointer loss inside ScrollView

**Impact**: Low, not reproduced by `camera_slider_keeps_pointer_while_dragging_inside_the_scroller` in `src/ui_tests.rs`, which dispatches continuous pointer input with small vertical deviations. Native pointer delivery still needs verification on the affected desktop. There is no confirmed upstream cause; do not duplicate slider controls or move them solely to hide the report.

### N21. Interactive file input cannot use the batch zero-copy source

**Impact**: High for large inputs. `Playback` uses `FfmpegFileSource`, which delivers owned CPU YUV planes even with hardware decoding; `PreviewBridge` uploads them before rendering. Sharing the output texture with Slint does not remove these input transfers or final buffer destruction on the UI thread. `SmartFileSource` provides the batch GPU-resident path but does not implement interactive polling or seeking. Reusing it needs a shared lifetime contract for a displayed frame retained while paused, cancelled or seeking, including safe GPU-slot release. A GUI-only disposal queue would not resolve the input ownership and transfer boundary.
