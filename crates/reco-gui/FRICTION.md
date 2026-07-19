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

### N19. fps-probe fallback is indistinguishable from a real 30fps source

**Impact**: Medium. When `VideoDecoder::frame_rate()` exhausts every
probe (container and codec context) it logs an error and returns
`Rational(30, 1)` - the same value a genuinely 30fps file produces.
The GUI cannot tell the difference, so it can't warn the user that
export timing is a guess (wrong speed, wrong trim). Surfacing it needs
the probe result to carry provenance (e.g. an
`fps_is_estimated` flag on `SourceInfo`).

