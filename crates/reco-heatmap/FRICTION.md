# FRICTION — reco-heatmap

Every friction point in `reco-highlights/FRICTION.md` (items 1, 2, 3, 6, 7,
8) applies to this crate too — not going to restate them. These are the ones
that only showed up once the consumer cared about *positions* rather than
just *timestamps*.

## 1. There is no documented panorama coordinate range

`MappedDetection.position.yaw` and `pitch` are radians, and the docs say
yaw `0.0` is the seam between the two cameras. That's it. There is no
`PanoramaBounds { yaw_min, yaw_max, pitch_min, pitch_max }` anywhere, and no
function on `StitchSession` or `SceneGeometry` that returns "the valid
yaw/pitch extent for the current calibration". I have the
`CoverageBoundary` type (accessible via `session.coverage()`) but that's
only exposed through `safe_clamp` / `max_fov_degrees` — the raw `yaw_min`,
`yaw_max`, `pitch_min`, `pitch_max` numbers aren't on the public API.

So this crate hard-codes `±45° yaw, ±20° pitch` as heatmap defaults and
clamps out-of-range samples into the border cells. That's wrong for any rig
with an unusual plane layout or FOV. A consumer that wants a *correct*
heatmap can't get one without reading the private scene geometry.

Suggestion: expose something like

```rust
impl CoverageBoundary {
    pub fn yaw_range(&self) -> (f32, f32);
    pub fn pitch_range(&self) -> (f32, f32);
}
```

or a higher-level helper:

```rust
impl StitchSession {
    pub fn panorama_extent(&self) -> PanoramaExtent;
}
```

## 2. Detection callback fires before session.coverage() is readable

The `on_session` hook fires *before* `StitchJob::run` starts the frame loop,
and that is the one place where we'd actually be able to read
`session.coverage()` — except by then we also want to hand out the
detection callback, which captures an `Arc<Mutex<HeatmapAccumulator>>`.
The accumulator needs the coverage bounds to build its grid, so now you
have to:

1. inside `on_session`, read `session.coverage()` (clone it),
2. stash it in an `Arc<OnceCell<_>>`,
3. have the callback lazy-init the accumulator the first time it fires
   with a non-empty detection list,
4. hope that works.

This crate punts on all of that by using hard-coded bounds. A cleaner fix
is either making the detection callback receive a context with the
coverage bounds attached, or letting `on_session` construct and register
the callback with access to the full session state (which it already has —
see `reco-highlights/FRICTION.md#2` — but you can't *return* a constructed
accumulator out of the `FnOnce`).

## 3. No timestamped "empty frame" signal

The callback is called every frame (even when no detections fire), so we
get the frame cadence for free — but on zero-copy paths detection is
skipped when there is no detector attached, and even when it's attached
it might get skipped by `detection_interval`. That means a silent period
in the heatmap could mean "no ball visible" *or* "this frame's detection
was skipped" *or* "no detector attached". The consumer cannot tell.

A `fresh_detection: bool` flag (like the one `DirectorContext` already
carries for the director trait) on the callback side would let consumers
distinguish "detection ran, found nothing" from "detection was skipped".

## 4. `image` crate is a heavy dep for "write one PNG"

Not a `reco-core` issue, just reporting cost: pulling in `image 0.25` for
a single `RgbaImage::save_png` adds measurable build time. A tiny
`reco-core::io::write_png_rgba8(path, w, h, &[u8])` helper would be nice
for analytics consumers that only need this.

## 5. Confusion between viewport space, camera space, and panorama space

`MappedDetection` carries:
- `camera_center: (f32, f32)` - normalized camera-pixel [0,1]
- `camera_size: (f32, f32)` - normalized camera-pixel
- `position: Option<ViewportPosition>` - yaw/pitch radians

The field is called `position` but the type is `ViewportPosition`, which
everywhere else in the crate refers to the *output* viewport the director
chose, not the detection's coordinate in the panorama. Same type, two
semantic meanings. A `PanoramaPoint` type with the same fields but a
different name would prevent readers from assuming the detection got
mapped into the output frame.
