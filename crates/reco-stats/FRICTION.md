# FRICTION — reco-stats

Same baseline friction as `reco-highlights` (items 1, 2, 3, 5, 6, 7, 8) and
`reco-heatmap`. Extra items below are the ones that only really hurt once
the consumer starts writing to a file from the callback.

## 1. Detection callback cannot propagate I/O errors

Signature:

```rust
pub type DetectionCallback = Box<dyn FnMut(&[MappedDetection], u64, f64) + Send>;
```

No return value. If the CSV writer fails halfway through a job (disk full,
broken pipe to `tee`, ...), the callback can only `log::error!` and keep
going. The stitch job will happily complete, produce a corrupt CSV, and
return `Ok`.

Suggestions:

- Give the callback a return type:
  ```rust
  pub type DetectionCallback =
      Box<dyn FnMut(&[MappedDetection], u64, f64) -> Result<(), Box<dyn Error + Send + Sync>> + Send>;
  ```
  and convert the error into a `SessionError` so `run()` returns it.

- Or, give the session a pluggable "sink" trait (`trait DetectionSink { fn on_detections(...) -> Result<(), E>; }`)
  so consumers can encode fallibility explicitly and the session can bubble it up.

## 2. Flushing the CSV requires reclaiming the `Arc<Mutex<_>>`

To flush a `BufWriter` you have to drop it, which means dropping the sink,
which means extracting it from the `Arc<Mutex<_>>` that the callback
captured. That only works because `StitchJob::run` happens to drop the
session (and thus the callback, and thus the session's clone of the Arc)
before returning. If a future refactor holds onto the session inside
`StitchJob` for longer (e.g. to expose metrics), every analytics consumer
silently stops flushing.

The whole pattern is fragile. Better options:

- `StitchJob::run` could take and return the set of registered sinks
  explicitly, so ownership is obvious.
- Or, `session.finish()` could call a `flush()` on registered sinks.
- Or, document the drop-order contract so consumers can rely on it instead
  of guessing.

## 3. `camera_size` is normalized per-camera, which is fine, but undocumented

`MappedDetection.camera_size: (f32, f32)` — the doc says "Bounding box size
in normalized camera coordinates." Every analytics consumer that wants real
pixel sizes has to multiply by the per-camera `width` / `height` from the
calibration. That's easy — but the field name doesn't hint at it. Renaming
to `camera_size_uv` or adding an example in the doc comment ("multiply by
`calibration.left.width` to get pixels") would save a round-trip to the
source.

## 4. CameraId has no `as_str()` / `Display`

To write `L` / `R` to CSV I have to `match det.camera` by hand:

```rust
let cam = match det.camera {
    CameraId::Left => 'L',
    CameraId::Right => 'R',
};
```

A `impl fmt::Display for CameraId` (or `CameraId::as_str(&self) -> &'static str`)
would let every logger / formatter / CSV writer treat it uniformly. Minor,
but the kind of thing every consumer duplicates.

## 5. `MappedDetection: Copy` but vectors of them get cloned a lot

`DirectorContext.detections: &'a [MappedDetection]` is borrowed for the
director call, and `session.detection.last_detections.clone()` makes a
fresh `Vec` for `step()`'s return. A CSV consumer only needs to *read*
them — an enumeration API or a visitor-style sink would avoid the
`Vec::clone` per frame when the consumer only projects a subset of
fields.

Not a blocker — just a heads-up that the current design pays for a copy
whether the consumer needs it or not.
