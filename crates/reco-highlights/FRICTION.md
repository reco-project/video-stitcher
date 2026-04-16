# FRICTION — reco-highlights

Notes from building an "auto-reel" consumer on top of `reco-core` + `reco-io`.
This is raw field-report data for API improvement — none of these were fatal,
but each one forced a detour.

## 1. `StitchJob` forces producing an output video file

`StitchJob::new(left, right, cal, output)` requires an output path, and
`StitchJob::run` unconditionally:

1. opens the source,
2. builds a session,
3. creates an FFmpeg encoder,
4. runs the frame loop,
5. calls `session.finish()`.

There is no way to ask it for "decode + detect only, no render, no encode".
For a detection-only consumer (this crate actually *wants* the stitched video
too, so we get away with it, but see `reco-stats` and `reco-heatmap`), you end
up paying for:

- full GPU render of every frame
- NV12 triple-buffer readback
- FFmpeg encode to disk

...just to get at the `set_detection_callback` stream. A `StitchJob::analyze()`
variant, or an `AnalyzeJob` sibling that wires up `SmartFileSource` + session
+ detector *without* an encoder, would save every analytics consumer from
re-implementing the zero-copy / GPU plumbing that lives in `StitchJob::run`.

## 2. The detection callback is `FnMut + Send + 'static`, but `on_session` is `FnOnce`

The shape of the callback chain is:

- `StitchJob::on_session(FnOnce(&mut StitchSession, &dyn FrameSource))`
- inside it, `session.set_detection_callback(Box<dyn FnMut(&[MappedDetection], u64, f64) + Send>)`
- after `job.run()` returns, the consumer wants the accumulated state back

Because the callback is `'static`, you can't capture a `&mut` to local state.
The only way to share accumulator state across the callback boundary and
the post-run code is `Arc<Mutex<_>>` (or `Arc<AtomicX>` for scalars).

That means every consumer writes the same boilerplate:

```rust
let acc = Arc::new(Mutex::new(State::default()));
let inner = acc.clone();
job.on_session(move |session, _| {
    let sink = inner.clone();
    session.set_detection_callback(Box::new(move |d, i, t| {
        sink.lock().unwrap().push(d, i, t);
    }));
});
job.run(...)?;
let state = Arc::try_unwrap(acc).unwrap().into_inner().unwrap().finish();
```

Two suggestions:

- Have `StitchJob::run` (or a new `run_with`) *return* some opaque "session
  result" carrier that lets the caller re-borrow attached consumer state.
- Or: make `set_detection_callback` accept an owned type, and `StitchJob::run`
  return whatever was attached. Think of it like `thread::spawn`'s return
  value.

## 3. The callback signature `fn(&[MappedDetection], u64, f64)` is positional and cryptic

Three unlabeled arguments: detections, frame index, timestamp ms. At the
call site it reads as `move |dets, idx, ts| ...` and you have to go read the
docs to know the order. A small struct `DetectionEvent { frame_index, timestamp_ms, detections }`
(similar to `DirectorContext`) would be consistent with the rest of the crate
and easier to extend later without breaking every consumer.

## 4. `MappedDetection.position` is `Option<ViewportPosition>` in radians

For a highlights reel this is fine — we only look at class id and confidence.
But any consumer that wants to bucket by position (heatmap, zone stats) has
to:

- check `Option` (OK — some detections are outside the camera FOV),
- convert radians to degrees or pixels,
- somehow figure out the panorama extent (there is no `full_panorama_bounds()`
  anywhere — the output viewport is a *cropped* subset of the panorama, not
  the whole thing).

At minimum, a helper on `MappedDetection` like `fn panorama_uv(&self) -> Option<(f32, f32)>`
that returns values in `[0,1]` relative to the full L-shaped panorama would
save every analytics consumer from duplicating projection math.

## 5. No way to learn the source fps from the callback

The detection callback receives a `timestamp_ms`, but the consumer has no
way to convert a pending highlight window back to a `(start_frame, end_frame)`
range without also knowing the source fps. We grab it via `_source.info().fps`
inside the `on_session` closure, but that has to be stashed into yet another
`Arc<AtomicU32>` to make it visible to the callback. A `DetectionEvent`
struct (see #3) that carries fps alongside timestamp would also solve this.

## 6. `AutocamConfig` doesn't pass through to `StitchJob`

`StitchJob` has a `.on_session` hook, which works. But since 90% of
analytics consumers want `setup_autocam_from_config(session, &config)` inside
it, a first-class `.autocam(AutocamConfig)` builder method on `StitchJob`
would avoid the closure dance. Today the example in `stitch_job.rs`'s own
docs literally shows how to wire autocam by hand.

## 7. `reco-autocam::setup_autocam_from_config` hard-codes fps to 30

Line 144 of `crates/reco-autocam/src/lib.rs`:

```rust
30.0, // default fps when not available from source
```

It comments "when not available from source" but the function never asks
the source. `session.pipeline().source_info()` returns dimensions, not fps.
The One Euro smoother and lookahead calculations downstream silently use
30 fps even on 50/60 fps footage. That is not *this* crate's problem to fix,
but it is friction I hit while debugging why my highlight timestamps drifted
on a 50fps test clip — I went looking for a fps bug in my code before I
spotted this one.

## 8. Any dep on `reco-io` drags in FFmpeg dev headers

`reco-io`'s default features include `ffmpeg`, which pulls `ffmpeg-next` and
`ffmpeg-sys-next`. The latter runs a `pkg-config` build script at *check*
time — so even `cargo check --lib` on a downstream crate fails on a machine
without `libavutil.pc` installed. This crate works around it by making
`reco-io` an *optional* dep gated behind a `cli` feature, splitting the
library (reco-core only, dead-simple to build) from the binary (full
FFmpeg toolchain). Every analytics consumer has to do the same dance.

A fix would be to either:

- keep FFmpeg behind a non-default feature in `reco-io`, or
- split `reco-io` into `reco-io-core` (trait definitions, no FFmpeg) and
  `reco-io-ffmpeg`, so consumers can depend on just the trait surface
  without inheriting the C toolchain dep.

## 9. `on_session` closure can't return an error

It is `FnOnce(&mut StitchSession, &dyn FrameSource)` — no `Result`. If
`setup_autocam_from_config` fails inside the closure, the only options are
`log::error!` + swallow, or `panic!`. The stitch job then silently runs
without a detector and writes an empty highlights reel, which looks like
"no highlights were found" to the user. Making `on_session` return
`Result<(), Box<dyn Error + Send + Sync>>` would let the consumer propagate
the failure as a clean `StitchError`.
