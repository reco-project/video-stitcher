# reco-highlights

Auto-highlight reel generator for Reco panoramic recordings.

Stitches a stereo recording end-to-end and writes a sidecar JSON "edit decision list"
describing moments of sustained ball activity. Downstream tools (an NLE, a script,
the coach's phone) can turn that list into clips.

```bash
reco-highlights left.mp4 right.mp4 \
    -c match.json \
    -m ball_v0.onnx \
    -o panorama.mp4 \
    -r highlights.json
```

The reel looks like:

```json
{
  "version": 1,
  "total_frames": 18000,
  "windows": [
    {
      "index": 1,
      "start_ms": 4500.0,
      "end_ms": 17800.0,
      "start_frame": 180,
      "end_frame": 534,
      "active_frames": 289,
      "peak_confidence": 0.91,
      "mean_confidence": 0.73
    }
  ]
}
```

## Tuning

- `--min-confidence` (default `0.45`): drop weak YOLO hits.
- `--min-duration-s` (default `2.5`): minimum window length.
- `--max-gap-s` (default `1.0`): bridge brief occlusions.
- `--pre-roll-s` / `--post-roll-s`: padding around each window.

See [`FRICTION.md`](FRICTION.md) for notes on the `reco-core` / `reco-io`
API that surfaced while building this consumer.
