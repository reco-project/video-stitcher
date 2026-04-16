# reco-heatmap

Render a ball-position heatmap PNG from a Reco stereo recording.

The heatmap buckets detections into a 2-D grid in panorama yaw/pitch space and
writes a PNG using a black -> red -> yellow -> white colormap. Log-scaled so
a few hot cells don't wash out the rest of the field.

```bash
reco-heatmap left.mp4 right.mp4 \
    -c match.json \
    -m ball_v0.onnx \
    -o panorama.mp4 \
    -p heatmap.png \
    --grid-width 640 --grid-height 180 \
    --yaw-degrees 45 --pitch-degrees 20
```

## Options

- `--grid-width` / `--grid-height`: heatmap resolution in cells.
- `--yaw-degrees` / `--pitch-degrees`: half-ranges of the panorama axes the
  heatmap will cover. Samples outside the range are clamped into the border
  cells — see [`FRICTION.md`](FRICTION.md) item 1.
- `--min-confidence`: drop weak detections.

## Pairs well with

- [`reco-highlights`](../reco-highlights) — feed the same callback stream
  into a highlight detector to ship both sidecars at once.
