# reco-cli

Terminal consumer binary. Proves the workspace API is thin enough for each command to stay ~100 lines.

## Commands

| Command | Purpose |
|---|---|
| `reco stitch` | Batch: stitch two files to a panoramic output, optionally with AI tracking + replay |
| `reco calibrate` | Compute L-shape placement from two videos, write `match.json` |
| `reco preview` | Interactive GPU preview (winit window) with keyboard pan/zoom |
| `reco camera` | Live stitching from two GStreamer sources (Jetson CSI, V4L2) |
| `reco analyze` | Run detection on a video, emit JSON ball track (no stitching) |
| `reco info` | Report GPU backend + capabilities, supported formats |

## Build

```bash
cargo build --release -p reco-cli                                  # default (ort-backed detection)
cargo build --release -p reco-cli --features tensorrt-native       # Jetson
cargo build --release -p reco-cli --features gstreamer,profiling   # live + tracing
```

## Output containers

`--container mp4` (default), `mp4-fragmented`, `mkv`, `mov`. MKV is the recommended live-record container: partial files survive unclean shutdowns.

## Reference

See [../../README.md](../../README.md) for the workspace-level architecture + install path. Per-command help: `reco <command> --help`.
