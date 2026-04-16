# reco-stats

Export Reco detections to CSV, one row per detection.

```bash
reco-stats left.mp4 right.mp4 \
    -c match.json \
    -m ball_v0.onnx \
    -o panorama.mp4 \
    -s detections.csv
```

Pipe straight into pandas:

```python
import pandas as pd
df = pd.read_csv("detections.csv")
df.groupby("camera")["confidence"].describe()
```

Columns are documented in the [crate-level docs](src/lib.rs). See
[`FRICTION.md`](FRICTION.md) for notes on the reco API that surfaced while
building this.
