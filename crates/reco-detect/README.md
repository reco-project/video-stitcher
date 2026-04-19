# reco-detect

Detection backends behind `reco-core::UnifiedDetector`.

## What it owns

- **CpuYoloDetector** — ORT CPU execution provider, ONNX models with built-in NMS.
- **OrtGpuDetector** — ORT with CUDA / TensorRT EP, GPU-resident NV12 input via the NDVEC zero-copy path. Preprocess (NPP color convert, NPP resize+pad, CUDA normalize+transpose) is GPU-side.
- **MetalYoloDetector** — ORT with CoreML EP, `CVPixelBuffer` zero-copy import, Metal compute preprocess.
- **NcnnYoloDetector** — NCNN backend for mobile / embedded.
- **TrtGpuDetector** — native TensorRT `.engine` runtime. Accepts NVDEC zero-copy frames or CSI-camera CPU-upload (lazy H2D memcpy, ~1-2 ms / 1080p frame on Orin Nano).

## Features

| Feature | Backend | Notes |
|---|---|---|
| `ort` (default) | CPU / CUDA / CoreML via ORT | default for most desktop builds |
| `cuda` | CUDA EP via ORT | requires CUDA 12 + ORT 1.23+ |
| `tensorrt` | TensorRT EP via ORT | requires TensorRT 10 |
| `tensorrt-native` | Native TRT (no ORT) | Jetson, avoids ORT glibc pin |
| `coreml` | CoreML EP via ORT | macOS only |
| `ncnn` | NCNN runtime | mobile / embedded |
| `load-dynamic` | ORT dylib at runtime | avoids linking libonnxruntime statically |

## Build

```bash
cargo build -p reco-detect
cargo build -p reco-detect --features tensorrt-native   # Jetson
cargo build -p reco-detect --features ort,cuda,tensorrt # desktop NVIDIA
```

reco-detect is a backend library; detector selection lives in `reco-autocam::setup_autocam`.
