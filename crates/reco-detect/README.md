# reco-detect

Detection backends behind `reco_core::detect::detector::Detector`.

## What it owns

- **CpuYoloDetector** — ORT CPU execution provider, ONNX models with built-in NMS.
- **OrtGpuDetector**: ORT with CUDA / TensorRT EP, CUDA-resident NV12/P010 input via the NVDEC zero-copy path. Preprocessing (NPP color conversion, NPP resize/padding, CUDA normalization) runs on the GPU.
- **MetalYoloDetector** — ORT with CoreML EP, `CVPixelBuffer` zero-copy import, Metal compute preprocess.
- **NcnnYoloDetector** — NCNN backend for mobile / embedded.
- **TrtGpuDetector**: native TensorRT `.engine` runtime. Accepts supported CUDA-resident frames or CPU frames uploaded to the GPU.

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
