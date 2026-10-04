# reco-detect

Detection backends behind `reco_core::detect::detector::Detector`.

## What it owns

- **CpuYoloDetector** — ORT CPU execution provider, ONNX models with built-in NMS.
- **OrtGpuDetector**: ORT with CUDA / TensorRT EP, CUDA-resident NV12/P010 input via the NVDEC zero-copy path. Preprocessing (NPP color conversion, NPP resize/padding, CUDA normalization) runs on the GPU.
- **MetalYoloDetector** — ORT with CoreML EP, `CVPixelBuffer` zero-copy import, Metal compute preprocess.
- **NcnnYoloDetector**: NCNN CPU backend for host-memory YUV420P/NV12 frames. One model handles different camera dimensions with aspect-preserving letterboxing. The model contract is documented on `NcnnYoloDetector::new`; [runtime tests](src/detectors/ncnn/tests.rs) cover placement, coordinate recovery and error paths.
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
NCNN_DIR=/path/to/ncnn cargo build -p reco-detect --no-default-features --features ncnn
```

reco-detect is a backend library; detector selection lives in `reco-autocam::setup_autocam`.

`NCNN_DIR` is the install prefix containing `lib/libncnn.a`. The Linux NCNN build links the C++ and GNU OpenMP runtimes; build NCNN with its C API, pixel operations and file/string support enabled. The [NCNN CI lane](../../.github/workflows/rust.yml) builds a pinned CPU runtime and runs generated-model regressions without downloading detector weights. NCNN does not accept GPU-resident frames; autocam setup rejects that combination instead of silently omitting inference.
