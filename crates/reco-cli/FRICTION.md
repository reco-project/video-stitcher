# CLI API friction

## NVMM detector preparation requirements

The NVMM camera path in [`camera.rs`](src/camera.rs) calls [`StitchSession::setup_nvmm_detection`](../reco-core/src/session/detection_dispatch.rs) with a hardcoded 1280-square input. The attached [`Detector`](../reco-core/src/detect/detector.rs) does not expose its prepared-input requirements, even though native TensorRT already knows the model's dimensions. A model with another input size cannot be configured correctly through this CLI path. Prepared CUDA RGBA frames carry their actual size and pitch, and TensorRT rejects a size mismatch instead of reading with a guessed layout; that validation does not choose the correct allocation size. The setup needs the detector's requirements without a consumer-side model parser or another guessed constant.
