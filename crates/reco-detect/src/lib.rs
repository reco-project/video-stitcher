//! Detection backends for reco.
//!
//! This crate owns all object detection implementations:
//!
//! - [`CpuYoloDetector`] - ONNX-based YOLO on CPU (all platforms)
//! - [`OrtGpuDetector`] - ONNX Runtime + TensorRT/CUDA EP on GPU-resident NV12 frames
//! - [`MetalYoloDetector`] - Metal compute + CoreML/ORT on macOS zero-copy frames
//! - [`TrtGpuDetector`] - Native TensorRT inference (no ORT dependency)
//!
//! All ORT-based detectors are gated behind the `ort` feature (on by default).
//! The native TensorRT backend is gated behind `tensorrt-native`.

pub mod detectors;
#[cfg(feature = "ort")]
pub mod ort_session;

// Re-export detector types at crate root for convenience.
#[cfg(feature = "ort")]
pub use detectors::cpu::CpuYoloDetector;
#[cfg(all(feature = "ort", target_os = "macos"))]
pub use detectors::metal::MetalYoloDetector;
#[cfg(all(feature = "ort", any(target_os = "linux", target_os = "windows")))]
pub use detectors::ort_gpu::OrtGpuDetector;
#[cfg(feature = "tensorrt-native")]
pub use detectors::trt::TrtGpuDetector;

// Re-export shared utilities.
pub use detectors::postprocess;
pub use detectors::read_labels_file;
#[cfg(feature = "ort")]
pub use ort_session::create_ort_session;
#[cfg(any(feature = "tensorrt", feature = "coreml"))]
pub use ort_session::reco_cache_dir;
