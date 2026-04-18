//! Detection backends for reco.
//!
//! This crate owns all object detection implementations:
//!
//! - [`CpuYoloDetector`] - ONNX-based YOLO on CPU (all platforms)
//! - [`OrtGpuDetector`] - ONNX Runtime + TensorRT/CUDA EP on GPU-resident NV12 frames
//! - `MetalYoloDetector` - Metal compute + CoreML/ORT on macOS zero-copy frames (cfg macos)
//! - `TrtGpuDetector` - Native TensorRT inference, no ORT dependency (feature `tensorrt-native`)
//! - `NcnnYoloDetector` - NCNN inference optimized for ARM/RPi5 (feature `ncnn`)
//!
//! All ORT-based detectors are gated behind the `ort` feature (on by default).
//! The native TensorRT backend is gated behind `tensorrt-native`.
//! The NCNN backend is gated behind `ncnn`.

pub mod detectors;
#[cfg(feature = "ort")]
pub mod ort_session;
#[cfg(feature = "ort")]
pub mod probe;

// Re-export detector types at crate root for convenience.
#[cfg(feature = "ort")]
pub use detectors::cpu::CpuYoloDetector;
#[cfg(all(feature = "ort", target_os = "macos"))]
pub use detectors::metal::MetalYoloDetector;
#[cfg(feature = "ncnn")]
pub use detectors::ncnn::NcnnYoloDetector;
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
#[cfg(feature = "ort")]
pub use probe::{AiProbeResult, probe_execution_providers};

/// Fuzz entry-point re-export for the ONNX `names` metadata parser.
///
/// `__` prefix + `doc(hidden)` keeps this out of the public API while
/// letting the `reco-fuzz` subcrate drive the parser directly. See
/// `fuzz/fuzz_targets/onnx_names.rs` and the N-C1 OOM cap fix.
#[cfg(feature = "ort")]
#[doc(hidden)]
pub fn __fuzz_parse_names_dict_string(names: &str) -> Option<Vec<String>> {
    detectors::cpu::__fuzz_parse_names_dict_string(names)
}
