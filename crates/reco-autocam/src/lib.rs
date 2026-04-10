//! Automatic camera control for reco.
//!
//! This crate provides implementations of the [`reco_core`] detection and
//! direction traits for sports camera automation:
//!
//! - [`YoloDetector`] — ONNX-based YOLO object detection on raw camera frames
//! - [`BallDirector`] — Ball-following director with smoothing and state machine logic
//!
//! # Usage
//!
//! ```rust,no_run
//! use reco_autocam::{YoloDetector, BallDirector};
//!
//! let detector = YoloDetector::from_file("ball_v0.onnx")?;
//! let director = BallDirector::new(30.0); // fps
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod detector;
mod director;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod gpu_detector;
#[cfg(target_os = "macos")]
mod metal_detector;
pub use detector::YoloDetector;
pub use director::BallDirector;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use gpu_detector::GpuYoloDetector;
#[cfg(target_os = "macos")]
pub use metal_detector::MetalYoloDetector;

use std::path::Path;

use ort::session::Session;

/// Create an ORT session with common settings and platform-specific EPs.
///
/// Shared by [`YoloDetector`] and [`GpuYoloDetector`] to avoid duplicating
/// the builder + EP setup + model metadata extraction logic.
///
/// Returns `(session, input_size, labels)` where `input_size` is extracted
/// from the model's BCHW input shape and `labels` are auto-detected from
/// the model's `names` metadata (or `fallback_labels` if provided).
pub(crate) fn create_ort_session(
    model_path: &Path,
    fallback_labels: Vec<String>,
) -> Result<(Session, u32, Vec<String>), ort::Error> {
    #[allow(unused_mut)]
    let mut builder = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?;

    // Try TensorRT EP first (JIT-compiles for any GPU arch including Blackwell),
    // then CUDA EP, then fall back to CPU.
    #[cfg(feature = "tensorrt")]
    let mut builder = {
        match builder.with_execution_providers([ort::ep::TensorRT::default()
            .with_fp16(true)
            .with_engine_cache(true)
            .with_engine_cache_path("/tmp/reco-trt-cache")
            .with_timing_cache(true)
            .with_timing_cache_path("/tmp/reco-trt-cache")
            .with_builder_optimization_level(3)
            .build()])
        {
            Ok(b) => {
                log::info!("ORT: TensorRT execution provider enabled");
                b
            }
            Err(e) => {
                log::warn!("ORT: TensorRT EP failed ({e}), falling back");
                e.recover()
            }
        }
    };

    // Try CUDA execution provider, fall back to CPU.
    #[cfg(all(feature = "cuda", not(feature = "tensorrt")))]
    let mut builder = {
        match builder.with_execution_providers([ort::ep::CUDA::default().build()]) {
            Ok(b) => {
                log::info!("ORT: CUDA execution provider enabled");
                b
            }
            Err(e) => {
                log::warn!("ORT: CUDA EP failed ({e}), falling back to CPU");
                e.recover()
            }
        }
    };

    // CoreML EP for macOS.
    #[cfg(feature = "coreml")]
    let mut builder = {
        match builder.with_execution_providers([ort::ep::CoreML::default()
            .with_compute_units(ort::ep::coreml::ComputeUnits::All)
            .with_model_cache_dir("/tmp/reco-coreml-cache")
            .build()])
        {
            Ok(b) => {
                log::info!("ORT: CoreML execution provider enabled");
                b
            }
            Err(e) => {
                log::warn!("ORT: CoreML EP failed ({e}), falling back to CPU");
                e.recover()
            }
        }
    };

    let session = builder.commit_from_file(model_path)?;

    // Extract input size from model metadata (BCHW layout).
    let input_size = match session.inputs()[0].dtype() {
        ort::value::ValueType::Tensor { shape, .. } => {
            let h = shape[2];
            if h > 0 { h as u32 } else { 1280 }
        }
        _ => 1280,
    };

    // Auto-detect labels from model metadata if not provided.
    let labels = if fallback_labels.is_empty() {
        detector::parse_onnx_names(&session).unwrap_or_else(|| vec!["ball".into()])
    } else {
        fallback_labels
    };

    Ok((session, input_size, labels))
}
