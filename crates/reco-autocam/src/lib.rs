//! Automatic camera control for reco.
//!
//! This crate provides detection and direction for sports camera automation:
//!
//! - [`YoloDetector`] - ONNX-based YOLO object detection on raw camera frames
//! - [`BallDirector`] - Ball-following director with plausibility rejection
//! - [`FieldDirector`] - Ball + player tracking for robust football coverage
//! - [`SmoothedDirector`] - Decorator that adds One Euro trajectory smoothing
//! - [`TrackingMode`] - Selects which director to use
//!
//! # Usage
//!
//! ```rust,no_run
//! use reco_autocam::{YoloDetector, BallDirector, SmoothedDirector};
//!
//! let detector = YoloDetector::from_file("ball_v0.onnx")?;
//! let director = BallDirector::new(30.0); // fps
//! let smoothed = SmoothedDirector::new(Box::new(director), 30.0, 15);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod detector;
mod directors;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod gpu_detector;
#[cfg(target_os = "macos")]
mod metal_detector;
mod smoother;
pub use detector::YoloDetector;
pub use directors::{BallDirector, FieldDirector, TrackingMode};
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use gpu_detector::GpuYoloDetector;
#[cfg(target_os = "macos")]
pub use metal_detector::MetalYoloDetector;
pub use smoother::{SmoothedDirector, TrajectorySmoother};

use std::path::Path;
#[cfg(any(feature = "tensorrt", feature = "coreml"))]
use std::path::PathBuf;

use ort::session::Session;
use reco_core::session::StitchSession;

/// Return a persistent cache directory for model engine/compilation caches.
///
/// Resolves to `{platform_cache_dir}/reco/{subdir}`:
/// - Linux: `~/.cache/reco/{subdir}`
/// - macOS: `~/Library/Caches/reco/{subdir}`
/// - Windows: `{FOLDERID_LocalAppData}/reco/{subdir}`
///
/// Falls back to `{temp_dir}/reco/{subdir}` if the platform cache directory
/// is unavailable. Creates the directory (with 0o700 permissions on Unix)
/// if it does not already exist.
#[cfg(any(feature = "tensorrt", feature = "coreml"))]
fn reco_cache_dir(subdir: &str) -> PathBuf {
    let base = dirs::cache_dir().unwrap_or_else(std::env::temp_dir);
    let dir = base.join("reco").join(subdir);

    if !dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&dir) {
            log::warn!("Failed to create cache dir {}: {e}", dir.display());
            // Return it anyway — ORT will get a clear error if it can't write.
            return dir;
        }

        // Restrict permissions on Unix (user-only).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o700);
            if let Err(e) = std::fs::set_permissions(&dir, perms) {
                log::warn!("Failed to set cache dir permissions: {e}");
            }
        }
    }

    dir
}

/// Set up automatic camera control on a [`StitchSession`].
///
/// Configures the appropriate detector (CPU, GPU/CUDA, or Metal) based on
/// the current platform and zero-copy mode, then attaches a [`BallDirector`]
/// for ball-following camera automation.
///
/// Returns `true` if detection was successfully activated, `false` if
/// detection could not be initialized (the session remains usable without
/// autocam in that case).
///
/// # Arguments
///
/// * `session` - The stitch session to attach detection and direction to.
/// * `model_path` - Path to a YOLO ONNX model (or `.mlmodelc` on macOS).
/// * `input_width`, `input_height` - Raw camera frame dimensions.
/// * `fps` - Video frame rate (used for director timing).
/// * `use_zero_copy` - Whether the pipeline is running in zero-copy mode.
/// * `detection_interval` - Run detection every N frames (1 = every frame).
/// * `lead_time` - Director lookahead in seconds (CPU path only).
/// * `tracking_mode` - Which director to use (Ball or Field).
#[allow(clippy::too_many_arguments)]
pub fn setup_autocam(
    session: &mut StitchSession,
    model_path: &str,
    input_width: u32,
    input_height: u32,
    fps: f32,
    use_zero_copy: bool,
    detection_interval: u64,
    lead_time: f64,
    tracking_mode: TrackingMode,
) -> Result<bool, Box<dyn std::error::Error>> {
    let mut detection_active = false;

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    if use_zero_copy {
        match GpuYoloDetector::try_new(model_path, input_width, input_height, 0.10, Vec::new()) {
            Ok(Some(gpu_det)) => {
                session.set_gpu_detector(Box::new(gpu_det));
                detection_active = true;
                log::info!("Autocam: GPU YOLO ball tracking enabled (model: {model_path})");
            }
            Ok(None) => {
                log::warn!("Autocam: NPP not available, ball tracking disabled in zero-copy mode");
            }
            Err(e) => {
                log::warn!("Autocam: GPU detector init failed ({e}), ball tracking disabled");
            }
        }
    }

    #[cfg(target_os = "macos")]
    if use_zero_copy {
        match MetalYoloDetector::try_new(
            model_path,
            session.gpu(),
            input_width,
            input_height,
            0.10,
            Vec::new(),
        ) {
            Ok(metal_det) => {
                session.set_metal_detector(Box::new(metal_det));
                detection_active = true;
                log::info!("Autocam: Metal YOLO ball tracking enabled (model: {model_path})");
            }
            Err(e) => {
                log::warn!("Autocam: Metal detector init failed ({e}), ball tracking disabled");
            }
        }
    }

    if !use_zero_copy {
        let detector = YoloDetector::from_file(model_path)?;
        session.set_detector(Box::new(detector));
        detection_active = true;
        log::info!("Autocam: YOLO ball tracking enabled (model: {model_path})");
    }

    if detection_active {
        if detection_interval > 1 {
            session.set_detection_interval(detection_interval);
            log::info!("Detection interval: every {detection_interval} frames");
        }

        let director: Box<dyn reco_core::director::Director> = match tracking_mode {
            TrackingMode::Ball => {
                let mut d = BallDirector::new(fps);
                if detection_interval > 1 {
                    d.set_detection_interval(detection_interval as u32);
                }
                log::info!("Tracking mode: ball");
                Box::new(d)
            }
            TrackingMode::Field => {
                let mut d = FieldDirector::new(fps);
                if detection_interval > 1 {
                    d.set_detection_interval(detection_interval as u32);
                }
                log::info!("Tracking mode: field (ball + players)");
                Box::new(d)
            }
        };

        let lookahead = if lead_time > 0.0 && !use_zero_copy {
            let frames = (fps as f64 * lead_time).round() as usize;
            if frames > 0 {
                session.set_lookahead(frames);
                log::info!("Director lead time: {lead_time:.1}s ({frames} frames)");
            }
            frames
        } else {
            0
        };

        let smoothed = SmoothedDirector::new(director, fps, lookahead);
        session.set_director(Box::new(smoothed));
    }

    Ok(detection_active)
}

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
        let trt_cache = reco_cache_dir("trt-cache");
        let trt_cache_str = trt_cache.to_string_lossy().into_owned();
        match builder.with_execution_providers([ort::ep::TensorRT::default()
            .with_fp16(true)
            .with_engine_cache(true)
            .with_engine_cache_path(&trt_cache_str)
            .with_timing_cache(true)
            .with_timing_cache_path(&trt_cache_str)
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
        let coreml_cache = reco_cache_dir("coreml-cache");
        let coreml_cache_str = coreml_cache.to_string_lossy().into_owned();
        match builder.with_execution_providers([ort::ep::CoreML::default()
            .with_compute_units(ort::ep::coreml::ComputeUnits::All)
            .with_model_cache_dir(&coreml_cache_str)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn create_ort_session_nonexistent_model_returns_error() {
        let path = PathBuf::from("/tmp/this_model_does_not_exist_12345.onnx");
        let result = create_ort_session(&path, Vec::new());
        assert!(
            result.is_err(),
            "loading a nonexistent model should return an error"
        );
    }
}
