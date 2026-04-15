//! Shared ORT session creation and caching utilities.
//!
//! Used by `CpuYoloDetector`, `OrtGpuDetector`, and
//! `MetalYoloDetector` (macOS) to avoid duplicating session builder
//! setup and model metadata extraction.

use std::path::Path;
#[cfg(any(feature = "tensorrt", feature = "coreml"))]
use std::path::PathBuf;

use ort::session::Session;

use crate::detectors::cpu::parse_onnx_names;

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
pub fn reco_cache_dir(subdir: &str) -> PathBuf {
    let base = dirs::cache_dir().unwrap_or_else(std::env::temp_dir);
    let dir = base.join("reco").join(subdir);

    // create_dir_all is a no-op if the directory already exists, avoiding
    // the TOCTOU race of checking exists() then creating.
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::warn!("Failed to create cache dir {}: {e}", dir.display());
        // Return it anyway - ORT will get a clear error if it can't write.
        return dir;
    }

    // Restrict permissions on Unix (user-only). Safe to call on an
    // existing directory - it just updates the mode.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o700);
        if let Err(e) = std::fs::set_permissions(&dir, perms) {
            log::warn!("Failed to set cache dir permissions: {e}");
        }
    }

    dir
}

/// Create an ORT session with common settings and platform-specific EPs.
///
/// Shared by [`CpuYoloDetector`](crate::CpuYoloDetector) and
/// [`OrtGpuDetector`](crate::OrtGpuDetector) to avoid duplicating
/// the builder + EP setup + model metadata extraction logic.
///
/// Returns `(session, input_size, labels)` where `input_size` is extracted
/// from the model's BCHW input shape and `labels` are auto-detected from
/// the model's `names` metadata (or `fallback_labels` if provided).
pub fn create_ort_session(
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
            if h > 0 {
                h as u32
            } else {
                log::warn!("Model input has dynamic height, defaulting to 1280");
                1280
            }
        }
        _ => {
            log::warn!("Could not determine model input size from metadata, defaulting to 1280");
            1280
        }
    };

    // Auto-detect labels from model metadata if not provided.
    let labels = if fallback_labels.is_empty() {
        parse_onnx_names(&session).unwrap_or_else(|| {
            log::warn!("No class names in model metadata, assuming single-class 'ball' model");
            vec!["ball".into()]
        })
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
