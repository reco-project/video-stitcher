//! Shared ORT session creation and caching utilities.
//!
//! Used by `CpuYoloDetector`, `OrtGpuDetector`, and
//! `MetalYoloDetector` (macOS) to avoid duplicating session builder
//! setup and model metadata extraction.

use std::path::Path;
#[cfg(any(feature = "tensorrt", feature = "coreml"))]
use std::path::PathBuf;

use ort::session::Session;
use ort::session::builder::SessionBuilder;

use crate::detectors::cpu::parse_onnx_names;
use crate::probe::AiProbeResult;

#[cfg(feature = "tensorrt")]
mod cache;

/// Errors from ORT session creation.
///
/// `RuntimeUnavailable` deliberately carries a plain `String`: with
/// load-dynamic and no loadable runtime, constructing any `ort::Error`
/// calls `CreateStatus` through `api()` and self-deadlocks in ort's init
/// (#446), so no `ort::Error` may exist for this case.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// Unsupported model input or output contract.
    #[error("{0}")]
    UnsupportedModel(String),
    #[error("{0}")]
    RuntimeUnavailable(String),
    #[error(transparent)]
    Ort(#[from] ort::Error),
    /// Reading the ONNX file for the TensorRT engine cache key failed.
    #[cfg(feature = "tensorrt")]
    #[error("Cannot read model {}: {source}", path.display())]
    ModelRead {
        /// Selected model file.
        path: std::path::PathBuf,
        /// Underlying filesystem failure.
        source: std::io::Error,
    },
}

/// One-shot verdict on whether the ONNX Runtime library can be loaded.
///
/// With load-dynamic, ort's first API call dlopens the runtime; when the
/// load fails, ort builds the error through `Error::new`, which calls
/// back into `api()` and re-enters the once-lock whose initializer is
/// still on this thread's stack - a self-deadlock, not an error (#446).
/// ort's failing init path must therefore never execute: the dylib is
/// probed with our own dlopen first, and ort is only entered once the
/// probe proves the load cannot fail.
static ORT_RUNTIME: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();

/// `Ok` when the ONNX Runtime library loaded; `Err` with the reason otherwise.
pub fn ort_runtime_available() -> Result<(), String> {
    ORT_RUNTIME
        .get_or_init(|| {
            probe_ort_dylib()?;
            // The probe proved the dylib loads and is API-compatible, so
            // ort's init cannot reach the deadlocking error path.
            // catch_unwind stays as a backstop for exotic init panics.
            match std::panic::catch_unwind(Session::builder) {
                Ok(Ok(_)) => Ok(()),
                Ok(Err(e)) => Err(format!("ONNX Runtime init failed: {e}")),
                Err(_) => Err("ONNX Runtime initialization panicked".to_string()),
            }
        })
        .clone()
}

/// Verify the ONNX Runtime dynamic library loads and is API-compatible
/// without going through ort (see [`ort_runtime_available`]). Mirrors
/// ort's resolution order: `ORT_DYLIB_PATH`, else the platform soname
/// next to the executable, else the plain soname via the system search
/// path.
#[cfg(feature = "load-dynamic")]
fn probe_ort_dylib() -> Result<(), String> {
    use std::ffi::{CStr, c_char, c_void};

    #[cfg(target_os = "windows")]
    const SONAME: &str = "onnxruntime.dll";
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const SONAME: &str = "libonnxruntime.so";
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const SONAME: &str = "libonnxruntime.dylib";

    // Field order mirrors OrtApiBase in onnxruntime_c_api.h.
    #[repr(C)]
    struct OrtApiBase {
        get_api: unsafe extern "system" fn(u32) -> *const c_void,
        get_version_string: unsafe extern "system" fn() -> *const c_char,
    }

    let name = match std::env::var("ORT_DYLIB_PATH") {
        Ok(s) if !s.is_empty() => std::path::PathBuf::from(s),
        _ => std::path::PathBuf::from(SONAME),
    };
    let path = if name.is_absolute() {
        name
    } else {
        match std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(|d| d.join(&name)))
        {
            Some(p) if p.exists() => p,
            _ => name,
        }
    };

    let lib = unsafe { libloading::Library::new(&path) }.map_err(|e| {
        format!(
            "ONNX Runtime library not found (`{}`: {e}). Install onnxruntime or \
             place the library next to the executable.",
            path.display()
        )
    })?;
    let version = unsafe {
        let base_getter: libloading::Symbol<unsafe extern "system" fn() -> *const OrtApiBase> = lib
            .get(b"OrtGetApiBase")
            .map_err(|e| format!("`{}` is not an ONNX Runtime library: {e}", path.display()))?;
        let base = base_getter();
        if base.is_null() {
            return Err(format!("`{}`: OrtGetApiBase returned null", path.display()));
        }
        CStr::from_ptr(((*base).get_version_string)())
            .to_string_lossy()
            .into_owned()
    };
    let minor = version
        .split('.')
        .nth(1)
        .and_then(|m| m.parse::<u32>().ok())
        .unwrap_or(0);
    if minor < ort::MINOR_VERSION {
        return Err(format!(
            "ONNX Runtime at `{}` is version {version}; ort needs >= 1.{}",
            path.display(),
            ort::MINOR_VERSION
        ));
    }
    // Keep the mapping alive: ort dlopens the same file next (refcounted),
    // and the probe's compatibility verdict must keep holding.
    std::mem::forget(lib);
    log::info!("ONNX Runtime {version} found at {}", path.display());
    Ok(())
}

/// Statically linked ort cannot fail to load.
#[cfg(not(feature = "load-dynamic"))]
fn probe_ort_dylib() -> Result<(), String> {
    Ok(())
}

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

/// Create an ORT session for host-memory input with platform-specific providers and CPU fallback.
///
/// CUDA-memory detectors use a separate checked entry point into the same builder.
///
/// Returns `(session, input_size, labels)` where `input_size` is extracted
/// from the model's BCHW input shape and `labels` are auto-detected from
/// the model's `names` metadata (or `fallback_labels` if provided).
pub fn create_ort_session(
    model_path: &Path,
    fallback_labels: Vec<String>,
) -> Result<(Session, u32, Vec<String>), SessionError> {
    create_session(model_path, fallback_labels, false)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(crate) fn create_cuda_ort_session(
    model_path: &Path,
    fallback_labels: Vec<String>,
) -> Result<(Session, u32, Vec<String>), SessionError> {
    create_session(model_path, fallback_labels, true)
}

/// Register providers with explicit failures, before any model or device input is used.
pub(crate) fn session_builder(
    cuda_input: bool,
    model_path: Option<&Path>,
) -> Result<(SessionBuilder, AiProbeResult), SessionError> {
    // Session::builder() must not run when the runtime cannot load: ort's
    // failing init self-deadlocks building its own error (#446), so the
    // gate's cached verdict is the only safe way to learn availability.
    // The same applies to constructing ANY ort::Error (its constructor
    // calls CreateStatus through api()), hence SessionError.
    if let Err(reason) = ort_runtime_available() {
        return Err(SessionError::RuntimeUnavailable(reason));
    }
    let mut builder = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(ort::Error::from)?;

    let mut report = AiProbeResult {
        providers: Vec::new(),
        errors: Vec::new(),
    };
    for (name, supports_cuda, provider) in configured_providers(model_path)? {
        if cuda_input && !supports_cuda {
            continue;
        }
        builder = match builder.with_execution_providers([provider.error_on_failure()]) {
            Ok(builder) => {
                log::info!("ORT: {name} execution provider registered");
                report.providers.push(name.into());
                builder
            }
            Err(error) => {
                log::warn!("ORT: {name} registration failed: {error}");
                report.errors.push(format!("{name}: {error}"));
                error.recover()
            }
        };
    }
    if cuda_input && report.providers.is_empty() {
        return Err(SessionError::RuntimeUnavailable(format!(
            "CUDA input requires a registered TensorRT or CUDA execution provider. {}",
            report.errors.join("; ")
        )));
    }
    // ORT supplies CPU fallback for host inputs and unsupported graph partitions.
    report.providers.push("CPU".into());
    Ok((builder, report))
}

fn configured_providers(
    model_path: Option<&Path>,
) -> Result<Vec<(&'static str, bool, ort::ep::ExecutionProviderDispatch)>, SessionError> {
    #[cfg(not(feature = "tensorrt"))]
    let _ = model_path;
    Ok(vec![
        #[cfg(feature = "tensorrt")]
        ("TensorRT", true, {
            let cache = reco_cache_dir("trt-cache");
            let path = cache.to_string_lossy();
            let mut provider = ort::ep::TensorRT::default()
                .with_fp16(true)
                .with_engine_cache(model_path.is_some())
                .with_engine_cache_path(&path)
                .with_timing_cache(true)
                .with_timing_cache_path(&path)
                .with_builder_optimization_level(3);
            if let Some(model) = model_path {
                let prefix = cache::engine_prefix(model, ort::info()).map_err(|source| {
                    SessionError::ModelRead {
                        path: model.to_owned(),
                        source,
                    }
                })?;
                provider = provider.with_engine_cache_prefix(prefix);
            }
            provider.build()
        }),
        #[cfg(feature = "cuda")]
        ("CUDA", true, ort::ep::CUDA::default().build()),
        #[cfg(feature = "coreml")]
        ("CoreML", false, {
            let cache = reco_cache_dir("coreml-cache");
            ort::ep::CoreML::default()
                .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                .with_model_cache_dir(cache.to_string_lossy())
                .build()
        }),
        #[cfg(target_os = "windows")]
        ("DirectML", false, ort::ep::DirectML::default().build()),
    ])
}

fn create_session(
    model_path: &Path,
    fallback_labels: Vec<String>,
    cuda_input: bool,
) -> Result<(Session, u32, Vec<String>), SessionError> {
    let (mut builder, _) = session_builder(cuda_input, Some(model_path))?;
    let session = {
        // ORT model loading registers custom-op domains in a global registry.
        // Serialize our loads to avoid registration races, never inference.
        static MODEL_LOAD: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _load = MODEL_LOAD.lock().unwrap_or_else(|e| e.into_inner());
        builder.commit_from_file(model_path)?
    };

    // Extract input size from model metadata (BCHW layout).
    if session.inputs().len() != 1 || session.outputs().len() != 1 {
        return Err(SessionError::UnsupportedModel(
            "expected one image input and one detection output".into(),
        ));
    }
    let input_size = match session.inputs()[0].dtype() {
        ort::value::ValueType::Tensor {
            shape,
            ty: ort::value::TensorElementType::Float32,
            ..
        } => match &shape[..] {
            [1 | -1, 3, h, w] if h == w && *h > 0 => u32::try_from(*h).map_err(|_| {
                SessionError::UnsupportedModel("input dimension exceeds supported range".into())
            })?,
            [1 | -1, 3, -1, -1] => 1280,
            _ => {
                return Err(SessionError::UnsupportedModel(format!(
                    "expected square RGB input [1, 3, H, H], got {shape:?}"
                )));
            }
        },
        _ => {
            return Err(SessionError::UnsupportedModel(
                "expected a float32 input tensor".into(),
            ));
        }
    };

    // Auto-detect labels from model metadata if not provided.
    let labels = if fallback_labels.is_empty() {
        parse_onnx_names(&session).unwrap_or_default()
    } else {
        fallback_labels
    };

    Ok((session, input_size, labels))
}

#[cfg(test)]
mod provider_tests;

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

    #[test]
    fn ort_gate_returns_same_verdict_on_repeat_calls() {
        // The second call must return (not deadlock) and agree with the
        // first - the load-dynamic missing-library case wedges ort's
        // once-lock if Session::builder is entered twice (#446).
        let first = ort_runtime_available();
        let second = ort_runtime_available();
        assert_eq!(first.is_ok(), second.is_ok());
    }
}
