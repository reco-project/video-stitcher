//! Runtime probe for ONNX Runtime execution providers.
//!
//! [`probe_execution_providers`] tests which inference backends are
//! actually available on the current machine, going beyond compile-time
//! feature flags. A build compiled with `features = ["tensorrt"]` will
//! report TensorRT as unavailable if the TensorRT shared libraries are
//! not installed or if the CUDA driver is missing.
//!
//! The GUI and CLI use the probe result to display an accurate
//! "AI: TensorRT" / "AI: CPU-only" / "AI: unavailable" status line at
//! startup, replacing the previous compile-time `cfg!()` checks.

use ort::session::Session;

/// Result of probing available ONNX Runtime execution providers.
///
/// Returned by [`probe_execution_providers`]. The GUI displays this
/// in the export dialog; the CLI prints one summary line on startup.
#[derive(Debug, Clone)]
pub struct AiProbeResult {
    /// Names of execution providers that initialized successfully
    /// (e.g. `["TensorRT", "CPU"]`).
    pub providers: Vec<String>,
    /// Whether any available provider can process GPU-resident NV12
    /// frames directly (TensorRT or native TRT). When `false`, only
    /// CPU-decoded frames can be detected.
    pub can_run_on_gpu_frames: bool,
    /// Errors from providers that were compiled in but failed to load
    /// at runtime (e.g. "TensorRT: library not found").
    pub errors: Vec<String>,
}

impl AiProbeResult {
    /// The best available provider name, or "unavailable" if none.
    pub fn best_provider(&self) -> &str {
        self.providers.first().map_or("unavailable", |s| s.as_str())
    }

    /// Whether any inference backend is available (at least CPU).
    pub fn is_available(&self) -> bool {
        !self.providers.is_empty()
    }
}

/// Probe which ONNX Runtime execution providers are available at runtime.
///
/// Tests each compiled-in EP by attempting to register it on a fresh
/// session builder. No model file is needed. The probe takes ~1-50ms
/// depending on how many EPs are compiled in and whether GPU drivers
/// are responsive.
///
/// The returned [`AiProbeResult`] lists all working providers (best
/// first) and any errors from EPs that failed to initialize.
pub fn probe_execution_providers() -> AiProbeResult {
    let mut providers = Vec::new();
    let mut errors = Vec::new();
    #[allow(unused_mut)]
    let mut can_run_on_gpu_frames = false;

    // Test ORT itself loads.
    let builder = match Session::builder() {
        Ok(b) => Some(b),
        Err(e) => {
            errors.push(format!("ORT init: {e}"));
            None
        }
    };

    if let Some(_builder) = builder {
        // TensorRT EP
        #[cfg(feature = "tensorrt")]
        {
            match Session::builder()
                .and_then(|b| b.with_execution_providers([ort::ep::TensorRT::default().build()]))
            {
                Ok(_) => {
                    providers.push("TensorRT".into());
                    can_run_on_gpu_frames = true;
                }
                Err(e) => {
                    errors.push(format!("TensorRT: {e}"));
                }
            }
        }

        // CUDA EP (only when TensorRT is not compiled in, matching ort_session.rs logic)
        #[cfg(all(feature = "cuda", not(feature = "tensorrt")))]
        {
            match Session::builder()
                .and_then(|b| b.with_execution_providers([ort::ep::CUDA::default().build()]))
            {
                Ok(_) => {
                    providers.push("CUDA".into());
                    // CUDA EP alone can't handle NV12 device pointers
                    // without NPP preprocessing, so can_run_on_gpu_frames
                    // stays false unless TensorRT succeeded above.
                }
                Err(e) => {
                    errors.push(format!("CUDA: {e}"));
                }
            }
        }

        // CoreML EP (macOS)
        #[cfg(feature = "coreml")]
        {
            match Session::builder().and_then(|b| {
                b.with_execution_providers([ort::ep::CoreML::default()
                    .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                    .build()])
            }) {
                Ok(_) => {
                    providers.push("CoreML".into());
                }
                Err(e) => {
                    errors.push(format!("CoreML: {e}"));
                }
            }
        }

        // CPU is always available when ORT loads.
        providers.push("CPU".into());
    }

    // Native TensorRT (no ORT dependency)
    #[cfg(feature = "tensorrt-native")]
    {
        providers.insert(0, "TensorRT (native)".into());
        can_run_on_gpu_frames = true;
    }

    AiProbeResult {
        providers,
        can_run_on_gpu_frames,
        errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_at_least_cpu() {
        let result = probe_execution_providers();
        // On any machine where ORT loads, CPU should be available.
        assert!(
            result.providers.contains(&"CPU".to_string()),
            "CPU provider should always be available, got: {:?}",
            result.providers
        );
        assert!(result.is_available());
    }

    #[test]
    fn best_provider_returns_first() {
        let result = AiProbeResult {
            providers: vec!["TensorRT".into(), "CPU".into()],
            can_run_on_gpu_frames: true,
            errors: vec![],
        };
        assert_eq!(result.best_provider(), "TensorRT");
    }

    #[test]
    fn best_provider_unavailable_when_empty() {
        let result = AiProbeResult {
            providers: vec![],
            can_run_on_gpu_frames: false,
            errors: vec!["ORT init: lib not found".into()],
        };
        assert_eq!(result.best_provider(), "unavailable");
        assert!(!result.is_available());
    }
}
