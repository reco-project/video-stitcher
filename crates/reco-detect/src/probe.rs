//! Runtime registration probe for ONNX Runtime execution providers.
//!
//! Uses the same provider configuration as model creation. Registration is not
//! a guarantee that a particular model or input residency is supported.

/// Inference providers registered at runtime, in preference order.
///
/// The GUI uses this information before a model has been selected.
#[derive(Debug, Clone)]
pub struct AiProbeResult {
    /// Successfully registered providers, including ORT's CPU fallback.
    pub providers: Vec<String>,
    /// Runtime or provider registration failures.
    pub errors: Vec<String>,
}

impl AiProbeResult {
    /// Whether ORT and at least its CPU fallback are available.
    pub fn is_available(&self) -> bool {
        !self.providers.is_empty()
    }
}

/// Probe provider registration without loading a model.
///
/// Model compilation and inference can still fail after successful registration.
/// This probe does not initialize or certify native TensorRT engine files.
pub fn probe_execution_providers() -> AiProbeResult {
    match crate::ort_session::session_builder(false, None) {
        Ok((_, report)) => report,
        Err(error) => AiProbeResult {
            providers: Vec::new(),
            errors: vec![error.to_string()],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_agrees_with_runtime_gate() {
        let result = probe_execution_providers();
        if crate::ort_session::ort_runtime_available().is_ok() {
            assert!(result.providers.contains(&"CPU".to_string()));
            assert!(result.is_available());
        } else {
            assert!(!result.is_available());
            assert!(!result.errors.is_empty());
        }
    }

    #[test]
    fn unavailable_when_empty() {
        let result = AiProbeResult {
            providers: vec![],
            errors: vec!["ORT init: library not found".into()],
        };
        assert!(!result.is_available());
    }
}
