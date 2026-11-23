//! Content identity for persistent inference engines.

use std::io::Read;
use std::path::Path;

/// ONNX Runtime can reuse an engine across different static shapes with the same graph name.
pub(super) fn engine_prefix(model: &Path, runtime: &str) -> std::io::Result<String> {
    let mut file = std::fs::File::open(model)?;
    let mut hash = hmac_sha256::Hash::new();
    hash.update(runtime.as_bytes());
    hash.update([0]);
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    let digest = hash.finalize();
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok(format!("reco-{hex}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_and_runtime_determine_identity() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.onnx");
        let second = dir.path().join("second.onnx");
        std::fs::write(&first, b"model-a").unwrap();
        std::fs::copy(&first, &second).unwrap();
        let original = engine_prefix(&first, "runtime-a").unwrap();
        assert_eq!(original, engine_prefix(&second, "runtime-a").unwrap());
        assert_ne!(original, engine_prefix(&first, "runtime-b").unwrap());
        // Same path and length cannot make a replacement reuse an old engine.
        std::fs::write(&first, b"model-b").unwrap();
        assert_ne!(original, engine_prefix(&first, "runtime-a").unwrap());
        std::fs::write(&first, b"model-a").unwrap();
        assert_eq!(original, engine_prefix(&first, "runtime-a").unwrap());
    }

    #[test]
    fn complete_file_is_hashed_with_bounded_reads() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("large.onnx");
        let mut bytes = vec![7; 128 * 1024 + 1];
        std::fs::write(&model, &bytes).unwrap();
        let original = engine_prefix(&model, "runtime").unwrap();
        *bytes.last_mut().unwrap() = 8;
        std::fs::write(&model, &bytes).unwrap();
        assert_ne!(original, engine_prefix(&model, "runtime").unwrap());
    }

    #[test]
    fn missing_model_has_no_cache_identity() {
        let dir = tempfile::tempdir().unwrap();
        assert!(engine_prefix(&dir.path().join("missing.onnx"), "runtime").is_err());
    }
}
