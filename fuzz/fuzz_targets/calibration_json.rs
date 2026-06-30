//! Fuzz target: calibration JSON parser + validator.
//!
//! Feeds arbitrary bytes to `serde_json::from_slice::<Calibration>`
//! followed by `Calibration::validate`. Exercises two attack
//! surfaces at once:
//!
//! 1. serde_json deserialization (zip bomb / depth / overflow parse).
//! 2. The validator's finite-float + sync_offset bounds checks that
//!    landed in M1 (B-10, B-29 defense chain).
//!
//! Expected behavior: always return Err, never panic, never hang.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 1_048_576 {
        // Matches MAX_CALIBRATION_FILE_SIZE in reco-core - from_file
        // enforces the cap; replicate here so the fuzz corpus doesn't
        // waste cycles on inputs we'd reject at the size check.
        return;
    }
    if let Ok(parsed) = serde_json::from_slice::<reco_core::calibration::Calibration>(data) {
        // validate() is the only non-parse-time invariant check. A
        // malicious calibration that parses cleanly but fails validate
        // must never panic here - only return the structured error.
        let _ = parsed.validate();
    }
});
