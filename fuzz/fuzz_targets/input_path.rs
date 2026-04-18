//! Fuzz target: input-path validator.
//!
//! `reco_core::source::validate_input_path` is the first-line filter
//! that every file-backed `FrameSource` runs before passing a caller-
//! supplied path to FFmpeg. It was designed to reject garbage paths
//! with a structured `InvalidPathReason` instead of a stringified
//! FFmpeg error, but the input is attacker-controlled (calibration JSON
//! references, CLI args, config files). Make sure no arbitrary bytes
//! can panic, hang, or corrupt the process state before the file is
//! even opened.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::path::Path;

fuzz_target!(|data: &[u8]| {
    // Only run valid UTF-8 inputs - Path::new(&str) requires &str.
    // Non-UTF-8 filesystem paths are a separate attack surface that
    // needs an OsStr-based harness; leave that for a future target.
    if let Ok(s) = std::str::from_utf8(data) {
        if s.len() > 4096 {
            // Skip paths longer than typical PATH_MAX - the validator
            // doesn't enforce a length cap and we don't want the fuzzer
            // to waste cycles on giant string constants.
            return;
        }
        let _ = reco_core::source::validate_input_path(Path::new(s));
    }
});
