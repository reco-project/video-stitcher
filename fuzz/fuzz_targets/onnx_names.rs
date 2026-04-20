//! Fuzz target: ONNX `names` metadata parser.
//!
//! The `names` metadata string inside an ONNX model is user-controlled
//! (whoever made the .onnx). A crafted value like `{999999999: 'ball'}`
//! drove an OOM-sized `Vec::with_capacity` before the N-C1 fix in M1.
//! This target feeds arbitrary strings to the parser to make sure the
//! cap (and every other parse branch) holds up under pathological
//! inputs.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        // Hard cap on input size so the fuzzer does not try megabyte
        // payloads: the real metadata never exceeds a few KB and we
        // want to find logic bugs, not OS-level allocator stress.
        if s.len() > 16 * 1024 {
            return;
        }
        let _ = reco_detect::__fuzz_parse_names_dict_string(s);
    }
});
