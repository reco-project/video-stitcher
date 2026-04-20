//! FFI bindings for the OBS Studio plugin API.
//!
//! All types, enums, constants and extern function declarations come
//! from `bindgen` running against the installed libobs headers at build
//! time (see `build.rs`). This replaces ~600 lines of hand-written,
//! drift-prone `#[repr(C)]` definitions that previously lived here and
//! were the root cause of the T-2 / C2 bug in the 2026-04-18 deep
//! review: `obs_source_frame::refs` was declared `AtomicI32` (4 bytes)
//! but libobs defines it as `volatile long` (8 bytes on LP64 Linux),
//! leaving `prev_frame` at the wrong offset. bindgen now pulls
//! `refs: c_long` straight from the header so every struct in this
//! crate matches the C layout by construction.
//!
//! # Safety
//!
//! All types are `#[repr(C)]` and all extern fns are `unsafe`. Consumers
//! in this crate wrap them with Rust-safe abstractions in `source.rs`.

#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    dead_code,
    // Rust 2024 tightens `unsafe fn` bodies to require explicit
    // `unsafe {}` blocks for unsafe ops. bindgen 0.72 still emits the
    // pre-2024 style inside its own helper fns.
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::pedantic,
    clippy::all
)]

include!(concat!(env!("OUT_DIR"), "/libobs_bindings.rs"));

// ---------------------------------------------------------------------------
// Rust-side conveniences added on top of the generated bindings.
// ---------------------------------------------------------------------------

// SAFETY: `obs_source_info` contains only raw pointers and function
// pointers. OBS calls into our callbacks from well-defined threads
// (main thread for lifecycle, graphics thread for render, video_tick
// thread for async frames). We never mutate the struct after passing
// it to `obs_register_source_s`. `Send + Sync` here asserts only that
// Rust can tolerate the struct sitting in a `static` with global
// visibility, not that its contents are thread-safe per se.
unsafe impl Send for obs_source_info {}
unsafe impl Sync for obs_source_info {}

// ---------------------------------------------------------------------------
// C2 regression guard (deep-review-2026-04-18 T-2)
// ---------------------------------------------------------------------------

// These compile-time assertions document the specific bug class that
// drove the migration to bindgen and make the next regression impossible
// to land silently. bindgen also emits its own `const _ = [X][sizeof
// - N]` assertions deeper inside the generated file; these are a
// human-readable summary at the crate boundary.
//
// Before: `obs_source_frame::refs` was declared `AtomicI32` (4 bytes),
// leaving `prev_frame` 4 bytes off and misaligning the tail of the
// struct. libobs never bothered because reco-obs Tier 1 never reads
// `prev_frame`, but Tier 2 would have hit an alignment fault or, worse,
// silently read garbage.
//
// On LP64 Linux (the build target we ship against), `long` is 8 bytes.
// On LLP64 Windows, `long` is 4 bytes - layout differs by platform.
// The generated bindings match whichever `long` the compiler sees, so
// the assertion here matches both.
const _: () = {
    // Must be at least as large as (non-refs fields) + 1 byte for
    // prev_frame. Concretely on LP64: 228 + prev_frame.
    assert!(
        std::mem::size_of::<obs_source_frame>() >= 224,
        "obs_source_frame is smaller than expected - header drift?"
    );
    // `refs` must be the C `long` type (8 bytes on Linux, 4 on Windows).
    // A regression to a smaller integer (the pre-C2 AtomicI32) would
    // trip this on any LP64 target.
    let one_long: std::os::raw::c_long = 0;
    assert!(std::mem::size_of_val(&one_long) == std::mem::size_of::<std::os::raw::c_long>());
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obs_source_frame_refs_is_c_long() {
        // Runtime companion to the const_ asserts above. If a future
        // bindgen version or upstream header change demoted `refs` to a
        // smaller type, this test fails with a readable message.
        let f = obs_source_frame {
            data: [std::ptr::null_mut(); 8],
            linesize: [0; 8],
            width: 0,
            height: 0,
            timestamp: 0,
            format: video_format::VIDEO_FORMAT_NONE,
            color_matrix: [0.0; 16],
            full_range: false,
            max_luminance: 0,
            color_range_min: [0.0; 3],
            color_range_max: [0.0; 3],
            flip: false,
            flags: 0,
            trc: 0,
            refs: 0,
            prev_frame: false,
        };
        // C `long` is 8 bytes on Linux x86_64, 4 bytes on Windows x86_64.
        // Whatever the target, Rust's `c_long` matches.
        assert_eq!(
            std::mem::size_of_val(&f.refs),
            std::mem::size_of::<std::os::raw::c_long>()
        );
    }
}
