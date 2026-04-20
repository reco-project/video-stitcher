//! Build script for reco-obs.
//!
//! Generates FFI bindings for the libobs C API at build time using the
//! installed libobs headers at `/usr/include/obs/` (or wherever `bindgen`
//! can find them via `clang -I` + `pkg-config`).
//!
//! # Why bindgen?
//!
//! Before this script, `src/ffi.rs` was ~600 lines of hand-written
//! `#[repr(C)]` struct definitions mirroring libobs headers. Hand-written
//! FFI is the root cause of the T-2 / C2 bug in the 2026-04-18 deep
//! review: `obs_source_frame::refs` was declared `AtomicI32` (4 bytes)
//! but libobs defines it as `volatile long` (8 bytes on Linux LP64),
//! which leaves `prev_frame` at the wrong offset and misaligns every
//! subsequent field.
//!
//! bindgen pulls struct layouts directly from the C headers, so sizeof
//! and field offsets are always correct for whatever libobs version the
//! plugin is built against.
//!
//! # Header source
//!
//! The OBS plugin is built against libobs 30.0.2 headers (the Ubuntu
//! `libobs-dev` package or the OBS project PPA). OBS 32.x is ABI-
//! compatible as long as the plugin passes `std::mem::size_of::<
//! obs_source_info>()` to `obs_register_source_s`, which we do.
//!
//! Override the header dir with the `OBS_INCLUDE_DIR` env var if your
//! installation uses a different path.

use std::env;
use std::path::PathBuf;

fn main() {
    let include_dir = env::var("OBS_INCLUDE_DIR").unwrap_or_else(|_| {
        // Default to the standard Linux libobs-dev install location.
        "/usr/include/obs".to_string()
    });

    let obs_header = PathBuf::from(&include_dir).join("obs.h");
    if !obs_header.exists() {
        panic!(
            "libobs header not found at {:?}. Install `libobs-dev` (Ubuntu) \
             or set OBS_INCLUDE_DIR to the directory containing obs.h.",
            obs_header,
        );
    }

    // Rerun if the header dir changes (new OBS install, version bump).
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OBS_INCLUDE_DIR");
    for entry in
        std::fs::read_dir(&include_dir).unwrap_or_else(|e| panic!("cannot read {include_dir}: {e}"))
    {
        let entry = entry.unwrap();
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }

    // Also pull in the frontend API header so we can hook
    // OBS_FRONTEND_EVENT_RECORDING_STARTED / STOPPED and mirror
    // OBS's global record/stream state to the replay recorder
    // (FRICTION reco-obs A20 — replay follows OBS Record/Stream
    // button by default). Optional: if the header is missing we
    // skip it and the replay feature falls back to the
    // "always record" mode.
    let frontend_header = PathBuf::from(&include_dir).join("obs-frontend-api.h");
    let have_frontend = frontend_header.exists();

    let mut builder = bindgen::Builder::default().header(obs_header.to_string_lossy().into_owned());
    if have_frontend {
        builder = builder.header(frontend_header.to_string_lossy().into_owned());
    }
    let bindings = builder
        .clang_arg(format!("-I{include_dir}"))
        // libobs headers are C, and we want standard types (not ptrdiff).
        .clang_arg("-std=c11")
        // Only generate bindings for items we actually use. Bindgen
        // pulls transitive types automatically, so the struct fields
        // are covered even if the field type isn't in the allowlist.
        .allowlist_type("obs_.*")
        .allowlist_type("gs_.*")
        .allowlist_type("audio_output_data")
        .allowlist_type("video_format")
        .allowlist_type("video_trc")
        .allowlist_function("obs_.*")
        .allowlist_function("gs_.*")
        .allowlist_var("OBS_.*")
        .allowlist_var("GS_.*")
        .allowlist_var("MAX_AV_PLANES")
        // Rust-style enums for the ones we pattern-match in source.rs.
        // Others default to `Consts` (no breakage if libobs adds new
        // out-of-range values).
        .rustified_enum("obs_source_type")
        .rustified_enum("obs_icon_type")
        .rustified_enum("obs_media_state")
        .rustified_enum("gs_color_space")
        .rustified_enum("gs_color_format")
        .rustified_enum("obs_path_type")
        .rustified_enum("obs_mouse_button_type")
        .rustified_enum("video_format")
        .rustified_enum("obs_combo_type")
        .rustified_enum("obs_combo_format")
        .rustified_enum("obs_base_effect_type")
        // Don't try to derive Default on opaque pointers or fn-pointer
        // structs: bindgen correctly emits the field as a union/ptr and
        // `impl Default` would need `std::mem::zeroed()` which is unsafe.
        .derive_default(false)
        .derive_debug(true)
        // Layout-altering options: keep C layout guarantees.
        .layout_tests(true)
        // Rust features we want in emitted code.
        .generate_comments(false)
        .formatter(bindgen::Formatter::Rustfmt)
        // Hide the low-level __ prefixed compiler helpers but keep SIMD
        // intrinsic types as opaque: libobs pulls `__m128` etc. in via
        // math/vector headers and some struct fields reference them, so
        // blocklisting them entirely creates dangling references.
        // Opaque emits a byte-array placeholder with the correct size
        // and alignment so the containing struct's layout stays right.
        .blocklist_type("__va_list_tag")
        .blocklist_type("_.*Iterator")
        .opaque_type(r"__m\d+.*")
        // Suppress ordinary re-exports of primitive types.
        .blocklist_type("size_t")
        // The `video_format` union/enum is exposed via typedef; make
        // sure bindgen picks it up by name.
        .generate()
        .expect("failed to generate libobs bindings; is libclang installed?");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out_path = out_dir.join("libobs_bindings.rs");
    bindings
        .write_to_file(&out_path)
        .unwrap_or_else(|e| panic!("failed to write bindings to {out_path:?}: {e}"));

    // Link the frontend-api shared library so symbols like
    // `obs_frontend_add_event_callback` resolve at plugin-load
    // time. libobs itself is loaded by OBS and doesn't need an
    // explicit link line here, but the frontend API is a separate
    // shared object (`libobs-frontend-api.so`) that plugins link
    // against explicitly. Only emit the link line when the header
    // was found, so builds against bare libobs installations still
    // succeed (they'll miss the OBS-record auto-follow behavior
    // but the plugin still loads).
    if have_frontend {
        println!("cargo:rustc-link-lib=obs-frontend-api");
        println!("cargo:rustc-cfg=have_frontend_api");
    }
    println!("cargo:rustc-check-cfg=cfg(have_frontend_api)");

    // Compile the `blog` shim so Rust can call OBS's variadic
    // logger through a fixed-arity entry point. libobs is loaded
    // into the process by OBS before any plugin, so the shim's
    // `blog(...)` reference resolves at plugin load time without
    // us needing to `-lobs` here. `cc` picks up system headers
    // automatically on Linux/macOS.
    let shim_path = PathBuf::from("src/blog_shim.c");
    println!("cargo:rerun-if-changed={}", shim_path.display());
    cc::Build::new()
        .file(&shim_path)
        .include(&include_dir)
        .flag_if_supported("-std=c11")
        // OBS ships headers that rely on GNU extensions (typeof
        // etc.) on Linux. `-D_GNU_SOURCE` avoids any surprises.
        .define("_GNU_SOURCE", None)
        .compile("reco_obs_blog_shim");
}
