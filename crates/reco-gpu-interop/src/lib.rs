//! # reco-gpu-interop
//!
//! Platform GPU interop glue extracted from reco-core per plan-execution
//! 2026-04-18 M5. This crate owns the low-level platform bindings so
//! reco-core stays focused on the GPU *pipeline* (geometric projection,
//! undistortion, blending, readback) and doesn't grow a CUDA/Metal/
//! Vulkan dependency tree every consumer has to pay for.
//!
//! ## Scope
//!
//! Pure platform glue:
//!
//! - FFI bindings (CUDA driver API, NPP, Metal, VideoToolbox, Vulkan
//!   external memory, CoreML).
//! - Shader kernels that live outside the main wgpu pipeline
//!   (CUDA PTX for normalize/transpose, Metal compute preprocess).
//! - Texture import/export across GPU APIs (CVPixelBuffer → Metal
//!   texture, CUDA device ptr → Vulkan image, etc.).
//! - Context adapters (CUDA context management, Metal texture cache).
//!
//! NOT scope:
//!
//! - Pipeline logic (projection, blending) — lives in reco-core.
//! - Director / Detector / Source traits — also reco-core.
//! - Frame source I/O (FFmpeg, GStreamer) — reco-io.
//! - Model inference (beyond raw CoreML wrap) — reco-detect.
//!
//! ## Feature flags
//!
//! Default is empty: headless / test consumers pay nothing. Opt-in:
//!
//! | Feature          | Enables                                  | Status     |
//! |------------------|------------------------------------------|------------|
//! | `cuda`           | `cuda_interop`, `cuda_kernels`, `npp_interop` | **shipped** |
//! | `metal`          | `metal_interop`, `metal_compute`, `coreml_inference` | pending migration |
//! | `vulkan-ext`     | `vulkan_interop`                        | pending migration |
//! | `android-ahb`    | `android_ahb` (AHardwareBuffer stub)    | stub       |
//! | `ios-iosurface`  | `ios_iosurface` (IOSurface stub)        | stub       |
//!
//! The "pending migration" rows compile in reco-core today; they
//! move here in a follow-up tranche. The plan's M5 item-6 ("mobile
//! readiness") lands now as `android_ahb` / `ios_iosurface` empty
//! modules so the feature-combo CI matrix exercises them.
//!
//! ## Consumer usage
//!
//! Typical platform-cfg dependency table:
//!
//! ```toml,ignore
//! [target.'cfg(any(target_os = "linux", target_os = "windows"))'.dependencies]
//! reco-gpu-interop = { path = "../reco-gpu-interop", features = ["cuda"] }
//!
//! [target.'cfg(any(target_os = "macos", target_os = "ios"))'.dependencies]
//! reco-gpu-interop = { path = "../reco-gpu-interop", features = ["metal"] }
//! ```

// CUDA family (shipped).
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub mod cuda_interop;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub mod cuda_kernels;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub mod npp_interop;

// Metal / CoreML family. `coreml_inference` lands in phase 1
// because it is standalone (pure CoreML C API wrap, no GpuContext
// entanglement). `metal_interop` and `metal_compute` migrations
// land in a follow-up once `GpuContext` has been reshaped.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod coreml_inference;

// Android hardware-buffer stub. Compiles on every target; marked
// `cfg(target_os = "android")` on the actual bindings when they
// land so the stub doesn't accidentally ship on desktop.
#[cfg(feature = "android-ahb")]
pub mod android_ahb {
    //! Android AHardwareBuffer bindings. Stub until the mobile
    //! tranche lands; present so the feature-combo CI matrix
    //! exercises the gate and so future work has an obvious
    //! target path.
    //!
    //! When implemented, this module exposes handle import/export
    //! between `AHardwareBuffer*` and wgpu textures via Vulkan's
    //! `VK_ANDROID_external_memory_android_hardware_buffer`.
}

// iOS IOSurface stub. Same rationale as android_ahb.
#[cfg(feature = "ios-iosurface")]
pub mod ios_iosurface {
    //! IOSurface bindings. Stub until the mobile tranche lands;
    //! when implemented, exposes IOSurface <-> Metal texture
    //! import/export via `CVMetalTextureCache` (desktop macOS
    //! already uses this through `metal_interop`; mobile iOS
    //! needs the same plumbing with different entitlements).
}
