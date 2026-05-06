//! Platform interop layer for zero-copy GPU decode paths.
//!
//! Each submodule bridges a platform-specific GPU API (CUDA, Vulkan,
//! Metal, D3D11, DMA-BUF) with wgpu so that decoded video frames can
//! be imported as GPU textures without a CPU round-trip.
//!
//! The [`zero_copy`] submodule defines cross-platform types that the
//! stitch session and decode threads share regardless of backend.

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub mod cuda;
#[cfg(target_os = "windows")]
pub mod d3d11;
#[cfg(target_os = "linux")]
pub mod dmabuf;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod metal;
#[cfg(target_os = "linux")]
pub mod vulkan;
pub mod zero_copy;
