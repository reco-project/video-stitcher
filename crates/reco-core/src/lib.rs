//! # reco-core
//!
//! GPU-accelerated panoramic video stitching engine.
//!
//! `reco-core` is the foundation of the Reco Video Stitcher. It provides the
//! complete GPU pipeline for stitching two camera feeds into a seamless panoramic
//! view, using [`wgpu`] for cross-platform GPU acceleration.
//!
//! ## Architecture
//!
//! The pipeline processes frames through these stages:
//!
//! ```text
//! Input frames (YUV/NV12)
//!   → GPU texture upload
//!   → Fisheye undistortion (per-camera)
//!   → Composite two planes with blending
//!   → Viewport crop (director-controlled pan)
//!   → Output frame for encoding
//! ```
//!
//! ## Geometric Model
//!
//! Two camera planes are arranged in an L-shape in 3D space:
//! - Left camera plane lies in the X-Z plane (faces right)
//! - Right camera plane lies in the Y-Z plane (faces forward)
//! - A virtual camera sits at the corner, looking at both planes
//! - Panning is achieved by rotating the virtual camera (yaw/pitch)
//!
//! ## Modularity
//!
//! The crate defines traits for pluggable components:
//! - [`source::FrameSource`] — delivers stereo frame pairs (files, cameras, streams)
//! - [`detector::Detector`] — detects objects in raw frames (e.g. ball tracking)
//! - [`director::Director`] — controls where the virtual camera pans
//! - [`encoder::Encoder`] — receives stitched GPU frames for encoding
//!
//! ## Usage
//!
//! ```rust,no_run
//! use reco_core::calibration::MatchCalibration;
//!
//! // Load calibration from a v1-compatible JSON file
//! let json = std::fs::read_to_string("match.json").unwrap();
//! let calibration: MatchCalibration = serde_json::from_str(&json).unwrap();
//! ```

/// Create a tracing span guard (no-op when `profiling` feature is disabled).
#[cfg(feature = "profiling")]
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _span = tracing::info_span!($name).entered();
    };
}

#[cfg(not(feature = "profiling"))]
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {};
}

/// Re-export of [`wgpu`] for windowed consumers that need surface management.
///
/// Headless consumers (CLI encode, cloud workers) should not need this -
/// use [`gpu::OutputFormat`] and the [`session`] API instead.
pub use wgpu;

/// Drain a channel receiver, keeping only the latest item.
///
/// After a blocking `recv()` returns the first item, this drains any
/// additional buffered items via `try_recv()` and returns the last one.
/// Useful for camera frame dropping: skip stale frames when the
/// renderer is slower than the capture rate.
///
/// Returns the number of items that were dropped (0 means the first
/// item was already the latest).
pub fn drain_to_latest<T>(rx: &std::sync::mpsc::Receiver<T>, item: &mut T) -> u64 {
    let mut dropped = 0;
    while let Ok(newer) = rx.try_recv() {
        *item = newer;
        dropped += 1;
    }
    dropped
}

pub mod async_encode;
pub mod calibration;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub mod cuda_interop;
pub mod detector;
pub mod director;
pub mod encoder;
pub mod gpu;
#[cfg(target_os = "macos")]
pub mod metal_interop;
pub mod nv12_converter;
pub mod pipeline;
pub mod renderer;
pub mod scene;
pub mod session;
pub mod source;
pub mod viewport;
#[cfg(target_os = "linux")]
pub mod vulkan_interop;
