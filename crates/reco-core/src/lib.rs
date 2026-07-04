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
//! - Right camera plane lies in the X-Y plane (faces forward)
//! - A virtual camera sits at the corner, looking at both planes
//! - Panning is achieved by rotating the virtual camera (yaw/pitch)
//!
//! ## Modularity
//!
//! The crate defines traits for pluggable components:
//! - [`source::FrameSource`] — delivers stereo frame pairs (files, cameras, streams)
//! - [`detect::detector::UnifiedDetector`] — detects objects in raw frames (e.g. ball tracking)
//! - [`detect::tracker::Tracker`] — turns detections into stable tracked entities
//! - [`detect::panner::Panner`] — turns the tracked world state into a viewport pose
//! - [`encoder::Encoder`] — receives stitched GPU frames for encoding
//!
//! ## Usage
//!
//! ```rust,no_run
//! use reco_core::calibration::Calibration;
//!
//! // Load calibration from a JSON file
//! let json = std::fs::read_to_string("match.json").unwrap();
//! let calibration: Calibration = serde_json::from_str(&json).unwrap();
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
/// Windowed apps need wgpu types like `Instance`, `Surface`,
/// `SurfaceConfiguration`, and `TextureFormat` for display setup.
/// This re-export ensures version compatibility with `reco-core`'s
/// internal wgpu usage - prefer this over adding `wgpu` as a
/// direct dependency.
///
/// Headless consumers (CLI encode, cloud workers) should not need this -
/// use [`gpu::OutputFormat`] and the [`session`] API instead.
#[cfg(feature = "gpu")]
pub use wgpu;

// The `gpu` feature (default-on) gates the wgpu render stack. What
// remains without it is the wgpu-free leaf for GPU-less targets:
// calibration, geometry, projection, the CPU stitch path, and the
// pure value/trait modules (source, detect sans GPU frames, encoder,
// telemetry).
#[cfg(feature = "gpu")]
pub(crate) mod async_encode;
#[cfg(feature = "gpu")]
pub mod bayer;
pub mod calibration;
/// Push-first `StitchCore` engine - the canonical entry point.
/// See [`core::StitchCore`] for details. Engine orchestration is
/// wgpu-free; only its GPU streaming surface is gated on `gpu`.
pub mod core;
pub mod detect;
pub mod encoder;
pub mod geometry;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod interop;
pub mod lens;
#[cfg(all(target_os = "linux", feature = "gpu"))]
pub mod nvbuf_transform;
pub mod projection;
pub mod render;
#[cfg(feature = "gpu")]
pub mod session;
pub mod source;
pub mod stitch;
pub mod telemetry;
