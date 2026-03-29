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

pub mod calibration;
pub mod cuda_interop;
pub mod detector;
pub mod director;
pub mod encoder;
pub mod gpu;
pub mod nv12_converter;
pub mod pipeline;
pub mod renderer;
pub mod scene;
pub mod source;
pub mod viewport;
#[cfg(target_os = "linux")]
pub mod vulkan_interop;
