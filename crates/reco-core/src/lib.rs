//! # reco-core
//!
//! Calibrated panoramic rendering with shared CPU and GPU engine orchestration.
//!
//! [`calibration::Calibration`] owns lenses, projection geometry and rig framing. It supports stereo L-shape and mono cylindrical inputs; [`core::StitchCore`] owns coverage, render-pose resolution and optional tracking. [`session::StitchSession`] drives the engine from a source through immediate or lookahead processing into output sinks.
//!
//! ## Architecture
//!
//! The pipeline processes frames through these stages:
//!
//! ```text
//! Source frames → detection/tracking → render-pose resolution
//!                                      → CPU/GPU executor → output sinks
//! ```
//!
//! CPU rendering, calibrated geometry and session orchestration build without the default `gpu` feature. GPU execution, native image imports and preview/readback require it. The isolated CPU test and dependency-graph checks in `.github/workflows/rust.yml` enforce this boundary.
//!
//! ## Modularity
//!
//! The crate defines traits for pluggable components:
//! - [`source::FrameSource`] — delivers per-camera frame sets (files, cameras, streams)
//! - [`detect::detector::Detector`] — detects objects in raw frames (e.g. ball tracking)
//! - [`detect::tracker::Tracker`] — turns detections into stable tracked entities
//! - [`detect::panner::Panner`] — turns the tracked world state into a viewport pose
//! - [`sink::OutputSink`] — consumes stitched frames (encoding, snapshots, streams)
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

// Exported macros must resolve dependencies through their defining crate.
#[cfg(feature = "profiling")]
#[doc(hidden)]
pub use tracing::info_span as __profile_span;

/// Create a scope-bound tracing span when reco-core's `profiling` feature is enabled. Otherwise, expands to nothing without evaluating the argument. Callers do not need a direct tracing dependency.
#[cfg(feature = "profiling")]
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _span = $crate::__profile_span!($name).entered();
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

// The `gpu` feature gates GPU rendering and platform interop. CPU rendering,
// engine/session orchestration and source/detection/sink traits remain available.
#[cfg(feature = "gpu")]
pub mod bayer;
pub mod calibration;
/// Push-first `StitchCore` engine - the canonical entry point.
/// See [`core::StitchCore`] for details. Engine orchestration is
/// wgpu-free; only its GPU streaming surface is gated on `gpu`.
pub mod core;
pub mod detect;
pub mod geometry;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod interop;
pub mod lens;
mod lookahead;
pub mod projection;
pub mod render;
pub mod session;
pub mod sink;
pub(crate) mod sink_thread;
pub mod source;
pub mod stitch;
pub mod telemetry;
