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
//! - [`detector::UnifiedDetector`] — detects objects in raw frames (e.g. ball tracking)
//! - [`tracker::Tracker`] — turns detections into stable tracked entities
//! - [`panner::Panner`] — turns the tracked world state into a viewport pose
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
/// Windowed apps need wgpu types like `Instance`, `Surface`,
/// `SurfaceConfiguration`, and `TextureFormat` for display setup.
/// This re-export ensures version compatibility with `reco-core`'s
/// internal wgpu usage - prefer this over adding `wgpu` as a
/// direct dependency.
///
/// Headless consumers (CLI encode, cloud workers) should not need this -
/// use [`gpu::OutputFormat`] and the [`session`] API instead.
pub use wgpu;

pub mod analyze;
pub(crate) mod async_encode;
pub mod calibration;
/// M3 push-first `StitchCore` shell — the canonical entry point.
/// See [`core::StitchCore`] for details.
pub mod core;
// `coreml_inference`, `cuda_kernels`, `npp_interop`, `metal_compute`
// moved to reco-detect (the only consumer) — per the revised M5
// analysis: those four are detection-preprocess, not GPU pipeline
// infrastructure. reco-core keeps `cuda_interop`, `metal_interop`,
// `vulkan_interop`, `zero_copy`, which are wgpu-native platform
// paths used by the stitch pipeline's zero-copy bridge.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub mod cuda_interop;
pub mod detector;
pub mod director;
pub mod encoder;
/// M4 timestamped multi-source ingest buffer. See
/// [`framesync::TimestampedIngestBuffer`] for the dual-source
/// pairing case and the N-source livestream-sync foundation.
pub mod framesync;
pub mod gpu;
pub mod lens;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod metal_interop;
pub mod nv12_converter;
/// Camera-motion policy contract — see [`Panner`](panner::Panner).
/// The panner half of the tracker/panner split: consumes a clean
/// [`WorldState`](tracker::WorldState) and decides where to point
/// the virtual camera. Camera smoothing, anticipation, and dead-zone
/// handling layer as panner decorators in `reco-autocam`.
pub mod panner;
pub mod pipeline;
/// M4 unified pose-control primitive. See [`pose_control::PoseControl`]
/// and [`pose_control::HotkeyIntent`] — the single source of truth for
/// mouse/drag/wheel/keyboard → yaw/pitch/FOV translation across
/// consumers.
pub mod pose_control;
pub mod projection;
pub mod renderer;
pub mod rgba_readback;
pub mod rig_correction;
pub mod scene;
pub mod session;
pub mod source;
pub mod stage;
pub mod stitch_renderer;
/// Tracker contract — see [`Tracker`](tracker::Tracker),
/// [`WorldState`](tracker::WorldState), and
/// [`TrackedEntity`](tracker::TrackedEntity). The tracker half of
/// the tracker/panner split: turns noisy per-frame detections into
/// stable tracked entities with velocity and lifecycle state.
/// Implementations (BallTracker, PlayerTracker, …) live in
/// `reco-autocam`.
pub mod tracker;
pub mod undistort;
pub mod viewport;
#[cfg(target_os = "linux")]
pub mod vulkan_interop;
pub mod yuv_stack_packer;
pub mod zero_copy;
