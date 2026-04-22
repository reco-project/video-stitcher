//! Pluggable frame I/O for Reco.
//!
//! This crate provides [`reco_core::source::FrameSource`] and
//! [`reco_core::encoder::Encoder`] implementations backed by
//! FFmpeg (files, network streams), GStreamer (live cameras),
//! and libcamera (Raspberry Pi CSI cameras).
//!
//! Enable backends via feature flags:
//! - `ffmpeg` (default): file decode/encode, RTMP/SRT/RTSP
//! - `gstreamer`: live camera capture (Jetson ISP, V4L2, etc.)
//! - `libcamera`: RPi CSI camera capture via rpicam-vid
//! - `config`: opt-in user-preference persistence (via the `settings`
//!   module) for consumers like reco-gui that need to remember
//!   recent files and defaults across sessions.

#[cfg(feature = "ffmpeg")]
pub mod ffmpeg;

#[cfg(feature = "gstreamer")]
pub mod gstreamer;

#[cfg(feature = "libcamera")]
pub mod libcamera;

pub mod adapters;
/// Default shipping [`reco_core::pipeline_event::PipelineEventSink`]:
/// writes each event as one JSON line. Wrap in `BackpressuredSink`
/// to keep serialization off the render thread.
pub mod jsonl_sink;
pub mod output;

#[cfg(feature = "ffmpeg")]
pub mod smart_source;

#[cfg(feature = "ffmpeg")]
pub mod stitch_job;

#[cfg(feature = "ffmpeg")]
pub mod analyze_job;

#[cfg(feature = "ffmpeg")]
pub mod zero_copy;

#[cfg(feature = "config")]
pub mod settings;

/// M6.5 stacked-video pack / unpack. Maps N YUV420P tiles into one
/// grid-layout frame (see `GridLayout`) and back, for single-file replay
/// recording, web panorama input, and cloud deployment. FFmpeg-backed
/// encoder/source stubs are gated behind the `stacked-output`
/// feature; the pure CPU pack/unpack primitives have no feature gate.
///
/// A GPU-accelerated pack/unpack path (wgpu compute shader that
/// blits N source textures into one render target) is a natural
/// follow-up — noted as a future-work item in the module docs.
pub mod stacked_video;

#[cfg(feature = "ffmpeg")]
pub use smart_source::SmartFileSource;

#[cfg(feature = "ffmpeg")]
pub use stitch_job::{InputPath, StitchJob, StitchResult};

#[cfg(feature = "ffmpeg")]
pub use analyze_job::{AnalyzeJob, AnalyzeResult};

/// Initialize enabled backends. Call once at program start.
///
/// Currently initializes FFmpeg when the `ffmpeg` feature is active.
/// GStreamer initializes lazily on first pipeline creation.
pub fn init() {
    #[cfg(feature = "ffmpeg")]
    ffmpeg::init();
}
