//! Pluggable frame I/O for Reco.
//!
//! This crate provides [`reco_core::source::FrameSource`] and
//! [`reco_core::encoder::Encoder`] implementations backed by
//! FFmpeg (files, network streams) and GStreamer (live cameras).
//!
//! Enable backends via feature flags:
//! - `ffmpeg` (default): file decode/encode, RTMP/SRT/RTSP
//! - `gstreamer`: live camera capture (Jetson ISP, V4L2, etc.)

#[cfg(feature = "ffmpeg")]
pub mod ffmpeg;

#[cfg(feature = "gstreamer")]
pub mod gstreamer;

pub mod adapters;
pub mod output;

#[cfg(feature = "ffmpeg")]
pub mod smart_source;

#[cfg(feature = "ffmpeg")]
pub mod stitch_job;

#[cfg(feature = "ffmpeg")]
pub mod zero_copy;

#[cfg(feature = "ffmpeg")]
pub use smart_source::SmartFileSource;

#[cfg(feature = "ffmpeg")]
pub use stitch_job::{InputPath, StitchJob, StitchResult};

/// Initialize enabled backends. Call once at program start.
///
/// Currently initializes FFmpeg when the `ffmpeg` feature is active.
/// GStreamer initializes lazily on first pipeline creation.
pub fn init() {
    #[cfg(feature = "ffmpeg")]
    ffmpeg::init();
}
