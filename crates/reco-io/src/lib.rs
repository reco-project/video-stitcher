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

/// Initialize all enabled backends. Call once at program start.
pub fn init() {
    #[cfg(feature = "ffmpeg")]
    ffmpeg::init();
}
