//! Pluggable frame I/O for Reco.
//!
//! This crate provides [`reco_core::source::FrameSource`] and
//! [`reco_core::sink::OutputSink`] implementations backed by
//! FFmpeg (files, network streams), GStreamer (live cameras),
//! and libcamera (Raspberry Pi CSI cameras).
//!
//! Enable backends via feature flags:
//! - `ffmpeg` (default): file decode/encode, RTSP video input, RTMP/RTMPS output, and stacked-video replay wrappers (protocol availability depends on the FFmpeg build).
//! - `gpu` (default): native texture sources, GPU stitching, and GPU atlas replay recording. FFmpeg hardware codecs remain available without this feature.
//! - `gstreamer`: live camera capture (Jetson ISP, V4L2, etc.)
//! - `libcamera`: RPi CSI camera capture via rpicam-vid
//! - `v4l2`: direct Linux V4L2 Bayer capture
//! - `rpi`: Raspberry Pi FFmpeg hardware codec hooks
//! - `profiling`: tracing spans (forwards `reco-core/profiling`)

#[cfg(feature = "ffmpeg")]
pub mod ffmpeg;

#[cfg(feature = "ffmpeg")]
mod decode_stream;

#[cfg(feature = "ffmpeg")]
mod decoder_cleanup;
#[cfg(feature = "ffmpeg")]
pub use decoder_cleanup::DecoderCleanup;

#[cfg(feature = "gstreamer")]
pub mod gstreamer;

#[cfg(feature = "libcamera")]
pub mod libcamera;

#[cfg(feature = "v4l2")]
pub mod v4l2;

pub mod adapters;
pub mod frame_sync;
/// Default shipping [`reco_core::detect::pipeline_event::PipelineEventSink`]:
/// writes each event as one JSON line. Wrap in `BackpressuredSink`
/// to keep serialization off the render thread.
pub mod jsonl_sink;
/// Application policy for retained lookahead images.
pub mod lookahead;
pub mod output;

#[cfg(feature = "ffmpeg")]
pub mod smart_source;

#[cfg(feature = "ffmpeg")]
pub mod rtsp;

#[cfg(feature = "ffmpeg")]
pub mod stitch_job;

#[cfg(all(feature = "ffmpeg", feature = "gpu"))]
pub mod zero_copy;

pub mod settings;

/// Pack N YUV420P camera tiles into one grid-layout frame and unpack them again. CPU primitives are unconditional; FFmpeg replay requires `ffmpeg`, and native GPU recording additionally requires `gpu`.
pub mod stacked_video;

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
