//! FFmpeg-based video decode and encode.
//!
//! Handles file I/O and network protocols (RTMP, SRT, RTSP).
//! Supports hardware-accelerated decode (NVDEC, VA-API) and
//! encode (NVENC, VideoToolbox, QSV).

pub mod calibration_io;
pub mod decoder;
pub mod encoder;

use std::sync::Once;

static FFMPEG_INIT: Once = Once::new();

/// Initialize FFmpeg (safe to call multiple times, only runs once).
pub fn init() {
    FFMPEG_INIT.call_once(|| {
        ffmpeg_next::init().expect("FFmpeg initialization failed");
    });
}
