//! FFmpeg-based video decode and encode.
//!
//! Handles file I/O and network protocols (RTMP, SRT, RTSP).
//! Supports hardware-accelerated decode (NVDEC, VA-API) and
//! encode (NVENC, VideoToolbox, QSV).

pub mod calibration_io;
pub mod decoder;
pub mod encoder;
mod hw_upload;

use std::sync::Once;

static FFMPEG_INIT: Once = Once::new();

/// Initialize FFmpeg (safe to call multiple times, only runs once).
pub fn init() {
    FFMPEG_INIT.call_once(|| {
        configure_external_logging();
        ffmpeg_next::init().expect("FFmpeg initialization failed");
    });
}

fn configure_external_logging() {
    // Default to ERROR (not FATAL): real FFmpeg encode/decode errors stay
    // visible for diagnostics, while the chatty WARNING/INFO output (codec
    // probes, "zero duration" stream notes, driver banners) is suppressed.
    // Override with RECO_FFMPEG_LOG (e.g. `warning`, `info`, `debug`).
    let ffmpeg_level = std::env::var("RECO_FFMPEG_LOG")
        .ok()
        .as_deref()
        .map(ffmpeg_log_level)
        .unwrap_or(ffmpeg_next::sys::AV_LOG_ERROR);

    unsafe {
        ffmpeg_next::sys::av_log_set_level(ffmpeg_level);
    }
}

fn ffmpeg_log_level(level: &str) -> i32 {
    match level.to_ascii_lowercase().as_str() {
        "quiet" => ffmpeg_next::sys::AV_LOG_QUIET,
        "panic" => ffmpeg_next::sys::AV_LOG_PANIC,
        "fatal" => ffmpeg_next::sys::AV_LOG_FATAL,
        "error" => ffmpeg_next::sys::AV_LOG_ERROR,
        "warn" | "warning" => ffmpeg_next::sys::AV_LOG_WARNING,
        "info" => ffmpeg_next::sys::AV_LOG_INFO,
        "verbose" => ffmpeg_next::sys::AV_LOG_VERBOSE,
        "debug" => ffmpeg_next::sys::AV_LOG_DEBUG,
        "trace" => ffmpeg_next::sys::AV_LOG_TRACE,
        _ => ffmpeg_next::sys::AV_LOG_ERROR,
    }
}
