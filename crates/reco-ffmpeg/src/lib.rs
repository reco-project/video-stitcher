//! FFmpeg-based video decode and encode for Reco.
//!
//! Provides [`decoder::VideoDecoder`] for reading video files frame-by-frame
//! as RGBA pixel data, and [`encoder::VideoEncoder`] for writing RGBA frames
//! to H.264 MP4 files.
//!
//! ## Architecture
//!
//! This crate wraps [`ffmpeg_next`] (Rust bindings for FFmpeg) and is
//! deliberately independent of `reco-core`. The CLI binary ties them together:
//!
//! ```text
//! reco-ffmpeg (decode) → RGBA frames → reco-core (GPU stitch) → RGBA frames → reco-ffmpeg (encode)
//! ```
//!
//! ## Phase 1
//!
//! Software decode (libavcodec) → swscale to RGBA → CPU buffer.
//! Software encode (libx264) ← swscale from RGBA ← CPU buffer.
//!
//! ## Future
//!
//! Hardware-accelerated decode (NVDEC, VideoToolbox, VAAPI) with
//! zero-copy GPU surface handoff to wgpu (following Gyroflow's pattern).

pub mod decoder;
pub mod encoder;
