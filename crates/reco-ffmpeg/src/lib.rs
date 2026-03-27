//! FFmpeg-based video decode and encode for Reco.
//!
//! Provides [`FrameSource`] for decoding video files via FFmpeg, and
//! an [`reco_core::encoder::Encoder`] implementation for encoding
//! the stitched output.
//!
//! ## Phase 1
//!
//! Software decode (libavcodec) → CPU buffer → wgpu texture upload.
//! This path works on all platforms without hardware-specific setup.
//!
//! ## Future
//!
//! Hardware-accelerated decode (NVDEC, VideoToolbox, VAAPI) with
//! zero-copy GPU surface handoff to wgpu.

// FFmpeg bindings will be added when we integrate actual video decode.
// For now, this crate defines the interfaces and stubs.

/// Placeholder for FFmpeg decoder implementation.
pub struct FfmpegDecoder;

/// Placeholder for FFmpeg encoder implementation.
pub struct FfmpegEncoder;
