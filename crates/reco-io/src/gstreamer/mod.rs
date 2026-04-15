//! GStreamer-based capture backends.
//!
//! Provides live camera capture using platform-appropriate GStreamer
//! elements:
//! - Jetson: `nvarguscamerasrc` (NVIDIA ISP: debayer, AWB, AE, denoise)
//! - Linux: `v4l2src` (generic V4L2 cameras)
//! - macOS: `avfvideosrc` (AVFoundation)
//! - Windows: `mfvideosrc` (Media Foundation)
//!
//! Pipelines output I420 (YUV420P) or NV12 via `appsink`. NV12
//! is the native NVIDIA ISP output, avoiding format conversion
//! on Jetson for lower latency.

pub mod camera;
