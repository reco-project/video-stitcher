//! GStreamer-based capture backends.
//!
//! Provides live camera capture using platform-appropriate GStreamer
//! elements:
//! - Jetson: `nvarguscamerasrc` (NVIDIA ISP: debayer, AWB, AE, denoise)
//! - Linux: `v4l2src` (generic V4L2 cameras)
//! - macOS: `avfvideosrc` (AVFoundation)
//! - Windows: `mfvideosrc` (Media Foundation)
//!
//! All pipelines output I420 (YUV420P planar) via `appsink` for
//! compatibility with the existing reco-core `YuvPlanes` interface.

pub mod camera;
