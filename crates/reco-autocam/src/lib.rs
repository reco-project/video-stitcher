//! Automatic camera control for reco.
//!
//! This crate provides implementations of the [`reco_core`] detection and
//! direction traits for sports camera automation:
//!
//! - [`YoloDetector`] — ONNX-based YOLO object detection on raw camera frames
//! - [`EkfTracker`] — Extended Kalman Filter tracker (utility for directors)
//! - [`BallDirector`] — Ball-following director with smoothing and state machine logic
//!
//! # Usage
//!
//! ```rust,no_run
//! use reco_autocam::{YoloDetector, BallDirector};
//!
//! let detector = YoloDetector::from_file("ball_v0.onnx")?;
//! let director = BallDirector::new(30.0); // fps
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod detector;
mod director;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod gpu_detector;
#[cfg(target_os = "macos")]
mod metal_detector;
mod tracker;

pub use detector::YoloDetector;
pub use director::BallDirector;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use gpu_detector::GpuYoloDetector;
#[cfg(target_os = "macos")]
pub use metal_detector::MetalYoloDetector;
pub use tracker::{EkfTracker, Track};
