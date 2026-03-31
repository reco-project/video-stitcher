//! Automatic camera control for reco.
//!
//! This crate provides implementations of the [`reco_core`] detection,
//! tracking, and direction traits for sports camera automation:
//!
//! - [`YoloDetector`] — ONNX-based YOLO object detection on raw camera frames
//! - [`EkfTracker`] — Extended Kalman Filter tracker with persistent object identity
//! - [`BallDirector`] — Ball-following director with smoothing and state machine logic
//!
//! # Usage
//!
//! ```rust,no_run
//! use reco_autocam::{YoloDetector, EkfTracker, BallDirector};
//!
//! let detector = YoloDetector::from_file("ball_v0.onnx")?;
//! let tracker = EkfTracker::new();
//! let director = BallDirector::new(30.0); // fps
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod detector;
mod director;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod gpu_detector;
mod tracker;

pub use detector::YoloDetector;
pub use director::BallDirector;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use gpu_detector::GpuYoloDetector;
pub use tracker::EkfTracker;
