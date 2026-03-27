//! Director trait for controlling virtual camera panning.
//!
//! A director determines where the virtual camera looks for each frame.
//! This is the primary extension point for AI-driven auto-panning
//! (e.g. ball tracking) and scripted camera movements.
//!
//! ## Design
//!
//! Directors receive detection data and output a viewport position.
//! They run asynchronously from the GPU pipeline — the renderer
//! simply reads the latest viewport position for each frame.

use crate::detector::Detection;

/// The viewport position output by a director.
///
/// Specifies the yaw and pitch of the virtual camera in radians.
#[derive(Debug, Clone, Copy)]
pub struct ViewportPosition {
    /// Horizontal pan angle in radians.
    ///
    /// `0.0` = centered on the seam between cameras.
    pub yaw: f32,

    /// Vertical tilt angle in radians.
    ///
    /// `0.0` = level. Positive = looking up.
    pub pitch: f32,
}

impl Default for ViewportPosition {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

/// Trait for virtual camera direction control.
///
/// Implement this trait to create custom panning behaviors:
/// - Static director: fixed viewport position
/// - Script director: keyframed pan/tilt over time
/// - AI director: follows ball detections from [`crate::detector::Detector`]
///
/// # Example
///
/// ```rust
/// use reco_core::director::{Director, ViewportPosition};
///
/// struct StaticDirector {
///     position: ViewportPosition,
/// }
///
/// impl Director for StaticDirector {
///     fn update(&mut self, _frame_index: u64, _timestamp_ms: f64, _detections: &[reco_core::detector::Detection]) {}
///
///     fn position(&self) -> ViewportPosition {
///         self.position
///     }
/// }
/// ```
pub trait Director: Send {
    /// Update the director state with new frame data and detections.
    ///
    /// Called once per frame. `detections` may be empty if no detector is
    /// configured or if detection hasn't completed for this frame yet.
    fn update(&mut self, frame_index: u64, timestamp_ms: f64, detections: &[Detection]);

    /// Get the current viewport position.
    ///
    /// Called by the renderer to determine where to crop the panorama.
    fn position(&self) -> ViewportPosition;
}
