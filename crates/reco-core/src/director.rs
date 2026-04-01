//! Director trait for controlling virtual camera panning.
//!
//! A director determines where the virtual camera looks for each frame.
//! This is the primary extension point for AI-driven auto-panning
//! (e.g. ball tracking) and scripted camera movements.
//!
//! ## Data Flow
//!
//! ```text
//! Detector (raw detections)
//!   -> coordinate mapping (camera pixels -> panorama yaw/pitch)
//!     -> Director (panning decisions)
//!     -> External consumers (coaching, stats, VAR)
//! ```
//!
//! Directors receive [`MappedDetection`]s (detections enriched with panorama
//! coordinates) via [`DirectorContext`], and output a [`ViewportPosition`]
//! that controls where the virtual camera pans. Tracking (persistent object
//! identity) is not part of the pipeline - directors that need it can use
//! tracking utilities internally.
//!
//! ## External Consumers
//!
//! Detection data isn't just for the director. The same [`MappedDetection`]s
//! are available to external consumers via
//! [`StitchSession::set_detection_callback`](crate::session::StitchSession::set_detection_callback).
//! This enables building coaching assistants, VAR systems, and stats
//! pipelines on top of the detection data without modifying the director.

use crate::detector::CameraId;
use crate::projection::CoverageBoundary;

/// The viewport position output by a director.
///
/// Specifies the yaw, pitch, and field of view of the virtual camera.
/// The FOV allows directors to express zoom: narrow FOV = zoomed in on
/// action, wide FOV = zoomed out for context.
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

    /// Horizontal field of view in degrees, or `None` to use the
    /// pipeline's default FOV.
    ///
    /// Typical range: 30.0 (zoomed in) to 120.0 (wide). The pipeline
    /// default is 75.0.
    pub fov_degrees: Option<f32>,
}

impl Default for ViewportPosition {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: None,
        }
    }
}

/// A detection mapped to panorama coordinates.
///
/// This is the primary data type that flows to both the [`Director`] and
/// external consumers. It enriches the raw camera-space detection with
/// panorama-space coordinates (yaw/pitch, computed via
/// [`camera_to_panorama`](crate::projection::camera_to_panorama)).
///
/// External consumers (coaching, VAR, stats) receive these via the
/// detection callback on [`StitchSession`](crate::session::StitchSession).
#[derive(Debug, Clone)]
pub struct MappedDetection {
    /// Which camera this detection came from.
    pub camera: CameraId,

    /// Detection class label (e.g. "ball", "player").
    pub label: String,

    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,

    /// Bounding box center in normalized camera coordinates `[0.0, 1.0]`.
    pub camera_center: (f32, f32),

    /// Bounding box size in normalized camera coordinates.
    pub camera_size: (f32, f32),

    /// Position in panorama space (yaw/pitch).
    /// `None` if the detection couldn't be mapped (e.g. outside camera FOV).
    pub position: Option<ViewportPosition>,
}

/// Context passed to [`Director::update`] each frame.
///
/// Provides everything a director needs to make panning decisions:
/// tracked objects with panorama coordinates, valid panning bounds,
/// and timing information.
#[derive(Debug)]
pub struct DirectorContext<'a> {
    /// Current frame index (0-based).
    pub frame_index: u64,

    /// Elapsed time since the start of processing, in milliseconds.
    pub timestamp_ms: f64,

    /// Detections mapped to panorama coordinates for this frame.
    /// Empty if no detector is configured or detection was skipped.
    pub detections: &'a [MappedDetection],

    /// Precomputed coverage boundary for safe panning (built once at startup).
    pub coverage: &'a CoverageBoundary,

    /// Current horizontal field of view in degrees.
    pub current_fov: f32,
}

/// Trait for virtual camera direction control.
///
/// Implement this trait to create custom panning behaviors:
/// - **Static**: fixed viewport position (debugging, manual override)
/// - **Scripted**: keyframed pan/tilt over time (replays, highlights)
/// - **AI**: follows tracked objects using smoothing, prediction, rules
///
/// Directors receive rich context via [`DirectorContext`] including
/// tracked objects with panorama coordinates and valid panning bounds.
///
/// # Example
///
/// ```rust
/// use reco_core::director::{Director, DirectorContext, ViewportPosition};
///
/// struct StaticDirector {
///     position: ViewportPosition,
/// }
///
/// impl Director for StaticDirector {
///     fn update(&mut self, _ctx: &DirectorContext<'_>) {}
///
///     fn position(&self) -> ViewportPosition {
///         self.position
///     }
/// }
/// ```
pub trait Director: Send {
    /// Update the director state with new frame context.
    ///
    /// Called once per frame. The context includes tracked objects
    /// (may be empty), viewport bounds, and timing. The director
    /// should update its internal state and be ready for a
    /// [`position`](Self::position) call.
    fn update(&mut self, ctx: &DirectorContext<'_>);

    /// Get the current viewport position.
    ///
    /// Called by the renderer to determine where to crop the panorama.
    fn position(&self) -> ViewportPosition;
}
