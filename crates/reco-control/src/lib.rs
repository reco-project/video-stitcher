//! Shared direct viewport input, independent of windowing and rendering backends.
//!
//! Consumers supply pixel drags, wheel ticks or [`ControlIntent`] values. Controls retain requested and accepted render-space poses; core owns coverage and calibration validation.

#![deny(unsafe_code)]

/// Render-space pose input state, sensitivity configuration and coverage acceptance.
pub mod pose_control;

/// A viewport command. Pan deltas are render-space radians; FOV values are degrees.
///
/// Native bindings and directional conventions belong to consumers. Serialization uses an `action` tag and a `value` payload for numeric commands.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "action", content = "value", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ControlIntent {
    /// Adjust yaw by a signed angle in radians.
    DeltaYawRad(f32),
    /// Adjust pitch by a signed angle in radians.
    DeltaPitchRad(f32),
    /// Set vertical FOV in degrees, bounded by the control's zoom limits.
    SetFovDeg(f32),
    /// Adjust vertical FOV by a signed angle in degrees.
    DeltaFovDeg(f32),
    /// Narrow FOV by the configured keyboard zoom step.
    ZoomIn,
    /// Widen FOV by the configured keyboard zoom step.
    ZoomOut,
    /// Restore the configured render-space resting pose.
    Reset,
}

/// Optional camera-device commands, separate from viewport input.
#[cfg(feature = "gopro")]
pub mod gopro;
