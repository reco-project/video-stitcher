//! # reco-control
//!
//! Operator pose-control state machine and intent dispatch for the Reco
//! pipeline.
//!
//! Consumers (reco-gui, reco-cli preview, reco-obs) feed input events to
//! a single [`pose_control::PoseControl`] and, where they speak a higher
//! level vocabulary, apply [`ControlIntent`] values via
//! [`PoseControl::apply_intent`](pose_control::PoseControl::apply_intent).
//!
//! # Why this crate
//!
//! The deep-review 2026-04-18 Agent 8 finding: consumer input paths were
//! duplicated three ways across reco-cli/preview, reco-gui, and reco-obs
//! (different key mappings, different pan sensitivity, different units).
//! [`PoseControl`](pose_control::PoseControl) fixes the state-machine
//! duplication; [`ControlIntent`] fixes the dispatch-vocabulary
//! duplication for the pose actions consumers share.

#![deny(unsafe_code)]

/// Unified pose-control primitive: `PoseControl` + `PoseControlConfig`
/// + `HotkeyIntent`. Single source of truth for mouse/drag/wheel/
/// keyboard -> yaw/pitch/FOV translation across consumers.
pub mod pose_control;

use pose_control::HotkeyIntent;

// ---------------------------------------------------------------------------
// Intent vocabulary
// ---------------------------------------------------------------------------

/// An operator pose action, independent of how it was triggered.
/// Consumers translate their native events (keystrokes, game-pad
/// buttons, mobile touch) into these and apply them via
/// [`PoseControl::apply_intent`](pose_control::PoseControl::apply_intent).
///
/// `#[non_exhaustive]` so new intent categories can be added without
/// breaking every consumer's match arm.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ControlIntent {
    /// Pan / zoom / reset driven by a hotkey binding. Wraps
    /// [`HotkeyIntent`] so consumers that already dispatch HotkeyIntents
    /// (via `PoseControl::apply_hotkey`) can forward directly.
    Hotkey(HotkeyIntent),

    /// Direct pose manipulation (set or delta). Used by input devices
    /// that speak absolute values (sliders, game-pad axes, mobile touch)
    /// rather than discrete keys.
    Pose(PoseIntent),
}

/// Pose-direct intents. Angles in radians for yaw/pitch, degrees
/// for FOV — same convention as
/// [`PoseControl`](crate::pose_control::PoseControl).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "action", content = "value", rename_all = "snake_case")]
#[non_exhaustive]
pub enum PoseIntent {
    /// Set target yaw (radians) to an absolute value.
    SetYawRad(f32),
    /// Set target pitch (radians) to an absolute value.
    SetPitchRad(f32),
    /// Set target FOV (degrees) to an absolute value.
    SetFovDeg(f32),
    /// Additively adjust target yaw by this many radians.
    DeltaYawRad(f32),
    /// Additively adjust target pitch by this many radians.
    DeltaPitchRad(f32),
    /// Additively adjust target FOV by this many degrees.
    DeltaFovDeg(f32),
    /// Return the target pose to the configured rest position.
    Reset,
}

#[cfg(feature = "gopro")]
pub mod gopro;

// ---------------------------------------------------------------------------
// Compile-time bound check
// ---------------------------------------------------------------------------

// `ControlIntent` and its payloads are `Clone + Send` so a worker
// thread can forward events to the UI thread through an mpsc channel.
const _: fn() = || {
    fn assert_clone_send<T: Clone + Send + 'static>() {}
    assert_clone_send::<ControlIntent>();
    assert_clone_send::<PoseIntent>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_intent_wraps_hotkey_intent() {
        let intent = ControlIntent::Hotkey(HotkeyIntent::ZoomIn);
        match intent {
            ControlIntent::Hotkey(HotkeyIntent::ZoomIn) => {}
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn pose_intent_distinguishes_set_from_delta() {
        let abs = PoseIntent::SetYawRad(1.0);
        let rel = PoseIntent::DeltaYawRad(0.1);
        assert_ne!(abs, rel);
    }

    #[test]
    fn non_exhaustive_allows_wildcard_match() {
        // Regression guard: `#[non_exhaustive]` on ControlIntent forces
        // downstream consumers to write a wildcard arm. This test
        // compile-checks that idiom stays valid.
        let intent = ControlIntent::Pose(PoseIntent::Reset);
        fn handle(i: &ControlIntent) -> &'static str {
            match i {
                ControlIntent::Pose(PoseIntent::Reset) => "reset",
                _ => "other",
            }
        }
        assert_eq!(handle(&intent), "reset");
    }

    #[test]
    fn serde_roundtrip_hotkey() {
        let intent = ControlIntent::Hotkey(HotkeyIntent::ZoomIn);
        let json = serde_json::to_string(&intent).unwrap();
        assert!(json.contains("\"kind\":\"hotkey\""));
        let back: ControlIntent = serde_json::from_str(&json).unwrap();
        assert_eq!(intent, back);
    }

    #[test]
    fn serde_roundtrip_pose_delta() {
        let intent = ControlIntent::Pose(PoseIntent::DeltaYawRad(0.1));
        let json = serde_json::to_string(&intent).unwrap();
        assert!(json.contains("\"kind\":\"pose\""));
        let back: ControlIntent = serde_json::from_str(&json).unwrap();
        assert_eq!(intent, back);
    }
}
