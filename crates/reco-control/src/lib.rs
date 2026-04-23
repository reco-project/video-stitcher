//! # reco-control
//!
//! Transport-agnostic control vocabulary for the Reco pipeline.
//!
//! Consumers (reco-obs, reco-gui, reco-cli, future standalone
//! livestream app) receive a stream of [`ControlIntent`] values and
//! dispatch them to their local state (pose control, encoder, replay
//! buffer, model selection). The transport — keyboard, gamepad, GoPro
//! USB HID, mobile WebRTC control channel, WebSocket — is
//! pluggable via the [`ControlTransport`] trait.
//!
//! # Why this crate
//!
//! The deep-review 2026-04-18 Agent 8 finding: consumer input paths
//! are duplicated three ways across reco-cli/preview, reco-gui, and
//! reco-obs (different key mappings, different pan sensitivity,
//! different units). [`PoseControl`](pose_control::PoseControl)
//! fixed the *state-machine* duplication; this crate fixes the
//! *vocabulary* duplication. A single `ControlIntent` enum describes
//! every operator action the pipeline cares about; transports
//! translate their native input events into these intents.
//!
//! # Planned transports
//!
//! | Feature       | Transport                       | Status       |
//! |---------------|----------------------------------|--------------|
//! | `keyboard`    | [`keyboard::KeyboardTransport`] | **shipped** (trivial pass-through) |
//! | `gopro`       | [`gopro`] module stub           | placeholder  |
//! | `mobile`      | [`mobile`] module stub          | placeholder  |
//! | `websocket`   | [`websocket`] module stub       | placeholder  |
//!
//! The placeholder modules contain `todo!()` bodies; they exist so
//! the feature-combo CI matrix exercises the gates and so future
//! work has an obvious target path.

#![deny(unsafe_code)]

/// Unified pose-control primitive: `PoseControl` + `PoseControlConfig`
/// + `HotkeyIntent`. Single source of truth for mouse/drag/wheel/
/// keyboard -> yaw/pitch/FOV translation across consumers.
/// Relocated from reco-core in Step 13 of the camera-stack plan.
pub mod pose_control;

use pose_control::HotkeyIntent;

// ---------------------------------------------------------------------------
// Intent vocabulary
// ---------------------------------------------------------------------------

/// Every operator action the pipeline cares about, independent of
/// how it was triggered. Transports translate their native events
/// (keystrokes, game-pad buttons, GoPro REST commands, WebSocket
/// messages) into these intents.
///
/// `#[non_exhaustive]` so new intent categories can be added without
/// breaking every consumer's match arm. Consumers already write
/// `_ =>` fallbacks per Rust's non-exhaustive rule.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ControlIntent {
    /// Pan / zoom / reset driven by a hotkey binding. Wraps
    /// [`reco_core::pose_control::HotkeyIntent`] so consumers that
    /// already dispatch HotkeyIntents (via `PoseControl::apply_hotkey`)
    /// can forward directly.
    Hotkey(HotkeyIntent),

    /// Direct pose manipulation (set or delta). Used by input
    /// devices that speak absolute values (sliders, game-pad axes,
    /// mobile touch) rather than discrete keys.
    Pose(PoseIntent),

    /// Encoder / output-quality adjustment.
    Quality(QualityIntent),

    /// Recording / replay / snapshot operations.
    Capture(CaptureIntent),

    /// AI detector / model selection.
    ModelSelect(ModelSelectIntent),
}

/// Pose-direct intents. Angles in radians for yaw/pitch, degrees
/// for FOV — same convention as
/// [`reco_core::pose_control::PoseControl`].
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Encoder / quality intents. String fields are the vocabulary the
/// encoder crate (reco-io) understands (codec names, preset names);
/// reco-control stays framework-agnostic.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QualityIntent {
    /// Change the active codec (e.g. `"h264"`, `"hevc"`, `"av1"`).
    SetCodec(String),
    /// Set the target bitrate in bits per second.
    SetBitrate(u64),
    /// Change the output resolution.
    SetResolution {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// Change the encoder preset (e.g. `"fast"`, `"medium"`, `"slow"`).
    SetPreset(String),
    /// Set the constant rate factor (0-51 for H.264; lower is better).
    SetCrf(u8),
}

/// Recording / replay / snapshot intents.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum CaptureIntent {
    /// Begin encoding to the configured output path.
    StartRecord,
    /// Stop encoding; finalize the output file.
    StopRecord,
    /// Capture a single frame to disk (still image).
    Snapshot,
    /// Clear the replay ring buffer
    /// ([`reco_core::core::ReplayBuffer::clear`]).
    ClearReplay,
    /// Save the current replay ring buffer to a file and clear it
    /// (maps to [`reco_core::core::ReplayBuffer::take`] + encode).
    SaveReplay,
}

/// Detector / model selection intents.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ModelSelectIntent {
    /// Switch to a detector model at the given path (or bundle
    /// name). Empty string disables detection.
    SetDetectorModel(String),
    /// Change how often detection runs (`1` = every frame).
    SetDetectionInterval(u64),
    /// Disable detection without changing the model path — useful
    /// for low-power mode.
    DisableDetection,
}

// ---------------------------------------------------------------------------
// Transport trait
// ---------------------------------------------------------------------------

/// A source of control intents.
///
/// Transports pull pending intents from their native event stream
/// and expose them as an iterator-like interface. The contract is
/// `non-blocking` — `poll` returns immediately with whatever is
/// available. Consumers call it each frame / tick and dispatch any
/// returned intents to their local state.
///
/// `Send` so transports can live on a worker thread (e.g. a GoPro
/// USB-HID reader loop that needs to block on `read`). Not `Sync`
/// because most transports hold mutable device state.
pub trait ControlTransport: Send {
    /// Short human-readable name for logs + telemetry bundles
    /// (e.g. `"keyboard"`, `"gopro-usb"`, `"websocket"`).
    fn name(&self) -> &'static str;

    /// Drain any currently-available intents into the supplied
    /// buffer. Returns the number appended. Non-blocking.
    ///
    /// Implementations typically read from an internal mpsc /
    /// ring buffer that a background thread fills from the
    /// underlying transport (USB HID, WebSocket message, etc.).
    fn poll(&mut self, out: &mut Vec<ControlIntent>) -> usize;
}

// ---------------------------------------------------------------------------
// Transport implementations
// ---------------------------------------------------------------------------

#[cfg(feature = "keyboard")]
pub mod keyboard;

#[cfg(feature = "gopro")]
pub mod gopro;

#[cfg(feature = "mobile")]
pub mod mobile;

#[cfg(feature = "websocket")]
pub mod websocket;

// ---------------------------------------------------------------------------
// Compile-time bound check
// ---------------------------------------------------------------------------

// `ControlIntent` and its payloads are `Clone + Send` so a worker
// thread can forward events to the UI thread through an mpsc
// channel. Not `Sync` (immutable sharing across threads isn't the
// typical use case), but `Send` is mandatory.
const _: fn() = || {
    fn assert_clone_send<T: Clone + Send + 'static>() {}
    assert_clone_send::<ControlIntent>();
    assert_clone_send::<PoseIntent>();
    assert_clone_send::<QualityIntent>();
    assert_clone_send::<CaptureIntent>();
    assert_clone_send::<ModelSelectIntent>();
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
    fn quality_intent_resolution_roundtrips() {
        let q = QualityIntent::SetResolution {
            width: 1920,
            height: 1080,
        };
        let cloned = q.clone();
        assert_eq!(q, cloned);
    }

    #[test]
    fn capture_intent_variants_are_copy() {
        // Copy-ness is useful for dispatch on the hot path.
        let i: CaptureIntent = CaptureIntent::SaveReplay;
        let j = i;
        let _both = (i, j); // proof of Copy
    }

    #[test]
    fn non_exhaustive_allows_wildcard_match() {
        // Regression guard: `#[non_exhaustive]` on ControlIntent
        // forces downstream consumers to write a wildcard arm. This
        // test compile-checks that idiom stays valid.
        let intent = ControlIntent::Capture(CaptureIntent::Snapshot);
        fn handle(i: &ControlIntent) -> &'static str {
            match i {
                ControlIntent::Capture(CaptureIntent::Snapshot) => "snapshot",
                _ => "other",
            }
        }
        assert_eq!(handle(&intent), "snapshot");
    }
}
