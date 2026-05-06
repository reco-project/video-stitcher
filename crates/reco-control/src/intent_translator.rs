//! Central dispatch from [`ControlIntent`] to consumer-side state.
//!
//! See `~/Documents/knowledge/projects/video-stitcher/architecture/
//! reco-control-design-2026-04-23.md` decisions #3, #5, #7.
//!
//! # Purpose
//!
//! Every consumer (reco-gui, reco-cli preview, reco-obs) turns a
//! transport's [`ControlIntent`] stream into calls on
//! [`PoseControl`](crate::pose_control::PoseControl) plus domain-specific side effects (start encoder,
//! swap detector model, change codec). Before this type, each
//! consumer wrote that dispatch inline. [`IntentTranslator`] owns
//! the translation once.
//!
//! Pose intents route directly to the borrowed [`PoseControl`](crate::pose_control::PoseControl).
//! Non-pose intents (quality / capture / model-select) route to
//! consumer-provided closures; those slots are `Option<Box<dyn FnMut>>`
//! rather than trait objects on the translator itself, matching the
//! Slint-callback shape most consumers already use.
//!
//! # Lifetime and threading
//!
//! Translators borrow `PoseControl` for a single dispatch tick and
//! are not `Send`. Cross-thread intent forwarding goes through an
//! mpsc channel of [`ControlIntent`] values (which *are* `Clone +
//! Send`); the UI thread pulls and dispatches. This matches
//! PoseControl's own non-`Sync` model.

use reco_core::detect::director::ViewportPosition;

use crate::pose_control::{HotkeyIntent, PoseControl};
use crate::{CaptureIntent, ControlIntent, ModelSelectIntent, PoseIntent, QualityIntent};

type QualityHandler = Box<dyn FnMut(&QualityIntent) + Send>;
type CaptureHandler = Box<dyn FnMut(&CaptureIntent) + Send>;
type ModelSelectHandler = Box<dyn FnMut(&ModelSelectIntent) + Send>;

/// Dispatches [`ControlIntent`] values to a borrowed
/// [`PoseControl`] and optional per-category closures.
///
/// Build with [`IntentTranslator::new`], optionally attach handlers
/// with the `with_*_handler` methods, then call [`dispatch`] per
/// intent or [`dispatch_all`] over a slice.
///
/// [`dispatch`]: Self::dispatch
/// [`dispatch_all`]: Self::dispatch_all
pub struct IntentTranslator<'a> {
    pose: &'a mut PoseControl,
    on_quality: Option<QualityHandler>,
    on_capture: Option<CaptureHandler>,
    on_model_select: Option<ModelSelectHandler>,
}

impl<'a> IntentTranslator<'a> {
    /// Construct a translator borrowing the given [`PoseControl`].
    /// Non-pose intents are dropped until the matching
    /// `with_*_handler` is called.
    pub fn new(pose: &'a mut PoseControl) -> Self {
        Self {
            pose,
            on_quality: None,
            on_capture: None,
            on_model_select: None,
        }
    }

    /// Install a handler for [`ControlIntent::Quality`] intents.
    pub fn with_quality_handler<F>(mut self, f: F) -> Self
    where
        F: FnMut(&QualityIntent) + Send + 'static,
    {
        self.on_quality = Some(Box::new(f));
        self
    }

    /// Install a handler for [`ControlIntent::Capture`] intents.
    pub fn with_capture_handler<F>(mut self, f: F) -> Self
    where
        F: FnMut(&CaptureIntent) + Send + 'static,
    {
        self.on_capture = Some(Box::new(f));
        self
    }

    /// Install a handler for [`ControlIntent::ModelSelect`] intents.
    pub fn with_model_select_handler<F>(mut self, f: F) -> Self
    where
        F: FnMut(&ModelSelectIntent) + Send + 'static,
    {
        self.on_model_select = Some(Box::new(f));
        self
    }

    /// Dispatch a single intent.
    pub fn dispatch(&mut self, intent: ControlIntent) {
        match intent {
            ControlIntent::Hotkey(h) => self.pose.apply_hotkey(h),
            ControlIntent::Pose(p) => self.dispatch_pose(p),
            ControlIntent::Quality(q) => {
                if let Some(h) = self.on_quality.as_mut() {
                    h(&q);
                }
            }
            ControlIntent::Capture(c) => {
                if let Some(h) = self.on_capture.as_mut() {
                    h(&c);
                }
            }
            ControlIntent::ModelSelect(m) => {
                if let Some(h) = self.on_model_select.as_mut() {
                    h(&m);
                }
            }
        }
    }

    /// Dispatch a slice of intents in arrival order.
    pub fn dispatch_all(&mut self, intents: &[ControlIntent]) {
        for intent in intents {
            self.dispatch(intent.clone());
        }
    }

    fn dispatch_pose(&mut self, intent: PoseIntent) {
        let current = self.pose.target_pose();
        match intent {
            PoseIntent::SetYawRad(yaw) => self.pose.set_target(ViewportPosition {
                yaw,
                pitch: current.pitch,
                fov_degrees: None,
            }),
            PoseIntent::SetPitchRad(pitch) => self.pose.set_target(ViewportPosition {
                yaw: current.yaw,
                pitch,
                fov_degrees: None,
            }),
            PoseIntent::SetFovDeg(fov) => self.pose.set_target_fov(fov),
            PoseIntent::DeltaYawRad(dy) => self.pose.set_target(ViewportPosition {
                yaw: current.yaw + dy,
                pitch: current.pitch,
                fov_degrees: None,
            }),
            PoseIntent::DeltaPitchRad(dp) => self.pose.set_target(ViewportPosition {
                yaw: current.yaw,
                pitch: current.pitch + dp,
                fov_degrees: None,
            }),
            PoseIntent::DeltaFovDeg(df) => {
                let fov = current.fov_degrees.unwrap_or(self.pose.current_fov_deg());
                self.pose.set_target_fov(fov + df);
            }
            PoseIntent::Reset => self.pose.apply_hotkey(HotkeyIntent::Reset),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose_control::PoseControlConfig;
    use std::sync::{Arc, Mutex};

    fn make_pose() -> PoseControl {
        PoseControl::new(PoseControlConfig::default())
    }

    #[test]
    fn hotkey_routes_to_pose_control() {
        let mut pose = make_pose();
        let yaw_before = pose.target_pose().yaw;
        {
            let mut t = IntentTranslator::new(&mut pose);
            t.dispatch(ControlIntent::Hotkey(HotkeyIntent::YawRight));
        }
        let yaw_after = pose.target_pose().yaw;
        assert!(
            yaw_after > yaw_before,
            "YawRight should increase target yaw (before={yaw_before}, after={yaw_after})"
        );
    }

    #[test]
    fn pose_delta_yaw_adds_to_current_target() {
        let mut pose = make_pose();
        let start = pose.target_pose().yaw;
        let delta = 0.1_f32;
        {
            let mut t = IntentTranslator::new(&mut pose);
            t.dispatch(ControlIntent::Pose(PoseIntent::DeltaYawRad(delta)));
        }
        let after = pose.target_pose().yaw;
        assert!(
            (after - (start + delta)).abs() < 1e-5,
            "expected yaw += delta, got {after} (start={start}, delta={delta})"
        );
    }

    #[test]
    fn pose_set_fov_clamps_to_config_bounds() {
        let mut pose = make_pose();
        let fov_max = pose.config().fov_max_degrees;
        {
            let mut t = IntentTranslator::new(&mut pose);
            t.dispatch(ControlIntent::Pose(PoseIntent::SetFovDeg(fov_max + 50.0)));
        }
        assert!(
            (pose.target_pose().fov_degrees.unwrap() - fov_max).abs() < 1e-3,
            "FOV target should clamp to fov_max ({})",
            fov_max
        );
    }

    #[test]
    fn pose_reset_invokes_reset_hotkey() {
        let mut pose = make_pose();
        pose.apply_hotkey(HotkeyIntent::YawRight);
        pose.apply_hotkey(HotkeyIntent::YawRight);
        let perturbed = pose.target_pose().yaw;
        assert!(perturbed.abs() > 0.0);
        {
            let mut t = IntentTranslator::new(&mut pose);
            t.dispatch(ControlIntent::Pose(PoseIntent::Reset));
        }
        let rest = pose.config().rest_pose.yaw;
        assert!(
            (pose.target_pose().yaw - rest).abs() < 1e-5,
            "Reset should restore yaw to rest_pose.yaw"
        );
    }

    #[test]
    fn quality_intent_invokes_installed_handler() {
        let mut pose = make_pose();
        let captured = Arc::new(Mutex::new(None));
        let captured_clone = Arc::clone(&captured);
        {
            let mut t = IntentTranslator::new(&mut pose).with_quality_handler(move |q| {
                *captured_clone.lock().unwrap() = Some(q.clone());
            });
            t.dispatch(ControlIntent::Quality(QualityIntent::SetBitrate(5_000_000)));
        }
        assert_eq!(
            *captured.lock().unwrap(),
            Some(QualityIntent::SetBitrate(5_000_000)),
        );
    }

    #[test]
    fn capture_intent_without_handler_is_a_noop() {
        let mut pose = make_pose();
        let before = pose.target_pose();
        {
            let mut t = IntentTranslator::new(&mut pose);
            t.dispatch(ControlIntent::Capture(CaptureIntent::Snapshot));
        }
        // No handler installed; dispatching should not touch pose.
        let after = pose.target_pose();
        assert_eq!(before.yaw, after.yaw);
        assert_eq!(before.pitch, after.pitch);
    }

    #[test]
    fn dispatch_all_preserves_arrival_order() {
        let mut pose = make_pose();
        let intents = vec![
            ControlIntent::Hotkey(HotkeyIntent::YawRight),
            ControlIntent::Hotkey(HotkeyIntent::YawRight),
            ControlIntent::Pose(PoseIntent::Reset),
        ];
        {
            let mut t = IntentTranslator::new(&mut pose);
            t.dispatch_all(&intents);
        }
        // Final intent is Reset, so yaw should be at rest.
        let rest = pose.config().rest_pose.yaw;
        assert!((pose.target_pose().yaw - rest).abs() < 1e-5);
    }
}
