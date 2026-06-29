//! Central dispatch from [`ControlIntent`] to consumer-side pose state.
//!
//! # Purpose
//!
//! Every consumer (reco-gui, reco-cli preview, reco-obs) turns a
//! [`ControlIntent`] stream into calls on
//! [`PoseControl`](crate::pose_control::PoseControl). Before this type,
//! each consumer wrote that dispatch inline. [`IntentTranslator`] owns
//! the translation once.
//!
//! # Lifetime and threading
//!
//! Translators borrow `PoseControl` for a single dispatch tick and are
//! not `Send`. Cross-thread intent forwarding goes through an mpsc
//! channel of [`ControlIntent`] values (which *are* `Clone + Send`); the
//! UI thread pulls and dispatches. This matches PoseControl's own
//! non-`Sync` model.

use reco_core::detect::director::ViewportPosition;

use crate::pose_control::{HotkeyIntent, PoseControl};
use crate::{ControlIntent, PoseIntent};

/// Dispatches [`ControlIntent`] values to a borrowed [`PoseControl`].
///
/// Build with [`IntentTranslator::new`], then call [`dispatch`] per
/// intent or [`dispatch_all`] over a slice.
///
/// [`dispatch`]: Self::dispatch
/// [`dispatch_all`]: Self::dispatch_all
pub struct IntentTranslator<'a> {
    pose: &'a mut PoseControl,
}

impl<'a> IntentTranslator<'a> {
    /// Construct a translator borrowing the given [`PoseControl`].
    pub fn new(pose: &'a mut PoseControl) -> Self {
        Self { pose }
    }

    /// Dispatch a single intent.
    pub fn dispatch(&mut self, intent: ControlIntent) {
        match intent {
            ControlIntent::Hotkey(h) => self.pose.apply_hotkey(h),
            ControlIntent::Pose(p) => self.dispatch_pose(p),
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
