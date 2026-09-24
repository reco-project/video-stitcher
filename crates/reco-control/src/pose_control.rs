//! Direct viewport input shared by CLI preview, GUI and OBS.
//!
//! Inputs update a render-space target. Call [`PoseControl::constrain`](crate::pose_control::PoseControl::constrain) after each input and before rendering to accept it through the engine's coverage policy. Blocked deltas are discarded, so releasing or reversing a drag has no residual motion. Automatic tracking and its timing belong to the engine, not these controls.
use reco_core::core::{StitchCore, types::StitchCoreError};
use reco_core::geometry::Pose;

use crate::{ControlIntent, PoseIntent};

/// Hotkey actions consumers bind to their input system (OBS hotkey
/// API, Slint key events, CLI keyboard, future SDL3 game-pad sidecar,
/// remote `reco-control` transport).
///
/// Consumers translate their native key events to these intents and
/// pass them to [`PoseControl::apply_hotkey`]. The mapping
/// key-code → intent lives in the consumer; this type is the
/// cross-consumer vocabulary.
///
/// `#[non_exhaustive]` so new intents can be added without a breaking
/// change (e.g. `SetFov(f32)`, `PresetRecall(u8)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum HotkeyIntent {
    /// Pan the target yaw left by [`PoseControlConfig::hotkey_yaw_step_rad`].
    YawLeft,
    /// Pan the target yaw right by the same step.
    YawRight,
    /// Pan the target pitch up by [`PoseControlConfig::hotkey_pitch_step_rad`].
    PitchUp,
    /// Pan the target pitch down by the same step.
    PitchDown,
    /// Zoom in (narrow FOV) by [`PoseControlConfig::hotkey_fov_step_deg`].
    ZoomIn,
    /// Zoom out (widen FOV) by the same step.
    ZoomOut,
    /// Reset target yaw/pitch/FOV to the configured resting pose.
    Reset,
    /// Toggle the "constrained look" mode. Consumers listen for this
    /// intent and flip a boolean that gates coverage-clamp behavior;
    /// PoseControl itself stays stateless on the toggle (it is merely
    /// forwarded through [`PoseControl::apply_hotkey`] so the consumer can react).
    ToggleConstrained,
}

/// Zoom limits, input steps, axis directions and reset pose for direct interaction.
#[derive(Debug, Clone, Copy)]
pub struct PoseControlConfig {
    /// Wheel sensitivity in FOV-degrees-per-tick. Positive tick
    /// counts narrow FOV (zoom in); negative widen. `3.0` is
    /// comfortable on mouse wheels and trackpad scrolls.
    pub wheel_fov_per_tick: f32,

    /// Minimum FOV in degrees (zoomed-in limit). The default matches the engine's one-degree coverage floor.
    pub fov_min_degrees: f32,

    /// Maximum FOV in degrees (zoomed-out limit), the baseline ceiling.
    /// The engine applies its geometric cap without changing this baseline.
    pub fov_max_degrees: f32,

    /// Whether drag deltas on the X axis are inverted. Off
    /// ([`false`], default) follows the "drag the scene" convention
    /// (drag right → content moves right → camera yaws left), same
    /// as Google Maps / photo viewers. Turn on ([`true`]) for the
    /// "drag the camera" / PTZ-head convention (drag right → camera
    /// yaws right → content moves left). reco-obs uses `true`.
    pub invert_drag_x: bool,

    /// Whether drag deltas on the Y axis are inverted. Off
    /// ([`false`], default) follows the "drag the scene" convention
    /// (drag down → content moves down → camera pitches up). Turn on
    /// ([`true`]) for the "drag the camera" convention (drag down →
    /// camera pitches down → content moves up).
    pub invert_drag_y: bool,

    /// Hotkey yaw step in radians per intent. `5.0 deg` default.
    pub hotkey_yaw_step_rad: f32,

    /// Hotkey pitch step in radians per intent. `5.0 deg` default.
    pub hotkey_pitch_step_rad: f32,

    /// Hotkey FOV step in degrees per intent. `5.0 deg` default.
    pub hotkey_fov_step_deg: f32,

    /// Resting pose the [`HotkeyIntent::Reset`] intent restores.
    pub rest_pose: Pose,
}

impl Default for PoseControlConfig {
    fn default() -> Self {
        Self {
            wheel_fov_per_tick: 3.0,
            fov_min_degrees: 1.0,
            fov_max_degrees: 150.0,
            invert_drag_x: false,
            invert_drag_y: false,
            hotkey_yaw_step_rad: (5.0_f32).to_radians(),
            hotkey_pitch_step_rad: (5.0_f32).to_radians(),
            hotkey_fov_step_deg: 5.0,
            rest_pose: Pose {
                yaw: 0.0,
                pitch: 0.0,
                fov_degrees: 75.0,
            },
        }
    }
}

/// Unified pose state machine driven by input events.
///
/// Owns the render-space input target and the last accepted pose. [`Self::constrain`] fits the target through the engine, then stores the accepted result in both. Do not apply world-to-render orientation again at the render site.
#[derive(Debug, Clone)]
pub struct PoseControl {
    target_yaw_rad: f32,
    target_pitch_rad: f32,
    target_fov_deg: f32,

    current_yaw_rad: f32,
    current_pitch_rad: f32,
    current_fov_deg: f32,

    config: PoseControlConfig,
}

impl PoseControl {
    /// Build a new `PoseControl` with the supplied config. The
    /// initial target + current pose are set to `config.rest_pose`.
    pub fn new(config: PoseControlConfig) -> Self {
        let rest = config.rest_pose;
        let fov = rest.fov_degrees;
        Self {
            target_yaw_rad: rest.yaw,
            target_pitch_rad: rest.pitch,
            target_fov_deg: fov,
            current_yaw_rad: rest.yaw,
            current_pitch_rad: rest.pitch,
            current_fov_deg: fov,
            config,
        }
    }

    /// Build a `PoseControl` with the default input settings.
    pub fn with_defaults() -> Self {
        Self::new(PoseControlConfig::default())
    }

    // ---------------------------------------------------------------
    // Input events
    // ---------------------------------------------------------------

    /// Apply a drag in displayed-viewport pixels, scaled by current vertical FOV and viewport height.
    ///
    /// Supply the displayed image height, not the source or render-target height. Invalid dimensions and non-finite deltas are ignored. Axis direction is controlled by the invert flags.
    pub fn apply_drag(&mut self, dx_pixels: f32, dy_pixels: f32, viewport_height: f32) {
        if !viewport_height.is_finite()
            || viewport_height <= 0.0
            || !dx_pixels.is_finite()
            || !dy_pixels.is_finite()
        {
            return;
        }
        let radians_per_pixel =
            2.0 * (self.current_fov_deg.to_radians() * 0.5).tan() / viewport_height;
        // The baseline convention is "drag the scene": drag right
        // (positive dx) yaws the camera left (yaw decreases), drag
        // down (positive dy) pitches the camera up (pitch increases).
        // `invert_drag_x` / `invert_drag_y` flip each axis
        // independently for consumers that prefer PTZ-head semantics.
        let raw_dx = if self.config.invert_drag_x {
            dx_pixels
        } else {
            -dx_pixels
        };
        let raw_dy = if self.config.invert_drag_y {
            -dy_pixels
        } else {
            dy_pixels
        };
        let dx_rad = raw_dx * radians_per_pixel;
        let dy_rad = raw_dy * radians_per_pixel;
        self.target_yaw_rad += dx_rad;
        self.target_pitch_rad += dy_rad;
    }

    /// Apply a scroll-wheel delta. `ticks > 0` zooms in
    /// (narrows FOV); `ticks < 0` zooms out. Fractional ticks
    /// from trackpads are respected.
    pub fn apply_wheel(&mut self, ticks: f32) {
        let delta_deg = -ticks * self.config.wheel_fov_per_tick;
        self.set_target_fov(self.target_fov_deg + delta_deg);
    }

    /// Apply a [`HotkeyIntent`]. Pan / zoom / reset intents move the
    /// target pose by the configured step; [`HotkeyIntent::ToggleConstrained`]
    /// is a no-op here so the consumer can observe the intent via
    /// their own dispatch (PoseControl holds no constrained-look state).
    pub fn apply_hotkey(&mut self, intent: HotkeyIntent) {
        match intent {
            HotkeyIntent::YawLeft => self.target_yaw_rad -= self.config.hotkey_yaw_step_rad,
            HotkeyIntent::YawRight => self.target_yaw_rad += self.config.hotkey_yaw_step_rad,
            HotkeyIntent::PitchUp => self.target_pitch_rad += self.config.hotkey_pitch_step_rad,
            HotkeyIntent::PitchDown => self.target_pitch_rad -= self.config.hotkey_pitch_step_rad,
            HotkeyIntent::ZoomIn => {
                self.set_target_fov(self.target_fov_deg - self.config.hotkey_fov_step_deg);
            }
            HotkeyIntent::ZoomOut => {
                self.set_target_fov(self.target_fov_deg + self.config.hotkey_fov_step_deg);
            }
            HotkeyIntent::Reset => {
                let rest = self.config.rest_pose;
                self.target_yaw_rad = rest.yaw;
                self.target_pitch_rad = rest.pitch;
                self.target_fov_deg = rest.fov_degrees;
            }
            HotkeyIntent::ToggleConstrained => { /* consumer-side */ }
        }
    }

    /// Apply a [`ControlIntent`]: hotkeys route through
    /// [`Self::apply_hotkey`]; pose intents set or nudge the target.
    pub fn apply_intent(&mut self, intent: ControlIntent) {
        match intent {
            ControlIntent::Hotkey(h) => self.apply_hotkey(h),
            ControlIntent::Pose(p) => self.apply_pose_intent(p),
        }
    }

    fn apply_pose_intent(&mut self, intent: PoseIntent) {
        let current = self.target_pose();
        match intent {
            PoseIntent::SetYawRad(yaw) => self.set_target(Pose { yaw, ..current }),
            PoseIntent::SetPitchRad(pitch) => self.set_target(Pose { pitch, ..current }),
            PoseIntent::SetFovDeg(fov) => self.set_target_fov(fov),
            PoseIntent::DeltaYawRad(dy) => self.set_target(Pose {
                yaw: current.yaw + dy,
                ..current
            }),
            PoseIntent::DeltaPitchRad(dp) => self.set_target(Pose {
                pitch: current.pitch + dp,
                ..current
            }),
            PoseIntent::DeltaFovDeg(df) => {
                self.set_target_fov(current.fov_degrees + df);
            }
            PoseIntent::Reset => self.apply_hotkey(HotkeyIntent::Reset),
        }
    }

    /// Replace the pending render-space pose. [`Self::constrain`] accepts it before rendering.
    pub fn set_target(&mut self, pose: Pose) {
        self.target_yaw_rad = pose.yaw;
        self.target_pitch_rad = pose.pitch;
        self.set_target_fov(pose.fov_degrees);
    }

    /// Set the target FOV directly, in degrees. Clamped to the
    /// configured `[fov_min, fov_max]` range.
    pub fn set_target_fov(&mut self, fov_deg: f32) {
        self.target_fov_deg = fov_deg.clamp(
            self.config.fov_min_degrees.min(self.config.fov_max_degrees),
            self.config.fov_max_degrees,
        );
    }

    // ---------------------------------------------------------------
    // Coverage clamping
    // ---------------------------------------------------------------

    /// Resolve direct render-space input immediately through the engine's coverage policy.
    ///
    /// Accepts the fitted pose as both current and target, discarding blocked input without smoothing or changing the configured zoom range. Errors preserve both poses. Call after each input and after viewport or calibration changes, before rendering.
    pub fn constrain(&mut self, core: &mut StitchCore) -> Result<(), StitchCoreError> {
        let accepted = core.constrain_render_pose(self.current_pose(), self.target_pose())?;
        self.set_pose(accepted);
        Ok(())
    }

    /// Replace current and target together after an external coordinate-frame change.
    ///
    /// Does not validate coverage or change the configured reset pose. Call [`Self::constrain`] before rendering under the new calibration.
    pub fn set_pose(&mut self, accepted: Pose) {
        self.target_yaw_rad = accepted.yaw;
        self.target_pitch_rad = accepted.pitch;
        self.target_fov_deg = accepted.fov_degrees;
        self.current_yaw_rad = accepted.yaw;
        self.current_pitch_rad = accepted.pitch;
        self.current_fov_deg = accepted.fov_degrees;
    }

    // ---------------------------------------------------------------
    // Readback
    // ---------------------------------------------------------------

    /// The target pose (what the input layer last set).
    pub fn target_pose(&self) -> Pose {
        Pose {
            yaw: self.target_yaw_rad,
            pitch: self.target_pitch_rad,
            fov_degrees: self.target_fov_deg,
        }
    }

    /// The current pose the renderer should draw this frame.
    pub fn current_pose(&self) -> Pose {
        Pose {
            yaw: self.current_yaw_rad,
            pitch: self.current_pitch_rad,
            fov_degrees: self.current_fov_deg,
        }
    }

    /// Current yaw in radians.
    pub fn current_yaw_rad(&self) -> f32 {
        self.current_yaw_rad
    }

    /// Current pitch in radians.
    pub fn current_pitch_rad(&self) -> f32 {
        self.current_pitch_rad
    }

    /// Current FOV in degrees.
    pub fn current_fov_deg(&self) -> f32 {
        self.current_fov_deg
    }
}

// Worker callbacks can own the controls; mutations require exclusive access.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    fn assert_clone<T: Clone>() {}
    assert_send::<PoseControl>();
    assert_clone::<PoseControl>();
    assert_clone::<PoseControlConfig>();
    assert_clone::<HotkeyIntent>();
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Shorthand: build with defaults + snap to a known rest pose.
    fn fresh() -> PoseControl {
        PoseControl::with_defaults()
    }

    // ---- Defaults -----------------------------------------------------

    #[test]
    fn defaults_use_degrees_for_zoom_and_radians_for_pan() {
        let cfg = PoseControlConfig::default();
        assert_eq!(cfg.fov_min_degrees, 1.0);
        assert_eq!(cfg.fov_max_degrees, 150.0);
        assert_eq!(cfg.wheel_fov_per_tick, 3.0);
        assert_eq!(cfg.rest_pose.fov_degrees, 75.0);
    }

    // ---- Drag ---------------------------------------------------------

    #[test]
    fn drag_right_decreases_yaw() {
        // Drag right = pan camera right = yaw decreases (camera looks
        // further left of the scene; the grab-the-world drag convention).
        let mut p = fresh();
        let y0 = p.target_yaw_rad;
        p.apply_drag(100.0, 0.0, 600.0);
        assert!(p.target_yaw_rad < y0);
    }

    #[test]
    fn drag_uses_perspective_scale() {
        let mut p = fresh();
        p.apply_drag(100.0, 0.0, 600.0);
        let expected = -2.0 * (p.current_fov_deg.to_radians() * 0.5).tan() * 100.0 / 600.0;
        assert!((p.target_yaw_rad - expected).abs() < 1e-5);
    }

    #[test]
    fn drag_scales_with_display_size_and_ignores_invalid_input() {
        let mut small = fresh();
        let mut large = fresh();
        small.apply_drag(30.0, -20.0, 300.0);
        large.apply_drag(60.0, -40.0, 600.0);
        assert_eq!(small.target_pose(), large.target_pose());
        let before = small.target_pose();
        for height in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            small.apply_drag(10.0, 10.0, height);
        }
        small.apply_drag(f32::NAN, 0.0, 300.0);
        small.apply_drag(0.0, f32::INFINITY, 300.0);
        assert_eq!(small.target_pose(), before);
    }

    #[test]
    fn narrow_coverage_does_not_prevent_zooming_in() {
        let mut p = PoseControl::new(PoseControlConfig {
            rest_pose: Pose {
                fov_degrees: 12.0,
                ..Pose::default()
            },
            ..Default::default()
        });
        p.apply_wheel(1.0);
        assert_eq!(p.target_pose().fov_degrees, 9.0);
    }

    #[test]
    fn invert_drag_y_flips_pitch_direction() {
        let mut p = PoseControl::new(PoseControlConfig {
            invert_drag_y: true,
            ..Default::default()
        });
        let p0 = p.target_pitch_rad;
        p.apply_drag(0.0, 50.0, 600.0);
        assert!(
            p.target_pitch_rad < p0,
            "invert_y: positive dy lowers pitch"
        );
    }

    // ---- Wheel --------------------------------------------------------

    #[test]
    fn wheel_positive_zooms_in() {
        let mut p = fresh();
        let f0 = p.target_fov_deg;
        p.apply_wheel(1.0);
        assert!(p.target_fov_deg < f0, "positive ticks narrow FOV");
    }

    #[test]
    fn wheel_clamps_to_fov_range() {
        let mut p = fresh();
        // Wheel way past min.
        for _ in 0..1000 {
            p.apply_wheel(1.0);
        }
        assert_eq!(p.target_fov_deg, p.config.fov_min_degrees);
        // Wheel way past max.
        for _ in 0..1000 {
            p.apply_wheel(-1.0);
        }
        assert_eq!(p.target_fov_deg, p.config.fov_max_degrees);
    }

    // ---- Hotkeys ------------------------------------------------------

    #[test]
    fn hotkey_yaw_left_right_are_symmetric() {
        let mut p = fresh();
        let y0 = p.target_yaw_rad;
        p.apply_hotkey(HotkeyIntent::YawLeft);
        p.apply_hotkey(HotkeyIntent::YawRight);
        assert!((p.target_yaw_rad - y0).abs() < 1e-6);
    }

    #[test]
    fn hotkey_reset_returns_target_to_rest() {
        let mut p = fresh();
        p.apply_drag(500.0, -300.0, 600.0);
        p.apply_wheel(10.0);
        p.apply_hotkey(HotkeyIntent::Reset);
        let rest = p.config.rest_pose;
        assert!((p.target_yaw_rad - rest.yaw).abs() < 1e-5);
        assert!((p.target_pitch_rad - rest.pitch).abs() < 1e-5);
        assert_eq!(p.target_fov_deg, rest.fov_degrees);
    }

    #[test]
    fn toggle_constrained_is_a_noop_on_pose() {
        let mut p = fresh();
        let before = (p.target_yaw_rad, p.target_pitch_rad, p.target_fov_deg);
        p.apply_hotkey(HotkeyIntent::ToggleConstrained);
        let after = (p.target_yaw_rad, p.target_pitch_rad, p.target_fov_deg);
        assert_eq!(before, after);
    }

    // ---- set_target ---------------------------------------------------

    #[test]
    fn set_target_waits_for_acceptance() {
        let mut p = fresh();
        let c0 = p.current_yaw_rad;
        p.set_target(Pose {
            yaw: 0.5,
            pitch: -0.2,
            fov_degrees: 60.0,
        });
        assert_eq!(p.current_yaw_rad, c0);
        assert_eq!(p.target_yaw_rad, 0.5);
        assert_eq!(p.target_fov_deg, 60.0);
    }

    // ---- apply_intent -------------------------------------------------

    #[test]
    fn apply_intent_hotkey_routes_to_pose() {
        let mut p = fresh();
        let before = p.target_yaw_rad;
        p.apply_intent(ControlIntent::Hotkey(HotkeyIntent::YawRight));
        assert!(p.target_yaw_rad > before);
    }

    #[test]
    fn apply_intent_pose_delta_adds_to_target() {
        let mut p = fresh();
        let start = p.target_yaw_rad;
        p.apply_intent(ControlIntent::Pose(PoseIntent::DeltaYawRad(0.1)));
        assert!((p.target_yaw_rad - (start + 0.1)).abs() < 1e-5);
    }

    #[test]
    fn apply_intent_pose_reset_restores_rest() {
        let mut p = fresh();
        p.apply_intent(ControlIntent::Hotkey(HotkeyIntent::YawRight));
        p.apply_intent(ControlIntent::Pose(PoseIntent::Reset));
        assert!((p.target_yaw_rad - p.config.rest_pose.yaw).abs() < 1e-5);
    }
}
#[test]
fn direct_acceptance_copies_the_target_exactly_after_large_reversals() {
    use reco_core::{
        calibration::{Calibration, Framing, Lens},
        projection::Cylinder,
        render::viewport::ViewportSize,
        stitch::{CpuExecutor, Executor},
    };
    let cal = Calibration::new(
        vec![Lens::flat(64, 32)],
        Cylinder::default(),
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let executor = CpuExecutor::new(
        cal,
        ViewportSize {
            width: 64,
            height: 32,
        },
        64,
        32,
        false,
    )
    .unwrap();
    let mut core = StitchCore::new(Executor::Cpu(Box::new(executor))).unwrap();
    core.set_constrained_look(false);
    let mut control = PoseControl::with_defaults();
    control.set_target_fov(1.0);
    control.constrain(&mut core).unwrap();
    assert_eq!(control.current_fov_deg(), 1.0);
    for yaw in [2.9, -0.0000001, -2.9, 0.0000001] {
        let p = Pose {
            yaw,
            pitch: -yaw * 0.1,
            fov_degrees: 37.12345,
        };
        control.set_target(p);
        control.constrain(&mut core).unwrap();
        assert_eq!(control.current_pose(), p);
    }
}
