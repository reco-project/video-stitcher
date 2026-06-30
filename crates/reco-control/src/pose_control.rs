//! Unified pose control: mouse / drag / wheel / keyboard → yaw / pitch / FOV.
//!
//! Solves the three divergent implementations the 2026-04-18 deep
//! review found across consumers (reco-cli/preview, reco-gui, reco-obs):
//! each had its own units (radians vs degrees), FOV clamp
//! (20-150deg in two of them, no clamp at all in reco-obs), drag
//! sensitivity (hand-tuned differently), and smoothing factor
//! (0.3 vs 0.25 vs zero). With PoseControl those all come from
//! one place, configurable per-consumer but with plan-mandated
//! defaults (radians internal, degrees for FOV).
//!
//! # Usage
//!
//! Consumers instantiate a `PoseControl` per session, feed it input events
//! via `apply_drag`, `apply_wheel`, `apply_hotkey`, and `set_target`, then
//! call `tick` once per frame to ease the *current* pose toward the
//! *target* pose. The current pose is what the renderer submits; the
//! target is what the input layer wanted. Optional coverage-based
//! clamping keeps the viewport inside the no-black region.
//!
//! ```rust,ignore
//! use reco_core::pose_control::{PoseControl, PoseControlConfig};
//!
//! let mut pose = PoseControl::with_defaults();
//!
//! // Mouse drag from UI layer:
//! pose.apply_drag(delta_x_pixels, delta_y_pixels);
//! // Scroll wheel (one notch up = zoom in a bit):
//! pose.apply_wheel(1.0);
//! // Per-frame smoothing tick:
//! pose.tick();
//! // (Optional) clamp current pose through the session's coverage:
//! pose.clamp_via_coverage(core.coverage().unwrap(), aspect);
//! // Feed the renderer (world-space pose -> StitchRenderer::orient_pose):
//! let vp = pose.current_pose();
//! ```
//!
//! # Units
//!
//! Per plan-execution §3 M4: **radians internal** for yaw/pitch;
//! **degrees** for FOV (matches the rest of the reco-core pipeline
//! and the viewport config). Conversion helpers live on
//! `apply_drag` / `apply_wheel` where the UI layer hands over pixel or
//! tick deltas.
//!
//! # Thread safety
//!
//! Not `Sync` (holds mutable drag state). Consumers that need
//! cross-thread access wrap in `Mutex`. Send-only is sufficient for
//! the worker / UI split every consumer actually needs.

use reco_core::detect::director::ViewportPosition;
use reco_core::projection::CoverageBoundary;

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

/// Configuration knobs for [`PoseControl`]. All have sensible
/// defaults; consumers typically only override `drag_deg_per_pixel`
/// or `smoothing` if the feel is wrong on their input device.
#[derive(Debug, Clone, Copy)]
pub struct PoseControlConfig {
    /// Drag sensitivity in degrees-per-pixel for both axes. reco-obs
    /// shipped `0.1`; reco-gui + reco-cli used hand-tuned multipliers
    /// that work out to roughly the same after accounting for their
    /// local radian conversions. `0.1` is the harmonized default.
    pub drag_deg_per_pixel: f32,

    /// Wheel sensitivity in FOV-degrees-per-tick. Positive tick
    /// counts narrow FOV (zoom in); negative widen. `3.0` is
    /// comfortable on mouse wheels and trackpad scrolls.
    pub wheel_fov_per_tick: f32,

    /// Per-tick smoothing factor in `[0, 1]`. `1.0` disables
    /// smoothing (current snaps to target); lower values give a
    /// critical-damped ease. Default `0.3` matches reco-cli; GUI
    /// consumers that want tighter response override to `0.5` or
    /// higher.
    pub smoothing: f32,

    /// Minimum FOV in degrees (zoomed-in limit). `20.0` matches
    /// reco-cli + reco-gui.
    pub fov_min_degrees: f32,

    /// Maximum FOV in degrees (zoomed-out limit), the baseline ceiling.
    /// `clamp_via_coverage` applies the tighter `coverage.max_fov_degrees()`
    /// limit transiently (without changing this baseline) so the viewport
    /// never reveals black edges while constrained look is on.
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
    pub rest_pose: ViewportPosition,
}

impl Default for PoseControlConfig {
    fn default() -> Self {
        Self {
            drag_deg_per_pixel: 0.1,
            wheel_fov_per_tick: 3.0,
            smoothing: 0.3,
            fov_min_degrees: 20.0,
            fov_max_degrees: 150.0,
            invert_drag_x: false,
            invert_drag_y: false,
            hotkey_yaw_step_rad: (5.0_f32).to_radians(),
            hotkey_pitch_step_rad: (5.0_f32).to_radians(),
            hotkey_fov_step_deg: 5.0,
            rest_pose: ViewportPosition {
                yaw: 0.0,
                pitch: 0.0,
                fov_degrees: Some(75.0),
            },
        }
    }
}

/// Unified pose state machine driven by input events.
///
/// Owns two poses: `target` (what the input layer just wanted) and
/// `current` (what the renderer should draw this frame). Each `tick`
/// eases current toward target by `config.smoothing`.
///
/// Poses are stored in **world space** (the panorama's native frame,
/// matching the AI/director path). The rig tilt+roll correction that
/// keeps the horizon level under pan is applied at the render site by
/// [`StitchRenderer::orient_pose`](reco_core::render::stitch_renderer::StitchRenderer::orient_pose),
/// not here; coverage clamping ([`Self::clamp_via_coverage`]) is also
/// world-space.
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
        let fov = rest.fov_degrees.unwrap_or(75.0);
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

    /// Build a `PoseControl` with plan-mandated defaults.
    pub fn with_defaults() -> Self {
        Self::new(PoseControlConfig::default())
    }

    // ---------------------------------------------------------------
    // Input events
    // ---------------------------------------------------------------

    /// Apply a pixel-space drag delta. The UI layer computes the
    /// delta between consecutive mouse-move events; this method
    /// translates it into radians using `config.drag_deg_per_pixel`.
    ///
    /// Dragging right (positive dx) pans the camera right, which
    /// means **decreasing** yaw (camera looks further left of the
    /// scene). Dragging down (positive dy) pans down (pitch
    /// decreases) unless `invert_drag_y` is set. This matches
    /// standard panning-camera conventions.
    pub fn apply_drag(&mut self, dx_pixels: f32, dy_pixels: f32) {
        let deg_per_pixel = self.config.drag_deg_per_pixel;
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
        let dx_rad = raw_dx * deg_per_pixel.to_radians();
        let dy_rad = raw_dy * deg_per_pixel.to_radians();
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
                self.target_fov_deg = rest.fov_degrees.unwrap_or(self.target_fov_deg);
            }
            HotkeyIntent::ToggleConstrained => { /* consumer-side */ }
        }
    }

    /// Replace the target pose outright. Used when an external
    /// source (an AI director, a calibration reset, a replay seek)
    /// produces a pose the UI layer did not; current still eases to
    /// it over the next few ticks.
    pub fn set_target(&mut self, pose: ViewportPosition) {
        self.target_yaw_rad = pose.yaw;
        self.target_pitch_rad = pose.pitch;
        if let Some(fov) = pose.fov_degrees {
            self.set_target_fov(fov);
        }
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
    // Per-frame step
    // ---------------------------------------------------------------

    /// Advance the current pose toward the target using the
    /// configured smoothing factor.
    pub fn tick(&mut self) {
        let s = self.config.smoothing;
        self.tick_with(s);
    }

    /// Advance with an explicit smoothing factor (called by [`Self::tick`]).
    fn tick_with(&mut self, smoothing: f32) {
        let s = smoothing.clamp(0.0, 1.0);
        self.current_yaw_rad += (self.target_yaw_rad - self.current_yaw_rad) * s;
        self.current_pitch_rad += (self.target_pitch_rad - self.current_pitch_rad) * s;
        self.current_fov_deg += (self.target_fov_deg - self.current_fov_deg) * s;
    }

    // ---------------------------------------------------------------
    // Coverage clamping
    // ---------------------------------------------------------------

    /// Clamp both target and current pose through the session's
    /// coverage boundary, and narrow `config.fov_max_degrees` to
    /// `coverage.max_fov_degrees()`. This keeps the viewport inside
    /// the no-black region the projection can render. Safe to call
    /// every tick.
    ///
    /// The stored poses are world-space (the panorama's native
    /// coordinate frame, matching the AI/director path), so this is a
    /// direct world-space clamp with no rig-tilt round-trip.
    pub fn clamp_via_coverage(&mut self, coverage: &CoverageBoundary, aspect: f32) {
        // Coverage FOV ceiling for this pass, bounded by the configured
        // baseline. Applied transiently to target/current FOV; the config's
        // `fov_max_degrees` baseline is left untouched, so disabling
        // constrained look (which stops calling this) frees the FOV again.
        let max_fov = coverage.max_fov_degrees().min(self.config.fov_max_degrees);
        self.target_fov_deg = self.target_fov_deg.min(max_fov);
        self.current_fov_deg = self.current_fov_deg.min(max_fov);

        // target_*/current_* are world-space, as is the coverage
        // boundary, so clamp directly (no human<->world mapping).
        let clamp = |yaw: f32, pitch: f32, fov: f32| -> (f32, f32) {
            let clamped = coverage.safe_clamp(yaw, pitch, fov, aspect);
            (clamped.yaw, clamped.pitch)
        };
        let (ty, tp) = clamp(
            self.target_yaw_rad,
            self.target_pitch_rad,
            self.target_fov_deg,
        );
        self.target_yaw_rad = ty;
        self.target_pitch_rad = tp;

        let (cy, cp) = clamp(
            self.current_yaw_rad,
            self.current_pitch_rad,
            self.current_fov_deg,
        );
        self.current_yaw_rad = cy;
        self.current_pitch_rad = cp;
    }

    // ---------------------------------------------------------------
    // Readback
    // ---------------------------------------------------------------

    /// The target pose (what the input layer last set).
    pub fn target_pose(&self) -> ViewportPosition {
        ViewportPosition {
            yaw: self.target_yaw_rad,
            pitch: self.target_pitch_rad,
            fov_degrees: Some(self.target_fov_deg),
        }
    }

    /// The current pose the renderer should draw this frame.
    pub fn current_pose(&self) -> ViewportPosition {
        ViewportPosition {
            yaw: self.current_yaw_rad,
            pitch: self.current_pitch_rad,
            fov_degrees: Some(self.current_fov_deg),
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

    /// Borrowed config.
    pub fn config(&self) -> &PoseControlConfig {
        &self.config
    }

    /// Replace the config. Does not re-clamp the current pose —
    /// call [`PoseControl::clamp_via_coverage`] or [`PoseControl::tick`] afterward if needed.
    pub fn set_config(&mut self, config: PoseControlConfig) {
        self.config = config;
    }
}

// Compile-time bound check: PoseControl is `Send` so a worker thread
// (e.g. OBS tick callback) can hold one. Not `Sync` — drags mutate
// state; cross-thread shared access requires a `Mutex`.
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
    fn defaults_match_plan_mandate() {
        // Plan: radians internal + degrees for FOV + 20-150deg clamp.
        let cfg = PoseControlConfig::default();
        assert_eq!(cfg.fov_min_degrees, 20.0);
        assert_eq!(cfg.fov_max_degrees, 150.0);
        assert_eq!(cfg.drag_deg_per_pixel, 0.1);
        assert_eq!(cfg.wheel_fov_per_tick, 3.0);
        assert_eq!(cfg.rest_pose.fov_degrees, Some(75.0));
    }

    // ---- Drag ---------------------------------------------------------

    #[test]
    fn drag_right_decreases_yaw() {
        // Drag right = pan camera right = yaw decreases (camera looks
        // further left of the scene; matches actionstitch convention).
        let mut p = fresh();
        let y0 = p.target_yaw_rad;
        p.apply_drag(100.0, 0.0);
        assert!(p.target_yaw_rad < y0);
    }

    #[test]
    fn drag_uses_configured_sensitivity() {
        // 100px at 0.1 deg/px = 10 degrees = 0.1745 rad.
        let mut p = fresh();
        p.apply_drag(100.0, 0.0);
        let expected = -(10.0_f32).to_radians();
        assert!((p.target_yaw_rad - expected).abs() < 1e-5);
    }

    #[test]
    fn invert_drag_y_flips_pitch_direction() {
        let mut p = PoseControl::new(PoseControlConfig {
            invert_drag_y: true,
            ..Default::default()
        });
        let p0 = p.target_pitch_rad;
        p.apply_drag(0.0, 50.0);
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
        p.apply_drag(500.0, -300.0);
        p.apply_wheel(10.0);
        p.apply_hotkey(HotkeyIntent::Reset);
        let rest = p.config.rest_pose;
        assert!((p.target_yaw_rad - rest.yaw).abs() < 1e-5);
        assert!((p.target_pitch_rad - rest.pitch).abs() < 1e-5);
        assert_eq!(p.target_fov_deg, rest.fov_degrees.unwrap());
    }

    #[test]
    fn toggle_constrained_is_a_noop_on_pose() {
        let mut p = fresh();
        let before = (p.target_yaw_rad, p.target_pitch_rad, p.target_fov_deg);
        p.apply_hotkey(HotkeyIntent::ToggleConstrained);
        let after = (p.target_yaw_rad, p.target_pitch_rad, p.target_fov_deg);
        assert_eq!(before, after);
    }

    // ---- Tick (smoothing) ---------------------------------------------

    #[test]
    fn tick_eases_current_toward_target() {
        let mut p = fresh();
        p.apply_drag(100.0, 0.0); // moves target only
        let before = p.current_yaw_rad;
        p.tick();
        let after = p.current_yaw_rad;
        let target = p.target_yaw_rad;
        assert!(
            (before - after).abs() > 0.0 && (after - target).abs() < (before - target).abs(),
            "tick moves current partway toward target"
        );
    }

    #[test]
    fn tick_with_one_snaps_to_target() {
        let mut p = fresh();
        p.apply_drag(100.0, 50.0);
        p.apply_wheel(2.0);
        p.tick_with(1.0);
        assert!((p.current_yaw_rad - p.target_yaw_rad).abs() < 1e-6);
        assert!((p.current_pitch_rad - p.target_pitch_rad).abs() < 1e-6);
        assert!((p.current_fov_deg - p.target_fov_deg).abs() < 1e-6);
    }

    #[test]
    fn tick_with_zero_does_not_move_current() {
        let mut p = fresh();
        p.apply_drag(100.0, 50.0);
        let before = (p.current_yaw_rad, p.current_pitch_rad);
        p.tick_with(0.0);
        assert_eq!(before, (p.current_yaw_rad, p.current_pitch_rad));
    }

    // ---- Convergence --------------------------------------------------

    #[test]
    fn repeated_ticks_converge_to_target() {
        let mut p = fresh();
        p.apply_drag(200.0, -100.0);
        p.apply_wheel(5.0);
        for _ in 0..200 {
            p.tick();
        }
        assert!((p.current_yaw_rad - p.target_yaw_rad).abs() < 1e-3);
        assert!((p.current_pitch_rad - p.target_pitch_rad).abs() < 1e-3);
        assert!((p.current_fov_deg - p.target_fov_deg).abs() < 1e-2);
    }

    // ---- set_target ---------------------------------------------------

    #[test]
    fn set_target_updates_target_but_not_current_until_tick() {
        let mut p = fresh();
        let c0 = p.current_yaw_rad;
        p.set_target(ViewportPosition {
            yaw: 0.5,
            pitch: -0.2,
            fov_degrees: Some(60.0),
        });
        assert_eq!(p.current_yaw_rad, c0);
        assert_eq!(p.target_yaw_rad, 0.5);
        assert_eq!(p.target_fov_deg, 60.0);
    }
}
