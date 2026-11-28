//! Direct viewport input shared by CLI preview and GUI.
//!
//! Inputs update a render-space target. Call [`PoseControl::constrain`](crate::pose_control::PoseControl::constrain) before rendering to accept it through coverage. Successful acceptance discards blocked deltas, so releasing or reversing a drag has no residual motion. Automatic tracking and its timing belong to core.

use crate::ControlIntent;
use reco_core::core::{StitchCore, types::StitchCoreError};
use reco_core::geometry::RenderPose;

/// Invalid zoom bounds, sensitivity or resting pose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidControlConfig(&'static str);

impl std::fmt::Display for InvalidControlConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for InvalidControlConfig {}

/// Zoom limits, sensitivities, axis directions and render-space reset pose.
#[derive(Debug, Clone, Copy)]
pub struct PoseControlConfig {
    /// Nonnegative FOV degrees per wheel tick. Positive ticks zoom in; fractional ticks are supported.
    pub wheel_fov_step_degrees: f32,
    /// Minimum requested vertical FOV in degrees, within `[1, 179]`.
    pub fov_min_degrees: f32,
    /// Maximum requested vertical FOV in degrees, within `[fov_min_degrees, 179]`. Coverage may impose a lower limit without changing this setting.
    pub fov_max_degrees: f32,
    /// Reverse horizontal drag direction. False makes positive pixel deltas decrease yaw.
    pub invert_drag_x: bool,
    /// Reverse vertical drag direction. False makes positive pixel deltas increase pitch.
    pub invert_drag_y: bool,
    /// Nonnegative vertical FOV degrees per keyboard zoom command.
    pub hotkey_fov_step_degrees: f32,
    /// Finite render-space resting pose whose FOV is within the configured zoom limits. Recreate controls when the calibrated rest direction changes.
    pub rest_pose: RenderPose,
}

impl Default for PoseControlConfig {
    fn default() -> Self {
        Self {
            wheel_fov_step_degrees: 3.0,
            fov_min_degrees: 1.0,
            fov_max_degrees: 150.0,
            invert_drag_x: false,
            invert_drag_y: false,
            hotkey_fov_step_degrees: 5.0,
            rest_pose: RenderPose::default(),
        }
    }
}

/// Pending render-space input and the last accepted viewport pose.
///
/// Input methods ignore non-finite values and overflowing results without partial mutation. Successful coverage acceptance replaces both poses; failure preserves both for inspection or replacement.
#[derive(Debug, Clone)]
pub struct PoseControl {
    target: RenderPose,
    current: RenderPose,
    config: PoseControlConfig,
}

impl Default for PoseControl {
    fn default() -> Self {
        Self::new(PoseControlConfig::default()).expect("valid default pose controls")
    }
}

impl PoseControl {
    /// Validate configuration and initialize both poses to its resting pose.
    pub fn new(config: PoseControlConfig) -> Result<Self, InvalidControlConfig> {
        if !config.fov_min_degrees.is_finite()
            || !config.fov_max_degrees.is_finite()
            || !(1.0..=179.0).contains(&config.fov_min_degrees)
            || !(config.fov_min_degrees..=179.0).contains(&config.fov_max_degrees)
        {
            return Err(InvalidControlConfig(
                "zoom bounds must satisfy 1 <= min <= max <= 179",
            ));
        }
        if [
            config.wheel_fov_step_degrees,
            config.hotkey_fov_step_degrees,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(InvalidControlConfig(
                "zoom sensitivities must be finite and nonnegative",
            ));
        }
        if !finite_pose(config.rest_pose)
            || !(config.fov_min_degrees..=config.fov_max_degrees)
                .contains(&config.rest_pose.fov_degrees)
        {
            return Err(InvalidControlConfig(
                "rest pose must be finite with FOV within zoom bounds",
            ));
        }
        Ok(Self {
            target: config.rest_pose,
            current: config.rest_pose,
            config,
        })
    }

    /// Apply pixel deltas using accepted FOV and displayed image height, not source dimensions. Invalid dimensions or deltas leave the target unchanged.
    pub fn apply_drag(&mut self, dx_pixels: f32, dy_pixels: f32, viewport_height: f32) {
        if !viewport_height.is_finite()
            || viewport_height <= 0.0
            || !dx_pixels.is_finite()
            || !dy_pixels.is_finite()
        {
            return;
        }
        let scale = 2.0 * (self.current.fov_degrees.to_radians() * 0.5).tan() / viewport_height;
        let dx = if self.config.invert_drag_x {
            dx_pixels
        } else {
            -dx_pixels
        };
        let dy = if self.config.invert_drag_y {
            -dy_pixels
        } else {
            dy_pixels
        };
        self.set_target(RenderPose {
            yaw: self.target.yaw + dx * scale,
            pitch: self.target.pitch + dy * scale,
            ..self.target
        });
    }

    /// Apply wheel ticks; positive values narrow FOV.
    pub fn apply_wheel(&mut self, ticks: f32) {
        self.set_target_fov(self.target.fov_degrees - ticks * self.config.wheel_fov_step_degrees);
    }

    /// Apply a command to pending input, without advancing motion or rendering.
    pub fn apply_intent(&mut self, intent: ControlIntent) {
        match intent {
            ControlIntent::DeltaYawRad(delta) => self.set_target(RenderPose {
                yaw: self.target.yaw + delta,
                ..self.target
            }),
            ControlIntent::DeltaPitchRad(delta) => self.set_target(RenderPose {
                pitch: self.target.pitch + delta,
                ..self.target
            }),
            ControlIntent::SetFovDeg(fov) => self.set_target_fov(fov),
            ControlIntent::DeltaFovDeg(delta) => {
                self.set_target_fov(self.target.fov_degrees + delta)
            }
            ControlIntent::ZoomIn => {
                self.set_target_fov(self.target.fov_degrees - self.config.hotkey_fov_step_degrees)
            }
            ControlIntent::ZoomOut => {
                self.set_target_fov(self.target.fov_degrees + self.config.hotkey_fov_step_degrees)
            }
            ControlIntent::Reset => self.target = self.config.rest_pose,
        }
    }

    /// Replace pending input, clamping finite FOV to configured limits. Non-finite components reject the entire update.
    pub fn set_target(&mut self, mut pose: RenderPose) {
        if finite_pose(pose) {
            pose.fov_degrees = pose
                .fov_degrees
                .clamp(self.config.fov_min_degrees, self.config.fov_max_degrees);
            self.target = pose;
        }
    }

    /// Set a finite requested FOV in degrees, bounded by configured zoom limits.
    pub fn set_target_fov(&mut self, fov: f32) {
        self.set_target(RenderPose {
            fov_degrees: fov,
            ..self.target
        });
    }

    /// Fit pending input through coverage and accept the result. Errors preserve both poses. Call before rendering and after geometry changes; do not orient the result again.
    pub fn constrain(&mut self, core: &mut StitchCore) -> Result<(), StitchCoreError> {
        let accepted = core.constrain_render_pose(self.current, self.target)?;
        self.set_pose(accepted);
        Ok(())
    }

    /// Accept interactive input, including preview recording, allowing black edges while core prepares coverage. Offline export and automatic tracking require strict coverage resolution.
    pub fn constrain_preview(&mut self, core: &mut StitchCore) -> Result<(), StitchCoreError> {
        let accepted = core.constrain_preview_pose(self.current, self.target)?;
        self.set_pose(accepted);
        Ok(())
    }

    /// Replace current and target after a coordinate-frame change. Accepts finite poses with FOV in `[1, 179]`, including coverage results below the requested zoom minimum. Does not validate coverage or change the reset pose; constrain before rendering.
    pub fn set_pose(&mut self, accepted: RenderPose) {
        if finite_pose(accepted) && (1.0..=179.0).contains(&accepted.fov_degrees) {
            self.target = accepted;
            self.current = accepted;
        }
    }

    /// Pending render-space input, not yet accepted through coverage.
    pub fn target_pose(&self) -> RenderPose {
        self.target
    }
    /// Accepted render-space pose for rendering.
    pub fn current_pose(&self) -> RenderPose {
        self.current
    }
    /// Accepted vertical FOV in degrees.
    pub fn current_fov_deg(&self) -> f32 {
        self.current.fov_degrees
    }
}

fn finite_pose(pose: RenderPose) -> bool {
    pose.yaw.is_finite() && pose.pitch.is_finite() && pose.fov_degrees.is_finite()
}
