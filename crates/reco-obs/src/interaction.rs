//! Pending UI input, transferred to the video callback without holding a lock during coverage.

use reco_control::pose_control::{PoseControl, PoseControlConfig};
use reco_core::geometry::RenderPose;

pub(super) struct Interaction {
    pub pose: PoseControl,
    pointer: Option<(i32, i32)>,
    property_angles_degrees: Option<[f32; 2]>,
}

impl Default for Interaction {
    fn default() -> Self {
        Self {
            pose: PoseControl::new(PoseControlConfig {
                invert_drag_x: true,
                ..Default::default()
            })
            .expect("valid OBS pose controls"),
            pointer: None,
            property_angles_degrees: None,
        }
    }
}

impl Interaction {
    pub fn apply_property_angles(&mut self, yaw_degrees: f32, pitch_degrees: f32) {
        if !yaw_degrees.is_finite() || !pitch_degrees.is_finite() {
            return;
        }
        let angles = [yaw_degrees, pitch_degrees];
        let mut target = self.pose.target_pose();
        // OBS supplies all settings on every update, including unchanged sliders.
        if self
            .property_angles_degrees
            .is_none_or(|old| old[0] != angles[0])
        {
            target.yaw = yaw_degrees.to_radians();
        }
        if self
            .property_angles_degrees
            .is_none_or(|old| old[1] != angles[1])
        {
            target.pitch = pitch_degrees.to_radians();
        }
        self.pose.set_target(target);
        self.property_angles_degrees = Some(angles);
    }

    pub fn click(&mut self, x: i32, y: i32, released: bool) {
        self.pointer = (!released).then_some((x, y));
    }

    pub fn move_pointer(&mut self, x: i32, y: i32, leave: bool, height: u32) {
        if leave {
            self.pointer = None;
        } else if let Some((previous_x, previous_y)) = self.pointer {
            self.pointer = Some((x, y));
            self.pose.apply_drag(
                x as f32 - previous_x as f32,
                y as f32 - previous_y as f32,
                height as f32,
            );
        }
    }

    pub fn take_request(&mut self) -> PoseControl {
        let request = self.pose.clone();
        self.pose.set_target(self.pose.current_pose());
        request
    }

    pub fn finish_request(&mut self, request: &PoseControl, accepted: bool) {
        let current = self.pose.current_pose();
        let pending = self.pose.target_pose();
        let base = if accepted {
            request.current_pose()
        } else {
            request.target_pose()
        };
        self.pose.set_pose(request.current_pose());
        self.pose.set_target(RenderPose {
            yaw: base.yaw + (pending.yaw - current.yaw),
            pitch: base.pitch + (pending.pitch - current.pitch),
            fov_degrees: base.fov_degrees + (pending.fov_degrees - current.fov_degrees),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_angles_initialize_once_and_invalid_updates_do_not_poison_them() {
        let mut input = Interaction::default();
        input.apply_property_angles(30.0, -20.0);
        assert_eq!(input.pose.target_pose().yaw, 30_f32.to_radians());
        assert_eq!(input.pose.target_pose().pitch, -20_f32.to_radians());
        let requested = RenderPose {
            yaw: 0.7,
            pitch: 0.3,
            fov_degrees: 42.0,
        };
        input.pose.set_target(requested);
        input.apply_property_angles(30.0, -20.0);
        assert_eq!(input.pose.target_pose(), requested);
        for angles in [
            [f32::NAN, 10.0],
            [10.0, f32::NAN],
            [f32::INFINITY, 10.0],
            [10.0, f32::NEG_INFINITY],
        ] {
            input.apply_property_angles(angles[0], angles[1]);
            assert_eq!(input.pose.target_pose(), requested);
        }
        input.apply_property_angles(30.0, 0.0);
        assert_eq!(
            input.pose.target_pose(),
            RenderPose {
                pitch: 0.0,
                ..requested
            }
        );
        input.apply_property_angles(0.0, 0.0);
        assert_eq!(
            input.pose.target_pose(),
            RenderPose {
                yaw: 0.0,
                pitch: 0.0,
                ..requested
            }
        );
    }

    #[test]
    fn coverage_discards_blocked_motion_but_keeps_new_input() {
        let mut input = Interaction::default();
        input.pose.set_target(RenderPose {
            yaw: 1.0,
            pitch: 0.4,
            fov_degrees: 90.0,
        });
        let mut request = input.take_request();
        input.pose.set_target(RenderPose {
            yaw: 0.1,
            pitch: -0.2,
            fov_degrees: 55.0,
        });
        let delta_fov = input.pose.target_pose().fov_degrees - input.pose.current_fov_deg();
        request.set_pose(RenderPose {
            yaw: 0.5,
            pitch: 0.3,
            fov_degrees: 51.0,
        });
        input.finish_request(&request, true);
        assert!((input.pose.target_pose().yaw - 0.6).abs() < 1e-6);
        assert!((input.pose.target_pose().pitch - 0.1).abs() < 1e-6);
        assert_eq!(input.pose.target_pose().fov_degrees, 51.0 + delta_fov);
        assert_eq!(input.pose.current_pose(), request.current_pose());
        let mut next = input.take_request();
        next.set_pose(input.pose.current_pose());
        input.finish_request(&next, true);
        assert_eq!(input.pose.target_pose(), input.pose.current_pose());
    }

    #[test]
    fn failed_coverage_restores_request_without_losing_new_input() {
        let mut input = Interaction::default();
        input.pose.set_target(RenderPose {
            yaw: 0.5,
            ..input.pose.target_pose()
        });
        let request = input.take_request();
        input.pose.set_target(RenderPose {
            yaw: 0.2,
            ..input.pose.target_pose()
        });
        input.finish_request(&request, false);
        assert!((input.pose.target_pose().yaw - 0.7).abs() < 1e-6);
        assert_eq!(input.pose.current_pose(), request.current_pose());
    }

    #[test]
    fn release_leave_and_extreme_coordinates_do_not_leave_a_drag() {
        let mut input = Interaction::default();
        input.move_pointer(20, 20, false, 100);
        assert_eq!(input.pose.current_pose(), input.pose.target_pose());
        input.click(i32::MIN, i32::MIN, false);
        input.move_pointer(i32::MAX, i32::MAX, false, 100);
        assert!(input.pose.target_pose().yaw.is_finite());
        input.move_pointer(0, 0, true, 100);
        let pose = input.pose.target_pose();
        input.move_pointer(100, 100, false, 100);
        assert_eq!(input.pose.target_pose(), pose);
        input.click(100, 100, false);
        input.click(100, 100, true);
        input.move_pointer(200, 200, false, 100);
        assert_eq!(input.pose.target_pose(), pose);
    }
}
