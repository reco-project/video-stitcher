//! A stopping continuation at the achieved FOV, without future zoom input.

use super::*;

// Covers a full-rate stop at 1000 fps, including f32 rounding in each step.
const MAX_STEPS: usize = 2048;

pub(super) fn path(window: &MotionWindow, next: Pose) -> Option<Vec<Pose>> {
    let mut poses = vec![next];
    let mut pose = next;
    let mut velocity = window.velocity_at(next);
    // The velocity is bounded, so the stopping horizon is bounded as well.
    let steps = (velocity[0].hypot(velocity[1]) / (ACCELERATION * window.dt)).ceil();
    // The analytic count is only an estimate: rounded positions can decelerate
    // more slowly. Certify the actual continuation, with a finite work budget.
    if !steps.is_finite() || steps > MAX_STEPS as f32 {
        return None;
    }
    for _ in 0..MAX_STEPS {
        if velocity[0].hypot(velocity[1]) < 1e-8 {
            return Some(poses);
        }
        let step = MotionWindow {
            previous: pose,
            velocity,
            dt: window.dt,
        };
        let next = step.braking_step();
        if !step.admits(next) {
            return None;
        }
        velocity = step.velocity_at(next);
        poses.push(next);
        pose = next;
    }
    (velocity[0].hypot(velocity[1]) < 1e-8).then_some(poses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_rate_stops_remain_feasible_near_both_yaw_seams() {
        for fps in [300.0, 1000.0] {
            for yaw in [-3.0, -1.0, 0.0, 1.0, 3.0] {
                for i in 0..32 {
                    let angle = i as f32 * std::f32::consts::TAU / 32.0;
                    let window = MotionWindow {
                        previous: Pose {
                            yaw,
                            pitch: 0.4,
                            fov_degrees: 30.0,
                        },
                        velocity: [RATE * angle.cos(), RATE * angle.sin()],
                        dt: 1.0 / fps,
                    };
                    let next = window.braking_step();
                    assert!(window.admits(next), "fps={fps}, yaw={yaw}, direction={i}");
                    let poses = path(&window, next)
                        .unwrap_or_else(|| panic!("fps={fps}, yaw={yaw}, direction={i}"));
                    assert!(poses.len() <= MAX_STEPS + 1);
                    assert_eq!(poses.last(), poses.get(poses.len() - 2));
                }
            }
        }
    }
}
