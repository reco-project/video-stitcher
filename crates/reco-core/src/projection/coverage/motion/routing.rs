//! Cached guidance; feasibility and braking remain the motion solver's responsibility.

use super::*;

/// Bound to one immutable coverage map by its owning controller.
pub(in super::super) struct Route {
    key: Option<(Vector3<f32>, f32)>,
    path: Vec<Pose>,
    views: Vec<(Matrix4<f32>, Vector3<f64>)>,
}

impl Route {
    pub(in super::super) fn new(cal: &Calibration, roof: &roof::Roof) -> Self {
        let views = roof
            .vertices()
            .map(|p| {
                let view = view(cal, p);
                let forward = view_direction(&view).cast();
                (view, forward)
            })
            .collect();
        Self {
            key: None,
            path: Vec::new(),
            views,
        }
    }

    pub(super) fn guide(
        &mut self,
        cal: &Calibration,
        size: &ViewportSize,
        current: Pose,
        target: &Vector3<f32>,
        roof: &roof::Roof,
    ) -> Option<Pose> {
        let key = (*target, current.fov_degrees);
        let scale = screen_scale(size, current.fov_degrees);
        if self.key != Some(key) {
            let at_fov = |p: Pose| Pose {
                fov_degrees: current.fov_degrees,
                ..p
            };
            // A changed request does not invalidate a route that still delivers
            // a visible target. Recheck its remaining geometry when widening.
            let reusable = self.path.last().is_some_and(|&p| {
                let p = at_fov(p);
                let cost = score(cal, size, p, target);
                cost[0] == 0.0 && cost[1] == 0.0 && roof.contains(p)
            }) && self.key.is_some_and(|(_, fov)| {
                fov >= current.fov_degrees
                    || self
                        .path
                        .windows(2)
                        .all(|p| roof.segment(at_fov(p[0]), at_fov(p[1])))
            });
            if reusable {
                for p in &mut self.path {
                    *p = at_fov(*p);
                }
            } else {
                self.path = roof.route(current, |i, _| self.score(i, scale, target));
            }
            self.key = Some(key);
        }
        let goal = *self.path.last()?;
        if !better(
            score(cal, size, goal, target),
            score(cal, size, current, target),
        ) {
            return None;
        }
        // The endpoint stays the endpoint even below motion resolution; never
        // replace it with an earlier waypoint after arriving.
        if roof.segment(current, goal) {
            self.path.clear();
            self.path.push(goal);
            return Some(goal);
        }
        let attachment = |path: &[Pose]| {
            path.iter()
                .enumerate()
                .rev()
                .find(|(_, p)| {
                    distance(current, **p) > CONTACT_TOLERANCE && roof.segment(current, **p)
                })
                .map(|(i, _)| i)
        };
        let index = if let Some(index) = attachment(&self.path) {
            index
        } else {
            // Widening or inertial movement can disconnect the current pose
            // from a cached shortcut even though its remaining edges are valid.
            self.path = roof.route(current, |i, _| self.score(i, scale, target));
            attachment(&self.path)?
        };
        self.path.drain(..index);
        self.path.first().copied()
    }

    pub(super) fn active(&self, target: &Vector3<f32>, fov: f32) -> bool {
        self.key == Some((*target, fov)) && !self.path.is_empty()
    }

    pub(super) fn score(&self, i: usize, scale: [f64; 2], target: &Vector3<f32>) -> [f64; 3] {
        let (matrix, forward) = &self.views[i];
        score_view(matrix, scale, target, || *forward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_scores_match_live_views_including_antipodal_targets() {
        let cylinder = Calibration::new(
            vec![Lens::flat(3840, 2000)],
            Cylinder::default(),
            Framing {
                axis_offset: 0.0,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        for mut cal in [stereo(), cylinder] {
            for (tilt, roll) in [(0.0_f64, 0.0_f64), (0.3, -0.4), (-0.25, 0.35)] {
                cal.framing.tilt = tilt;
                cal.framing.roll = roll;
                let roof = roof::Roof::from_cells(
                    roof::Grid {
                        yaw: [-1.5, 1.5],
                        pitch: [-0.8, 0.6],
                        columns: 16,
                        rows: 12,
                    },
                    10.0,
                    &[100.0; 192],
                );
                let route = Route::new(&cal, &roof);
                for size in [
                    ViewportSize {
                        width: 1920,
                        height: 1080,
                    },
                    ViewportSize {
                        width: 720,
                        height: 1280,
                    },
                ] {
                    for fov in [10.0, 51.98, 90.0] {
                        for (i, p) in roof.vertices().enumerate() {
                            let p = Pose {
                                fov_degrees: fov,
                                ..p
                            };
                            let forward = direction(&cal, p);
                            for target in [
                                forward,
                                -forward,
                                (-forward + Vector3::new(1e-7, -1e-7, 0.0)).normalize(),
                                Vector3::x(),
                                -Vector3::z(),
                            ] {
                                assert_eq!(
                                    route.score(i, screen_scale(&size, fov), &target),
                                    score(&cal, &size, p, &target)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cached_initial_pose_matches_exhaustive_live_scoring() {
        let mut cal = stereo();
        cal.framing.tilt = 0.3;
        cal.framing.roll = -0.4;
        let cells: Vec<_> = (0..143).map(|i| 30.0 + ((i * 37) % 53) as f32).collect();
        let roof = roof::Roof::from_cells(
            roof::Grid {
                yaw: [-1.3, 1.3],
                pitch: [-0.7, 0.7],
                columns: 13,
                rows: 11,
            },
            10.0,
            &cells,
        );
        let route = Route::new(&cal, &roof);
        let size = ViewportSize {
            width: 1920,
            height: 1080,
        };
        for i in 0..90 {
            let target = direction(
                &cal,
                Pose {
                    yaw: (i as f32 * 0.29).sin() * 3.0,
                    pitch: (i as f32 * 0.61).cos(),
                    fov_degrees: 30.0,
                },
            );
            for fov in [20.0, 51.98, 90.0] {
                let wanted = aim(&cal, &target, fov);
                let expected = if roof.contains(wanted) {
                    Some(wanted)
                } else {
                    roof.supported_vertices(fov).map(|(_, p)| p).min_by(|a, b| {
                        let a = score(&cal, &size, *a, &target);
                        let b = score(&cal, &size, *b, &target);
                        a[0].total_cmp(&b[0])
                            .then_with(|| a[1].total_cmp(&b[1]))
                            .then_with(|| a[2].total_cmp(&b[2]))
                    })
                };
                assert_eq!(initial(&cal, &size, &target, fov, &roof, &route), expected);
            }
        }
    }

    #[test]
    fn widening_rebuilds_a_cached_shortcut_when_its_attachment_is_lost() {
        let cal = stereo();
        let size = ViewportSize {
            width: 640,
            height: 360,
        };
        let cells: Vec<_> = (0..100)
            .map(|i| {
                if (4..6).contains(&(i % 10)) && i / 10 < 7 {
                    50.0
                } else {
                    80.0
                }
            })
            .collect();
        let roof = roof::Roof::from_cells(
            roof::Grid {
                yaw: [-1.0, 1.0],
                pitch: [-0.5, 0.5],
                columns: 10,
                rows: 10,
            },
            10.0,
            &cells,
        );
        let start = Pose {
            yaw: -0.8,
            pitch: -0.3,
            fov_degrees: 30.0,
        };
        let goal = Pose { yaw: 0.8, ..start };
        let target = direction(&cal, goal);
        let mut route = Route::new(&cal, &roof);
        let narrow = route.guide(&cal, &size, start, &target, &roof).unwrap();
        assert!(distance(narrow, goal) < 1e-6);
        assert_eq!(route.path.len(), 1);
        let start = Pose {
            fov_degrees: 60.0,
            ..start
        };
        let guide = route.guide(&cal, &size, start, &target, &roof).unwrap();
        assert!(route.path.len() > 1);
        assert!(roof.segment(start, guide));
        assert!(!roof.segment(
            start,
            Pose {
                fov_degrees: 60.0,
                ..goal
            }
        ));
    }

    #[test]
    fn detour_tracking_reverses_without_zooming_or_relaxing_motion_bounds() {
        let cal = stereo();
        let size = ViewportSize {
            width: 640,
            height: 360,
        };
        let cells: Vec<_> = (0..100)
            .map(|i| {
                if (4..6).contains(&(i % 10)) && i / 10 < 7 {
                    0.0
                } else {
                    80.0
                }
            })
            .collect();
        let roof = roof::Roof::from_cells(
            roof::Grid {
                yaw: [-1.0, 1.0],
                pitch: [-0.5, 0.5],
                columns: 10,
                rows: 10,
            },
            10.0,
            &cells,
        );
        for fps in [30, 60] {
            let dt = 1.0 / fps as f32;
            let mut route = Route::new(&cal, &roof);
            let mut pose = Pose {
                yaw: -0.8,
                pitch: -0.3,
                fov_degrees: 30.0,
            };
            let mut velocity = [0.0; 2];
            for yaw in [0.8, -0.8] {
                let target = direction(
                    &cal,
                    Pose {
                        yaw,
                        pitch: -0.3,
                        ..pose
                    },
                );
                let mut tail_travel = 0.0;
                for frame in 0..fps * 30 {
                    let window = MotionWindow {
                        previous: pose,
                        velocity,
                        dt,
                    };
                    let next = solve_motion(&cal, &size, window, &target, 30.0, &roof, &mut route)
                        .unwrap();
                    assert_eq!(next.pose.fov_degrees, 30.0);
                    assert!(roof.segment(pose, next.pose));
                    assert!(next.velocity[0].hypot(next.velocity[1]) <= RATE + 1e-5);
                    assert!(
                        (next.velocity[0] - velocity[0]).hypot(next.velocity[1] - velocity[1]) / dt
                            <= ACCELERATION + 0.001
                    );
                    let mut stopping = braking::path(&window, next.pose).unwrap();
                    stopping.insert(0, pose);
                    assert!(stopping.windows(2).all(|p| roof.segment(p[0], p[1])));
                    if frame >= fps * 25 {
                        assert!(
                            screen(&cal, &size, next.pose, &target)
                                .is_some_and(|xy| xy.iter().all(|x| x.abs() <= 1.0))
                        );
                        tail_travel += distance(pose, next.pose);
                    }
                    pose = next.pose;
                    velocity = next.velocity;
                }
                assert!(
                    tail_travel < 0.001,
                    "fps={fps} yaw={yaw} tail={tail_travel} pose={pose:?}"
                );
            }
        }
    }
}
