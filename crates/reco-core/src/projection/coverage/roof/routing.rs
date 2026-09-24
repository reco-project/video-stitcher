//! Paths through the same piecewise-linear support used to certify motion.

use super::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

impl Roof {
    pub(in super::super) fn route(
        &self,
        start: Pose,
        score: impl Fn(usize, Pose) -> [f64; 3],
    ) -> Vec<Pose> {
        if !self.contains(start) {
            return Vec::new();
        }
        let stride = self.grid.columns + 1;
        let minimum = f64::from(start.fov_degrees) + FOV_EPSILON;
        let supported = |i: usize| f64::from(self.heights[i]) >= minimum;
        let pose = |i| {
            self.grid.near_yaw(
                self.grid.pose(i % stride, i / stride, start.fov_degrees),
                start.yaw,
            )
        };
        let mut goal = None;
        let mut goal_score = [f64::INFINITY; 3];
        let mut unique = false;
        for i in (0..self.heights.len()).filter(|&i| supported(i)) {
            let cost = score(i, pose(i));
            if cost < goal_score {
                goal = Some(i);
                goal_score = cost;
                unique = true;
            } else if cost == goal_score {
                unique = false;
            }
        }
        let Some(goal) = goal else { return Vec::new() };
        if unique && self.segment(start, pose(goal)) {
            return vec![start, pose(goal)];
        }
        let dx = f64::from(self.grid.yaw[1] - self.grid.yaw[0]) / self.grid.columns as f64;
        let dy = f64::from(self.grid.pitch[1] - self.grid.pitch[0]) / self.grid.rows as f64;
        let [mut x, y] = self.coordinates(start).unwrap();
        if self.grid.wraps_yaw() {
            x = x.rem_euclid(self.grid.columns as f64);
        }
        let ix = (x.floor() as usize).min(self.grid.columns - 1);
        let iy = (y.floor() as usize).min(self.grid.rows - 1);
        let mut lengths = vec![f64::INFINITY; self.heights.len()];
        let mut parents = vec![usize::MAX; self.heights.len()];
        let mut pending = BinaryHeap::new();
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let i = (iy + dy) * stride + ix + dx;
            if supported(i) && self.segment(start, pose(i)) {
                let p = pose(i);
                let length = f64::from(p.yaw - start.yaw).hypot(f64::from(p.pitch - start.pitch));
                lengths[i] = length;
                parents[i] = i;
                pending.push(Reverse((length.to_bits(), i)));
            }
        }
        let steps = [[0.0, dy], [dx, dx.hypot(dy)]];
        let mut best = None;
        let mut best_score = [f64::INFINITY; 3];
        while let Some(Reverse((bits, i))) = pending.pop() {
            let length = f64::from_bits(bits);
            if length > lengths[i] {
                continue;
            }
            let cost = score(i, pose(i));
            if cost < best_score {
                best = Some(i);
                best_score = cost;
            }
            // Distance-first ordering selects the nearest equally ranked goal.
            // Once the global best score is reached, no search remains.
            if cost == goal_score {
                best = Some(i);
                break;
            }
            for (j, [dx, dy]) in self.neighbors(i) {
                let next = length + steps[dx][dy];
                // These are edges of the actual triangulation, so their heights are affine.
                if supported(j) && next < lengths[j] {
                    lengths[j] = next;
                    parents[j] = i;
                    pending.push(Reverse((next.to_bits(), j)));
                }
            }
        }
        // If the global goal is disconnected, exhaustion still finds the
        // best view in the start component, never a route to the other side.
        let Some(mut i) = best else {
            return Vec::new();
        };
        let mut path = Vec::new();
        loop {
            path.push(pose(i));
            if parents[i] == i {
                break;
            }
            i = parents[i];
        }
        path.push(start);
        path.reverse();
        for i in 1..path.len() {
            path[i] = self.grid.near_yaw(path[i], path[i - 1].yaw);
        }
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equally_ranked_goals_keep_the_nearest_reachable_destination() {
        let grid = Grid {
            yaw: [-1.0, 1.0],
            pitch: [-0.5, 0.5],
            columns: 10,
            rows: 10,
        };
        let roof = Roof::from_cells(grid, 10.0, &[80.0; 100]);
        let start = grid.pose(9, 7, 30.0);
        let path = roof.route(start, |_, p| {
            [
                0.0,
                0.0,
                if p.yaw.abs() > 0.5 && p.pitch == 0.0 {
                    0.0
                } else {
                    1.0
                },
            ]
        });
        let goal = path.last().unwrap();
        assert!(goal.yaw > 0.5);
        assert_eq!(goal.pitch, 0.0);
        assert!(path.windows(2).all(|p| roof.segment(p[0], p[1])));
    }

    #[test]
    fn routing_matches_exhaustive_reachability_on_fragmented_maps() {
        let grid = Grid {
            yaw: [-1.0, 1.0],
            pitch: [-0.5, 0.5],
            columns: 8,
            rows: 8,
        };
        let mut seed = 17_u64;
        for case in 0..200 {
            let cells: Vec<_> = (0..64)
                .map(|i| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if (i % 8 < 2 && i / 8 < 2) || seed >> 61 > 1 {
                        80.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let roof = Roof::from_cells(grid, 10.0, &cells);
            let start = grid.pose(1, 1, 30.0);
            let target = grid.pose(case % 9, case / 9 % 9, 30.0);
            let score = |_: usize, p: Pose| {
                [
                    0.0,
                    0.0,
                    f64::from(p.yaw - target.yaw).powi(2)
                        + f64::from(p.pitch - target.pitch).powi(2),
                ]
            };
            let pose = |i: usize| grid.pose(i % 9, i / 9, 30.0);
            let mut visited = [false; 81];
            let mut pending = vec![10];
            let mut best = [f64::INFINITY; 3];
            while let Some(i) = pending.pop() {
                if visited[i] {
                    continue;
                }
                visited[i] = true;
                let cost = score(i, pose(i));
                if cost < best {
                    best = cost;
                }
                for (dx, dy) in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (1, 1),
                    (-1, 1),
                    (1, -1),
                ] {
                    let (x, y) = ((i % 9) as isize + dx, (i / 9) as isize + dy);
                    if !(0..9).contains(&x) || !(0..9).contains(&y) {
                        continue;
                    }
                    let j = y as usize * 9 + x as usize;
                    if !visited[j] && roof.segment(pose(i), pose(j)) {
                        pending.push(j);
                    }
                }
            }
            let path = roof.route(start, score);
            assert_eq!(score(0, *path.last().unwrap()), best, "case {case}");
            assert!(path.windows(2).all(|p| roof.segment(p[0], p[1])));
        }
    }

    #[test]
    fn constant_fov_ceiling_matches_segment_certification_at_contacts() {
        let grid = Grid {
            yaw: [-1.0, 1.0],
            pitch: [-0.5, 0.5],
            columns: 16,
            rows: 8,
        };
        let cells: Vec<_> = (0..128).map(|i| 20.0 + ((i * 37) % 67) as f32).collect();
        let roof = Roof::from_cells(grid, 10.0, &cells);
        let mut seed = 7_u64;
        let mut random = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 40) as f32 / ((1_u32 << 24) - 1) as f32
        };
        for i in 0..2000 {
            let a = Pose {
                yaw: random() * 2.0 - 1.0,
                pitch: random() - 0.5,
                fov_degrees: 30.0,
            };
            let b = if i % 10 == 0 {
                a
            } else {
                Pose {
                    yaw: random() * 2.0 - 1.0,
                    pitch: random() - 0.5,
                    ..a
                }
            };
            let ceiling = roof.segment_ceiling(a, b).unwrap();
            let contact = (ceiling - FOV_EPSILON) as f32;
            for fov in [10.0, contact - 1e-4, contact, contact + 1e-4, 100.0] {
                let a = Pose {
                    fov_degrees: fov,
                    ..a
                };
                let b = Pose {
                    fov_degrees: fov,
                    ..b
                };
                assert_eq!(f64::from(fov) + FOV_EPSILON <= ceiling, roof.segment(a, b));
                assert_eq!(roof.segment(a, b), roof.segment(b, a));
            }
        }
    }

    #[test]
    fn route_uses_certified_detour_and_never_crosses_an_unsupported_cut() {
        let grid = Grid {
            yaw: [-1.0, 1.0],
            pitch: [-0.5, 0.5],
            columns: 10,
            rows: 10,
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
        let roof = Roof::from_cells(grid, 10.0, &cells);
        let from = Pose {
            yaw: -0.8,
            pitch: -0.3,
            fov_degrees: 30.0,
        };
        let to = Pose { yaw: 0.8, ..from };
        assert!(!roof.segment(from, to));
        let route = roof.route(from, |_, p| {
            [
                0.0,
                0.0,
                f64::from(p.yaw - to.yaw).powi(2) + f64::from(p.pitch - to.pitch).powi(2),
            ]
        });
        assert!(route.len() > 2);
        assert!(route.iter().any(|p| p.pitch >= 0.3));
        assert!(route.windows(2).all(|p| roof.segment(p[0], p[1])));
        assert!(super::super::super::motion::distance(*route.last().unwrap(), to) < 1e-6);
    }

    #[test]
    fn routing_never_selects_a_goal_in_another_component() {
        let grid = Grid {
            yaw: [-1.0, 1.0],
            pitch: [-0.5, 0.5],
            columns: 10,
            rows: 10,
        };
        let cells: Vec<_> = (0..100)
            .map(|i| {
                if (4..6).contains(&(i % 10)) {
                    0.0
                } else {
                    80.0
                }
            })
            .collect();
        let roof = Roof::from_cells(grid, 10.0, &cells);
        let from = Pose {
            yaw: -0.8,
            pitch: 0.0,
            fov_degrees: 30.0,
        };
        let route = roof.route(from, |_, p| {
            [
                0.0,
                0.0,
                f64::from(p.yaw - 0.8).powi(2) + f64::from(p.pitch).powi(2),
            ]
        });
        assert!(!route.is_empty());
        assert!(route.iter().all(|p| p.yaw < 0.0));
        assert!(route.windows(2).all(|p| roof.segment(p[0], p[1])));
    }
}
