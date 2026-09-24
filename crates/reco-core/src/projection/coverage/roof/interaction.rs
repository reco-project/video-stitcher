//! Direct manipulation of the same piecewise-affine coverage used by motion.

use super::*;
use nalgebra::Vector2;

impl Roof {
    /// Fit direct movement to supported segments, without velocity, routing or input debt.
    pub(in super::super) fn constrain(&self, from: Pose, requested: Pose) -> Pose {
        let requested = self.grid.near_yaw(requested, from.yaw);
        if self.segment(from, requested) {
            return requested;
        }
        // Retain a supported prefix even if the requested zoom cannot fit anywhere.
        let (mut lo, mut hi) = (0.0, 1.0);
        let mut best = from;
        for _ in 0..24 {
            let t = (lo + hi) * 0.5;
            let p = interpolate(from, requested, t);
            if self.segment(from, p) {
                lo = t;
                best = p;
            } else {
                hi = t;
            }
        }
        let point = |p: Pose| Vector2::new(f64::from(p.yaw), f64::from(p.pitch));
        let wanted = point(requested);
        let origin = point(from);
        let radius2 = (wanted - origin).norm_squared();
        let cost = |p: Pose| {
            [
                f64::from((p.fov_degrees - requested.fov_degrees).abs()),
                (point(p) - wanted).norm_squared(),
            ]
        };
        let mut best_cost = cost(best);
        let height = f64::from(requested.fov_degrees) + FOV_EPSILON;
        let stride = self.grid.columns + 1;
        // Once the requested zoom fits, no point farther away than the
        // supported prefix can win. Restrict the search to that bounding box;
        // a larger angular chart must not make a small drag more expensive.
        let cells = |axis: [f32; 2], count: usize, coordinate: f64, best_cost: [f64; 2]| {
            if best_cost[0] != 0.0 {
                return 0..count;
            }
            let radius = best_cost[1].sqrt();
            let scale = count as f64 / (f64::from(axis[1]) - f64::from(axis[0]));
            let first = ((coordinate - radius - f64::from(axis[0])) * scale).floor() - 1.0;
            let last = ((coordinate + radius - f64::from(axis[0])) * scale).ceil() + 1.0;
            (first.max(0.0) as usize).min(count)..(last.max(0.0) as usize).min(count)
        };
        let nearest = |axis: [f32; 2], count: usize, coordinate: f64| {
            (((coordinate - f64::from(axis[0])) / (f64::from(axis[1]) - f64::from(axis[0]))
                * count as f64)
                .floor()
                .max(0.0) as usize)
                .min(count - 1)
        };
        let column = nearest(
            self.grid.yaw,
            self.grid.columns,
            f64::from(self.grid.near_yaw(requested, 0.0).yaw),
        );
        let dx =
            (f64::from(self.grid.yaw[1]) - f64::from(self.grid.yaw[0])) / self.grid.columns as f64;
        let row = nearest(self.grid.pitch, self.grid.rows, wanted.y);
        // Start near the request so a fitting zoom tightens the search radius
        // before distant cells are considered. Row bounds then skip whole strips.
        for y in near_first(row, self.grid.rows, false) {
            if !cells(self.grid.pitch, self.grid.rows, wanted.y, best_cost).contains(&y) {
                continue;
            }
            for (order, x) in
                near_first(column, self.grid.columns, self.grid.wraps_yaw()).enumerate()
            {
                if best_cost[0] == 0.0
                    && order > 2 * ((best_cost[1].sqrt() / dx).ceil() as usize + 2)
                {
                    break;
                }
                if !self.grid.wraps_yaw()
                    && !cells(self.grid.yaw, self.grid.columns, wanted.x, best_cost).contains(&x)
                {
                    continue;
                }
                let base = self.grid.pose(x, y, requested.fov_degrees);
                let shift = self.grid.near_yaw(base, requested.yaw).yaw - base.yaw;
                let cell_pose = |dx, dy| {
                    let mut p = self.grid.pose(x + dx, y + dy, requested.fov_degrees);
                    p.yaw += shift;
                    p
                };
                let low = point(cell_pose(0, 0));
                let high = point(cell_pose(1, 1));
                let near = wanted.zip_map(&low, f64::max).zip_map(&high, f64::min);
                if best_cost[0] == 0.0 && (near - wanted).norm_squared() >= best_cost[1] {
                    continue;
                }
                for offsets in self.triangles(x, y) {
                    let vertices = offsets.map(|(dx, dy)| {
                        (
                            point(cell_pose(dx, dy)),
                            f64::from(self.heights[(y + dy) * stride + x + dx]),
                        )
                    });
                    // Intersect the triangle with its affine FOV half-plane.
                    let mut polygon = [Vector2::zeros(); 4];
                    let mut count = 0;
                    for i in 0..3 {
                        let (a, ha) = vertices[i];
                        let (b, hb) = vertices[(i + 1) % 3];
                        if ha >= height {
                            polygon[count] = a;
                            count += 1;
                        }
                        if (ha >= height) != (hb >= height) {
                            polygon[count] = a + (b - a) * ((height - ha) / (hb - ha));
                            count += 1;
                        }
                    }
                    if count == 0 {
                        continue;
                    }
                    let center =
                        polygon[..count].iter().copied().sum::<Vector2<f64>>() / count as f64;
                    for i in 0..count {
                        let a = polygon[i];
                        let edge = polygon[(i + 1) % count] - a;
                        let t = if edge.norm_squared() > 0.0 {
                            ((wanted - a).dot(&edge) / edge.norm_squared()).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let mut q = a + edge * t;
                        // Keep the f32 pose on the supported side after rounding.
                        q += (center - q) * 1e-4;
                        let p = Pose {
                            yaw: q.x as f32,
                            pitch: q.y as f32,
                            fov_degrees: requested.fov_degrees,
                        };
                        let candidate_cost = cost(p);
                        if candidate_cost < best_cost
                            && (from.fov_degrees != requested.fov_degrees
                                || (point(p) - origin).norm_squared() <= radius2 + 1e-12)
                            && self.segment(from, p)
                        {
                            best = p;
                            best_cost = candidate_cost;
                        }
                    }
                }
            }
        }
        best
    }
}

fn near_first(center: usize, count: usize, wrap: bool) -> impl Iterator<Item = usize> {
    (0..if wrap { count } else { count * 2 }).filter_map(move |i| {
        let offset = if i % 2 == 0 {
            -((i / 2) as isize)
        } else {
            (i / 2 + 1) as isize
        };
        if wrap {
            Some((center as isize + offset).rem_euclid(count as isize) as usize)
        } else {
            center.checked_add_signed(offset).filter(|&j| j < count)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat() -> Roof {
        Roof::from_cells(
            Grid {
                yaw: [-1.0, 1.0],
                pitch: [-1.0, 1.0],
                columns: 10,
                rows: 10,
            },
            10.0,
            &[80.0; 100],
        )
    }

    fn pose(yaw: f32, pitch: f32, fov_degrees: f32) -> Pose {
        Pose {
            yaw,
            pitch,
            fov_degrees,
        }
    }

    #[test]
    fn contact_slides_and_release_and_reversal_have_no_input_debt() {
        let roof = flat();
        let mut p = pose(0.0, 0.0, 30.0);
        assert_eq!(
            roof.constrain(p, pose(0.4, 0.1, 30.0)),
            pose(0.4, 0.1, 30.0)
        );
        for _ in 0..100 {
            let next = roof.constrain(p, pose(p.yaw + 0.1, p.pitch + 0.005, 30.0));
            assert!(roof.segment(p, next));
            p = next;
        }
        assert!((p.yaw - 1.0).abs() < 1e-4, "{p:?}");
        assert!((p.pitch - 0.5).abs() < 1e-3, "{p:?}");
        for _ in 0..100 {
            assert_eq!(roof.constrain(p, p), p);
        }
        let reverse = pose(p.yaw - 0.02, p.pitch, 30.0);
        assert_eq!(roof.constrain(p, reverse), reverse);
    }

    #[test]
    fn diagonal_contact_is_not_axis_aligned_or_grid_quantized() {
        let mut roof = flat();
        roof.heights = roof
            .vertices()
            .map(|p| 50.0 + 20.0 * p.pitch - 10.0 * p.yaw)
            .collect();
        let from = pose(-0.3, 0.1, 50.0);
        let result = roof.constrain(from, pose(0.5, -0.2, 50.0));
        assert!(roof.segment(from, result));
        assert!((result.yaw - 0.32).abs() < 1e-4, "{result:?}");
        assert!((result.pitch - 0.16).abs() < 1e-4, "{result:?}");
    }

    #[test]
    fn cannot_cross_an_unsupported_valley_in_either_direction() {
        let roof = Roof::from_cells(
            Grid {
                columns: 5,
                rows: 1,
                ..flat().grid
            },
            10.0,
            &[80.0, 80.0, 20.0, 80.0, 80.0],
        );
        for sign in [-1.0, 1.0] {
            let from = pose(sign * 0.9, 0.0, 50.0);
            let result = roof.constrain(from, pose(-from.yaw, 0.1, 50.0));
            assert!(roof.segment(from, result));
            assert!(result.yaw * sign > 0.2, "{result:?}");
        }
    }

    #[test]
    fn zoom_moves_inward_to_fit_without_crossing_the_roof() {
        let mut roof = flat();
        roof.heights = roof.vertices().map(|p| 50.0 - 30.0 * p.yaw).collect();
        let from = pose(0.8, 0.2, 20.0);
        let result = roof.constrain(from, pose(0.8, 0.2, 60.0));
        assert!(roof.segment(from, result));
        assert_eq!(result.fov_degrees, 60.0);
        assert!((result.yaw + 1.0 / 3.0).abs() < 1e-4, "{result:?}");
        assert!((result.pitch - from.pitch).abs() < 1e-4);
    }

    #[test]
    fn long_drag_stays_supported_and_never_exceeds_input_distance() {
        let mut roof = flat();
        roof.heights = roof
            .vertices()
            .map(|p| 45.0 + 20.0 * (3.0 * p.yaw).cos() + 15.0 * p.pitch)
            .collect();
        let mut p = pose(0.0, 0.0, 50.0);
        for i in 0..900 {
            let angle = i as f32 * 0.037;
            let request = pose(
                p.yaw + angle.cos() * 0.08,
                p.pitch + angle.sin() * 0.08,
                p.fov_degrees,
            );
            let q = roof.constrain(p, request);
            assert!(roof.segment(p, q), "{i}: {p:?} -> {q:?}");
            assert!(motion::distance(p, q) <= motion::distance(p, request) + 1e-6);
            assert_eq!(roof.constrain(q, q), q);
            p = q;
        }
    }
}
