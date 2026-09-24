//! Budgeted recovery of map bottlenecks, certified against the projection.

use super::*;

impl Roof {
    pub(super) fn refine_bottlenecks(
        &mut self,
        cal: &Calibration,
        size: &ViewportSize,
        mut budget: usize,
        original_cells: Option<&[f32]>,
    ) {
        let stride = self.grid.columns + 1;
        while budget > 0 {
            let (capacity, parents) = self.connections();
            let Some(i) = (0..self.heights.len())
                .filter(|&i| {
                    f64::from(self.heights[i]) >= f64::from(self.minimum_fov) + FOV_EPSILON
                        && self.heights[i] > capacity[i]
                })
                .min_by(|&a, &b| capacity[a].total_cmp(&capacity[b]))
            else {
                break;
            };
            let saddle = capacity[i];
            // Recover the whole separated peak, not successively higher slivers
            // of a meaningful branch. A genuine bottleneck cannot pass its proof.
            let mut seen = vec![false; self.heights.len()];
            let mut queue = vec![i];
            seen[i] = true;
            let mut peak = i;
            while let Some(j) = queue.pop() {
                if self.heights[j] > self.heights[peak] {
                    peak = j;
                }
                for (k, _) in self.neighbors(j) {
                    if !seen[k] && self.heights[k] > saddle {
                        seen[k] = true;
                        queue.push(k);
                    }
                }
            }
            let height = self.heights[peak];
            let mut changes = Vec::new();
            let mut j = peak;
            while parents[j] != j {
                if self.heights[j] < height {
                    changes.push(j);
                    let offset = self.grid.seam_offset(j % stride);
                    if offset != 0 {
                        changes.push(j.checked_add_signed(offset).unwrap());
                    }
                }
                j = parents[j];
            }
            changes.sort_unstable();
            changes.dedup();
            let changes: Vec<_> = changes.into_iter().map(|i| (i, self.heights[i])).collect();
            let mut cells = Vec::new();
            for &(i, _) in &changes {
                for (dx, dy) in [(-1, -1), (0, -1), (-1, 0), (0, 0)] {
                    let Some(x) = (i % stride)
                        .checked_add_signed(dx)
                        .filter(|&x| x < self.grid.columns)
                    else {
                        continue;
                    };
                    let Some(y) = (i / stride)
                        .checked_add_signed(dy)
                        .filter(|&y| y < self.grid.rows)
                    else {
                        continue;
                    };
                    cells.push((x, y));
                }
            }
            cells.sort_unstable();
            cells.dedup();
            let cells: Vec<_> = cells
                .into_iter()
                .map(|(x, y)| (x, y, self.main_diagonal(x, y)))
                .collect();
            for &(i, _) in &changes {
                self.heights[i] = height;
            }
            // A lift can change a cell's diagonal. Certify both resulting triangles
            // in every incident cell, including the other side of a periodic seam.
            let supported = cells.iter().all(|&(x, y, old_split)| {
                self.triangles(x, y).into_iter().all(|triangle| {
                    if self.main_diagonal(x, y) == old_split
                        && triangle.iter().all(|&(dx, dy)| {
                            changes
                                .binary_search_by_key(&((y + dy) * stride + x + dx), |&(i, _)| i)
                                .is_err()
                        })
                    {
                        return true;
                    }
                    let poses = triangle.map(|(dx, dy)| {
                        [
                            f64::from(self.grid.yaw[0])
                                + (f64::from(self.grid.yaw[1]) - f64::from(self.grid.yaw[0]))
                                    * (x + dx) as f64
                                    / self.grid.columns as f64,
                            f64::from(self.grid.pitch[0])
                                + (f64::from(self.grid.pitch[1]) - f64::from(self.grid.pitch[0]))
                                    * (y + dy) as f64
                                    / self.grid.rows as f64,
                            f64::from(self.heights[(y + dy) * stride + x + dx]),
                        ]
                    });
                    let original_ceiling = original_cells
                        .map(|cells| cells[y * self.grid.columns + x])
                        .unwrap_or(0.0);
                    self.certify_triangle(cal, size, poses, original_ceiling, &mut budget)
                })
            });
            if !supported {
                for (i, height) in changes {
                    self.heights[i] = height;
                }
                break;
            }
        }
    }

    fn certify_triangle(
        &self,
        cal: &Calibration,
        size: &ViewportSize,
        vertices: [[f64; 3]; 3],
        original_ceiling: f32,
        budget: &mut usize,
    ) -> bool {
        // The cell was certified in full during construction. Only its newly
        // admitted portion needs another projection proof.
        if vertices.iter().all(|p| p[2] <= f64::from(original_ceiling)) {
            return true;
        }
        // Only the portion at or above the admitted FOV floor represents views.
        // An uncertified corner below that floor must not veto a valid repair.
        let floor = f64::from(self.minimum_fov);
        if vertices.iter().any(|p| p[2] < floor) {
            let mut polygon = Vec::with_capacity(4);
            for i in 0..3 {
                let a = vertices[i];
                let b = vertices[(i + 1) % 3];
                if a[2] >= floor {
                    polygon.push(a);
                }
                if (a[2] >= floor) != (b[2] >= floor) {
                    let t = (floor - a[2]) / (b[2] - a[2]);
                    polygon.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), floor]);
                }
            }
            return (1..polygon.len().saturating_sub(1)).all(|i| {
                self.certify_triangle(
                    cal,
                    size,
                    [polygon[0], polygon[i], polygon[i + 1]],
                    original_ceiling,
                    budget,
                )
            });
        }
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        // Enclose each affine patch in yaw, pitch and FOV. Subdivision tightens
        // the enclosure, never samples away a gap. Round the f64 bounds outward.
        let lo = Pose {
            yaw: (vertices.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min) as f32).next_down(),
            pitch: (vertices.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min) as f32).next_down(),
            fov_degrees: self.minimum_fov,
        };
        let hi = Pose {
            yaw: (vertices
                .iter()
                .map(|p| p[0])
                .fold(f64::NEG_INFINITY, f64::max) as f32)
                .next_up(),
            pitch: (vertices
                .iter()
                .map(|p| p[1])
                .fold(f64::NEG_INFINITY, f64::max) as f32)
                .next_up(),
            fov_degrees: (vertices.iter().map(|p| p[2]).fold(0.0, f64::max) as f32).next_up(),
        };
        if boundary::exterior_enclosure(cal, size, lo, hi, 2048) == Certificate::Covered {
            return true;
        }
        let [a, b, c] = vertices;
        let midpoint = |a: [f64; 3], b: [f64; 3]| std::array::from_fn(|i| (a[i] + b[i]) * 0.5);
        let ab = midpoint(a, b);
        let bc = midpoint(b, c);
        let ca = midpoint(c, a);
        [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
            .into_iter()
            .all(|t| self.certify_triangle(cal, size, t, original_ceiling, budget))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cylinder() -> Calibration {
        Calibration::new(
            vec![Lens::flat(3840, 2000)],
            Cylinder {
                focal_length: 1000.0,
                ..Default::default()
            },
            Framing {
                axis_offset: 0.0,
                tilt: 0.0,
                roll: 0.0,
            },
        )
    }

    fn valley() -> Roof {
        Roof::from_cells(
            Grid {
                yaw: [-0.1, 0.1],
                pitch: [-0.05, 0.05],
                columns: 5,
                rows: 1,
            },
            10.0,
            &[40.0, 40.0, 20.0, 30.0, 30.0],
        )
    }

    #[test]
    fn an_existing_cell_certificate_needs_no_duplicate_projection_proof() {
        let cal = cylinder();
        let size = ViewportSize {
            width: 160,
            height: 90,
        };
        let cells = [40.0; 5];
        let mut roof = Roof::from_cells(valley().grid, 10.0, &cells);
        assert!((0..5).all(|x| {
            boundary::exterior_enclosure(
                &cal,
                &size,
                roof.grid.pose(x, 0, 10.0),
                roof.grid.pose(x + 1, 1, 40.0),
                2048,
            ) == Certificate::Covered
        }));
        roof.heights[3] = 20.0;
        roof.heights[roof.grid.columns + 4] = 20.0;
        assert!(roof.connected_cap(None, 0.0).unwrap() < 30.0);
        roof.refine_bottlenecks(&cal, &size, 1, Some(&cells));
        assert!(roof.connected_cap(None, 0.0).unwrap() > 39.99);
    }

    #[test]
    fn certified_lift_recovers_a_passage_without_lowering_a_peak() {
        let cal = cylinder();
        let size = ViewportSize {
            width: 160,
            height: 90,
        };
        let mut roof = valley();
        let before = roof.heights.clone();
        let a = roof.grid.pose(0, 0, 29.0);
        let b = roof.grid.pose(5, 1, 29.0);
        assert!(!roof.segment(a, b));
        roof.refine_bottlenecks(&cal, &size, 256, None);
        assert!(
            roof.heights
                .iter()
                .zip(before)
                .all(|(after, before)| *after >= before)
        );
        let path = roof.route(a, |_, p| {
            [
                0.0,
                0.0,
                f64::from(p.yaw - b.yaw).powi(2) + f64::from(p.pitch - b.pitch).powi(2),
            ]
        });
        assert_eq!(path.last(), Some(&b));
        assert!(
            path.windows(2)
                .all(|p| roof.segment(p[0], p[1]) && roof.segment(p[1], p[0]))
        );
        let cap = roof.connected_cap(Some(roof.grid.yaw), 1e-6).unwrap();
        assert!(cap > 29.99 && cap < 30.0);
        for pair in path.windows(2) {
            for i in 0..=100 {
                assert_eq!(
                    boundary::pose(&cal, &size, interpolate(pair[0], pair[1], i as f32 / 100.0)),
                    Certificate::Covered
                );
            }
        }
    }

    #[test]
    fn failed_or_budget_limited_lifts_restore_every_vertex() {
        let size = ViewportSize {
            width: 160,
            height: 90,
        };
        for budget in [0, 1] {
            let mut roof = valley();
            let before = roof.heights.clone();
            roof.refine_bottlenecks(&cylinder(), &size, budget, None);
            assert_eq!(before, roof.heights);
        }
        let mut cal = cylinder();
        cal.framing.tilt = 1.4;
        let mut roof = valley();
        let before = roof.heights.clone();
        roof.refine_bottlenecks(&cal, &size, 256, None);
        assert_eq!(before, roof.heights);
    }

    #[test]
    fn disconnected_map_vertices_require_projection_proof_before_joining() {
        let cal = cylinder();
        let size = ViewportSize {
            width: 160,
            height: 90,
        };
        let mut roof = Roof::from_cells(valley().grid, 10.0, &[40.0, 40.0, 0.0, 30.0, 30.0]);
        assert!(roof.connected_cap(None, 1e-6).is_none());
        roof.refine_bottlenecks(&cal, &size, 256, None);
        assert!(roof.connected_cap(Some(roof.grid.yaw), 1e-6).unwrap() > 29.99);
    }

    #[test]
    fn refinement_preserves_periodic_seams_and_source_support() {
        let mut cal = cylinder();
        if let crate::calibration::Topology::Cylinder(cylinder) = &mut cal.topology {
            cylinder.sweep_deg = 360.0;
        }
        let size = ViewportSize {
            width: 160,
            height: 90,
        };
        let grid = Grid {
            yaw: [-std::f32::consts::PI, std::f32::consts::PI],
            pitch: [-0.05, 0.05],
            columns: 64,
            rows: 2,
        };
        for case in 0..4 {
            let cells: Vec<_> = (0..128)
                .map(|i| 20.0 + ((i * 37 + case * 17) % 41) as f32)
                .collect();
            let mut roof = Roof::from_cells(grid, 10.0, &cells);
            roof.refine_bottlenecks(&cal, &size, 256, None);
            for y in 0..=grid.rows {
                assert_eq!(roof.heights[y * 65], roof.heights[y * 65 + 64]);
            }
            for i in 0..100 {
                let p = Pose {
                    yaw: -std::f32::consts::PI + std::f32::consts::TAU * i as f32 / 100.0,
                    pitch: 0.0,
                    fov_degrees: 10.0,
                };
                let p = Pose {
                    fov_degrees: roof.ceiling(p).unwrap() as f32 - 0.001,
                    ..p
                };
                assert!(roof.contains(p));
                assert_eq!(boundary::pose(&cal, &size, p), Certificate::Covered);
            }
        }
    }
}
