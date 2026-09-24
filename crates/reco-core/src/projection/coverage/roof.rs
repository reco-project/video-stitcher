//! Continuous inner approximation of exterior viewport coverage.
//! Enclosed unsupported components are outside the construction contract.

use super::*;

mod interaction;
mod refinement;
mod routing;

#[cfg(test)]
mod diagnostics;
#[cfg(test)]
mod tests;

pub(super) const FOV_EPSILON: f64 = 1e-4;

#[derive(Clone, Copy)]
pub(super) struct Grid {
    pub(super) yaw: [f32; 2],
    pub(super) pitch: [f32; 2],
    pub(super) columns: usize,
    pub(super) rows: usize,
}

impl Grid {
    fn wraps_yaw(self) -> bool {
        self.yaw == [-std::f32::consts::PI, std::f32::consts::PI]
    }

    fn seam_offset(self, x: usize) -> isize {
        if !self.wraps_yaw() {
            0
        } else if x == 0 {
            self.columns as isize
        } else if x == self.columns {
            -(self.columns as isize)
        } else {
            0
        }
    }

    fn near_yaw(self, mut pose: Pose, reference: f32) -> Pose {
        if self.wraps_yaw() {
            let period = f64::from(self.yaw[1]) - f64::from(self.yaw[0]);
            let reference = f64::from(reference);
            let delta = f64::from(pose.yaw) - reference;
            if delta.abs() > period * 0.5 {
                // Reduce before subtracting: large finite angles must not leave
                // an enormous cancellation residual or an unbounded grid walk.
                let delta = (f64::from(pose.yaw).rem_euclid(period) - reference.rem_euclid(period)
                    + period * 0.5)
                    .rem_euclid(period)
                    - period * 0.5;
                pose.yaw = (reference + delta) as f32;
            }
        }
        pose
    }
    fn pose(self, x: usize, y: usize, fov_degrees: f32) -> Pose {
        Pose {
            yaw: self.yaw[0] + (self.yaw[1] - self.yaw[0]) * x as f32 / self.columns as f32,
            pitch: self.pitch[0] + (self.pitch[1] - self.pitch[0]) * y as f32 / self.rows as f32,
            fov_degrees,
        }
    }

    fn valid(self) -> bool {
        [self.yaw, self.pitch]
            .iter()
            .all(|r| r.iter().all(|v| v.is_finite()) && r[0] < r[1])
            && self.columns > 0
            && self.rows > 0
            && self
                .columns
                .checked_mul(self.rows)
                .is_some_and(|n| n <= 200_000)
    }
}

pub(super) struct Roof {
    grid: Grid,
    minimum_fov: f32,
    heights: Vec<f32>,
}

impl Roof {
    pub(super) fn near_yaw(&self, pose: Pose, reference: f32) -> Pose {
        self.grid.near_yaw(pose, reference)
    }

    pub(super) fn vertices(&self) -> impl Iterator<Item = Pose> + '_ {
        let stride = self.grid.columns + 1;
        (0..self.heights.len())
            .map(move |i| self.grid.pose(i % stride, i / stride, self.minimum_fov))
    }

    pub(super) fn build(
        cal: &Calibration,
        size: &ViewportSize,
        grid: Grid,
        fov: [f32; 2],
        tolerance: f32,
    ) -> Option<Self> {
        if !grid.valid()
            || cal.validate().is_err()
            || size.validate().is_err()
            || !fov
                .iter()
                .all(|v| v.is_finite() && (1.0..=178.0).contains(v))
            || fov[0] >= fov[1]
            || !tolerance.is_finite()
            || tolerance <= 0.0
        {
            return None;
        }
        let workers = std::thread::available_parallelism().map_or(1, usize::from);
        Some(Self::build_cells(cal, size, grid, fov, tolerance, workers))
    }

    fn build_cells(
        cal: &Calibration,
        size: &ViewportSize,
        grid: Grid,
        fov: [f32; 2],
        tolerance: f32,
        workers: usize,
    ) -> Self {
        let count = grid.columns * grid.rows;
        let workers = workers.clamp(1, count);
        let compute = |i| {
            Self::cell_ceiling(
                cal,
                size,
                grid,
                fov,
                tolerance,
                [i % grid.columns, i / grid.columns],
            )
        };
        let cells = if workers == 1 {
            (0..count).map(compute).collect()
        } else {
            std::thread::scope(|scope| {
                // Interleave cells so workers share the more expensive boundary regions.
                let jobs: Vec<_> = (0..workers)
                    .map(|worker| {
                        let compute = &compute;
                        scope.spawn(move || {
                            (worker..count)
                                .step_by(workers)
                                .map(|i| (i, compute(i)))
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                let mut cells = vec![0.0; count];
                for job in jobs {
                    for (i, ceiling) in job.join().unwrap() {
                        cells[i] = ceiling;
                    }
                }
                cells
            })
        };
        let mut roof = Self::from_cells(grid, fov[0], &cells);
        // Bound additional projection proofs across the entire refinement pass.
        roof.refine_bottlenecks(cal, size, 256, Some(&cells));
        roof
    }

    fn cell_ceiling(
        cal: &Calibration,
        size: &ViewportSize,
        grid: Grid,
        fov: [f32; 2],
        tolerance: f32,
        [x, y]: [usize; 2],
    ) -> f32 {
        let yaw = |x: usize| {
            f64::from(grid.yaw[0])
                + (f64::from(grid.yaw[1]) - f64::from(grid.yaw[0])) * x as f64 / grid.columns as f64
        };
        let pitch = |y: usize| {
            f64::from(grid.pitch[0])
                + (f64::from(grid.pitch[1]) - f64::from(grid.pitch[0])) * y as f64
                    / grid.rows as f64
        };
        let lo = Pose {
            yaw: (yaw(x) as f32).next_down(),
            pitch: (pitch(y) as f32).next_down(),
            fov_degrees: fov[0],
        };
        let hi = Pose {
            yaw: (yaw(x + 1) as f32).next_up(),
            pitch: (pitch(y + 1) as f32).next_up(),
            fov_degrees: fov[0],
        };
        let supported = |maximum| {
            boundary::exterior_enclosure(
                cal,
                size,
                lo,
                Pose {
                    fov_degrees: maximum,
                    ..hi
                },
                2048,
            ) == Certificate::Covered
        };
        let mut low = 0.0;
        if supported(fov[0]) {
            low = fov[0];
            let mut high = fov[1];
            if supported(high) {
                low = high;
            } else {
                while high - low > tolerance {
                    let middle = (low + high) * 0.5;
                    if middle == low || middle == high {
                        break;
                    }
                    if supported(middle) {
                        low = middle;
                    } else {
                        high = middle;
                    }
                }
            }
        }
        low
    }

    pub(super) fn from_cells(grid: Grid, minimum_fov: f32, cells: &[f32]) -> Self {
        assert_eq!(cells.len(), grid.columns * grid.rows);
        assert!(cells.iter().all(|h| h.is_finite() && *h >= 0.0));
        let mut heights = vec![f32::INFINITY; (grid.columns + 1) * (grid.rows + 1)];
        for y in 0..grid.rows {
            for x in 0..grid.columns {
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let vertex = &mut heights[(y + dy) * (grid.columns + 1) + x + dx];
                    *vertex = vertex.min(cells[y * grid.columns + x]);
                }
            }
        }
        if grid.wraps_yaw() {
            for y in 0..=grid.rows {
                let first = y * (grid.columns + 1);
                let last = first + grid.columns;
                let height = heights[first].min(heights[last]);
                heights[first] = height;
                heights[last] = height;
            }
        }
        Self {
            grid,
            minimum_fov,
            heights,
        }
    }

    fn main_diagonal(&self, x: usize, y: usize) -> bool {
        let i = y * (self.grid.columns + 1) + x;
        let below = i + self.grid.columns + 1;
        // Connect the higher pair so a fixed diagonal cannot cut a supported
        // ridge in two. Refinement re-certifies any triangles changed by a lift.
        self.heights[i].min(self.heights[below + 1]) >= self.heights[i + 1].min(self.heights[below])
    }

    fn triangles(&self, x: usize, y: usize) -> [[(usize, usize); 3]; 2] {
        if self.main_diagonal(x, y) {
            [[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]]
        } else {
            [[(0, 0), (1, 0), (0, 1)], [(1, 0), (1, 1), (0, 1)]]
        }
    }

    // Vertex index and unsigned grid travel, with zero travel across the seam.
    fn neighbors(&self, i: usize) -> impl Iterator<Item = (usize, [usize; 2])> + '_ {
        let stride = self.grid.columns + 1;
        let (x, y) = (i % stride, i / stride);
        let edges = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
        ]
        .into_iter()
        .filter_map(move |(dx, dy)| {
            let nx = x
                .checked_add_signed(dx)
                .filter(|x| *x <= self.grid.columns)?;
            let ny = y.checked_add_signed(dy).filter(|y| *y <= self.grid.rows)?;
            if dx != 0 && dy != 0 && self.main_diagonal(x.min(nx), y.min(ny)) != (dx == dy) {
                return None;
            }
            Some((ny * stride + nx, [dx.unsigned_abs(), dy.unsigned_abs()]))
        });
        let seam = self.grid.seam_offset(x);
        edges.chain((seam != 0).then(|| (i.checked_add_signed(seam).unwrap(), [0, 0])))
    }

    // Keep every supported orientation connected at every FOV from the
    // minimum through this cap, and retain a passage across the supplied yaw
    // span. One left-to-right route alone says nothing about side branches.
    pub(super) fn connected_cap(
        &self,
        yaw: Option<[f32; 2]>,
        motion_precision: f32,
    ) -> Option<f32> {
        if yaw.is_some_and(|yaw| {
            !yaw.iter().all(|v| v.is_finite())
                || yaw[0] >= yaw[1]
                || yaw[0] < self.grid.yaw[0]
                || yaw[1] > self.grid.yaw[1]
        }) || !motion_precision.is_finite()
            || motion_precision < 0.0
        {
            return None;
        }
        let minimum = f64::from(self.minimum_fov) + FOV_EPSILON;
        let mut ceiling = self.heights.iter().copied().fold(0.0_f32, f32::max);
        if f64::from(ceiling) < minimum {
            return None;
        }
        let stride = self.grid.columns + 1;
        let (capacity, _) = self.connections();
        for (&height, &return_fov) in self.heights.iter().zip(&capacity) {
            if f64::from(height) < minimum {
                continue;
            }
            if f64::from(return_fov) < minimum {
                return None;
            }
            if height > return_fov {
                ceiling = ceiling.min(return_fov);
            }
        }
        let column = |yaw| {
            self.coordinates(Pose {
                yaw,
                pitch: self.grid.pitch[0],
                fov_degrees: self.minimum_fov,
            })
            .unwrap()[0]
        };
        if let Some(yaw) = yaw {
            let first = column(yaw[0]).floor() as usize;
            let last = column(yaw[1]).ceil() as usize;
            for x in first..=last {
                let highest = (0..=self.grid.rows)
                    .map(|y| self.heights[y * stride + x])
                    .fold(0.0_f32, f32::max);
                ceiling = ceiling.min(highest);
            }
        }
        let dx =
            (f64::from(self.grid.yaw[1]) - f64::from(self.grid.yaw[0])) / self.grid.columns as f64;
        let dy =
            (f64::from(self.grid.pitch[1]) - f64::from(self.grid.pitch[0])) / self.grid.rows as f64;
        let mut slope = 0.0_f64;
        for y in 0..self.grid.rows {
            for x in 0..self.grid.columns {
                let at = |ox, oy| f64::from(self.heights[(y + oy) * stride + x + ox]);
                let (a, b, c, d) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
                if self.main_diagonal(x, y) {
                    slope = slope.max(((b - a) / dx).hypot((d - b) / dy));
                    slope = slope.max(((d - c) / dx).hypot((c - a) / dy));
                } else {
                    slope = slope.max(((b - a) / dx).hypot((c - a) / dy));
                    slope = slope.max(((d - c) / dx).hypot((d - b) / dy));
                }
            }
        }
        // A point-sized passage cannot be resolved by a finite-precision
        // controller. The global slope bounds FOV loss within its positional
        // tolerance. This is applied once to the cap, not to runtime viewports.
        let clearance = slope * f64::from(motion_precision);
        let cap = ((f64::from(ceiling) - FOV_EPSILON - clearance) as f32).next_down();
        (cap >= self.minimum_fov).then_some(cap)
    }

    fn connections(&self) -> (Vec<f32>, Vec<usize>) {
        use std::cmp::Reverse;
        let minimum = f64::from(self.minimum_fov) + FOV_EPSILON;
        let peak = (0..self.heights.len())
            .max_by(|&a, &b| self.heights[a].total_cmp(&self.heights[b]))
            .unwrap();
        let ceiling = self.heights[peak];
        let mut capacity = vec![0.0_f32; self.heights.len()];
        let mut parents = vec![usize::MAX; self.heights.len()];
        parents[peak] = peak;
        capacity[peak] = ceiling;
        let mut remaining = self
            .heights
            .iter()
            .filter(|&&h| f64::from(h) >= minimum)
            .count();
        if remaining == 0 {
            return (capacity, parents);
        }
        // Uncertified vertices are candidates for preparation-time repair, not
        // runtime routes. Prefer the fewest such vertices when capacities tie.
        let mut gaps = vec![usize::MAX; self.heights.len()];
        gaps[peak] = 0;
        // Finite nonnegative FOVs have the same numeric and unsigned bit order.
        let mut pending =
            std::collections::BinaryHeap::from([(ceiling.to_bits(), Reverse(0), peak)]);
        while let Some((bits, Reverse(gap), i)) = pending.pop() {
            let level = f32::from_bits(bits);
            if level < capacity[i] || gap > gaps[i] {
                continue;
            }
            if f64::from(self.heights[i]) >= minimum {
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
            for (j, _) in self.neighbors(i) {
                let next = level.min(self.heights[j]);
                let next_gap = gap + usize::from(f64::from(self.heights[j]) < minimum);
                if next > capacity[j] || (next == capacity[j] && next_gap < gaps[j]) {
                    capacity[j] = next;
                    gaps[j] = next_gap;
                    parents[j] = i;
                    pending.push((next.to_bits(), Reverse(next_gap), j));
                }
            }
        }
        (capacity, parents)
    }

    fn coordinates(&self, p: Pose) -> Option<[f64; 2]> {
        if !p.yaw.is_finite()
            || !p.pitch.is_finite()
            || (!self.grid.wraps_yaw() && (p.yaw < self.grid.yaw[0] || p.yaw > self.grid.yaw[1]))
            || p.pitch < self.grid.pitch[0]
            || p.pitch > self.grid.pitch[1]
        {
            return None;
        }
        Some([
            (f64::from(p.yaw) - f64::from(self.grid.yaw[0]))
                / (f64::from(self.grid.yaw[1]) - f64::from(self.grid.yaw[0]))
                * self.grid.columns as f64,
            (f64::from(p.pitch) - f64::from(self.grid.pitch[0]))
                / (f64::from(self.grid.pitch[1]) - f64::from(self.grid.pitch[0]))
                * self.grid.rows as f64,
        ])
    }

    fn height_at(&self, [mut x, y]: [f64; 2]) -> f64 {
        if self.grid.wraps_yaw() {
            x = x.rem_euclid(self.grid.columns as f64);
        }
        let ix = (x.floor() as usize).min(self.grid.columns - 1);
        let iy = (y.floor() as usize).min(self.grid.rows - 1);
        let u = x - ix as f64;
        let v = y - iy as f64;
        let at = |dx, dy| f64::from(self.heights[(iy + dy) * (self.grid.columns + 1) + ix + dx]);
        // Shared edge values keep the certified triangles continuous even
        // when adjacent cells split differently.
        if self.main_diagonal(ix, iy) {
            if u >= v {
                (1.0 - u) * at(0, 0) + (u - v) * at(1, 0) + v * at(1, 1)
            } else {
                (1.0 - v) * at(0, 0) + (v - u) * at(0, 1) + u * at(1, 1)
            }
        } else if u + v <= 1.0 {
            (1.0 - u - v) * at(0, 0) + u * at(1, 0) + v * at(0, 1)
        } else {
            (1.0 - v) * at(1, 0) + (u + v - 1.0) * at(1, 1) + (1.0 - u) * at(0, 1)
        }
    }

    fn ceiling(&self, p: Pose) -> Option<f64> {
        self.coordinates(p).map(|xy| self.height_at(xy))
    }

    pub(super) fn contains(&self, p: Pose) -> bool {
        p.fov_degrees.is_finite()
            && p.fov_degrees >= self.minimum_fov
            && self
                .ceiling(p)
                .is_some_and(|h| f64::from(p.fov_degrees) + FOV_EPSILON <= h)
    }

    pub(super) fn supported_vertices(
        &self,
        fov_degrees: f32,
    ) -> impl Iterator<Item = (usize, Pose)> + '_ {
        self.heights.iter().enumerate().filter_map(move |(i, _)| {
            let p = self.grid.pose(
                i % (self.grid.columns + 1),
                i / (self.grid.columns + 1),
                fov_degrees,
            );
            self.contains(p).then_some((i, p))
        })
    }

    pub(super) fn segment(&self, from: Pose, to: Pose) -> bool {
        let Some(a) = self.coordinates(from) else {
            return false;
        };
        let Some(b) = self.coordinates(to) else {
            return false;
        };
        let supported = |p: Pose, xy| {
            p.fov_degrees.is_finite()
                && p.fov_degrees >= self.minimum_fov
                && f64::from(p.fov_degrees) + FOV_EPSILON <= self.height_at(xy)
        };
        if !supported(from, a) || !supported(to, b) {
            return false;
        }
        let supported = |t: f64| {
            let xy = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
            let fov = f64::from(from.fov_degrees)
                + t * (f64::from(to.fov_degrees) - f64::from(from.fov_degrees));
            fov + FOV_EPSILON <= self.height_at(xy)
        };
        Self::at_crossings(a, b, supported)
    }

    // Constant-FOV paths share one exact ceiling across all zoom trials.
    pub(super) fn segment_ceiling(&self, from: Pose, to: Pose) -> Option<f64> {
        let a = self.coordinates(from)?;
        let b = self.coordinates(to)?;
        let mut ceiling = self.height_at(a).min(self.height_at(b));
        Self::at_crossings(a, b, |t| {
            let xy = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
            ceiling = ceiling.min(self.height_at(xy));
            true
        });
        Some(ceiling)
    }

    fn at_crossings(a: [f64; 2], b: [f64; 2], mut supported: impl FnMut(f64) -> bool) -> bool {
        // Within each triangle both the roof and the queried segment's FOV
        // are affine. Their difference is bounded by its endpoint values.
        // Include both diagonal families. Extra crossings only subdivide affine spans.
        for (start, end) in [
            (a[0], b[0]),
            (a[1], b[1]),
            (a[0] - a[1], b[0] - b[1]),
            (a[0] + a[1], b[0] + b[1]),
        ] {
            if start == end {
                continue;
            }
            for line in start.min(end).ceil() as i64..=start.max(end).floor() as i64 {
                let t = (line as f64 - start) / (end - start);
                if t > 0.0 && t < 1.0 && !supported(t) {
                    return false;
                }
            }
        }
        true
    }
}
