//! Precomputed coverage boundary for "no-black" viewport constraining.

use std::f32::consts::FRAC_PI_2;

use crate::geometry::VirtualCamera;

// -- Coverage Boundary --
//
// A precomputed, pitch-indexed lookup table of valid yaw ranges for
// "no-black" viewport constraining. Replaces the per-frame frontier
// sampling approach in `viewport_bounds` with O(1) runtime lookups.

/// Precomputed coverage boundary for the stitched panorama.
///
/// Maps each pitch angle to the valid yaw range where the active projection
/// provides pixels. Projections build this analytically or from grouped
/// angular samples. Runtime lookups are O(1) via pitch-indexed interpolation.
///
/// Use [`safe_clamp`](Self::safe_clamp) to constrain a viewport position
/// so no black edges appear.
#[derive(Debug, Clone)]
pub struct CoverageBoundary {
    n_slices: usize,
    /// Global minimum pitch with any coverage (radians).
    pub pitch_min: f32,
    /// Global maximum pitch with any coverage (radians).
    pub pitch_max: f32,
    /// Per-slice combined coverage: `(yaw_min, yaw_max)`.
    slices: Vec<(f32, f32)>,
    /// Minimum pitch range across all yaw positions.
    /// Determines the maximum safe FOV.
    min_pitch_range: f32,
    /// Rig tilt captured at construction (radians). Panning a tilted rig
    /// rolls the rendered viewport against the panorama, so the clamp
    /// must margin against the rotated rectangle, not an axis-aligned one.
    rig_tilt: f32,
    /// Rig roll captured at construction (radians). Same role as `rig_tilt`.
    rig_roll: f32,
    /// Virtual-camera basis of the scene the boundary was built from,
    /// used to evaluate the viewport roll at a candidate pose.
    cam: VirtualCamera,
}

/// Result of clamping a viewport position to the safe panning region.
#[derive(Debug, Clone, Copy)]
pub struct ClampedPosition {
    /// Clamped yaw in radians.
    pub yaw: f32,
    /// Clamped pitch in radians.
    pub pitch: f32,
}

/// Full angular extent of the stitched panorama (radians).
///
/// Returned by [`StitchSession::panorama_extent`](crate::session::StitchSession::panorama_extent).
/// Analytics consumers (heatmaps, zone stats) use this to size grids that
/// span the full coverage rather than hardcoding `+-45 deg yaw, +-20 deg pitch`.
#[derive(Debug, Clone, Copy)]
pub struct PanoramaExtent {
    /// Minimum yaw with coverage from either camera (radians).
    pub yaw_min: f32,
    /// Maximum yaw with coverage from either camera (radians).
    pub yaw_max: f32,
    /// Minimum pitch with coverage from either camera (radians).
    pub pitch_min: f32,
    /// Maximum pitch with coverage from either camera (radians).
    pub pitch_max: f32,
}

impl PanoramaExtent {
    /// Width of the yaw range in radians.
    pub fn yaw_span(&self) -> f32 {
        self.yaw_max - self.yaw_min
    }

    /// Width of the pitch range in radians.
    pub fn pitch_span(&self) -> f32 {
        self.pitch_max - self.pitch_min
    }

    /// Map an angular position in radians to normalized `[0, 1]`
    /// coordinates within this extent.
    ///
    /// Returns `None` if the extent is degenerate (zero span on either
    /// axis). Values outside the extent are returned as-is (not clamped),
    /// so callers can detect out-of-bounds detections.
    pub fn normalize(&self, yaw: f32, pitch: f32) -> Option<(f32, f32)> {
        let yaw_span = self.yaw_span();
        let pitch_span = self.pitch_span();
        if yaw_span <= 0.0 || pitch_span <= 0.0 {
            return None;
        }
        Some((
            (yaw - self.yaw_min) / yaw_span,
            (pitch - self.pitch_min) / pitch_span,
        ))
    }
}

impl CoverageBoundary {
    /// Analytic rectangular coverage: the same yaw range at every
    /// pitch. Projection implementations supply their virtual-camera
    /// basis so roll-aware clamping is not coupled to one topology.
    pub(crate) fn rectangular(
        yaw_min: f32,
        yaw_max: f32,
        pitch_min: f32,
        pitch_max: f32,
        cam: VirtualCamera,
        rig_tilt: f32,
        rig_roll: f32,
    ) -> Self {
        let n_slices: usize = 16;
        Self {
            n_slices,
            pitch_min,
            pitch_max,
            slices: vec![(yaw_min, yaw_max); n_slices],
            min_pitch_range: pitch_max - pitch_min,
            rig_tilt,
            rig_roll,
            cam,
        }
    }

    /// Build the representation from source-grouped angular samples.
    ///
    /// Source grouping is retained while estimating vertical room at each
    /// yaw. Flattening disjoint cameras would invent coverage across the gap
    /// between their individual yaw ranges.
    pub(crate) fn from_samples(
        per_camera: &[&[(f32, f32)]],
        n_slices: usize,
        cam: VirtualCamera,
        rig_tilt: f32,
        rig_roll: f32,
    ) -> Self {
        assert!(n_slices > 0, "coverage needs at least one pitch slice");

        // Find the global pitch range and reject values that would make the
        // slice lookup or percentile sort partial and non-deterministic.
        let mut global_pitch_min = f32::MAX;
        let mut global_pitch_max = f32::MIN;
        for points in per_camera {
            for &(yaw, pitch) in *points {
                assert!(
                    yaw.is_finite() && pitch.is_finite(),
                    "coverage samples must be finite"
                );
                global_pitch_min = global_pitch_min.min(pitch);
                global_pitch_max = global_pitch_max.max(pitch);
            }
        }

        if global_pitch_min >= global_pitch_max {
            return Self {
                n_slices,
                pitch_min: 0.0,
                pitch_max: 0.0,
                slices: vec![(0.0, 0.0); n_slices],
                min_pitch_range: 0.0,
                rig_tilt,
                rig_roll,
                cam,
            };
        }

        let pitch_range = global_pitch_max - global_pitch_min;
        let slice_size = pitch_range / n_slices as f32;

        // Bucket points into merged and per-camera pitch slices.
        let mut slices = vec![(f32::MAX, f32::MIN); n_slices];
        let mut camera_slices = vec![vec![(f32::MAX, f32::MIN); n_slices]; per_camera.len()];

        let pitch_to_slice = |pitch: f32| -> usize {
            let idx = ((pitch - global_pitch_min) / slice_size) as usize;
            idx.min(n_slices - 1)
        };

        for (points, source_slices) in per_camera.iter().zip(&mut camera_slices) {
            for &(yaw, pitch) in *points {
                let s = pitch_to_slice(pitch);
                source_slices[s].0 = source_slices[s].0.min(yaw);
                source_slices[s].1 = source_slices[s].1.max(yaw);
                slices[s].0 = slices[s].0.min(yaw);
                slices[s].1 = slices[s].1.max(yaw);
            }
        }

        // Compute min pitch range (determines max FOV)
        let min_pitch_range = {
            let quarter = n_slices / 4;
            let three_quarter = 3 * n_slices / 4;
            let mut yaw_lo = f32::MAX;
            let mut yaw_hi = f32::MIN;
            for s in &slices[quarter..three_quarter] {
                if s.0 <= s.1 {
                    yaw_lo = yaw_lo.min(s.0);
                    yaw_hi = yaw_hi.max(s.1);
                }
            }

            if yaw_lo >= yaw_hi {
                pitch_range
            } else {
                // Sample pitch range at multiple yaw positions and use the
                // 10th percentile instead of absolute minimum. The absolute
                // minimum is dominated by narrow seam edges which the director
                // rarely visits. The 10th percentile gives a practical FOV
                // that works across most of the useful yaw range.
                let n_samples = 50;
                let mut ranges = Vec::with_capacity(n_samples + 1);
                for j in 0..=n_samples {
                    let t = j as f32 / n_samples as f32;
                    let test_yaw = yaw_lo + t * (yaw_hi - yaw_lo);
                    let mut p_lo = f32::MAX;
                    let mut p_hi = f32::MIN;
                    for s in 0..n_slices {
                        let covered = camera_slices.iter().any(|source| {
                            let (yaw_min, yaw_max) = source[s];
                            yaw_min <= yaw_max && test_yaw >= yaw_min && test_yaw <= yaw_max
                        });
                        if covered {
                            let p = global_pitch_min + (s as f32 + 0.5) * slice_size;
                            p_lo = p_lo.min(p);
                            p_hi = p_hi.max(p);
                        }
                    }
                    if p_hi > p_lo {
                        ranges.push(p_hi - p_lo);
                    }
                }
                if ranges.is_empty() {
                    pitch_range
                } else {
                    ranges.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    // 10th percentile: skip the narrowest 10%
                    let idx = (ranges.len() / 10).min(ranges.len() - 1);
                    ranges[idx]
                }
            }
        };

        // Fill gaps: slices with no samples inherit from neighbors
        for i in 1..n_slices {
            if slices[i].0 > slices[i].1 {
                slices[i] = slices[i - 1];
            }
        }
        for i in (0..n_slices - 1).rev() {
            if slices[i].0 > slices[i].1 {
                slices[i] = slices[i + 1];
            }
        }

        log::debug!(
            "CoverageBoundary: pitch [{:.3}, {:.3}] ({:.1} deg), min pitch range {:.1} deg",
            global_pitch_min,
            global_pitch_max,
            pitch_range.to_degrees(),
            min_pitch_range.to_degrees(),
        );

        Self {
            n_slices,
            pitch_min: global_pitch_min,
            pitch_max: global_pitch_max,
            slices,
            min_pitch_range,
            rig_tilt,
            rig_roll,
            cam,
        }
    }

    /// Refresh the captured rig orientation without resampling.
    ///
    /// Coverage geometry is world-fixed; rig tilt and roll only change how
    /// the viewport traverses it. Only the roll-aware clamp margins read
    /// these scalars, so a live orientation change needs no resampling.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // live-orientation engine hook
    pub(crate) fn set_rig_orientation(&mut self, rig_tilt: f32, rig_roll: f32) {
        self.rig_tilt = rig_tilt;
        self.rig_roll = rig_roll;
    }

    /// Global yaw coverage range across the full panorama (radians).
    ///
    /// Returns `(yaw_min, yaw_max)`, the widest-point extremes of the
    /// stitched panorama. Useful for heatmap consumers that need to
    /// bucket detections by yaw across the full coverage.
    ///
    /// This is the global extent, not the per-pitch range. For a
    /// pitch-aware range, sample with [`yaw_range_at`](Self::yaw_range_at).
    pub fn yaw_range(&self) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &(a, b) in &self.slices {
            if a <= b {
                lo = lo.min(a);
                hi = hi.max(b);
            }
        }
        if lo > hi { (0.0, 0.0) } else { (lo, hi) }
    }

    /// Global pitch coverage range across the full panorama (radians).
    ///
    /// Returns `(pitch_min, pitch_max)`. Used alongside
    /// [`yaw_range`](Self::yaw_range) by heatmap and analytics consumers
    /// that need panorama bounds without reaching into private state.
    pub fn pitch_range(&self) -> (f32, f32) {
        (self.pitch_min, self.pitch_max)
    }

    /// Look up the combined yaw coverage range at a given pitch.
    ///
    /// Returns the interpolated `(yaw_min, yaw_max)` where the projection
    /// provides coverage. Used by the director for
    /// perspective-aware clamping; analytics consumers typically want
    /// [`yaw_range`](Self::yaw_range) for the global extent instead.
    pub fn yaw_range_at(&self, pitch: f32) -> (f32, f32) {
        self.interpolate_slice(&self.slices, pitch)
    }

    /// Interpolate a slice table at the given pitch.
    fn interpolate_slice(&self, table: &[(f32, f32)], pitch: f32) -> (f32, f32) {
        if self.n_slices == 0 || self.pitch_max <= self.pitch_min {
            return (0.0, 0.0);
        }
        let pitch_range = self.pitch_max - self.pitch_min;
        let t = (pitch - self.pitch_min) / pitch_range;
        let idx_f = t * (self.n_slices - 1) as f32;
        let idx_lo = (idx_f.floor() as usize).min(self.n_slices - 1);
        let idx_hi = (idx_lo + 1).min(self.n_slices - 1);
        let frac = idx_f - idx_lo as f32;

        let lo = table[idx_lo];
        let hi = table[idx_hi];
        if lo.0 > lo.1 {
            return hi;
        }
        if hi.0 > hi.1 {
            return lo;
        }
        (lo.0 + frac * (hi.0 - lo.0), lo.1 + frac * (hi.1 - lo.1))
    }

    /// Clamp a viewport center to the safe (no-black) panning region for
    /// the given FOV, in world space, with perspective-correct margins.
    ///
    /// Inputs and `self` are both world-space (the panorama's native
    /// frame). Rig tilt/roll correction is a render-site concern applied
    /// after clamping (see `rig_correction::world_to_render_pose`), so no
    /// human<->world mapping happens here.
    pub fn safe_clamp(&self, yaw: f32, pitch: f32, fov_v_deg: f32, aspect: f32) -> ClampedPosition {
        // B-30 defense: non-finite inputs would propagate through the
        // clamp / comparisons and emit NaN, which then flows into the
        // MVP matrix and produces a black or garbage frame. Upstream
        // guards (B-28 detector boundary, B-29 director EMA) stop most
        // NaN at the source, but user overrides and external clients
        // can still hand us non-finite values. Fall back to the
        // coverage center.
        if !yaw.is_finite() || !pitch.is_finite() || !fov_v_deg.is_finite() || !aspect.is_finite() {
            let safe_pitch = (self.pitch_min + self.pitch_max) * 0.5;
            let (yaw_lo, yaw_hi) = self.yaw_range_at(safe_pitch);
            let safe_yaw = (yaw_lo + yaw_hi) * 0.5;
            return ClampedPosition {
                yaw: safe_yaw,
                pitch: safe_pitch,
            };
        }

        let half_vfov = (fov_v_deg * 0.5).to_radians();
        let half_hfov = (aspect * half_vfov.tan()).atan();

        // First pass: axis-aligned viewport margins.
        let first = self.clamp_with_margins(yaw, pitch, half_vfov, half_hfov);
        if self.rig_tilt.abs() < 1e-6 && self.rig_roll.abs() < 1e-6 {
            return first;
        }

        // Panning a tilted/rolled rig rolls the rendered viewport against
        // the panorama, so its corners overhang the axis-aligned margin
        // box (measured 4-9% black leak at ~19 deg tilt before this).
        // Evaluate the roll at the first-pass pose - it varies slowly, so
        // one refinement pass suffices - and re-clamp with the rotated
        // rectangle's extents as margins.
        let vroll = crate::geometry::render_viewport_roll(
            &self.cam,
            first.yaw,
            first.pitch,
            self.rig_tilt,
            self.rig_roll,
        )
        .abs()
        .min(FRAC_PI_2);
        let (sin_r, cos_r) = (vroll.sin(), vroll.cos());
        self.clamp_with_margins(
            yaw,
            pitch,
            half_hfov * sin_r + half_vfov * cos_r,
            half_hfov * cos_r + half_vfov * sin_r,
        )
    }

    /// One clamp pass with explicit half-viewport margins: pitch against
    /// the global bounds, then yaw against the coverage range at that
    /// pitch. Degenerate ranges (viewport taller/wider than the coverage)
    /// fall back to the range midpoint.
    fn clamp_with_margins(
        &self,
        yaw: f32,
        pitch: f32,
        half_v: f32,
        half_h: f32,
    ) -> ClampedPosition {
        let clamped_pitch = if self.pitch_min + half_v <= self.pitch_max - half_v {
            pitch.clamp(self.pitch_min + half_v, self.pitch_max - half_v)
        } else {
            (self.pitch_min + self.pitch_max) * 0.5
        };

        let (yaw_lo, yaw_hi) = self.yaw_range_at(clamped_pitch);
        let clamped_yaw = if yaw_lo + half_h <= yaw_hi - half_h {
            yaw.clamp(yaw_lo + half_h, yaw_hi - half_h)
        } else {
            (yaw_lo + yaw_hi) * 0.5
        };

        ClampedPosition {
            yaw: clamped_yaw,
            pitch: clamped_pitch,
        }
    }

    /// Maximum vertical FOV (degrees) that fits within the coverage.
    ///
    /// This is the widest zoom-out where at least one valid viewport
    /// position exists. Determined by the narrowest pitch range across
    /// the projection's useful yaw positions.
    pub fn max_fov_degrees(&self) -> f32 {
        if self.min_pitch_range <= 0.0 {
            return 20.0;
        }
        self.min_pitch_range.to_degrees()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_coverage_uses_supplied_camera_basis() {
        let camera = VirtualCamera::new(&[0.2, 0.0, 0.3]);
        let coverage = CoverageBoundary::rectangular(-1.0, 1.0, -0.5, 0.5, camera, 0.0, 0.0);
        assert_eq!(coverage.cam.eye, camera.eye);
        assert_eq!(coverage.cam.base_forward, camera.base_forward);
        assert_eq!(coverage.cam.base_right, camera.base_right);
    }

    #[test]
    fn grouped_samples_do_not_bridge_disjoint_camera_yaw_ranges() {
        let mut left = Vec::new();
        for pitch in [0.0, 0.2, 0.4, 0.6] {
            left.push((-2.0, pitch));
            left.push((-1.95, pitch));
        }
        let mut right = Vec::new();
        for pitch in [0.4, 0.6, 0.8, 1.0] {
            right.push((1.95, pitch));
            right.push((2.0, pitch));
        }

        let grouped = CoverageBoundary::from_samples(
            &[left.as_slice(), right.as_slice()],
            20,
            VirtualCamera::mono(),
            0.0,
            0.0,
        );
        let reversed = CoverageBoundary::from_samples(
            &[right.as_slice(), left.as_slice()],
            20,
            VirtualCamera::mono(),
            0.0,
            0.0,
        );

        let mut flattened = left;
        flattened.extend(right);
        let ungrouped = CoverageBoundary::from_samples(
            &[flattened.as_slice()],
            20,
            VirtualCamera::mono(),
            0.0,
            0.0,
        );

        assert_eq!(grouped.yaw_range(), reversed.yaw_range());
        assert_eq!(grouped.pitch_range(), reversed.pitch_range());
        assert_eq!(grouped.max_fov_degrees(), reversed.max_fov_degrees());
        assert!(
            grouped.max_fov_degrees() > ungrouped.max_fov_degrees() + 10.0,
            "flattening sources must not invent coverage across their yaw gap"
        );
    }

    #[test]
    #[should_panic(expected = "coverage samples must be finite")]
    fn sampled_coverage_rejects_non_finite_points() {
        let samples = [(0.0, f32::NAN)];
        CoverageBoundary::from_samples(&[&samples], 4, VirtualCamera::mono(), 0.0, 0.0);
    }
}
