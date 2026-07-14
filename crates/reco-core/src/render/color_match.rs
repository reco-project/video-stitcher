//! Per-camera exposure/white-balance matching at the stitch seam.
//!
//! Two independently-metering action cameras rarely agree on exposure or
//! white balance, showing up as a visible color/brightness step at the
//! seam, independent of geometric alignment. This periodically samples a
//! coarse grid in the seam-adjacent band of each camera's raw frame,
//! derives a small per-camera YUV offset that nudges both toward their
//! shared mean, and feeds it into the shader's existing (previously
//! always-identity) `color_scale`/`color_offset_blend` uniforms.
//!
//! ## Sampling geometry - the one thing this MUST get exactly right
//!
//! The sampling point in plane UV -> source pixel mapping here
//! deliberately reuses the same forward-KB4 primitive
//! ([`crate::lens::kb4::kb4_forward_scale_with_correction`]) and the same
//! normalized-intrinsics convention (`fx/width`, `fy/height`, ...) as
//! [`crate::stitch::geometry`]'s `PlaneMap` and `fisheye.wgsl`'s
//! `fs_main` - NOT [`crate::lens::undistorted_to_distorted`], which uses
//! a different (halved-FOV, plane-fitted) intrinsics convention built for
//! a different consumer (the single-camera undistort preview) and would
//! silently sample the wrong region here. Getting this mapping wrong is
//! exactly the class of bug this feature's own history already hit twice
//! (see FRICTION.md/git history) - sampling unrelated scene content and
//! applying the resulting "correction" made the seam worse, not better.

use crate::calibration::Lens;
use crate::lens::kb4;
use crate::render::planes::{Nv12Planes, YuvPlanes};

/// Width of the seam-adjacent sampling band, in the plane's own local UV
/// units (same space as `Topology::blend_width`/`seam_offset`).
const BAND_WIDTH: f64 = 0.08;
/// Sampling grid density within the band.
const GRID_COLS: usize = 4;
const GRID_ROWS: usize = 6;
/// Re-measure every N frames - the correction only needs to track slow
/// lighting drift, not per-frame noise.
const INTERVAL_FRAMES: u64 = 15;
/// Exponential-moving-average smoothing factor for the correction
/// (higher = faster to react, noisier).
const EMA_ALPHA: f32 = 0.15;
/// Clamp on the luma (Y) correction, in normalized `[0, 1]` units.
const MAX_Y_OFFSET: f32 = 0.06;
/// Clamp on each chroma (U/V) correction, in normalized `[0, 1]` units.
const MAX_CHROMA_OFFSET: f32 = 0.03;

/// Map a plane-local UV coordinate to the corresponding source-frame UV
/// via forward KB4 distortion. `None` if the point falls outside the
/// source frame (matches `fisheye.wgsl`'s bounds check and
/// `PlaneMap::sample_uv`'s forward-mapping section exactly).
fn plane_uv_to_source_uv(uv_x: f64, uv_y: f64, cam: &Lens) -> Option<(f64, f64)> {
    let euv_x = uv_x * 2.0 - 0.5;
    let euv_y = uv_y * 2.0 - 0.5;
    let fx_n = cam.fx / cam.width as f64;
    let fy_n = cam.fy / cam.height as f64;
    let cx_n = cam.cx / cam.width as f64;
    let cy_n = cam.cy / cam.height as f64;
    let xn = (euv_x - cx_n) / fx_n;
    let yn = (euv_y - cy_n) / fy_n;
    let r = (xn * xn + yn * yn).sqrt();
    let scale = kb4::kb4_forward_scale_with_correction(r, &cam.distortion, cam.correction as f64);
    let du = fx_n * xn * scale + cx_n;
    let dv = fy_n * yn * scale + cy_n;
    if !(0.0..=1.0).contains(&du) || !(0.0..=1.0).contains(&dv) {
        return None;
    }
    Some((du, dv))
}

/// Nearest-sample a normalized `[0, 1]` Y/U/V byte value at source UV
/// `(u, v)` from tightly-packed YUV420P planes.
fn sample_yuv420p(planes: &YuvPlanes<'_>, w: u32, h: u32, u: f64, v: f64) -> (f32, f32, f32) {
    let x = ((u * w as f64) as u32).min(w - 1);
    let y = ((v * h as f64) as u32).min(h - 1);
    let cw = w / 2;
    let ch = h / 2;
    let cx = (x / 2).min(cw.saturating_sub(1));
    let cy = (y / 2).min(ch.saturating_sub(1));
    let yv = planes.y[(y * w + x) as usize] as f32 / 255.0;
    let uv_idx = (cy * cw + cx) as usize;
    let uu = planes.u.get(uv_idx).copied().unwrap_or(128) as f32 / 255.0;
    let vv = planes.v.get(uv_idx).copied().unwrap_or(128) as f32 / 255.0;
    (yv, uu, vv)
}

/// Same as [`sample_yuv420p`] but for interleaved-UV NV12 planes.
fn sample_nv12(planes: &Nv12Planes<'_>, w: u32, h: u32, u: f64, v: f64) -> (f32, f32, f32) {
    let x = ((u * w as f64) as u32).min(w - 1);
    let y = ((v * h as f64) as u32).min(h - 1);
    let cw = w / 2;
    let ch = h / 2;
    let cx = (x / 2).min(cw.saturating_sub(1));
    let cy = (y / 2).min(ch.saturating_sub(1));
    let yv = planes.y[(y * w + x) as usize] as f32 / 255.0;
    let uv_idx = ((cy * cw + cx) * 2) as usize;
    let uu = planes.uv.get(uv_idx).copied().unwrap_or(128) as f32 / 255.0;
    let vv = planes.uv.get(uv_idx + 1).copied().unwrap_or(128) as f32 / 255.0;
    (yv, uu, vv)
}

/// Average Y/U/V over a coarse grid in the seam-adjacent band of one
/// camera's raw frame. `is_right` selects which edge is seam-adjacent:
/// the right plane fades in from its own left edge (`uv_x` near `0`, see
/// `fisheye.wgsl`'s `fs_main`), so it samples `[0, BAND_WIDTH]`; the left
/// plane sits geometrically adjacent to that seam along its own right
/// edge, so it samples `[1 - BAND_WIDTH, 1]`.
fn measure_band_mean(
    sample: impl Fn(f64, f64) -> Option<(f32, f32, f32)>,
    is_right: bool,
) -> Option<(f32, f32, f32)> {
    let (band_lo, band_hi) = if is_right {
        (0.0, BAND_WIDTH)
    } else {
        (1.0 - BAND_WIDTH, 1.0)
    };
    let mut sum = (0.0f64, 0.0f64, 0.0f64);
    let mut n = 0u32;
    for row in 0..GRID_ROWS {
        let uv_y = (row as f64 + 0.5) / GRID_ROWS as f64;
        for col in 0..GRID_COLS {
            let uv_x = band_lo + (band_hi - band_lo) * (col as f64 + 0.5) / GRID_COLS as f64;
            if let Some((y, u, v)) = sample(uv_x, uv_y) {
                sum.0 += y as f64;
                sum.1 += u as f64;
                sum.2 += v as f64;
                n += 1;
            }
        }
    }
    if n == 0 {
        return None;
    }
    Some((
        (sum.0 / n as f64) as f32,
        (sum.1 / n as f64) as f32,
        (sum.2 / n as f64) as f32,
    ))
}

/// EMA-smoothed, clamped per-camera YUV offset derived from periodic
/// seam-band measurements. `[0.0; 3]` (identity, matching the shader's
/// long-standing hardcoded default) until the first successful
/// measurement.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ColorMatchState {
    left_offset: [f32; 3],
    right_offset: [f32; 3],
    frames_since_measure: u64,
}

impl Default for ColorMatchState {
    fn default() -> Self {
        Self {
            left_offset: [0.0; 3],
            right_offset: [0.0; 3],
            frames_since_measure: INTERVAL_FRAMES, // measure on frame 0
        }
    }
}

impl ColorMatchState {
    /// Current per-camera YUV offset uniforms: `(left, right)`, each
    /// `[y, u, v]` in the shader's `color_offset_blend.xyz` convention.
    pub(crate) fn offsets(&self) -> ([f32; 3], [f32; 3]) {
        (self.left_offset, self.right_offset)
    }

    /// Re-measure and update the smoothed offsets if the interval has
    /// elapsed. `sample_left`/`sample_right` map a source UV to a Y/U/V
    /// triple (`None` outside the frame).
    fn update(
        &mut self,
        sample_left: impl Fn(f64, f64) -> Option<(f32, f32, f32)>,
        sample_right: impl Fn(f64, f64) -> Option<(f32, f32, f32)>,
    ) {
        self.frames_since_measure += 1;
        if self.frames_since_measure < INTERVAL_FRAMES {
            return;
        }
        self.frames_since_measure = 0;

        let (Some(left_mean), Some(right_mean)) = (
            measure_band_mean(sample_left, false),
            measure_band_mean(sample_right, true),
        ) else {
            return;
        };

        // Nudge both cameras toward their shared mean, split evenly, so
        // neither camera is treated as "the reference" - a static rig
        // where one camera happens to be correctly exposed would
        // otherwise get needlessly corrected too.
        let shared = (
            (left_mean.0 + right_mean.0) * 0.5,
            (left_mean.1 + right_mean.1) * 0.5,
            (left_mean.2 + right_mean.2) * 0.5,
        );
        let target_left = [
            (shared.0 - left_mean.0).clamp(-MAX_Y_OFFSET, MAX_Y_OFFSET),
            (shared.1 - left_mean.1).clamp(-MAX_CHROMA_OFFSET, MAX_CHROMA_OFFSET),
            (shared.2 - left_mean.2).clamp(-MAX_CHROMA_OFFSET, MAX_CHROMA_OFFSET),
        ];
        let target_right = [
            (shared.0 - right_mean.0).clamp(-MAX_Y_OFFSET, MAX_Y_OFFSET),
            (shared.1 - right_mean.1).clamp(-MAX_CHROMA_OFFSET, MAX_CHROMA_OFFSET),
            (shared.2 - right_mean.2).clamp(-MAX_CHROMA_OFFSET, MAX_CHROMA_OFFSET),
        ];
        for i in 0..3 {
            self.left_offset[i] += (target_left[i] - self.left_offset[i]) * EMA_ALPHA;
            self.right_offset[i] += (target_right[i] - self.right_offset[i]) * EMA_ALPHA;
        }
    }

    /// Re-measure from YUV420P source planes, if the interval has elapsed.
    pub(crate) fn update_yuv420p(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        left_cam: &Lens,
        right_cam: &Lens,
    ) {
        let (lw, lh) = (left_cam.width, left_cam.height);
        let (rw, rh) = (right_cam.width, right_cam.height);
        self.update(
            |u, v| {
                plane_uv_to_source_uv(u, v, left_cam)
                    .map(|(su, sv)| sample_yuv420p(left, lw, lh, su, sv))
            },
            |u, v| {
                plane_uv_to_source_uv(u, v, right_cam)
                    .map(|(su, sv)| sample_yuv420p(right, rw, rh, su, sv))
            },
        );
    }

    /// Re-measure from NV12 source planes, if the interval has elapsed.
    pub(crate) fn update_nv12(
        &mut self,
        left: &Nv12Planes<'_>,
        right: &Nv12Planes<'_>,
        left_cam: &Lens,
        right_cam: &Lens,
    ) {
        let (lw, lh) = (left_cam.width, left_cam.height);
        let (rw, rh) = (right_cam.width, right_cam.height);
        self.update(
            |u, v| {
                plane_uv_to_source_uv(u, v, left_cam)
                    .map(|(su, sv)| sample_nv12(left, lw, lh, su, sv))
            },
            |u, v| {
                plane_uv_to_source_uv(u, v, right_cam)
                    .map(|(su, sv)| sample_nv12(right, rw, rh, su, sv))
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Solid-color lens + plane pair so the measured band mean is exactly
    /// the fill value regardless of exactly which pixels the KB4 forward
    /// map lands on - isolates "does the correction move in the right
    /// direction by the right rough magnitude" from "is the UV mapping
    /// pixel-exact" (the latter is what the GPU agreement tests in
    /// `stitch::executor::tests` cover for the shared geometry primitive).
    fn solid_lens_and_planes(w: u32, h: u32, y_fill: u8) -> (Lens, Vec<u8>, Vec<u8>, Vec<u8>) {
        let lens = Lens::fisheye(
            w,
            h,
            w as f64 * 0.5,
            w as f64 * 0.5,
            w as f64 * 0.5,
            h as f64 * 0.5,
            [0.0, 0.0, 0.0, 0.0], // zero distortion: keeps the math simple
        );
        let y = vec![y_fill; (w * h) as usize];
        let u = vec![128u8; (w * h / 4) as usize];
        let v = vec![128u8; (w * h / 4) as usize];
        (lens, y, u, v)
    }

    #[test]
    fn identical_cameras_produce_no_correction() {
        let (lens_l, ly, lu, lv) = solid_lens_and_planes(64, 64, 120);
        let (lens_r, ry, ru, rv) = solid_lens_and_planes(64, 64, 120);
        let left = YuvPlanes {
            y: &ly,
            u: &lu,
            v: &lv,
        };
        let right = YuvPlanes {
            y: &ry,
            u: &ru,
            v: &rv,
        };

        let mut state = ColorMatchState::default();
        state.update_yuv420p(&left, &right, &lens_l, &lens_r);

        let (l, r) = state.offsets();
        assert_eq!(l, [0.0, 0.0, 0.0]);
        assert_eq!(r, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn mismatched_cameras_nudge_toward_each_other() {
        // Left is darker (Y=80/255), right is brighter (Y=170/255).
        let (lens_l, ly, lu, lv) = solid_lens_and_planes(64, 64, 80);
        let (lens_r, ry, ru, rv) = solid_lens_and_planes(64, 64, 170);
        let left = YuvPlanes {
            y: &ly,
            u: &lu,
            v: &lv,
        };
        let right = YuvPlanes {
            y: &ry,
            u: &ru,
            v: &rv,
        };

        let mut state = ColorMatchState::default();
        state.update_yuv420p(&left, &right, &lens_l, &lens_r);

        let (l, r) = state.offsets();
        // Left is darker than the shared mean -> its Y offset must be
        // positive (brighten it). Right is brighter -> negative (darken
        // it). Chroma is identical (128 both sides) -> stays at zero.
        assert!(l[0] > 0.0, "left Y offset should be positive, got {l:?}");
        assert!(r[0] < 0.0, "right Y offset should be negative, got {r:?}");
        assert!(
            (l[0] + r[0]).abs() < 1e-6,
            "offsets should be symmetric: {l:?} vs {r:?}"
        );
        assert_eq!(l[1], 0.0);
        assert_eq!(l[2], 0.0);
        assert_eq!(r[1], 0.0);
        assert_eq!(r[2], 0.0);
        // One EMA step (alpha=0.15) must not have already fully converged.
        assert!(l[0] < MAX_Y_OFFSET);
    }

    #[test]
    fn repeated_measurement_converges_toward_the_clamped_target() {
        let (lens_l, ly, lu, lv) = solid_lens_and_planes(64, 64, 40);
        let (lens_r, ry, ru, rv) = solid_lens_and_planes(64, 64, 220);
        let left = YuvPlanes {
            y: &ly,
            u: &lu,
            v: &lv,
        };
        let right = YuvPlanes {
            y: &ry,
            u: &ru,
            v: &rv,
        };

        let mut state = ColorMatchState::default();
        let mut prev = 0.0f32;
        for _ in 0..(INTERVAL_FRAMES * 40) {
            state.update_yuv420p(&left, &right, &lens_l, &lens_r);
            let (l, _) = state.offsets();
            assert!(
                l[0] >= prev - 1e-6,
                "offset should monotonically increase toward the clamp"
            );
            prev = l[0];
        }
        // A huge, sustained mismatch must saturate at the configured clamp,
        // not drift past it.
        assert!(
            (prev - MAX_Y_OFFSET).abs() < 1e-4,
            "expected convergence to {MAX_Y_OFFSET}, got {prev}"
        );
    }
}
