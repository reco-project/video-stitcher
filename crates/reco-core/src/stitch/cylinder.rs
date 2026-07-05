//! Inverse map for the mono cylindrical projection.
//!
//! The pre-stitched panorama is painted on the inside of a cylinder of
//! radius `focal_length`; the virtual camera sits on the cylinder axis
//! at the origin. Each output pixel casts a ray through the virtual
//! camera and intersects the cylinder; the hit's angle and height give
//! the video UV.
//!
//! SYNC_WITH: shaders/cylindrical_mono.wgsl - ray construction (top
//! row of the output looks up), the screen-rotation fold, the
//! theta-from-forward convention (straight ahead samples u = 0.5, u
//! grows with yaw), and the bounds discard must match; the CPU/GPU
//! cylinder oracle pins the agreement.

use crate::calibration::CylinderTopology;
use crate::render::viewport::ViewportConfig;
use crate::stitch::{SurfaceMap, SurfaceUv};

/// Per-frame inverse map: output pixel -> pre-stitched panorama UV.
///
/// All quantities are precomputed f64 (the CPU side's precision
/// convention; the GPU runs the same math in f32 and the oracle
/// absorbs the difference).
pub(crate) struct CylinderMap {
    /// Rotated camera basis scaled for ray construction: the ray for
    /// NDC `(x, y)` is `forward + right_eff * x * tan_h + up_eff * y * tan_v`.
    forward: [f64; 3],
    right_eff: [f64; 3],
    up_eff: [f64; 3],
    tan_half_h: f64,
    tan_half_v: f64,
    /// Cylinder radius (world units).
    radius: f64,
    /// Full angular sweep in radians.
    sweep: f64,
    /// Half the painted height (world units).
    half_height: f64,
    out_w: f64,
    out_h: f64,
}

impl CylinderMap {
    /// Build the map for one output frame at the given pose.
    ///
    /// The yaw/pitch convention is [`VirtualCamera`]'s
    /// (`yaw_pitch_to_direction`): the mono camera basis looks along
    /// `-Z` with `+X` right and `+Y` up. Positive yaw turns toward
    /// `-X` (VirtualCamera's sense), i.e. toward the video's left
    /// half - what matters is that screen-right samples video-right.
    ///
    /// [`VirtualCamera`]: crate::geometry::VirtualCamera
    pub fn new(
        topology: &CylinderTopology,
        source_height_px: f64,
        config: &ViewportConfig,
        yaw: f32,
        pitch: f32,
    ) -> Self {
        let (yaw, pitch) = (yaw as f64, pitch as f64);
        let (sin_y, cos_y) = yaw.sin_cos();
        let (sin_p, cos_p) = pitch.sin_cos();

        // dir = base_forward*(cosP*cosY) - base_right*(cosP*sinY) + up*sinP
        // with base_forward = -Z, base_right = +X (SYNC_WITH
        // VirtualCamera::yaw_pitch_to_direction).
        let forward = [-cos_p * sin_y, sin_p, -cos_p * cos_y];
        // The camera's right stays horizontal under pitch; up completes
        // the right-handed basis (up = right x forward).
        let right = [cos_y, 0.0, -sin_y];
        let up = [
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
        ];

        // Fold the screen rotation (tilt around the view axis) into the
        // basis: offsets (a, b) rotate to (a*cr - b*sr, a*sr + b*cr).
        let (sr, cr) = topology.screen_rotation_deg.to_radians().sin_cos();
        let right_eff = [
            right[0] * cr + up[0] * sr,
            right[1] * cr + up[1] * sr,
            right[2] * cr + up[2] * sr,
        ];
        let up_eff = [
            up[0] * cr - right[0] * sr,
            up[1] * cr - right[1] * sr,
            up[2] * cr - right[2] * sr,
        ];

        let tan_half_v = (f64::from(config.fov_degrees).to_radians() * 0.5).tan();
        let aspect = f64::from(config.width) / f64::from(config.height);

        Self {
            forward,
            right_eff,
            up_eff,
            tan_half_h: tan_half_v * aspect,
            tan_half_v,
            radius: topology.focal_length,
            sweep: topology.sweep_deg.to_radians(),
            half_height: topology.video_height.unwrap_or(source_height_px) * 0.5,
            out_w: f64::from(config.width),
            out_h: f64::from(config.height),
        }
    }
}

impl SurfaceMap for CylinderMap {
    fn sample_uv(&self, out_x: u32, out_y: u32) -> Option<SurfaceUv> {
        // Pixel centre -> NDC, +Y up (the top output row looks up).
        let ndc_x = (f64::from(out_x) + 0.5) / self.out_w * 2.0 - 1.0;
        let ndc_y = 1.0 - (f64::from(out_y) + 0.5) / self.out_h * 2.0;

        let a = ndc_x * self.tan_half_h;
        let b = ndc_y * self.tan_half_v;
        let ray = [
            self.forward[0] + self.right_eff[0] * a + self.up_eff[0] * b,
            self.forward[1] + self.right_eff[1] * a + self.up_eff[1] * b,
            self.forward[2] + self.right_eff[2] * a + self.up_eff[2] * b,
        ];

        // Intersect with the cylinder x^2 + z^2 = r^2 (camera on the
        // axis, so the positive hit is r over the ray's horizontal
        // reach). A near-vertical ray never meets the wall.
        let horiz = (ray[0] * ray[0] + ray[2] * ray[2]).sqrt();
        if horiz < 1e-9 {
            return None;
        }
        let t = self.radius / horiz;
        let hit_y = ray[1] * t;

        // Angle from the world forward axis (-Z), positive toward
        // screen-right (+X at pose zero): straight ahead is 0 and the
        // right half of the output samples the right half of the
        // video. VirtualCamera's positive yaw turns toward -X, so
        // panning right means decreasing yaw - the sign never
        // surfaces (coverage is symmetric and panners work in the
        // same basis), but a mismatch here mirrors the image.
        let theta = ray[0].atan2(-ray[2]);

        let u = 0.5 + theta / self.sweep;
        let v = 0.5 - hit_y / (self.half_height * 2.0);
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }

        Some(SurfaceUv {
            u,
            v,
            // Single opaque surface: the blend edge is unused.
            edge: 1.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1080px source; the defaults (r=2400px, 180-deg sweep) give a
    /// vertical band of atan(540/2400) = +-0.221 rad of pitch.
    const SRC_H: f64 = 1080.0;

    fn cfg() -> ViewportConfig {
        ViewportConfig {
            width: 200,
            height: 100,
            fov_degrees: 60.0,
        }
    }

    fn map(yaw: f32, pitch: f32) -> CylinderMap {
        CylinderMap::new(&CylinderTopology::default(), SRC_H, &cfg(), yaw, pitch)
    }

    /// Half-pixel slack: the output center pixel (100, 50) sits half a
    /// pixel off the exact optical axis, which at r=2400 is ~0.013 in v.
    const TOL: f64 = 2e-2;

    #[test]
    fn center_pixel_at_zero_pose_samples_the_video_center() {
        let s = map(0.0, 0.0).sample_uv(100, 50).expect("center covered");
        assert!((s.u - 0.5).abs() < TOL, "u = {}", s.u);
        assert!((s.v - 0.5).abs() < TOL, "v = {}", s.v);
    }

    #[test]
    fn screen_right_samples_video_right() {
        // The un-mirrored invariant: at pose zero, the right side of
        // the output shows the right half of the panorama.
        let m = map(0.0, 0.0);
        let left_px = m.sample_uv(10, 50).unwrap();
        let right_px = m.sample_uv(190, 50).unwrap();
        assert!(
            left_px.u < 0.5 && right_px.u > 0.5,
            "screen left/right must sample video left/right: {} / {}",
            left_px.u,
            right_px.u
        );
    }

    #[test]
    fn yaw_follows_the_virtual_camera_sense() {
        // VirtualCamera's positive yaw turns toward -X = the video's
        // left half; the magnitude is exact (0.4 rad over a PI sweep).
        let ahead = map(0.0, 0.0).sample_uv(100, 50).unwrap();
        let panned = map(0.4, 0.0).sample_uv(100, 50).unwrap();
        assert!(
            panned.u < ahead.u - 0.05,
            "yaw +0.4 turns toward the video's left: {} -> {}",
            ahead.u,
            panned.u
        );
        let expected = 0.5 - 0.4 / std::f64::consts::PI;
        assert!((panned.u - expected).abs() < 5e-3, "u = {}", panned.u);
    }

    #[test]
    fn positive_pitch_looks_up_toward_lower_v() {
        let up = map(0.0, 0.2).sample_uv(100, 50).unwrap();
        // Center ray at pitch p hits at y = r*tan(p): v = 0.5 - r*tan(p)/h.
        let t = CylinderTopology::default();
        let expected = 0.5 - t.focal_length * (0.2f64).tan() / SRC_H;
        assert!(
            up.v < 0.5 && (up.v - expected).abs() < TOL,
            "v = {} vs {expected}",
            up.v
        );
    }

    #[test]
    fn top_output_row_samples_above_the_bottom_row() {
        // A tall painted band so both extreme rows land inside it.
        let tall = CylinderMap::new(
            &CylinderTopology {
                video_height: Some(100_000.0),
                ..Default::default()
            },
            SRC_H,
            &cfg(),
            0.0,
            0.0,
        );
        let top = tall.sample_uv(100, 0).unwrap();
        let bottom = tall.sample_uv(100, 99).unwrap();
        assert!(
            top.v < bottom.v,
            "the top output row must sample the upper video region: {} vs {}",
            top.v,
            bottom.v
        );
    }

    #[test]
    fn rays_beyond_the_sweep_are_discarded() {
        // 180-degree sweep: looking straight backward has no coverage.
        let s = map(std::f32::consts::PI, 0.0).sample_uv(100, 50);
        assert!(s.is_none(), "the back of the cylinder is unpainted");
    }

    #[test]
    fn rays_beyond_the_painted_height_are_discarded() {
        // Looking up past the band: pitch well beyond atan(540/2400).
        let s = map(0.0, 0.5).sample_uv(100, 50);
        assert!(s.is_none(), "above the painted band");
    }

    #[test]
    fn screen_rotation_tilts_the_sampling() {
        let level = map(0.0, 0.0);
        let tilted = CylinderMap::new(
            &CylinderTopology {
                screen_rotation_deg: 10.0,
                ..Default::default()
            },
            SRC_H,
            &cfg(),
            0.0,
            0.0,
        );
        // Off-center horizontally: a tilt shifts its vertical sample.
        let l = level.sample_uv(180, 50).unwrap();
        let t = tilted.sample_uv(180, 50).unwrap();
        assert!(
            (l.v - t.v).abs() > 1e-3,
            "screen rotation must displace off-center samples: {} vs {}",
            l.v,
            t.v
        );
        // The center pixel stays put (it lies on the rotation axis).
        let lc = level.sample_uv(100, 50).unwrap();
        let tc = tilted.sample_uv(100, 50).unwrap();
        assert!((lc.u - tc.u).abs() < TOL && (lc.v - tc.v).abs() < TOL);
    }
}
