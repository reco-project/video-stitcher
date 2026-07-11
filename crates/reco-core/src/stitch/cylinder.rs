//! Inverse map for the mono cylindrical projection.
//!
//! The pre-stitched panorama is painted on the inside of a cylinder of
//! radius `focal_length`; the virtual camera sits on the cylinder axis
//! at the origin. Each output pixel casts a ray through the virtual
//! camera and intersects the cylinder; the hit's angle and height give
//! the video UV.
//!
//! The pan frame honours the calibration's rig orientation
//! ([`Framing`] tilt/roll) exactly like the L-shape does: yaw rotates
//! around the tilted+rolled up axis, so panning a tilted rig rolls
//! the rendered viewport progressively toward the edges.
//! SYNC_WITH: geometry/rig_correction.rs `rig_frame` - the frame
//! construction must match or the two topologies disagree on what a
//! calibrated tilt means.
//! SYNC_WITH: shaders/cylindrical_mono.wgsl - ray construction, the
//! theta sign, and the bounds discard; the CPU/GPU cylinder oracle
//! pins the agreement.

use crate::calibration::Framing;
use crate::projection::Cylinder;
use crate::render::viewport::ViewportConfig;
use crate::stitch::{SurfaceMap, SurfaceUv};

/// Rotate `v` around the unit axis `k` by `angle` (Rodrigues).
fn rotate(v: [f64; 3], k: [f64; 3], angle: f64) -> [f64; 3] {
    let (s, c) = angle.sin_cos();
    let kv = [
        k[1] * v[2] - k[2] * v[1],
        k[2] * v[0] - k[0] * v[2],
        k[0] * v[1] - k[1] * v[0],
    ];
    let kdv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
    [
        v[0] * c + kv[0] * s + k[0] * kdv * (1.0 - c),
        v[1] * c + kv[1] * s + k[1] * kdv * (1.0 - c),
        v[2] * c + kv[2] * s + k[2] * kdv * (1.0 - c),
    ]
}

/// Per-frame inverse map: output pixel -> pre-stitched panorama UV.
///
/// All quantities are precomputed f64 (the CPU side's precision
/// convention; the GPU runs the same math in f32 and the oracle
/// absorbs the difference).
pub(crate) struct CylinderMap {
    /// Rotated camera basis for ray construction: the ray for NDC
    /// `(x, y)` is `forward + right * x * tan_h + up * y * tan_v`.
    forward: [f64; 3],
    right: [f64; 3],
    up: [f64; 3],
    /// Half-frustum extents at unit distance: `tan(fov/2)` vertically,
    /// times the output aspect horizontally. They scale NDC into the
    /// ray basis above.
    tan_half_h: f64,
    tan_half_v: f64,
    /// Cylinder radius (world units) = `Cylinder::focal_length`:
    /// how far the painted surface sits from the axis camera.
    radius: f64,
    /// Full angular sweep in radians: a hit's azimuth inside
    /// `[-sweep/2, +sweep/2]` maps linearly to source U.
    sweep: f64,
    /// Half the painted height (world units): bounds the ray-cylinder
    /// hit vertically and maps linearly to source V.
    half_height: f64,
    /// Output dimensions, cast once here because `sample_uv` runs per
    /// output pixel.
    out_w: f64,
    out_h: f64,
}

impl CylinderMap {
    /// Build the map for one output frame at the given pose.
    ///
    /// The parameters split by lifetime: `topology` and `framing` are
    /// the calibrated document (static), `yaw`/`pitch` are the
    /// per-frame render pose, and the viewport FOV plus output
    /// dimensions ride in `config`. `source_height_px` backs
    /// `Cylinder::video_height`'s default.
    ///
    /// The mono camera basis looks along `-Z` with `+X` right and
    /// `+Y` up ([`VirtualCamera::mono`](crate::geometry::VirtualCamera::mono)),
    /// and the pose composition mirrors `view_matrix`: yaw around the
    /// rig frame's up axis, pitch around the yaw-rotated base right.
    pub fn new(
        topology: &Cylinder,
        framing: &Framing,
        source_height_px: f64,
        config: &ViewportConfig,
        yaw: f32,
        pitch: f32,
    ) -> Self {
        let base_forward = [0.0, 0.0, -1.0];
        let base_right = [1.0, 0.0, 0.0];
        let world_up = [0.0, 1.0, 0.0];

        // Rig frame (SYNC_WITH rig_correction::rig_frame): tilt rotates
        // forward + up around base right; roll rotates up around the
        // tilted forward, leaving it unchanged. The epsilon guards
        // skip the no-op rotations so a level rig keeps bit-exact
        // base axes.
        let mut f0 = base_forward;
        let mut u = world_up;
        if framing.tilt.abs() > 1e-9 {
            f0 = rotate(f0, base_right, framing.tilt);
            u = rotate(u, base_right, framing.tilt);
        }
        if framing.roll.abs() > 1e-9 {
            u = rotate(u, f0, -framing.roll);
        }

        // Pose (SYNC_WITH geometry::view_matrix): yaw around the rig
        // up, pitch around the yaw-rotated base right; the SCREEN axes
        // come from the rotated forward + up pair, exactly like the
        // look-at construction - deriving right from the pitch axis
        // would silently drop the rig roll.
        let (yaw, pitch) = (f64::from(yaw), f64::from(pitch));
        let pitch_axis = rotate(base_right, u, yaw);
        let forward = rotate(rotate(f0, u, yaw), pitch_axis, pitch);
        let up = rotate(rotate(u, u, yaw), pitch_axis, pitch);
        // f0 and u stay orthonormal through every rotation, so the
        // cross product is already unit length.
        let right = [
            forward[1] * up[2] - forward[2] * up[1],
            forward[2] * up[0] - forward[0] * up[2],
            forward[0] * up[1] - forward[1] * up[0],
        ];

        let tan_half_v = (f64::from(config.fov_degrees).to_radians() * 0.5).tan();
        let aspect = f64::from(config.width) / f64::from(config.height);

        Self {
            forward,
            right,
            up,
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
            self.forward[0] + self.right[0] * a + self.up[0] * b,
            self.forward[1] + self.right[1] * a + self.up[1] * b,
            self.forward[2] + self.right[2] * a + self.up[2] * b,
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

    fn level() -> Framing {
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        }
    }

    fn map(yaw: f32, pitch: f32) -> CylinderMap {
        CylinderMap::new(&Cylinder::default(), &level(), SRC_H, &cfg(), yaw, pitch)
    }

    /// Rig-frame map over a tall painted band, so tilted/rolled
    /// samples at pan edges stay inside the coverage.
    fn map_rig(tilt: f64, roll: f64, yaw: f32) -> CylinderMap {
        CylinderMap::new(
            &Cylinder {
                video_height: Some(20_000.0),
                ..Default::default()
            },
            &Framing {
                axis_offset: 0.0,
                tilt,
                roll,
            },
            SRC_H,
            &cfg(),
            yaw,
            0.0,
        )
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
        let t = Cylinder::default();
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
            &Cylinder {
                video_height: Some(100_000.0),
                ..Default::default()
            },
            &level(),
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
    fn rig_tilt_shifts_the_band_at_pan_center() {
        // At yaw 0 a tilted rig frame points the rest-forward up by
        // the tilt: the center pixel samples r*tan(t) above the video
        // center, exactly like pitching by t (band height 20k here).
        let tilted = map_rig(0.15, 0.0, 0.0);
        let s = tilted.sample_uv(100, 50).unwrap();
        let t = Cylinder::default();
        let expected = 0.5 - t.focal_length * (0.15f64).tan() / 20_000.0;
        assert!((s.v - expected).abs() < TOL, "v = {} vs {expected}", s.v);
    }

    #[test]
    fn rig_tilt_rolls_the_view_at_pan_edges() {
        // THE tilt signature: pan a tilted rig sideways and the
        // horizon rolls. Level rig at yaw 0.9: two pixels on the same
        // output row sample the same video height (symmetry). Tilted
        // rig: they diverge. Pixels sit at +-23 deg of horizontal FOV
        // so yaw + offset stays inside the 180-degree sweep.
        let level_v = {
            let m = map_rig(0.0, 0.0, 0.9);
            let l = m.sample_uv(60, 50).unwrap();
            let r = m.sample_uv(140, 50).unwrap();
            (l.v - r.v).abs()
        };
        let tilted_v = {
            let m = map_rig(0.15, 0.0, 0.9);
            let l = m.sample_uv(60, 50).unwrap();
            let r = m.sample_uv(140, 50).unwrap();
            (l.v - r.v).abs()
        };
        assert!(level_v < 5e-3, "level rig stays symmetric: {level_v}");
        assert!(
            tilted_v > 5e-3,
            "tilted rig must roll the view when panned: {tilted_v}"
        );
    }

    #[test]
    fn rig_roll_tilts_the_sampling_like_the_surface_roll() {
        // Rig roll = the painted surface rolled around the view axis
        // (the player's screen-tilt correction): off-center samples
        // displace vertically, the center pixel stays on the axis.
        let level_m = map_rig(0.0, 0.0, 0.0);
        let rolled = map_rig(0.0, 0.15, 0.0);
        let l = level_m.sample_uv(180, 50).unwrap();
        let r = rolled.sample_uv(180, 50).unwrap();
        assert!(
            (l.v - r.v).abs() > 1e-3,
            "roll must displace off-center samples: {} vs {}",
            l.v,
            r.v
        );
        let lc = level_m.sample_uv(100, 50).unwrap();
        let rc = rolled.sample_uv(100, 50).unwrap();
        assert!((lc.u - rc.u).abs() < TOL && (lc.v - rc.v).abs() < TOL);
    }
}
