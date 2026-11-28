//! The cylindrical projection: one pre-stitched panorama painted on
//! the inside of a cylinder, viewed from its axis.
//!
//! This module owns everything cylinder: the calibration parameters
//! (serialized inside the document's `projection` object), their
//! validation, and the CPU inverse map ([`CylinderMap`]).

use serde::{Deserialize, Serialize};

use crate::calibration::{
    CalibrationError, Framing, Lens, expect_finite, expect_in_range, expect_positive,
};
use crate::geometry::{CameraIndex, RenderPose, VirtualCamera};
use crate::projection::{ProjectionContext, ProjectionGeometry, SourceMapError};
use crate::render::viewport::ViewportSize;
use crate::stitch::{BlendRule, SurfaceMap, SurfaceUv};

// Functions, not constants: serde's `default = "..."` attribute takes
// a function path, never a value expression.
fn default_radius() -> f64 {
    2400.0
}

fn default_arc_deg() -> f64 {
    180.0
}

/// Cylinder placement for a single pre-stitched panorama: the video is
/// painted on the inside of a cylinder and the virtual camera sits on
/// its axis. Defaults match the established 180-degree
/// cylindrical-player convention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cylinder {
    /// Cylinder radius in the same units as the painted height. Their ratio sets the vertical angular scale; horizontal mapping depends only on `arc_deg`.
    #[serde(default = "default_radius")]
    pub radius: f64,
    /// Full horizontal arc in degrees. 180 is the canonical
    /// pre-stitched action-camera case; 360 is a full cylinder.
    #[serde(default = "default_arc_deg")]
    pub arc_deg: f64,
    /// Painted height in the same units as `radius`. Omitted = the calibration image's pixel height, not the current decoded frame height.
    #[serde(default)]
    pub surface_height: Option<f64>,
}

impl Default for Cylinder {
    fn default() -> Self {
        Self {
            radius: default_radius(),
            arc_deg: default_arc_deg(),
            surface_height: None,
        }
    }
}

impl Cylinder {
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_uniforms(
        &self,
        framing: &Framing,
        source_height: f64,
        aspect: f32,
        pose: RenderPose,
        flags: [u32; 4],
    ) -> CylinderUniforms {
        let map = CylinderMap::new(
            self,
            framing,
            source_height,
            &ViewportSize {
                width: 1,
                height: 1,
            },
            pose,
        );
        let vector = |v: [f64; 3], scale: f64| {
            [
                (v[0] * scale) as f32,
                (v[1] * scale) as f32,
                (v[2] * scale) as f32,
                0.0,
            ]
        };
        CylinderUniforms {
            forward: vector(map.forward, 1.0),
            right: vector(map.right, map.tan_half_v * f64::from(aspect)),
            up: vector(map.up, map.tan_half_v),
            shape: [
                (map.radius / (2.0 * map.half_height)) as f32,
                map.arc_rad as f32,
                0.0,
                0.0,
            ],
            flags,
        }
    }

    /// Validate the cylinder's parameters: positive finite
    /// radius/height, arc in `(0, 360]` degrees.
    pub(crate) fn validate(&self) -> Result<(), CalibrationError> {
        expect_finite("projection.radius", self.radius)?;
        // Same guard as lens focal lengths: the coverage math divides
        // by the radius.
        expect_positive("projection.radius", self.radius)?;

        expect_finite("projection.arc_deg", self.arc_deg)?;
        expect_positive("projection.arc_deg", self.arc_deg)?;
        expect_in_range("projection.arc_deg", self.arc_deg, 0.0, 360.0)?;

        // Omitted = the source pixel height, always valid.
        if let Some(height) = self.surface_height {
            expect_finite("projection.surface_height", height)?;
            expect_positive("projection.surface_height", height)?;
        }
        Ok(())
    }
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CylinderUniforms {
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    shape: [f32; 4],
    flags: [u32; 4],
}

impl ProjectionGeometry for Cylinder {
    fn source_rays(
        &self,
        camera: CameraIndex,
        lenses: &[Lens],
        _framing: &Framing,
    ) -> Result<super::SourceRays, SourceMapError> {
        let lens = lenses.get(camera).ok_or(SourceMapError::InvalidCamera)?;
        let height = self.surface_height.unwrap_or(f64::from(lens.height));
        let arc = self.arc_deg.to_radians();
        let radius = self.radius;
        Ok(Box::new(move |[u, v]| {
            let theta = (u - 0.5) * arc;
            nalgebra::Vector3::new(
                radius * theta.sin(),
                (0.5 - v) * height,
                -radius * theta.cos(),
            )
            .try_normalize(1e-12)
        }))
    }

    fn source_correspondence(
        &self,
        _from: CameraIndex,
        _to: CameraIndex,
        _lenses: &[Lens],
        _framing: &Framing,
    ) -> Result<super::SourceCorrespondence, SourceMapError> {
        Ok(super::SourceCorrespondence::new(Some))
    }
    fn name(&self) -> &'static str {
        "cylindrical-mono-1camera"
    }

    fn camera_count(&self) -> usize {
        1
    }

    fn virtual_camera(&self, _framing: &Framing) -> VirtualCamera {
        VirtualCamera::new(&[0.0, 0.0, 0.0], &[0.0, 0.0, -1.0])
    }

    fn map_source_point(
        &self,
        camera: CameraIndex,
        point: [f32; 2],
        lenses: &[Lens],
        _framing: &Framing,
    ) -> Result<crate::geometry::YawPitch, SourceMapError> {
        let lens = lenses.get(camera).ok_or(SourceMapError::InvalidCamera)?;
        let height = self.surface_height.unwrap_or(f64::from(lens.height));
        let theta = (f64::from(point[0]) - 0.5) * self.arc_deg.to_radians();
        let y = (0.5 - f64::from(point[1])) * height;
        // World-reference angles around the cylinder's origin. Rig tilt/roll
        // belongs to world_to_render_pose, not the painted surface coordinates.
        Ok(crate::geometry::YawPitch {
            yaw: -theta as f32,
            pitch: y.atan2(self.radius) as f32,
        })
    }

    fn surface_maps(&self, ctx: &ProjectionContext) -> Vec<(Box<dyn SurfaceMap>, BlendRule)> {
        vec![(
            Box::new(CylinderMap::new(
                self,
                ctx.framing,
                f64::from(ctx.lenses[0].height),
                ctx.viewport_size,
                ctx.pose,
            )),
            // Single surface: nothing underneath to blend with.
            BlendRule::Opaque,
        )]
    }

    #[cfg(feature = "gpu")]
    fn gpu_program(&self) -> Option<crate::render::GpuProgram> {
        Some(crate::render::GpuProgram {
            wgsl: concat!(
                include_str!("../shaders/video_sampling.wgsl"),
                include_str!("../shaders/cylinder.wgsl")
            ),
            vs_entry: "vs_main",
            fs_entry: "fs_main",
            blend: wgpu::BlendState::REPLACE,
            pass: crate::render::GpuPass::Cylinder,
        })
    }
}

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
/// The panorama is painted on the inside of a cylinder of radius
/// `radius`; the virtual camera sits on the cylinder axis at the
/// origin. Each output pixel casts a ray through the virtual camera
/// and intersects the cylinder; the hit's angle and height give the
/// video UV.
///
/// The pan frame honours the calibration's rig orientation
/// ([`Framing`] tilt/roll) exactly like the L-shape does: yaw rotates
/// around the tilted+rolled up axis, so panning a tilted rig rolls
/// the rendered viewport progressively toward the edges.
/// SYNC_WITH: geometry/rig_correction.rs `rig_frame` - the frame
/// construction must match or the two projections disagree on what a
/// calibrated tilt means.
/// Ray-basis quantities are precomputed in f64 for CPU sampling and GPU uniforms.
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
    /// Cylinder radius (world units) = `Cylinder::radius`:
    /// how far the painted surface sits from the axis camera.
    radius: f64,
    /// Full angular arc in radians: a hit's azimuth inside
    /// `[-arc_rad/2, +arc_rad/2]` maps linearly to source U.
    arc_rad: f64,
    /// Half the painted height (world units): bounds the ray-cylinder
    /// hit vertically and maps linearly to source V.
    half_height: f64,
}

impl CylinderMap {
    /// Build the map for one output frame at the given pose.
    ///
    /// The parameters split by lifetime: `cylinder` and `framing` are
    /// the calibrated document (static), `pose` is the per-frame
    /// render pose (pan + fov zoom), and the output dimensions ride
    /// in `viewport_size`. `source_height_px` backs `Cylinder::surface_height`'s
    /// default.
    ///
    /// The mono camera basis looks along `-Z` with `+X` right and
    /// `+Y` up, and the pose composition
    /// mirrors `view_matrix`: yaw around the rig frame's up axis,
    /// pitch around the yaw-rotated base right.
    fn new(
        cylinder: &Cylinder,
        framing: &Framing,
        source_height_px: f64,
        viewport_size: &ViewportSize,
        pose: RenderPose,
    ) -> Self {
        let camera = cylinder.virtual_camera(framing);
        let base_forward = [
            f64::from(camera.base_forward.x),
            f64::from(camera.base_forward.y),
            f64::from(camera.base_forward.z),
        ];
        let base_right = [
            f64::from(camera.base_right.x),
            f64::from(camera.base_right.y),
            f64::from(camera.base_right.z),
        ];
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

        // RenderPose (SYNC_WITH geometry::view_matrix): yaw around the rig
        // up, pitch around the yaw-rotated base right; the SCREEN axes
        // come from the rotated forward + up pair, exactly like the
        // look-at construction - deriving right from the pitch axis
        // would silently drop the rig roll.
        let (yaw, pitch) = (f64::from(pose.yaw), f64::from(pose.pitch));
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

        let tan_half_v = (f64::from(pose.render_fov()).to_radians() * 0.5).tan();
        let aspect = f64::from(viewport_size.width) / f64::from(viewport_size.height);

        Self {
            forward,
            right,
            up,
            tan_half_h: tan_half_v * aspect,
            tan_half_v,
            radius: cylinder.radius,
            arc_rad: cylinder.arc_deg.to_radians(),
            half_height: cylinder.surface_height.unwrap_or(source_height_px) * 0.5,
        }
    }
}

/// Calibration-constant cylinder state for allocation-free coverage sweeps.
pub(crate) struct PreparedCylinder {
    f0: [f64; 3],
    u: [f64; 3],
    base_right: [f64; 3],
    aspect: f64,
    radius: f64,
    arc_rad: f64,
    half_height: f64,
}

impl PreparedCylinder {
    pub(crate) fn new(
        cylinder: &Cylinder,
        framing: &Framing,
        source_height_px: f64,
        viewport_size: &ViewportSize,
    ) -> Self {
        let camera = cylinder.virtual_camera(framing);
        let base_forward = [
            f64::from(camera.base_forward.x),
            f64::from(camera.base_forward.y),
            f64::from(camera.base_forward.z),
        ];
        let base_right = [
            f64::from(camera.base_right.x),
            f64::from(camera.base_right.y),
            f64::from(camera.base_right.z),
        ];
        let mut f0 = base_forward;
        let mut u = [0.0, 1.0, 0.0];
        if framing.tilt.abs() > 1e-9 {
            f0 = rotate(f0, base_right, framing.tilt);
            u = rotate(u, base_right, framing.tilt);
        }
        if framing.roll.abs() > 1e-9 {
            u = rotate(u, f0, -framing.roll);
        }
        Self {
            f0,
            u,
            base_right,
            aspect: f64::from(viewport_size.width) / f64::from(viewport_size.height),
            radius: cylinder.radius,
            arc_rad: cylinder.arc_deg.to_radians(),
            half_height: cylinder.surface_height.unwrap_or(source_height_px) * 0.5,
        }
    }

    pub(crate) fn orientation(&self, yaw: f32, pitch: f32) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let (yaw, pitch) = (f64::from(yaw), f64::from(pitch));
        let pitch_axis = rotate(self.base_right, self.u, yaw);
        let forward = rotate(rotate(self.f0, self.u, yaw), pitch_axis, pitch);
        let up = rotate(rotate(self.u, self.u, yaw), pitch_axis, pitch);
        let right = [
            forward[1] * up[2] - forward[2] * up[1],
            forward[2] * up[0] - forward[0] * up[2],
            forward[0] * up[1] - forward[1] * up[0],
        ];
        (forward, right, up)
    }

    pub(crate) fn map_for_orientation(
        &self,
        (forward, right, up): ([f64; 3], [f64; 3], [f64; 3]),
        fov_degrees: f32,
    ) -> CylinderMap {
        let fov = RenderPose {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees,
        }
        .render_fov();
        let tan_half_v = (f64::from(fov).to_radians() * 0.5).tan();
        CylinderMap {
            forward,
            right,
            up,
            tan_half_h: tan_half_v * self.aspect,
            tan_half_v,
            radius: self.radius,
            arc_rad: self.arc_rad,
            half_height: self.half_height,
        }
    }
}

impl SurfaceMap for CylinderMap {
    fn covers_ndc(&self, min: [f64; 2], max: [f64; 2], rotation_bound: f64) -> bool {
        use super::interval::Interval as I;
        if !rotation_bound.is_finite() || !(0.0..=std::f64::consts::PI).contains(&rotation_bound) {
            return false;
        }
        let (x, y) = (I::new(min[0], max[0]), I::new(min[1], max[1]));
        let ray = |i: usize| {
            I::point(self.forward[i])
                + x * I::point(self.right[i] * self.tan_half_h)
                + y * I::point(self.up[i] * self.tan_half_v)
        };
        let (rx, ry, rz) = (ray(0), ray(1), ray(2));
        let horiz = (rx.square() + rz.square()).sqrt();
        let height = ry.abs_max() * self.radius;
        let limit = self.half_height * horiz.lo;
        if !horiz.finite()
            || !height.is_finite()
            || !limit.is_finite()
            || !(self.radius / horiz.lo).is_finite()
            || horiz.lo < 1e-9
            || height > limit - 1e-7
        {
            return false;
        }
        if self.arc_rad >= std::f64::consts::TAU {
            return true;
        }
        let (s, c) = (self.arc_rad * 0.5).sin_cos();
        let a = I::point(-s) * rz + I::point(c) * rx;
        let b = I::point(-s) * rz - I::point(c) * rx;
        if self.arc_rad <= std::f64::consts::PI {
            a.lo > 1e-7 && b.lo > 1e-7
        } else {
            a.lo > 1e-7 || b.lo > 1e-7
        }
    }

    fn sample_uv(&self, [ndc_x, ndc_y]: [f64; 2]) -> Option<SurfaceUv> {
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

        let u = 0.5 + theta / self.arc_rad;
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
    use crate::calibration::Calibration;
    use crate::geometry::WorldPose;

    #[test]
    fn source_mapping_round_trips_through_framed_render_surfaces() {
        for arc_deg in [1.0, 90.0, 180.0, 270.0, 360.0] {
            for surface_height in [None, Some(80.0)] {
                for (tilt, roll) in [(0.0, 0.0), (0.21, -0.15)] {
                    let cal = Calibration::new(
                        vec![Lens::flat(128, 32)],
                        Cylinder {
                            radius: 40.0,
                            arc_deg,
                            surface_height,
                        },
                        Framing {
                            axis_offset: 0.7,
                            tilt,
                            roll,
                        },
                    );
                    cal.validate().unwrap();
                    for u in [0.001, 0.1, 0.5, 0.9, 0.999] {
                        for v in [0.02, 0.3, 0.5, 0.98] {
                            let angle = cal.map_source_point(0, [u, v]).unwrap();
                            let pose = cal.world_to_render_pose(WorldPose {
                                yaw: angle.yaw,
                                pitch: angle.pitch,
                                fov_degrees: 30.0,
                            });
                            let maps = cal.surface_maps(
                                &ViewportSize {
                                    width: 1,
                                    height: 1,
                                },
                                pose,
                            );
                            let sample = maps[0].0.sample_uv([0.0, 0.0]).unwrap();
                            assert!(
                                (sample.u - f64::from(u)).abs() < 2e-5,
                                "u={} expected={u} arc={arc_deg} tilt={tilt} roll={roll}",
                                sample.u
                            );
                            assert!((sample.v - f64::from(v)).abs() < 2e-5);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn source_mapping_has_correct_scale_edges_and_full_circle_seam() {
        let mut cal = Calibration::new(
            vec![Lens::flat(128, 32)],
            Cylinder {
                radius: 16.0,
                arc_deg: 180.0,
                surface_height: None,
            },
            level(),
        );
        let left_top = cal.map_source_point(0, [0.0, 0.0]).unwrap();
        let right_bottom = cal.map_source_point(0, [1.0, 1.0]).unwrap();
        assert!((left_top.yaw - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((left_top.pitch - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
        assert_eq!(left_top.yaw, -right_bottom.yaw);
        assert_eq!(left_top.pitch, -right_bottom.pitch);
        cal.projection = Cylinder {
            radius: 16.0,
            arc_deg: 360.0,
            surface_height: Some(64.0),
        }
        .into();
        let a = cal.map_source_point(0, [0.0, 0.5]).unwrap();
        let b = cal.map_source_point(0, [1.0, 0.5]).unwrap();
        assert!((a.yaw - std::f32::consts::PI).abs() < 1e-6);
        assert!((b.yaw + std::f32::consts::PI).abs() < 1e-6);
        assert!((a.yaw - b.yaw).sin().abs() < 1e-6);
        assert!((cal.map_source_point(0, [0.5, 0.0]).unwrap().pitch - 2.0_f32.atan()).abs() < 1e-6);
        assert_eq!(
            cal.map_source_point(1, [0.5, 0.5]),
            Err(SourceMapError::InvalidCamera)
        );
        assert_eq!(
            cal.map_source_point(0, [f32::NAN, 0.5]),
            Err(SourceMapError::InvalidPoint)
        );
    }

    /// 1080px source; the defaults (r=2400px, 180-degree arc) give a
    /// vertical band of atan(540/2400) = +-0.221 rad of pitch.
    const SRC_H: f64 = 1080.0;

    fn cfg() -> ViewportSize {
        ViewportSize {
            width: 200,
            height: 100,
        }
    }

    /// The 60-degree test frustum, riding in the pose like production.
    fn pose(yaw: f32, pitch: f32) -> RenderPose {
        RenderPose {
            yaw,
            pitch,
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
        CylinderMap::new(
            &Cylinder::default(),
            &level(),
            SRC_H,
            &cfg(),
            pose(yaw, pitch),
        )
    }

    /// Rig-frame map over a tall painted band, so tilted/rolled
    /// samples at pan edges stay inside the coverage.
    fn map_rig(tilt: f64, roll: f64, yaw: f32) -> CylinderMap {
        CylinderMap::new(
            &Cylinder {
                surface_height: Some(20_000.0),
                ..Default::default()
            },
            &Framing {
                axis_offset: 0.0,
                tilt,
                roll,
            },
            SRC_H,
            &cfg(),
            pose(yaw, 0.0),
        )
    }

    /// Half-pixel slack: the output center pixel (100, 50) sits half a
    /// pixel off the exact optical axis, which at r=2400 is ~0.013 in v.
    const TOL: f64 = 2e-2;

    #[test]
    fn declared_camera_matches_the_sampling_origin_and_orientation() {
        let cylinder = Cylinder {
            surface_height: Some(20_000.0),
            ..Default::default()
        };
        for (tilt, roll) in [(0.0, 0.0), (0.21, -0.08)] {
            let framing = Framing {
                axis_offset: 0.7,
                tilt,
                roll,
            };
            let camera = cylinder.virtual_camera(&framing);
            assert_eq!(camera.eye, nalgebra::Point3::origin());
            assert_eq!(camera.base_forward, nalgebra::Vector3::new(0.0, 0.0, -1.0));
            for yaw in [-0.7, 0.0, 0.7] {
                for pitch in [-0.2, 0.0, 0.2] {
                    let map = CylinderMap::new(
                        &cylinder,
                        &framing,
                        SRC_H,
                        &ViewportSize {
                            width: 1,
                            height: 1,
                        },
                        pose(yaw, pitch),
                    );
                    let sample = map.sample_uv([0.0, 0.0]).expect("center ray covered");
                    let theta = (sample.u - 0.5) * cylinder.arc_deg.to_radians();
                    let hit = nalgebra::Vector4::new(
                        (cylinder.radius * theta.sin()) as f32,
                        ((0.5 - sample.v) * cylinder.surface_height.unwrap()) as f32,
                        (-cylinder.radius * theta.cos()) as f32,
                        1.0,
                    );
                    let view =
                        crate::geometry::view_matrix(&camera, yaw, pitch, tilt as f32, roll as f32);
                    let rendered = view * hit;
                    assert!(rendered.z < 0.0);
                    let error = rendered.x.hypot(rendered.y).atan2(-rendered.z);
                    assert!(
                        error < 1e-6,
                        "center ray error {error} at yaw={yaw} pitch={pitch} tilt={tilt} roll={roll}"
                    );
                }
            }
        }
    }

    #[test]
    fn center_pixel_at_zero_pose_samples_the_video_center() {
        let s = map(0.0, 0.0)
            .sample_uv(cfg().pixel_ndc(100, 50))
            .expect("center covered");
        assert!((s.u - 0.5).abs() < TOL, "u = {}", s.u);
        assert!((s.v - 0.5).abs() < TOL, "v = {}", s.v);
    }

    #[test]
    fn screen_right_samples_video_right() {
        // The un-mirrored invariant: at pose zero, the right side of
        // the output shows the right half of the panorama.
        let m = map(0.0, 0.0);
        let left_px = m.sample_uv(cfg().pixel_ndc(10, 50)).unwrap();
        let right_px = m.sample_uv(cfg().pixel_ndc(190, 50)).unwrap();
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
        // left half; the magnitude is exact (0.4 rad over a PI arc).
        let ahead = map(0.0, 0.0).sample_uv(cfg().pixel_ndc(100, 50)).unwrap();
        let panned = map(0.4, 0.0).sample_uv(cfg().pixel_ndc(100, 50)).unwrap();
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
        let up = map(0.0, 0.2).sample_uv(cfg().pixel_ndc(100, 50)).unwrap();
        // Center ray at pitch p hits at y = r*tan(p): v = 0.5 - r*tan(p)/h.
        let t = Cylinder::default();
        let expected = 0.5 - t.radius * (0.2f64).tan() / SRC_H;
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
                surface_height: Some(100_000.0),
                ..Default::default()
            },
            &level(),
            SRC_H,
            &cfg(),
            pose(0.0, 0.0),
        );
        let top = tall.sample_uv(cfg().pixel_ndc(100, 0)).unwrap();
        let bottom = tall.sample_uv(cfg().pixel_ndc(100, 99)).unwrap();
        assert!(
            top.v < bottom.v,
            "the top output row must sample the upper video region: {} vs {}",
            top.v,
            bottom.v
        );
    }

    #[test]
    fn rays_beyond_the_sweep_are_discarded() {
        // 180-degree arc: looking straight backward has no coverage.
        let s = map(std::f32::consts::PI, 0.0).sample_uv(cfg().pixel_ndc(100, 50));
        assert!(s.is_none(), "the back of the cylinder is unpainted");
    }

    #[test]
    fn rays_beyond_the_painted_height_are_discarded() {
        // Looking up past the band: pitch well beyond atan(540/2400).
        let s = map(0.0, 0.5).sample_uv(cfg().pixel_ndc(100, 50));
        assert!(s.is_none(), "above the painted band");
    }

    #[test]
    fn rig_tilt_shifts_the_band_at_pan_center() {
        // At yaw 0 a tilted rig frame points the rest-forward up by
        // the tilt: the center pixel samples r*tan(t) above the video
        // center, exactly like pitching by t (band height 20k here).
        let tilted = map_rig(0.15, 0.0, 0.0);
        let s = tilted.sample_uv(cfg().pixel_ndc(100, 50)).unwrap();
        let t = Cylinder::default();
        let expected = 0.5 - t.radius * (0.15f64).tan() / 20_000.0;
        assert!((s.v - expected).abs() < TOL, "v = {} vs {expected}", s.v);
    }

    #[test]
    fn rig_tilt_rolls_the_view_at_pan_edges() {
        // THE tilt signature: pan a tilted rig sideways and the
        // horizon rolls. Level rig at yaw 0.9: two pixels on the same
        // output row sample the same video height (symmetry). Tilted
        // rig: they diverge. Pixels sit at +-23 deg of horizontal FOV
        // so yaw + offset stays inside the 180-degree arc.
        let level_v = {
            let m = map_rig(0.0, 0.0, 0.9);
            let l = m.sample_uv(cfg().pixel_ndc(60, 50)).unwrap();
            let r = m.sample_uv(cfg().pixel_ndc(140, 50)).unwrap();
            (l.v - r.v).abs()
        };
        let tilted_v = {
            let m = map_rig(0.15, 0.0, 0.9);
            let l = m.sample_uv(cfg().pixel_ndc(60, 50)).unwrap();
            let r = m.sample_uv(cfg().pixel_ndc(140, 50)).unwrap();
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
        let l = level_m.sample_uv(cfg().pixel_ndc(180, 50)).unwrap();
        let r = rolled.sample_uv(cfg().pixel_ndc(180, 50)).unwrap();
        assert!(
            (l.v - r.v).abs() > 1e-3,
            "roll must displace off-center samples: {} vs {}",
            l.v,
            r.v
        );
        let lc = level_m.sample_uv(cfg().pixel_ndc(100, 50)).unwrap();
        let rc = rolled.sample_uv(cfg().pixel_ndc(100, 50)).unwrap();
        assert!((lc.u - rc.u).abs() < TOL && (lc.v - rc.v).abs() < TOL);
    }
}
