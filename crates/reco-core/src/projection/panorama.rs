//! Fixed cylindrical output over the existing calibrated stereo surfaces.
//!
//! This changes the output chart, not the fitted rig or its lens model. Missing
//! source support stays black; the chart never invents a complete rectangle.

use nalgebra::{Matrix4, Vector3, Vector4};
use serde::{Deserialize, Serialize};

use super::{LShape, ProjectionContext, ProjectionGeometry, SourceMapError};
use crate::calibration::{CalibrationError, Framing, Lens, expect_finite, expect_in_range};
use crate::geometry::{CameraIndex, VirtualCamera, YawPitch, view_matrix};
use crate::stitch::{BlendRule, SurfaceMap, SurfaceUv};

/// Full cylindrical panorama from two calibrated camera inputs.
///
/// For output NDC `(x, y)`, the view-space ray is
/// `(sin(x * arc/2), y * tan(vertical_fov/2), -cos(x * arc/2))`.
/// Framing tilt/roll orient this fixed chart. Runtime pan/zoom poses and output
/// aspect do not change its extent. Resolution only controls sampling density.
/// Use equal angular/pixel scale at the centre by choosing
/// `width/height = arc_radians / (2*tan(vertical_fov/2))`.
///
/// The per-camera mapping is invertible only on supported, monotonic lens
/// branches and positive ray-plane intersections. Blending, occlusion and
/// resampling are not invertible. No depth or parallax correction is performed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Panorama {
    /// Existing fitted L-shape geometry, unchanged by the output chart.
    pub rig: LShape,
    /// Horizontal extent in degrees, in `[1, 359]` (no wrap seam).
    pub arc_deg: f64,
    /// Vertical extent at every cylinder longitude, in degrees, in `[1, 179]`.
    pub vertical_fov_deg: f64,
}

impl Panorama {
    pub(crate) fn validate(&self, framing: &Framing) -> Result<(), CalibrationError> {
        self.rig.validate(framing)?;
        expect_finite("projection.arc_deg", self.arc_deg)?;
        expect_in_range("projection.arc_deg", self.arc_deg, 1.0, 359.0)?;
        expect_finite("projection.vertical_fov_deg", self.vertical_fov_deg)?;
        expect_in_range(
            "projection.vertical_fov_deg",
            self.vertical_fov_deg,
            1.0,
            179.0,
        )
    }

    fn maps(&self, lenses: &[Lens], framing: &Framing) -> [PanoramaMap; 2] {
        let camera = self.rig.virtual_camera(framing);
        let view = view_matrix(&camera, 0.0, 0.0, framing.tilt as f32, framing.roll as f32);
        let camera_to_world = view.cast::<f64>().try_inverse().expect("camera basis");
        let models = self.rig.plane_models([
            lenses[0].width as f64 / lenses[0].height as f64,
            lenses[1].width as f64 / lenses[1].height as f64,
        ]);
        std::array::from_fn(|i| {
            let to_plane = models[i].try_inverse().expect("rigid plane") * camera_to_world;
            PanoramaMap {
                to_plane,
                eye: (to_plane * Vector4::new(0.0, 0.0, 0.0, 1.0)).xyz(),
                optics: super::l_shape::NormalizedLens::new(&lenses[i]),
                half_arc: self.arc_deg.to_radians() * 0.5,
                tan_half_v: (self.vertical_fov_deg.to_radians() * 0.5).tan(),
            }
        })
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_uniforms(
        &self,
        lenses: &[Lens],
        framing: &Framing,
        flags: [[u32; 4]; 2],
    ) -> [PanoramaUniforms; 2] {
        let maps = self.maps(lenses, framing);
        std::array::from_fn(|i| {
            let m = &maps[i];
            let l = &lenses[i];
            PanoramaUniforms {
                to_plane: m.to_plane.cast::<f32>().into(),
                shape: [
                    m.half_arc as f32,
                    m.tan_half_v as f32,
                    self.rig.blend_width,
                    l.correction,
                ],
                intrinsics: [
                    (l.fx / l.width as f64) as f32,
                    (l.fy / l.height as f64) as f32,
                    (l.cx / l.width as f64) as f32,
                    (l.cy / l.height as f64) as f32,
                ],
                distortion: l.distortion.map(|v| v as f32),
                flags: flags[i],
                base_to_plane: maps[0].to_plane.cast::<f32>().into(),
                base_intrinsics: [
                    (lenses[0].fx / lenses[0].width as f64) as f32,
                    (lenses[0].fy / lenses[0].height as f64) as f32,
                    (lenses[0].cx / lenses[0].width as f64) as f32,
                    (lenses[0].cy / lenses[0].height as f64) as f32,
                ],
                base_distortion: lenses[0]
                    .distortion
                    .map(|v| (v * lenses[0].correction as f64) as f32),
            }
        })
    }
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PanoramaUniforms {
    to_plane: [[f32; 4]; 4],
    shape: [f32; 4],
    intrinsics: [f32; 4],
    distortion: [f32; 4],
    flags: [u32; 4],
    base_to_plane: [[f32; 4]; 4],
    base_intrinsics: [f32; 4],
    base_distortion: [f32; 4],
}

#[derive(Clone)]
struct PanoramaMap {
    to_plane: Matrix4<f64>,
    eye: Vector3<f64>,
    optics: super::l_shape::NormalizedLens,
    half_arc: f64,
    tan_half_v: f64,
}

impl SurfaceMap for PanoramaMap {
    fn sample_uv(&self, [x, y]: [f64; 2]) -> Option<SurfaceUv> {
        if !x.is_finite() || !y.is_finite() || x.abs() > 1.0 || y.abs() > 1.0 {
            return None;
        }
        let angle = x * self.half_arc;
        let ray = self.to_plane * Vector4::new(angle.sin(), y * self.tan_half_v, -angle.cos(), 0.0);
        let distance = -self.eye.z / ray.z;
        if !distance.is_finite() || distance <= 0.0 {
            return None;
        }
        self.optics
            .sample_plane([self.eye.x + distance * ray.x, self.eye.y + distance * ray.y])
    }

    fn covers_ndc(&self, _min: [f64; 2], _max: [f64; 2], _rotation_bound: f64) -> bool {
        // A fixed panorama deliberately retains unsupported borders. The
        // perspective coverage controller is not an output-extent solver.
        false
    }
}

// Only fade where there is an actual base sample. A valid right-only border
// must not fade into the clear colour, even with a vertically displaced rig.
struct PanoramaOverlay {
    right: PanoramaMap,
    base: PanoramaMap,
    blend_width: f64,
}
impl SurfaceMap for PanoramaOverlay {
    fn sample_uv(&self, ndc: [f64; 2]) -> Option<SurfaceUv> {
        let mut sample = self.right.sample_uv(ndc)?;
        if sample.edge < self.blend_width && self.base.sample_uv(ndc).is_none() {
            sample.edge = self.blend_width;
        }
        Some(sample)
    }
    fn covers_ndc(&self, _min: [f64; 2], _max: [f64; 2], _rotation_bound: f64) -> bool {
        false
    }
}

impl ProjectionGeometry for Panorama {
    fn name(&self) -> &'static str {
        "cylindrical-stereo-panorama"
    }
    fn camera_count(&self) -> usize {
        2
    }
    fn virtual_camera(&self, framing: &Framing) -> VirtualCamera {
        self.rig.virtual_camera(framing)
    }
    fn source_rays(
        &self,
        camera: CameraIndex,
        lenses: &[Lens],
        framing: &Framing,
    ) -> Result<super::SourceRays, SourceMapError> {
        self.rig.source_rays(camera, lenses, framing)
    }
    fn source_correspondence(
        &self,
        from: CameraIndex,
        to: CameraIndex,
        lenses: &[Lens],
        framing: &Framing,
    ) -> Result<super::SourceCorrespondence, SourceMapError> {
        self.rig.source_correspondence(from, to, lenses, framing)
    }
    fn map_source_point(
        &self,
        camera: CameraIndex,
        point: [f32; 2],
        lenses: &[Lens],
        framing: &Framing,
    ) -> Result<YawPitch, SourceMapError> {
        self.rig.map_source_point(camera, point, lenses, framing)
    }
    fn surface_maps(&self, ctx: &ProjectionContext) -> Vec<(Box<dyn SurfaceMap>, BlendRule)> {
        let [left, right] = self.maps(ctx.lenses, ctx.framing);
        let right = PanoramaOverlay {
            right,
            base: left.clone(),
            blend_width: self.rig.blend_width as f64,
        };
        vec![
            (Box::new(left), BlendRule::Opaque),
            (
                Box::new(right),
                BlendRule::Smoothstep(self.rig.blend_width as f64),
            ),
        ]
    }
    #[cfg(feature = "gpu")]
    fn gpu_program(&self) -> Option<crate::render::GpuProgram> {
        Some(crate::render::GpuProgram {
            wgsl: concat!(
                include_str!("../shaders/video_sampling.wgsl"),
                include_str!("../shaders/lens_mapping.wgsl"),
                include_str!("../shaders/panorama.wgsl")
            ),
            vs_entry: "vs_main",
            fs_entry: "fs_main",
            blend: wgpu::BlendState::ALPHA_BLENDING,
            pass: crate::render::GpuPass::Panorama,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{Calibration, Projection};
    use crate::geometry::RenderPose;
    use crate::render::viewport::ViewportSize;
    use crate::stitch::CpuExecutor;
    #[cfg(feature = "gpu")]
    use crate::stitch::StitchExecutor;

    fn calibration() -> Calibration {
        let mut cal = crate::stitch::test_support::calib(192, 108);
        let rig = cal.projection.l_shape_mut().unwrap();
        rig.x_rx = 0.024;
        rig.x_rz = -0.013;
        rig.z_rz = 0.017;
        rig.z_rx = 0.009;
        cal.framing.tilt = 0.11;
        cal.framing.roll = -0.07;
        cal.projection = Panorama {
            rig: rig.clone(),
            arc_deg: 240.0,
            vertical_fov_deg: 110.0,
        }
        .into();
        cal
    }

    #[test]
    fn panorama_document_validation_and_round_trip() {
        let cal = calibration();
        cal.validate().unwrap();
        let restored: Calibration = serde_json::from_str(&cal.to_json_pretty()).unwrap();
        assert_eq!(cal, restored);
        let mut bad = cal.clone();
        bad.lenses.pop();
        assert!(bad.validate().is_err());
        for (arc, vertical) in [
            (0.0, 90.0),
            (360.0, 90.0),
            (180.0, 0.0),
            (180.0, 180.0),
            (f64::NAN, 90.0),
            (180.0, f64::INFINITY),
        ] {
            let mut bad = cal.clone();
            let Projection::Panorama(p) = &mut bad.projection else {
                unreachable!()
            };
            p.arc_deg = arc;
            p.vertical_fov_deg = vertical;
            assert!(bad.validate().is_err());
        }
        let mut value = serde_json::to_value(&cal).unwrap();
        value["projection"]["typo"] = true.into();
        assert!(serde_json::from_value::<Calibration>(value).is_err());
    }

    #[test]
    fn panorama_supported_camera_domains_round_trip() {
        let mut cal = calibration();
        for correction in [0.0, 0.4, 1.0] {
            for lens in &mut cal.lenses {
                lens.correction = correction;
            }
            let Projection::Panorama(p) = &cal.projection else {
                unreachable!()
            };
            let maps = p.maps(&cal.lenses, &cal.framing);
            let camera = cal.virtual_camera();
            let view = view_matrix(
                &camera,
                0.0,
                0.0,
                cal.framing.tilt as f32,
                cal.framing.roll as f32,
            )
            .cast::<f64>();
            for (i, map) in maps.iter().enumerate() {
                let rays = p.source_rays(i, &cal.lenses, &cal.framing).unwrap();
                let mut supported = 0;
                for x in 1..20 {
                    for y in 1..20 {
                        let point = [x as f64 / 20.0, y as f64 / 20.0];
                        let Some(ray) = rays(point) else { continue };
                        let ray = view * ray.push(0.0);
                        let ndc = [
                            ray.x.atan2(-ray.z) / map.half_arc,
                            ray.y / ray.x.hypot(ray.z) / map.tan_half_v,
                        ];
                        if ndc.iter().any(|v| v.abs() > 1.0) {
                            continue;
                        }
                        let sample = map
                            .sample_uv(ndc)
                            .expect("supported forward ray must return to its source");
                        let error = (sample.u - point[0]).hypot(sample.v - point[1]);
                        assert!(
                            error < 1e-6,
                            "camera={i} correction={correction} source={point:?} normalized error={error}"
                        );
                        supported += 1;
                    }
                }
                assert!(supported > 200, "camera {i}: {supported}");
                assert!(map.sample_uv([f64::NAN, 0.0]).is_none());
                assert!(map.sample_uv([1.01, 0.0]).is_none());
            }
        }
    }

    #[test]
    fn field_retains_dominant_camera_from_rendered_panorama() {
        let mut cal = calibration();
        let size = ViewportSize {
            width: 320,
            height: 180,
        };
        let pose = RenderPose::default();
        let corners = [[0.4, 0.45], [0.6, 0.45], [0.6, 0.55], [0.4, 0.55]]
            .map(|p| cal.pick_source_point(&size, pose, p).unwrap());
        cal.field_roi = Some(crate::calibration::FieldRoi { corners });
        let boundary = cal.field_boundary().unwrap().unwrap();
        let mut checked = [0; 2];
        for y in 45..55 {
            for x in 40..60 {
                if let Some(p) =
                    cal.pick_source_point(&size, pose, [x as f64 / 100.0, y as f64 / 100.0])
                    && boundary.contains(p)
                {
                    assert!(boundary.owns(p), "panorama pixel ({x}, {y}), source {p:?}");
                    checked[p.camera] += 1;
                }
            }
        }
        assert!(checked.iter().all(|n| *n > 0), "{checked:?}");
    }

    #[test]
    fn panorama_fixed_extent_support_and_overlap() {
        let cal = calibration();
        let size = ViewportSize {
            width: 240,
            height: 110,
        };
        let maps = cal.surface_maps(&size, RenderPose::default());
        let changed = cal.surface_maps(
            &ViewportSize {
                width: 80,
                height: 200,
            },
            RenderPose {
                yaw: 1.0,
                pitch: -0.2,
                fov_degrees: 20.0,
            },
        );
        let mut counts = [0; 4];
        let mut blended = 0;
        for y in 0..size.height {
            for x in 0..size.width {
                let ndc = size.pixel_ndc(x, y);
                let left = maps[0].0.sample_uv(ndc);
                let right = maps[1].0.sample_uv(ndc);
                counts[usize::from(left.is_some()) + 2 * usize::from(right.is_some())] += 1;
                if let (None, Some(r)) = (left, right) {
                    assert_eq!(
                        maps[1].1.alpha(r.edge),
                        1.0,
                        "right-only support must not fade to black"
                    );
                }
                if let (Some(_), Some(r)) = (left, right) {
                    let alpha = maps[1].1.alpha(r.edge);
                    if alpha > 0.0 && alpha < 1.0 {
                        blended += 1;
                    }
                }
                for i in 0..2 {
                    assert_eq!(
                        maps[i].0.sample_uv(ndc).map(|s| [s.u, s.v]),
                        changed[i].0.sample_uv(ndc).map(|s| [s.u, s.v])
                    );
                }
            }
        }
        assert!(
            counts.iter().all(|&n| n > 0),
            "black/left/right/overlap counts: {counts:?}"
        );
        assert!(blended > 0);
        let executor = CpuExecutor::new(cal, size, vec![(192, 108); 2], false).unwrap();
        let mut core =
            crate::core::StitchCore::new(crate::stitch::Executor::Cpu(Box::new(executor))).unwrap();
        assert!(!core.constrained_look());
        assert!(core.prepare_coverage().is_err());
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn panorama_gpu_matches_cpu_support_seams_and_optics() {
        use crate::render::{planes::Nv12Planes, renderer::InputFormat};
        use crate::stitch::{GpuExecutor, GpuExecutorConfig};
        let Some(gpu) = crate::stitch::test_support::gpu_or_skip() else {
            return;
        };
        eprintln!("panorama oracle adapter: {}", gpu.gpu_name());
        let (y0, uv0) = crate::stitch::test_support::nv12(192, 108, 0);
        let (y1, uv1) = crate::stitch::test_support::nv12(192, 108, 45);
        let frames = [
            Nv12Planes { y: &y0, uv: &uv0 },
            Nv12Planes { y: &y1, uv: &uv1 },
        ];
        for (arc, vertical, blend, correction) in [
            (180.0, 70.0, 0.0, 1.0),
            (240.0, 110.0, 0.15, 0.4),
            (300.0, 150.0, 0.05, 0.0),
        ] {
            for full_range in [false, true] {
                let mut cal = calibration();
                let Projection::Panorama(p) = &mut cal.projection else {
                    unreachable!()
                };
                p.arc_deg = arc;
                p.vertical_fov_deg = vertical;
                p.rig.blend_width = blend;
                cal.lenses[1].fx *= 1.04;
                cal.lenses[1].cx += 2.0;
                for l in &mut cal.lenses {
                    l.correction = correction;
                }
                let size = ViewportSize {
                    width: 240,
                    height: 100,
                };
                let cpu =
                    CpuExecutor::new(cal.clone(), size.clone(), vec![(192, 108); 2], full_range)
                        .unwrap();
                let mut config =
                    GpuExecutorConfig::new(cal, vec![(192, 108); 2], InputFormat::Nv12);
                config.viewport_size = size;
                config.full_range = full_range;
                let mut render = GpuExecutor::new(gpu.clone(), config).unwrap();
                let expected = cpu.stitch_nv12(&frames, RenderPose::default()).unwrap();
                let actual = render.stitch(&frames, RenderPose::default()).unwrap();
                let max_error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap();
                assert!(
                    max_error <= 2,
                    "arc={arc} vertical={vertical} max error={max_error}"
                );
                assert!(render.stitch(&frames[..1], RenderPose::default()).is_err());
            }
        }
    }
}
