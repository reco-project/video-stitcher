//! Continuous viewport-side support. Enclosed interior gaps are outside this contract.

use super::*;

#[cfg(test)]
const SIDES: [([f64; 2], [f64; 2]); 4] = [
    ([-1.0, -1.0], [1.0, -1.0]),
    ([1.0, -1.0], [1.0, 1.0]),
    ([-1.0, 1.0], [1.0, 1.0]),
    ([-1.0, -1.0], [-1.0, 1.0]),
];

#[cfg(test)]
pub(super) fn pose(cal: &Calibration, size: &ViewportSize, pose: RenderPose) -> Certificate {
    if !valid_query(cal, size, pose) {
        return Certificate::Unknown;
    }
    let maps = cal.surface_maps(size, pose);
    certify_regions(&maps, &SIDES, 0.0, 2048)
}

// Each strip encloses one complete side over the orientation and FOV range.
// Checking only the perimeter of an enlarged rectangle would miss the sides
// that move through its interior, especially during a zoom.
fn strips(
    size: &ViewportSize,
    min_fov: f32,
    max_fov: f32,
    rotation: f64,
) -> Option<[([f64; 2], [f64; 2]); 4]> {
    let v = (f64::from(max_fov).to_radians() * 0.5).tan();
    let h = v * f64::from(size.aspect_ratio());
    let k = (f64::from(min_fov).to_radians() * 0.5).tan() / v;
    let e = 2.0 * (rotation * 0.5).sin() * (1.0 + h * h + v * v).sqrt();
    if e >= 1.0 || rotation > 0.5 {
        return None;
    }
    // A rotated ray (x,y,-1) changes each component by at most e. Divide by
    // its new depth to bound displacement in the reference view plane.
    let dx = e * (1.0 + h) / (h * (1.0 - e)) + 1e-6;
    let dy = e * (1.0 + v) / (v * (1.0 - e)) + 1e-6;
    Some([
        ([-1.0 - dx, -1.0 - dy], [1.0 + dx, -k + dy]),
        ([k - dx, -1.0 - dy], [1.0 + dx, 1.0 + dy]),
        ([-1.0 - dx, k - dy], [1.0 + dx, 1.0 + dy]),
        ([-1.0 - dx, -1.0 - dy], [-k + dx, 1.0 + dy]),
    ])
}

#[cfg(test)]
pub(super) fn enclosure(
    cal: &Calibration,
    size: &ViewportSize,
    lo: RenderPose,
    hi: RenderPose,
    budget: usize,
) -> Certificate {
    enclosure_impl(cal, size, lo, hi, budget, false)
}

// Excludes enclosed invalid components. Every exterior
// invalid path entering the enclosed rectangle must cross its continuous outline.
pub(super) fn exterior_enclosure(
    cal: &Calibration,
    size: &ViewportSize,
    lo: RenderPose,
    hi: RenderPose,
    budget: usize,
) -> Certificate {
    enclosure_impl(cal, size, lo, hi, budget, true)
}

fn outline(lo: [f64; 2], hi: [f64; 2]) -> [([f64; 2], [f64; 2]); 4] {
    [
        (lo, [hi[0], lo[1]]),
        ([hi[0], lo[1]], hi),
        ([lo[0], hi[1]], hi),
        (lo, [lo[0], hi[1]]),
    ]
}

fn enclosure_impl(
    cal: &Calibration,
    size: &ViewportSize,
    lo: RenderPose,
    hi: RenderPose,
    budget: usize,
    exterior_only: bool,
) -> Certificate {
    let rotation = f64::from((hi.yaw - lo.yaw).abs() + (hi.pitch - lo.pitch).abs()) * 0.5 + 2e-6;
    let min_fov = lo.fov_degrees.min(hi.fov_degrees);
    let max_fov = lo.fov_degrees.max(hi.fov_degrees);
    let Some(regions) = strips(size, min_fov, max_fov, rotation) else {
        return Certificate::Unknown;
    };
    let regions = if exterior_only {
        outline(regions[0].0, regions[2].1)
    } else {
        regions
    };
    let maps = cal.surface_maps(
        size,
        RenderPose {
            fov_degrees: max_fov,
            ..interpolate(lo, hi, 0.5)
        },
    );
    certify_regions(&maps, &regions, rotation, budget)
}

/// Calibration- and viewport-constant exterior enclosure evaluator for roof sweeps.
pub(super) enum PreparedExterior<'a> {
    LShape(Box<crate::projection::l_shape::PreparedLShape>),
    Cylinder(crate::projection::cylinder::PreparedCylinder),
    Generic(&'a Calibration, &'a ViewportSize),
}

pub(super) enum PreparedCell {
    LShape(crate::projection::l_shape::PreparedLShapeCell),
    Cylinder(([f64; 3], [f64; 3], [f64; 3])),
    Generic(f32, f32),
}

impl<'a> PreparedExterior<'a> {
    pub(super) fn new(cal: &'a Calibration, size: &'a ViewportSize) -> Self {
        match &cal.projection {
            crate::calibration::Projection::LShape(lshape) => {
                Self::LShape(Box::new(crate::projection::l_shape::PreparedLShape::new(
                    lshape,
                    &cal.lenses,
                    &cal.framing,
                    size,
                )))
            }
            crate::calibration::Projection::Cylinder(cylinder) => {
                Self::Cylinder(crate::projection::cylinder::PreparedCylinder::new(
                    cylinder,
                    &cal.framing,
                    f64::from(cal.lenses[0].height),
                    size,
                ))
            }
            _ => Self::Generic(cal, size),
        }
    }

    pub(super) fn prepare_cell(&self, mid_yaw: f32, mid_pitch: f32) -> PreparedCell {
        match self {
            Self::LShape(p) => PreparedCell::LShape(p.prepare_cell(mid_yaw, mid_pitch)),
            Self::Cylinder(p) => PreparedCell::Cylinder(p.orientation(mid_yaw, mid_pitch)),
            Self::Generic(_, _) => PreparedCell::Generic(mid_yaw, mid_pitch),
        }
    }

    pub(super) fn exterior_enclosure_cell(
        &self,
        cell: &PreparedCell,
        size: &ViewportSize,
        min_fov: f32,
        max_fov: f32,
        rotation: f64,
        budget: usize,
    ) -> bool {
        let Some(regions) = strips(size, min_fov, max_fov, rotation) else {
            return false;
        };
        let (lo, hi) = (regions[0].0, regions[2].1);
        match (self, cell) {
            (Self::LShape(p), PreparedCell::LShape(lcell)) => {
                let maps = p.plane_maps(lcell, max_fov);
                certify_outline_covered(&maps, lo, hi, rotation, budget)
            }
            (Self::Cylinder(p), PreparedCell::Cylinder(orientation)) => {
                let map = [p.map_for_orientation(*orientation, max_fov)];
                certify_outline_covered(&map, lo, hi, rotation, budget)
            }
            (Self::Generic(cal, size), PreparedCell::Generic(yaw, pitch)) => {
                let maps = cal.surface_maps(
                    size,
                    RenderPose {
                        yaw: *yaw,
                        pitch: *pitch,
                        fov_degrees: max_fov,
                    },
                );
                certify_outline_covered(&maps, lo, hi, rotation, budget)
            }
            _ => unreachable!("PreparedCell always matches PreparedExterior"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stitch::SurfaceUv;

    struct Gap {
        at: [f64; 2],
        radius: f64,
    }
    impl SurfaceMap for Gap {
        fn sample_uv(&self, p: [f64; 2]) -> Option<SurfaceUv> {
            ((p[0] - self.at[0]).abs() > self.radius || (p[1] - self.at[1]).abs() > self.radius)
                .then_some(SurfaceUv {
                    u: 0.5,
                    v: 0.5,
                    edge: 0.5,
                })
        }
        fn covers_ndc(&self, lo: [f64; 2], hi: [f64; 2], _: f64) -> bool {
            (0..2).any(|i| hi[i] < self.at[i] - self.radius || lo[i] > self.at[i] + self.radius)
        }
    }

    #[test]
    fn interior_gap_is_ignored_but_a_gap_between_side_samples_is_not() {
        let maps: Vec<(Box<dyn SurfaceMap>, BlendRule)> = vec![(
            Box::new(Gap {
                at: [0.0, 0.0],
                radius: 0.1,
            }),
            BlendRule::Opaque,
        )];
        assert_eq!(
            certify_regions(&maps, &SIDES, 0.0, 2048),
            Certificate::Covered
        );
        assert_eq!(certify(&maps, 0.0), Certificate::Uncovered);
        let maps: Vec<(Box<dyn SurfaceMap>, BlendRule)> = vec![(
            Box::new(Gap {
                at: [0.12345, 1.0],
                radius: 0.00001,
            }),
            BlendRule::Opaque,
        )];
        assert_eq!(
            certify_regions(&maps, &SIDES, 0.0, 2048),
            Certificate::Uncovered
        );
        assert_ne!(certify_regions(&maps, &SIDES, 0.0, 1), Certificate::Covered);
    }

    #[test]
    fn swept_sides_include_intermediate_zoom_not_only_outer_perimeter() {
        let size = ViewportSize {
            width: 97,
            height: 55,
        };
        let maps: Vec<(Box<dyn SurfaceMap>, BlendRule)> = vec![(
            Box::new(Gap {
                at: [0.0, 0.8],
                radius: 0.01,
            }),
            BlendRule::Opaque,
        )];
        assert_eq!(
            certify_regions(&maps, &SIDES, 0.0, 2048),
            Certificate::Covered
        );
        let regions = strips(&size, 30.0, 45.0, 0.0).unwrap();
        assert_eq!(
            certify_regions(&maps, &regions, 0.0, 2048),
            Certificate::Uncovered
        );
        // This gap is enclosed, so it is deliberately outside the cheaper contract.
        assert_eq!(
            certify_regions(&maps, &outline(regions[0].0, regions[2].1), 0.0, 2048),
            Certificate::Covered
        );
    }

    #[test]
    fn outline_detects_narrow_exterior_notches_between_corners() {
        // A rectangle reaching beyond the upper edge, not an enclosed hole.
        for x in [-0.73123, 0.12345, 0.67891] {
            for width in [1e-2, 1e-5, 1e-8] {
                struct Notch(f64, f64);
                impl SurfaceMap for Notch {
                    fn sample_uv(&self, p: [f64; 2]) -> Option<SurfaceUv> {
                        (p[1] < 0.8 || (p[0] - self.0).abs() > self.1).then_some(SurfaceUv {
                            u: 0.5,
                            v: 0.5,
                            edge: 0.5,
                        })
                    }
                    fn covers_ndc(&self, lo: [f64; 2], hi: [f64; 2], _: f64) -> bool {
                        hi[1] < 0.8 || hi[0] < self.0 - self.1 || lo[0] > self.0 + self.1
                    }
                }
                let maps: Vec<(Box<dyn SurfaceMap>, BlendRule)> =
                    vec![(Box::new(Notch(x, width)), BlendRule::Opaque)];
                assert_eq!(
                    certify_regions(&maps, &outline([-1.0; 2], [1.0; 2]), 0.0, 2048),
                    Certificate::Uncovered
                );
                assert_ne!(
                    certify_regions(&maps, &outline([-1.0; 2], [1.0; 2]), 0.0, 1),
                    Certificate::Covered
                );
            }
        }
    }

    #[test]
    fn exterior_enclosures_agree_with_rotated_and_zoomed_source_samples() {
        let cylinder = Calibration::new(
            vec![Lens::flat(3840, 2000)],
            Cylinder {
                radius: 1000.0,
                ..Default::default()
            },
            Framing {
                axis_offset: 0.0,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        let mut checked = 0;
        for mut cal in [stereo(), cylinder] {
            for (tilt, roll) in [(0.0, 0.0), (0.25, -0.3), (-0.3, 0.4)] {
                cal.framing.tilt = tilt;
                cal.framing.roll = roll;
                for (width, height) in [(160, 90), (90, 160), (160, 40)] {
                    let size = ViewportSize { width, height };
                    for i in 0..40 {
                        let from = RenderPose {
                            yaw: -0.9 + (i * 23 % 40) as f32 * 0.045,
                            pitch: -0.4 + (i * 13 % 40) as f32 * 0.02,
                            fov_degrees: 20.0,
                        };
                        let to = RenderPose {
                            yaw: from.yaw + 0.01,
                            pitch: from.pitch + 0.01,
                            fov_degrees: 45.0,
                        };
                        if exterior_enclosure(&cal, &size, from, to, 2048) != Certificate::Covered {
                            continue;
                        }
                        for (dx, dy) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.5)]
                        {
                            for fov in [20.0, 32.5, 45.0] {
                                let pose = RenderPose {
                                    yaw: from.yaw + dx * 0.01,
                                    pitch: from.pitch + dy * 0.01,
                                    fov_degrees: fov,
                                };
                                let maps = cal.surface_maps(&size, pose);
                                assert_eq!(exhaustive(&maps, &size), 0, "{pose:?}");
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 100);
    }

    #[test]
    fn strips_contain_rotated_sides_at_every_intermediate_fov() {
        use nalgebra::Vector4;
        for (width, height) in [(1920, 1080), (1080, 1920), (800, 100)] {
            let size = ViewportSize { width, height };
            for (tilt, roll) in [(0.0, 0.0), (0.4, -0.3), (-0.7, 0.5)] {
                let mut cal = stereo();
                cal.framing.tilt = tilt;
                cal.framing.roll = roll;
                let camera = cal.projection.geometry().virtual_camera(&cal.framing);
                let view = |p: RenderPose| {
                    crate::geometry::view_matrix(&camera, p.yaw, p.pitch, tilt as f32, roll as f32)
                };
                for (fov_a, fov_b) in [(25.0, 60.0), (60.0, 25.0), (60.0, 60.0)] {
                    let from = RenderPose {
                        yaw: -0.6,
                        pitch: 0.25,
                        fov_degrees: fov_a,
                    };
                    let to = RenderPose {
                        yaw: -0.52,
                        pitch: 0.21,
                        fov_degrees: fov_b,
                    };
                    let rotation =
                        f64::from((to.yaw - from.yaw).abs() + (to.pitch - from.pitch).abs()) * 0.5
                            + 2e-6;
                    let max_fov = fov_a.max(fov_b);
                    let regions = strips(&size, fov_a.min(fov_b), max_fov, rotation).unwrap();
                    let reference = view(interpolate(from, to, 0.5));
                    let v_ref = (max_fov.to_radians() * 0.5).tan();
                    for i in 0..=16 {
                        let p = interpolate(from, to, i as f32 / 16.0);
                        let transform = reference * view(p).try_inverse().unwrap();
                        let v = (p.fov_degrees.to_radians() * 0.5).tan();
                        for ((lo, hi), (start, end)) in regions.iter().zip(SIDES) {
                            for j in 0..=64 {
                                let t = j as f64 / 64.0;
                                let x = start[0] + t * (end[0] - start[0]);
                                let y = start[1] + t * (end[1] - start[1]);
                                let q = transform
                                    * Vector4::new(
                                        x as f32 * v * size.aspect_ratio(),
                                        y as f32 * v,
                                        -1.0,
                                        0.0,
                                    );
                                let ndc = [
                                    f64::from(q.x / (-q.z * v_ref * size.aspect_ratio())),
                                    f64::from(q.y / (-q.z * v_ref)),
                                ];
                                assert!(
                                    (0..2)
                                        .all(|axis| ndc[axis] >= lo[axis] && ndc[axis] <= hi[axis]),
                                    "{ndc:?} outside {lo:?}..{hi:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
