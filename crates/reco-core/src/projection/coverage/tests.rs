//! Numerical and behavioral checks for the coverage candidate.

use super::*;

#[test]
fn spatial_refinement_is_bounded_and_never_uses_samples_as_proof() {
    use crate::stitch::SurfaceUv;
    use std::{cell::Cell, rc::Rc};

    struct RefinedSheet {
        leaf_width: f64,
        hole: bool,
        queries: Rc<Cell<usize>>,
    }
    impl SurfaceMap for RefinedSheet {
        fn sample_uv(&self, p: [f64; 2]) -> Option<SurfaceUv> {
            (!(self.hole && p[0].abs() < 0.01 && p[1].abs() < 0.01)).then_some(SurfaceUv {
                u: 0.5,
                v: 0.5,
                edge: 0.5,
            })
        }
        fn covers_ndc(&self, lo: [f64; 2], hi: [f64; 2], _: f64) -> bool {
            self.queries.set(self.queries.get() + 1);
            !self.hole && (0..2).all(|i| hi[i] - lo[i] <= self.leaf_width)
        }
    }
    let queries = Rc::new(Cell::new(0));
    let maps = |leaf_width, hole| {
        vec![(
            Box::new(RefinedSheet {
                leaf_width,
                hole,
                queries: queries.clone(),
            }) as Box<dyn SurfaceMap>,
            BlendRule::Opaque,
        )]
    };
    // 64 x 64 leaves require 8,191 nodes in this binary subdivision.
    for budget in [0, 1, 2048, 8190, 8191, 8192] {
        queries.set(0);
        let result = certify_rect(&maps(1.0 / 32.0, false), [-1.0; 2], [1.0; 2], 0.0, budget);
        assert_eq!(
            result,
            if budget < 8191 {
                Certificate::Unknown
            } else {
                Certificate::Covered
            }
        );
        assert_eq!(queries.get(), budget.min(8191));
    }
    queries.set(0);
    assert_eq!(
        certify_rect(&maps(2.0, false), [-1.0; 2], [1.0; 2], 0.0, 8192),
        Certificate::Covered
    );
    assert_eq!(
        queries.get(),
        1,
        "easy regions do not spend the larger allowance"
    );
    assert_eq!(
        certify_rect(&maps(0.0, false), [-1.0; 2], [1.0; 2], 0.0, 8192),
        Certificate::Unknown
    );
    assert_eq!(
        certify_rect(&maps(2.0, true), [-1.0; 2], [1.0; 2], 0.0, 8192),
        Certificate::Uncovered
    );
}

#[test]
fn narrow_cylinder_fit_matches_perspective_limit_and_keeps_orientation() {
    let cal = Calibration::new(
        vec![Lens::flat(3840, 2160)],
        Cylinder {
            sweep_deg: 30.0,
            ..Default::default()
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    for (width, height) in [(65, 37), (37, 65), (400, 20)] {
        let size = ViewportSize { width, height };
        let request = Pose {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: 100.0,
        };
        let expected = 2.0
            * ((15_f64.to_radians().tan() / size.aspect_ratio() as f64).min(0.45))
                .atan()
                .to_degrees();
        let result = fit_fov(&cal, &size, request).unwrap();
        assert_eq!((result.yaw, result.pitch), (request.yaw, request.pitch));
        assert!(
            (f64::from(result.fov_degrees) - expected).abs() < 0.003,
            "{size:?}: {result:?}, expected={expected}"
        );
        let missing = fit_fov(
            &cal,
            &size,
            Pose {
                yaw: 2.0,
                ..request
            },
        );
        assert!(
            missing.is_none(),
            "must not invent a fit facing the unpainted side"
        );
    }
}

#[test]
fn overflowing_cylinder_bounds_are_not_a_certificate() {
    let size = ViewportSize {
        width: 65,
        height: 37,
    };
    let cal = Calibration::new(
        vec![Lens::flat(3840, 2160)],
        Cylinder {
            focal_length: 1e308,
            video_height: Some(1e308),
            sweep_deg: 360.0,
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let maps = cal.surface_maps(
        &size,
        Pose {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: 170.0,
        },
    );
    assert!(!maps[0].0.covers_ndc([-1.0; 2], [1.0; 2], 0.0));
}

#[test]
fn continuous_certificates_never_accept_missing_pixels() {
    let mut cal = stereo();
    let size = ViewportSize {
        width: 129,
        height: 73,
    };
    let mut seed = 41_u64;
    let mut random = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 32) as u32) as f64 / u32::MAX as f64
    };
    let mut accepted = 0;
    let mut unknown = 0;
    for _ in 0..1200 {
        for lens in &mut cal.lenses {
            lens.correction = random() as f32;
            lens.fx = 1200.0 + 1800.0 * random();
            lens.fy = 1200.0 + 1800.0 * random();
            lens.cx = 1700.0 + 400.0 * random();
            lens.cy = 850.0 + 350.0 * random();
        }
        cal.framing.tilt = (random() - 0.5) * 0.8;
        cal.framing.roll = (random() - 0.5) * 0.8;
        let topo = cal.topology.l_shape_mut().unwrap();
        topo.x_rx = (random() - 0.5) * 0.2;
        topo.z_rz = (random() - 0.5) * 0.2;
        topo.x_ty = (random() - 0.5) * 0.08;
        let pose = Pose {
            yaw: ((random() - 0.5) * 3.0) as f32,
            pitch: ((random() - 0.5) * 0.8) as f32,
            fov_degrees: (10.0 + 60.0 * random()) as f32,
        };
        let maps = cal.surface_maps(&size, pose);
        match certify(&maps, 0.0) {
            Certificate::Covered => {
                assert_eq!(exhaustive(&maps, &size), 0, "{pose:?} {cal:?}");
                accepted += 1;
            }
            Certificate::Unknown => unknown += 1,
            Certificate::Uncovered => {}
        }
    }
    eprintln!("random viewports: certified={accepted}, unknown={unknown}, total=1200");
    assert!(accepted > 100);
}

#[test]
fn extreme_lenses_and_placements_cannot_invent_support() {
    let size = ViewportSize {
        width: 97,
        height: 55,
    };
    let mut seed = 317_u64;
    let mut random = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 32) as u32) as f64 / u32::MAX as f64
    };
    let mut accepted = 0;
    for i in 0..2400 {
        let mut cal = stereo();
        cal.framing.axis_offset =
            [0.001, 0.006, 0.012, 0.02, 0.1, 0.25, 0.6, 1.0, 3.0, 6.0][i % 10];
        cal.framing.tilt = (random() - 0.5) * 3.0;
        cal.framing.roll = (random() - 0.5) * 3.0;
        for lens in &mut cal.lenses {
            lens.fx = 200.0 + random() * 6000.0;
            lens.fy = 200.0 + random() * 6000.0;
            lens.cx = random() * 3840.0;
            lens.cy = random() * 2160.0;
            lens.distortion = [
                (random() - 0.5) * 2.0,
                (random() - 0.5) * 1.5,
                (random() - 0.5) * 0.3,
                (random() - 0.5) * 0.08,
            ];
            lens.correction = [0.0, 0.4, 1.0][i % 3];
        }
        let topo = cal.topology.l_shape_mut().unwrap();
        topo.intersect = random();
        topo.x_rx = (random() - 0.5) * 2.0;
        topo.x_rz = (random() - 0.5) * 2.0;
        topo.z_rx = (random() - 0.5) * 2.0;
        topo.z_rz = (random() - 0.5) * 2.0;
        topo.x_ty = (random() - 0.5) * 0.6;
        cal.validate().unwrap();
        let pose = Pose {
            yaw: ((random() - 0.5) * 6.0) as f32,
            pitch: ((random() - 0.5) * 2.0) as f32,
            fov_degrees: (1.0 + random() * 150.0) as f32,
        };
        let maps = cal.surface_maps(&size, pose);
        if certify(&maps, 0.0) == Certificate::Covered {
            accepted += 1;
            assert_eq!(exhaustive(&maps, &size), 0, "case {i}: {pose:?}, {cal:?}");
        }
    }
    eprintln!("extreme-domain certified cases={accepted}/2400");
    assert!(accepted > 10);
}

#[test]
fn interior_holes_and_more_than_two_surfaces_are_not_sampling_special_cases() {
    use crate::stitch::SurfaceUv;
    struct Rect {
        lo: [f64; 2],
        hi: [f64; 2],
    }
    impl SurfaceMap for Rect {
        fn sample_uv(&self, p: [f64; 2]) -> Option<SurfaceUv> {
            (0..2)
                .all(|i| p[i] >= self.lo[i] && p[i] <= self.hi[i])
                .then_some(SurfaceUv {
                    u: 0.5,
                    v: 0.5,
                    edge: 0.5,
                })
        }
        fn covers_ndc(&self, lo: [f64; 2], hi: [f64; 2], _: f64) -> bool {
            (0..2).all(|i| lo[i] >= self.lo[i] && hi[i] <= self.hi[i])
        }
    }
    let size = ViewportSize {
        width: 101,
        height: 99,
    };
    let rect = |lo, hi| {
        (
            Box::new(Rect { lo, hi }) as Box<dyn SurfaceMap>,
            BlendRule::Opaque,
        )
    };
    let maps = vec![
        rect([-1.0, -1.0], [-0.2, 1.0]),
        rect([-0.21, -1.0], [0.21, 1.0]),
        rect([0.2, -1.0], [0.6, 1.0]),
        rect([0.59, -1.0], [1.0, 1.0]),
    ];
    assert_eq!(certify(&maps, 0.0), Certificate::Covered);
    // Four surfaces surround a subpixel hole that misses every pixel centre, all corners, and the viewport centre.
    let maps = vec![
        rect([-1.0, -1.0], [0.2340, 1.0]),
        rect([0.2341, -1.0], [1.0, 1.0]),
        rect([-1.0, -1.0], [1.0, 0.3450]),
        rect([-1.0, 0.3451], [1.0, 1.0]),
    ];
    assert_eq!(exhaustive(&maps, &size), 0);
    assert_eq!(certify(&maps, 0.0), Certificate::Uncovered);
}

#[test]
fn swept_certificates_contain_intermediate_rendered_views() {
    let mut cases = vec![stereo()];
    for sweep_deg in [30.0, 180.0, 270.0, 360.0] {
        cases.push(Calibration::new(
            vec![Lens::flat(3840, 2160)],
            Cylinder {
                sweep_deg,
                ..Default::default()
            },
            Framing {
                axis_offset: 0.0,
                tilt: 0.0,
                roll: 0.0,
            },
        ));
    }
    let size = ViewportSize {
        width: 65,
        height: 37,
    };
    let mut seed = 819_u64;
    let mut random = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 32) as u32) as f64 / u32::MAX as f64
    };
    let mut accepted = 0;
    for mut cal in cases {
        for _ in 0..300 {
            cal.framing.tilt = (random() - 0.5) * 0.8;
            cal.framing.roll = (random() - 0.5) * 0.8;
            let a = Pose {
                yaw: ((random() - 0.5) * 6.0) as f32,
                pitch: ((random() - 0.5) * 0.8) as f32,
                fov_degrees: (5.0 + 45.0 * random()) as f32,
            };
            let b = Pose {
                yaw: a.yaw + ((random() - 0.5) * 0.25) as f32,
                pitch: a.pitch + ((random() - 0.5) * 0.15) as f32,
                fov_degrees: (a.fov_degrees + ((random() - 0.5) * 12.0) as f32).max(1.0),
            };
            if certify_sweep_enclosure(&cal, &size, a, b, 2048) != Certificate::Covered {
                continue;
            }
            accepted += 1;
            for step in 0..=20 {
                let pose = interpolate(a, b, step as f32 / 20.0);
                let maps = cal.surface_maps(&size, pose);
                assert_eq!(
                    exhaustive(&maps, &size),
                    0,
                    "{a:?} -> {b:?}, step={step}, {cal:?}"
                );
            }
        }
    }
    eprintln!("certified swept paths={accepted}/1500, 21 exhaustive intermediate views each");
    assert!(accepted > 100);
}

#[test]
fn four_corners_and_center_do_not_prove_cylinder_support() {
    let cal = Calibration::new(
        vec![Lens::flat(3840, 2160)],
        Cylinder {
            sweep_deg: 360.0,
            ..Default::default()
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let size = ViewportSize {
        width: 401,
        height: 101,
    };
    let maps = cal.surface_maps(
        &size,
        Pose {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: 90.0,
        },
    );
    for (x, y) in [(0, 0), (400, 0), (0, 100), (400, 100), (200, 50)] {
        assert!(maps[0].0.sample_uv(size.pixel_ndc(x, y)).is_some());
    }
    assert!(maps[0].0.sample_uv(size.pixel_ndc(200, 0)).is_none());
    assert_eq!(certify(&maps, 0.0), Certificate::Uncovered);
}

#[test]
fn certified_rectangles_match_all_pixel_oracle() {
    let mut cases = vec![stereo()];
    for sweep_deg in [30.0, 180.0, 270.0, 360.0] {
        cases.push(Calibration::new(
            vec![Lens::flat(3840, 2160)],
            Cylinder {
                sweep_deg,
                ..Default::default()
            },
            Framing {
                axis_offset: 0.0,
                tilt: 0.0,
                roll: 0.0,
            },
        ));
    }
    let mut checked = 0;
    for mut cal in cases {
        for (tilt, roll) in [(0.0, 0.0), (0.25, 0.3)] {
            cal.framing.tilt = tilt;
            cal.framing.roll = roll;
            for (width, height) in [(65, 37), (37, 65)] {
                let size = ViewportSize { width, height };
                for yaw in [-2.8, -1.6, -1.0, 0.0, 1.0, 1.6, 2.8] {
                    for pitch in [-0.4, 0.0, 0.4] {
                        for fov_degrees in [10.0, 30.0, 60.0, 100.0] {
                            let pose = Pose {
                                yaw,
                                pitch,
                                fov_degrees,
                            };
                            let maps = cal.surface_maps(&size, pose);
                            let (result, work) = covered(&maps, &size);
                            let missing = exhaustive(&maps, &size);
                            assert_eq!(
                                result,
                                missing == 0,
                                "{} tilt={tilt} roll={roll} {size:?} {pose:?} missing={missing}, work={work}",
                                cal.topology.projection().name()
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("checked {checked} complete viewports against every output pixel");
}

#[test]
fn coverage_check_work_at_output_resolution() {
    let cal = stereo();
    for (width, height) in [(65, 37), (1920, 1080), (3840, 2160)] {
        let size = ViewportSize { width, height };
        for yaw in [-1.0, 0.0, 1.0] {
            let pose = Pose {
                yaw,
                pitch: 0.0,
                fov_degrees: 30.0,
            };
            let maps = cal.surface_maps(&size, pose);
            let start = std::time::Instant::now();
            let (result, work) = covered(&maps, &size);
            eprintln!(
                "{size:?} yaw={yaw}: covered={result}, work={work}, time={:?}",
                start.elapsed()
            );
            assert!(result);
        }
    }
}
