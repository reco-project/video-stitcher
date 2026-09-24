//! External-renderer contract: use calibrated maps without duplicating projection or blend math.

use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::geometry::Pose;
use reco_core::projection::{Cylinder, LShape};
use reco_core::render::planes::Nv12Planes;
use reco_core::render::viewport::ViewportSize;
use reco_core::stitch::{BlendRule, CpuExecutor};

fn calibrations() -> [Calibration; 2] {
    let lens = Lens::flat(64, 32);
    let framing = Framing {
        axis_offset: 0.25,
        tilt: 0.12,
        roll: -0.08,
    };
    [
        Calibration::new(
            vec![lens.clone(), lens.clone()],
            LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.2,
            },
            framing,
        ),
        Calibration::new(
            vec![lens],
            Cylinder {
                focal_length: 20.0,
                sweep_deg: 180.0,
                video_height: None,
            },
            Framing {
                axis_offset: 0.0,
                tilt: 0.12,
                roll: -0.08,
            },
        ),
    ]
}

#[test]
fn external_compositor_matches_cpu_for_both_projections() {
    let size = ViewportSize {
        width: 96,
        height: 54,
    };
    for cal in calibrations() {
        cal.validate().unwrap();
        let count = cal.topology.camera_count();
        let executor = CpuExecutor::new(cal.clone(), size.clone(), 64, 32, false).unwrap();
        let mut seen = vec![false; count];
        let mut saw_blend = false;
        let mut saw_uncovered = false;
        for yaw in [-1.0, 0.0, 1.0] {
            for fov_degrees in [35.0, 110.0] {
                let pose = Pose {
                    yaw,
                    pitch: 0.1,
                    fov_degrees,
                };
                let maps = cal.surface_maps(&size, pose);
                assert_eq!(maps.len(), count);
                // Activate each input separately to detect incorrect source order or blending.
                for active in 0..count {
                    let luma: Vec<_> = (0..count)
                        .map(|camera| vec![if camera == active { 235 } else { 16 }; 64 * 32])
                        .collect();
                    let chroma = vec![128; 64 * 32 / 2];
                    let planes: Vec<_> =
                        luma.iter().map(|y| Nv12Planes { y, uv: &chroma }).collect();
                    let rendered = executor.stitch_nv12(&planes, pose).unwrap();
                    for y in 0..size.height {
                        for x in 0..size.width {
                            let ndc = [
                                (f64::from(x) + 0.5) / f64::from(size.width) * 2.0 - 1.0,
                                1.0 - (f64::from(y) + 0.5) / f64::from(size.height) * 2.0,
                            ];
                            let mut value = 0.0;
                            let mut supported = false;
                            for (camera, (map, blend)) in maps.iter().enumerate() {
                                if let Some(sample) = map.sample_uv(ndc) {
                                    assert!((0.0..=1.0).contains(&sample.u));
                                    assert!((0.0..=1.0).contains(&sample.v));
                                    seen[camera] = true;
                                    supported = true;
                                    let alpha = blend.alpha(sample.edge);
                                    assert!((0.0..=1.0).contains(&alpha));
                                    saw_blend |= alpha > 0.0 && alpha < 1.0;
                                    let source = if camera == active { 1.0 } else { 0.0 };
                                    value += (source - value) * alpha;
                                }
                            }
                            saw_uncovered |= !supported;
                            let expected = (value * 255.0).round() as u8;
                            let offset = ((y * size.width + x) * 4) as usize;
                            for channel in &rendered[offset..offset + 3] {
                                assert!(channel.abs_diff(expected) <= 1);
                            }
                            assert_eq!(rendered[offset + 3], 255);
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|seen| *seen));
        assert!(saw_uncovered);
        if count == 2 {
            assert!(saw_blend);
        }
    }
}

#[test]
fn public_blend_weights_cover_seam_endpoints() {
    assert_eq!(BlendRule::Opaque.alpha(-1.0), 1.0);
    assert_eq!(BlendRule::Smoothstep(0.0).alpha(-1.0), 1.0);
    let blend = BlendRule::Smoothstep(0.2);
    assert_eq!(blend.alpha(-0.1), 0.0);
    assert_eq!(blend.alpha(0.0), 0.0);
    assert_eq!(blend.alpha(0.1), 0.5);
    assert_eq!(blend.alpha(0.2), 1.0);
    assert_eq!(blend.alpha(0.3), 1.0);
}
