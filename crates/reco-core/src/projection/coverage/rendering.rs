//! White-frame and gradient image oracles for the coverage candidate.

use super::*;

fn gradient_y() -> Vec<u8> {
    (0..320 * 180)
        .map(|i| {
            let (x, y) = (i % 320, i / 320);
            if x % 32 < 2 || y % 20 < 2 {
                220
            } else {
                (70 + x * 100 / 320 + y * 50 / 180) as u8
            }
        })
        .collect()
}

#[test]
#[ignore = "compares an external pre-refactor image oracle"]
fn normalized_coordinates_preserve_the_recorded_pixel_output() {
    use crate::{render::planes::Nv12Planes, stitch::CpuExecutor};
    let dir = std::path::PathBuf::from(std::env::var("RECO_COVERAGE_ARTIFACT_DIR").unwrap());
    let mut cal = stereo();
    cal.framing.tilt = 0.23;
    cal.framing.roll = 0.17;
    let size = ViewportSize {
        width: 320,
        height: 180,
    };
    let render = {
        let source_sizes = vec![(320, 180); cal.projection.camera_count()];
        CpuExecutor::new(cal, size, source_sizes, false)
    }
    .unwrap();
    let y = gradient_y();
    let left: Vec<_> = (0..320 * 90)
        .map(|i| if i % 2 == 0 { 110 } else { 150 })
        .collect();
    let right: Vec<_> = (0..320 * 90)
        .map(|i| if i % 2 == 0 { 150 } else { 110 })
        .collect();
    let frames = [
        Nv12Planes { y: &y, uv: &left },
        Nv12Planes { y: &y, uv: &right },
    ];
    let csv = std::fs::read_to_string(dir.join("tilted-motion.csv")).unwrap();
    let baseline = std::fs::read(dir.join("tilted-gradient-sheet.ppm")).unwrap();
    let header = b"P6\n960 540\n255\n";
    assert!(baseline.starts_with(header));
    let baseline = &baseline[header.len()..];
    for (i, line) in csv.lines().skip(1).enumerate().step_by(40) {
        let values: Vec<_> = line.split(',').collect();
        let image = render
            .stitch_nv12(
                &frames,
                RenderPose {
                    yaw: values[3].parse().unwrap(),
                    pitch: values[4].parse().unwrap(),
                    fov_degrees: 30.0,
                },
            )
            .unwrap();
        let tile = i / 40;
        for y in 0..180 {
            for x in 0..320 {
                let dst = ((tile / 3 * 180 + y) * 960 + tile % 3 * 320 + x) * 3;
                let src = (y * 320 + x) * 4;
                assert_eq!(
                    &image[src..src + 3],
                    &baseline[dst..dst + 3],
                    "frame={i},x={x},y={y}"
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn fitted_random_lenses_and_placements_render_without_gpu_holes() {
    use crate::{
        render::{planes::Nv12Planes, renderer::InputFormat},
        stitch::{GpuExecutor, GpuExecutorConfig, StitchExecutor, test_support::gpu_or_skip},
    };
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let y = vec![235; 320 * 180];
    let uv = vec![128; 320 * 90];
    let frames = [Nv12Planes { y: &y, uv: &uv }; 2];
    let mut seed = 1927_u64;
    let mut random = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 32) as u32) as f64 / u32::MAX as f64
    };
    let mut checked = 0;
    for case in 0..120 {
        let mut cal = stereo();
        cal.framing.axis_offset = [0.02, 0.08, 0.24, 0.8, 3.0][case % 5];
        cal.framing.tilt = (random() - 0.5) * 0.9;
        cal.framing.roll = (random() - 0.5) * 0.9;
        for lens in &mut cal.lenses {
            lens.correction = random() as f32;
            lens.cx = 3840.0 * (0.4 + 0.2 * random());
            lens.cy = 2160.0 * (0.4 + 0.2 * random());
            lens.distortion = [
                (random() - 0.5) * 0.5,
                (random() - 0.5) * 0.4,
                (random() - 0.5) * 0.1,
                (random() - 0.5) * 0.03,
            ];
        }
        if let crate::calibration::Projection::LShape(layout) = &mut cal.projection {
            layout.x_rx = (random() - 0.5) * 0.2;
            layout.z_rz = (random() - 0.5) * 0.2;
        }
        let size = if case % 2 == 0 {
            ViewportSize {
                width: 384,
                height: 96,
            }
        } else {
            ViewportSize {
                width: 96,
                height: 384,
            }
        };
        let mut render = GpuExecutor::new(
            gpu.clone(),
            GpuExecutorConfig {
                viewport_size: size.clone(),
                ..{
                    let source_sizes = vec![(320, 180); cal.projection.camera_count()];
                    GpuExecutorConfig::new(cal.clone(), source_sizes, InputFormat::Nv12)
                }
            },
        )
        .unwrap();
        for yaw in [-1.5, -0.6, 0.0, 0.6, 1.5] {
            let request = cal.world_to_render_pose(crate::geometry::WorldPose {
                yaw,
                pitch: (random() as f32 - 0.5) * 0.4,
                fov_degrees: 120.0,
            });
            let Some(p) = fit_fov(&cal, &size, request) else {
                continue;
            };
            let image = render.stitch(&frames, p).unwrap();
            let missing = image
                .chunks_exact(4)
                .filter(|p| p[..3].iter().any(|v| *v < 250))
                .count();
            assert_eq!(missing, 0, "case={case},{p:?},{cal:?}");
            checked += 1;
        }
    }
    eprintln!("Random boundary fits with GPU white oracle: {checked}/600");
    assert!(checked > 200);
}

#[test]
#[cfg(feature = "gpu")]
fn fitted_viewports_render_white_across_lens_correction_and_aspects() {
    use crate::{
        render::{planes::Nv12Planes, renderer::InputFormat},
        stitch::{GpuExecutor, GpuExecutorConfig, StitchExecutor, test_support::gpu_or_skip},
    };
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    eprintln!(
        "GPU fit oracle: {} / {}",
        gpu.gpu_name(),
        gpu.backend_name()
    );
    let y = vec![235; 320 * 180];
    let uv = vec![128; 320 * 90];
    let planes = [Nv12Planes { y: &y, uv: &uv }; 2];
    let mut checked = 0;
    for correction in [0.0, 0.4, 1.0] {
        for (tilt, roll) in [(0.0, 0.0), (0.23, 0.17)] {
            let mut cal = stereo();
            cal.framing.tilt = tilt;
            cal.framing.roll = roll;
            cal.lenses[0].correction = correction;
            cal.lenses[1].correction = 1.0 - correction;
            for size in [
                ViewportSize {
                    width: 640,
                    height: 360,
                },
                ViewportSize {
                    width: 360,
                    height: 640,
                },
            ] {
                let mut render = GpuExecutor::new(
                    gpu.clone(),
                    GpuExecutorConfig {
                        viewport_size: size.clone(),
                        ..{
                            let source_sizes = vec![(320, 180); cal.projection.camera_count()];
                            GpuExecutorConfig::new(cal.clone(), source_sizes, InputFormat::Nv12)
                        }
                    },
                )
                .unwrap();
                let cam = cal.projection.geometry().virtual_camera(&cal.framing);
                for yaw in [-1.5, -0.8, 0.0, 0.8, 1.5] {
                    for pitch in [-0.15, 0.0, 0.15] {
                        let (yaw, pitch) = crate::geometry::world_to_render_angles(
                            &cam,
                            yaw,
                            pitch,
                            tilt as f32,
                            roll as f32,
                        );
                        let Some(pose) = fit_fov(
                            &cal,
                            &size,
                            RenderPose {
                                yaw,
                                pitch,
                                fov_degrees: 70.0,
                            },
                        ) else {
                            continue;
                        };
                        let pixels = render.stitch(&planes, pose).unwrap();
                        let dark = pixels
                            .chunks_exact(4)
                            .filter(|p| p[..3].iter().any(|v| *v < 250))
                            .count();
                        assert_eq!(
                            dark, 0,
                            "correction={correction},tilt={tilt},roll={roll},{size:?},{pose:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    eprintln!("white GPU fitted viewports={checked}");
    assert!(checked >= 100);
}
