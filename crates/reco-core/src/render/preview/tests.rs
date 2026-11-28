//! Pixel and lifetime oracles for prepared preview inputs.

use super::*;
use crate::calibration::{Calibration, Framing, Lens};
use crate::core::StitchCore;
use crate::geometry::RenderPose;
use crate::gpu::rgba_readback::RgbaReadback;
use crate::lens::preview::LensPreviewRenderer;
use crate::projection::Cylinder;
use crate::render::color::{YuvMatrix, yuv_to_rgb};
use crate::render::viewport::ViewportSize;
use crate::source::{Nv12Data, Yuv420pData};
use crate::stitch::{Executor, GpuExecutor, GpuExecutorConfig, test_support::gpu_or_skip};

const SIZE: (u32, u32) = (64, 48);

fn engine(gpu: &GpuContext) -> StitchCore {
    let calibration = Calibration::new(
        vec![Lens::flat(SIZE.0, SIZE.1)],
        Cylinder {
            radius: 1.0,
            arc_deg: 180.0,
            surface_height: None,
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    StitchCore::new(Executor::Gpu(Box::new(
        GpuExecutor::new(
            gpu.clone(),
            GpuExecutorConfig {
                viewport_size: ViewportSize {
                    width: SIZE.0,
                    height: SIZE.1,
                },
                ..GpuExecutorConfig::new(calibration, vec![SIZE], InputFormat::Yuv420p)
            },
        )
        .unwrap(),
    )))
    .unwrap()
}

fn capture(gpu: &GpuContext, target: &wgpu::Texture) -> Vec<u8> {
    let mut readback = RgbaReadback::new(gpu, target.width(), target.height()).unwrap();
    let empty = gpu
        .device()
        .create_command_encoder(&Default::default())
        .finish();
    assert!(readback.readback(gpu, target, empty).unwrap().is_none());
    readback.flush_pending(gpu).unwrap().unwrap().to_vec()
}

fn panorama(core: &StitchCore, frame: &PreviewFrame, pose: RenderPose) -> Vec<u8> {
    let (width, height) = core.output_dims();
    let target = core
        .gpu()
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("preview_pixel_oracle"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
    core.render_to_view(frame, pose, &target.create_view(&Default::default()))
        .unwrap();
    capture(core.gpu(), &target)
}

fn record(core: &mut StitchCore, frame: &PreviewFrame, pose: RenderPose) -> Vec<u8> {
    assert!(
        core.render_and_readback_nv12(frame, pose)
            .unwrap()
            .is_none()
    );
    let bytes = core.flush_nv12().unwrap().unwrap().to_vec();
    assert!(core.flush_nv12().unwrap().is_none());
    bytes
}

fn transfers(core: &StitchCore) -> usize {
    core.executor
        .gpu()
        .unwrap()
        .pipeline
        .preview_plane_transfers()
}

fn lens(gpu: &GpuContext) -> LensPreviewRenderer {
    LensPreviewRenderer::new(gpu, SIZE.0, SIZE.1, wgpu::TextureFormat::Rgba8Unorm)
}

fn nv12(yuv: [u8; 3]) -> Nv12Data {
    let [y, u, v] = yuv;
    let pixels = (SIZE.0 * SIZE.1) as usize;
    Nv12Data {
        size: SIZE,
        y: vec![y; pixels],
        uv: [u, v].repeat(pixels / 4),
    }
}

fn center_index() -> usize {
    ((SIZE.1 / 2) * SIZE.0 + SIZE.0 / 2) as usize
}

fn assert_center_color(rgba: &[u8], expected: [f64; 3]) {
    let center = center_index() * 4;
    for (actual, expected) in rgba[center..center + 3].iter().zip(expected) {
        let code = (expected * 255.0).round() as u8;
        assert!(actual.abs_diff(code) <= 1, "got {actual}, expected {code}");
    }
}

#[test]
fn host_layouts_share_panorama_lens_and_recording_color() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut core = engine(&gpu);
    let mut lens = lens(&gpu);
    let pose = RenderPose::default();
    let cal_lens = core.calibration().lenses[0].clone();
    for matrix in [YuvMatrix::Bt709, YuvMatrix::Bt601, YuvMatrix::Bt2020Ncl] {
        for full_range in [false, true] {
            let yuv = [140, 82, 149];
            let image = nv12(yuv);
            let color = YuvColor {
                matrix,
                full_range,
                ..Default::default()
            };
            let [y, u, v] = yuv.map(|code| f64::from(code) / 255.0);
            let expected = yuv_to_rgb(y, u, v, color);
            let planar = core
                .prepare_preview_frame(
                    FrameSet::Yuv420p(vec![
                        Yuv420pData {
                            size: SIZE,
                            y: image.y.clone(),
                            u: image.uv.iter().step_by(2).copied().collect(),
                            v: image.uv.iter().skip(1).step_by(2).copied().collect(),
                        }
                        .into(),
                    ]),
                    color,
                    [0; 2],
                )
                .unwrap();
            let interleaved = core
                .prepare_preview_frame(FrameSet::Nv12(vec![image.into()]), color, [0; 2])
                .unwrap();
            let planar_panorama = panorama(&core, &planar, pose);
            assert_center_color(&planar_panorama, expected);
            assert_eq!(panorama(&core, &interleaved, pose), planar_panorama);
            let planar_lens = capture(&gpu, &lens.render(&planar, 0, &cal_lens).unwrap());
            assert_center_color(&planar_lens, expected);
            assert_eq!(
                capture(&gpu, &lens.render(&interleaved, 0, &cal_lens).unwrap()),
                planar_lens
            );
            let recording = record(&mut core, &planar, pose);
            assert_eq!(recording.len(), (SIZE.0 * SIZE.1 * 3 / 2) as usize);
            let rgb_center = &planar_panorama[center_index() * 4..][..3];
            let y = matrix
                .luma()
                .iter()
                .zip(rgb_center)
                .map(|(weight, code)| weight * f32::from(*code) / 255.0)
                .sum::<f32>();
            assert!(recording[center_index()].abs_diff((16.0 + 219.0 * y).round() as u8) <= 1);
            assert_eq!(record(&mut core, &interleaved, pose), recording);
        }
    }
}

#[test]
fn failed_candidate_preserves_display_and_slots_reuse_after_release() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut core = engine(&gpu);
    let pose = RenderPose::default();
    let prepare = |core: &mut StitchCore, y| {
        core.prepare_preview_frame(
            FrameSet::Nv12(vec![nv12([y, 128, 128]).into()]),
            Default::default(),
            [0; 2],
        )
    };
    let displayed = prepare(&mut core, 75).unwrap();
    let before = panorama(&core, &displayed, pose);
    for invalid in [
        vec![],
        vec![
            Nv12Data {
                size: (SIZE.0 * 2, SIZE.1 / 2),
                ..nv12([170, 128, 128])
            }
            .into(),
        ],
        vec![
            Nv12Data {
                uv: vec![128],
                ..nv12([170, 128, 128])
            }
            .into(),
        ],
    ] {
        assert!(
            core.prepare_preview_frame(FrameSet::Nv12(invalid), Default::default(), [0; 2])
                .is_err()
        );
        assert_eq!(panorama(&core, &displayed, pose), before);
    }
    let candidate = prepare(&mut core, 170).unwrap();
    assert!(prepare(&mut core, 210).is_err());
    assert_eq!(panorama(&core, &displayed, pose), before);
    assert_ne!(panorama(&core, &candidate, pose), before);

    let old_plane = displayed.slot.textures[0][0].clone();
    let old_binding = displayed.slot.bindings[0].clone();
    drop(displayed);
    let next = prepare(&mut core, 210).unwrap();
    assert_eq!(next.slot.textures[0][0], old_plane);
    assert_eq!(next.slot.bindings[0], old_binding);
    assert_ne!(panorama(&core, &next, pose), before);
}

#[test]
fn paused_host_redraw_releases_source_and_leaves_planes_untouched() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut core = engine(&gpu);
    let image = Arc::new(nv12([120, 110, 145]));
    let source = Arc::downgrade(&image);
    let frame = core
        .prepare_preview_frame(FrameSet::Nv12(vec![image]), Default::default(), [0; 2])
        .unwrap();
    assert!(source.upgrade().is_none());
    assert_eq!(transfers(&core), 2);

    let original = panorama(&core, &frame, RenderPose::default());
    core.resize_preview(92, 44).unwrap();
    let pose = RenderPose {
        yaw: 0.17,
        ..Default::default()
    };
    let mut calibration = core.calibration().clone();
    calibration.lenses[0].correction = 0.4;
    core.update_calibration_preview(calibration).unwrap();
    let lens_image = lens(&gpu)
        .render(&frame, 0, &core.calibration().lenses[0])
        .unwrap();
    assert!(
        capture(&gpu, &lens_image)
            .chunks_exact(4)
            .any(|px| px[0] > 64)
    );
    assert_eq!(panorama(&core, &frame, pose).len(), 92 * 44 * 4);
    assert_eq!(record(&mut core, &frame, pose).len(), 92 * 44 * 3 / 2);
    assert_eq!(transfers(&core), 2);
    core.resize_preview(SIZE.0, SIZE.1).unwrap();
    assert_eq!(panorama(&core, &frame, RenderPose::default()), original);
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires NVIDIA CUDA and Vulkan external-memory interop"]
fn cuda_preview_recycles_leases_while_paused_and_preserves_nv12_p010_pixels() {
    use crate::interop::cuda::{cuda_ensure_context, cuda_memcpy_htod_2d, cuda_synchronize};
    use crate::interop::cuda_lease::CudaSlotPool;
    use std::time::{Duration, Instant};

    let gpu = GpuContext::new_blocking().unwrap();
    assert!(
        gpu.supports_zero_copy(),
        "select the CUDA device's Vulkan adapter"
    );
    cuda_ensure_context().unwrap();
    for format in [GpuPixelFormat::Nv12, GpuPixelFormat::P010] {
        for full_range in [false, true] {
            let source_color = YuvColor {
                matrix: YuvMatrix::Bt2020Ncl,
                full_range,
                ..Default::default()
            };
            let pool = CudaSlotPool::new(&gpu, SIZE, format, source_color, 180).unwrap();
            let lease = pool.try_acquire().unwrap();
            let held = pool.try_acquire().unwrap();
            assert!(pool.try_acquire().is_none());
            let frame = lease.cuda_frame();

            let mut upright = nv12([180, 104, 155]);
            upright.y[..(SIZE.0 * SIZE.1 / 4) as usize].fill(56);
            upright.uv[..(SIZE.0 * SIZE.1 / 8) as usize]
                .chunks_exact_mut(2)
                .for_each(|pair| {
                    pair.copy_from_slice(&[113, 80]);
                });
            let native_y: Vec<_> = upright.y.iter().rev().copied().collect();
            let native_uv: Vec<_> = upright
                .uv
                .chunks_exact(2)
                .rev()
                .flatten()
                .copied()
                .collect();
            let pack = |codes: &[u8]| {
                if format == GpuPixelFormat::P010 {
                    codes
                        .iter()
                        .flat_map(|&sample| (u16::from(sample) << 8).to_le_bytes())
                        .collect()
                } else {
                    codes.to_vec()
                }
            };
            let upload_cuda = |y: &[u8], uv: &[u8]| {
                let row = SIZE.0 as usize * format.bytes_per_sample();
                cuda_memcpy_htod_2d(
                    frame.y_ptr,
                    frame.y_pitch,
                    y.as_ptr(),
                    row,
                    row,
                    SIZE.1 as usize,
                )
                .unwrap();
                cuda_memcpy_htod_2d(
                    frame.uv_ptr,
                    frame.uv_pitch,
                    uv.as_ptr(),
                    row,
                    row,
                    (SIZE.1 / 2) as usize,
                )
                .unwrap();
                cuda_synchronize().unwrap();
            };
            upload_cuda(&pack(&native_y), &pack(&native_uv));

            let mut core = engine(&gpu);
            let mut reference = engine(&gpu);
            let host = reference
                .prepare_preview_frame(FrameSet::Nv12(vec![upright.into()]), source_color, [0; 2])
                .unwrap();
            let preview = core
                .prepare_preview_frame(
                    FrameSet::GpuResident(vec![lease]),
                    Default::default(),
                    [0; 2],
                )
                .unwrap();

            assert_eq!(transfers(&core), 2);

            // No rendering or manual device poll while waiting for retirement.
            let deadline = Instant::now() + Duration::from_secs(5);
            let recycled = loop {
                if let Some(recycled) = pool.try_acquire() {
                    break recycled;
                }
                assert!(
                    Instant::now() < deadline,
                    "paused preview pinned the CUDA slot"
                );
                std::thread::sleep(Duration::from_millis(1));
            };
            assert_eq!(recycled.cuda_frame().y_ptr, frame.y_ptr);
            upload_cuda(
                &pack(&vec![16; native_y.len()]),
                &pack(&vec![128; native_uv.len()]),
            );
            drop((recycled, held, pool));

            let mut lens = lens(&gpu);
            for (width, height, yaw, correction) in [(64, 48, 0.0, 1.0), (92, 44, 0.21, 0.4)] {
                for engine in [&mut core, &mut reference] {
                    engine.resize_preview(width, height).unwrap();
                }
                let mut params = core.calibration().lenses[0].clone();
                params.correction = correction;
                let pose = RenderPose {
                    yaw,
                    ..Default::default()
                };
                let assert_pixels = |actual: &[u8], expected: &[u8]| {
                    assert_eq!(actual.len(), expected.len());
                    let limit = if format == GpuPixelFormat::P010 { 2 } else { 1 };
                    let max = actual
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| a.abs_diff(*b))
                        .max()
                        .unwrap();
                    assert!(
                        max <= limit,
                        "{format:?}, full range={full_range}: max pixel difference {max}"
                    );
                };
                assert_pixels(
                    &panorama(&core, &preview, pose),
                    &panorama(&reference, &host, pose),
                );
                let image = lens.render(&preview, 0, &params).unwrap();
                let expected = lens.render(&host, 0, &params).unwrap();
                assert_pixels(&capture(&gpu, &image), &capture(&gpu, &expected));
                assert_pixels(
                    &record(&mut core, &preview, pose),
                    &record(&mut reference, &host, pose),
                );
            }
            assert_eq!(transfers(&core), 2);
        }
    }
}
