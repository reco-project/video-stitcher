//! GPU pixel checks independent of model execution and native decoder APIs.

use super::*;
use reco_core::gpu::GpuContext;

fn gpu() -> Option<GpuContext> {
    match GpuContext::new_blocking() {
        Ok(gpu) => {
            eprintln!("preprocess adapter: {}", gpu.gpu_name());
            Some(gpu)
        }
        Err(error) if std::env::var_os("RECO_REQUIRE_GPU").is_none() => {
            eprintln!("GPU unavailable: {error}");
            None
        }
        Err(error) => panic!("GPU required: {error}"),
    }
}

fn plane(gpu: &GpuContext, size: (u32, u32), uv: bool) -> wgpu::TextureView {
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("preprocess fixture"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if uv {
            wgpu::TextureFormat::Rg8Unorm
        } else {
            wgpu::TextureFormat::R8Unorm
        },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let row = size.0 * if uv { 2 } else { 1 };
    gpu.queue().write_texture(
        texture.as_image_copy(),
        &vec![if uv { 128 } else { 200 }; (row * size.1) as usize],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row),
            rows_per_image: None,
        },
        texture.size(),
    );
    texture.create_view(&Default::default())
}

#[test]
fn odd_letterbox_padding_is_grey() {
    let Some(gpu) = gpu() else { return };
    let mut preprocess = WgpuPreprocessor::new(gpu.device(), 8).unwrap();
    let y = plane(&gpu, (6, 4), false);
    let uv = plane(&gpu, (3, 2), true);
    let tensor = preprocess
        .preprocess(gpu.device(), gpu.queue(), (&y, &uv), (6, 4), 0)
        .unwrap();
    // 6x4 scales to 8x5: one row above, two below, not six content rows.
    for channel in 0..3 {
        for row in [0, 6, 7] {
            for col in 0..8 {
                assert!((tensor[channel * 64 + row * 8 + col] - 114.0 / 255.0).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn changing_shapes_rotation_and_bit_depth_match_pixels_and_reuse_storage() {
    let Some(gpu) = gpu() else { return };
    let mut preprocess = WgpuPreprocessor::new(gpu.device(), 8).unwrap();
    let tensor_ptr = preprocess.tensor.as_ptr();
    let output_buffer = preprocess.output_buffer.clone();
    let staging_buffer = preprocess.staging_buffer.clone();
    for bits in [8, 10] {
        if bits == 10
            && !gpu
                .device()
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            eprintln!("10-bit textures unavailable");
            continue;
        }
        for (w, h) in [(6, 4), (4, 6), (12, 4), (4, 12), (8, 8), (10, 6), (6, 4)] {
            let mut planes = Vec::new();
            for uv in [false, true] {
                let size = if uv { (w / 2, h / 2) } else { (w, h) };
                let channels = if uv { 2 } else { 1 };
                let values: Vec<u8> = (0..size.1)
                    .flat_map(|y| {
                        (0..size.0 * channels).map(move |x| {
                            if uv {
                                [100, 170][x as usize % 2]
                            } else {
                                (32 + 7 * x + 9 * y) as u8
                            }
                        })
                    })
                    .collect();
                let data = if bits == 10 {
                    values
                        .iter()
                        .flat_map(|&v| ((u16::from(v) * 257) & 0xffc0).to_le_bytes())
                        .collect()
                } else {
                    values
                };
                let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("changing shape fixture"),
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: match (uv, bits) {
                        (false, 8) => wgpu::TextureFormat::R8Unorm,
                        (true, 8) => wgpu::TextureFormat::Rg8Unorm,
                        (false, _) => wgpu::TextureFormat::R16Unorm,
                        (true, _) => wgpu::TextureFormat::Rg16Unorm,
                    },
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                gpu.queue().write_texture(
                    texture.as_image_copy(),
                    &data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(size.0 * channels * if bits == 10 { 2 } else { 1 }),
                        rows_per_image: None,
                    },
                    texture.size(),
                );
                planes.push(texture.create_view(&Default::default()));
            }
            for rotation in [0, 180, 0] {
                preprocess.tensor.fill(-99.0);
                let tensor = preprocess
                    .preprocess(
                        gpu.device(),
                        gpu.queue(),
                        (&planes[0], &planes[1]),
                        (w, h),
                        rotation,
                    )
                    .unwrap();
                assert_eq!(tensor.as_ptr(), tensor_ptr);
                // Independent scalar oracle, including asymmetric integer padding.
                let scale = 8.0 / w.max(h) as f32;
                let (cw, ch) = (
                    (w as f32 * scale).round() as usize,
                    (h as f32 * scale).round() as usize,
                );
                let (px, py) = ((8 - cw) / 2, (8 - ch) / 2);
                for oy in 0..8 {
                    for ox in 0..8 {
                        let expected = if ox < px || ox >= px + cw || oy < py || oy >= py + ch {
                            [114.0 / 255.0; 3]
                        } else {
                            let mut x = (ox - px) as f32 / scale;
                            let mut y = (oy - py) as f32 / scale;
                            if rotation == 180 {
                                x = (w - 1) as f32 - x;
                                y = (h - 1) as f32 - y;
                            }
                            let luma = (32.0
                                + 7.0 * x.clamp(0.0, (w - 1) as f32)
                                + 9.0 * y.clamp(0.0, (h - 1) as f32))
                                / 255.0;
                            let (cb, cr) = (100.0 / 255.0 - 0.5, 170.0 / 255.0 - 0.5);
                            [
                                (luma + 1.5748 * cr).clamp(0.0, 1.0),
                                (luma - 0.1873 * cb - 0.4681 * cr).clamp(0.0, 1.0),
                                (luma + 1.8556 * cb).clamp(0.0, 1.0),
                            ]
                        };
                        for channel in 0..3 {
                            let actual = tensor[channel * 64 + oy * 8 + ox];
                            assert!(
                                (actual - expected[channel]).abs() < 0.008,
                                "{w}x{h} {bits}-bit rotation {rotation}, ({ox},{oy}) channel {channel}: {actual} != {}",
                                expected[channel]
                            );
                        }
                    }
                }
            }
        }
        eprintln!("{bits}-bit shape/rotation oracle passed");
    }
    assert_eq!(preprocess.output_buffer, output_buffer);
    assert_eq!(preprocess.staging_buffer, staging_buffer);
}

#[test]
fn malformed_inputs_fail_before_submission_and_valid_input_still_works() {
    let Some(gpu) = gpu() else { return };
    for size in [0, u32::MAX] {
        assert!(WgpuPreprocessor::new(gpu.device(), size).is_err());
    }
    let mut preprocess = WgpuPreprocessor::new(gpu.device(), 8).unwrap();
    let y = plane(&gpu, (6, 4), false);
    let uv = plane(&gpu, (3, 2), true);
    preprocess.tensor.fill(-99.0);
    for (size, rotation) in [((0, 4), 0), ((3, 4), 0), ((4, 6), 0), ((6, 4), 90)] {
        assert!(
            preprocess
                .preprocess(gpu.device(), gpu.queue(), (&y, &uv), size, rotation)
                .is_err()
        );
        assert!(preprocess.tensor.iter().all(|&v| v == -99.0));
    }
    assert!(
        preprocess
            .preprocess(gpu.device(), gpu.queue(), (&y, &y), (6, 4), 0)
            .is_err()
    );
    assert!(
        preprocess
            .preprocess(gpu.device(), gpu.queue(), (&y, &uv), (6, 4), 0)
            .unwrap()
            .iter()
            .all(|v| (0.0..=1.0).contains(v))
    );
}
