//! Independent CPU reference and layout checks for the P010 shader.

use super::*;
use crate::gpu::rgba_readback::{RgbaReadback, RgbaReadbackError};
use crate::render::color::{YuvMatrix, YuvSampleFormat};

fn p010_reference(rgb: &[[u16; 3]], width: usize, height: usize, color: YuvColor) -> Vec<u16> {
    let (kr, kb) = match color.matrix {
        YuvMatrix::Bt709 => (0.2126, 0.0722),
        YuvMatrix::Bt2020Ncl => (0.2627, 0.0593),
        YuvMatrix::Bt601 => (0.299, 0.114),
    };
    let (black, y_span, c_span) = if color.full_range {
        (0.0, 1023.0, 1023.0)
    } else {
        (64.0, 876.0, 896.0)
    };
    let quantize = |code: f64| code.round().clamp(0.0, 1023.0) as u16;
    let ycbcr: Vec<_> = rgb
        .iter()
        .map(|sample| {
            let [r, g, b] = sample.map(|code| f64::from(code) / 1023.0);
            let y = kr * r + (1.0 - kr - kb) * g + kb * b;
            (
                y,
                (b - y) / (2.0 * (1.0 - kb)),
                (r - y) / (2.0 * (1.0 - kr)),
            )
        })
        .collect();
    let mut codes = Vec::with_capacity(width * height * 3 / 2);
    codes.extend(ycbcr.iter().map(|&(y, _, _)| quantize(black + y_span * y)));
    for row in ycbcr.chunks_exact(width * 2) {
        for x in (0..width).step_by(2) {
            let mut chroma = [0.0; 2];
            for i in [x, x + 1, x + width, x + width + 1] {
                chroma[0] += row[i].1;
                chroma[1] += row[i].2;
            }
            codes.extend(chroma.map(|c| quantize(512.0 + c_span * c / 4.0)));
        }
    }
    codes
}

#[test]
fn gpu_p010_matches_cpu_reference_in_all_matrices_and_ranges() {
    let Some(gpu) = crate::stitch::test_support::gpu_or_skip() else {
        return;
    };
    // Exercise partial workgroups, even widths not divisible by four,
    // heterogeneous 2x2 blocks, neutral ramps and both chroma extremes.
    let size = wgpu::Extent3d {
        width: 38,
        height: 10,
        depth_or_array_layers: 1,
    };
    let colors = [
        [0; 3],
        [1023; 3],
        [512; 3],
        [513; 3],
        [1023, 0, 0],
        [0, 1023, 1023],
        [0, 0, 1023],
        [1023, 1023, 0],
    ];
    let rgb: Vec<_> = (0..size.height)
        .flat_map(|y| {
            (0..size.width).map(move |x| {
                if y < 2 && x < 16 {
                    colors[(x / 2) as usize]
                } else {
                    [233 * x + 91 * y, 599 * y + 79 * x, 885 * x + 199 * y]
                        .map(|code| (code % 1024) as u16)
                }
            })
        })
        .collect();
    let packed: Vec<_> = rgb
        .iter()
        .flat_map(|&[r, g, b]| {
            (u32::from(r) | (u32::from(g) << 10) | (u32::from(b) << 20) | (3 << 30)).to_le_bytes()
        })
        .collect();
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("P010 oracle input"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgb10a2Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue().write_texture(
        texture.as_image_copy(),
        &packed,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.width * 4),
            rows_per_image: Some(size.height),
        },
        size,
    );
    let commands = || {
        gpu.device()
            .create_command_encoder(&Default::default())
            .finish()
    };

    let format = DeliveryFormat::P010;
    for matrix in [YuvMatrix::Bt709, YuvMatrix::Bt2020Ncl, YuvMatrix::Bt601] {
        for full_range in [false, true] {
            let color = YuvColor {
                matrix,
                full_range,
                samples: YuvSampleFormat::P010,
            };
            let mut converter =
                YuvConverter::with_format(&gpu, size.width, size.height, format, color).unwrap();
            assert!(
                converter
                    .convert_and_readback(&gpu, &texture, commands())
                    .unwrap()
                    .is_none()
            );
            let bytes = converter.flush_pending(&gpu).unwrap().unwrap();
            assert_eq!(bytes.len(), size.width as usize * size.height as usize * 3);
            let expected = p010_reference(&rgb, size.width as usize, size.height as usize, color);
            let actual: Vec<_> = bytes
                .chunks_exact(2)
                .map(|bytes| {
                    let word = u16::from_le_bytes([bytes[0], bytes[1]]);
                    assert_eq!(word & 63, 0);
                    word >> 6
                })
                .collect();
            assert_eq!(actual.len(), expected.len());
            for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    actual.abs_diff(expected) <= 1,
                    "{color:?} sample {index}: {actual} != {expected}"
                );
            }
            assert_eq!(
                (actual[0], actual[2]),
                if full_range { (0, 1023) } else { (64, 940) }
            );
            let chroma = &actual[(size.width * size.height) as usize..];
            assert_eq!(&chroma[..8], &[512; 8]);
            assert_eq!(
                (chroma[9], chroma[11], chroma[12], chroma[14]),
                if full_range {
                    (1023, 1, 1023, 1)
                } else {
                    (960, 64, 960, 64)
                }
            );
            assert!(
                actual[4] < actual[6],
                "adjacent ten-bit greys must stay distinct"
            );
            assert!(converter.flush_pending(&gpu).unwrap().is_none());
        }
    }

    // The 8-bit RGBA API must never reinterpret a packed RGB10A2 texel.
    let mut readback = RgbaReadback::new(&gpu, size.width, size.height).unwrap();
    assert!(matches!(
        readback.readback(&gpu, &texture, commands()),
        Err(RgbaReadbackError::PackedRgb10a2)
    ));
    assert!(readback.flush_pending(&gpu).unwrap().is_none());
}
