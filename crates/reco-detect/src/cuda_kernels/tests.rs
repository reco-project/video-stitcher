//! Pixel-level CUDA preprocessing checks.

use super::*;
use reco_core::interop::cuda::{
    cuda_mem_alloc, cuda_mem_free, cuda_memcpy_dtoh, cuda_memcpy_htod_2d,
};

#[test]
fn invalid_rgba_layouts_fail_before_cuda_initialization() {
    for (src, pitch, dst, width, height) in [
        (1, 4, 4, 0, 1),
        (1, 4, 4, 1, 0),
        (0, 4, 4, 1, 1),
        (1, 4, 0, 1, 1),
        (1, 4, 1, 1, 1),
        (1, 3, 4, 1, 1),
        (1, usize::MAX, 4, 1, 1),
        (1, 4, 4, u32::MAX, 1),
        (1, 4, 4, 1, u32::MAX),
        (u64::MAX - 1, 4, 4, 1, 1),
        (1, 4, u64::MAX - 3, 1, 1),
    ] {
        assert!(matches!(
            normalize_rgba_to_chw(src, pitch, dst, width, height),
            Err(CudaInteropError::CudaError {
                function: "normalize_rgba_to_chw",
                code: 1
            })
        ));
    }
}

#[test]
#[ignore = "requires NVIDIA CUDA"]
fn rgba_normalization_preserves_channels_stride_and_guards() {
    reco_core::interop::cuda::cuda_ensure_context().unwrap();
    const GUARD: usize = 8;
    for (width, height) in [(1, 1), (17, 9), (31, 33), (640, 640), (1280, 1280)] {
        for padding in [0, 1, 32, 256] {
            let pitch = width as usize * 4 + padding;
            let count = (width * height) as usize;
            let mut rgba = vec![0xA5; pitch * height as usize + 1];
            let mut expected = vec![0.0; 3 * count];
            for y in 0..height as usize {
                for x in 0..width as usize {
                    for channel in 0..3 {
                        let value = (y * 7 + x * 3 + channel * 20) as u8;
                        rgba[1 + y * pitch + x * 4 + channel] = value;
                        expected[channel * count + y * width as usize + x] = value as f32 / 255.0;
                    }
                }
            }
            let source = Buffer::bytes(&rgba);
            let output =
                Buffer::bytes(bytemuck::cast_slice(&vec![-99.0f32; 3 * count + 2 * GUARD]));
            normalize_rgba_to_chw(
                source.0 + 1,
                pitch,
                output.0 + (GUARD * 4) as u64,
                width,
                height,
            )
            .unwrap();
            let actual = output.floats(3 * count + 2 * GUARD);
            assert!(actual[..GUARD].iter().all(|&value| value == -99.0));
            assert!(
                actual[GUARD + 3 * count..]
                    .iter()
                    .all(|&value| value == -99.0)
            );
            for (index, (&actual, expected)) in actual[GUARD..GUARD + 3 * count]
                .iter()
                .zip(expected)
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "{width}x{height}, pitch {pitch}, tensor[{index}]: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn invalid_p010_layouts_fail_before_cuda_initialization() {
    for (src, pitch, dst, width, height) in [
        (2, 2, 1, 0, 1),
        (2, 2, 1, 1, 0),
        (0, 2, 1, 1, 1),
        (2, 2, 0, 1, 1),
        (1, 2, 1, 1, 1),
        (2, 1, 1, 1, 1),
        (2, 3, 1, 1, 1),
        (2, usize::MAX, 1, 1, 1),
        (2, 2, 1, u32::MAX, 1),
        (2, 2, 1, 1, u32::MAX),
        (u64::MAX - 1, 2, 1, 1, 1),
        (2, 2, u64::MAX, 1, 1),
    ] {
        assert!(matches!(
            p010_plane_to_nv12(src, pitch, dst, width, height),
            Err(CudaInteropError::CudaError {
                function: "p010_plane_to_nv12",
                code: 1
            })
        ));
    }
}

#[test]
#[ignore = "requires NVIDIA CUDA"]
fn p010_conversion_matches_high_bytes_with_padding_and_partial_blocks() {
    reco_core::interop::cuda::cuda_ensure_context().unwrap();
    const GUARD: usize = 32;
    for (width, height) in [
        (1, 1),
        (1, 9),
        (31, 7),
        (32, 8),
        (33, 9),
        (127, 1),
        (128, 2),
        (129, 3),
        (257, 256),
        (1920, 1080),
        (3840, 2160),
        (8192, 3),
        (3, 8192),
    ] {
        for padding in [0, 2, 30, 256] {
            let pitch = width as usize * 2 + padding;
            let count = width as usize * height as usize;
            let mut source = vec![0xBEEFu16; 3 + pitch / 2 * height as usize];
            let expected: Vec<u8> = (0..count)
                .map(|i| {
                    let value = (i as u16).wrapping_mul(71).wrapping_add(13);
                    source[3 + (i / width as usize) * (pitch / 2) + i % width as usize] = value;
                    (value >> 8) as u8
                })
                .collect();
            let source = Buffer::bytes(bytemuck::cast_slice(&source));
            let output = Buffer::bytes(&vec![0xA5; count + 2 * GUARD]);
            p010_plane_to_nv12(source.0 + 6, pitch, output.0 + GUARD as u64, width, height)
                .unwrap();
            let mut actual = vec![0u8; count + 2 * GUARD];
            // SAFETY: destination length matches the device allocation, and conversion has completed.
            unsafe {
                cuda_memcpy_dtoh(actual.as_mut_ptr().cast(), output.0, actual.len()).unwrap();
            }
            assert!(
                actual[..GUARD]
                    .iter()
                    .chain(&actual[GUARD + count..])
                    .all(|&v| v == 0xA5)
            );
            assert_eq!(
                &actual[GUARD..GUARD + count],
                &expected,
                "{width}x{height}, pitch {pitch}"
            );
        }
    }
}

struct Buffer(CUdeviceptr);

impl Buffer {
    fn bytes(data: &[u8]) -> Self {
        let buffer = Self(cuda_mem_alloc(data.len()).unwrap());
        cuda_memcpy_htod_2d(
            buffer.0,
            data.len(),
            data.as_ptr(),
            data.len(),
            data.len(),
            1,
        )
        .unwrap();
        buffer
    }

    fn floats(&self, len: usize) -> Vec<f32> {
        let mut data = vec![0.0f32; len];
        // SAFETY: the allocation and destination contain `len` f32 values.
        unsafe {
            cuda_memcpy_dtoh(data.as_mut_ptr().cast(), self.0, len * 4).unwrap();
        }
        data
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        cuda_mem_free(self.0).unwrap();
    }
}

fn check_tensor(sizes: &[(u32, u32)], y_pitch: u32, uv_pitch: u32, p010: bool) {
    reco_core::interop::cuda::cuda_ensure_context().unwrap();
    const DST: u32 = 16;
    let poison = vec![-99.0f32; (DST * DST * 3) as usize];
    let tensor = Buffer::bytes(bytemuck::cast_slice(&poison));
    for &(width, height) in sizes {
        let mut y = vec![0; (y_pitch * height) as usize];
        let mut uv = vec![0; (uv_pitch * height / 2) as usize];
        for row in 0..height {
            for col in 0..width {
                y[(row * y_pitch + col) as usize] = (50 + row * 7 + col * 3) as u8;
            }
        }
        for row in 0..height / 2 {
            for col in 0..width / 2 {
                uv[(row * uv_pitch + col * 2) as usize] = (100 + row * 7) as u8;
                uv[(row * uv_pitch + col * 2 + 1) as usize] = (140 + col * 5) as u8;
            }
        }
        let (y, uv, y_stride, uv_stride) = if p010 {
            let y16: Vec<u16> = y.iter().map(|&v| u16::from(v) << 8).collect();
            let uv16: Vec<u16> = uv.iter().map(|&v| u16::from(v) << 8).collect();
            let y16 = Buffer::bytes(bytemuck::cast_slice(&y16));
            let uv16 = Buffer::bytes(bytemuck::cast_slice(&uv16));
            let y = Buffer::bytes(&vec![0; (width * height) as usize]);
            let uv = Buffer::bytes(&vec![0; (width * height / 2) as usize]);
            p010_plane_to_nv12(y16.0, (y_pitch * 2) as usize, y.0, width, height).unwrap();
            p010_plane_to_nv12(uv16.0, (uv_pitch * 2) as usize, uv.0, width, height / 2).unwrap();
            (y, uv, width, width)
        } else {
            (Buffer::bytes(&y), Buffer::bytes(&uv), y_pitch, uv_pitch)
        };
        let scale = (DST as f32 / width as f32).min(DST as f32 / height as f32);
        let content_w = (width as f32 * scale).round() as u32;
        let content_h = (height as f32 * scale).round() as u32;
        let pad_x = (DST - content_w) / 2;
        let pad_y = (DST - content_h) / 2;
        for rotation in [0, 180] {
            nv12_to_rgb_chw_fullrange(
                y.0, uv.0, tensor.0, y_stride, uv_stride, width, height, DST, DST, pad_x, pad_y,
                scale, rotation,
            )
            .unwrap();
            let result = tensor.floats(poison.len());
            for channel in 0..3 {
                for oy in 0..DST {
                    for ox in 0..DST {
                        let expected = if ox < pad_x
                            || ox >= pad_x + content_w
                            || oy < pad_y
                            || oy >= pad_y + content_h
                        {
                            114.0 / 255.0
                        } else {
                            let mut x = (((ox - pad_x) as f32 / scale) as u32).min(width - 1);
                            let mut y = (((oy - pad_y) as f32 / scale) as u32).min(height - 1);
                            if rotation == 180 {
                                x = width - 1 - x;
                                y = height - 1 - y;
                            }
                            let luma = (50 + y * 7 + x * 3) as f32;
                            let u = (100 + y / 2 * 7) as f32 - 128.0;
                            let v = (140 + x / 2 * 5) as f32 - 128.0;
                            let rgb = [
                                luma + 1.5748 * v,
                                luma - 0.1873 * u - 0.4681 * v,
                                luma + 1.8556 * u,
                            ];
                            rgb[channel as usize].clamp(0.0, 255.0) / 255.0
                        };
                        let actual = result[(channel * DST * DST + oy * DST + ox) as usize];
                        assert!(
                            (actual - expected).abs() < 1e-6,
                            "{width}x{height}, rotation {rotation}, channel {channel}, ({ox},{oy}): {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires NVIDIA CUDA"]
fn nv12_letterbox_overwrites_padding_when_shape_changes() {
    check_tensor(
        &[(8, 4), (4, 8), (8, 8), (6, 4), (4, 6), (8, 4)],
        16,
        16,
        false,
    );
}

#[test]
#[ignore = "requires NVIDIA CUDA"]
fn nv12_planes_have_independent_strides() {
    check_tensor(&[(8, 8)], 16, 24, false);
}

#[test]
#[ignore = "requires NVIDIA CUDA"]
fn p010_strides_and_shape_changes_preserve_tensor_pixels() {
    check_tensor(&[(8, 4), (4, 8), (8, 8), (6, 4), (4, 6)], 16, 24, true);
}
