//! CPU RGBA -> NV12 conversion for the software delivery path.
//!
//! Mirrors the GPU compute converter byte-for-byte (within float
//! rounding): BT.709 sRGB to limited-range YCbCr, one chroma sample
//! per 2x2 block via plain averaging, `+0.5`-then-truncate rounding.
//!
//! SYNC_WITH: shaders/rgba_to_nv12.wgsl - the coefficient set, the
//! chroma averaging order, and the rounding must match exactly; the
//! GPU-vs-CPU oracle test pins the agreement to 1 LSB.

use thiserror::Error;

/// Errors from the CPU NV12 conversion.
#[derive(Debug, Clone, Error)]
pub enum Nv12CpuError {
    /// NV12 needs even dimensions (one chroma sample per 2x2 block).
    #[error("NV12 dimensions must be even, got {width}x{height}")]
    OddDimensions {
        /// Requested output width.
        width: u32,
        /// Requested output height.
        height: u32,
    },
    /// The RGBA buffer is too small for the requested conversion.
    #[error("RGBA buffer too small: {actual} bytes, need {expected}")]
    ShortBuffer {
        /// Bytes required for `stride * height * 4`.
        expected: usize,
        /// Bytes the supplied buffer actually contains.
        actual: usize,
    },
}

/// NV12-legal dimensions for a render output: width rounded down to a
/// multiple of 4, height to a multiple of 2.
///
/// The single home for the rounding rule the GPU converter and the CPU
/// kernel share; both delivery paths hand sinks identically-sized
/// frames for the same viewport.
pub fn nv12_dims(width: u32, height: u32) -> (u32, u32) {
    (width & !3, height & !1)
}

/// Convert sRGB-domain RGBA bytes to NV12 (BT.709 limited range).
///
/// Reads a `width x height` region from `rgba`, whose rows are
/// `stride_px` pixels wide (`stride_px >= width`; pass the render
/// width when converting a rounded region of a wider frame). Output
/// layout matches the GPU converter: Y plane (`width * height`
/// bytes) followed by the interleaved UV plane (`width * height / 2`
/// bytes). `out` is resized to fit and fully overwritten.
pub fn rgba_to_nv12_into(
    rgba: &[u8],
    stride_px: u32,
    width: u32,
    height: u32,
    out: &mut Vec<u8>,
) -> Result<(), Nv12CpuError> {
    if width == 0
        || height == 0
        || !width.is_multiple_of(2)
        || !height.is_multiple_of(2)
        || stride_px < width
    {
        return Err(Nv12CpuError::OddDimensions { width, height });
    }
    let stride = stride_px as usize;
    let (w, h) = (width as usize, height as usize);
    let expected = stride * h * 4;
    if rgba.len() < expected {
        return Err(Nv12CpuError::ShortBuffer {
            expected,
            actual: rgba.len(),
        });
    }

    let y_size = w * h;
    out.resize(y_size + y_size / 2, 0);
    let (y_plane, uv_plane) = out.split_at_mut(y_size);

    // sRGB bytes as-is (no transfer decode), scaled to [0,1] exactly
    // like the GPU's Rgba8Unorm texture load.
    let f = |b: u8| b as f32 / 255.0;
    let px_rgb = |x: usize, y: usize| {
        let i = (y * stride + x) * 4;
        (f(rgba[i]), f(rgba[i + 1]), f(rgba[i + 2]))
    };

    for row in 0..h {
        for col in 0..w {
            let (r, g, b) = px_rgb(col, row);
            let y = 16.0 + 219.0 * (0.2126 * r + 0.7152 * g + 0.0722 * b);
            y_plane[row * w + col] = (y + 0.5).clamp(0.0, 255.0) as u8;
        }

        // One (Cb, Cr) pair per 2x2 block, averaging in the shader's
        // exact order: top-left + top-right + bottom-left + bottom-right.
        if row.is_multiple_of(2) {
            for pair in 0..w / 2 {
                let x = pair * 2;
                let (tl_r, tl_g, tl_b) = px_rgb(x, row);
                let (tr_r, tr_g, tr_b) = px_rgb(x + 1, row);
                let (bl_r, bl_g, bl_b) = px_rgb(x, row + 1);
                let (br_r, br_g, br_b) = px_rgb(x + 1, row + 1);
                let r = (tl_r + tr_r + bl_r + br_r) * 0.25;
                let g = (tl_g + tr_g + bl_g + br_g) * 0.25;
                let b = (tl_b + tr_b + bl_b + br_b) * 0.25;

                let cb = 128.0 + 224.0 * (-0.1146 * r - 0.3854 * g + 0.5 * b);
                let cr = 128.0 + 224.0 * (0.5 * r - 0.4542 * g - 0.0458 * b);
                let uv_idx = (row / 2) * w + x;
                uv_plane[uv_idx] = (cb + 0.5).clamp(0.0, 255.0) as u8;
                uv_plane[uv_idx + 1] = (cr + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(r: u8, g: u8, b: u8, w: usize, h: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            v.extend_from_slice(&[r, g, b, 255]);
        }
        v
    }

    fn convert(rgba: &[u8], stride: u32, w: u32, h: u32) -> Vec<u8> {
        let mut out = Vec::new();
        rgba_to_nv12_into(rgba, stride, w, h, &mut out).unwrap();
        out
    }

    #[test]
    fn black_maps_to_limited_range_floor() {
        let nv12 = convert(&solid(0, 0, 0, 4, 2), 4, 4, 2);
        assert!(nv12[..8].iter().all(|&y| y == 16), "black Y = 16");
        assert!(nv12[8..].iter().all(|&uv| uv == 128), "neutral chroma");
    }

    #[test]
    fn white_maps_to_limited_range_ceiling() {
        let nv12 = convert(&solid(255, 255, 255, 4, 2), 4, 4, 2);
        assert!(nv12[..8].iter().all(|&y| y == 235), "white Y = 235");
        assert!(nv12[8..].iter().all(|&uv| uv == 128), "neutral chroma");
    }

    #[test]
    fn pure_red_matches_bt709() {
        // Y = 16 + 219*0.2126 = 62.56 -> 63; Cb = 128 - 224*0.1146 ->
        // 102.3 -> 102; Cr = 128 + 224*0.5 = 240.
        let nv12 = convert(&solid(255, 0, 0, 4, 2), 4, 4, 2);
        assert_eq!(nv12[0], 63);
        assert_eq!(nv12[8], 102);
        assert_eq!(nv12[9], 240);
    }

    #[test]
    fn chroma_averages_the_2x2_block() {
        // Left 2x2 block: one white pixel, three black -> Y avg not
        // relevant, chroma stays neutral (grey axis); instead use
        // red/blue split so the average is measurable.
        // Top row red, bottom row blue: avg = (0.5, 0, 0.5).
        let w = 2usize;
        let mut rgba = Vec::new();
        rgba.extend_from_slice(&[255, 0, 0, 255, 255, 0, 0, 255]); // row 0: red red
        rgba.extend_from_slice(&[0, 0, 255, 255, 0, 0, 255, 255]); // row 1: blue blue
        let mut out = Vec::new();
        rgba_to_nv12_into(&rgba, w as u32, w as u32, 2, &mut out).unwrap();
        // avg rgb = (0.5, 0, 0.5): Cb = 128 + 224*(-0.0573 + 0.25) ->
        // 171.2 -> 171; Cr = 128 + 224*(0.25 - 0.0229) -> 178.9 -> 179.
        assert_eq!(out[4], 171, "Cb of the averaged block");
        assert_eq!(out[5], 179, "Cr of the averaged block");
    }

    #[test]
    fn stride_reads_the_left_region() {
        // 6px-wide source, convert the left 4x2: right 2 columns are
        // poison values that must not leak into the output.
        let mut rgba = solid(100, 100, 100, 6, 2);
        for row in 0..2 {
            for col in 4..6 {
                let i = (row * 6 + col) * 4;
                rgba[i..i + 4].copy_from_slice(&[255, 0, 255, 255]);
            }
        }
        let nv12 = convert(&rgba, 6, 4, 2);
        let uniform = convert(&solid(100, 100, 100, 4, 2), 4, 4, 2);
        assert_eq!(nv12, uniform, "poison columns leaked through stride");
    }

    #[test]
    fn rejects_odd_dimensions_and_short_buffers() {
        let mut out = Vec::new();
        assert!(rgba_to_nv12_into(&solid(0, 0, 0, 3, 2), 3, 3, 2, &mut out).is_err());
        assert!(rgba_to_nv12_into(&[0u8; 8], 4, 4, 2, &mut out).is_err());
    }

    #[test]
    fn nv12_dims_rounds_like_the_gpu_converter() {
        assert_eq!(nv12_dims(1920, 1080), (1920, 1080));
        assert_eq!(nv12_dims(1923, 1081), (1920, 1080));
        assert_eq!(nv12_dims(2, 3), (0, 2));
    }
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_oracle {
    use super::rgba_to_nv12_into;
    use crate::gpu::nv12_converter::Nv12Converter;
    use crate::stitch::test_support::gpu_or_skip;

    /// The CPU kernel and the GPU compute shader run the same f32 math
    /// on the same bytes; only FMA contraction can split them, so the
    /// agreement bound is 1 LSB on every byte (unlike the full-stitch
    /// oracle, there is no sampling or rasterization noise to admit).
    #[test]
    fn cpu_kernel_agrees_with_the_gpu_converter() {
        let Some(gpu) = gpu_or_skip() else { return };
        let (w, h) = (64u32, 32u32);

        // Deterministic pseudo-random RGBA covering the value range.
        let mut state = 0x1234_5678u32;
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for px in rgba.chunks_exact_mut(4) {
            for channel in px.iter_mut().take(3) {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *channel = (state >> 24) as u8;
            }
            px[3] = 255;
        }

        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("nv12-oracle-input"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        let mut converter = Nv12Converter::new(&gpu, w, h).expect("converter");
        let empty = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
            .finish();
        assert!(
            converter
                .convert_and_readback(&gpu, &texture, empty)
                .expect("convert")
                .is_none(),
            "first call is a warmup"
        );
        let gpu_nv12 = converter
            .flush_pending(&gpu)
            .expect("flush")
            .expect("one pending frame")
            .to_vec();

        let mut cpu_nv12 = Vec::new();
        rgba_to_nv12_into(&rgba, w, w, h, &mut cpu_nv12).expect("cpu kernel");

        assert_eq!(gpu_nv12.len(), cpu_nv12.len(), "NV12 layout mismatch");
        let max = gpu_nv12
            .iter()
            .zip(&cpu_nv12)
            .map(|(g, c)| (*g as i32 - *c as i32).abs())
            .max()
            .unwrap();
        let diff_count = gpu_nv12
            .iter()
            .zip(&cpu_nv12)
            .filter(|(g, c)| g != c)
            .count();
        eprintln!(
            "[nv12 oracle] {} bytes, {diff_count} differ, max diff {max}",
            gpu_nv12.len()
        );
        assert!(max <= 1, "CPU/GPU NV12 divergence beyond 1 LSB: {max}");
    }
}
