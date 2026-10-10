//! BT.709 pixel conversion shared by CPU rendering and image consumers.

/// Convert normalized BT.709 YCbCr to clamped, nonlinear R'G'B' in `[0, 1]`.
///
/// `full_range` selects full-range samples instead of 8-bit limited-range
/// scaling (Y: 16-235, Cb/Cr: 16-240). This performs no transfer-function
/// conversion. The matrix and range scaling match `video_sampling.wgsl`.
#[inline]
pub fn bt709_yuv_to_rgb(y_raw: f64, u_raw: f64, v_raw: f64, full_range: bool) -> [f64; 3] {
    let (y, cb, cr) = if full_range {
        (y_raw, u_raw - 0.5, v_raw - 0.5)
    } else {
        (
            (y_raw - 16.0 / 255.0) * (255.0 / 219.0),
            (u_raw - 128.0 / 255.0) * (255.0 / 224.0),
            (v_raw - 128.0 / 255.0) * (255.0 / 224.0),
        )
    };
    [
        (y + 1.5748 * cr).clamp(0.0, 1.0),
        (y - 0.1873 * cb - 0.4681 * cr).clamp(0.0, 1.0),
        (y + 1.8556 * cb).clamp(0.0, 1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limited_range_neutral_endpoints() {
        for (y, expected) in [(0, 0.0), (16, 0.0), (235, 1.0), (255, 1.0)] {
            let rgb = bt709_yuv_to_rgb(y as f64 / 255.0, 128.0 / 255.0, 128.0 / 255.0, false);
            assert!(rgb.iter().all(|v| (v - expected).abs() < 1e-12));
        }
    }

    #[test]
    fn full_range_neutral_is_unchanged() {
        for y in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(bt709_yuv_to_rgb(y, 0.5, 0.5, true), [y; 3]);
        }
    }

    #[test]
    fn limited_range_colours_round_trip_the_output_converter() {
        for rgb in [[255, 0, 0], [0, 255, 0], [0, 0, 255], [50, 150, 40]] {
            let rgba = [rgb[0], rgb[1], rgb[2], 255].repeat(4);
            let mut nv12 = Vec::new();
            crate::render::nv12_cpu::rgba_to_nv12_into(&rgba, 2, 2, 2, &mut nv12).unwrap();
            let actual = bt709_yuv_to_rgb(
                nv12[0] as f64 / 255.0,
                nv12[4] as f64 / 255.0,
                nv12[5] as f64 / 255.0,
                false,
            );
            for (actual, expected) in actual.into_iter().zip(rgb) {
                // Quantized luma and chroma each contribute up to half a code.
                let tolerance = 0.5 * 255.0 / 219.0 + 1.8556 * 0.5 * 255.0 / 224.0;
                assert!((actual * 255.0 - expected as f64).abs() < tolerance + 0.01);
            }
        }
    }
}
