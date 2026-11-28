//! Nonlinear YCbCr matrix and range conversion shared by render and delivery paths.

/// Supported YCbCr matrix coefficients (ITU-T H.273).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum YuvMatrix {
    /// BT.709.
    #[default]
    Bt709 = 0,
    /// BT.2020 non-constant luminance.
    Bt2020Ncl = 1,
    /// Conventional BT.601 (SMPTE 170M or BT.470BG).
    Bt601 = 2,
}

impl YuvMatrix {
    pub(crate) fn luma(self) -> [f32; 3] {
        match self {
            Self::Bt709 => [0.2126, 0.7152, 0.0722],
            Self::Bt2020Ncl => [0.2627, 0.6780, 0.0593],
            Self::Bt601 => [0.299, 0.587, 0.114],
        }
    }
}

/// Storage of the normalized YCbCr samples supplied to the renderer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum YuvSampleFormat {
    /// 8-bit samples normalized by 255.
    #[default]
    Unorm8,
    /// 10-bit samples in the upper bits of a 16-bit UNORM plane.
    P010,
}

/// Matrix, range, and sample storage of a YCbCr image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct YuvColor {
    /// Matrix relating nonlinear RGB to YCbCr.
    pub matrix: YuvMatrix,
    /// Use the full code range, with chroma centered at 128 (8-bit) or 512 (10-bit).
    pub full_range: bool,
    /// Sample storage and normalization.
    pub samples: YuvSampleFormat,
}

impl YuvColor {
    /// Rendered delivery uses 8-bit limited-range samples in the source matrix.
    pub const fn delivery(matrix: YuvMatrix) -> Self {
        Self {
            matrix,
            full_range: false,
            samples: YuvSampleFormat::Unorm8,
        }
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn flags(self) -> u32 {
        u32::from(self.full_range)
            | ((self.matrix as u32) << 1)
            | (u32::from(self.samples == YuvSampleFormat::P010) << 3)
    }

    pub(crate) fn encode_range(self) -> (f32, f32, f32) {
        if self.full_range {
            (0.0, 255.0, 255.0)
        } else {
            (16.0, 219.0, 224.0)
        }
    }

    pub(crate) fn encode_coefficients(self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let [kr, kg, kb] = self.matrix.luma();
        let (cb, cr) = match self.matrix {
            YuvMatrix::Bt709 => ([-0.1146, -0.3854, 0.5], [0.5, -0.4542, -0.0458]),
            _ => (
                [-kr / (2.0 * (1.0 - kb)), -kg / (2.0 * (1.0 - kb)), 0.5],
                [0.5, -kg / (2.0 * (1.0 - kr)), -kb / (2.0 * (1.0 - kr))],
            ),
        };
        ([kr, kg, kb], cb, cr)
    }
}

/// Convert normalized YCbCr samples to clamped nonlinear R'G'B' in `[0, 1]`.
///
/// Applies only matrix coefficients and code-range scaling. P010 samples use
/// `(code << 6) / 65535`, including when interpolated between texels.
#[inline]
pub fn yuv_to_rgb(y: f64, cb: f64, cr: f64, color: YuvColor) -> [f64; 3] {
    YuvToRgb::new(color).convert(y, cb, cr)
}

#[derive(Clone, Copy)]
pub(crate) struct YuvToRgb {
    black: f64,
    neutral: f64,
    y_scale: f64,
    c_scale: f64,
    matrix: [f64; 4],
}

impl YuvToRgb {
    pub(crate) fn new(color: YuvColor) -> Self {
        let (black, neutral, y_scale, c_scale) = match (color.samples, color.full_range) {
            (YuvSampleFormat::Unorm8, false) => {
                (16.0 / 255.0, 128.0 / 255.0, 255.0 / 219.0, 255.0 / 224.0)
            }
            (YuvSampleFormat::Unorm8, true) => (0.0, 128.0 / 255.0, 1.0, 1.0),
            (YuvSampleFormat::P010, full) => {
                // The R16Unorm plane stores 64 times each code.
                let codes_per_unorm = 65535.0 / 64.0;
                let neutral = 512.0 / codes_per_unorm;
                if full {
                    (
                        0.0,
                        neutral,
                        codes_per_unorm / 1023.0,
                        codes_per_unorm / 1023.0,
                    )
                } else {
                    (
                        64.0 / codes_per_unorm,
                        neutral,
                        codes_per_unorm / 876.0,
                        codes_per_unorm / 896.0,
                    )
                }
            }
        };
        let matrix = match color.matrix {
            YuvMatrix::Bt709 => [1.5748, 0.1873, 0.4681, 1.8556],
            YuvMatrix::Bt2020Ncl => [1.4746, 0.1645531268, 0.5713531268, 1.8814],
            YuvMatrix::Bt601 => [1.402, 0.3441362862, 0.7141362862, 1.772],
        };
        Self {
            black,
            neutral,
            y_scale,
            c_scale,
            matrix,
        }
    }

    #[inline]
    pub(crate) fn convert(self, y: f64, cb: f64, cr: f64) -> [f64; 3] {
        let y = (y - self.black) * self.y_scale;
        let cb = (cb - self.neutral) * self.c_scale;
        let cr = (cr - self.neutral) * self.c_scale;
        let [r_cr, g_cb, g_cr, b_cb] = self.matrix;
        [
            (y + r_cr * cr).clamp(0.0, 1.0),
            (y - g_cb * cb - g_cr * cr).clamp(0.0, 1.0),
            (y + b_cb * cb).clamp(0.0, 1.0),
        ]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    // H.273 YCbCr codes rounded to the nearest integer for six uniform RGB patches.
    pub(crate) const RGB_PATCHES: [[u8; 3]; 6] = [
        [0, 0, 0],
        [255, 255, 255],
        [128, 128, 128],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ];

    pub(crate) fn vectors() -> impl Iterator<Item = (YuvColor, [[u8; 3]; 6])> {
        [
            (
                YuvMatrix::Bt709,
                false,
                [
                    [16, 128, 128],
                    [235, 128, 128],
                    [126, 128, 128],
                    [63, 102, 240],
                    [173, 42, 26],
                    [32, 240, 118],
                ],
            ),
            (
                YuvMatrix::Bt709,
                true,
                [
                    [0, 128, 128],
                    [255, 128, 128],
                    [128, 128, 128],
                    [54, 99, 255],
                    [182, 30, 12],
                    [18, 255, 116],
                ],
            ),
            (
                YuvMatrix::Bt2020Ncl,
                false,
                [
                    [16, 128, 128],
                    [235, 128, 128],
                    [126, 128, 128],
                    [74, 97, 240],
                    [164, 47, 25],
                    [29, 240, 119],
                ],
            ),
            (
                YuvMatrix::Bt2020Ncl,
                true,
                [
                    [0, 128, 128],
                    [255, 128, 128],
                    [128, 128, 128],
                    [67, 92, 255],
                    [173, 36, 11],
                    [15, 255, 118],
                ],
            ),
            (
                YuvMatrix::Bt601,
                false,
                [
                    [16, 128, 128],
                    [235, 128, 128],
                    [126, 128, 128],
                    [81, 90, 240],
                    [145, 54, 34],
                    [41, 240, 110],
                ],
            ),
            (
                YuvMatrix::Bt601,
                true,
                [
                    [0, 128, 128],
                    [255, 128, 128],
                    [128, 128, 128],
                    [76, 85, 255],
                    [150, 44, 21],
                    [29, 255, 107],
                ],
            ),
        ]
        .into_iter()
        .map(|(matrix, full_range, codes)| {
            (
                YuvColor {
                    matrix,
                    full_range,
                    ..Default::default()
                },
                codes,
            )
        })
    }

    #[test]
    fn independent_color_vectors_match_cpu_decode_and_encode() {
        for (color, codes) in vectors() {
            for (expected_rgb, expected_yuv) in RGB_PATCHES.into_iter().zip(codes) {
                let [y, cb, cr] = expected_yuv.map(|c| f64::from(c) / 255.0);
                let rgb = yuv_to_rgb(y, cb, cr, color).map(|c| (c * 255.0).round() as u8);
                for (actual, expected) in rgb.into_iter().zip(expected_rgb) {
                    assert!(
                        actual.abs_diff(expected) <= 2,
                        "{color:?}: {rgb:?} vs {expected_rgb:?}"
                    );
                }

                let [r, g, b] = expected_rgb;
                let mut nv12 = Vec::new();
                crate::render::nv12_cpu::rgba_to_nv12_with_color_into(
                    &[r, g, b, 255].repeat(4),
                    2,
                    2,
                    2,
                    color,
                    &mut nv12,
                )
                .unwrap();
                assert_eq!([nv12[0], nv12[4], nv12[5]], expected_yuv, "{color:?}");
            }
        }
    }

    pub(crate) const P010_VECTORS: [(bool, [u16; 3], [u8; 3]); 8] = [
        (false, [64, 512, 512], [0; 3]),
        (false, [940, 512, 512], [255; 3]),
        (false, [502, 512, 512], [128; 3]),
        (false, [385, 221, 920], [255, 41, 0]),
        (true, [0, 512, 512], [0; 3]),
        (true, [1023, 512, 512], [255; 3]),
        (true, [512, 512, 512], [128; 3]),
        (true, [500, 260, 855], [251, 86, 6]),
    ];

    #[test]
    fn p010_uses_shifted_unorm_codes_for_range_and_chroma() {
        let unorm = |code: u16| f64::from(code << 6) / 65535.0;
        for (full_range, code, expected) in P010_VECTORS {
            let color = YuvColor {
                matrix: YuvMatrix::Bt2020Ncl,
                full_range,
                samples: YuvSampleFormat::P010,
            };
            let [y, cb, cr] = code.map(unorm);
            let rgb = yuv_to_rgb(y, cb, cr, color);
            for (actual, expected) in rgb.into_iter().zip(expected) {
                assert!(
                    (actual * 255.0 - f64::from(expected)).abs() <= 0.51,
                    "{code:?}: {rgb:?}"
                );
            }
        }
        let color = YuvColor {
            samples: YuvSampleFormat::P010,
            ..Default::default()
        };
        let rgb = yuv_to_rgb(unorm(502), unorm(512), unorm(512), color);
        assert!(rgb.iter().all(|x| (x - 0.5).abs() < 1e-12), "{rgb:?}");
    }

    #[test]
    fn limited_range_neutral_endpoints() {
        for (y, expected) in [(0, 0.0), (16, 0.0), (235, 1.0), (255, 1.0)] {
            let rgb = yuv_to_rgb(
                y as f64 / 255.0,
                128.0 / 255.0,
                128.0 / 255.0,
                YuvColor::default(),
            );
            assert!(rgb.iter().all(|v| (v - expected).abs() < 1e-12));
        }
    }

    #[test]
    fn full_range_neutral_is_unchanged() {
        for y in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(
                yuv_to_rgb(
                    y,
                    128.0 / 255.0,
                    128.0 / 255.0,
                    YuvColor {
                        full_range: true,
                        ..Default::default()
                    }
                ),
                [y; 3]
            );
        }
    }
}
