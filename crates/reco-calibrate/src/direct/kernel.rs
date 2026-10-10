//! Native normalized-patch objective, using the prepared canonical core mapper.

use reco_core::calibration::Calibration;

pub struct Pair {
    pub target: Vec<f64>,
    pub right: Vec<f64>,
}

pub struct Stage {
    width: usize,
    height: usize,
    samples: usize,
    points: Vec<[f64; 2]>,
    pairs: Vec<Pair>,
}

pub fn validate_layout(
    width: usize,
    height: usize,
    points: usize,
    samples: usize,
    frames: usize,
) -> Result<(), String> {
    if width < 2
        || height < 2
        || width.checked_mul(height).is_none_or(|n| n > 4_000_000)
        || !(2..=4096).contains(&samples)
        || points == 0
        || points > 200_000
        || !points.is_multiple_of(samples)
        || !(1..=4).contains(&frames)
        || (width * height + points) * frames > 8_000_000
    {
        return Err("invalid or excessive objective layout".into());
    }
    Ok(())
}

impl Stage {
    pub fn output_len(&self) -> usize {
        self.points.len() * self.pairs.len()
    }

    pub(crate) fn frame_count(&self) -> usize {
        self.pairs.len()
    }

    pub fn new(
        width: usize,
        height: usize,
        samples: usize,
        points: Vec<[f64; 2]>,
        pairs: Vec<Pair>,
    ) -> Result<Self, String> {
        validate_layout(width, height, points.len(), samples, pairs.len())?;
        if points
            .iter()
            .flatten()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || pairs.iter().any(|p| {
                p.target.len() != points.len()
                    || p.right.len() != width * height
                    || p.target.iter().chain(&p.right).any(|x| !x.is_finite())
            })
        {
            return Err("invalid objective data".into());
        }
        Ok(Self {
            width,
            height,
            samples,
            points,
            pairs,
        })
    }

    #[cfg(test)]
    pub fn residual(&self, cal: &Calibration, output: &mut Vec<f64>) -> Result<usize, String> {
        self.residual_checked(cal, output, || Ok(()))
    }

    pub fn residual_checked(
        &self,
        cal: &Calibration,
        output: &mut Vec<f64>,
        check: impl Fn() -> Result<(), String>,
    ) -> Result<usize, String> {
        check()?;
        cal.validate().map_err(|e| e.to_string())?;
        let mapping = cal.source_correspondence(0, 1).map_err(|e| e.to_string())?;
        output.resize(self.points.len() * self.pairs.len(), 4.0);
        output.fill(4.0);
        let mut mapped = vec![[0.0; 2]; self.samples];
        let mut values = vec![0.0; self.samples];
        let mut lost = 0;
        for (patch, points) in self.points.chunks_exact(self.samples).enumerate() {
            check()?;
            let valid = points.iter().zip(&mut mapped).all(|(&point, result)| {
                let Some(uv) = mapping.map(point) else {
                    return false;
                };
                *result = [
                    uv[0] * self.width as f64 - 0.5,
                    uv[1] * self.height as f64 - 0.5,
                ];
                result.iter().all(|v| v.is_finite())
                    && result[0] >= 0.0
                    && result[1] >= 0.0
                    && result[0] < (self.width - 1) as f64
                    && result[1] < (self.height - 1) as f64
            });
            if !valid {
                lost += 1;
                continue;
            }
            let begin = patch * self.samples;
            for (frame, pair) in self.pairs.iter().enumerate() {
                for (value, &point) in values.iter_mut().zip(&mapped) {
                    *value = sample(&pair.right, self.width, self.height, point);
                }
                normalize(&mut values);
                for (j, &value) in values.iter().enumerate() {
                    output[frame * self.points.len() + begin + j] = value - pair.target[begin + j];
                }
            }
        }
        check()?;
        Ok(lost)
    }
}

pub(crate) fn sample<T: Copy + Into<f64>>(
    image: &[T],
    width: usize,
    height: usize,
    point: [f64; 2],
) -> f64 {
    sample_with_gradient(image, width, height, point).0
}

pub(crate) fn sample_with_gradient<T: Copy + Into<f64>>(
    image: &[T],
    width: usize,
    height: usize,
    point: [f64; 2],
) -> (f64, [f64; 2]) {
    let x = point[0].clamp(0.0, width as f64 - 1.001);
    let y = point[1].clamp(0.0, height as f64 - 1.001);
    let ix = x as usize;
    let iy = y as usize;
    let a = x - ix as f64;
    let b = y - iy as f64;
    let [tl, tr, bl, br] = [
        image[iy * width + ix].into(),
        image[iy * width + ix + 1].into(),
        image[(iy + 1) * width + ix].into(),
        image[(iy + 1) * width + ix + 1].into(),
    ];
    let value = (1.0 - b) * ((1.0 - a) * tl + a * tr) + b * ((1.0 - a) * bl + a * br);
    let dx = if x == point[0] {
        (1.0 - b) * (tr - tl) + b * (br - bl)
    } else {
        0.0
    };
    let dy = if y == point[1] {
        (1.0 - a) * (bl - tl) + a * (br - tr)
    } else {
        0.0
    };
    (value, [dx, dy])
}

pub(crate) fn normalize(values: &mut [f64]) -> f64 {
    let origin = values[0];
    let mean = pairwise_sum(values, |x| x - origin) / values.len() as f64;
    let variance = pairwise_sum(values, |x| ((x - origin) - mean).powi(2)) / values.len() as f64;
    let scale = variance.sqrt();
    if scale == 0.0 {
        values.fill(0.0);
        return 0.0;
    }
    for v in values {
        *v = ((*v - origin) - mean) / scale;
    }
    scale
}

// Stable reduction order also makes the numerical port reproducible against the reference.
fn pairwise_sum(values: &[f64], transform: impl Fn(f64) -> f64 + Copy) -> f64 {
    if values.len() < 8 {
        return values
            .iter()
            .copied()
            .map(transform)
            .fold(-0.0, |a, b| a + b);
    }
    if values.len() > 128 {
        let middle = (values.len() / 2) / 8 * 8;
        return pairwise_sum(&values[..middle], transform)
            + pairwise_sum(&values[middle..], transform);
    }
    let mut lanes = std::array::from_fn::<_, 8, _>(|i| transform(values[i]));
    let mut chunks = values[8..].chunks_exact(8);
    for chunk in &mut chunks {
        for (lane, &value) in lanes.iter_mut().zip(chunk) {
            *lane += transform(value);
        }
    }
    let total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    chunks
        .remainder()
        .iter()
        .copied()
        .map(transform)
        .fold(total, |a, b| a + b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_sizes_before_allocating() {
        for (w, h, n, s, f) in [
            (0, 2, 2, 2, 1),
            (2, 1, 2, 2, 1),
            (usize::MAX, 2, 2, 2, 1),
            (2, 2, 0, 2, 1),
            (2, 2, 3, 2, 1),
            (2, 2, 2, 0, 1),
            (2, 2, 2, 2, 0),
            (4000, 4000, 2, 2, 1),
        ] {
            assert!(validate_layout(w, h, n, s, f).is_err());
        }
        assert!(validate_layout(960, 720, 7840, 49, 2).is_ok());
    }

    #[test]
    fn normalization_handles_flat_and_low_contrast() {
        let mut flat = [0.4; 49];
        normalize(&mut flat);
        assert!(flat.iter().all(|x| x.abs() < 1e-12));
        let mut low = [0.4, 0.40001, 0.4, 0.39999];
        normalize(&mut low);
        assert!((low[1] - 2.0_f64.sqrt()).abs() < 1e-11);
        assert!((low[3] + 2.0_f64.sqrt()).abs() < 1e-11);
        let mut high = [0.0, 1.0, 0.0, 1.0];
        normalize(&mut high);
        assert_eq!(high, [-1.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn normalization_preserves_positive_gain_and_offset() {
        for contrast in [0.1, 0.01, 0.0001] {
            let source: Vec<_> = (0..49)
                .map(|i| 0.4 + contrast * (i as f64 * 0.73).sin())
                .collect();
            let mut reference = source.clone();
            normalize(&mut reference);
            for gain in [0.2, 0.65, 1.0, 2.0] {
                let mut changed: Vec<_> = source.iter().map(|x| x * gain + 0.12).collect();
                normalize(&mut changed);
                assert!(
                    reference
                        .iter()
                        .zip(changed)
                        .all(|(a, b)| (a - b).abs() < 1e-10)
                );
            }
        }
    }

    #[test]
    fn bilinear_has_no_subpixel_quantization() {
        let image = [0.0, 1.0, 2.0, 3.0];
        assert!((sample(&image, 2, 2, [0.25, 0.375]) - 1.0).abs() < 1e-15);
        assert!((sample(&image, 2, 2, [0.251, 0.375]) - 1.001).abs() < 1e-15);
    }

    #[test]
    fn rejects_nonfinite_and_mismatched_data() {
        assert!(
            Stage::new(
                2,
                2,
                2,
                vec![[0.5, 0.5]; 2],
                vec![Pair {
                    target: vec![0.; 2],
                    right: vec![f64::NAN; 4]
                }]
            )
            .is_err()
        );
        assert!(
            Stage::new(
                2,
                2,
                2,
                vec![[0.5, 0.5]; 2],
                vec![Pair {
                    target: vec![0.; 1],
                    right: vec![0.; 4]
                }]
            )
            .is_err()
        );
        assert!(
            Stage::new(
                2,
                2,
                2,
                vec![[f64::NAN, 0.5]; 2],
                vec![Pair {
                    target: vec![0.; 2],
                    right: vec![0.; 4]
                }]
            )
            .is_err()
        );
    }

    #[test]
    fn reduction_handles_block_boundaries_and_transforms() {
        for n in [2, 7, 8, 9, 49, 127, 128, 129, 255, 256, 1025, 4096] {
            let values: Vec<_> = (0..n).map(|i| (i % 7) as f64 / 8.0).collect();
            assert_eq!(pairwise_sum(&values, |x| x), values.iter().sum::<f64>());
            assert_eq!(
                pairwise_sum(&values, |x| x * x),
                values.iter().map(|x| x * x).sum::<f64>()
            );
        }
    }
}
