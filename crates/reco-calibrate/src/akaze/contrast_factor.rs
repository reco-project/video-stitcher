use super::image::{GrayFloatImage, gaussian_blur};
use log::*;

/// Compute contrast factor from pre-computed Scharr derivatives.
///
/// Accepts the horizontal and vertical Scharr derivatives (`Lx`, `Ly`)
/// directly, avoiding redundant gaussian+scharr computation when the
/// caller already has them (e.g. from `create_nonlinear_scale_space`).
#[allow(non_snake_case)]
pub fn compute_contrast_factor_from_gradients(
    Lx: &GrayFloatImage,
    Ly: &GrayFloatImage,
    percentile: f64,
    num_bins: usize,
) -> f64 {
    let mut num_points: f64 = 0.0;
    let mut histogram = vec![0; num_bins];
    let hmax = (1..Lx.height().saturating_sub(1))
        .flat_map(|y| (1..Lx.width().saturating_sub(1)).map(move |x| (x, y)))
        .map(|(x, y)| Lx.get(x, y).powi(2) as f64 + Ly.get(x, y).powi(2) as f64)
        .map(float_ord::FloatOrd)
        .max()
        .unwrap_or(float_ord::FloatOrd(0.0))
        .0
        .sqrt();
    for y in 1..(Lx.height() - 1) {
        for x in 1..(Lx.width() - 1) {
            let modg = (Lx.get(x, y).powi(2) as f64 + Ly.get(x, y).powi(2) as f64).sqrt();
            if modg != 0.0 {
                let mut bin_number = f64::floor((num_bins as f64) * (modg / hmax)) as usize;
                if bin_number == num_bins {
                    bin_number -= 1;
                }
                histogram[bin_number] += 1;
                num_points += 1f64;
            }
        }
    }
    let threshold: usize = (num_points * percentile) as usize;
    let mut k: usize = 0;
    let mut num_elements: usize = 0;
    while num_elements < threshold && k < num_bins {
        num_elements += histogram[k];
        k += 1;
    }
    debug!(
        "hmax: {}, threshold: {}, num_elements: {}",
        hmax, threshold, num_elements
    );
    let result = if num_elements >= threshold {
        hmax * (k as f64) / (num_bins as f64)
    } else {
        0.03
    };
    // Clamp to a minimum to prevent zero contrast factor on blank/uniform
    // images, which would cause division-by-zero (Infinity) in pm_g2.
    result.max(1e-6)
}

/// Compute contrast factor by blurring the image and computing Scharr
/// derivatives internally.
///
/// This is the original entry point, kept for cases where pre-computed
/// derivatives are not available.
#[allow(non_snake_case)]
pub fn compute_contrast_factor(
    image: &GrayFloatImage,
    percentile: f64,
    gradient_histogram_scale: f64,
    num_bins: usize,
) -> f64 {
    let gaussian = gaussian_blur(image, gradient_histogram_scale as f32);
    let Lx = super::derivatives::scharr_horizontal(&gaussian, 1);
    let Ly = super::derivatives::scharr_vertical(&gaussian, 1);
    compute_contrast_factor_from_gradients(&Lx, &Ly, percentile, num_bins)
}
