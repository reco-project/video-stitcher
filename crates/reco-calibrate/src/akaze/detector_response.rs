use super::derivatives::ScharrScratch;
use super::evolution::EvolutionStep;
use super::image::GrayFloatImage;
use super::{Akaze, derivatives};
use ndarray::azip;

impl Akaze {
    fn compute_multiscale_derivatives(&self, evolutions: &mut [EvolutionStep]) {
        if evolutions.is_empty() {
            return;
        }
        // Pre-allocate a single scratch buffer, reused across all Scharr
        // calls for every evolution step.
        let (h, w) = (evolutions[0].lt.height(), evolutions[0].lt.width());
        let mut scratch = ScharrScratch::new(h, w);

        for evolution in evolutions.iter_mut() {
            let ratio = 2.0f64.powi(evolution.octave as i32);
            let sigma_size = f64::round(evolution.esigma * self.derivative_factor / ratio) as u32;
            compute_multiscale_derivatives_for_evolution(evolution, sigma_size, &mut scratch);
        }
    }

    pub fn detector_response(&self, evolutions: &mut [EvolutionStep]) {
        self.compute_multiscale_derivatives(evolutions);
        for evolution in evolutions.iter_mut() {
            let ratio = f64::powi(2.0, evolution.octave as i32);
            let sigma_size = f64::round(evolution.esigma * self.derivative_factor / ratio);
            let sigma_size_quat = sigma_size.powi(4) as f32;
            evolution.ldet = GrayFloatImage::new(evolution.lxx.width(), evolution.lxx.height());
            azip!((
                ldet in evolution.ldet.mut_array2(),
                &lxx in evolution.lxx.ref_array2(),
                &lyy in evolution.lyy.ref_array2(),
                &lxy in evolution.lxy.ref_array2(),
            ) {
                *ldet = ((lxx * lyy) - (lxy * lxy)) * sigma_size_quat;
            });
        }
    }
}

fn compute_multiscale_derivatives_for_evolution(
    evolution: &mut EvolutionStep,
    sigma_size: u32,
    scratch: &mut ScharrScratch,
) {
    evolution.lx = derivatives::scharr_horizontal_buffered(&evolution.lsmooth, sigma_size, scratch);
    evolution.ly = derivatives::scharr_vertical_buffered(&evolution.lsmooth, sigma_size, scratch);
    evolution.lxx = derivatives::scharr_horizontal_buffered(&evolution.lx, sigma_size, scratch);
    evolution.lyy = derivatives::scharr_vertical_buffered(&evolution.ly, sigma_size, scratch);
    evolution.lxy = derivatives::scharr_vertical_buffered(&evolution.lx, sigma_size, scratch);
}
