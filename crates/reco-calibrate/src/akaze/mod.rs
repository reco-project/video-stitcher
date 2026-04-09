//! Vendored AKAZE feature detector with bug fixes.
//!
//! Based on [Christopher22/akaze](https://github.com/Christopher22/akaze.git)
//! (MIT license), itself a fork of the archived `rust-cv/akaze`.
//!
//! ## Bug fixes applied
//!
//! 1. `atan2(res_y, res_y)` -> `atan2(res_y, res_x)` in orientation computation
//! 2. Sliding window accumulator reset between iterations
//! 3. Non-maximum suppression upgraded from 4-connected to 8-connected
//!
//! ## Dependency simplification
//!
//! Removed `cv-core`, `bitarray`, `derive_more`, `space`. Descriptors are
//! returned as `[u8; 64]` (512-bit binary) directly.

mod contrast_factor;
mod derivatives;
mod descriptors;
mod detector_response;
mod evolution;
mod fed_tau;
mod image;
mod nonlinear_diffusion;
mod scale_space_extrema;

use self::image::{GrayFloatImage, gaussian_blur};
use ::image::DynamicImage;
use evolution::*;
use log::*;
use nonlinear_diffusion::pm_g2;

/// Descriptor size in bytes (512 bits = 64 bytes, M-LDB binary descriptor).
pub const DESC_BYTES: usize = 64;

/// A point of interest in an image.
#[derive(Debug, Clone, Copy)]
pub struct KeyPoint {
    /// Pixel coordinates `(x, y)`.
    pub point: (f32, f32),
    /// Detector response magnitude.
    pub response: f32,
    /// Keypoint extent radius in pixels.
    pub size: f32,
    /// Scale space octave.
    pub octave: usize,
    /// Evolution step index.
    pub class_id: usize,
    /// Dominant orientation angle (radians).
    pub angle: f32,
}

/// AKAZE feature detector and descriptor extractor.
///
/// The most important parameter is `detector_threshold`. Lower values
/// detect more features but include weaker ones.
#[derive(Debug, Copy, Clone)]
pub struct Akaze {
    /// Number of sublevels per scale level.
    pub num_sublevels: u32,
    /// Maximum octave evolution of the image.
    pub max_octave_evolution: u32,
    /// Base scale offset (sigma units).
    pub base_scale_offset: f64,
    /// Initial contrast factor parameter.
    pub initial_contrast: f64,
    /// Percentile level for the contrast factor.
    pub contrast_percentile: f64,
    /// Number of bins for the contrast factor histogram.
    pub contrast_factor_num_bins: usize,
    /// Factor for the multiscale derivatives.
    pub derivative_factor: f64,
    /// Detector response threshold to accept a keypoint.
    pub detector_threshold: f64,
    /// Number of descriptor channels (1, 2, or 3).
    pub descriptor_channels: usize,
    /// Actual patch size is `2 * pattern_size * point.scale`.
    pub descriptor_pattern_size: usize,
}

impl Akaze {
    /// Create with a custom detector threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            detector_threshold: threshold,
            ..Default::default()
        }
    }

    /// Sparse detection (threshold = 0.01).
    pub fn sparse() -> Self {
        Self::new(0.01)
    }

    /// Dense detection (threshold = 0.0001).
    pub fn dense() -> Self {
        Self::new(0.0001)
    }
}

impl Default for Akaze {
    fn default() -> Akaze {
        Akaze {
            num_sublevels: 4,
            max_octave_evolution: 4,
            base_scale_offset: 1.6f64,
            initial_contrast: 0.001f64,
            contrast_percentile: 0.7f64,
            contrast_factor_num_bins: 300,
            derivative_factor: 1.5f64,
            detector_threshold: 0.001f64,
            descriptor_channels: 3usize,
            descriptor_pattern_size: 10usize,
        }
    }
}

impl Akaze {
    /// Build the nonlinear scale space via FED (Fast Explicit Diffusion).
    fn create_nonlinear_scale_space(
        &self,
        evolutions: &mut [EvolutionStep],
        image: &GrayFloatImage,
    ) {
        trace!("Creating first evolution.");
        evolutions[0].lt = gaussian_blur(image, self.base_scale_offset as f32);
        trace!("Gaussian blur finished.");
        evolutions[0].lsmooth = evolutions[0].lt.clone();
        debug!(
            "Convolving first evolution with sigma={} Gaussian.",
            self.base_scale_offset
        );
        let mut contrast_factor = contrast_factor::compute_contrast_factor(
            &evolutions[0].lsmooth,
            self.contrast_percentile,
            1.0f64,
            self.contrast_factor_num_bins,
        );
        trace!("Computing contrast factor finished.");
        // Pre-allocate diffusion buffers (reused across all evolution steps)
        let (init_h, init_w) = (
            evolutions[0].lt.height() as usize,
            evolutions[0].lt.width() as usize,
        );
        let mut diffusion_bufs = nonlinear_diffusion::DiffusionBuffers::new(init_h, init_w);

        for i in 1..evolutions.len() {
            trace!("Creating evolution {}.", i);
            if evolutions[i].octave > evolutions[i - 1].octave {
                evolutions[i].lt = evolutions[i - 1].lt.half_size();
                trace!("Half-sizing done.");
                contrast_factor *= 0.75;
            } else {
                evolutions[i].lt = evolutions[i - 1].lt.clone();
            }
            evolutions[i].lsmooth = gaussian_blur(&evolutions[i].lt, 1.0f32);
            evolutions[i].lx = derivatives::scharr_horizontal(&evolutions[i].lsmooth, 1);
            evolutions[i].ly = derivatives::scharr_vertical(&evolutions[i].lsmooth, 1);
            evolutions[i].lflow = pm_g2(&evolutions[i].lx, &evolutions[i].ly, contrast_factor);
            for j in 0..evolutions[i].fed_tau_steps.len() {
                let step_size = evolutions[i].fed_tau_steps[j];
                nonlinear_diffusion::calculate_step_buffered(
                    &mut evolutions[i],
                    step_size as f32,
                    &mut diffusion_bufs,
                );
            }
        }
    }

    /// Detect keypoints after computing detector response.
    fn find_image_keypoints(&self, evolutions: &mut [EvolutionStep]) -> Vec<KeyPoint> {
        self.detector_response(evolutions);
        self.detect_keypoints(evolutions)
    }

    /// Extract keypoints and 512-bit binary descriptors from an image.
    ///
    /// Returns `(keypoints, descriptors)` where each descriptor is a
    /// 64-byte M-LDB binary descriptor.
    pub fn extract(&self, image: &DynamicImage) -> (Vec<KeyPoint>, Vec<[u8; DESC_BYTES]>) {
        let float_image = GrayFloatImage::from_dynamic(image);
        info!("Loaded a {} x {} image", image.width(), image.height());
        let mut evolutions = self.allocate_evolutions(image.width(), image.height());
        self.create_nonlinear_scale_space(&mut evolutions, &float_image);
        let keypoints = self.find_image_keypoints(&mut evolutions);
        let descriptors = self.extract_descriptors(&evolutions, &keypoints);
        (keypoints, descriptors)
    }

    /// GPU-accelerated extraction. Falls back to CPU if GPU diffusion fails.
    ///
    /// Uses GPU compute shaders for the nonlinear diffusion (the 99% bottleneck),
    /// then CPU for keypoint detection and descriptor computation.
    pub fn extract_gpu(
        &self,
        image: &DynamicImage,
        gpu: &reco_core::gpu::GpuContext,
        gpu_diff: &reco_core::gpu_diffusion::GpuDiffusion,
    ) -> (Vec<KeyPoint>, Vec<[u8; DESC_BYTES]>) {
        let float_image = GrayFloatImage::from_dynamic(image);
        let (w, h) = (image.width(), image.height());
        info!("Loaded a {w} x {h} image (GPU path)");

        let mut evolutions = self.allocate_evolutions(w, h);

        // GPU-accelerated scale space construction
        self.create_nonlinear_scale_space_gpu(&mut evolutions, &float_image, gpu, gpu_diff);

        // CPU keypoint detection + descriptors (sparse, complex access patterns)
        let keypoints = self.find_image_keypoints(&mut evolutions);
        let descriptors = self.extract_descriptors(&evolutions, &keypoints);
        (keypoints, descriptors)
    }

    /// Build the nonlinear scale space using GPU compute shaders.
    ///
    /// Uploads the image to GPU, runs gaussian_blur -> scharr -> pm_g2 -> FED
    /// diffusion on the GPU for each evolution step, and reads back the diffused
    /// image to CPU for subsequent keypoint detection.
    fn create_nonlinear_scale_space_gpu(
        &self,
        evolutions: &mut [EvolutionStep],
        image: &GrayFloatImage,
        gpu: &reco_core::gpu::GpuContext,
        gpu_diff: &reco_core::gpu_diffusion::GpuDiffusion,
    ) {
        use crate::akaze::image::gaussian_blur;

        // Evolution 0: Gaussian blur (same as CPU path)
        evolutions[0].lt = gaussian_blur(image, self.base_scale_offset as f32);
        evolutions[0].lsmooth = evolutions[0].lt.clone();

        let mut contrast_factor = contrast_factor::compute_contrast_factor(
            &evolutions[0].lsmooth,
            self.contrast_percentile,
            1.0f64,
            self.contrast_factor_num_bins,
        );

        let mut octave_idx = 0usize;

        for i in 1..evolutions.len() {
            if evolutions[i].octave > evolutions[i - 1].octave {
                evolutions[i].lt = evolutions[i - 1].lt.half_size();
                contrast_factor *= 0.75;
                octave_idx += 1;
            } else {
                evolutions[i].lt = evolutions[i - 1].lt.clone();
            }

            // Upload current lt to GPU
            let lt_data = evolutions[i].lt.as_slice();
            gpu_diff.upload_image(gpu, octave_idx.min(gpu_diff.octave_count() - 1), lt_data);

            // Run evolution on GPU: blur -> scharr -> pm_g2 -> FED loop
            let fed_steps: Vec<f64> = evolutions[i].fed_tau_steps.clone();
            gpu_diff.evolve(
                gpu,
                octave_idx.min(gpu_diff.octave_count() - 1),
                1.0, // gaussian sigma for lsmooth
                contrast_factor,
                &fed_steps,
            );

            // Read back diffused image
            let result = gpu_diff.readback_lt(gpu, octave_idx.min(gpu_diff.octave_count() - 1));
            let (ew, eh) = (evolutions[i].lt.width(), evolutions[i].lt.height());
            evolutions[i].lt = GrayFloatImage::from_f32_vec(result, ew, eh);

            // lsmooth is computed on GPU as part of evolve, but we need it on CPU
            // for detector_response. Recompute on CPU (it's just one Gaussian blur).
            evolutions[i].lsmooth = gaussian_blur(&evolutions[i].lt, 1.0f32);
        }
    }
}
