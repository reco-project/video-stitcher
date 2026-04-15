#![allow(dead_code)] // Vendored code - not all paths are used
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

/// Maximum allowed `descriptor_pattern_size` (OOB write if larger).
const MAX_PATTERN_SIZE: usize = 10;

/// AKAZE feature detector and descriptor extractor.
///
/// The most important parameter is `detector_threshold`. Lower values
/// detect more features but include weaker ones.
///
/// Construct via [`Akaze::new`], [`Akaze::sparse`], [`Akaze::dense`],
/// or [`AkazeBuilder`].
#[derive(Debug, Copy, Clone)]
pub struct Akaze {
    /// Number of sublevels per scale level.
    num_sublevels: u32,
    /// Maximum octave evolution of the image.
    max_octave_evolution: u32,
    /// Base scale offset (sigma units).
    base_scale_offset: f64,
    /// Initial contrast factor parameter.
    initial_contrast: f64,
    /// Percentile level for the contrast factor.
    contrast_percentile: f64,
    /// Number of bins for the contrast factor histogram.
    contrast_factor_num_bins: usize,
    /// Factor for the multiscale derivatives.
    derivative_factor: f64,
    /// Detector response threshold to accept a keypoint.
    detector_threshold: f64,
    /// Number of descriptor channels (1, 2, or 3).
    descriptor_channels: usize,
    /// Actual patch size is `2 * pattern_size * point.scale`.
    /// Must be <= [`MAX_PATTERN_SIZE`] (10).
    descriptor_pattern_size: usize,
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

    /// Return a builder for fine-grained configuration.
    pub fn builder() -> AkazeBuilder {
        AkazeBuilder(Akaze::default())
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

/// Builder for [`Akaze`] with validated parameters.
///
/// All setters clamp or validate inputs to prevent unsafe configurations
/// (e.g. `descriptor_pattern_size > 10` causes an out-of-bounds write).
pub struct AkazeBuilder(Akaze);

impl AkazeBuilder {
    /// Set the detector response threshold (must be > 0).
    pub fn threshold(mut self, t: f64) -> Self {
        self.0.detector_threshold = t.max(1e-12);
        self
    }

    /// Set the descriptor pattern size (clamped to 1..=10).
    pub fn pattern_size(mut self, size: usize) -> Self {
        self.0.descriptor_pattern_size = size.clamp(1, MAX_PATTERN_SIZE);
        self
    }

    /// Set the number of descriptor channels (clamped to 1..=3).
    pub fn descriptor_channels(mut self, ch: usize) -> Self {
        self.0.descriptor_channels = ch.clamp(1, 3);
        self
    }

    /// Set the number of sublevels per octave (clamped to 1..=8).
    pub fn num_sublevels(mut self, n: u32) -> Self {
        self.0.num_sublevels = n.clamp(1, 8);
        self
    }

    /// Set the max octave evolution (clamped to 1..=8).
    pub fn max_octave_evolution(mut self, n: u32) -> Self {
        self.0.max_octave_evolution = n.clamp(1, 8);
        self
    }

    /// Build the validated [`Akaze`] instance.
    pub fn build(self) -> Akaze {
        self.0
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

        // Pre-allocate diffusion buffers (reused across all evolution steps)
        let (init_h, init_w) = (evolutions[0].lt.height(), evolutions[0].lt.width());
        let mut diffusion_bufs = nonlinear_diffusion::DiffusionBuffers::new(init_h, init_w);

        // Compute contrast factor. When the first evolution step stays in
        // the same octave (the common case), we can compute evolution[1]'s
        // lsmooth/lx/ly first and reuse those gradients for the contrast
        // factor - avoiding the redundant gaussian+scharr that
        // compute_contrast_factor would do internally.
        let mut contrast_factor =
            if evolutions.len() > 1 && evolutions[1].octave == evolutions[0].octave {
                evolutions[1].lt = evolutions[0].lt.clone();
                evolutions[1].lsmooth = gaussian_blur(&evolutions[1].lt, 1.0f32);
                evolutions[1].lx = derivatives::scharr_horizontal(&evolutions[1].lsmooth, 1);
                evolutions[1].ly = derivatives::scharr_vertical(&evolutions[1].lsmooth, 1);

                let cf = contrast_factor::compute_contrast_factor_from_gradients(
                    &evolutions[1].lx,
                    &evolutions[1].ly,
                    self.contrast_percentile,
                    self.contrast_factor_num_bins,
                );
                trace!("Computing contrast factor (reused gradients) finished.");

                // Complete evolution[1]'s diffusion using the contrast factor.
                evolutions[1].lflow = pm_g2(&evolutions[1].lx, &evolutions[1].ly, cf);
                for j in 0..evolutions[1].fed_tau_steps.len() {
                    let step_size = evolutions[1].fed_tau_steps[j];
                    nonlinear_diffusion::calculate_step_buffered(
                        &mut evolutions[1],
                        step_size as f32,
                        &mut diffusion_bufs,
                    );
                }
                cf
            } else {
                let cf = contrast_factor::compute_contrast_factor(
                    &evolutions[0].lsmooth,
                    self.contrast_percentile,
                    1.0f64,
                    self.contrast_factor_num_bins,
                );
                trace!("Computing contrast factor finished.");
                cf
            };

        // Start from evolution 2 if we already processed 1, otherwise from 1.
        let start = if evolutions.len() > 1 && evolutions[1].octave == evolutions[0].octave {
            2
        } else {
            1
        };

        for i in start..evolutions.len() {
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
}
