use super::evolution::EvolutionStep;
use super::image::GrayFloatImage;
use ndarray::{Array2, azip, s};

/// Pre-allocated buffers for nonlinear diffusion to avoid per-step allocations.
pub struct DiffusionBuffers {
    pub horizontal_flow: Array2<f32>,
    pub vertical_flow: Array2<f32>,
}

impl DiffusionBuffers {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            horizontal_flow: Array2::zeros((height, width.saturating_sub(1).max(1))),
            vertical_flow: Array2::zeros((height.saturating_sub(1).max(1), width)),
        }
    }

    /// Resize buffers if the current dimensions don't match.
    pub fn ensure_size(&mut self, height: usize, width: usize) {
        let h_dim = (height, width.saturating_sub(1).max(1));
        let v_dim = (height.saturating_sub(1).max(1), width);
        if self.horizontal_flow.dim() != h_dim {
            self.horizontal_flow = Array2::zeros(h_dim);
        }
        if self.vertical_flow.dim() != v_dim {
            self.vertical_flow = Array2::zeros(v_dim);
        }
    }
}

/// Calculate a diffusion step using pre-allocated buffers.
#[allow(non_snake_case)]
pub fn calculate_step_buffered(
    evolution_step: &mut EvolutionStep,
    step_size: f32,
    buffers: &mut DiffusionBuffers,
) {
    let mut input = evolution_step.lt.mut_array2();
    let conductivities = evolution_step.lflow.ref_array2();
    let dim = input.dim();

    buffers.ensure_size(dim.0, dim.1);

    let horizontal_flow = &mut buffers.horizontal_flow;
    azip!((
        flow in &mut *horizontal_flow,
        &a in input.slice(s![.., ..-1]),
        &b in input.slice(s![.., 1..]),
        &ca in conductivities.slice(s![.., ..-1]),
        &cb in conductivities.slice(s![.., 1..]),
    ) {
        *flow = step_size * ca * cb * (b - a);
    });

    let vertical_flow = &mut buffers.vertical_flow;
    azip!((
        flow in &mut *vertical_flow,
        &a in input.slice(s![..-1, ..]),
        &b in input.slice(s![1.., ..]),
        &ca in conductivities.slice(s![..-1, ..]),
        &cb in conductivities.slice(s![1.., ..]),
    ) {
        *flow = step_size * ca * cb * (b - a);
    });

    input
        .slice_mut(s![.., ..-1])
        .zip_mut_with(horizontal_flow, |acc, &i| *acc += i);
    input
        .slice_mut(s![.., 1..])
        .zip_mut_with(horizontal_flow, |acc, &i| *acc -= i);
    input
        .slice_mut(s![..-1, ..])
        .zip_mut_with(vertical_flow, |acc, &i| *acc += i);
    input
        .slice_mut(s![1.., ..])
        .zip_mut_with(vertical_flow, |acc, &i| *acc -= i);
}

#[allow(non_snake_case)]
pub fn pm_g2(Lx: &GrayFloatImage, Ly: &GrayFloatImage, k: f64) -> GrayFloatImage {
    assert!(Lx.width() == Ly.width());
    assert!(Lx.height() == Ly.height());
    let inverse_k = (1.0f64 / (k * k)) as f32;
    let mut conductivities = Lx.zero_array();
    azip!((
        c in &mut conductivities,
        &x in Lx.ref_array2(),
        &y in Ly.ref_array2(),
    ) {
        *c = 1.0 / (1.0 + inverse_k * (x * x + y * y));
    });
    GrayFloatImage::from_array2(conductivities)
}
