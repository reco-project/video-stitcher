use super::evolution::EvolutionStep;
use super::image::GrayFloatImage;
use ndarray::{Array2, azip, s};

#[allow(non_snake_case)]
pub fn calculate_step(evolution_step: &mut EvolutionStep, step_size: f32) {
    let mut input = evolution_step.lt.mut_array2();
    let conductivities = evolution_step.lflow.ref_array2();
    let dim = input.dim();
    let mut horizontal_flow = Array2::<f32>::zeros((dim.0, dim.1 - 1));
    azip!((
        flow in &mut horizontal_flow,
        &a in input.slice(s![.., ..-1]),
        &b in input.slice(s![.., 1..]),
        &ca in conductivities.slice(s![.., ..-1]),
        &cb in conductivities.slice(s![.., 1..]),
    ) {
        *flow = step_size * ca * cb * (b - a);
    });
    let mut vertical_flow = Array2::<f32>::zeros((dim.0 - 1, dim.1));
    azip!((
        flow in &mut vertical_flow,
        &a in input.slice(s![..-1, ..]),
        &b in input.slice(s![1.., ..]),
        &ca in conductivities.slice(s![..-1, ..]),
        &cb in conductivities.slice(s![1.., ..]),
    ) {
        *flow = step_size * ca * cb * (b - a);
    });

    input
        .slice_mut(s![.., ..-1])
        .zip_mut_with(&horizontal_flow, |acc, &i| *acc += i);
    input
        .slice_mut(s![.., 1..])
        .zip_mut_with(&horizontal_flow, |acc, &i| *acc -= i);
    input
        .slice_mut(s![..-1, ..])
        .zip_mut_with(&vertical_flow, |acc, &i| *acc += i);
    input
        .slice_mut(s![1.., ..])
        .zip_mut_with(&vertical_flow, |acc, &i| *acc -= i);
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
