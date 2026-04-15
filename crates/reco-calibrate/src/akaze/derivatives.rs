use super::image::{GrayFloatImage, fill_border};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, s};

/// Pre-allocated scratch buffer for Scharr convolutions.
///
/// Each separable Scharr call needs one intermediate image. By reusing
/// this buffer across the 5 Scharr calls per evolution step, we avoid
/// 10 `Array2` allocations per step (2 per Scharr x 5 calls).
pub struct ScharrScratch {
    /// Intermediate buffer for the first separable pass.
    buf: GrayFloatImage,
}

impl ScharrScratch {
    /// Create a scratch buffer sized for the given image dimensions.
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            buf: GrayFloatImage::new(width, height),
        }
    }

    /// Resize if the current dimensions don't match.
    fn ensure_size(&mut self, height: usize, width: usize) {
        if self.buf.height() != height || self.buf.width() != width {
            self.buf = GrayFloatImage::new(width, height);
        }
    }
}

pub fn scharr_horizontal(image: &GrayFloatImage, sigma_size: u32) -> GrayFloatImage {
    let img_horizontal = scharr_axis(
        image,
        sigma_size,
        FilterDirection::Horizontal,
        FilterOrder::Main,
    );
    scharr_axis(
        &img_horizontal,
        sigma_size,
        FilterDirection::Vertical,
        FilterOrder::Off,
    )
}

pub fn scharr_vertical(image: &GrayFloatImage, sigma_size: u32) -> GrayFloatImage {
    let img_horizontal = scharr_axis(
        image,
        sigma_size,
        FilterDirection::Horizontal,
        FilterOrder::Off,
    );
    scharr_axis(
        &img_horizontal,
        sigma_size,
        FilterDirection::Vertical,
        FilterOrder::Main,
    )
}

/// Compute horizontal Scharr derivative reusing a scratch buffer.
pub fn scharr_horizontal_buffered(
    image: &GrayFloatImage,
    sigma_size: u32,
    scratch: &mut ScharrScratch,
) -> GrayFloatImage {
    scratch.ensure_size(image.height(), image.width());
    scharr_axis_into(
        image,
        sigma_size,
        FilterDirection::Horizontal,
        FilterOrder::Main,
        &mut scratch.buf,
    );
    scharr_axis(
        &scratch.buf,
        sigma_size,
        FilterDirection::Vertical,
        FilterOrder::Off,
    )
}

/// Compute vertical Scharr derivative reusing a scratch buffer.
pub fn scharr_vertical_buffered(
    image: &GrayFloatImage,
    sigma_size: u32,
    scratch: &mut ScharrScratch,
) -> GrayFloatImage {
    scratch.ensure_size(image.height(), image.width());
    scharr_axis_into(
        image,
        sigma_size,
        FilterDirection::Horizontal,
        FilterOrder::Off,
        &mut scratch.buf,
    );
    scharr_axis(
        &scratch.buf,
        sigma_size,
        FilterDirection::Vertical,
        FilterOrder::Main,
    )
}

fn accumulate_mul_offset(
    mut accumulator: ArrayViewMut2<f32>,
    source: ArrayView2<f32>,
    val: f32,
    border: usize,
    xoff: usize,
    yoff: usize,
) {
    assert_eq!(source.dim(), accumulator.dim());
    let dims = source.dim();
    let mut accumulator =
        accumulator.slice_mut(s![border..dims.0 - border, border..dims.1 - border]);
    accumulator.scaled_add(
        val,
        &source.slice(s![
            yoff..dims.0 + yoff - 2 * border,
            xoff..dims.1 + xoff - 2 * border
        ]),
    );
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum FilterDirection {
    Horizontal,
    Vertical,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum FilterOrder {
    Main,
    Off,
}

fn scharr_axis(
    image: &GrayFloatImage,
    sigma_size: u32,
    dir: FilterDirection,
    order: FilterOrder,
) -> GrayFloatImage {
    let mut output = Array2::<f32>::zeros([image.height(), image.width()]);
    let border = sigma_size as usize;
    let w = 10.0 / 3.0;
    let norm = (1.0 / (2.0 * f64::from(sigma_size) * (w + 2.0))) as f32;
    let middle = norm * w as f32;

    let mut offsets = match order {
        FilterOrder::Main => vec![
            (norm, [border, 0]),
            (middle, [border, border]),
            (norm, [border, 2 * border]),
        ],
        FilterOrder::Off => vec![(-1.0, [border, 0]), (1.0, [border, 2 * border])],
    };

    if dir == FilterDirection::Horizontal {
        for (_, [x, y]) in &mut offsets {
            std::mem::swap(x, y);
        }
    }

    for (val, [x, y]) in offsets {
        accumulate_mul_offset(output.view_mut(), image.ref_array2(), val, border, x, y);
    }
    let mut output = GrayFloatImage::from_array2(output);
    fill_border(&mut output, border);
    output
}

/// Compute a Scharr axis pass, writing results into `output` instead
/// of allocating a new image.
fn scharr_axis_into(
    image: &GrayFloatImage,
    sigma_size: u32,
    dir: FilterDirection,
    order: FilterOrder,
    output: &mut GrayFloatImage,
) {
    let border = sigma_size as usize;
    let w = 10.0 / 3.0;
    let norm = (1.0 / (2.0 * f64::from(sigma_size) * (w + 2.0))) as f32;
    let middle = norm * w as f32;

    let mut offsets = match order {
        FilterOrder::Main => vec![
            (norm, [border, 0]),
            (middle, [border, border]),
            (norm, [border, 2 * border]),
        ],
        FilterOrder::Off => vec![(-1.0, [border, 0]), (1.0, [border, 2 * border])],
    };

    if dir == FilterDirection::Horizontal {
        for (_, [x, y]) in &mut offsets {
            std::mem::swap(x, y);
        }
    }

    // Scope the mutable borrow so fill_border can borrow output afterwards.
    {
        let mut out_arr = output.mut_array2();
        out_arr.fill(0.0);
        for (val, [x, y]) in offsets {
            accumulate_mul_offset(out_arr.view_mut(), image.ref_array2(), val, border, x, y);
        }
    }
    fill_border(output, border);
}
