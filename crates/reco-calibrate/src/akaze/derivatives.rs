use super::image::{GrayFloatImage, fill_border};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, s};

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
