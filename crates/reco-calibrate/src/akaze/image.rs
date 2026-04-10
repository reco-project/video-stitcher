use image::{DynamicImage, ImageBuffer, Luma, imageops};
use log::*;
use ndarray::{Array2, ArrayView2, ArrayViewMut2, azip, s};
use nshare::{AsNdarray2, AsNdarray2Mut};
use std::f32;
use std::ops::{Deref, DerefMut};

/// Grayscale float image (pixel values in [0, 1]).
#[derive(Debug, Clone)]
pub struct GrayFloatImage(pub ImageBuffer<Luma<f32>, Vec<f32>>);

impl Deref for GrayFloatImage {
    type Target = ImageBuffer<Luma<f32>, Vec<f32>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for GrayFloatImage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GrayFloatImage {
    /// Convert a `DynamicImage` to unit-float grayscale.
    pub fn from_dynamic(input_image: &DynamicImage) -> Self {
        // Always convert to Luma8 first to handle any input variant safely.
        let gray_image = input_image.to_luma8();
        info!("Loaded an 8-bit image");
        Self(ImageBuffer::from_fn(
            gray_image.width(),
            gray_image.height(),
            |x, y| Luma([f32::from(gray_image[(x, y)][0]) / 255f32]),
        ))
    }

    pub fn from_array2(arr: Array2<f32>) -> Self {
        let (rows, cols) = arr.dim();
        let raw = arr.into_raw_vec_and_offset().0;
        // ImageBuffer::from_raw checks that raw.len() == width * height.
        // For a contiguous Array2 this is always true, but handle it
        // gracefully instead of panicking in case of future refactors.
        Self(
            ImageBuffer::from_raw(cols as u32, rows as u32, raw).unwrap_or_else(|| {
                log::error!(
                    "BUG: Array2 raw vec size mismatch for {}x{} image, using empty image",
                    cols,
                    rows,
                );
                ImageBuffer::from_pixel(cols as u32, rows as u32, Luma([0.0]))
            }),
        )
    }

    pub fn ref_array2(&self) -> ArrayView2<'_, f32> {
        self.0.as_ndarray2()
    }

    pub fn mut_array2(&mut self) -> ArrayViewMut2<'_, f32> {
        self.0.as_ndarray2_mut()
    }

    pub fn zero_array(&self) -> Array2<f32> {
        Array2::zeros((self.height(), self.width()))
    }

    pub fn width(&self) -> usize {
        self.0.width() as usize
    }

    pub fn height(&self) -> usize {
        self.0.height() as usize
    }

    pub fn new(width: usize, height: usize) -> Self {
        Self(ImageBuffer::from_pixel(
            width as u32,
            height as u32,
            Luma([0.0]),
        ))
    }

    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.get_pixel(x as u32, y as u32)[0]
    }

    pub fn put(&mut self, x: usize, y: usize, pixel_value: f32) {
        self.put_pixel(x as u32, y as u32, Luma([pixel_value]));
    }

    pub fn half_size(&self) -> Self {
        let width = self.width() / 2;
        let height = self.height() / 2;
        Self(imageops::resize(
            &self.0,
            width as u32,
            height as u32,
            imageops::FilterType::Nearest,
        ))
    }
}

/// Fill border pixels with neighboring values.
pub fn fill_border(output: &mut GrayFloatImage, half_width: usize) {
    for x in 0..output.width() {
        let plus = output.get(x, half_width);
        let minus = output.get(x, output.height() - half_width - 1);
        for y in 0..half_width {
            output.put(x, y, plus);
        }
        for y in (output.height() - half_width)..output.height() {
            output.put(x, y, minus);
        }
    }
    for y in 0..output.height() {
        let plus = output.get(half_width, y);
        let minus = output.get(output.width() - half_width - 1, y);
        for x in 0..half_width {
            output.put(x, y, plus);
        }
        for x in (output.width() - half_width)..output.width() {
            output.put(x, y, minus);
        }
    }
}

/// Horizontal separable filter.
#[inline(always)]
pub fn horizontal_filter(image: &GrayFloatImage, kernel: &[f32]) -> GrayFloatImage {
    debug_assert!(kernel.len() % 2 == 1);
    let half = kernel.len() / 2;
    let img = image.ref_array2(); // shape: (height, width)
    let (h, w) = img.dim();
    let mut out_arr = Array2::<f32>::zeros((h, w));
    // Convolve along columns (axis 1) for the interior region.
    for (ki, &kv) in kernel.iter().enumerate() {
        let offset = ki; // source column starts at ki, output at half
        let len = w - kernel.len() + 1; // number of valid output columns
        let src = img.slice(s![.., offset..offset + len]);
        let mut dst = out_arr.slice_mut(s![.., half..half + len]);
        azip!((d in &mut dst, &s in src) { *d += kv * s; });
    }
    let mut output = GrayFloatImage::from_array2(out_arr);
    fill_border(&mut output, half);
    output
}

/// Vertical separable filter.
#[inline(always)]
pub fn vertical_filter(image: &GrayFloatImage, kernel: &[f32]) -> GrayFloatImage {
    debug_assert!(kernel.len() % 2 == 1);
    let half = kernel.len() / 2;
    let img = image.ref_array2(); // shape: (height, width)
    let (h, _w) = img.dim();
    let mut out_arr = Array2::<f32>::zeros(img.dim());
    // Convolve along rows (axis 0) for the interior region.
    for (ki, &kv) in kernel.iter().enumerate() {
        let offset = ki; // source row starts at ki, output at half
        let len = h - kernel.len() + 1; // number of valid output rows
        let src = img.slice(s![offset..offset + len, ..]);
        let mut dst = out_arr.slice_mut(s![half..half + len, ..]);
        azip!((d in &mut dst, &s in src) { *d += kv * s; });
    }
    let mut output = GrayFloatImage::from_array2(out_arr);
    fill_border(&mut output, half);
    output
}

fn gaussian(x: f32, r: f32) -> f32 {
    ((2.0 * f32::consts::PI).sqrt() * r).recip() * (-x.powi(2) / (2.0 * r.powi(2))).exp()
}

fn gaussian_kernel(r: f32, kernel_size: usize) -> Vec<f32> {
    let mut kernel = vec![0f32; kernel_size];
    let half_width = (kernel_size / 2) as i32;
    let mut sum = 0f32;
    for i in -half_width..=half_width {
        let val = gaussian(i as f32, r);
        kernel[(i + half_width) as usize] = val;
        sum += val;
    }
    for val in kernel.iter_mut() {
        *val /= sum;
    }
    kernel
}

/// Gaussian blur via separable filter.
///
/// Returns a clone of the input unchanged if the image is too small
/// for the kernel (width or height <= kernel radius).
pub fn gaussian_blur(image: &GrayFloatImage, r: f32) -> GrayFloatImage {
    let kernel_size = (f32::ceil(r) as usize) * 2 + 1usize;
    if image.width() < kernel_size || image.height() < kernel_size {
        return image.clone();
    }
    let kernel = gaussian_kernel(r, kernel_size);
    let img_horizontal = horizontal_filter(image, &kernel);
    vertical_filter(&img_horizontal, &kernel)
}

#[cfg(test)]
mod tests {
    use super::gaussian_kernel;
    #[test]
    fn gaussian_kernel_correct() {
        let kernel = gaussian_kernel(3.0, 7);
        let known_correct_kernel = [
            0.1062_8852,
            0.1403_2133,
            0.1657_7007,
            0.1752_4014,
            0.1657_7007,
            0.1403_2133,
            0.1062_8852,
        ];
        for it in kernel.iter().zip(known_correct_kernel.iter()) {
            let (i, j) = it;
            assert!(f32::abs(*i - *j) < 0.0001);
        }
    }
}
