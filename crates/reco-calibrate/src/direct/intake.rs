//! Bounded analysis images from the decoded luma plane, independent of transport.

use super::matching::Image;
use super::solver::Control;
use image::{ImageBuffer, Luma, imageops};

const ANALYSIS_SIDE: u32 = 960;

pub(crate) fn valid_dimensions(width: u32, height: u32) -> bool {
    (64..=8192).contains(&width)
        && (64..=8192).contains(&height)
        && u64::from(width) * u64::from(height) <= 16_000_000
}

/// Copy a tightly packed luma plane into a bounded fitting image without changing its field of view.
/// Resampling is one bounded library call; cancellation is checked around it and per conversion row.
pub(crate) fn prepare(
    luma: &[u8],
    width: u32,
    height: u32,
    control: &Control<'_>,
) -> Result<Image, String> {
    control.check()?;
    if !valid_dimensions(width, height) || luma.len() != width as usize * height as usize {
        return Err("invalid calibration luma plane".into());
    }
    let largest = width.max(height);
    let size = if largest <= ANALYSIS_SIDE {
        [width, height]
    } else {
        [width, height].map(|v| {
            ((u64::from(v) * u64::from(ANALYSIS_SIDE) + u64::from(largest / 2))
                / u64::from(largest)) as u32
        })
    };
    if size.iter().any(|&n| n < 64) {
        return Err("calibration analysis image is too narrow".into());
    }
    let resized;
    let pixels = if size == [width, height] {
        luma
    } else {
        let image = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, luma)
            .ok_or("invalid calibration luma plane")?;
        resized = imageops::resize(&image, size[0], size[1], imageops::FilterType::Triangle);
        resized.as_raw().as_slice()
    };
    control.check()?;
    let mut values = Vec::with_capacity(pixels.len());
    for row in pixels.chunks_exact(size[0] as usize) {
        control.check()?;
        values.extend(row.iter().map(|&v| f64::from(v) / 255.0));
    }
    Image::new(size[0] as usize, size[1] as usize, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    #[test]
    fn small_luma_planes_keep_exact_pixel_values_and_dimensions() {
        let cancelled = AtomicBool::new(false);
        let control = Control {
            interrupted: &cancelled,
            deadline: Instant::now() + Duration::from_secs(10),
        };
        let pixels: Vec<_> = (0..320 * 240).map(|i| (i % 256) as u8).collect();
        let image = prepare(&pixels, 320, 240, &control).unwrap();
        assert_eq!(image.dimensions(), [320, 240]);
        assert!(
            image
                .values()
                .iter()
                .zip(&pixels)
                .all(|(&a, &b)| a == f64::from(b) / 255.0)
        );
    }

    #[test]
    fn constant_and_asymmetric_frames_preserve_range_and_aspect() {
        let cancelled = AtomicBool::new(false);
        let control = Control {
            interrupted: &cancelled,
            deadline: Instant::now() + Duration::from_secs(10),
        };
        for ([w, h], expected) in [
            ([1920, 1440], [960, 720]),
            ([1440, 1920], [720, 960]),
            ([1280, 721], [960, 541]),
        ] {
            let image = prepare(&vec![71; w as usize * h as usize], w, h, &control).unwrap();
            assert_eq!(image.dimensions(), expected);
            assert!(image.values().iter().all(|&v| v == 71.0 / 255.0));
        }
    }

    #[test]
    fn invalid_cancelled_and_expired_inputs_return_no_image() {
        let cancelled = AtomicBool::new(true);
        let mut control = Control {
            interrupted: &cancelled,
            deadline: Instant::now() + Duration::from_secs(10),
        };
        assert!(matches!(prepare(&[], 64, 64, &control), Err(e) if e == "calibration cancelled"));
        cancelled.store(false, std::sync::atomic::Ordering::Relaxed);
        for dims in [
            [0, 0],
            [63, 64],
            [8193, 64],
            [8192, 8192],
            [u32::MAX, u32::MAX],
        ] {
            assert!(!valid_dimensions(dims[0], dims[1]));
            assert!(prepare(&[], dims[0], dims[1], &control).is_err());
        }
        assert!(prepare(&[0; 4095], 64, 64, &control).is_err());
        control.deadline = Instant::now() - Duration::from_millis(1);
        assert!(
            matches!(prepare(&[0; 4096], 64, 64, &control), Err(e) if e == "calibration time limit reached")
        );
    }
}
