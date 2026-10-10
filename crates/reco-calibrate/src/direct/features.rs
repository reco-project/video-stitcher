//! Bounded raw-image fallback using the calibrator's existing feature implementation.

use super::matching::Image;
use super::solver::Control;
use crate::features::{DetectRegion, detect_with_border, match_descriptors};

pub(crate) struct Correspondences {
    pub(crate) source: Vec<[f64; 2]>,
    pub(crate) target: Vec<[f64; 2]>,
}

pub(crate) fn measure(
    images: &[Image; 2],
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<Correspondences, String> {
    control.check()?;
    for image in images {
        let [w, h] = image.dimensions();
        if w < 64 || h < 64 || w > 1920 || h > 1920 || w * h > 1920 * 1080 {
            return Err("invalid feature analysis image".into());
        }
    }
    let extract = |side: usize| -> Result<_, String> {
        control.check()?;
        let image = &images[side];
        let [w, h] = image.dimensions();
        let mut rgba = Vec::with_capacity(w * h * 4);
        for row in image.values().chunks_exact(w) {
            control.check()?;
            rgba.extend(row.iter().flat_map(|v| {
                let value = (v * 255.0) as u8;
                [value, value, value, 255]
            }));
        }
        let region = DetectRegion {
            x_min: if side == 0 { 0.45 } else { 0.0 },
            x_max: if side == 0 { 1.0 } else { 0.65 },
            y_min: 0.0,
            y_max: 1.0,
        };
        let result = pool.install(|| {
            detect_with_border(&rgba, w as u32, h as u32, Some(region), 2000, 0.0001, 0)
        });
        control.check()?;
        Ok(result)
    };
    let (left, a) = extract(0)?;
    let (right, b) = extract(1)?;
    control.check()?;
    let matches = match_descriptors(&a, &b, 0.75);
    control.check()?;
    let [lw, lh] = images[0].dimensions().map(|v| v as f64);
    let [rw, rh] = images[1].dimensions().map(|v| v as f64);
    let mut source = Vec::new();
    let mut target = Vec::new();
    for matched in matches {
        control.check()?;
        let a = &left[matched.left_idx];
        let b = &right[matched.right_idx];
        let [ax, ay, bx, by] = [a.x, a.y, b.x, b.y].map(f64::from);
        if [ax, ay, bx, by].iter().all(|v| v.is_finite())
            && ax > 0.45 * lw
            && ax < lw - 1.0
            && ay >= 0.0
            && ay < lh - 1.0
            && bx >= 0.0
            && bx < 0.65 * rw
            && by >= 0.0
            && by < rh - 1.0
            && (ay / lh - by / rh).abs() < 0.4
        {
            source.push([ax, ay]);
            target.push([bx, by]);
        }
    }
    Ok(Correspondences { source, target })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    #[test]
    fn feature_fallback_rejects_cancelled_or_oversized_work_and_blank_has_no_matches() {
        let images = [
            Image::new(64, 64, vec![0.3; 4096]).unwrap(),
            Image::new(64, 64, vec![0.6; 4096]).unwrap(),
        ];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let cancelled = AtomicBool::new(true);
        let control = Control {
            interrupted: &cancelled,
            deadline: Instant::now() + Duration::from_secs(10),
        };
        assert!(
            matches!(measure(&images, &pool, &control), Err(e) if e == "calibration cancelled")
        );
        let active = AtomicBool::new(false);
        let control = Control {
            interrupted: &active,
            deadline: Instant::now() + Duration::from_secs(10),
        };
        let result = measure(&images, &pool, &control).unwrap();
        assert!(result.source.is_empty() && result.target.is_empty());
        let images = [
            Image::new(1921, 64, vec![0.3; 1921 * 64]).unwrap(),
            Image::new(64, 64, vec![0.6; 4096]).unwrap(),
        ];
        assert!(
            matches!(measure(&images, &pool, &control), Err(e) if e == "invalid feature analysis image")
        );
    }
}
