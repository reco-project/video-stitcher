//! Spatially fixed source patches for the two-plane calibration workflow.

use super::kernel::{normalize, sample};
use super::matching::Image;
use reco_core::calibration::Calibration;

pub(crate) struct Patch {
    pub(crate) pixels: Vec<[f64; 2]>,
    pub(crate) fitting: bool,
}

struct Grid {
    count: [usize; 2],
    start: [f64; 2],
    step: [f64; 2],
    margin: f64,
}

/// Local search and confirmation use the expected seam side of the source sensor.
pub(crate) fn select(
    seed: &Calibration,
    source: &Image,
    check: impl Fn() -> Result<(), String>,
) -> Result<Vec<Patch>, String> {
    let left_flipped = seed.projection.l_shape().is_some_and(|p| p.z_rx.cos() < 0.);
    select_models(
        std::slice::from_ref(seed),
        source,
        Grid {
            count: [14, 28],
            start: [if left_flipped { 0.045 } else { 0.50 }, 0.025],
            step: [0.035, 0.035],
            margin: 0.005,
        },
        check,
    )
}

/// Select textured patches whose complete footprint maps into the other sensor.
/// The spatial split is fixed before matching or fitting, not chosen from residuals.
pub(crate) fn select_overlap(
    seed: &Calibration,
    source: &Image,
    check: impl Fn() -> Result<(), String>,
) -> Result<Vec<Patch>, String> {
    check()?;
    seed.validate().map_err(|e| e.to_string())?;
    let map = seed
        .source_correspondence(0, 1)
        .map_err(|e| e.to_string())?;
    let mut lo = [1.0_f64; 2];
    let mut hi = [0.0_f64; 2];
    for row in 0..24 {
        check()?;
        for col in 0..32 {
            let uv = [(f64::from(col) + 0.5) / 32., (f64::from(row) + 0.5) / 24.];
            if map.map(uv).is_some() {
                for i in 0..2 {
                    lo[i] = lo[i].min(uv[i]);
                    hi[i] = hi[i].max(uv[i]);
                }
            }
        }
    }
    if hi[0] <= lo[0] || hi[1] <= lo[1] {
        return Ok(Vec::new());
    }
    let count = [24, 48];
    select_models(
        std::slice::from_ref(seed),
        source,
        Grid {
            count,
            start: lo,
            step: std::array::from_fn(|i| (hi[i] - lo[i]) / (count[i] - 1) as f64),
            margin: 0.005,
        },
        check,
    )
}

/// Common support for all generic hypotheses, fixed before their costs are known.
pub(crate) fn select_generic(
    seeds: &[Calibration],
    source: &Image,
    check: impl Fn() -> Result<(), String>,
) -> Result<Vec<Patch>, String> {
    select_models(
        seeds,
        source,
        Grid {
            count: [10, 18],
            start: [0.62, 0.12],
            step: [0.035, 0.045],
            margin: 0.015,
        },
        check,
    )
}

fn select_models(
    seeds: &[Calibration],
    source: &Image,
    grid: Grid,
    check: impl Fn() -> Result<(), String>,
) -> Result<Vec<Patch>, String> {
    check()?;
    if seeds.is_empty() || seeds.len() > 6 {
        return Err("invalid support hypotheses".into());
    }
    let mappings = seeds
        .iter()
        .map(|seed| {
            check()?;
            seed.validate().map_err(|e| e.to_string())?;
            if seed.lenses.len() != 2 || seed.projection.l_shape().is_none() {
                return Err("source support requires a two-source L-shape".into());
            }
            seed.source_correspondence(0, 1).map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, String>>()?;
    let [width, height] = source.dimensions();
    let mut selected = Vec::new();
    for row in 0..grid.count[1] {
        for column in 0..grid.count[0] {
            check()?;
            let centre = [
                (grid.start[0] + column as f64 * grid.step[0]) * width as f64 - 0.5,
                (grid.start[1] + row as f64 * grid.step[1]) * height as f64 - 0.5,
            ];
            let pixels: Vec<_> = (-12..=12)
                .step_by(4)
                .flat_map(|y| {
                    (-12..=12)
                        .step_by(4)
                        .map(move |x| [centre[0] + x as f64, centre[1] + y as f64])
                })
                .collect();
            if !pixels.iter().all(|p| {
                p[0] >= 0.0
                    && p[1] >= 0.0
                    && p[0] < (width - 1) as f64
                    && p[1] < (height - 1) as f64
                    && mappings.iter().all(|mapping| {
                        mapping
                            .map([(p[0] + 0.5) / width as f64, (p[1] + 0.5) / height as f64])
                            .is_some_and(|q| {
                                q.iter().all(|&v| v > grid.margin && v < 1.0 - grid.margin)
                            })
                    })
            }) {
                continue;
            }
            let mut values: Vec<_> = pixels
                .iter()
                .map(|&p| sample(source.values(), width, height, p))
                .collect();
            if normalize(&mut values) > 0.012 {
                selected.push(Patch {
                    pixels,
                    fitting: (row + column) % 3 != 0,
                });
            }
        }
    }
    check()?;
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::calibration::{Framing, Lens};
    use reco_core::projection::LShape;
    use std::cell::Cell;

    fn seed() -> Calibration {
        let lens = Lens::fisheye(640, 480, 384.0, 384.0, 320.0, 240.0, [0.0; 4]);
        Calibration::new(
            vec![lens.clone(), lens],
            LShape {
                intersect: 0.59,
                x_ty: 0.0,
                x_rz: 0.0,
                x_rx: 0.0,
                z_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            Framing {
                axis_offset: 0.27,
                tilt: 0.0,
                roll: 0.0,
            },
        )
    }

    fn textured(width: usize, height: usize) -> Image {
        Image::new(
            width,
            height,
            (0..width * height)
                .map(|i| 0.5 + (i as f64 * 0.73).sin() * 0.25)
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn selection_is_bounded_and_reproducible_with_a_disjoint_spatial_split() {
        let seed = seed();
        let image = textured(640, 480);
        let a = select(&seed, &image, || Ok(())).unwrap();
        let b = select(&seed, &image, || Ok(())).unwrap();
        assert!(a.iter().filter(|p| p.fitting).count() >= 15);
        assert!(a.iter().any(|p| !p.fitting));
        assert!(a.len() <= 392);
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(&b) {
            assert_eq!(a.pixels.len(), 49);
            assert_eq!(a.pixels, b.pixels);
            assert_eq!(a.fitting, b.fitting);
        }
        let map = seed.source_correspondence(0, 1).unwrap();
        for patch in &a {
            for p in &patch.pixels {
                assert!(p[0] >= 0.0 && p[0] < 639.0 && p[1] >= 0.0 && p[1] < 479.0);
                assert!(
                    map.map([(p[0] + 0.5) / 640.0, (p[1] + 0.5) / 480.0])
                        .unwrap()
                        .iter()
                        .all(|&v| v > 0.005 && v < 0.995)
                );
            }
        }
    }

    #[test]
    fn selection_has_no_support_for_blank_or_too_small_images() {
        assert!(
            select(
                &seed(),
                &Image::new(640, 480, vec![0.5; 640 * 480]).unwrap(),
                || Ok(())
            )
            .unwrap()
            .is_empty()
        );
        assert!(
            select(&seed(), &textured(16, 16), || Ok(()))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn cancellation_does_not_publish_partial_support() {
        let image = textured(640, 480);
        let calls = Cell::new(0);
        let result = select(&seed(), &image, || {
            calls.set(calls.get() + 1);
            if calls.get() == 12 {
                Err("cancelled".into())
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(e) if e == "cancelled"));
        assert_eq!(calls.get(), 12);
        let mut invalid = seed();
        invalid.lenses[0].fx = f64::NAN;
        assert!(select(&invalid, &image, || Ok(())).is_err());
    }
}
