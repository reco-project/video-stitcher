//! Coarse placement from normalized luminance across the predicted overlap.

use super::generic::{FOCALS, guess};
use super::kernel::sample;
use super::matching::Image;
use super::preparation::blur;
use super::solver::Control;
use image::{ImageBuffer, Luma, imageops};
use rayon::prelude::*;
use reco_core::calibration::Calibration;
use std::f64::consts::PI;

struct Raster {
    values: Vec<f64>,
    size: [usize; 2],
}

impl Raster {
    fn new(image: &Image, side: usize, control: &Control<'_>) -> Result<Self, String> {
        let [w, h] = image.dimensions();
        let longest = w.max(h);
        let [width, height] = [w, h].map(|n| ((n * side + longest / 2) / longest).max(4));
        let input = ImageBuffer::from_fn(w as u32, h as u32, |x, y| {
            Luma([image.values()[y as usize * w + x as usize] as f32])
        });
        let small = imageops::resize(
            &input,
            width as u32,
            height as u32,
            imageops::FilterType::Triangle,
        );
        let values: Vec<_> = small.as_raw().iter().copied().map(f64::from).collect();
        let square = Image::new(width, height, values.iter().map(|x| x * x).collect())?;
        let image = Image::new(width, height, values)?;
        let mean = blur(&image, 1.7, || control.check())?;
        let moment = blur(&square, 1.7, || control.check())?;
        let values = image
            .values()
            .iter()
            .zip(mean.values())
            .zip(moment.values())
            .map(|((&v, &m), &s)| {
                ((v - m) / (s - m * m).max(0.).sqrt().max(3. / 255.)).clamp(-2., 2.)
            })
            .collect();
        Ok(Self {
            values,
            size: [width, height],
        })
    }
}

// Score the same source raster for every placement. A small energy floor penalizes
// accidental correlations on very small or nearly constant overlaps.
fn score(seed: &Calibration, [left, right]: &[Raster; 2]) -> Result<f64, String> {
    let mapping = seed
        .source_correspondence(0, 1)
        .map_err(|e| e.to_string())?;
    let [w, h] = left.size;
    let [tw, th] = right.size;
    let (mut cross, mut aa, mut bb, mut count) = (0., 0., 0., 0);
    for (i, &a) in left.values.iter().enumerate() {
        let uv = [
            ((i % w) as f64 + 0.5) / w as f64,
            ((i / w) as f64 + 0.5) / h as f64,
        ];
        if let Some([x, y]) = mapping.map(uv) {
            let b = sample(
                &right.values,
                tw,
                th,
                [x * tw as f64 - 0.5, y * th as f64 - 0.5],
            );
            cross += a * b;
            aa += a * a;
            bb += b * b;
            count += 1;
        }
    }
    if count < 20 || count * 12 < w * h {
        return Ok(-1.);
    }
    Ok(cross / ((aa * bb).sqrt() + 0.017 * (w * h) as f64))
}

/// Start perpendicular planes at the mean normalized focal distance. The rendered
/// plane is twice the sensor width in undistorted UV, giving the extra factor of two.
pub(super) fn center_planes(seed: &mut Calibration) -> Result<(), String> {
    let axis = 0.25
        * seed
            .lenses
            .iter()
            .map(|lens| lens.fx / f64::from(lens.width))
            .sum::<f64>();
    seed.framing.axis_offset = axis;
    seed.projection
        .l_shape_mut()
        .ok_or("expected L-shape projection")?
        .intersect = (1. - 2. * axis).clamp(0., 1.);
    Ok(())
}

// Keep a separate shortlist per focal length and mounting orientation. A good
// low-resolution score can also come from repeated texture in the wrong overlap.
fn coarse_starts(
    seed: &Calibration,
    flip: f64,
    pair: &[Raster; 2],
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<Vec<Calibration>, String> {
    // Roll steps are radians. Translations are fractions of one modeled plane width.
    let mut placements = Vec::with_capacity(9 * 20 * 7);
    for roll in -4..=4 {
        for overlap in 0..20 {
            for vertical in -3..=3 {
                placements.push((
                    0.125 * f64::from(roll),
                    0.18 + 0.035 * f64::from(overlap),
                    0.02 * f64::from(vertical),
                ));
            }
        }
    }
    let mut ranked = pool.install(|| {
        placements
            .into_par_iter()
            .map(|(roll, intersect, height)| {
                control.check()?;
                let mut c = seed.clone();
                let p = c
                    .projection
                    .l_shape_mut()
                    .ok_or("expected L-shape projection")?;
                p.z_rx = flip - roll;
                p.x_rz = roll;
                p.intersect = intersect;
                p.x_ty = height;
                Ok::<_, String>((score(&c, pair)?, c))
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(ranked.into_iter().take(4).map(|(_, c)| c).collect())
}

fn refine_placement(
    mut seed: Calibration,
    fine: &[Raster; 2],
    control: &Control<'_>,
) -> Result<(f64, Calibration), String> {
    let mut best = score(&seed, fine)?;
    // Halve the step in the solver's own scaled placement coordinates at each pass.
    for step in [0.6, 0.3, 0.15, 0.075] {
        for _ in 0..2 {
            for axis in 0..7 {
                control.check()?;
                let start = seed.clone();
                for signed_step in [-step, step] {
                    let mut delta = [0.; 7];
                    delta[axis] = signed_step;
                    let candidate = super::solver::decode_parameters(&start, &delta)?;
                    let intersect = candidate
                        .projection
                        .l_shape()
                        .ok_or("expected L-shape projection")?
                        .intersect;
                    if !(0.02..0.98).contains(&intersect) || candidate.framing.axis_offset <= 0.05 {
                        continue;
                    }
                    let value = score(&candidate, fine)?;
                    if value > best {
                        best = value;
                        seed = candidate;
                    }
                }
            }
        }
    }
    Ok((best, seed))
}

pub(super) fn discover(
    template: &Calibration,
    images: &[Image; 2],
    optics: [bool; 2],
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<Vec<Calibration>, String> {
    let pair = |side| {
        Ok::<_, String>([
            Raster::new(&images[0], side, control)?,
            Raster::new(&images[1], side, control)?,
        ])
    };
    let coarse = pair(40)?;
    if coarse
        .iter()
        .any(|r| r.values.iter().all(|v| v.abs() < 1e-6))
    {
        return Err("insufficient textured overlap".into());
    }
    let fine = pair(80)?;
    let mut seeds = Vec::new();
    if optics != [true; 2] {
        // Includes the supplied lens rescaled to initialize an unspecified peer.
        seeds.push(template.clone());
    }
    if optics != [false; 2] {
        for focal in FOCALS {
            seeds.push(guess(template, focal, [0.; 4], optics)?);
        }
    }
    let mut candidates = Vec::new();
    for mut seed in seeds {
        center_planes(&mut seed)?;
        let mut starts = Vec::new();
        for flip in [0., PI] {
            starts.extend(coarse_starts(&seed, flip, &coarse, pool, control)?);
        }
        let ranked = pool.install(|| {
            starts
                .into_par_iter()
                .map(|s| refine_placement(s, &fine, control))
                .collect::<Result<Vec<_>, _>>()
        })?;
        if let Some(best) = ranked.into_iter().max_by(|a, b| a.0.total_cmp(&b.0)) {
            candidates.push(best);
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(candidates.into_iter().map(|(_, seed)| seed).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::{
        calibration::{Framing, Lens},
        projection::LShape,
    };
    use std::{
        sync::atomic::AtomicBool,
        time::{Duration, Instant},
    };

    #[test]
    fn pixel_score_locates_the_overlap_on_an_inverted_sensor() {
        let lens = Lens::fisheye(256, 144, 150., 150., 128., 72., [0.; 4]);
        let mut rig = Calibration::new(
            vec![lens.clone(), lens],
            LShape {
                intersect: 0.45,
                x_ty: 0.,
                x_rz: 0.,
                x_rx: 0.,
                z_rz: 0.,
                z_rx: PI,
                blend_width: 0.05,
            },
            Framing {
                axis_offset: 0.3,
                tilt: 0.,
                roll: 0.,
            },
        );
        let reverse = rig.source_correspondence(1, 0).unwrap();
        let texture = |[u, v]: [f64; 2]| {
            0.5 + 0.20 * (31. * u + 43. * v).sin() + 0.15 * (62. * u - 21. * v).cos()
        };
        let image = |left| {
            Image::new(
                256,
                144,
                (0..256 * 144)
                    .map(|i| {
                        let uv = [
                            (f64::from(i % 256) + 0.5) / 256.,
                            (f64::from(i / 256) + 0.5) / 144.,
                        ];
                        if left {
                            texture(uv)
                        } else {
                            reverse.map(uv).map_or(0.5, texture)
                        }
                    })
                    .collect(),
            )
            .unwrap()
        };
        let control = Control {
            interrupted: &AtomicBool::new(false),
            deadline: Instant::now() + Duration::from_secs(10),
        };
        let pair = [true, false].map(|left| Raster::new(&image(left), 80, &control).unwrap());
        let inverted = score(&rig, &pair).unwrap();
        rig.projection.l_shape_mut().unwrap().z_rx = 0.;
        assert!(inverted > 0.5 && inverted > score(&rig, &pair).unwrap() + 0.3);
    }
}
