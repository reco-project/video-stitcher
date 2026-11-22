//! One bounded image-pair proposal using shared native preparation and geometry.

use super::initialization::{Initialization, initialize_native};
use super::matching::{Image, measure};
use super::preparation::{blur, refine};
use super::solver::{Control, Solution};
use super::support::{select, select_overlap};
use reco_core::calibration::Calibration;
use std::time::{Duration, Instant};

pub(crate) struct Proposal {
    pub(crate) initialization: Initialization,
    pub(crate) refinement: Vec<Solution>,
    pub(crate) initial_patches: usize,
    pub(crate) matched_training_patches: usize,
    pub(crate) fitting_patches: usize,
    pub(crate) matching_elapsed: Duration,
    pub(crate) elapsed: Duration,
}

/// Try local pixel alignment first, then search the full frame for an overlap.
pub(crate) fn fit(
    seed: Calibration,
    images: &[Image; 2],
    optics: [bool; 2],
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<Proposal, String> {
    let local = if optics == [true; 2] {
        super::generic::discover(&seed, images, optics, pool, control)
    } else {
        Ok(seed.clone())
    };
    let mut error = match local.and_then(|s| propose(s, images, optics, false, pool, control)) {
        Ok(proposal) => return Ok(proposal),
        Err(error) => error,
    };
    control.check()?;
    log::debug!("Local placement: {error}");
    for start in super::overlap::discover(&seed, images, optics, pool, control)? {
        control.check()?;
        let patches = select_overlap(&start, &images[0], || control.check())?
            .into_iter()
            .filter(|p| p.fitting)
            .map(|p| p.pixels)
            .collect::<Vec<_>>();
        let candidate = super::preparation::coarse(start, images, &patches, optics, pool, control)
            .and_then(|fit| fit.into_calibration())
            .and_then(|s| propose(s, images, optics, true, pool, control));
        match candidate {
            Ok(proposal) => return Ok(proposal),
            Err(reason) => {
                log::debug!("Pixel overlap candidate: {reason}");
                error = reason;
            }
        }
    }
    control.check()?;
    Err(error)
}

/// Propose an alignment from one raw analysis pair and a supplied starting model.
/// Independent confirmation and generic starts belong to the caller.
fn propose(
    seed: Calibration,
    images: &[Image; 2],
    estimate_optics: [bool; 2],
    wide: bool,
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<Proposal, String> {
    let started = Instant::now();
    control.check()?;
    let support = if wide {
        select_overlap(&seed, &images[0], || control.check())?
    } else {
        select(&seed, &images[0], || control.check())?
    };
    if support.iter().filter(|p| wide || p.fitting).count() < 15 {
        return Err("insufficient textured overlap".into());
    }
    let initial_patches = support.len();
    let fitting: Vec<_> = support.iter().map(|p| p.fitting).collect();
    let patches: Vec<_> = support.into_iter().map(|p| p.pixels).collect();
    let matching_started = Instant::now();
    let measured = {
        let prepared = [
            blur(&images[0], 1.2, || control.check())?,
            blur(&images[1], 1.2, || control.check())?,
        ];
        measure(&seed, &prepared, &patches, pool, &|| control.check())?
    };
    let matching_elapsed = matching_started.elapsed();
    let mut accepted_patches = Vec::new();
    let mut source = Vec::new();
    let mut target = Vec::new();
    // The full-frame search already used the training image. Keep every verified
    // pair in that case. Independent assessment always uses different frames.
    for ((measurement, patch), train) in measured.into_iter().zip(patches).zip(fitting) {
        if (wide || train)
            && measurement.accepted
            && let Some(point) = measurement.target
        {
            source.push(patch[patch.len() / 2]);
            target.push(point);
            accepted_patches.push(patch);
        }
    }
    let matched_training_patches = source.len();
    log::debug!(
        "Bidirectional pixel matches: {matched_training_patches} fitting, {initial_patches} initial patches"
    );
    if matched_training_patches < 15 {
        return Err("insufficient image-measured initialization".into());
    }
    let initialization = initialize_native(
        seed,
        &source,
        &target,
        [images[0].dimensions(), images[1].dimensions()],
        control,
    )?;
    let patches: Vec<_> = if wide {
        accepted_patches
    } else {
        select(&initialization.calibration, &images[0], || control.check())?
            .into_iter()
            .filter(|p| p.fitting)
            .map(|p| p.pixels)
            .collect()
    };
    let fitting_patches = patches.len();
    let refinement = refine(
        initialization.calibration.clone(),
        images,
        &patches,
        estimate_optics,
        pool,
        control,
    )?;
    control.check()?;
    Ok(Proposal {
        initialization,
        refinement,
        initial_patches,
        matched_training_patches,
        fitting_patches,
        matching_elapsed,
        elapsed: started.elapsed(),
    })
}
