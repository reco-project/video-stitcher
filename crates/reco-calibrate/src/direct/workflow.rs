//! One bounded image-pair proposal using shared native preparation and geometry.

use super::initialization::{Initialization, initialize_native};
use super::matching::{Image, measure};
use super::preparation::{blur, refine};
use super::solver::{Control, Solution};
use super::support::select;
use reco_core::calibration::Calibration;
use std::time::{Duration, Instant};

pub(crate) enum ProposalError {
    InsufficientMatches,
    Failed(String),
}

impl From<String> for ProposalError {
    fn from(error: String) -> Self {
        Self::Failed(error)
    }
}

impl std::fmt::Display for ProposalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientMatches => f.write_str("insufficient image-measured initialization"),
            Self::Failed(error) => f.write_str(error),
        }
    }
}

pub(crate) struct Proposal {
    pub(crate) initialization: Initialization,
    pub(crate) refinement: Vec<Solution>,
    pub(crate) initial_patches: usize,
    pub(crate) matched_training_patches: usize,
    pub(crate) fitting_patches: usize,
    pub(crate) matching_elapsed: Duration,
    pub(crate) elapsed: Duration,
}

/// Propose an alignment from one raw analysis pair and a supplied starting model.
/// Independent confirmation, generic starts and feature fallback belong to the caller.
pub(crate) fn propose(
    seed: Calibration,
    images: &[Image; 2],
    estimate_optics: [bool; 2],
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<Proposal, ProposalError> {
    let started = Instant::now();
    control.check()?;
    let support = select(&seed, &images[0], || control.check())?;
    if support.iter().filter(|p| p.fitting).count() < 15 {
        return Err(ProposalError::Failed(
            "insufficient textured overlap".into(),
        ));
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
    let (source, target): (Vec<_>, Vec<_>) = measured
        .into_iter()
        .zip(&patches)
        .zip(fitting)
        .filter_map(|((measurement, patch), fitting)| {
            (fitting && measurement.accepted)
                .then(|| measurement.target.map(|p| (patch[patch.len() / 2], p)))
                .flatten()
        })
        .unzip();
    let matched_training_patches = source.len();
    if matched_training_patches < 15 {
        return Err(ProposalError::InsufficientMatches);
    }
    drop(patches);
    let initialization = initialize_native(
        seed,
        &source,
        &target,
        [images[0].dimensions(), images[1].dimensions()],
        control,
    )?;
    let patches: Vec<_> = select(&initialization.calibration, &images[0], || control.check())?
        .into_iter()
        .filter(|p| p.fitting)
        .map(|p| p.pixels)
        .collect();
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

pub(crate) struct DiscoveredProposal {
    pub(crate) proposal: Proposal,
    pub(crate) used_features: bool,
    pub(crate) focal: f64,
}

/// Recover placement from raw features when supplied optics start outside the pixel search radius.
pub(crate) fn from_optics(
    seed: Calibration,
    optics: [bool; 2],
    images: &[Image; 2],
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<Proposal, String> {
    match propose(seed.clone(), images, optics, pool, control) {
        Ok(result) => Ok(result),
        Err(ProposalError::InsufficientMatches) => {
            let measured = super::features::measure(images, pool, control)?;
            let placement = initialize_native(
                seed,
                &measured.source,
                &measured.target,
                [images[0].dimensions(), images[1].dimensions()],
                control,
            )?;
            propose(placement.calibration, images, optics, pool, control).map_err(|e| e.to_string())
        }
        Err(error) => Err(error.to_string()),
    }
}

/// Generic optical discovery with one feature fallback, only for insufficient pixel matches.
pub(crate) fn discover_and_propose(
    template: &Calibration,
    optics: [bool; 2],
    images: &[Image; 2],
    pool: &rayon::ThreadPool,
    control: &Control<'_>,
) -> Result<DiscoveredProposal, String> {
    let mut features = None;
    loop {
        let start =
            super::generic::discover(template, images, features.as_ref(), optics, pool, control)?;
        let selected = &start.hypotheses[start.selected];
        let fit = selected.fit.as_ref().map_err(Clone::clone)?;
        let seed = fit
            .stages
            .last()
            .ok_or("missing selected generic fit")?
            .calibration
            .clone();
        match propose(seed, images, optics, pool, control) {
            Ok(proposal) => {
                return Ok(DiscoveredProposal {
                    proposal,
                    used_features: features.is_some(),
                    focal: selected.focal,
                });
            }
            Err(ProposalError::InsufficientMatches) if features.is_none() => {
                let measured = super::features::measure(images, pool, control)?;
                if measured.source.len() < 15 {
                    return Err("insufficient raw feature correspondences".into());
                }
                features = Some(measured);
            }
            Err(error) => return Err(error.to_string()),
        }
    }
}
