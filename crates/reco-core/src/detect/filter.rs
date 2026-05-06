//! Composable detection-filter trait.
//!
//! A [`DetectionFilter`] transforms a `Vec<MappedDetection>` in place.
//! Filters run in a pre-tracker chain owned by `StitchSession`:
//! detections → filter₁ → filter₂ → ... → trackers → panner.
//!
//! # Composability
//!
//! Same type in, same type out. Every stage is auditable through the
//! pipeline event sink: when a sink is attached, the session emits
//! `PipelineEvent::DetectionFilter { before, after, filter_name }`
//! per stage so consumers can see what each filter actually did.
//!
//! # Trait lives here, impls live in reco-autocam
//!
//! Session owns the chain, so the trait must be visible to session
//! code. The concrete filters (FlickerFilter, FeetInRoiFilter, etc.)
//! live in reco-autocam where they can depend on autocam-internal
//! helpers without polluting reco-core.

use super::director::MappedDetection;
use crate::calibration::MatchCalibration;

/// Per-frame context handed to every filter.
///
/// Minimal on purpose: `frame_index` + `timestamp_ms` for time-based
/// filters (flicker windows), `calibration` for filters that need
/// projection math (feet-in-ROI, geometric plausibility).
#[derive(Debug, Clone, Copy)]
pub struct FilterContext<'a> {
    /// Monotonic frame counter, starting at 0.
    pub frame_index: u64,
    /// Milliseconds since session start.
    pub timestamp_ms: f64,
    /// Current calibration - borrowed for the duration of the filter
    /// call. Filters must not retain it.
    pub calibration: &'a MatchCalibration,
}

/// A detection-list filter stage.
///
/// Implementations are `Send` because `StitchSession` moves them
/// across the session boundary when attached. Runs on the render
/// thread, once per frame - keep filtering fast (ideally O(n) in
/// detection count).
pub trait DetectionFilter: Send {
    /// Short identifying name. Shown in logs, and attached to every
    /// emitted `PipelineEvent::DetectionFilter` so consumers can
    /// distinguish stages.
    fn name(&self) -> &'static str;

    /// Mutate `detections` in place. The filter may drop elements,
    /// reorder them, or replace them entirely; it must not panic on
    /// an empty input.
    fn filter(&mut self, detections: &mut Vec<MappedDetection>, ctx: &FilterContext<'_>);
}
