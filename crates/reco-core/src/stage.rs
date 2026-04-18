//! Post-undistort, pre-composite pipeline stage trait.
//!
//! Added 2026-04-18 as part of the plan-execution M3 foundation (§7
//! decision 7, "Pipeline-stage trait for per-camera transforms").
//!
//! # Why
//!
//! Today the stitch pipeline is fixed-function: `fisheye undistort ->
//! composite -> viewport crop`. Future work the user flagged during
//! the plan iteration (color correction / exposure normalization /
//! tone mapping / distributed-compute shims) needs an insertion point
//! between undistort and composite. Hardcoding those features would
//! bake the feature list into pipeline.rs and make alternatives
//! expensive to add.
//!
//! # Design
//!
//! A [`PipelineStage`] is a pluggable per-camera GPU transform. Each
//! stage owns its own wgpu bind group layout, fragment shader, and
//! optional CPU-side state. StitchCore (M3 refactor) carries a
//! `Vec<Box<dyn PipelineStage>>` executed in order between the
//! per-camera undistort pass and the composite pass.
//!
//! This file defines only the trait + a minimal [`StageContext`]
//! input type. No concrete stages ship in this commit - the slot is
//! reserved so future stages (color correction first) drop in without
//! another API-shape revision.
//!
//! # Relation to distributed compute
//!
//! A future remote-compute stage can fit the same trait: transform
//! the input texture by shipping its contents to a remote worker and
//! blocking on the reply. The trait contract makes no assumption
//! about locality.

use crate::detector::CameraId;

/// Per-frame context passed to [`PipelineStage::apply`].
///
/// Kept minimal on purpose: every field added here becomes a
/// commitment every stage implementor has to handle. Extend only when
/// a concrete stage needs it.
#[derive(Debug)]
pub struct StageContext {
    /// Which camera the current invocation is processing.
    pub camera: CameraId,
    /// Frame index (0-based) within the current session.
    pub frame_index: u64,
    /// Input texture width in pixels (the post-undistort texture the
    /// stage reads from).
    pub input_width: u32,
    /// Input texture height in pixels.
    pub input_height: u32,
}

/// Errors a pipeline stage can report.
#[derive(Debug, Clone)]
pub enum PipelineStageError {
    /// The stage's GPU resources are not initialized (e.g. the shader
    /// module or bind group layout was never built). Applying such a
    /// stage is a bug in the orchestrator, not the stage itself.
    NotInitialized(&'static str),
    /// The stage refused the current frame because its precondition
    /// (e.g. a required texture format) was not met. Callers may
    /// choose to skip this stage for this frame and continue.
    UnsupportedInput(String),
    /// Runtime failure inside the stage's shader or CPU-side work.
    /// The string is the stage's own error message.
    Runtime(String),
}

impl std::fmt::Display for PipelineStageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInitialized(why) => write!(f, "stage not initialized: {why}"),
            Self::UnsupportedInput(msg) => write!(f, "stage rejected input: {msg}"),
            Self::Runtime(msg) => write!(f, "stage runtime error: {msg}"),
        }
    }
}

impl std::error::Error for PipelineStageError {}

/// A pluggable per-camera transform that runs between undistort and
/// composite.
///
/// Stages are held by the pipeline as `Box<dyn PipelineStage>` and
/// executed in registration order. A stage that cannot run on the
/// current frame should return `Ok(())` and leave the output
/// unchanged, or return `Err(PipelineStageError::UnsupportedInput)`
/// for the orchestrator to log and skip.
///
/// Thread safety: the trait is `Send` so a stage can be constructed
/// on one thread and used on the wgpu render thread. It is *not*
/// `Sync` by default - stages that need to be shared across threads
/// must add the bound explicitly. This matches the mobile-friendly
/// trait-bound policy from the plan-execution doc §2.8.
pub trait PipelineStage: Send {
    /// Short human-readable name for logs and diagnostic bundles.
    fn name(&self) -> &'static str;

    /// Run the stage against the pipeline's intermediate per-camera
    /// texture.
    ///
    /// # Contract
    ///
    /// - The implementor is responsible for encoding any wgpu
    ///   commands into the orchestrator-provided `CommandEncoder`.
    ///   This commit intentionally leaves the encoder parameter
    ///   implicit - the method signature will grow when StitchCore
    ///   lands and needs to pass it; we are shipping the trait shape
    ///   first so early stage authors can design against the name.
    /// - On success, return `Ok(())`. On a frame-skippable problem,
    ///   return [`PipelineStageError::UnsupportedInput`]. On a real
    ///   failure (e.g. GPU allocation), return
    ///   [`PipelineStageError::Runtime`].
    /// - Stages MUST NOT panic. A panic here will be caught by the
    ///   session's catch_unwind wrapper (M2), but the stage loses
    ///   the ability to report the cause via `PipelineStageError`.
    fn apply(&mut self, ctx: &StageContext) -> Result<(), PipelineStageError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Noop;
    impl PipelineStage for Noop {
        fn name(&self) -> &'static str {
            "noop"
        }
        fn apply(&mut self, _ctx: &StageContext) -> Result<(), PipelineStageError> {
            Ok(())
        }
    }

    #[test]
    fn stages_can_live_in_a_vec_of_dyn() {
        // Core invariant: StitchCore will hold `Vec<Box<dyn PipelineStage>>`.
        // Verify the trait bounds allow that composition today.
        let mut stages: Vec<Box<dyn PipelineStage>> = vec![Box::new(Noop), Box::new(Noop)];
        let ctx = StageContext {
            camera: CameraId::Left,
            frame_index: 0,
            input_width: 1920,
            input_height: 1080,
        };
        for s in stages.iter_mut() {
            assert_eq!(s.name(), "noop");
            s.apply(&ctx).unwrap();
        }
    }

    #[test]
    fn pipeline_stage_error_displays() {
        let e = PipelineStageError::UnsupportedInput("test".into());
        let s = format!("{e}");
        assert!(s.contains("rejected input"));
    }
}
